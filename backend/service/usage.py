"""LLM usage tracing — one row per LLM call, persisted to Postgres.

The two LLM providers (``backend/service/llm_service.py``) are the single
choke point every LLM call passes through.  This module gives them a cheap,
**synchronous** way to observe each call without touching the database on the
hot path:

- :func:`begin_call` snapshots trace/thread context + timing + prompt size.
- :func:`record_call` appends one observation to an in-memory, thread-safe,
  bounded buffer and emits one structured ``llm_call`` log line.  It never
  raises — observability must never back-pressure the LLM path.
- :func:`flush_usage_buffer` drains the buffer into the ``llm_usage`` table
  in a single batched INSERT.  A background task
  (:func:`usage_flusher_loop`, started in ``backend/main.py`` lifespan)
  runs it every ``USAGE_FLUSH_INTERVAL_SECONDS``; the app also flushes once
  on shutdown so no buffered rows are lost on a clean exit.

Trace linkage: :data:`backend.shared.config.current_trace_id` is set by the
entry points (agent chat routes, patrol runner) to one id per run; every LLM
call made inside that run carries it, so ``GET /api/usage/trace/{id}``
replays a run end-to-end.  Background tasks that never set a trace still get
rows (empty ``trace_id``) with their ``scenario`` tag.

``scenario`` is the existing cost-observability tag from ``metrics.py``; the
in-memory counters there keep working independently — this module is the
persistent complement, not a replacement.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import threading
import time
from typing import Any

from sqlalchemy import text

from backend.db import get_session_factory
from backend.shared.config import config, current_thread_id, current_trace_id
from backend.shared.metrics import extract_tokens

logger = logging.getLogger(__name__)

# ── Recording buffer ─────────────────────────────────────────────────
# Module-level state + lock, same pattern as metrics.py.  The provider layer
# appends observations here synchronously; the flusher drains it in batches.

_pending: list[dict[str, Any]] = []
_pending_lock = threading.Lock()


def pending_count() -> int:
    """Number of observations waiting to be flushed (tests / observability)."""
    with _pending_lock:
        return len(_pending)


def pending_rows() -> list[dict[str, Any]]:
    """Return a copy of the buffered observations (tests / debugging)."""
    with _pending_lock:
        return list(_pending)


# ── Call observation helpers (used by the provider layer) ─────────────

# Prompt/response text is sampled (not always stored) for post-hoc quality
# analysis, so the persisted columns stay bounded.  Error calls are always
# sampled — failures are what analysis needs most; success calls sample at
# ``USAGE_SAMPLE_RATE``.  Columns are NULL for calls not sampled.
_SAMPLE_MAX_CHARS = 2000


def begin_call(messages: list[dict[str, Any]]) -> dict[str, Any]:
    """Snapshot the context of an upcoming LLM call.

    Returns an opaque context dict to pass to :func:`record_call`.  Reads the
    current trace/thread from their context variables, so the provider layer
    does not need them threaded through its signatures.
    """
    prompt_parts: list[str] = []
    prompt_chars = 0
    for m in messages:
        text = str(m.get("content", ""))
        prompt_chars += len(text)
        prompt_parts.append(text)
    return {
        "t0": time.perf_counter(),
        "trace_id": current_trace_id.get(),
        "thread_id": current_thread_id.get(),
        "prompt_chars": prompt_chars,
        # Truncated prompt text for the sampling path — a full transcript of
        # a long tool turn would exceed the sample cap anyway.
        "prompt_text": "\n".join(prompt_parts)[:_SAMPLE_MAX_CHARS],
    }


def record_call(
    ctx: dict[str, Any],
    *,
    model: str,
    provider: str,
    scenario: str,
    usage: Any = None,
    status: str = "success",
    error: str | None = None,
    response_text: str = "",
    attempts: int | None = None,
) -> None:
    """Record one LLM call observation into the buffer + structured log.

    Best-effort and non-blocking: never raises, and appends a row even when
    token usage is missing (``total_tokens=0``) so the call count and latency
    are still observable.  When ``config.usage_enabled`` is false this is a
    no-op (the in-memory ``/api/agent/usage`` counters are unaffected).

    *attempts* is how many times the provider was actually hit before this
    outcome — 1 for a clean first try, >1 when tenacity retried transient
    failures before the success (or the final failure).  None (the default)
    leaves the ``attempts`` column NULL: non-retryable call paths and rows
    written before the column existed must not get a fabricated value.
    """
    try:
        (
            input_tokens,
            output_tokens,
            total_tokens,
            cache_read_tokens,
            cache_creation_tokens,
        ) = extract_tokens(usage)
        latency_ms = round((time.perf_counter() - ctx["t0"]) * 1000)

        # Runtime health metrics — fed from the same event as the persisted
        # row but independent of usage_enabled (the Prometheus series are
        # process-local health; the cost pipeline is opt-out separately).
        try:
            from backend.shared.runtime_metrics import record_llm_call

            record_llm_call(
                scenario=scenario,
                status=status,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
        except Exception:
            pass  # metrics must never back-pressure the LLM hot path

        if not config.usage_enabled:
            return

        # Sample prompt/response text for post-hoc quality analysis (see the
        # module note above _SAMPLE_MAX_CHARS).  NULL columns mean "not
        # sampled" — no separate flag needed.  Sampled text is redacted before
        # it is stored so a credential a tool echoed back (an API key, a
        # captured Authorization header) never lands in llm_usage in the clear.
        prompt_sample = response_sample = None
        if status == "error" or random.random() < config.usage_sample_rate:
            prompt_text = (ctx.get("prompt_text") or "").strip()
            resp_text = str(response_text or "").strip()
            prompt_sample = _redact(prompt_text[:_SAMPLE_MAX_CHARS]) or None
            response_sample = _redact(resp_text[:_SAMPLE_MAX_CHARS]) or None

        row: dict[str, Any] = {
            "trace_id": ctx.get("trace_id") or None,
            "thread_id": ctx.get("thread_id") or None,
            "scenario": scenario,
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "latency_ms": latency_ms,
            "status": status,
            "error": _redact((error or "")[:500]) if error else None,
            "prompt_chars": ctx.get("prompt_chars", 0),
            "response_chars": len(str(response_text)),
            "prompt_sample": prompt_sample,
            "response_sample": response_sample,
            "attempts": attempts,
        }

        with _pending_lock:
            if len(_pending) >= config.usage_buffer_max:
                logger.warning(
                    "Usage buffer full (%d) — dropping oldest observation",
                    config.usage_buffer_max,
                )
                _pending.pop(0)
            _pending.append(row)

        logger.info(
            "llm_call trace=%s thread=%s scenario=%s provider=%s model=%s "
            "in=%d out=%d total=%d cache_read=%d cache_creation=%d "
            "latency_ms=%d status=%s attempts=%s",
            row["trace_id"],
            row["thread_id"],
            scenario,
            provider,
            model,
            input_tokens,
            output_tokens,
            total_tokens,
            cache_read_tokens,
            cache_creation_tokens,
            latency_ms,
            status,
            attempts,
        )
    except Exception:
        logger.exception("Failed to record LLM usage observation (swallowed)")


# ── Sample redaction ─────────────────────────────────────────────────
# Sampled prompt/response text can carry credentials (an API key a tool
# echoed back, an Authorization header captured in an error response).  A
# lightweight regex pass masks the obvious secret shapes before the text is
# stored or returned — pragmatic, not a replacement for secrets management.

_SECRET_VALUE_RE = re.compile(
    r"(?i)((?:api[_-]?key|access[_-]?token|auth[_-]?token|authorization|"
    r"client[_-]?secret|private[_-]?key|password|passwd|secret|token)\b"
    r"[^\n]{0,40}?[:=]\s*[\"']?)[A-Za-z0-9_\-\./+=]{8,}"
)
_BEARER_VALUE_RE = re.compile(
    r"(?i)((?:authorization|proxy-authorization)\s*[:=]\s*(?:[\"']?bearer\s+)"
    r"|\bbearer\s+)[A-Za-z0-9_\-\.=+/]{8,}"
)
_TOKEN_PREFIX_RE = re.compile(
    r"\b(sk-[A-Za-z0-9_\-]{12,}|ghp_[A-Za-z0-9_\-]{12,}|AKIA[A-Za-z0-9]{12,}|"
    r"hf_[A-Za-z0-9]{12,}|xox[baprs]-[A-Za-z0-9-]{10,})"
)
_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
    r".*?-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----",
    re.DOTALL,
)


def _redact(text: str) -> str:
    """Mask obvious secrets in sampled prompt/response text.

    Replaces the value of secret-shaped ``key=value`` / ``key: value``
    patterns, ``Authorization: Bearer …`` headers, well-known token prefixes
    (``sk-``, ``ghp_``, ``AKIA``, …) and PEM private-key blocks with ``***``.
    Best-effort: bounds accidental PII in ``llm_usage`` without pretending to
    hide every possible secret.
    """
    if not text:
        return text
    text = _BEARER_VALUE_RE.sub(lambda m: f"{m.group(1)}***", text)
    text = _SECRET_VALUE_RE.sub(lambda m: f"{m.group(1)}***", text)
    text = _TOKEN_PREFIX_RE.sub("***", text)
    return _PRIVATE_KEY_RE.sub("***", text)


# ── Flush: buffer → llm_usage table ──────────────────────────────────

# A failed flush requeues its rows at the head of the buffer and retries on
# the next tick, up to _FLUSH_MAX_ATTEMPTS per row — a transient DB hiccup
# no longer silently loses every buffered observation, while a persistent
# outage still caps memory growth (rows are dropped with an ERROR log, never
# blocking the LLM hot path, whose synchronous append stays unaffected).
_FLUSH_MAX_ATTEMPTS = 5           # drop a row after this many failed flushes
_FLUSH_ATTEMPT_KEY = "_flush_attempts"  # internal per-row retry counter
_FLUSH_BACKOFF_MAX_SECONDS = 60.0  # cap on the flusher's exponential backoff

# Consecutive flush failures drive the flusher loop's backoff (see
# usage_flusher_loop); reset to 0 on the first successful drain.
_consecutive_flush_failures = 0


async def flush_usage_buffer() -> int:
    """Drain the buffer into ``llm_usage`` with one batched INSERT.

    Returns the number of rows inserted (0 when the buffer was empty).  On a
    DB failure the drained rows are requeued at the head of the buffer (see
    :func:`_requeue_failed_rows`) so the next flush retries them — observability
    stays best-effort and must never back-pressure the LLM path, but a single
    failed batch is not simply discarded.
    """
    global _consecutive_flush_failures
    with _pending_lock:
        if not _pending:
            # Nothing drained — leave the failure counter untouched so a
            # series of empty flushes doesn't masquerade as recovery and
            # weaken the backoff.  Only a successful non-empty drain resets it.
            return 0
        rows = list(_pending)
        _pending.clear()

    stmt = text(
        """\
        INSERT INTO llm_usage (
            trace_id, thread_id, scenario, provider, model,
            input_tokens, output_tokens, total_tokens, latency_ms,
            cache_read_tokens, cache_creation_tokens,
            status, error, prompt_chars, response_chars,
            prompt_sample, response_sample, attempts
        ) VALUES (
            :trace_id, :thread_id, :scenario, :provider, :model,
            :input_tokens, :output_tokens, :total_tokens, :latency_ms,
            :cache_read_tokens, :cache_creation_tokens,
            :status, :error, :prompt_chars, :response_chars,
            :prompt_sample, :response_sample, :attempts
        )"""
    )
    try:
        async with get_session_factory()() as session:
            await session.execute(stmt, rows)
            await session.commit()
    except asyncio.CancelledError:
        # Shutdown can cancel the flusher while it awaits the DB.  ``except
        # Exception`` never sees CancelledError (a BaseException in 3.8+), so
        # without this branch the already-drained rows would be silently
        # discarded.  Requeue them for a later flush, then propagate the
        # cancellation — the per-row retry counters are preserved so a
        # persistent outage still caps memory growth.
        _consecutive_flush_failures += 1
        _requeue_failed_rows(rows)
        raise
    except Exception:
        _consecutive_flush_failures += 1
        _requeue_failed_rows(rows)
        return 0
    _consecutive_flush_failures = 0
    logger.info("Flushed %d LLM usage rows to llm_usage", len(rows))
    return len(rows)


def _requeue_failed_rows(rows: list[dict[str, Any]]) -> None:
    """Put *rows* back at the head of ``_pending``, dropping over-cap rows.

    Each failed flush increments the row's internal ``_flush_attempts``
    counter; rows past ``_FLUSH_MAX_ATTEMPTS`` are dropped with an ERROR log
    rather than growing the buffer without bound.  The extra key is harmless
    on insert — the batched INSERT names its columns explicitly.
    """
    with _pending_lock:
        retained: list[dict[str, Any]] = []
        for row in rows:
            row[_FLUSH_ATTEMPT_KEY] = row.get(_FLUSH_ATTEMPT_KEY, 0) + 1
            if row[_FLUSH_ATTEMPT_KEY] < _FLUSH_MAX_ATTEMPTS:
                retained.append(row)
        if retained:
            _pending[:0] = retained
    dropped = len(rows) - len(retained)
    if dropped:
        logger.error(
            "Dropping %d LLM usage rows after %d failed flush attempts",
            dropped,
            _FLUSH_MAX_ATTEMPTS,
        )
    else:
        logger.warning(
            "Failed to flush %d usage rows — requeued for retry",
            len(rows),
            exc_info=True,
        )


# Sampled prompt/response text is released after ``usage_sample_retention_days``
# (default 30): the metadata row stays for usage summaries and cost reporting,
# only the two large text columns are nulled.  The purge runs at most once a
# day inside the flusher loop (cheap UPDATE, hit by idx_llm_usage_created).
_last_sample_purge: float = 0.0
_SAMPLE_PURGE_INTERVAL_SECONDS = 24 * 3600


async def _purge_expired_samples() -> int:
    """Null out sampled prompt/response text older than the retention window.

    Returns the number of rows whose samples were released.  The row itself is
    kept — token/latency/status metadata still feeds the usage summaries and
    error-rate alerts; only the sampled text (bounded at ``_SAMPLE_MAX_CHARS``
    per column but unbounded in row count over time) is discarded.
    """
    async with get_session_factory()() as session:
        result = await session.execute(
            text(
                """\
                UPDATE llm_usage
                SET prompt_sample = NULL, response_sample = NULL
                WHERE created_at < now() - make_interval(days => :days)
                  AND (prompt_sample IS NOT NULL OR response_sample IS NOT NULL)
                """
            ),
            {"days": config.usage_sample_retention_days},
        )
        await session.commit()
        return result.rowcount or 0


async def usage_flusher_loop() -> None:
    """Background task: periodically drain the buffer into the DB.

    When a flush fails, waits an exponentially-growing extra interval (capped
    at ``_FLUSH_BACKOFF_MAX_SECONDS``) so a down database is not hammered with
    a drain attempt every tick; the counter resets on the first successful
    flush.  Also purges expired sample text at most once a day (see
    ``_purge_expired_samples``) — the metadata rows stay for usage summaries,
    only the sampled prompt/response columns are released.
    """
    global _last_sample_purge
    while True:
        await asyncio.sleep(config.usage_flush_interval_seconds)
        await flush_usage_buffer()
        failures = _consecutive_flush_failures
        if failures > 0:
            # 2^(failures-1) * base, capped — a persistent outage backs off to
            # a retry every ~2 minutes instead of spinning every tick.
            backoff = min(
                config.usage_flush_interval_seconds * (2 ** min(failures - 1, 5)),
                _FLUSH_BACKOFF_MAX_SECONDS,
            )
            if backoff > 0:
                logger.warning(
                    "Usage flush failing (%d consecutive) — backing off %.0fs",
                    failures,
                    backoff,
                )
                await asyncio.sleep(backoff)
        if time.monotonic() - _last_sample_purge >= _SAMPLE_PURGE_INTERVAL_SECONDS:
            _last_sample_purge = time.monotonic()
            try:
                purged = await _purge_expired_samples()
                if purged:
                    logger.info("Purged sampled text from %d expired usage rows", purged)
            except Exception:
                # Purging is best-effort — a failure must not break the flush
                # loop; it retries on the next daily tick.
                logger.exception("Sample purge failed")


# ── Cost estimation ──────────────────────────────────────────────────
# Approximate $ per 1M tokens (input, output) by model family, used only for
# reporting — real billing comes from the provider invoice.  Unknown models
# fall back to a conservative default rather than silently reporting $0.
_PRICE_RULES: list[tuple[str, tuple[float, float]]] = [
    ("claude-opus", (15.0, 75.0)),
    ("claude-sonnet", (3.0, 15.0)),
    ("claude-haiku", (1.0, 5.0)),
    ("gpt-4o-mini", (0.15, 0.6)),
    ("gpt-4o", (2.5, 10.0)),
    ("gpt-4.1", (2.0, 8.0)),
    ("deepseek-reasoner", (0.55, 2.19)),
    ("deepseek", (0.27, 1.1)),
]
_DEFAULT_PRICE = (1.0, 3.0)
# Cached input tokens are billed at a fraction of the input price; the
# fraction is provider-specific — Anthropic prompt caching and DeepSeek cache
# hits discount to ~10% of the input price, OpenAI-family cache reads to ~50%.
# The stored ``provider`` column uses "openai-compatible" for both DeepSeek
# and OpenAI, so DeepSeek is recognised by its model name (the Anthropic
# "claude" model-name fallback only matters for direct callers that omit
# *provider*).  Cache-creation tokens are billed at full input price — they
# are the tokens that populated the cache.
def _cache_read_price_factor(provider: str | None, model: str) -> float:
    key = (model or "").lower()
    if provider == "anthropic" or "claude" in key or "deepseek" in key:
        return 0.1
    return 0.5


def estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
    provider: str | None = None,
) -> float:
    """Estimate USD cost for one call from its token counts.

    ``model`` is matched by substring against a small built-in price table
    (input/output $ per 1M tokens); unknown models use ``_DEFAULT_PRICE``.
    ``input_tokens`` is the full-price input (see ``metrics.extract_tokens``);
    ``cache_read_tokens`` are billed at a provider-specific fraction of the
    input price (see :func:`_cache_read_price_factor`) and
    ``cache_creation_tokens`` at the full input price.
    """
    key = (model or "").lower()
    price_in, price_out = _DEFAULT_PRICE
    for fragment, (p_in, p_out) in _PRICE_RULES:
        if fragment in key:
            price_in, price_out = p_in, p_out
            break
    return (
        input_tokens / 1_000_000 * price_in
        + output_tokens / 1_000_000 * price_out
        + cache_read_tokens / 1_000_000 * price_in * _cache_read_price_factor(provider, model)
        + cache_creation_tokens / 1_000_000 * price_in
    )


# ── Query API ─────────────────────────────────────────────────────────
# All summaries group by their key *and* (provider, model), then fold in
# Python so per-model cost can be summed into the aggregate.  No separate
# aggregate table — raw rows are the source of truth (simple first).

_GROUPED_COLUMNS = (
    "provider, model, COUNT(*) AS calls, "
    "COALESCE(SUM(input_tokens), 0) AS input_tokens, "
    "COALESCE(SUM(output_tokens), 0) AS output_tokens, "
    "COALESCE(SUM(total_tokens), 0) AS total_tokens, "
    "COALESCE(SUM(cache_read_tokens), 0) AS cache_read_tokens, "
    "COALESCE(SUM(cache_creation_tokens), 0) AS cache_creation_tokens, "
    "COUNT(*) FILTER (WHERE status = 'error') AS error_calls, "
    "COALESCE(SUM(latency_ms), 0) AS latency_ms"
)


async def _summary_rows(days: int, key_expr: str) -> list[dict[str, Any]]:
    """Run one grouped summary query over the last *days* days."""
    async with get_session_factory()() as session:
        result = await session.execute(
            text(
                f"""\
                SELECT {key_expr} AS k, {_GROUPED_COLUMNS}
                FROM llm_usage
                WHERE created_at >= now() - make_interval(days => :days)
                GROUP BY k, provider, model
                ORDER BY k DESC, calls DESC
                """
            ),
            {"days": days},
        )
        return [dict(r._mapping) for r in result]


def _fold_summary(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fold (k, provider, model) grouped rows into per-key aggregates.

    Each output row carries calls / tokens / errors / latency + an estimated
    USD cost summed across the models in that key.
    """
    merged: dict[Any, dict[str, Any]] = {}
    for r in rows:
        k = r["k"]
        agg = merged.setdefault(
            k,
            {
                "key": k,
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cache_read_tokens": 0,
                "cache_creation_tokens": 0,
                "error_calls": 0,
                "latency_ms": 0,
                "est_cost": 0.0,
            },
        )
        agg["calls"] += r["calls"]
        agg["input_tokens"] += r["input_tokens"]
        agg["output_tokens"] += r["output_tokens"]
        agg["total_tokens"] += r["total_tokens"]
        agg["cache_read_tokens"] += r["cache_read_tokens"]
        agg["cache_creation_tokens"] += r["cache_creation_tokens"]
        agg["error_calls"] += r["error_calls"]
        agg["latency_ms"] += r["latency_ms"]
        agg["est_cost"] += estimate_cost(
            r["model"],
            r["input_tokens"],
            r["output_tokens"],
            r["cache_read_tokens"],
            r["cache_creation_tokens"],
            provider=r["provider"],
        )
    return list(merged.values())


async def get_daily_summary(days: int = 7) -> list[dict[str, Any]]:
    """Per-day totals (calls / tokens / errors / avg latency / est. cost)."""
    rows = await _summary_rows(days, "created_at::date")
    out = []
    for r in _fold_summary(rows):
        out.append(
            {
                "date": str(r["key"]),
                "calls": r["calls"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "total_tokens": r["total_tokens"],
                "cache_read_tokens": r["cache_read_tokens"],
                "cache_creation_tokens": r["cache_creation_tokens"],
                "error_calls": r["error_calls"],
                "avg_latency_ms": round(r["latency_ms"] / r["calls"]) if r["calls"] else 0,
                "est_cost": round(r["est_cost"], 4),
            }
        )
    return out


async def get_scenario_summary(days: int = 7) -> list[dict[str, Any]]:
    """Per-scenario totals — the cost/observability breakdown per call site."""
    rows = await _summary_rows(days, "scenario")
    out = []
    for r in _fold_summary(rows):
        out.append(
            {
                "scenario": r["key"],
                "calls": r["calls"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "total_tokens": r["total_tokens"],
                "cache_read_tokens": r["cache_read_tokens"],
                "cache_creation_tokens": r["cache_creation_tokens"],
                "error_calls": r["error_calls"],
                "avg_latency_ms": round(r["latency_ms"] / r["calls"]) if r["calls"] else 0,
                "est_cost": round(r["est_cost"], 4),
            }
        )
    return out


async def get_model_summary(days: int = 7) -> list[dict[str, Any]]:
    """Per-(provider, model) totals with estimated cost per model."""
    rows = await _summary_rows(days, "provider || '/' || model")
    out = []
    for r in _fold_summary(rows):
        out.append(
            {
                "model": r["key"],
                "calls": r["calls"],
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "total_tokens": r["total_tokens"],
                "cache_read_tokens": r["cache_read_tokens"],
                "cache_creation_tokens": r["cache_creation_tokens"],
                "error_calls": r["error_calls"],
                "avg_latency_ms": round(r["latency_ms"] / r["calls"]) if r["calls"] else 0,
                "est_cost": round(r["est_cost"], 4),
            }
        )
    return out


async def get_thread_usage(thread_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """Recent LLM calls for one conversation thread."""
    async with get_session_factory()() as session:
        result = await session.execute(
            text(
                """\
                SELECT trace_id, thread_id, scenario, provider, model,
                       input_tokens, output_tokens, total_tokens,
                       cache_read_tokens, cache_creation_tokens,
                       latency_ms, status, error, prompt_chars,
                       response_chars, created_at
                FROM llm_usage
                WHERE thread_id = :tid
                ORDER BY seq DESC
                LIMIT :limit
                """
            ),
            {"tid": thread_id, "limit": limit},
        )
        return [dict(r._mapping) for r in result]


async def get_trace(trace_id: str, limit: int = 50) -> list[dict[str, Any]]:
    """All LLM calls in one trace — replay a single agent run end-to-end.

    ``limit`` mirrors :func:`get_thread_usage`; pass a larger value to replay
    a run that exceeded the default window.
    """
    async with get_session_factory()() as session:
        result = await session.execute(
            text(
                """\
                SELECT trace_id, thread_id, scenario, provider, model,
                       input_tokens, output_tokens, total_tokens,
                       cache_read_tokens, cache_creation_tokens,
                       latency_ms, status, error, prompt_chars,
                       response_chars, created_at
                FROM llm_usage
                WHERE trace_id = :tid
                ORDER BY seq ASC
                LIMIT :limit
                """
            ),
            {"tid": trace_id, "limit": limit},
        )
        return [dict(r._mapping) for r in result]


async def get_samples(
    *,
    scenario: str | None = None,
    status: str | None = None,
    trace_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Return calls that carried sampled prompt/response text, newest first.

    The ``llm_usage`` sample columns feed post-hoc quality analysis — "what
    prompt produced this hallucinated answer", "which tool call had malformed
    JSON".  Only rows where a sample was stored (``prompt_sample`` or
    ``response_sample`` non-NULL) are returned, optionally narrowed by
    *scenario* / *status* / *trace_id*.
    """
    conditions = [
        "(prompt_sample IS NOT NULL OR response_sample IS NOT NULL)"
    ]
    params: dict[str, Any] = {"limit": limit}
    if scenario:
        conditions.append("scenario = :scenario")
        params["scenario"] = scenario
    if status:
        conditions.append("status = :status")
        params["status"] = status
    if trace_id:
        conditions.append("trace_id = :trace_id")
        params["trace_id"] = trace_id
    where_clause = " AND ".join(conditions)

    async with get_session_factory()() as session:
        result = await session.execute(
            text(
                f"""\
                SELECT trace_id, thread_id, scenario, provider, model,
                       input_tokens, output_tokens, total_tokens, latency_ms,
                       status, error, prompt_chars, response_chars,
                       prompt_sample, response_sample, created_at
                FROM llm_usage
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT :limit
                """
            ),
            params,
        )
        return [dict(r._mapping) for r in result]
