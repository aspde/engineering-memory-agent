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
import threading
import time
from typing import Any

from sqlalchemy import text

from backend.db import get_session_factory
from backend.shared.config import config, current_thread_id, current_trace_id
from backend.shared.metrics import _extract_total_tokens

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


def reset_usage_buffer() -> None:
    """Drop all buffered observations — tests use this to isolate state."""
    with _pending_lock:
        _pending.clear()


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
) -> None:
    """Record one LLM call observation into the buffer + structured log.

    Best-effort and non-blocking: never raises, and appends a row even when
    token usage is missing (``total_tokens=0``) so the call count and latency
    are still observable.  When ``config.usage_enabled`` is false this is a
    no-op (the in-memory ``/api/agent/usage`` counters are unaffected).
    """
    try:
        if not config.usage_enabled:
            return
        input_tokens, output_tokens, total_tokens = _extract_tokens(usage)
        latency_ms = round((time.perf_counter() - ctx["t0"]) * 1000)

        # Sample prompt/response text for post-hoc quality analysis (see the
        # module note above _SAMPLE_MAX_CHARS).  NULL columns mean "not
        # sampled" — no separate flag needed.
        prompt_sample = response_sample = None
        if status == "error" or random.random() < config.usage_sample_rate:
            prompt_text = (ctx.get("prompt_text") or "").strip()
            resp_text = str(response_text or "").strip()
            prompt_sample = prompt_text[:_SAMPLE_MAX_CHARS] or None
            response_sample = resp_text[:_SAMPLE_MAX_CHARS] or None

        row: dict[str, Any] = {
            "trace_id": ctx.get("trace_id") or None,
            "thread_id": ctx.get("thread_id") or None,
            "scenario": scenario,
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency_ms": latency_ms,
            "status": status,
            "error": (error or "")[:500] if error else None,
            "prompt_chars": ctx.get("prompt_chars", 0),
            "response_chars": len(str(response_text)),
            "prompt_sample": prompt_sample,
            "response_sample": response_sample,
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
            "in=%d out=%d total=%d latency_ms=%d status=%s",
            row["trace_id"],
            row["thread_id"],
            scenario,
            provider,
            model,
            input_tokens,
            output_tokens,
            total_tokens,
            latency_ms,
            status,
        )
    except Exception:
        logger.exception("Failed to record LLM usage observation (swallowed)")


def _extract_tokens(usage: Any) -> tuple[int, int, int]:
    """Return ``(input, output, total)`` tokens from a provider usage object.

    Accepts OpenAI ``CompletionUsage`` (``prompt_tokens``/``completion_tokens``),
    Anthropic ``Usage`` (``input_tokens``/``output_tokens``), and plain dicts.
    """
    if usage is None:
        return 0, 0, 0
    if hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens"):
        i = int(usage.prompt_tokens)
        o = int(usage.completion_tokens)
        return i, o, i + o
    if hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
        i = int(usage.input_tokens)
        o = int(usage.output_tokens)
        return i, o, i + o
    if isinstance(usage, dict):
        i = int(usage.get("prompt_tokens") or 0) + int(usage.get("input_tokens") or 0)
        o = int(usage.get("completion_tokens") or 0) + int(usage.get("output_tokens") or 0)
        total = int(usage.get("total_tokens") or 0)
        return i, o, total if total else i + o
    # Unknown shape — fall back to the shared total-token extractor so the
    # call is still counted; input/output stay 0.
    return 0, 0, _extract_total_tokens(usage)


# ── Flush: buffer → llm_usage table ──────────────────────────────────


async def flush_usage_buffer() -> int:
    """Drain the buffer into ``llm_usage`` with one batched INSERT.

    Returns the number of rows inserted (0 when the buffer was empty).  A DB
    failure logs a warning and drops the drained rows — observability is
    best-effort and must never back-pressure the LLM path.
    """
    with _pending_lock:
        if not _pending:
            return 0
        rows = list(_pending)
        _pending.clear()

    stmt = text(
        """\
        INSERT INTO llm_usage (
            trace_id, thread_id, scenario, provider, model,
            input_tokens, output_tokens, total_tokens, latency_ms,
            status, error, prompt_chars, response_chars,
            prompt_sample, response_sample
        ) VALUES (
            :trace_id, :thread_id, :scenario, :provider, :model,
            :input_tokens, :output_tokens, :total_tokens, :latency_ms,
            :status, :error, :prompt_chars, :response_chars,
            :prompt_sample, :response_sample
        )"""
    )
    try:
        async with get_session_factory()() as session:
            await session.execute(stmt, rows)
            await session.commit()
    except Exception:
        logger.warning(
            "Failed to flush %d usage rows (dropped)", len(rows), exc_info=True
        )
        return 0
    logger.info("Flushed %d LLM usage rows to llm_usage", len(rows))
    return len(rows)


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

    Also purges expired sample text at most once a day (see
    ``_purge_expired_samples``) — the metadata rows stay for usage summaries,
    only the sampled prompt/response columns are released.
    """
    global _last_sample_purge
    while True:
        await asyncio.sleep(config.usage_flush_interval_seconds)
        await flush_usage_buffer()
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


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate USD cost for one call from its token counts.

    ``model`` is matched by substring against a small built-in price table
    (input/output $ per 1M tokens); unknown models use ``_DEFAULT_PRICE``.
    """
    key = (model or "").lower()
    price_in, price_out = _DEFAULT_PRICE
    for fragment, (p_in, p_out) in _PRICE_RULES:
        if fragment in key:
            price_in, price_out = p_in, p_out
            break
    return input_tokens / 1_000_000 * price_in + output_tokens / 1_000_000 * price_out


# ── Query API ─────────────────────────────────────────────────────────
# All summaries group by their key *and* (provider, model), then fold in
# Python so per-model cost can be summed into the aggregate.  No separate
# aggregate table — raw rows are the source of truth (simple first).

_GROUPED_COLUMNS = (
    "provider, model, COUNT(*) AS calls, "
    "COALESCE(SUM(input_tokens), 0) AS input_tokens, "
    "COALESCE(SUM(output_tokens), 0) AS output_tokens, "
    "COALESCE(SUM(total_tokens), 0) AS total_tokens, "
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
                "error_calls": 0,
                "latency_ms": 0,
                "est_cost": 0.0,
            },
        )
        agg["calls"] += r["calls"]
        agg["input_tokens"] += r["input_tokens"]
        agg["output_tokens"] += r["output_tokens"]
        agg["total_tokens"] += r["total_tokens"]
        agg["error_calls"] += r["error_calls"]
        agg["latency_ms"] += r["latency_ms"]
        agg["est_cost"] += estimate_cost(r["model"], r["input_tokens"], r["output_tokens"])
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
                       input_tokens, output_tokens, total_tokens, latency_ms,
                       status, error, prompt_chars, response_chars, created_at
                FROM llm_usage
                WHERE thread_id = :tid
                ORDER BY seq DESC
                LIMIT :limit
                """
            ),
            {"tid": thread_id, "limit": limit},
        )
        return [dict(r._mapping) for r in result]


async def get_trace(trace_id: str) -> list[dict[str, Any]]:
    """All LLM calls in one trace — replay a single agent run end-to-end."""
    async with get_session_factory()() as session:
        result = await session.execute(
            text(
                """\
                SELECT trace_id, thread_id, scenario, provider, model,
                       input_tokens, output_tokens, total_tokens, latency_ms,
                       status, error, prompt_chars, response_chars, created_at
                FROM llm_usage
                WHERE trace_id = :tid
                ORDER BY seq ASC
                """
            ),
            {"tid": trace_id},
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
