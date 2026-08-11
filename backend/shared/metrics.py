"""LLM cost monitoring — token usage accounting by scenario.

Module-level counters keyed by *scenario* (e.g. ``"agent_chat"``,
``"conflict_detection"``).  Callers pass ``scenario=...`` as a kwarg
to ``LLMProvider.chat()`` / ``chat_raw()`` / ``chat_sync()``; the
provider pops it before sending the request so the SDK never sees it.

Thread-safe via a Lock — the agent runtime may invoke tools concurrently
from multiple asyncio tasks, and ``reset_token_usage()`` is used in tests.

Public API:
    - ``record_usage(scenario, usage)`` — accumulate tokens (called by providers)
    - ``pop_scenario(kwargs)`` — extract & remove the scenario kwarg
    - ``get_token_usage()`` — snapshot for the /api/agent/usage endpoint
    - ``reset_token_usage()`` — clear counters (tests / fresh measurement)

The leading underscore on ``_extract_total_tokens`` marks it as an
implementation detail for the LLMProvider layer; ``record_usage`` and
``pop_scenario`` are also intended for provider use only (called from
``llm_service``), while ``get_token_usage`` / ``reset_token_usage`` are
meant for external callers (API endpoints, tests).
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

# ── Module-level state ───────────────────────────────────────────────

_token_usage: dict[str, int] = defaultdict(int)
_token_usage_lock = threading.Lock()

# Structured-output failures (enrichment call sites degrade to []/defaults
# after retries).  Keyed by scenario so degradations are visible, not silent.
_structured_failures: dict[str, int] = defaultdict(int)
_structured_failures_lock = threading.Lock()


# ── Token extraction (shared with backend/service/usage.py) ──────────
# One extractor for every provider usage shape, so the in-memory counters
# here and the persisted ``llm_usage`` rows can never disagree on how a
# token count is derived.


def extract_tokens(usage: Any) -> tuple[int, int, int, int, int]:
    """Return ``(input, output, total, cache_read, cache_creation)`` tokens.

    Handles OpenAI ``CompletionUsage`` (``prompt_tokens``/``completion_tokens``
    with ``prompt_tokens_details.cached_tokens``), Anthropic ``Usage``
    (``input_tokens``/``output_tokens`` plus ``cache_read_input_tokens`` /
    ``cache_creation_input_tokens``), and plain dicts (OpenAI-compatible
    ``prompt_cache_hit_tokens``/``prompt_cache_miss_tokens``, or the same
    attribute names).

    ``input`` is the *full-price* input: OpenAI's ``prompt_tokens`` already
    includes its cached subset, so the cached tokens are subtracted and
    reported separately in ``cache_read``; Anthropic's ``input_tokens``
    already excludes cache and is used as-is.  ``total`` always equals
    ``input + output + cache_read + cache_creation``, which lets
    ``usage.estimate_cost`` apply the provider-specific cache-read discount
    without recomputing the split itself.
    """
    if usage is None:
        return 0, 0, 0, 0, 0

    # OpenAI CompletionUsage — prompt_tokens includes cached_tokens.
    if hasattr(usage, "prompt_tokens") and hasattr(usage, "completion_tokens"):
        prompt = int(usage.prompt_tokens or 0)
        output = int(usage.completion_tokens or 0)
        # Cache hits arrive either in ``prompt_tokens_details.cached_tokens``
        # (OpenAI's SDK shape) or as top-level ``prompt_cache_hit_tokens``
        # (DeepSeek's object shape — its ``prompt_tokens_details`` is empty,
        # so reading only the details would bill every cache hit at full
        # price).  Prefer the top-level field and fall back to the details,
        # mirroring the dict branch so both shapes of the same provider never
        # disagree on a token count.
        cache_read = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
        if not cache_read:
            cache_read = _cached_tokens_from_details(
                getattr(usage, "prompt_tokens_details", None)
            )
        return max(prompt - cache_read, 0), output, prompt + output, cache_read, 0

    # Anthropic Usage — input_tokens excludes cache tokens.
    if hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
        input_tok = int(usage.input_tokens or 0)
        output = int(usage.output_tokens or 0)
        cache_read = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        cache_creation = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
        return (
            input_tok,
            output,
            input_tok + output + cache_read + cache_creation,
            cache_read,
            cache_creation,
        )

    # Objects that only report a total (stream usage snapshots, older shapes).
    if hasattr(usage, "total_tokens"):
        return 0, 0, int(usage.total_tokens or 0), 0, 0

    # Plain dicts — prefer one alias per field, never sum across aliases
    # (a proxy returning both prompt_tokens and input_tokens must not have
    # its input counted twice).
    if isinstance(usage, dict):
        total = int(usage.get("total_tokens") or 0)
        if "prompt_tokens" in usage or "completion_tokens" in usage:
            prompt = int(usage.get("prompt_tokens") or 0)
            output = int(usage.get("completion_tokens") or 0)
            cache_read = int(usage.get("prompt_cache_hit_tokens") or 0)
            if not cache_read:
                cache_read = _cached_tokens_from_details(
                    usage.get("prompt_tokens_details")
                )
            return (
                max(prompt - cache_read, 0),
                output,
                total or (prompt + output),
                cache_read,
                0,
            )
        input_tok = int(usage.get("input_tokens") or 0)
        output = int(usage.get("output_tokens") or 0)
        cache_read = int(usage.get("cache_read_input_tokens") or 0)
        cache_creation = int(usage.get("cache_creation_input_tokens") or 0)
        return (
            input_tok,
            output,
            total or (input_tok + output + cache_read + cache_creation),
            cache_read,
            cache_creation,
        )

    # Unknown shape — nothing to count.
    return 0, 0, 0, 0, 0


def _cached_tokens_from_details(details: Any) -> int:
    """Cached-token count from ``prompt_tokens_details`` (object or dict)."""
    if isinstance(details, dict):
        return int(details.get("cached_tokens") or 0)
    return int(getattr(details, "cached_tokens", 0) or 0)


def _extract_total_tokens(usage: Any) -> int:
    """Normalise *usage* (OpenAI / Anthropic / dict / None) to a total int.

    Thin wrapper over :func:`extract_tokens` kept for the LLMProvider layer
    (``record_usage``) and for callers that only need the total.
    """
    return extract_tokens(usage)[2]


def pop_scenario(kwargs: dict[str, Any]) -> str:
    """Pop the ``scenario`` kwarg from *kwargs*, defaulting to ``"default"``.

    Called by providers before forwarding *kwargs* to the SDK so the
    scenario tag never leaks into the API request.
    """
    scenario = kwargs.pop("scenario", "default")
    return scenario if isinstance(scenario, str) and scenario else "default"


# ── Recording ────────────────────────────────────────────────────────


def record_usage(scenario: str, usage: Any) -> None:
    """Accumulate token *usage* under *scenario* for cost reporting.

    Called by each provider's ``chat`` / ``chat_raw`` / ``chat_sync``
    after a successful LLM response.  No-op when *usage* is ``None`` or
    zero (e.g. fake providers in tests).
    """
    total = _extract_total_tokens(usage)
    if total <= 0:
        return
    with _token_usage_lock:
        _token_usage[scenario] += total
        logger.info(
            "LLM token usage: scenario=%s this_call=%d cumulative=%d",
            scenario, total, _token_usage[scenario],
        )


# ── Snapshot / reset (public API) ────────────────────────────────────


def get_token_usage() -> dict[str, int]:
    """Return a snapshot of cumulative token usage per scenario.

    Returns cumulative token usage per scenario — call after a few agent
    turns to show the per-scenario cost breakdown (agent_chat /
    memory_search / conflict_detection / entity_extraction /
    relation_extraction / rerank).
    """
    with _token_usage_lock:
        return dict(_token_usage)


def reset_token_usage() -> None:
    """Reset all counters — used in tests and to start a fresh measurement."""
    with _token_usage_lock:
        _token_usage.clear()


def record_structured_failure(scenario: str) -> None:
    """Count one structured-output degradation for *scenario*.

    Called by enrichment call sites (entity/relation extraction) when a
    structured call exhausts its retries and falls back to ``[]``.  The
    counter is surfaced on ``GET /api/agent/usage`` so degradations are
    observable rather than silent.
    """
    with _structured_failures_lock:
        _structured_failures[scenario] += 1
        logger.warning(
            "Structured output degraded: scenario=%s total=%d",
            scenario,
            _structured_failures[scenario],
        )


def get_structured_failures() -> dict[str, int]:
    """Return a snapshot of structured-output failure counts per scenario."""
    with _structured_failures_lock:
        return dict(_structured_failures)


def reset_structured_failures() -> None:
    """Reset structured-failure counters — used in tests."""
    with _structured_failures_lock:
        _structured_failures.clear()
