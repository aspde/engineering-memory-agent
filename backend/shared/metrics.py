"""Token accounting helpers shared by the LLM provider layer and usage.py.

The cost pipeline persists one row per LLM call into the ``llm_usage`` table
(``backend/service/usage.py``), fed from the provider layer's single choke
point.  This module carries the *parsing* side of that pipeline — extracting
token counts from the different provider usage shapes (OpenAI / Anthropic /
dict) and pulling the ``scenario`` kwarg out of a call's kwargs before the
SDK sees it.

It deliberately holds no counters of its own: the historical cost rows in
``llm_usage`` and the live Prometheus series in ``runtime_metrics.py`` are
the two observability sources, and both are fed by the same ``record_call``
event.  A third in-process counter used to exist here (``/api/agent/usage``)
and was removed as redundant — process-memory counts reset on restart and
added nothing over the persisted rows.

Public API:
    - ``extract_tokens(usage)`` — split any provider usage shape into
      (input, output, total, cache_read, cache_creation)
    - ``pop_scenario(kwargs)`` — extract & remove the scenario kwarg
"""

from __future__ import annotations

from typing import Any

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


def pop_scenario(kwargs: dict[str, Any]) -> str:
    """Pop the ``scenario`` kwarg from *kwargs*, defaulting to ``"default"``.

    Called by providers before forwarding *kwargs* to the SDK so the
    scenario tag never leaks into the API request.
    """
    scenario = kwargs.pop("scenario", "default")
    return scenario if isinstance(scenario, str) and scenario else "default"
