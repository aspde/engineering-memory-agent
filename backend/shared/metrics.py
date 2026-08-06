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
    - ``get_token_usage()`` — snapshot forinterview demo / API endpoint
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


# ── Internal helpers ─────────────────────────────────────────────────


def _extract_total_tokens(usage: Any) -> int:
    """Normalise *usage* (OpenAI / Anthropic / dict / None) to a total int."""
    if usage is None:
        return 0
    # OpenAI CompletionUsage — has total_tokens
    if hasattr(usage, "total_tokens"):
        return int(usage.total_tokens)
    # Anthropic Usage — has input_tokens + output_tokens
    if hasattr(usage, "input_tokens") and hasattr(usage, "output_tokens"):
        return int(usage.input_tokens) + int(usage.output_tokens)
    # Dict shape
    if isinstance(usage, dict):
        total = usage.get("total_tokens")
        if total is not None:
            return int(total)
        return int(usage.get("input_tokens", 0)) + int(usage.get("output_tokens", 0))
    return 0


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

    Used forinterview demo — call after a few agent turns to show per-scenario
    cost breakdown (agent_chat / memory_search / conflict_detection /
    entity_extraction / relation_extraction / rerank).
    """
    with _token_usage_lock:
        return dict(_token_usage)


def reset_token_usage() -> None:
    """Reset all counters — used in tests and to start a fresh measurement."""
    with _token_usage_lock:
        _token_usage.clear()
