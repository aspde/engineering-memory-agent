"""Executors for the memory write-path eval — the real production calls.

Each default executor drives the same function production uses, so the eval
measures shipped behavior rather than a harness copy:

- ``make_conflict_detector`` → ``backend.service.memory._detect_conflict``
  (the ``memory.conflict`` prompt + structured output channel);
- ``make_merge_summarizer`` → ``backend.service.memory.merge_summaries``
  (the shared ``memory.merge`` prompt call — single entry point for
  ``_merge_memory``, conflict resolution, and this eval);
- ``make_gate_checker`` → ``backend.agent.nodes._llm_gate_verdict`` (the
  raising core of the auto-memory gate — the fail-open wrapper is NOT used,
  so a provider outage records as an execution error instead of silently
  scoring every item "worthy").

All executors accept an injection point for unit tests.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

# (existing_summary, new_summary) → contradiction verdict
ConflictDetector = Callable[[str, str], Awaitable[bool]]
# (existing_summary, new_summary) → merged summary text
MergeSummarizer = Callable[[str, str], Awaitable[str]]
# (content) → gate verdict; raises on provider failure
GateChecker = Callable[[str], Awaitable[bool]]


def make_conflict_detector() -> ConflictDetector:
    """Executor that runs the real conflict-detection call."""

    async def _detect(existing: str, new: str) -> bool:
        from backend.service.memory import _detect_conflict

        return await _detect_conflict(
            {"summary": existing}, {"summary": new}
        )

    return _detect


def make_merge_summarizer() -> MergeSummarizer:
    """Executor that runs the real merge-prompt call."""

    async def _merge(existing: str, new: str) -> str:
        from backend.service.memory import merge_summaries

        return await merge_summaries(existing, new)

    return _merge


def make_gate_checker() -> GateChecker:
    """Executor that runs the real auto-memory gate verdict.

    Uses ``_llm_gate_verdict`` (raising) rather than ``_llm_gate_worthy``
    (fail-open): in the eval a gate outage must surface as a per-item error
    and drag the aggregate down, not masquerade as unanimous allow.
    """

    async def _check(content: str) -> bool:
        from backend.agent.nodes import _llm_gate_verdict

        return await _llm_gate_verdict(content)

    return _check
