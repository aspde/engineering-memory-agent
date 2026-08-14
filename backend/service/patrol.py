"""Patrol service — runs agent.ainvoke() with patrol prompts and logs results.

The core patrol execution function.  Each patrol is one agent.ainvoke() call
with a patrol-specific System Prompt.  Results are persisted to patrol_logs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from langgraph.types import Command

from backend.db import get_session_factory
from backend.service.agent_service import get_agent
from backend.service.patrol_prompts import (
    DAILY_PATROL_PROMPT,
    WEEKLY_PATROL_PROMPT,
)
from backend.shared.config import config, current_thread_id, current_trace_id

logger = logging.getLogger(__name__)

# Unattended patrols auto-resolve write-conflict pauses (see run_patrol): a
# conflict interrupt pauses the graph for a human, but no human is watching a
# cron/webhook run, so without auto-resolution the patrol would sit
# 'interrupted' forever.  keep_both inserts the new content alongside the
# existing memory — nothing is dropped or overwritten — and the bound below
# stops a loop of repeated conflicts from spinning the whole patrol.
_MAX_AUTO_CONFLICT_RESOLUTIONS = 3

# Patrol scans get their own ReAct budget instead of borrowing the interactive
# MAX_AGENT_STEPS (default 5).  The graph force-routes to ``generate_final``
# once ``step_count`` hits the bound, which used to cut patrols off mid-scan
# with an incomplete report.  A full-memory scan needs many more search steps;
# types absent from the map keep the interactive default (``get_agent`` falls
# back to ``config.max_agent_steps``).
_PATROL_MAX_STEPS: dict[str, int] = {
    "daily": 15,
    "weekly": 20,
}

# LangGraph's default recursion_limit (50) counts *graph node executions*, not
# ReAct call-LLM iterations.  Each loop pass is ~4 nodes (call_llm →
# check_approval → tools → check_conflict), so a weekly patrol's 20
# max_steps ≈ 80 node executions — the default 50 cut every patrol mid-scan
# with GraphRecursionError.  This must stay comfortably above
# max(_PATROL_MAX_STEPS) × nodes-per-loop (20 × 4 = 80); 200 leaves headroom
# for the conflict-auto-resolve resume loop (each ainvoke restarts the count).
_PATROL_RECURSION_LIMIT = 200


def _interrupt_payload(interrupt: Any) -> Any:
    """The payload carried by a LangGraph interrupt object."""
    return interrupt.value if hasattr(interrupt, "value") else interrupt


def _is_conflict_interrupt(result: dict) -> bool:
    """True when the graph is paused on a write conflict (type='conflict')."""
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return False
    payload = _interrupt_payload(interrupts[0])
    return isinstance(payload, dict) and payload.get("type") == "conflict"

_PATROL_PROMPTS: dict[str, str] = {
    "daily": DAILY_PATROL_PROMPT,
    "weekly": WEEKLY_PATROL_PROMPT,
}

# Valid patrol_type values for manual trigger
VALID_PATROL_TYPES = {"daily", "weekly"}

# Minimal structural contract per patrol type (see prompts.py templates):
# the daily/weekly templates instruct the model to emit every category key —
# an empty array when a category has nothing to report — so a missing key
# means the model deviated from the JSON contract, not that the category is
# empty.
_PATROL_REQUIRED_KEYS: dict[str, set[str]] = {
    "daily": {"pattern_matches", "knowledge_gaps", "new_entities"},
    "weekly": {"contradictions", "stale_memories", "entity_coverage"},
}


def _validate_findings(findings: dict | None, patrol_type: str) -> str | None:
    """Return a parse_error string when *findings* are unusable for this
    patrol type, else None.

    Unknown patrol types have no contract to validate against and pass
    through unchanged.  A malformed scan must never persist as 'completed' —
    downstream consumers (e.g. ``persist_patrol_conflict`` reads
    ``memory_a_id`` from weekly contradiction findings) would crash on the
    resulting None.
    """
    required = _PATROL_REQUIRED_KEYS.get(patrol_type)
    if required is None:
        return None
    if not isinstance(findings, dict):
        return f"findings must be a JSON object, got {type(findings).__name__}"
    missing = sorted(required - set(findings.keys()))
    if missing:
        return f"findings missing required keys: {', '.join(missing)}"
    return None


def _parse_findings(raw_text: str) -> dict | None:
    """Try to parse the agent's final response as JSON findings.

    The agent is instructed to output pure JSON, but it may wrap with
    markdown fences or include explanatory text.  This function attempts
    to extract a JSON object from the response.
    """
    text = raw_text.strip()
    if not text:
        return None

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract from markdown code fences
    if "```" in text:
        lines = text.split("\n")
        in_fence = False
        fence_lines: list[str] = []
        for line in lines:
            if line.strip().startswith("```"):
                if in_fence:
                    break
                in_fence = True
                continue
            if in_fence:
                fence_lines.append(line)
        if fence_lines:
            try:
                return json.loads("\n".join(fence_lines))
            except json.JSONDecodeError:
                pass

    # Try to find a JSON object with regex (greedy — last { to first })
    import re

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Could not parse patrol findings as JSON, storing raw text")
    return {"raw_output": text[:5000]}


async def run_patrol(
    patrol_type: str,
    trigger: str,
    system_prompt: str,
    scope: str | None = None,
    patrol_id: str | None = None,
) -> str:
    """Execute a patrol and persist results to ``patrol_logs``.

    Args:
        patrol_type: ``daily`` or ``weekly``.
        trigger: ``cron``, ``webhook``, or ``manual``.
        system_prompt: The patrol System Prompt to use.
        scope: Optional scope filter (``"all"`` or ``"entity:<name>"``).
        patrol_id: Optional log id to use.  When omitted, a fresh UUID is
            generated.  The manual-trigger route passes one in so the caller
            gets the real id up front instead of a "pending" placeholder.

    Returns:
        The patrol log ID (UUID string), or ``""`` when an overlapping run
        of the same type was already in flight (the run was skipped).
    """
    patrol_id = patrol_id or str(uuid.uuid4())
    started_at = datetime.now(UTC)

    # Build the user message for the agent
    user_message_parts = [
        f"Execute {patrol_type} patrol.",
        f"Patrol ID: {patrol_id}",
    ]
    if scope and scope != "all":
        user_message_parts.append(f"Scope: {scope}")

    user_message = "\n".join(user_message_parts)

    # ── Overlap guard: skip if a patrol of this type is already running ──
    # The scheduler has one loop per type, but manual triggers and
    # event-driven runs can overlap a cron run.  Two concurrent patrols of
    # the same type would double provider spend and fight over the same
    # findings.  (Application-level check — a tiny race window between the
    # check and the insert is acceptable: the scheduler is single-loop per
    # type, so the realistic case is one contender already holding the row.)
    session_factory = get_session_factory()
    async with session_factory() as session:
        from sqlalchemy import text

        running = await session.execute(
            text(
                """SELECT id FROM patrol_logs
                   WHERE patrol_type = :type AND status = 'running'
                   LIMIT 1"""
            ),
            {"type": patrol_type},
        )
        running_row = running.fetchone()
    if running_row is not None:
        logger.warning(
            "Patrol %s already running as %s — skipping overlapping run",
            patrol_type,
            running_row[0],
        )
        return ""

    # ── Write initial "running" log entry ──
    async with session_factory() as session:
        from sqlalchemy import text

        await session.execute(
            text(
                """INSERT INTO patrol_logs (id, patrol_type, trigger, status, started_at)
                   VALUES (:id, :patrol_type, :trigger, 'running', :started_at)"""
            ),
            {
                "id": patrol_id,
                "patrol_type": patrol_type,
                "trigger": trigger,
                "started_at": started_at,
            },
        )
        await session.commit()

    # ── Execute agent ──
    findings: dict | None = None
    status = "completed"
    error_msg: str | None = None
    cancelled = False

    try:
        # Use a dedicated thread_id to isolate patrol from user conversations
        patrol_thread_id = f"patrol-{patrol_id}"
        token = current_thread_id.set(patrol_thread_id)
        # Link every LLM call in this patrol run to the patrol's own trace id.
        trace_token = current_trace_id.set(patrol_id)

        try:
            # Automated patrol runs are unattended — no human can approve a
            # paused write/ingest call, so pass an empty approval set and let
            # write tools execute directly.  The conflict HITL gate
            # (check_conflict_node) still pauses on a write conflict — that
            # interrupt is surfaced below, not swallowed.
            agent = get_agent(
                approval_required_tools=frozenset(),
                max_steps=_PATROL_MAX_STEPS.get(patrol_type),
            )
            # Bounded by PATROL_TIMEOUT (default 600s) — a hung provider or a
            # runaway ReAct loop must not leave the patrol in 'running' forever.
            async with asyncio.timeout(config.patrol_timeout):
                result = await agent.ainvoke(
                    input={
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt,
                            },
                            {
                                "role": "user",
                                "content": user_message,
                            },
                        ]
                    },
                    config={
                        "configurable": {"thread_id": patrol_thread_id},
                        "recursion_limit": _PATROL_RECURSION_LIMIT,
                    },
                )
                # A write conflict pauses the graph (check_conflict_node), and
                # patrols are unattended — no human can resolve the pause, so
                # without this the run would sit 'interrupted' forever.
                # Auto-resolve conflict pauses with keep_both (the new content
                # is inserted alongside the existing memory — nothing is
                # dropped or overwritten), bounded so a loop of repeated
                # conflicts can't spin the whole patrol.
                auto_resolved = 0
                while (
                    _is_conflict_interrupt(result)
                    and auto_resolved < _MAX_AUTO_CONFLICT_RESOLUTIONS
                ):
                    auto_resolved += 1
                    logger.warning(
                        "Patrol %s (%s) hit a memory conflict — auto-resolving "
                        "keep_both (%d/%d)",
                        patrol_id, patrol_type, auto_resolved,
                        _MAX_AUTO_CONFLICT_RESOLUTIONS,
                    )
                    result = await agent.ainvoke(
                        Command(resume={"resolution": "keep_both"}),
                        config={
                            "configurable": {"thread_id": patrol_thread_id},
                            "recursion_limit": _PATROL_RECURSION_LIMIT,
                        },
                    )

            # A HITL gate paused the graph: ``ainvoke`` returns *normally*
            # with ``__interrupt__`` set — it is not an exception.  Without
            # this check the pause was silently swallowed and the patrol was
            # persisted as 'completed' with no findings (the empty
            # final_response fell back to the last AIMessage).  Mark the run
            # interrupted so the pause is visible instead of pretending the
            # scan completed.
            interrupts = result.get("__interrupt__")
            if interrupts:
                status = "interrupted"
                interrupt_payload = _interrupt_payload(interrupts[0])
                findings = {"interrupt": interrupt_payload}
                logger.warning(
                    "Patrol %s (%s) interrupted for human review: %r",
                    patrol_id, patrol_type, interrupt_payload,
                )
            else:
                # Extract findings from the agent's final response
                raw_text = ""
                final_response = result.get("final_response", "")
                if final_response:
                    raw_text = str(final_response)
                else:
                    # Fallback: check last AIMessage content
                    messages = result.get("messages", [])
                    for m in reversed(messages):
                        if hasattr(m, "content") and m.content:
                            raw_text = str(m.content)
                            break
                findings = _parse_findings(raw_text) if raw_text else None
                parse_error = _validate_findings(findings, patrol_type)
                if parse_error:
                    # A malformed scan must never persist as 'completed' — the
                    # UI would render "done" while every finding click fails.
                    status = "failed"
                    error_msg = parse_error
                    if not isinstance(findings, dict):
                        findings = {"raw_output": raw_text[:5000]}
                    findings.setdefault("raw_output", raw_text[:5000])
                    findings["parse_error"] = parse_error
                    logger.warning(
                        "Patrol %s (%s) findings failed validation: %s",
                        patrol_id, patrol_type, parse_error,
                    )
        finally:
            current_thread_id.reset(token)
            current_trace_id.reset(trace_token)

    except Exception as exc:
        logger.exception("Patrol %s (%s) failed", patrol_id, patrol_type)
        status = "failed"
        error_msg = str(exc)
    except asyncio.CancelledError:
        # Task cancelled (e.g. scheduler shutdown).  ``except Exception``
        # doesn't catch this — CancelledError is a BaseException in 3.12 —
        # so without this branch a cancelled patrol would leave its row
        # stuck in 'running', and the overlap guard would skip every future
        # patrol of this type until restart.  Persist the failed status
        # below, then re-raise so the task ends as cancelled (not as a
        # successful run).
        cancelled = True
        status = "failed"
        error_msg = "cancelled mid-run"
        logger.warning("Patrol %s (%s) cancelled mid-run", patrol_id, patrol_type)

    # ── Update log entry with results ──
    completed_at = datetime.now(UTC)
    try:
        findings_json = json.dumps(findings) if findings else None
    except (TypeError, ValueError) as exc:
        # A payload that JSON can't serialise (e.g. an interrupt value
        # carrying an unserialisable object) must not leave the patrol stuck
        # in 'running' — record it as failed instead.
        logger.exception("Patrol %s findings are not JSON-serialisable", patrol_id)
        status = "failed"
        error_msg = f"findings not JSON-serialisable: {exc}"
        findings_json = None

    async with session_factory() as session:
        from sqlalchemy import text

        await session.execute(
            text(
                """UPDATE patrol_logs
                   SET status = :status,
                       findings = :findings,
                       completed_at = :completed_at
                   WHERE id = :id"""
            ),
            {
                "id": patrol_id,
                "status": status,
                "findings": findings_json,
                "completed_at": completed_at,
            },
        )
        await session.commit()

    if cancelled:
        logger.warning(
            "Patrol %s cancelled after %.1fs — marked failed",
            patrol_id,
            (completed_at - started_at).total_seconds(),
        )
    elif status == "interrupted":
        logger.warning(
            "Patrol %s interrupted after %.1fs — awaiting human review",
            patrol_id,
            (completed_at - started_at).total_seconds(),
        )
    elif error_msg:
        logger.error(
            "Patrol %s failed after %.1fs: %s",
            patrol_id,
            (completed_at - started_at).total_seconds(),
            error_msg,
        )
    else:
        finding_count = (
            sum(len(v) for v in findings.values() if isinstance(v, list))
            if findings
            else 0
        )
        logger.info(
            "Patrol %s completed in %.1fs — %d findings",
            patrol_id,
            (completed_at - started_at).total_seconds(),
            finding_count,
        )

    if cancelled:
        # Propagate the cancellation now that the failed status is persisted.
        raise asyncio.CancelledError()

    return patrol_id


def get_patrol_prompt(patrol_type: str) -> str:
    """Return the pre-defined System Prompt for a patrol type."""
    return _PATROL_PROMPTS.get(patrol_type, DAILY_PATROL_PROMPT)


async def mark_stale_patrols_failed() -> int:
    """Mark patrol_logs stuck in ``running`` as failed.

    Called at startup before the scheduler starts: a ``running`` row left by
    a previous process was killed mid-run and never completed, so showing it
    as in-flight would be a lie.  Returns the number of rows marked.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        from sqlalchemy import text

        result = await session.execute(
            text(
                """UPDATE patrol_logs
                   SET status = 'failed', completed_at = COALESCE(completed_at, now())
                   WHERE status = 'running'"""
            ),
        )
        await session.commit()
    count = result.rowcount or 0  # type: ignore[attr-defined]
    if count:
        logger.info("Marked %d stale patrol(s) as failed", count)
    return count
