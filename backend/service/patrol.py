"""Patrol service — runs agent.ainvoke() with patrol prompts and logs results.

The core patrol execution function.  Each patrol is one agent.ainvoke() call
with a patrol-specific System Prompt.  Results are persisted to patrol_logs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone

from backend.db import get_session_factory
from backend.service.agent_service import get_agent
from backend.service.patrol_prompts import (
    CI_FAILURE_PATROL_PROMPT,
    DAILY_PATROL_PROMPT,
    JIRA_RESOLVED_PATROL_PROMPT,
    WEEKLY_PATROL_PROMPT,
)
from backend.shared.config import config, current_thread_id

logger = logging.getLogger(__name__)

_PATROL_PROMPTS: dict[str, str] = {
    "daily": DAILY_PATROL_PROMPT,
    "weekly": WEEKLY_PATROL_PROMPT,
    "contradiction_scan": WEEKLY_PATROL_PROMPT,  # reuse weekly prompt
    "event_driven": "",  # set dynamically based on event source
}

_EVENT_PROMPTS: dict[str, str] = {
    "ci_failure": CI_FAILURE_PATROL_PROMPT,
    "jira_resolved": JIRA_RESOLVED_PATROL_PROMPT,
}

# Valid patrol_type values for manual trigger
VALID_PATROL_TYPES = {"daily", "weekly", "contradiction_scan"}


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
    event_source: str | None = None,
) -> str:
    """Execute a patrol and persist results to ``patrol_logs``.

    Args:
        patrol_type: ``daily``, ``weekly``, ``contradiction_scan``,
            or ``event_driven``.
        trigger: ``cron``, ``webhook``, or ``manual``.
        system_prompt: The patrol System Prompt to use.
        scope: Optional scope filter (``"all"`` or ``"entity:<name>"``).
        event_source: For event_driven patrols, the connector source name
            (e.g. ``"ci"``, ``"pingcode"``).

    Returns:
        The patrol log ID (UUID string).
    """
    patrol_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc)

    # Build the user message for the agent
    user_message_parts = [
        f"Execute {patrol_type} patrol.",
        f"Patrol ID: {patrol_id}",
    ]
    if scope and scope != "all":
        user_message_parts.append(f"Scope: {scope}")
    if event_source:
        user_message_parts.append(f"Event source: {event_source}")

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

    try:
        # Use a dedicated thread_id to isolate patrol from user conversations
        patrol_thread_id = f"patrol-{patrol_id}"
        token = current_thread_id.set(patrol_thread_id)

        try:
            agent = get_agent()
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
                        "recursion_limit": 50,  # higher limit for patrol scans
                    },
                )

            # Extract findings from the agent's final response
            final_response = result.get("final_response", "")
            if final_response:
                findings = _parse_findings(final_response)
            else:
                # Fallback: check last AIMessage content
                messages = result.get("messages", [])
                for m in reversed(messages):
                    if hasattr(m, "content") and m.content:
                        findings = _parse_findings(str(m.content))
                        break
        finally:
            current_thread_id.reset(token)

    except Exception as exc:
        logger.exception("Patrol %s (%s) failed", patrol_id, patrol_type)
        status = "failed"
        error_msg = str(exc)

    # ── Update log entry with results ──
    completed_at = datetime.now(timezone.utc)
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
                "findings": json.dumps(findings) if findings else None,
                "completed_at": completed_at,
            },
        )
        await session.commit()

    if error_msg:
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

    return patrol_id


def get_patrol_prompt(patrol_type: str) -> str:
    """Return the pre-defined System Prompt for a patrol type."""
    return _PATROL_PROMPTS.get(patrol_type, DAILY_PATROL_PROMPT)


def get_event_prompt(event_source: str) -> str:
    """Return the event-driven System Prompt for a given event source."""
    return _EVENT_PROMPTS.get(event_source, CI_FAILURE_PATROL_PROMPT)


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
    count = result.rowcount or 0
    if count:
        logger.info("Marked %d stale patrol(s) as failed", count)
    return count
