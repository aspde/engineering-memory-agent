"""Patrol API routes — trigger, logs, findings."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text

from backend.db import get_session_factory
from backend.service.conflicts import persist_patrol_conflict
from backend.service.patrol import (
    VALID_PATROL_TYPES,
    get_patrol_prompt,
    run_patrol,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/patrol", tags=["patrol"])

# Strong references to in-flight background patrol tasks.
# ``asyncio.create_task`` keeps no strong reference of its own: the loop's
# bookkeeping is weak, so a fire-and-forget manual patrol can be
# garbage-collected at its first await — while it is still running the agent —
# and the run silently vanishes (only a log line would remain).  Holding the
# task here keeps it alive until it finishes; ``_run_and_capture`` drops it in
# its ``finally``.  (Same pattern as ``_normalization_tasks`` in
# ``backend/service/memory.py``.)
_patrol_tasks: set[asyncio.Task] = set()


# ── Request / Response models ──────────────────────────────────────────


class PatrolTriggerRequest(BaseModel):
    patrol_type: str = Field(..., description="daily | weekly | contradiction_scan")
    scope: str | None = Field(default="all", description="all | entity:<name>")


class PatrolTriggerResponse(BaseModel):
    patrol_id: str
    status: str = "accepted"


class FindingDismissRequest(BaseModel):
    finding_id: str


class PatrolConflictQueueResponse(BaseModel):
    conflict_id: str
    status: str = "queued"
    message: str | None = None


class PatrolLogSummary(BaseModel):
    id: str
    patrol_type: str
    trigger: str
    status: str
    finding_count: int
    started_at: str
    completed_at: str | None


class PatrolLogList(BaseModel):
    items: list[PatrolLogSummary]
    total: int


# ── Routes ─────────────────────────────────────────────────────────────


@router.post("/trigger", response_model=PatrolTriggerResponse, status_code=202)
async def trigger_patrol(body: PatrolTriggerRequest):
    """Manually trigger a patrol run.  Runs asynchronously — returns 202 immediately."""
    if body.patrol_type not in VALID_PATROL_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid patrol_type. Must be one of: {', '.join(VALID_PATROL_TYPES)}",
        )

    prompt = get_patrol_prompt(body.patrol_type)

    # Fire-and-forget: run in background, return patrol_id immediately
    async def _run() -> None:
        try:
            await run_patrol(
                patrol_type=body.patrol_type,
                trigger="manual",
                system_prompt=prompt,
                scope=body.scope,
            )
        except Exception:
            logger.exception("Manual patrol (%s) failed", body.patrol_type)

    # Get patrol_id from a quick initial insert in run_patrol — but run_patrol
    # creates its own ID internally.  We need the ID to return to the caller.
    # Strategy: we create a preliminary patrol_id here and pass it through.
    # For simplicity, we just fire the background task and return a tracking note.
    # run_patrol creates its own UUID, so we capture it by wrapping.
    patrol_id_holder: list[str] = []

    async def _run_and_capture() -> None:
        try:
            pid = await run_patrol(
                patrol_type=body.patrol_type,
                trigger="manual",
                system_prompt=prompt,
                scope=body.scope,
            )
            patrol_id_holder.append(pid)
        except Exception:
            logger.exception("Manual patrol (%s) failed", body.patrol_type)
        finally:
            _patrol_tasks.discard(asyncio.current_task())

    task = asyncio.create_task(_run_and_capture())
    _patrol_tasks.add(task)

    # We don't have the patrol_id yet (run_patrol creates it internally).
    # Return a "scheduled" response and let the client poll /logs.
    return PatrolTriggerResponse(
        patrol_id="pending",
        status="accepted",
    )


@router.get("/logs", response_model=PatrolLogList)
async def list_patrol_logs(
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    patrol_type: str | None = Query(default=None),
):
    """List patrol log summaries, newest first.  Does NOT include findings content."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        conditions = ["1=1"]
        params: dict[str, Any] = {"limit": limit, "offset": offset}

        if patrol_type:
            conditions.append("patrol_type = :patrol_type")
            params["patrol_type"] = patrol_type

        where_clause = " AND ".join(conditions)

        # Count total
        total_result = await session.execute(
            text(f"SELECT COUNT(*) FROM patrol_logs WHERE {where_clause}"),
            params,
        )
        total = total_result.scalar() or 0

        # Fetch page
        rows = await session.execute(
            text(
                f"""SELECT id, patrol_type, trigger, status, findings,
                           started_at, completed_at
                    FROM patrol_logs
                    WHERE {where_clause}
                    ORDER BY started_at DESC
                    LIMIT :limit OFFSET :offset"""
            ),
            params,
        )

        items: list[PatrolLogSummary] = []
        for row in rows:
            findings_raw = row.findings
            finding_count = 0
            if findings_raw:
                try:
                    if isinstance(findings_raw, str):
                        findings_data = json.loads(findings_raw)
                    else:
                        findings_data = findings_raw
                    finding_count = sum(
                        len(v)
                        for v in findings_data.values()
                        if isinstance(v, list)
                    )
                except (json.JSONDecodeError, TypeError, AttributeError):
                    pass

            items.append(
                PatrolLogSummary(
                    id=str(row.id),
                    patrol_type=str(row.patrol_type),
                    trigger=str(row.trigger),
                    status=str(row.status),
                    finding_count=finding_count,
                    started_at=row.started_at.isoformat() if row.started_at else "",
                    completed_at=row.completed_at.isoformat() if row.completed_at else None,
                )
            )

    return PatrolLogList(items=items, total=total)


@router.get("/logs/{log_id}")
async def get_patrol_log(log_id: str):
    """Return a single patrol log with full findings."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        row = await session.execute(
            text(
                """SELECT id, patrol_type, trigger, status, findings,
                          dismissed_findings, started_at, completed_at
                   FROM patrol_logs WHERE id = :id"""
            ),
            {"id": log_id},
        )
        result = row.fetchone()
        if result is None:
            raise HTTPException(status_code=404, detail="Patrol log not found")

        findings_raw = result.findings
        if findings_raw and isinstance(findings_raw, str):
            try:
                findings_raw = json.loads(findings_raw)
            except json.JSONDecodeError:
                pass

        dismissed = result.dismissed_findings or []

        return {
            "id": str(result.id),
            "patrol_type": str(result.patrol_type),
            "trigger": str(result.trigger),
            "status": str(result.status),
            "findings": findings_raw,
            "dismissed_findings": [str(d) for d in dismissed],
            "started_at": result.started_at.isoformat() if result.started_at else "",
            "completed_at": result.completed_at.isoformat() if result.completed_at else None,
        }


@router.post("/findings/{log_id}/dismiss")
async def dismiss_finding(log_id: str, body: FindingDismissRequest):
    """Mark a finding as dismissed (adds its key to dismissed_findings array).

    ``finding_id`` is a *finding key*, not a memory UUID — the patrol finding
    JSON carries no per-finding id, so the frontend keys dismissals by
    ``<group>-<index>`` (and contradiction findings by their memory pair).
    The column is TEXT[]; append is idempotent via the containment check.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        # Verify the log exists
        exists = await session.execute(
            text("SELECT id FROM patrol_logs WHERE id = :id"),
            {"id": log_id},
        )
        if exists.fetchone() is None:
            raise HTTPException(status_code=404, detail="Patrol log not found")

        # PostgreSQL array append — idempotent due to array containment check
        await session.execute(
            text(
                """UPDATE patrol_logs
                   SET dismissed_findings = array_append(
                       COALESCE(dismissed_findings, ARRAY[]::text[]),
                       :finding_id
                   )
                   WHERE id = :id
                     AND NOT (:finding_id = ANY(COALESCE(dismissed_findings, ARRAY[]::text[])))"""
            ),
            {"id": log_id, "finding_id": body.finding_id},
        )
        await session.commit()

    return {"ok": True, "log_id": log_id, "dismissed_finding_id": body.finding_id}


@router.post("/findings/{log_id}/conflict", response_model=PatrolConflictQueueResponse)
async def queue_patrol_conflict(log_id: str, finding: dict[str, Any]):
    """Queue a patrol contradiction finding for HITL arbitration.

    The finding is keyed by its memory pair (memory_a_id / memory_b_id) rather
    than a finding index — contradictions carry no id, and an index would drift
    as findings change.  The memory pair is the stable key, matching the
    backend's unordered-pair dedup.  Returns the queued conflict row (or the
    already-queued / already-arbitrated row for duplicates).
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        exists = await session.execute(
            text("SELECT id FROM patrol_logs WHERE id = :id"),
            {"id": log_id},
        )
        if exists.fetchone() is None:
            raise HTTPException(status_code=404, detail="Patrol log not found")

    try:
        queued = await persist_patrol_conflict(log_id, finding)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return PatrolConflictQueueResponse(
        conflict_id=queued["id"],
        status=queued["status"],
        message=(
            None
            if queued["status"] == "queued"
            else (
                "该矛盾已在待处理列表中"
                if queued["status"] == "already_pending"
                else "该矛盾已仲裁过（两者都保留）"
            )
        ),
    )
