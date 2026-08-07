"""Pending-conflict queue — surfacing webhook/connector conflicts for HITL.

The agent path resolves memory conflicts interactively (``interrupt()`` →
human → ``resolve_conflict()``).  Webhook deliveries have no interactive
session, so a conflict detected during ingestion is persisted here and
resolved later by a human through the *same* ``resolve_conflict()`` — the
four options and their semantics are identical:
keep_existing / overwrite / merge / keep_both.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import text

from backend.db import get_session_factory
from backend.service.memory import resolve_conflict

logger = logging.getLogger(__name__)

_RESOLUTIONS = {"keep_existing", "overwrite", "merge", "keep_both"}


async def persist_pending_conflict(source: str, result: dict[str, Any]) -> dict[str, Any]:
    """Persist a ``write_memory`` conflict result for later HITL resolution.

    *result* is the dict returned by ``write_memory`` when it detects a
    conflict (``action == "conflict"``), carrying the ``_deferred`` payload
    that ``resolve_conflict`` needs.  Returns the new pending-conflict row.
    """
    deferred = result.get("_deferred") or {}
    existing_id = result.get("existing_id")
    if not existing_id or not deferred:
        raise ValueError("Conflict result missing existing_id or _deferred")

    session_factory = get_session_factory()
    async with session_factory() as session:
        row = await session.execute(
            text(
                """\
                INSERT INTO pending_conflicts
                    (source, source_type, existing_id, existing_summary,
                     new_summary, deferred)
                VALUES
                    (:source, :source_type, :existing_id, :existing_summary,
                     :new_summary, :deferred ::jsonb)
                ON CONFLICT DO NOTHING
                RETURNING id, created_at
                """
            ),
            {
                "source": source,
                "source_type": deferred.get("source_type"),
                "existing_id": existing_id,
                "existing_summary": result.get("existing_summary", ""),
                "new_summary": result.get("summary", ""),
                "deferred": json.dumps(deferred, ensure_ascii=False),
            },
        )
        new_row = row.fetchone()
        if new_row is None:
            # An identical conflict is already queued (webhook redelivery of
            # the same payload) — the unique index rejected our insert.  Return
            # the existing pending row instead of stacking a duplicate that
            # would multiply HITL work.
            existing = await session.execute(
                text(
                    """\
                    SELECT id, created_at FROM pending_conflicts
                    WHERE existing_id = :existing_id
                      AND status = 'pending'
                      AND (deferred->>'content_hash') IS NOT DISTINCT FROM :content_hash
                    ORDER BY created_at
                    LIMIT 1
                    """
                ),
                {
                    "existing_id": existing_id,
                    "content_hash": deferred.get("content_hash"),
                },
            )
            new_row = existing.fetchone()
        await session.commit()
        if new_row is None:  # duplicate resolved concurrently — retry is safe
            raise RuntimeError(
                "Conflict already queued and resolved concurrently; please retry"
            )

    logger.info(
        "Persisted pending conflict (source=%s, existing=%s) as %s",
        source,
        existing_id,
        new_row[0],
    )
    return {
        "id": str(new_row[0]),
        "source": source,
        "status": "pending",
        "created_at": new_row[1].isoformat() if new_row[1] else None,
    }


async def list_pending_conflicts(limit: int = 50) -> list[dict[str, Any]]:
    """Return unresolved conflicts, newest first."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """\
                SELECT id, source, source_type, existing_id, existing_summary,
                       new_summary, status, resolution, created_at
                FROM pending_conflicts
                WHERE status = 'pending'
                ORDER BY created_at DESC
                LIMIT :limit
                """
            ),
            {"limit": limit},
        )
        rows = result.fetchall()

    return [
        {
            "id": str(r.id),
            "source": r.source,
            "source_type": r.source_type,
            "existing_id": str(r.existing_id),
            "existing_summary": r.existing_summary,
            "new_summary": r.new_summary,
            "status": r.status,
            "resolution": r.resolution,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


async def resolve_pending_conflict(
    conflict_id: str, resolution: str
) -> dict[str, Any]:
    """Apply *resolution* via the shared ``resolve_conflict`` pipeline.

    Marks the pending conflict ``resolved`` on success.  On failure the row
    is left ``pending`` so it can be retried (e.g. after an LLM hiccup).
    """
    if resolution not in _RESOLUTIONS:
        raise ValueError(
            f"Unknown resolution: {resolution} — expected one of {sorted(_RESOLUTIONS)}"
        )

    session_factory = get_session_factory()
    async with session_factory() as session:
        row = await session.execute(
            text(
                """\
                SELECT existing_id, deferred, status
                FROM pending_conflicts WHERE id = :id
                """
            ),
            {"id": conflict_id},
        )
        conflict = row.fetchone()

    if conflict is None:
        raise ValueError(f"Pending conflict {conflict_id} not found")
    if conflict.status != "pending":
        raise ValueError(f"Pending conflict {conflict_id} already resolved")

    existing_id = str(conflict.existing_id)
    # asyncpg parses JSONB columns into Python dicts; tolerate the raw string
    # form for direct-SQL callers.
    deferred = conflict.deferred
    if isinstance(deferred, str):
        deferred = json.loads(deferred)

    # Same pipeline the agent HITL uses — identical options & semantics.
    outcome = await resolve_conflict(resolution, existing_id, deferred)

    async with session_factory() as session:
        await session.execute(
            text(
                """\
                UPDATE pending_conflicts
                SET status = 'resolved', resolution = :resolution,
                    resolved_at = now()
                WHERE id = :id
                """
            ),
            {"id": conflict_id, "resolution": resolution},
        )
        await session.commit()

    logger.info("Pending conflict %s resolved via %s", conflict_id, resolution)
    return {
        "id": conflict_id,
        "resolution": resolution,
        "outcome": outcome,
    }
