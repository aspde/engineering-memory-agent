"""Pending-conflict queue — surfacing webhook/connector conflicts for HITL.

The agent path resolves memory conflicts interactively (``interrupt()`` →
human → ``resolve_conflict()``).  Webhook deliveries have no interactive
session, so a conflict detected during ingestion is persisted here and
resolved later by a human through the *same* ``resolve_conflict()`` — the
four options and their semantics are identical:
keep_existing / overwrite / merge / keep_both.

Patrol (weekly inspection) contradictions are a second source: two
*already-stored* memories (A, B) that the patrol LLM found to contradict
each other.  They are queued manually (one click per finding from the
Patrol page) with ``conflict_type='patrol'``, and resolved through the same
``resolve_conflict()`` with ``peer_id=B`` — B loses the arbitration
(soft-deleted via ``memories.deleted_at``) and A survives.
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


async def list_pending_conflicts(
    limit: int = 50,
    conflict_type: str | None = None,
    status: str = "pending",
) -> list[dict[str, Any]]:
    """Return conflicts, newest first.

    Defaults to unresolved rows; pass ``status="resolved"`` for the arbitration
    ledger (used by the "reopen" surface for patrol keep_both records).  An
    optional ``conflict_type`` narrows to ``ingestion`` or ``patrol`` rows.
    """
    conditions = ["status = :status"]
    params: dict[str, Any] = {"status": status, "limit": limit}
    if conflict_type:
        conditions.append("conflict_type = :conflict_type")
        params["conflict_type"] = conflict_type
    where_clause = " AND ".join(conditions)

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                f"""\
                SELECT id, source, source_type, existing_id, existing_summary,
                       new_summary, status, resolution, created_at,
                       conflict_type, peer_id
                FROM pending_conflicts
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT :limit
                """
            ),
            params,
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
            "conflict_type": r.conflict_type,
            "peer_id": str(r.peer_id) if r.peer_id else None,
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
                SELECT existing_id, deferred, status, conflict_type, peer_id
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

    # One shared pipeline.  ``peer_id`` is set only for patrol contradictions
    # (both memories already stored): the resolver soft-deletes the losing
    # side (B) as part of the resolution.  Ingestion conflicts pass None and
    # keep the one-stored-memory + one-deferred-new-side semantics.  Both mark
    # the row resolved below — one unified arbitration exit.
    peer_id = str(conflict.peer_id) if conflict.peer_id else None
    outcome = await resolve_conflict(resolution, existing_id, deferred, peer_id=peer_id)

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


async def persist_patrol_conflict(log_id: str, finding: dict[str, Any]) -> dict[str, Any]:
    """Queue a patrol contradiction for later HITL arbitration.

    Patrol contradictions are two *already-stored* memories (A, B).  ``finding``
    carries ``memory_a_id`` / ``memory_b_id``; the full rows are re-read from
    the store so the ``deferred`` payload (needed by ``resolve_conflict``)
    is self-contained and immutable — the memories may change while the
    conflict sits in the queue.

    A is the surviving side (``existing_id``) and B the losing side
    (``peer_id``): keep_existing/overwrite/merge soft-delete B.  Users who
    prefer B over A express it via overwrite (B's content replaces A's) or
    merge (both folded into A) — the four options cover both directions.

    Returns ``{"id", "status", "queued", ...}`` where status is one of
    ``"queued"``, ``"already_pending"`` (same pair already in the queue), or
    ``"already_resolved"`` (this pair was arbitrated before and both memories
    are still live — i.e. the prior choice was keep_both).
    """
    a_id = str(finding.get("memory_a_id") or "").strip()
    b_id = str(finding.get("memory_b_id") or "").strip()
    if not a_id or not b_id:
        raise ValueError("Patrol finding missing memory_a_id or memory_b_id")
    if a_id == b_id:
        raise ValueError("Patrol finding memory_a_id and memory_b_id must differ")

    session_factory = get_session_factory()
    async with session_factory() as session:
        rows = await session.execute(
            text(
                """\
                SELECT id, summary, entities, relations, embedding, source_type,
                       meta, content_hash
                FROM memories
                WHERE id IN (:a_id, :b_id) AND deleted_at IS NULL
                """
            ),
            {"a_id": a_id, "b_id": b_id},
        )
        stored = {str(r.id): dict(r._mapping) for r in rows.fetchall()}
        memory_a = stored.get(a_id)
        memory_b = stored.get(b_id)

    if memory_a is None or memory_b is None:
        raise ValueError(
            "memory_a or memory_b not found or already deleted — patrol finding is stale"
        )
    if memory_a.get("embedding") is None or memory_b.get("embedding") is None:
        raise ValueError(
            "memory_a or memory_b has no embedding — cannot arbitrate; re-embed the memories first"
        )

    # ── Already-arbitrated check (keep_both record) ─────────────────────
    # Resolved rows leave the partial unique index, so a re-run of the same
    # pair must be suppressed explicitly.  Both memories being live (verified
    # above) means the prior resolution did not soft-delete B → keep_both.
    async with session_factory() as session:
        resolved = await session.execute(
            text(
                """\
                SELECT id FROM pending_conflicts
                WHERE conflict_type = 'patrol' AND status = 'resolved'
                  AND LEAST(existing_id, peer_id) = :lo
                  AND GREATEST(existing_id, peer_id) = :hi
                LIMIT 1
                """
            ),
            {
                "lo": min(a_id, b_id),
                "hi": max(a_id, b_id),
            },
        )
        resolved_row = resolved.fetchone()
    if resolved_row is not None:
        logger.info(
            "Patrol conflict (%s, %s) already arbitrated as %s — skipping",
            a_id,
            b_id,
            resolved_row[0],
        )
        return {
            "id": str(resolved_row[0]),
            "status": "already_resolved",
            "queued": False,
        }

    deferred = {
        "kind": "patrol",
        "extracted": {
            "summary": memory_b["summary"],
            "entities": _as_list(memory_b.get("entities")),
            "relations": _as_list(memory_b.get("relations")),
        },
        "embedding": str(memory_b["embedding"]),
        "source_type": memory_b.get("source_type"),
        "metadata": {
            "conflicts_with": a_id,
            "conflicting_summary": memory_a["summary"],
            "peer_id": b_id,
            "peer_summary": memory_b["summary"],
            "conflict_description": finding.get("conflict_description", ""),
            "severity": finding.get("severity", "info"),
            "patrol_log_id": log_id,
        },
        "content_hash": memory_b.get("content_hash"),
    }

    new_summary = memory_b["summary"]
    existing_summary = memory_a["summary"]
    async with session_factory() as session:
        row = await session.execute(
            text(
                """\
                INSERT INTO pending_conflicts
                    (source, source_type, existing_id, existing_summary,
                     new_summary, deferred, conflict_type, peer_id)
                VALUES
                    ('patrol', :source_type, :existing_id, :existing_summary,
                     :new_summary, :deferred ::jsonb, 'patrol', :peer_id)
                ON CONFLICT DO NOTHING
                RETURNING id, created_at
                """
            ),
            {
                "source_type": memory_b.get("source_type"),
                "existing_id": a_id,
                "existing_summary": existing_summary,
                "new_summary": new_summary,
                "deferred": json.dumps(deferred, ensure_ascii=False),
                "peer_id": b_id,
            },
        )
        new_row = row.fetchone()
        already_pending = False
        if new_row is None:
            # Same pair already queued (unique patrol-pair index rejected our
            # insert) — return the pending row instead of stacking a duplicate.
            already_pending = True
            existing = await session.execute(
                text(
                    """\
                    SELECT id, created_at FROM pending_conflicts
                    WHERE conflict_type = 'patrol' AND status = 'pending'
                      AND LEAST(existing_id, peer_id) = :lo
                      AND GREATEST(existing_id, peer_id) = :hi
                    ORDER BY created_at
                    LIMIT 1
                    """
                ),
                {
                    "lo": min(a_id, b_id),
                    "hi": max(a_id, b_id),
                },
            )
            new_row = existing.fetchone()
        await session.commit()
        if new_row is None:  # duplicate resolved concurrently — retry is safe
            raise RuntimeError(
                "Conflict already queued and resolved concurrently; please retry"
            )

    logger.info(
        "Persisted patrol conflict (a=%s, b=%s) as %s",
        a_id,
        b_id,
        new_row[0],
    )
    return {
        "id": str(new_row[0]),
        "source": "patrol",
        "status": "already_pending" if already_pending else "queued",
        "queued": not already_pending,
        "created_at": new_row[1].isoformat() if new_row[1] else None,
    }


async def _require_live_memory(
    session: Any, memory_id: str, role: str
) -> None:
    """Raise ValueError unless *memory_id* is a live (non-deleted) memory row.

    Patrol arbitration resolves against two *already-stored* memories; if the
    surviving side (A) was deleted while the conflict sat in the queue, every
    resolution is meaningless — keep_both would bless a dead pair, and
    overwrite/merge/keep_existing would report success against a gone row.
    """
    row = await session.execute(
        text("SELECT id FROM memories WHERE id = :id AND deleted_at IS NULL"),
        {"id": memory_id},
    )
    if row.fetchone() is None:
        raise ValueError(
            f"{role.capitalize()} memory {memory_id} no longer exists"
        )


class ConflictNotFoundError(ValueError):
    """Raised when a conflict row does not exist (maps to 404, not 409)."""


async def reopen_patrol_conflict(conflict_id: str) -> dict[str, Any]:
    """Reset a resolved *patrol* conflict to pending for re-arbitration.

    Only patrol conflicts can be reopened, and only when both participating
    memories are still live.  The meaningful case is a mistaken keep_both:
    both memories survived, so re-arbitration is safe.  keep_existing/
    overwrite/merge already soft-deleted B, so reopening would resolve against
    a gone memory — refused.  Likewise if A was deleted separately while the
    conflict sat resolved.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        row = await session.execute(
            text(
                """\
                SELECT existing_id, peer_id, status, conflict_type
                FROM pending_conflicts WHERE id = :id
                """
            ),
            {"id": conflict_id},
        )
        conflict = row.fetchone()
        if conflict is None:
            raise ConflictNotFoundError(f"Pending conflict {conflict_id} not found")
        if conflict.status != "resolved":
            raise ValueError(f"Conflict {conflict_id} is not resolved; nothing to reopen")
        if conflict.conflict_type != "patrol":
            raise ValueError(f"Conflict {conflict_id} is not a patrol conflict")

        # Re-arbitration resolves against two *live* memories; either side
        # deleted makes it meaningless.
        await _require_live_memory(session, str(conflict.existing_id), "surviving")

        peer_id = str(conflict.peer_id) if conflict.peer_id else ""
        if peer_id:
            await _require_live_memory(session, peer_id, "losing")

        await session.execute(
            text(
                """\
                UPDATE pending_conflicts
                SET status = 'pending', resolution = NULL, resolved_at = NULL
                WHERE id = :id
                """
            ),
            {"id": conflict_id},
        )
        await session.commit()

    logger.info("Reopened patrol conflict %s for re-arbitration", conflict_id)
    return {"id": conflict_id, "status": "pending"}


def _as_list(value: Any) -> list[Any]:
    """Tolerate JSONB column forms (parsed list, raw JSON string, or None)."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return []
        return parsed if isinstance(parsed, list) else []
    return []
