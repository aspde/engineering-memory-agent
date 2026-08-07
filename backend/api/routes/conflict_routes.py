"""Pending memory-conflict resolution — the HITL surface for webhook conflicts.

Webhook/connector deliveries that contradict an existing memory land in the
``pending_conflicts`` queue; a human lists them here and resolves each with
the same four options the agent interrupt offers.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.service.conflicts import list_pending_conflicts, resolve_pending_conflict

_RESOLUTIONS = {"keep_existing", "overwrite", "merge", "keep_both"}

router = APIRouter(prefix="/conflicts", tags=["conflicts"])


class ConflictListItem(BaseModel):
    id: str
    source: str
    source_type: str | None = None
    existing_id: str
    existing_summary: str
    new_summary: str
    status: str
    resolution: str | None = None
    created_at: str | None = None


class ConflictResolveRequest(BaseModel):
    resolution: str


class ConflictResolveResponse(BaseModel):
    id: str
    resolution: str
    outcome: dict[str, Any]


@router.get("", response_model=list[ConflictListItem])
async def get_conflicts(limit: int = 50) -> list[ConflictListItem]:
    """List unresolved memory conflicts awaiting a human decision."""
    return [
        ConflictListItem(**item)
        for item in await list_pending_conflicts(limit=limit)
    ]


@router.post("/{conflict_id}/resolve", response_model=ConflictResolveResponse)
async def resolve_conflict(
    conflict_id: str, req: ConflictResolveRequest
) -> ConflictResolveResponse:
    """Apply a human resolution (keep_existing / overwrite / merge / keep_both)."""
    if req.resolution not in _RESOLUTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown resolution: {req.resolution} — expected one of {sorted(_RESOLUTIONS)}",
        )
    try:
        return await resolve_pending_conflict(conflict_id, req.resolution)
    except ValueError as exc:  # not found / already resolved
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Conflict resolution error: {exc}"
        ) from exc
