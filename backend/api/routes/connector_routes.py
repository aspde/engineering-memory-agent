"""Connector management API — list connectors, view delivery logs."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text

from backend.connectors.registry import get_connector, list_connectors
from backend.db import get_session_factory

router = APIRouter(prefix="/connectors", tags=["connectors"])


# ── Response models ───────────────────────────────────────────────────


class ConnectorInfo(BaseModel):
    source_type: str
    display_name: str
    status: str  # active | pending | error
    batch_mode: str  # supported | pending


class ConnectorListResponse(BaseModel):
    connectors: list[ConnectorInfo]


class DeliveryLogEntry(BaseModel):
    id: str
    source: str
    event_type: str | None
    status: str
    payload_summary: str | None
    memory_id: str | None
    error: str | None
    created_at: str


class DeliveryLogResponse(BaseModel):
    logs: list[DeliveryLogEntry]


# ── Routes ─────────────────────────────────────────────────────────────


@router.get("", response_model=ConnectorListResponse)
async def list_connectors_api() -> ConnectorListResponse:
    """Return every registered connector with its status."""
    items = list_connectors()
    return ConnectorListResponse(
        connectors=[
            ConnectorInfo(
                source_type=c["source_type"],
                display_name=c["display_name"],
                status=c["status"],
                batch_mode=c["batch_mode"],
            )
            for c in items
        ]
    )


@router.get("/{source}/logs", response_model=DeliveryLogResponse)
async def get_connector_logs(
    source: str,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> DeliveryLogResponse:
    """Return recent webhook delivery logs for *source*."""
    connector = get_connector(source)
    if connector is None:
        raise HTTPException(status_code=404, detail=f"Unknown source: {source}")

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """\
                SELECT id, source, event_type, status, payload_summary,
                       memory_id, error, created_at
                FROM webhook_logs
                WHERE source = :source
                ORDER BY created_at DESC
                LIMIT :limit OFFSET :offset
                """
            ),
            {"source": source, "limit": limit, "offset": offset},
        )
        rows = result.fetchall()

    return DeliveryLogResponse(
        logs=[
            DeliveryLogEntry(
                id=str(row.id),
                source=str(row.source),
                event_type=row.event_type,
                status=str(row.status),
                payload_summary=row.payload_summary,
                memory_id=str(row.memory_id) if row.memory_id else None,
                error=row.error,
                created_at=row.created_at.isoformat() if row.created_at else "",
            )
            for row in rows
        ]
    )
