"""Connector registry — explicit registration, not plugin scanning.

Every connector is registered at app startup.  If a connector is
missing required configuration (e.g. no API key), it is still
registered but marked as "pending" so the frontend can show its
status.
"""

from __future__ import annotations

from typing import Any

from backend.connectors.base import Connector

# source_type → Connector instance
CONNECTOR_REGISTRY: dict[str, Connector] = {}

# Track per-connector status so the frontend knows which are active.
# Keys are source_type strings; values are "active" | "pending" | "error".
_connector_status: dict[str, str] = {}


def register_connector(source_type: str, connector: Connector, *, status: str = "active") -> None:
    """Register a connector instance.

    Args:
        source_type: The unique key used in ``memories.source_type`` and
            the webhook URL path segment.
        connector: The connector instance.
        status: One of ``"active"``, ``"pending"``, or ``"error"``.
    """
    CONNECTOR_REGISTRY[source_type] = connector
    _connector_status[source_type] = status


def get_connector(source_type: str) -> Connector | None:
    """Return the registered connector for *source_type*, or None."""
    return CONNECTOR_REGISTRY.get(source_type)


def list_connectors() -> list[dict[str, Any]]:
    """Return metadata for every registered connector.

    Used by ``GET /api/connectors`` to power the frontend settings page.
    """
    return [
        {
            "source_type": source_type,
            "display_name": conn.display_name,
            "status": _connector_status.get(source_type, "pending"),
            "batch_mode": conn.batch_mode,
        }
        for source_type, conn in CONNECTOR_REGISTRY.items()
    ]
