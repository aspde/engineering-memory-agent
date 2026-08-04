"""Connector package — unified interface for external data source adapters."""

from backend.connectors.base import Connector
from backend.connectors.registry import (
    CONNECTOR_REGISTRY,
    get_connector,
    list_connectors,
    register_connector,
)

__all__ = [
    "Connector",
    "CONNECTOR_REGISTRY",
    "get_connector",
    "list_connectors",
    "register_connector",
]
