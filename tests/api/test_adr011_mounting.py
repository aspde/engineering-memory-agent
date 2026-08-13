"""Tests for ADR-011 breadth-layer mounting — unmounted routers return 404.

The ``*_active`` flags fold in an ``APP_ENV=test`` exemption (config.py), so
under the test suite every breadth router is mounted and the unmounted-404
behaviour was never exercised.  These tests rebuild the api_router with the
breadth flags forced off and assert the breadth routes 404 while the core
loop stays mounted.  See backend/api/router.py.
"""

from __future__ import annotations

import importlib

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

import backend.api.router as router_module
from backend.shared.config import config

# Snapshot the flags-on router (built under APP_ENV=test) so this module can
# restore it after each test — a later ``from backend.api.router import
# api_router`` in another module must never pick up a flags-off build.
_ORIGINAL_API_ROUTER = router_module.api_router


def _flags_off_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    connectors_enabled: bool = False,
    scenarios_enabled: bool = False,
    patrol_enabled: bool = False,
) -> FastAPI:
    """Rebuild the api_router with breadth flags off; return a fresh app."""
    # Drop the APP_ENV=test exemption so *_active follows *_enabled.
    monkeypatch.setattr(config, "app_env", "prod")
    monkeypatch.setattr(config, "connectors_enabled", connectors_enabled)
    monkeypatch.setattr(config, "scenarios_enabled", scenarios_enabled)
    monkeypatch.setattr(config, "patrol_enabled", patrol_enabled)
    importlib.reload(router_module)

    app = FastAPI()
    app.include_router(router_module.api_router, prefix="/api")
    return app


@pytest.mark.asyncio
async def test_unmounted_breadth_routers_return_404(monkeypatch) -> None:
    app = _flags_off_app(monkeypatch)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Connectors + webhooks share the ``connectors_enabled`` flag.
            assert (await client.get("/api/connectors")).status_code == 404
            assert (await client.get("/api/connectors/feishu/logs")).status_code == 404
            assert (await client.post("/api/webhook/feishu")).status_code == 404
            # Scenarios and patrol have their own flags.
            assert (await client.get("/api/scenarios")).status_code == 404
            assert (await client.get("/api/patrol/logs")).status_code == 404

            # The core loop stays mounted regardless of the breadth flags.
            assert (await client.get("/api/memory/stats")).status_code == 200
            assert (await client.get("/api/agent/threads")).status_code == 200
    finally:
        # Restore the flags-on router before monkeypatch teardown reverts
        # config — a later test importing backend.main must see the mounted
        # breadth routes again.
        router_module.api_router = _ORIGINAL_API_ROUTER


@pytest.mark.asyncio
async def test_breadth_flags_mount_routers_independently(monkeypatch) -> None:
    """Each breadth layer mounts only when its own flag is set."""
    app = _flags_off_app(monkeypatch, connectors_enabled=True)
    try:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            # Connectors mounted, scenarios/patrol still 404.
            assert (await client.get("/api/connectors")).status_code == 200
            assert (await client.get("/api/scenarios")).status_code == 404
            assert (await client.get("/api/patrol/logs")).status_code == 404
    finally:
        router_module.api_router = _ORIGINAL_API_ROUTER
