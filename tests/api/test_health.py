"""Tests for the /health liveness endpoint."""

from __future__ import annotations

import pytest


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_ok_when_db_reachable(self, async_client) -> None:
        from backend.db.schema import init_db

        await init_db()
        resp = await async_client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["database"] == "ok"
        # Provider health is reported cheaply: config presence + breaker state.
        assert "llm" in data and "embedding" in data
        assert data["llm"]["circuit"] in ("closed", "open")
        assert data["embedding"]["circuit"] in ("closed", "open", "n/a")
        assert "configured" in data["llm"]
        assert "configured" in data["embedding"]

    @pytest.mark.asyncio
    async def test_health_degraded_when_db_down(self, async_client, monkeypatch) -> None:
        """DB unreachable → 503, so load balancers stop routing to this node."""

        class _BrokenFactory:
            def __call__(self):
                raise ConnectionError("db down")

        monkeypatch.setattr("backend.main.get_session_factory", lambda: _BrokenFactory())

        resp = await async_client.get("/health")

        assert resp.status_code == 503
        assert resp.json()["status"] == "degraded"
        assert resp.json()["database"] == "unreachable"
        # Provider fields are still reported on the degraded path.
        assert "llm" in resp.json() and "embedding" in resp.json()
