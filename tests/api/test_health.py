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


class TestSpaFallbackApiBoundary:
    """The SPA catch-all must not swallow API paths (ADR-011 depends on 404
    signalling a disabled breadth layer; an API typo should never return the
    SPA index as a 200 HTML)."""

    @pytest.mark.asyncio
    async def test_unknown_api_path_returns_404(self, async_client) -> None:
        resp = await async_client.get("/api/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_spa_path_still_serves_frontend(self, async_client) -> None:
        # /memories is a client-side route — the fallback must serve index.html
        # (200) rather than 404, keeping SPA routing intact.
        resp = await async_client.get("/memories")
        assert resp.status_code == 200

