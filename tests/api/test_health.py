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

    @pytest.mark.asyncio
    async def test_index_html_is_not_cached(self, async_client) -> None:
        # index.html must revalidate on every load so a new deploy's HTML
        # (which references new hashed asset filenames) is picked up
        # immediately instead of serving the stale bundle.
        resp = await async_client.get("/memories")
        assert resp.status_code == 200
        assert resp.headers.get("cache-control") == "no-cache"

    @pytest.mark.asyncio
    async def test_index_html_revalidates_to_304(self, async_client) -> None:
        # no-cache means "revalidate every load": a matching If-None-Match
        # must yield 304 (the fallback uses a FileResponse, which — unlike
        # StaticFiles — doesn't emit 304 itself, so main.py handles it).
        resp = await async_client.get("/memories")
        assert resp.status_code == 200
        etag = resp.headers.get("etag")
        assert etag
        resp304 = await async_client.get(
            "/memories", headers={"If-None-Match": etag}
        )
        assert resp304.status_code == 304

    @pytest.mark.asyncio
    async def test_built_assets_are_long_cached(self, async_client) -> None:
        # Vite hashes asset filenames, so each asset URL is immutable — the
        # browser may cache it for a year and never revalidate; a code change
        # produces a new filename that fetches fresh.  Use the first real
        # built asset (the hash changes every build, so pick it at runtime).
        from pathlib import Path

        assets_dir = (
            Path(__file__).resolve().parents[2] / "frontend" / "dist" / "assets"
        )
        files = sorted(assets_dir.glob("index-*.js"))
        if not files:
            import pytest

            pytest.skip("frontend/dist/assets has no built JS")
        url = f"/assets/{files[0].name}"
        resp = await async_client.get(url)
        assert resp.status_code == 200
        assert (
            resp.headers.get("cache-control") == "public, max-age=31536000, immutable"
        )

    @pytest.mark.asyncio
    async def test_built_assets_revalidate_to_304(self, async_client) -> None:
        from pathlib import Path

        assets_dir = (
            Path(__file__).resolve().parents[2] / "frontend" / "dist" / "assets"
        )
        files = sorted(assets_dir.glob("index-*.js"))
        if not files:
            import pytest

            pytest.skip("frontend/dist/assets has no built JS")
        url = f"/assets/{files[0].name}"
        resp = await async_client.get(url)
        assert resp.status_code == 200
        etag = resp.headers.get("etag")
        assert etag
        resp304 = await async_client.get(url, headers={"If-None-Match": etag})
        assert resp304.status_code == 304
        # The 304 keeps the cache directive so the browser honors the policy.
        assert (
            resp304.headers.get("cache-control")
            == "public, max-age=31536000, immutable"
        )

