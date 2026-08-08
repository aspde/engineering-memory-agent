"""Tests for the API-key guard on ``/api`` routes (``backend/api/auth.py``).

``APP_ENV=test`` bypasses the guard — that is what lets the existing API
tests run unauthenticated.  These tests flip ``APP_ENV`` to a non-test value
to prove the guard actually rejects unauthenticated / wrong-key requests and
accepts the right key, both at the dependency level (unit) and through the
ASGI app (router wiring).
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from backend.api.auth import require_api_key


# ── Dependency-level (unit) ──────────────────────────────────────────


class TestRequireApiKeyUnit:
    def test_test_env_bypasses_guard(self, monkeypatch) -> None:
        monkeypatch.setenv("APP_ENV", "test")
        monkeypatch.delenv("EMA_API_KEY", raising=False)
        # No header, no configured key — still passes in test env.
        assert require_api_key(authorization=None) is None
        assert require_api_key(authorization="Bearer whatever") is None

    def test_rejects_when_no_key_configured(self, monkeypatch) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.delenv("EMA_API_KEY", raising=False)
        with pytest.raises(HTTPException) as excinfo:
            require_api_key(authorization="Bearer secret")
        assert excinfo.value.status_code == 401
        assert "secret" not in excinfo.value.detail

    def test_rejects_missing_header(self, monkeypatch) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("EMA_API_KEY", "secret-key")
        with pytest.raises(HTTPException) as excinfo:
            require_api_key(authorization=None)
        assert excinfo.value.status_code == 401

    def test_rejects_non_bearer_scheme(self, monkeypatch) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("EMA_API_KEY", "secret-key")
        with pytest.raises(HTTPException) as excinfo:
            require_api_key(authorization="Basic dXNlcjpwYXNz")
        assert excinfo.value.status_code == 401

    def test_rejects_bearer_without_token(self, monkeypatch) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("EMA_API_KEY", "secret-key")
        with pytest.raises(HTTPException) as excinfo:
            require_api_key(authorization="Bearer   ")
        assert excinfo.value.status_code == 401

    def test_rejects_wrong_key(self, monkeypatch) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("EMA_API_KEY", "secret-key")
        with pytest.raises(HTTPException) as excinfo:
            require_api_key(authorization="Bearer wrong-key")
        assert excinfo.value.status_code == 401
        # The rejected token must not be reflected back to the caller.
        assert "wrong-key" not in excinfo.value.detail

    def test_accepts_matching_key(self, monkeypatch) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("EMA_API_KEY", "secret-key")
        assert require_api_key(authorization="Bearer secret-key") is None
        # Bearer scheme is case-insensitive.
        assert require_api_key(authorization="bearer secret-key") is None


# ── Router wiring (integration through the ASGI app) ─────────────────


class TestRouterGuardIntegration:
    """The guard is wired onto every ``/api`` route.

    Uses ``GET /api/connectors`` (registry-only, no DB) so the assertions
    are about auth, not data.
    """

    @pytest.mark.asyncio
    async def test_unauthenticated_request_rejected(
        self, monkeypatch, async_client
    ) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("EMA_API_KEY", "secret-key")

        resp = await async_client.get("/api/connectors")

        assert resp.status_code == 401
        # No internal details leaked in the body.
        assert "secret-key" not in resp.text

    @pytest.mark.asyncio
    async def test_wrong_key_rejected(self, monkeypatch, async_client) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("EMA_API_KEY", "secret-key")

        resp = await async_client.get(
            "/api/connectors",
            headers={"Authorization": "Bearer wrong-key"},
        )

        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_key_accepted(self, monkeypatch, async_client) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("EMA_API_KEY", "secret-key")

        resp = await async_client.get(
            "/api/connectors",
            headers={"Authorization": "Bearer secret-key"},
        )

        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_route_stays_unauthenticated(
        self, monkeypatch, async_client
    ) -> None:
        # /health lives outside api_router — it must keep working without a key
        # so load balancers / probes are unaffected by the new guard.
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("EMA_API_KEY", "secret-key")

        resp = await async_client.get("/health")

        assert resp.status_code == 200
