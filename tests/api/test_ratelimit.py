"""Integration tests for the API rate-limit middleware.

``APP_ENV=test`` bypasses the limiter (same convention as the API-key guard),
so — like ``tests/api/test_auth.py`` — these tests flip ``APP_ENV`` to a
non-test value and shrink the per-tier quotas via the module-level ``config``
singleton to prove the middleware actually rejects over-quota requests.

``GET /api/connectors`` (registry-only, no DB) exercises the general tier;
``GET /api/scenarios`` (registry-only too) is on the chat-tier prefix
(``/api/scenarios``) without running a real scenario.
"""

from __future__ import annotations

import pytest

from backend.api.ratelimit import reset_rate_limits
from backend.shared.config import config


@pytest.fixture(autouse=True)
def _reset_limiter() -> None:
    """Isolate bucket state between cases (each test mutates quotas/keys)."""
    reset_rate_limits()
    yield
    reset_rate_limits()


class TestRateLimitMiddleware:
    @pytest.mark.asyncio
    async def test_test_env_is_not_limited(
        self, monkeypatch, async_client
    ) -> None:
        # APP_ENV stays "test" (set in tests/conftest.py) — auth is bypassed
        # and the limiter must not throttle real handlers even with a tiny quota.
        monkeypatch.setattr(config, "rate_limit_enabled", True)
        monkeypatch.setattr(config, "rate_limit_general_requests", 1)
        monkeypatch.setattr(config, "rate_limit_general_window_seconds", 60)

        for _ in range(3):
            resp = await async_client.get("/api/connectors")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_general_tier_rejects_over_quota(
        self, monkeypatch, async_client
    ) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("EMA_API_KEY", "secret-key")
        monkeypatch.setattr(config, "rate_limit_enabled", True)
        monkeypatch.setattr(config, "rate_limit_general_requests", 3)
        monkeypatch.setattr(config, "rate_limit_general_window_seconds", 60)
        headers = {"Authorization": "Bearer secret-key"}

        for _ in range(3):
            resp = await async_client.get("/api/connectors", headers=headers)
            assert resp.status_code == 200

        resp = await async_client.get("/api/connectors", headers=headers)
        assert resp.status_code == 429
        # Retry-After tells the client when a token refills.
        assert "retry-after" in resp.headers
        assert "请求过于频繁" in resp.text

    @pytest.mark.asyncio
    async def test_chat_tier_uses_stricter_bucket(
        self, monkeypatch, async_client
    ) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("EMA_API_KEY", "secret-key")
        monkeypatch.setattr(config, "rate_limit_enabled", True)
        monkeypatch.setattr(config, "rate_limit_chat_requests", 1)
        monkeypatch.setattr(config, "rate_limit_chat_window_seconds", 60)
        # general quota is huge — only the chat bucket could trip at quota 1.
        monkeypatch.setattr(config, "rate_limit_general_requests", 1000)
        monkeypatch.setattr(config, "rate_limit_general_window_seconds", 60)
        headers = {"Authorization": "Bearer secret-key"}

        # /api/scenarios is on the chat prefix but runs no LLM work (list only).
        resp = await async_client.get("/api/scenarios", headers=headers)
        assert resp.status_code == 200
        resp = await async_client.get("/api/scenarios", headers=headers)
        assert resp.status_code == 429

    @pytest.mark.asyncio
    async def test_anonymous_requests_share_a_bucket(
        self, monkeypatch, async_client
    ) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("EMA_API_KEY", "secret-key")
        monkeypatch.setattr(config, "rate_limit_enabled", True)
        monkeypatch.setattr(config, "rate_limit_general_requests", 1)
        monkeypatch.setattr(config, "rate_limit_general_window_seconds", 60)

        # No Authorization header → the limiter uses the shared "anonymous"
        # bucket, and it runs *before* the auth dependency: once the anonymous
        # bucket is empty the request is refused 429 rather than reaching the
        # auth layer's 401.  This proves unauthenticated requests are still
        # throttled (no unbounded brute-force / cost burn on a bad key).
        assert (await async_client.get("/api/connectors")).status_code == 401
        assert (await async_client.get("/api/connectors")).status_code == 429

    @pytest.mark.asyncio
    async def test_health_route_is_not_limited(
        self, monkeypatch, async_client
    ) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("EMA_API_KEY", "secret-key")
        monkeypatch.setattr(config, "rate_limit_enabled", True)
        monkeypatch.setattr(config, "rate_limit_general_requests", 1)
        monkeypatch.setattr(config, "rate_limit_general_window_seconds", 60)
        headers = {"Authorization": "Bearer secret-key"}

        # Exhaust the general bucket via an /api route.
        assert (
            await async_client.get("/api/connectors", headers=headers)
        ).status_code == 200
        # /health lives outside /api — probes must keep working under load.
        assert (await async_client.get("/health")).status_code == 200

    @pytest.mark.asyncio
    async def test_disabled_limiter_does_not_429(
        self, monkeypatch, async_client
    ) -> None:
        monkeypatch.setenv("APP_ENV", "production")
        monkeypatch.setenv("EMA_API_KEY", "secret-key")
        monkeypatch.setattr(config, "rate_limit_enabled", False)
        monkeypatch.setattr(config, "rate_limit_general_requests", 1)
        monkeypatch.setattr(config, "rate_limit_general_window_seconds", 60)
        headers = {"Authorization": "Bearer secret-key"}

        for _ in range(3):
            resp = await async_client.get("/api/connectors", headers=headers)
            assert resp.status_code == 200
