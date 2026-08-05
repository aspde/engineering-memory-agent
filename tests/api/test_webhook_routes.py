"""Tests for webhook routes — POST /api/webhook/{source}."""

import hashlib
import hmac
import json
import os
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from backend.connectors.base import Connector
from backend.connectors.registry import (
    CONNECTOR_REGISTRY,
    register_connector,
)
from backend.db import get_session_factory
from backend.main import app


# ── Minimal test connector registered before each test ────────────────


class _FakeConnector(Connector):
    """A test connector that always validates and returns fixed content."""

    display_name = "Fake CI"

    @property
    def source_type(self) -> str:
        return "fake_ci"

    def validate(self, payload: dict) -> bool:
        return "job_name" in payload

    def normalize(self, payload: dict) -> str:
        return f"Build {payload.get('job_name', '?')} failed"

    def build_metadata(self, payload: dict) -> dict:
        return {"job_name": payload.get("job_name", "")}


@pytest.fixture(autouse=True)
async def _ensure_tables() -> None:
    """Create tables (including webhook_logs) before tests."""
    from backend.db.schema import init_db

    await init_db()


@pytest.fixture(autouse=True)
def _register_test_connector(monkeypatch):
    """Register the fake connector before each test, clean up after."""
    # Set a known secret for signature verification tests
    monkeypatch.setenv("WEBHOOK_FAKE_CI_SECRET", "testsecret123")
    register_connector("fake_ci", _FakeConnector(), status="active")
    # Mock write_memory so we don't hit the real embedding pipeline
    with patch(
        "backend.service.memory.write_memory",
        new_callable=AsyncMock,
    ) as mock_write:
        mock_write.return_value = {
            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "action": "inserted",
            "summary": "Build test-job failed",
            "entity_ids": [],
        }
        yield
    CONNECTOR_REGISTRY.clear()


@pytest.fixture
async def async_client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# ── Helpers ───────────────────────────────────────────────────────────


def _sign(body: bytes, secret: str = "testsecret123") -> str:
    """Compute a valid HMAC-SHA256 signature for *body*."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


# ── Signature verification ────────────────────────────────────────────


class TestWebhookSignature:
    @pytest.mark.asyncio
    async def test_missing_signature_returns_401(self, async_client: AsyncClient):
        body = {"job_name": "test-job"}
        resp = await async_client.post(
            "/api/webhook/fake_ci",
            json=body,
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_invalid_signature_returns_401(self, async_client: AsyncClient):
        body = {"job_name": "test-job"}
        raw = json.dumps(body).encode("utf-8")
        resp = await async_client.post(
            "/api/webhook/fake_ci",
            content=raw,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": "sha256=deadbeef",
            },
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_valid_signature_returns_200(self, async_client: AsyncClient):
        body = {"job_name": "test-job"}
        raw = json.dumps(body).encode("utf-8")
        sig = _sign(raw)
        resp = await async_client.post(
            "/api/webhook/fake_ci",
            content=raw,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": sig,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "fake_ci"
        assert data["status"] == "processed"
        assert data["memory_id"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

    @pytest.mark.asyncio
    async def test_hub_signature_256_header_accepted(self, async_client: AsyncClient):
        """X-Hub-Signature-256 is also accepted (GitHub-style)."""
        body = {"job_name": "gh-job"}
        raw = json.dumps(body).encode("utf-8")
        sig = _sign(raw)
        resp = await async_client.post(
            "/api/webhook/fake_ci",
            content=raw,
            headers={
                "Content-Type": "application/json",
                "X-Hub-Signature-256": sig,
            },
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_no_secret_configured_skips_verification(
        self, async_client: AsyncClient, monkeypatch
    ):
        """When no secret is set, unauthenticated requests succeed."""
        monkeypatch.delenv("WEBHOOK_FAKE_CI_SECRET", raising=False)
        body = {"job_name": "unauthenticated-job"}
        resp = await async_client.post(
            "/api/webhook/fake_ci",
            json=body,
        )
        # Without secret, signature check is skipped
        assert resp.status_code == 200


# ── Source / payload errors ───────────────────────────────────────────


class TestWebhookErrors:
    @pytest.mark.asyncio
    async def test_unknown_source_returns_404(self, async_client: AsyncClient, monkeypatch):
        monkeypatch.delenv("WEBHOOK_UNKNOWN_SECRET", raising=False)
        body = {"data": "x"}
        raw = json.dumps(body).encode("utf-8")
        resp = await async_client.post(
            "/api/webhook/unknown",
            content=raw,
            headers={
                "Content-Type": "application/json",
            },
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_json_returns_400(self, async_client: AsyncClient):
        raw = b"not json"
        sig = _sign(raw)
        resp = await async_client.post(
            "/api/webhook/fake_ci",
            content=raw,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": sig,
            },
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_validation_failure_returns_400(self, async_client: AsyncClient):
        """Payload missing required fields → validate() returns False → 400."""
        body = {"not_job_name": "missing required field"}
        raw = json.dumps(body).encode("utf-8")
        sig = _sign(raw)
        resp = await async_client.post(
            "/api/webhook/fake_ci",
            content=raw,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": sig,
            },
        )
        assert resp.status_code == 400


# ── Delivery logging ──────────────────────────────────────────────────


class TestWebhookLogging:
    @pytest.mark.asyncio
    async def test_successful_delivery_is_logged(self, async_client: AsyncClient):
        """A processed webhook creates a row in webhook_logs."""
        body = {"job_name": "logged-job"}
        raw = json.dumps(body).encode("utf-8")
        sig = _sign(raw)
        resp = await async_client.post(
            "/api/webhook/fake_ci",
            content=raw,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": sig,
            },
        )
        assert resp.status_code == 200

        # Verify the log row exists
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                text("SELECT source, status, memory_id FROM webhook_logs ORDER BY created_at DESC LIMIT 1")
            )
            row = result.fetchone()
            assert row is not None
            assert row.source == "fake_ci"
            assert row.status == "processed"
            assert str(row.memory_id) == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

    @pytest.mark.asyncio
    async def test_failed_validation_is_logged(self, async_client: AsyncClient):
        """A validation failure is logged with status='failed'."""
        body = {"not_valid": True}
        raw = json.dumps(body).encode("utf-8")
        sig = _sign(raw)
        resp = await async_client.post(
            "/api/webhook/fake_ci",
            content=raw,
            headers={
                "Content-Type": "application/json",
                "X-Webhook-Signature": sig,
            },
        )
        assert resp.status_code == 400

        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                text("SELECT source, status, error FROM webhook_logs ORDER BY created_at DESC LIMIT 1")
            )
            row = result.fetchone()
            assert row is not None
            assert row.source == "fake_ci"
            assert row.status == "failed"
