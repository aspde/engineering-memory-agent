"""Tests for webhook routes — POST /api/webhook/{source}.

Webhook processing is async by design: the request answers 202 immediately
and the connector pipeline runs in the background, updating the delivery
log.  Tests therefore assert the fast sync response first, then *poll* the
``webhook_logs`` row until the background task records its outcome.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
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

POLL_TIMEOUT = 5.0
POLL_INTERVAL = 0.05


async def _wait_for_status(
    delivery_id: str, expected: str, timeout: float = POLL_TIMEOUT
) -> tuple:
    """Poll webhook_logs until the delivery reaches *expected*.

    Returns the row; raises AssertionError on timeout or a different
    terminal status.
    """
    row = None
    deadline = time.monotonic() + timeout
    async with get_session_factory()() as session:
        while True:
            result = await session.execute(
                text(
                    "SELECT source, status, memory_id, conflict_id, error "
                    "FROM webhook_logs WHERE id = :id"
                ),
                {"id": delivery_id},
            )
            row = result.fetchone()
            if row is not None and row.status == expected:
                return row
            if row is not None and row.status != "received":
                break
            if time.monotonic() > deadline:
                break
            await asyncio.sleep(POLL_INTERVAL)
    raise AssertionError(
        f"delivery {delivery_id} did not reach status={expected!r}; last={row}"
    )


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


def _signed_post(body: dict, secret: str = "testsecret123") -> dict:
    """Return (content, headers) for a signed POST request."""
    raw = json.dumps(body).encode("utf-8")
    return raw, {
        "Content-Type": "application/json",
        "X-Webhook-Signature": _sign(raw, secret),
    }


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
        raw, _ = _signed_post({"job_name": "test-job"}, secret="wrong-secret")
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
    async def test_valid_signature_accepted_with_delivery_id(
        self, async_client: AsyncClient
    ):
        """A valid delivery answers 202 immediately, then processes."""
        raw, headers = _signed_post({"job_name": "test-job"})
        resp = await async_client.post("/api/webhook/fake_ci", content=raw, headers=headers)

        assert resp.status_code == 202
        data = resp.json()
        assert data["source"] == "fake_ci"
        assert data["status"] == "accepted"
        assert data["delivery_id"]
        assert data["memory_id"] is None  # not known until background finishes

        # The background task eventually records the processed outcome.
        row = await _wait_for_status(data["delivery_id"], "processed")
        assert str(row.memory_id) == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

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
        assert resp.status_code == 202

    @pytest.mark.asyncio
    async def test_no_secret_configured_skips_verification(
        self, async_client: AsyncClient, monkeypatch
    ):
        """When no secret is set, unauthenticated requests succeed (dev)."""
        monkeypatch.delenv("WEBHOOK_FAKE_CI_SECRET", raising=False)
        body = {"job_name": "unauthenticated-job"}
        resp = await async_client.post(
            "/api/webhook/fake_ci",
            json=body,
        )
        # Without secret, signature check is skipped
        assert resp.status_code == 202

    @pytest.mark.asyncio
    async def test_production_rejects_when_no_secret(
        self, async_client: AsyncClient, monkeypatch
    ):
        """In production, a source without a secret is refused (fail closed)."""
        import backend.api.routes.webhook_routes as mod

        monkeypatch.delenv("WEBHOOK_FAKE_CI_SECRET", raising=False)
        monkeypatch.setattr(mod.config, "app_env", "production")
        body = {"job_name": "prod-job"}
        resp = await async_client.post(
            "/api/webhook/fake_ci",
            json=body,
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_production_accepts_when_secret_configured(
        self, async_client: AsyncClient, monkeypatch
    ):
        """Production + valid signature + secret configured → accepted."""
        import backend.api.routes.webhook_routes as mod

        monkeypatch.setattr(mod.config, "app_env", "production")
        raw, headers = _signed_post({"job_name": "prod-signed-job"})
        resp = await async_client.post(
            "/api/webhook/fake_ci", content=raw, headers=headers
        )
        assert resp.status_code == 202


# ── Source / payload errors (stay synchronous) ────────────────────────


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
        raw, headers = _signed_post({"not_job_name": "missing required field"})
        resp = await async_client.post("/api/webhook/fake_ci", content=raw, headers=headers)
        assert resp.status_code == 400


# ── Delivery logging ──────────────────────────────────────────────────


class TestWebhookLogging:
    @pytest.mark.asyncio
    async def test_successful_delivery_is_logged(self, async_client: AsyncClient):
        """A processed webhook updates its delivery log row."""
        raw, headers = _signed_post({"job_name": "logged-job"})
        resp = await async_client.post("/api/webhook/fake_ci", content=raw, headers=headers)
        delivery_id = resp.json()["delivery_id"]

        row = await _wait_for_status(delivery_id, "processed")
        assert row.source == "fake_ci"
        assert str(row.memory_id) == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert row.conflict_id is None
        assert row.error is None

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

    @pytest.mark.asyncio
    async def test_background_failure_is_logged(
        self, async_client: AsyncClient
    ):
        """A failure in the background pipeline marks the delivery failed."""
        body = {"job_name": "boom-job"}
        raw, headers = _signed_post(body)

        # The background task runs after the response returns, so the failure
        # patch must stay installed until the delivery log shows the outcome.
        with patch(
            "backend.service.memory.write_memory",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM provider down"),
        ):
            resp = await async_client.post(
                "/api/webhook/fake_ci", content=raw, headers=headers
            )
            assert resp.status_code == 202
            delivery_id = resp.json()["delivery_id"]

            row = await _wait_for_status(delivery_id, "failed")
        assert row.error == "LLM provider down"

    @pytest.mark.asyncio
    async def test_concurrency_cap_returns_503(
        self, async_client: AsyncClient, monkeypatch
    ):
        """Beyond the in-flight cap the request is refused, not queued."""
        import backend.api.routes.webhook_routes as mod

        monkeypatch.setattr(mod, "_WEBHOOK_MAX_CONCURRENCY", 0)
        session_factory = get_session_factory()
        async with session_factory() as session:
            before = (
                await session.execute(text("SELECT COUNT(*) FROM webhook_logs"))
            ).fetchone()[0]

        raw, headers = _signed_post({"job_name": "overload-job"})
        resp = await async_client.post("/api/webhook/fake_ci", content=raw, headers=headers)
        assert resp.status_code == 503

        # Nothing was dispatched — no new delivery row was created.
        async with session_factory() as session:
            after = (
                await session.execute(text("SELECT COUNT(*) FROM webhook_logs"))
            ).fetchone()[0]
        assert after == before


# ── Trace linkage ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_background_processing_sets_trace_id(async_client) -> None:
    """Every LLM call in a webhook intake is linked to one delivery trace.

    The agent chat and patrol paths already stamp ``current_trace_id``; the
    webhook ingestion pipeline must too, so ``GET /api/usage/trace/{id}`` can
    replay a whole intake (extraction → grading → merge) end-to-end.
    """
    from backend.shared.config import current_trace_id

    captured: list[str] = []

    async def _capture(content, source_type="conversation", metadata=None):
        captured.append(current_trace_id.get())
        return {
            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "action": "inserted",
            "summary": "x",
            "entity_ids": [],
        }

    with patch(
        "backend.service.memory.write_memory",
        new_callable=AsyncMock,
        side_effect=_capture,
    ):
        raw, headers = _signed_post({"job_name": "trace-job"})
        resp = await async_client.post("/api/webhook/fake_ci", content=raw, headers=headers)
        assert resp.status_code == 202
        delivery_id = resp.json()["delivery_id"]
        await _wait_for_status(delivery_id, "processed")

    assert captured == [f"webhook:{delivery_id}"]


# ── Conflict handling ─────────────────────────────────────────────────


class TestWebhookConflict:
    @pytest.mark.asyncio
    async def test_conflict_is_persisted_not_dropped(
        self, async_client: AsyncClient
    ):
        """A webhook whose content conflicts with an existing memory is queued
        for HITL instead of being silently discarded."""
        raw, headers = _signed_post({"job_name": "conflict-job"})

        conflict_result = {
            "action": "conflict",
            "summary": "New conflicting summary",
            "existing_id": "11111111-1111-1111-1111-111111111111",
            "existing_summary": "Existing summary",
            "entities": [],
            "relations": [],
            "_deferred": {
                "extracted": {
                    "summary": "New conflicting summary",
                    "entities": [],
                    "relations": [],
                },
                "embedding": "[0.1]",
                "source_type": "fake_ci",
                "metadata": {"conflicts_with": "11111111-1111-1111-1111-111111111111"},
                "content_hash": "abc123",
            },
        }
        # Override the autouse "inserted" stub for this test.  The background
        # task runs after the response returns, so the override must stay
        # installed until the delivery log shows the conflict outcome.
        with patch(
            "backend.service.memory.write_memory",
            new_callable=AsyncMock,
            return_value=conflict_result,
        ):
            resp = await async_client.post(
                "/api/webhook/fake_ci", content=raw, headers=headers
            )
            assert resp.status_code == 202
            delivery_id = resp.json()["delivery_id"]

            # Background marks the delivery conflict_pending and points at the row.
            log_row = await _wait_for_status(delivery_id, "conflict_pending")
        assert log_row.conflict_id

        # The conflict landed in pending_conflicts, not dropped
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT source, status, existing_summary, new_summary "
                    "FROM pending_conflicts WHERE id = :id"
                ),
                {"id": log_row.conflict_id},
            )
            row = result.fetchone()
            assert row is not None
            assert row.source == "fake_ci"
            assert row.status == "pending"
            assert row.new_summary == "New conflicting summary"


# ── Background task lifetime (GC guard) ───────────────────────────────


@pytest.mark.asyncio
async def test_background_delivery_task_is_held_by_strong_reference(
    async_client: AsyncClient,
) -> None:
    """The fire-and-forget delivery task is tracked, not left to the GC.

    ``asyncio.create_task`` keeps no strong reference: an unreferenced task
    can be garbage-collected at its first await (the 10-40s LLM call), which
    would leave the delivery-log row stuck at ``received`` forever with no
    retry.  The task must be held in a module-level set for its whole lifetime
    and dropped only after it finishes.
    """
    import backend.api.routes.webhook_routes as mod

    release = asyncio.Event()

    async def _blocking_write(content, **kwargs):
        # Hold the pipeline open so the test can inspect the task set while
        # the background delivery is genuinely in flight.
        await release.wait()
        return {
            "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "action": "inserted",
            "summary": "Build held-job failed",
            "entity_ids": [],
        }

    with patch(
        "backend.service.memory.write_memory",
        new_callable=AsyncMock,
        side_effect=_blocking_write,
    ):
        raw, headers = _signed_post({"job_name": "gc-guard-job"})
        resp = await async_client.post(
            "/api/webhook/fake_ci", content=raw, headers=headers
        )
        assert resp.status_code == 202
        delivery_id = resp.json()["delivery_id"]

        # While the task is awaiting the (slow) pipeline, it must be tracked.
        assert mod._webhook_tasks, (
            "background delivery task must be held by a strong reference"
        )

        # Let the pipeline finish; the task then drops its own reference.
        release.set()
        row = await _wait_for_status(delivery_id, "processed")
        assert row is not None

    # The completed delivery task must have dropped its reference from the set.
    deadline = time.monotonic() + 5.0
    while mod._webhook_tasks and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
    assert not mod._webhook_tasks, (
        "completed delivery task must be removed from the task set"
    )
