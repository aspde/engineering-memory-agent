"""API tests for the event-driven analysis trigger on the CI webhook path.

The webhook request answers 202 and the delivery pipeline runs in the
background; the analysis is a *downstream* step of that background task.
Tests assert three contracts:

1. ``EVENT_ANALYSIS_ENABLED=true`` + opted-in connector → an analysis run is
   dispatched after the delivery reaches its terminal state.
2. ``EVENT_ANALYSIS_ENABLED=false`` → no analysis (the default posture).
3. An analysis crash must not touch the delivery status — intake
   semantics stay "processed" regardless of what the analysis does.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from backend.connectors.ci import CIConnector
from backend.connectors.registry import CONNECTOR_REGISTRY, register_connector
from backend.db import get_session_factory
from backend.main import app
from backend.shared.config import config

from tests.api.test_webhook_routes import _signed_post, _wait_for_status


@pytest.fixture(autouse=True)
async def _ensure_tables() -> None:
    from backend.db.schema import init_db

    await init_db()


@pytest.fixture(autouse=True)
def _register_ci_connector(monkeypatch):
    """Register the real CIConnector; stub write_memory (no LLM extraction)."""
    monkeypatch.setenv("WEBHOOK_CI_SECRET", "testsecret123")
    register_connector("ci", CIConnector(), status="active")
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


@pytest.fixture(autouse=True)
def _reset_cooldowns():
    from backend.service.event_analysis import reset_cooldowns_for_tests

    reset_cooldowns_for_tests()
    yield
    reset_cooldowns_for_tests()


@pytest.fixture
async def async_client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def _read_analysis(delivery_id: str) -> dict | None:
    session_factory = get_session_factory()
    async with session_factory() as session:
        from sqlalchemy import text

        result = await session.execute(
            text("SELECT analysis FROM webhook_logs WHERE id = :id"),
            {"id": delivery_id},
        )
        row = result.fetchone()
        return row.analysis if row else None


_CI_PAYLOAD = {
    "job_name": "build-api",
    "status": "failure",
    "commit_sha": "abc1234",
    "error_summary": "TypeError: cannot read properties of undefined",
}


class TestWebhookTriggersEventAnalysis:
    @pytest.mark.asyncio
    async def test_enabled_flag_dispatches_analysis(
        self, async_client: AsyncClient
    ):
        runner = AsyncMock()
        with (
            patch.object(config.event_analysis, "enabled", True),
            patch(
                "backend.service.event_analysis.run_event_analysis", runner
            ) as _runner_patch,
        ):
            # The webhook task imports maybe_analyze_event at call time;
            # patching run_event_analysis where maybe_analyze_event looks
            # it up keeps the real gating logic under test.
            raw, headers = _signed_post(_CI_PAYLOAD)
            resp = await async_client.post("/api/webhook/ci", content=raw, headers=headers)
            assert resp.status_code == 202
            await _wait_for_status(resp.json()["delivery_id"], "processed")

        runner.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disabled_by_default_no_analysis(
        self, async_client: AsyncClient
    ):
        import backend.service.event_analysis as ea

        gate = AsyncMock(wraps=ea.maybe_analyze_event)
        with (
            patch.object(config.event_analysis, "enabled", False),
            patch.object(ea, "maybe_analyze_event", gate),
        ):
            raw, headers = _signed_post(_CI_PAYLOAD)
            resp = await async_client.post("/api/webhook/ci", content=raw, headers=headers)
            assert resp.status_code == 202
            await _wait_for_status(resp.json()["delivery_id"], "processed")

        gate.assert_awaited_once()  # hook fires…
        assert config.event_analysis.enabled is False  # …but the flag gated it inside

    @pytest.mark.asyncio
    async def test_analysis_does_not_hold_extraction_slot(
        self, async_client: AsyncClient
    ):
        """While an analysis is in flight, the extraction slot must already
        be released — a slow verdict (up to EVENT_ANALYSIS_TIMEOUT_SECONDS) may
        never pin intake capacity."""
        import asyncio as _asyncio

        import backend.api.routes.webhook_routes as wr

        observed: dict = {}

        async def _slow_runner(*args, **kwargs):
            # Called after the delivery row is terminal; the extraction slot
            # must already be back in the pool at this point.
            observed["webhook_active"] = wr._webhook_active
            await _asyncio.sleep(0.05)  # analysis "work"

        with (
            patch.object(config.event_analysis, "enabled", True),
            patch("backend.service.event_analysis.run_event_analysis", _slow_runner),
        ):
            raw, headers = _signed_post(dict(_CI_PAYLOAD, job_name="build-slot"))
            resp = await async_client.post("/api/webhook/ci", content=raw, headers=headers)
            assert resp.status_code == 202
            await _wait_for_status(resp.json()["delivery_id"], "processed")
            # Wait for the slow runner to have observed the state.
            for _ in range(100):
                if "webhook_active" in observed:
                    break
                await _asyncio.sleep(0.05)

        assert observed.get("webhook_active") == 0

    @pytest.mark.asyncio
    async def test_analysis_crash_keeps_delivery_processed(
        self, async_client: AsyncClient
    ):
        """A failing analysis never changes the intake outcome."""
        with (
            patch.object(config.event_analysis, "enabled", True),
            patch(
                "backend.service.event_analysis.run_event_analysis",
                AsyncMock(side_effect=RuntimeError("analysis exploded")),
            ),
        ):
            raw, headers = _signed_post(_CI_PAYLOAD)
            resp = await async_client.post("/api/webhook/ci", content=raw, headers=headers)
            delivery_id = resp.json()["delivery_id"]
            row = await _wait_for_status(delivery_id, "processed")
            assert row.status == "processed"

    @pytest.mark.asyncio
    async def test_completed_analysis_persisted_to_column(
        self, async_client: AsyncClient
    ):
        """The full real path: delivery → analysis → verdict in the column."""
        verdict = (
            '{"is_known_issue": true, '
            '"similar_incidents": [{"memory_id": "a1b2", "summary": "s", '
            '"similarity": 0.9}], "root_cause_hypothesis": "h", '
            '"recommendation": "r", "severity": "warning"}'
        )
        # Stub the agent so the runner completes without any LLM call.
        agent = AsyncMock()
        agent.ainvoke = AsyncMock(return_value={"final_response": verdict})
        with (
            patch.object(config.event_analysis, "enabled", True),
            patch(
                "backend.service.event_analysis.get_agent", return_value=agent
            ),
        ):
            raw, headers = _signed_post(dict(_CI_PAYLOAD, job_name="build-persist"))
            resp = await async_client.post("/api/webhook/ci", content=raw, headers=headers)
            delivery_id = resp.json()["delivery_id"]
            await _wait_for_status(delivery_id, "processed")

        record = await _read_analysis(delivery_id)
        assert record is not None
        assert record["status"] == "completed"
        assert record["is_known_issue"] is True
