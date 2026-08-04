"""Tests for connector management API — GET /api/connectors and GET /api/connectors/{source}/logs."""

import json

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from backend.connectors.base import Connector
from backend.connectors.registry import (
    CONNECTOR_REGISTRY,
    register_connector,
)
from backend.db import close_db, get_session_factory
from backend.db.schema import init_db
from backend.main import app


# ── Test connector ────────────────────────────────────────────────────


class _ConnA(Connector):
    display_name = "Connector A"

    @property
    def source_type(self) -> str:
        return "conn_a"

    def validate(self, payload: dict) -> bool:
        return True

    def normalize(self, payload: dict) -> str:
        return str(payload)


class _ConnB(Connector):
    display_name = "Connector B"

    @property
    def source_type(self) -> str:
        return "conn_b"

    def validate(self, payload: dict) -> bool:
        return True

    def normalize(self, payload: dict) -> str:
        return str(payload)


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
async def _dispose_db_engine() -> None:
    yield
    await close_db()


@pytest.fixture(autouse=True)
async def _ensure_tables() -> None:
    """Create tables and clean webhook_logs before each test."""
    await init_db()
    # Clean up any log rows left by previous tests
    session_factory = get_session_factory()
    async with session_factory() as session:
        await session.execute(text("DELETE FROM webhook_logs"))
        await session.commit()


@pytest.fixture(autouse=True)
def _setup_registry():
    """Register test connectors, clean up after."""
    register_connector("conn_a", _ConnA(), status="active")
    register_connector("conn_b", _ConnB(), status="pending")
    yield
    CONNECTOR_REGISTRY.clear()


@pytest.fixture
async def async_client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def _insert_log(source: str, status: str, summary: str) -> None:
    session_factory = get_session_factory()
    async with session_factory() as session:
        await session.execute(
            text(
                """\
                INSERT INTO webhook_logs (source, status, payload_summary)
                VALUES (:source, :status, :summary)
                """
            ),
            {"source": source, "status": status, "summary": summary},
        )
        await session.commit()


# ── List connectors ───────────────────────────────────────────────────


class TestListConnectors:
    @pytest.mark.asyncio
    async def test_returns_all_registered_connectors(self, async_client: AsyncClient):
        resp = await async_client.get("/api/connectors")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["connectors"]) == 2

        sources = {c["source_type"] for c in data["connectors"]}
        assert "conn_a" in sources
        assert "conn_b" in sources

    @pytest.mark.asyncio
    async def test_includes_status_and_batch_mode(self, async_client: AsyncClient):
        resp = await async_client.get("/api/connectors")
        data = resp.json()

        a = next(c for c in data["connectors"] if c["source_type"] == "conn_a")
        assert a["status"] == "active"
        assert a["batch_mode"] == "pending"  # default

        b = next(c for c in data["connectors"] if c["source_type"] == "conn_b")
        assert b["status"] == "pending"

    @pytest.mark.asyncio
    async def test_empty_registry_returns_empty_list(self, async_client: AsyncClient):
        CONNECTOR_REGISTRY.clear()
        resp = await async_client.get("/api/connectors")
        assert resp.status_code == 200
        assert resp.json()["connectors"] == []


# ── Connector logs ────────────────────────────────────────────────────


class TestConnectorLogs:
    @pytest.mark.asyncio
    async def test_returns_logs_for_source(self, async_client: AsyncClient):
        await _insert_log("conn_a", "processed", '{"key": "val"}')
        await _insert_log("conn_a", "failed", '{"broken": true}')

        resp = await async_client.get("/api/connectors/conn_a/logs?limit=10")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["logs"]) == 2

    @pytest.mark.asyncio
    async def test_unknown_source_returns_404(self, async_client: AsyncClient):
        resp = await async_client.get("/api/connectors/unknown/logs")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_no_logs_returns_empty(self, async_client: AsyncClient):
        resp = await async_client.get("/api/connectors/conn_a/logs")
        assert resp.status_code == 200
        assert resp.json()["logs"] == []

    @pytest.mark.asyncio
    async def test_respects_limit(self, async_client: AsyncClient):
        for i in range(5):
            await _insert_log("conn_a", "processed", f"payload {i}")

        resp = await async_client.get("/api/connectors/conn_a/logs?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()["logs"]) == 2

    @pytest.mark.asyncio
    async def test_respects_offset(self, async_client: AsyncClient):
        for i in range(5):
            await _insert_log("conn_a", "processed", f"payload {i}")

        resp_all = await async_client.get("/api/connectors/conn_a/logs?limit=10")
        resp_offset = await async_client.get(
            "/api/connectors/conn_a/logs?limit=10&offset=3"
        )
        assert resp_all.status_code == 200
        assert resp_offset.status_code == 200
        # offset=3 should return 2 items (5 total - 3 offset)
        assert len(resp_offset.json()["logs"]) == 2

    @pytest.mark.asyncio
    async def test_log_entry_has_expected_fields(self, async_client: AsyncClient):
        await _insert_log("conn_a", "processed", "test payload")

        resp = await async_client.get("/api/connectors/conn_a/logs?limit=1")
        data = resp.json()
        entry = data["logs"][0]

        assert "id" in entry
        assert entry["source"] == "conn_a"
        assert entry["status"] == "processed"
        assert entry["payload_summary"] == "test payload"
        assert "created_at" in entry
