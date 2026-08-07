"""Tests for the pending-conflict HITL API — GET/POST /api/conflicts."""

import json
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from backend.db import get_session_factory
from backend.main import app


@pytest.fixture(autouse=True)
async def _ensure_tables() -> None:
    """Create tables (including pending_conflicts) before tests."""
    from backend.db.schema import init_db

    await init_db()


@pytest.fixture(autouse=True)
async def _clean_conflict_rows() -> None:
    """Isolate each test from prior runs (the test DB persists between runs)."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        await session.execute(text("DELETE FROM pending_conflicts"))
        await session.execute(text("DELETE FROM webhook_logs"))
        await session.commit()
    yield


@pytest.fixture
async def async_client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def _deferred_payload(existing_id: str) -> dict:
    """A minimal _deferred payload for resolve_conflict(..., 'keep_existing')."""
    return {
        "extracted": {"summary": "New summary", "entities": [], "relations": []},
        "embedding": "[0.1]",
        "source_type": "ci_build",
        "metadata": {"conflicts_with": existing_id},
        "content_hash": "abc123",
    }


async def insert_pending_conflict(
    source: str = "ci",
    new_summary: str = "New summary",
    existing_summary: str = "Existing summary",
) -> str:
    """Insert a pending_conflicts row directly; return its UUID string."""
    existing_id = str(uuid4())
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """\
                INSERT INTO pending_conflicts
                    (source, source_type, existing_id, existing_summary,
                     new_summary, deferred)
                VALUES (:source, :source_type, :existing_id, :existing_summary,
                        :new_summary, :deferred ::jsonb)
                RETURNING id
                """
            ),
            {
                "source": source,
                "source_type": "ci_build",
                "existing_id": existing_id,
                "existing_summary": existing_summary,
                "new_summary": new_summary,
                "deferred": json.dumps(_deferred_payload(existing_id)),
            },
        )
        await session.commit()
        return str(result.fetchone()[0])


class TestListConflicts:
    @pytest.mark.asyncio
    async def test_empty_list(self, async_client: AsyncClient) -> None:
        resp = await async_client.get("/api/conflicts")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_lists_pending_conflicts(self, async_client: AsyncClient) -> None:
        conflict_id = await insert_pending_conflict()

        resp = await async_client.get("/api/conflicts")

        assert resp.status_code == 200
        items = resp.json()
        assert len(items) == 1
        assert items[0]["id"] == conflict_id
        assert items[0]["source"] == "ci"
        assert items[0]["status"] == "pending"
        assert items[0]["new_summary"] == "New summary"


class TestResolveConflict:
    @pytest.mark.asyncio
    async def test_resolve_keep_existing_marks_done(
        self, async_client: AsyncClient
    ) -> None:
        conflict_id = await insert_pending_conflict()

        resp = await async_client.post(
            f"/api/conflicts/{conflict_id}/resolve",
            json={"resolution": "keep_existing"},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == conflict_id
        assert data["resolution"] == "keep_existing"

        # Row is marked resolved; no longer listed as pending
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                text("SELECT status FROM pending_conflicts WHERE id = :id"),
                {"id": conflict_id},
            )
            assert result.fetchone()[0] == "resolved"

        list_resp = await async_client.get("/api/conflicts")
        assert list_resp.json() == []

    @pytest.mark.asyncio
    async def test_resolve_twice_returns_404(self, async_client: AsyncClient) -> None:
        conflict_id = await insert_pending_conflict()
        await async_client.post(
            f"/api/conflicts/{conflict_id}/resolve",
            json={"resolution": "keep_existing"},
        )

        resp = await async_client.post(
            f"/api/conflicts/{conflict_id}/resolve",
            json={"resolution": "overwrite"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_unknown_resolution_returns_400(
        self, async_client: AsyncClient
    ) -> None:
        conflict_id = await insert_pending_conflict()
        resp = await async_client.post(
            f"/api/conflicts/{conflict_id}/resolve",
            json={"resolution": "nuke"},
        )
        assert resp.status_code == 400

    @pytest.mark.asyncio
    async def test_missing_conflict_returns_404(self, async_client: AsyncClient) -> None:
        resp = await async_client.post(
            "/api/conflicts/00000000-0000-0000-0000-000000000000/resolve",
            json={"resolution": "keep_existing"},
        )
        assert resp.status_code == 404


class TestQueueDedup:
    """An identical conflict (webhook redelivery) is queued at most once."""

    @pytest.mark.asyncio
    async def test_identical_conflict_persisted_once(self) -> None:
        from backend.service.conflicts import persist_pending_conflict

        existing_id = str(uuid4())
        result = {
            "action": "conflict",
            "summary": "New summary",
            "existing_id": existing_id,
            "existing_summary": "Existing summary",
            "entities": [],
            "relations": [],
            "_deferred": {
                "extracted": {"summary": "New summary", "entities": [], "relations": []},
                "embedding": "[0.1]",
                "source_type": "ci_build",
                "metadata": {"conflicts_with": existing_id},
                "content_hash": "dedup-test-hash",
            },
        }

        first = await persist_pending_conflict("ci", result)
        second = await persist_pending_conflict("ci", result)

        # Same row returned, not a second queue entry.
        assert first["id"] == second["id"]

        session_factory = get_session_factory()
        async with session_factory() as session:
            row = await session.execute(
                text("SELECT count(*) FROM pending_conflicts WHERE status = 'pending'")
            )
            assert row.fetchone()[0] == 1
