"""Tests for the memory soft-delete API — DELETE /api/memory/memories/{id}.

Covers 200 / 404 behavior, GET-after-delete, and that deleted memories are
excluded from search and stats.
"""

import json
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from backend.db import close_db, get_session_factory
from backend.main import app


@pytest.fixture(autouse=True)
async def _dispose_db_engine() -> None:
    """Dispose the async engine after each test.

    pytest-asyncio runs each test in a fresh event loop.  The SQLAlchemy
    engine's asyncpg connections are bound to the loop that created them, so
    pooled connections orphaned by a closed loop break the next test on
    Windows (ProactorEventLoop).  Disposing the engine here keeps every test
    independent and leak-free.
    """
    yield
    await close_db()


@pytest.fixture
async def async_client() -> AsyncClient:
    """Create an async HTTP client for testing FastAPI endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def insert_test_memory(
    session_factory,
    source_type: str = "test",
    summary: str = "Test memory for delete API",
    entities: list | None = None,
    relations: list | None = None,
) -> str:
    """Insert a memory row directly via the DB session; return its UUID string."""
    entities_json = json.dumps(entities or [])
    relations_json = json.dumps(relations or [])
    # Dummy 1024-dim embedding to satisfy the vector(1024) column.
    embedding = "[{}]".format(",".join(["0.1", "0.2"] * 512))

    async with session_factory() as session:
        result = await session.execute(
            text(
                "INSERT INTO memories (source_type, summary, entities, relations, embedding) "
                "VALUES (:source_type, :summary, :entities ::jsonb, :relations ::jsonb, "
                "        :embedding ::vector) "
                "RETURNING id"
            ),
            {
                "source_type": source_type,
                "summary": summary,
                "entities": entities_json,
                "relations": relations_json,
                "embedding": embedding,
            },
        )
        await session.commit()
        return str(result.fetchone()[0])


class TestDeleteMemory:
    @pytest.mark.asyncio
    async def test_delete_memory_returns_200(self, async_client: AsyncClient) -> None:
        """Deleting an existing memory returns 200 with deleted=True."""
        session_factory = get_session_factory()
        memory_id = await insert_test_memory(session_factory)

        response = await async_client.delete(f"/api/memory/memories/{memory_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["deleted"] is True
        assert data["id"] == memory_id

    @pytest.mark.asyncio
    async def test_delete_nonexistent_memory_returns_404(self, async_client: AsyncClient) -> None:
        """Deleting a memory that does not exist returns 404."""
        memory_id = str(uuid4())

        response = await async_client.delete(f"/api/memory/memories/{memory_id}")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_already_deleted_returns_404(self, async_client: AsyncClient) -> None:
        """Deleting a memory that was already soft-deleted returns 404."""
        session_factory = get_session_factory()
        memory_id = await insert_test_memory(session_factory)

        first = await async_client.delete(f"/api/memory/memories/{memory_id}")
        assert first.status_code == 200

        second = await async_client.delete(f"/api/memory/memories/{memory_id}")
        assert second.status_code == 404

    @pytest.mark.asyncio
    async def test_get_deleted_memory_returns_404(self, async_client: AsyncClient) -> None:
        """GET on a soft-deleted memory returns 404."""
        session_factory = get_session_factory()
        memory_id = await insert_test_memory(session_factory)

        delete_resp = await async_client.delete(f"/api/memory/memories/{memory_id}")
        assert delete_resp.status_code == 200

        get_resp = await async_client.get(f"/api/memory/memories/{memory_id}")
        assert get_resp.status_code == 404


class TestSoftDeleteEffects:
    @pytest.mark.asyncio
    async def test_search_excludes_deleted_memory(self, async_client: AsyncClient) -> None:
        """A soft-deleted memory never appears in search results."""
        session_factory = get_session_factory()
        unique_token = f"searchable-token-{uuid4().hex}"
        memory_id = await insert_test_memory(
            session_factory,
            summary=f"Memory uniquely matching {unique_token}",
        )

        delete_resp = await async_client.delete(f"/api/memory/memories/{memory_id}")
        assert delete_resp.status_code == 200

        search_resp = await async_client.post(
            "/api/memory/memories/search",
            json={"query": unique_token, "top_k": 10},
        )
        assert search_resp.status_code == 200
        results = search_resp.json()["results"]
        result_ids = {str(r.get("id")) for r in results}
        assert memory_id not in result_ids

    @pytest.mark.asyncio
    async def test_stats_exclude_deleted_memory(self, async_client: AsyncClient) -> None:
        """A soft-deleted memory is excluded from total_memories stats."""
        session_factory = get_session_factory()
        memory_id = await insert_test_memory(session_factory)

        before_resp = await async_client.get("/api/memory/stats")
        assert before_resp.status_code == 200
        before_total = before_resp.json()["total_memories"]

        delete_resp = await async_client.delete(f"/api/memory/memories/{memory_id}")
        assert delete_resp.status_code == 200

        after_resp = await async_client.get("/api/memory/stats")
        assert after_resp.status_code == 200
        after_total = after_resp.json()["total_memories"]

        assert after_total == before_total - 1
