"""Tests for the memory soft-delete API — DELETE /api/memory/memories/{id}.

Covers 200 / 404 behavior, GET-after-delete, and that deleted memories are
excluded from search and stats.
"""

import json
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from backend.db import get_session_factory
from backend.main import app


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
    embedding: list | None = None,
) -> str:
    """Insert a memory row directly via the DB session; return its UUID string."""
    entities_json = json.dumps(entities or [])
    relations_json = json.dumps(relations or [])
    if embedding is None:
        # Dummy 1024-dim embedding to satisfy the vector(1024) column.
        embedding = [0.1, 0.2] * 512
    embedding_str = "[{}]".format(",".join(str(x) for x in embedding))

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
                "embedding": embedding_str,
            },
        )
        await session.commit()
        return str(result.fetchone()[0])


def embedding_near(stored: list, cos_sim: float) -> list:
    """Return a unit vector whose cosine similarity to *stored* is *cos_sim*.

    Decomposes along an orthonormal basis: ``v_hat = stored / |stored|`` and a
    fixed ``w_hat`` perpendicular to the all-ones direction (which every test
    embedding below is parallel to).  The result
    ``cos_sim * v_hat + sqrt(1 - cos_sim**2) * w_hat`` has unit length and dot
    product ``cos_sim`` with ``v_hat``, so its cosine similarity to *stored* is
    exactly ``cos_sim`` — letting the test pin a new-content embedding into a
    chosen similarity band (e.g. the conflict band [0.72, 0.85)).
    """
    v = [float(x) for x in stored]
    norm = sum(x * x for x in v) ** 0.5
    v_hat = [x / norm for x in v]
    w_hat = [0.0] * len(v)
    if len(v) >= 2:
        w_hat[0] = 1.0 / 2 ** 0.5
        w_hat[1] = -1.0 / 2 ** 0.5
    r = (1 - cos_sim ** 2) ** 0.5
    return [cos_sim * a + r * b for a, b in zip(v_hat, w_hat)]


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
    async def test_search_excludes_deleted_memory(self, async_client: AsyncClient, monkeypatch) -> None:
        """A soft-deleted memory never appears in search results."""
        # Search embeds the query via the real BGE model.  CI has no HF model
        # cache and runs offline, so the loaded model emits a mismatched-dim
        # vector that fails against the vector(1024) column → 500.  Fake the
        # provider with a 1024-dim vector matching the test-data dummy above.
        provider = AsyncMock()
        provider.embed.return_value = [[0.1, 0.2] * 512]
        monkeypatch.setattr(
            "backend.service.retrieval.get_embedding_provider", lambda: provider
        )

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


class TestStatsEntityNormalization:
    """GET /api/memory/stats top_entities must rank *normalized* entities.

    Regression: top_entities scanned the raw memories.entities JSONB names,
    so "PG"/"PostgreSQL"/"Postgres" ranked as three separate entries and
    diverged from total_entities (which counts the normalized entities
    table).  The endpoint now ranks the normalized entities via
    memory_entities, so the two numbers share one source.
    """

    async def _link_entity(
        self, session_factory, memory_id: str, name: str, canonical: str
    ) -> None:
        async with session_factory() as session:
            entity_id = await session.execute(
                text(
                    "INSERT INTO entities (name, canonical_name, type) "
                    "VALUES (:name, :canonical, 'technology') RETURNING id"
                ),
                {"name": name, "canonical": canonical},
            )
            eid = entity_id.fetchone()[0]
            await session.execute(
                text(
                    "INSERT INTO memory_entities (memory_id, entity_id) "
                    "VALUES (:mid, :eid)"
                ),
                {"mid": memory_id, "eid": eid},
            )
            await session.commit()

    @pytest.mark.asyncio
    async def test_top_entities_uses_normalized_names(
        self, async_client: AsyncClient
    ) -> None:
        session_factory = get_session_factory()
        # The memory's raw JSONB carries the un-normalized name "postgres";
        # the normalized entities table stores it as "PostgreSQL".  The
        # endpoint must report the normalized name — and rank by link count.
        memory_id = await insert_test_memory(
            session_factory, entities=[{"name": "postgres", "type": "technology"}]
        )
        await self._link_entity(session_factory, memory_id, "postgres", "PostgreSQL")

        resp = await async_client.get("/api/memory/stats")
        assert resp.status_code == 200
        top = resp.json()["top_entities"]
        names = [e["name"] for e in top]

        assert "PostgreSQL" in names, f"normalized name missing from {names}"
        # The raw JSONB spelling must NOT appear (that is the divergence the
        # regression fixes).
        assert "postgres" not in names, f"raw name leaked into {names}"
        entry = next(e for e in top if e["name"] == "PostgreSQL")
        assert entry["count"] == 1

    @pytest.mark.asyncio
    async def test_top_entities_excludes_deleted_memories(
        self, async_client: AsyncClient
    ) -> None:
        # A soft-deleted memory's entity link must not count toward the rank.
        session_factory = get_session_factory()
        memory_id = await insert_test_memory(
            session_factory, entities=[{"name": "BGE-M3", "type": "technology"}]
        )
        await self._link_entity(session_factory, memory_id, "BGE-M3", "BGE-M3")
        delete_resp = await async_client.delete(f"/api/memory/memories/{memory_id}")
        assert delete_resp.status_code == 200

        resp = await async_client.get("/api/memory/stats")
        top = resp.json()["top_entities"]
        entry = next((e for e in top if e["name"] == "BGE-M3"), None)
        # The JOIN filters m.deleted_at IS NULL, so the only memory linking
        # this entity is soft-deleted and the entity cannot appear.
        assert entry is None, f"deleted memory's entity leaked into {top}"


class TestWriteConflict:
    """POST /api/memory/memories/write must not 500 when the new content lands
    in the conflict-detection band.

    Regression: ``_mark_conflict`` returns a dict without an ``id`` key, and the
    route used to do ``id=result["id"]`` directly — a conflict was an uncaught
    KeyError → 500, and the conflicting content was neither stored nor queued.
    The conflict branch must instead persist to the pending_conflicts queue and
    return conflict semantics with a ``conflict_id``.
    """

    @pytest.mark.asyncio
    async def test_conflict_branch_returns_conflict_and_queues(
        self, async_client: AsyncClient, monkeypatch
    ) -> None:
        # Existing memory whose embedding is the all-ones vector.
        session_factory = get_session_factory()
        stored_embedding = [1.0] * 1024
        memory_id = await insert_test_memory(
            session_factory,
            source_type="test",
            summary="Existing memory about service X",
            embedding=stored_embedding,
        )

        # New-content embedding pinned at ~0.80 cosine-similarity to the stored
        # one — inside the conflict band [0.72, 0.85), below the merge 0.85.
        new_embedding = embedding_near(stored_embedding, 0.80)

        provider = AsyncMock()
        provider.embed.return_value = [new_embedding]
        monkeypatch.setattr(
            "backend.service.memory.get_embedding_provider", lambda: provider
        )
        monkeypatch.setattr(
            "backend.service.memory.extract_memory",
            AsyncMock(
                return_value={
                    "summary": "New content that contradicts existing memory",
                    "entities": [],
                    "relations": [],
                }
            ),
        )
        # The conflict-detection LLM says: contradiction found.
        monkeypatch.setattr(
            "backend.service.memory._detect_conflict", AsyncMock(return_value=True)
        )

        response = await async_client.post(
            "/api/memory/memories/write",
            json={"content": "new content", "source_type": "api", "metadata": {}},
        )

        # No longer a 500: the route returns explicit conflict semantics.
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["action"] == "conflict"
        assert data["conflict_id"]
        assert data["existing_id"] == memory_id
        assert data["id"] is None

        # The conflicting content landed in the pending_conflicts queue (HITL),
        # not silently dropped.
        async with session_factory() as session:
            row = await session.execute(
                text(
                    "SELECT existing_id, status, new_summary FROM pending_conflicts "
                    "WHERE id = :cid"
                ),
                {"cid": data["conflict_id"]},
            )
            pending = row.fetchone()
        assert pending is not None
        assert str(pending[0]) == memory_id
        assert pending[1] == "pending"
        assert pending[2] == "New content that contradicts existing memory"
