"""Tests for entity query & search API endpoints."""

from __future__ import annotations

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


async def _make_embedding() -> str:
    """Return a dummy 1024-dim vector as a pgvector string literal."""
    return "[{}]".format(",".join(["0.1", "0.2"] * 512))


async def _insert_entity(
    name: str,
    canonical_name: str | None = None,
    entity_type: str = "technology",
) -> str:
    """Insert an entity row; return its UUID string."""
    emb = await _make_embedding()
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                "INSERT INTO entities (name, canonical_name, type, embedding) "
                "VALUES (:name, :canonical_name, :type, :embedding ::vector) "
                "RETURNING id"
            ),
            {
                "name": name,
                "canonical_name": canonical_name or name,
                "type": entity_type,
                "embedding": emb,
            },
        )
        await session.commit()
        return str(result.fetchone()[0])


async def _insert_memory(
    source_type: str = "test",
    summary: str = "A test memory",
    entities_json: str = "[]",
    relations_json: str = "[]",
) -> str:
    """Insert a memory row; return its UUID string."""
    emb = await _make_embedding()
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                "INSERT INTO memories (source_type, summary, entities, relations, embedding) "
                "VALUES (:st, :summary, :entities ::jsonb, :relations ::jsonb, :emb ::vector) "
                "RETURNING id"
            ),
            {
                "st": source_type,
                "summary": summary,
                "entities": entities_json,
                "relations": relations_json,
                "emb": emb,
            },
        )
        await session.commit()
        return str(result.fetchone()[0])


async def _link_memory_entity(memory_id: str, entity_id: str) -> None:
    """Insert a memory_entities link."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        await session.execute(
            text(
                "INSERT INTO memory_entities (memory_id, entity_id) "
                "VALUES (:mid, :eid) ON CONFLICT DO NOTHING"
            ),
            {"mid": memory_id, "eid": entity_id},
        )
        await session.commit()


def _uniq(prefix: str) -> str:
    """Return a unique name for this test run to avoid cross-test collisions."""
    return f"{prefix}-{uuid4().hex[:8]}"


class TestGetEntity:
    @pytest.mark.asyncio
    async def test_get_entity_returns_profile(self, async_client: AsyncClient) -> None:
        """GET /api/entities/{id} returns full entity profile."""
        name = _uniq("GetEntityProfile")
        eid = await _insert_entity(name)
        mid = await _insert_memory(summary=f"We use {name} for persistence")
        await _link_memory_entity(mid, eid)

        resp = await async_client.get(f"/api/entities/{eid}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == eid
        assert data["canonical_name"] == name
        assert data["type"] == "technology"
        assert data["memory_count"] == 1
        assert len(data["source_breakdown"]) == 1
        assert data["source_breakdown"][0]["source_type"] == "test"

    @pytest.mark.asyncio
    async def test_get_entity_not_found_404(self, async_client: AsyncClient) -> None:
        """GET /api/entities/{nonexistent} returns 404."""
        fake_id = str(uuid4())
        resp = await async_client.get(f"/api/entities/{fake_id}")
        assert resp.status_code == 404


class TestGetEntityRelations:
    @pytest.mark.asyncio
    async def test_relations_returns_structure(self, async_client: AsyncClient) -> None:
        """GET /api/entities/{id}/relations returns entity + related entities."""
        import json

        name1 = _uniq("RelTest1")
        name2 = _uniq("RelTest2")
        e1 = await _insert_entity(name1)
        e2 = await _insert_entity(name2)

        mid = await _insert_memory(
            summary=f"{name1} with {name2} for vector search",
            relations_json=json.dumps([
                {"from": name1, "to": name2, "type": "depends_on"}
            ]),
        )
        await _link_memory_entity(mid, e1)
        await _link_memory_entity(mid, e2)

        resp = await async_client.get(f"/api/entities/{e1}/relations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity"]["id"] == e1
        assert len(data["related_entities"]) >= 1
        rel_ids = [r["entity_id"] for r in data["related_entities"]]
        assert e2 in rel_ids
        assert len(data["recent_memories"]) >= 1

    @pytest.mark.asyncio
    async def test_relations_empty_for_isolated_entity(
        self, async_client: AsyncClient
    ) -> None:
        """An entity with no linked memories has empty relations list."""
        name = _uniq("Orphan")
        eid = await _insert_entity(name)
        resp = await async_client.get(f"/api/entities/{eid}/relations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["related_entities"] == []
        assert data["recent_memories"] == []

    @pytest.mark.asyncio
    async def test_relations_404_for_missing_entity(
        self, async_client: AsyncClient
    ) -> None:
        """GET /api/entities/{nonexistent}/relations returns 404."""
        fake_id = str(uuid4())
        resp = await async_client.get(f"/api/entities/{fake_id}/relations")
        assert resp.status_code == 404


class TestSearchEntities:
    @pytest.mark.asyncio
    async def test_search_by_name(self, async_client: AsyncClient) -> None:
        """GET /api/entities/search?q=... finds matching entities."""
        name = _uniq("SearchTestDB")
        other = _uniq("OtherDB")
        await _insert_entity(name, name)
        await _insert_entity(other, other)

        resp = await async_client.get(f"/api/entities/search?q={name}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) >= 1
        names = [r["canonical_name"] for r in data["results"]]
        assert name in names
        assert other not in names

    @pytest.mark.asyncio
    async def test_search_with_type_filter(self, async_client: AsyncClient) -> None:
        """GET /api/entities/search?q=...&type=technology filters by type."""
        prefix = _uniq("TypeFilter")
        await _insert_entity(f"{prefix}-tech", entity_type="technology")
        await _insert_entity(f"{prefix}-event", entity_type="event")

        resp = await async_client.get(
            f"/api/entities/search?q={prefix}&type=technology"
        )
        assert resp.status_code == 200
        data = resp.json()
        types = {r["type"] for r in data["results"]}
        assert "event" not in types

    @pytest.mark.asyncio
    async def test_search_empty_query_rejected(self, async_client: AsyncClient) -> None:
        """Search without q parameter returns 422."""
        resp = await async_client.get("/api/entities/search")
        assert resp.status_code == 422
