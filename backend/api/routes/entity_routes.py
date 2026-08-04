"""Entity query & search API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text

from backend.db import get_session_factory

router = APIRouter(prefix="/entities", tags=["entities"])


# ── Response models ───────────────────────────────────────────────────


class SourceBreakdown(BaseModel):
    source_type: str
    count: int


class EntityProfile(BaseModel):
    id: str
    name: str
    canonical_name: str
    type: str
    memory_count: int
    source_breakdown: list[SourceBreakdown]
    first_seen_at: str


class RelatedEntity(BaseModel):
    entity_id: str
    name: str
    type: str
    relation_type: str
    memory_count: int


class RecentMemory(BaseModel):
    memory_id: str
    summary: str
    source_type: str
    created_at: str


class EntityRelationsResponse(BaseModel):
    entity: EntityProfile
    related_entities: list[RelatedEntity]
    recent_memories: list[RecentMemory]


class EntitySearchResult(BaseModel):
    id: str
    name: str
    canonical_name: str
    type: str
    memory_count: int


class EntitySearchResponse(BaseModel):
    results: list[EntitySearchResult]


# ── Helpers ────────────────────────────────────────────────────────────


async def _get_entity_profile(entity_id: str) -> dict[str, Any] | None:
    """Fetch a single entity with memory count and source breakdown."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        # Entity row + memory count
        result = await session.execute(
            text(
                """\
                SELECT e.id, e.name, e.canonical_name, e.type,
                       e.first_seen_at,
                       COUNT(me.memory_id)::int AS memory_count
                FROM entities e
                LEFT JOIN memory_entities me ON me.entity_id = e.id
                WHERE e.id = :id
                GROUP BY e.id
                """
            ),
            {"id": entity_id},
        )
        row = result.fetchone()
        if row is None:
            return None

        profile = dict(row._mapping)

        # Source breakdown
        result2 = await session.execute(
            text(
                """\
                SELECT m.source_type, COUNT(*)::int AS cnt
                FROM memory_entities me
                JOIN memories m ON m.id = me.memory_id
                WHERE me.entity_id = :id AND m.deleted_at IS NULL
                GROUP BY m.source_type
                ORDER BY cnt DESC
                """
            ),
            {"id": entity_id},
        )
        profile["source_breakdown"] = [
            {"source_type": r[0], "count": r[1]} for r in result2.fetchall()
        ]

        return profile


def _dominant_relation_type(
    relation_types: list[str],
) -> str:
    """Pick the most informative relation type from a list."""
    if not relation_types:
        return "relates_to"
    # Priority order — more specific types win
    priority = ["causes", "contradicts", "supersedes", "depends_on", "part_of", "relates_to"]
    for p in priority:
        if p in relation_types:
            return p
    return relation_types[0]


# ── Routes (order matters: /search must be registered before /{entity_id}) ──


@router.get("/search", response_model=EntitySearchResponse)
async def search_entities(
    q: str = Query(..., min_length=1, description="Search query — matches name or canonical_name"),
    type: str | None = Query(None, description="Filter by entity type"),
) -> EntitySearchResponse:
    """Search entities by name.

    Matches against both ``name`` and ``canonical_name`` using
    case-insensitive substring matching.  Optionally filtered by entity
    type.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        query = """
            SELECT e.id, e.name, e.canonical_name, e.type,
                   COUNT(me.memory_id)::int AS memory_count
            FROM entities e
            LEFT JOIN memory_entities me ON me.entity_id = e.id
            WHERE (e.canonical_name ILIKE :q OR e.name ILIKE :q)
            """
        params: dict[str, Any] = {"q": f"%{q}%"}

        if type:
            query += " AND e.type = :type"
            params["type"] = type

        query += " GROUP BY e.id ORDER BY memory_count DESC LIMIT 20"

        result = await session.execute(text(query), params)
        rows = result.fetchall()

    return EntitySearchResponse(
        results=[
            EntitySearchResult(
                id=str(row.id),
                name=str(row.name),
                canonical_name=str(row.canonical_name),
                type=str(row.type),
                memory_count=row.memory_count,
            )
            for row in rows
        ]
    )


@router.get("/{entity_id}", response_model=EntityProfile)
async def get_entity(entity_id: str) -> EntityProfile:
    """Return a single entity's full profile."""
    profile = await _get_entity_profile(entity_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Entity not found")

    return EntityProfile(
        id=str(profile["id"]),
        name=str(profile["name"]),
        canonical_name=str(profile["canonical_name"]),
        type=str(profile["type"]),
        memory_count=profile["memory_count"],
        source_breakdown=[
            SourceBreakdown(source_type=s["source_type"], count=s["count"])
            for s in profile["source_breakdown"]
        ],
        first_seen_at=profile["first_seen_at"].isoformat() if profile["first_seen_at"] else "",
    )


@router.get("/{entity_id}/relations", response_model=EntityRelationsResponse)
async def get_entity_relations(entity_id: str) -> EntityRelationsResponse:
    """Return one-degree relations for an entity.

    Related entities are those that co-occur in the same memories.
    The relation type is derived from the ``relations`` JSONB column
    of those memories.
    """
    profile = await _get_entity_profile(entity_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="Entity not found")

    session_factory = get_session_factory()
    async with session_factory() as session:
        # Co-occurring entities with shared memory count
        result = await session.execute(
            text(
                """\
                SELECT e2.id AS entity_id, e2.canonical_name AS name, e2.type,
                       COUNT(DISTINCT me2.memory_id)::int AS memory_count
                FROM memory_entities me1
                JOIN memory_entities me2
                  ON me2.memory_id = me1.memory_id
                 AND me2.entity_id != :entity_id
                JOIN entities e2 ON e2.id = me2.entity_id
                WHERE me1.entity_id = :entity_id
                GROUP BY e2.id, e2.canonical_name, e2.type
                ORDER BY memory_count DESC
                LIMIT 20
                """
            ),
            {"entity_id": entity_id},
        )
        related_rows = result.fetchall()

        # Collect all relation types in a single batch query
        rel_ids = [str(row.entity_id) for row in related_rows]
        rel_type_map: dict[str, list[str]] = {rid: [] for rid in rel_ids}
        if rel_ids:
            type_result = await session.execute(
                text(
                    """\
                    SELECT me2.entity_id::text AS rel_id,
                           jsonb_array_elements(m.relations)->>'type' AS rel_type
                    FROM memory_entities me1
                    JOIN memory_entities me2
                      ON me2.memory_id = me1.memory_id
                     AND me2.entity_id = ANY(:rel_ids)
                    JOIN memories m ON m.id = me1.memory_id
                    WHERE me1.entity_id = :entity_id
                      AND m.relations IS NOT NULL
                      AND jsonb_typeof(m.relations) = 'array'
                      AND jsonb_array_length(m.relations) > 0
                    """
                ),
                {"entity_id": entity_id, "rel_ids": rel_ids},
            )
            for row in type_result.fetchall():
                rid = row.rel_id
                rt = row.rel_type
                if rt:
                    rel_type_map.setdefault(rid, []).append(rt)

        related_entities: list[RelatedEntity] = []
        for row in related_rows:
            rel_id = str(row.entity_id)
            rel_name = str(row.name)
            relation_type = _dominant_relation_type(rel_type_map.get(rel_id, []))

            related_entities.append(
                RelatedEntity(
                    entity_id=rel_id,
                    name=rel_name,
                    type=str(row.type),
                    relation_type=relation_type,
                    memory_count=row.memory_count,
                )
            )

        # Recent memories
        mem_result = await session.execute(
            text(
                """\
                SELECT m.id, m.summary, m.source_type, m.created_at
                FROM memory_entities me
                JOIN memories m ON m.id = me.memory_id
                WHERE me.entity_id = :entity_id AND m.deleted_at IS NULL
                ORDER BY m.created_at DESC
                LIMIT 5
                """
            ),
            {"entity_id": entity_id},
        )
        recent_memories: list[RecentMemory] = []
        for mr in mem_result.fetchall():
            recent_memories.append(
                RecentMemory(
                    memory_id=str(mr.id),
                    summary=str(mr.summary)[:200],
                    source_type=str(mr.source_type),
                    created_at=mr.created_at.isoformat() if mr.created_at else "",
                )
            )

    return EntityRelationsResponse(
        entity=EntityProfile(
            id=str(profile["id"]),
            name=str(profile["name"]),
            canonical_name=str(profile["canonical_name"]),
            type=str(profile["type"]),
            memory_count=profile["memory_count"],
            source_breakdown=[
                SourceBreakdown(source_type=s["source_type"], count=s["count"])
                for s in profile["source_breakdown"]
            ],
            first_seen_at=profile["first_seen_at"].isoformat()
            if profile["first_seen_at"]
            else "",
        ),
        related_entities=related_entities,
        recent_memories=recent_memories,
    )
