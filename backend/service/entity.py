"""Entity normalization service.

After a memory is written, this module normalises the extracted entity
names against the ``entities`` table — linking equivalent names to the
same canonical entity so that cross-memory queries work.

Flow for each extracted entity:
  embed(name) → cosine search top-3 (threshold 0.85)
  → LLM confirmation → link existing or insert new
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text

from backend.db import get_session_factory
from backend.service.embedding_service import get_embedding_provider

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.85
TOP_K_CANDIDATES = 3

_ENTITY_MATCH_PROMPT = """\
Are "{new_name}" and "{existing_name}" the same {entity_type} entity?
Consider: abbreviations, aliases, version-specific references, and common synonyms.

Reply with ONLY a JSON object: {{"match": true}} or {{"match": false}}"""


async def normalize_entities(
    memory_id: str,
    extracted_entities: list[dict[str, Any]],
) -> list[str]:
    """Normalise extracted entity names against the ``entities`` table.

    For each extracted entity, embed its name and search for similar
    existing entities.  When a close match is found, ask the LLM to
    confirm equivalence.  Entities that match an existing record are
    linked; unmatched entities are inserted as new rows.

    Args:
        memory_id: UUID of the memory these entities were extracted from.
        extracted_entities: List of dicts with ``name`` and ``type`` keys.

    Returns:
        List of entity UUIDs that the memory is now linked to.
    """
    if not extracted_entities:
        return []

    provider = get_embedding_provider()
    session_factory = get_session_factory()
    entity_ids: list[str] = []

    for ent in extracted_entities:
        name = ent.get("name", "").strip()
        entity_type = ent.get("type", "concept").strip()
        if not name:
            continue

        try:
            # 1. Embed entity name
            vectors = await provider.embed([name])
            embedding = vectors[0]

            # 2. Search for similar existing entities
            async with session_factory() as session:
                result = await session.execute(
                    text(
                        """\
                        SELECT id, canonical_name, name, type,
                               1 - (embedding <=> :vec ::vector) AS similarity
                        FROM entities
                        WHERE embedding IS NOT NULL
                          AND 1 - (embedding <=> :vec ::vector) > :threshold
                        ORDER BY embedding <=> :vec ::vector
                        LIMIT :limit
                        """
                    ),
                    {
                        "vec": str(embedding),
                        "threshold": SIMILARITY_THRESHOLD,
                        "limit": TOP_K_CANDIDATES,
                    },
                )
                candidates = [dict(row._mapping) for row in result.fetchall()]

            # 3. If candidates exist, ask LLM to confirm
            matched_id: str | None = None
            for candidate in candidates:
                is_match = await _llm_confirm_match(
                    new_name=name,
                    existing_name=candidate["canonical_name"],
                    entity_type=entity_type,
                )
                if is_match:
                    matched_id = str(candidate["id"])
                    logger.info(
                        "Entity '%s' matched existing '%s' (%s)",
                        name,
                        candidate["canonical_name"],
                        matched_id,
                    )
                    break

            # 4. Link or insert
            if matched_id:
                entity_id = matched_id
            else:
                async with session_factory() as session:
                    result = await session.execute(
                        text(
                            """\
                            INSERT INTO entities (name, canonical_name, type, embedding)
                            VALUES (:name, :canonical_name, :type, :embedding)
                            ON CONFLICT (canonical_name, type) DO UPDATE
                                SET name = EXCLUDED.name
                            RETURNING id
                            """
                        ),
                        {
                            "name": name,
                            "canonical_name": name,  # initial canonical = first seen name
                            "type": entity_type,
                            "embedding": str(embedding),
                        },
                    )
                    await session.commit()
                    entity_id = str(result.fetchone()[0])
                logger.info("Created new entity '%s' (%s)", name, entity_id)

            # 5. Link memory → entity
            async with session_factory() as session:
                await session.execute(
                    text(
                        """\
                        INSERT INTO memory_entities (memory_id, entity_id)
                        VALUES (:memory_id, :entity_id)
                        ON CONFLICT DO NOTHING
                        """
                    ),
                    {"memory_id": memory_id, "entity_id": entity_id},
                )
                await session.commit()

            entity_ids.append(entity_id)

        except Exception:
            logger.exception("Failed to normalise entity '%s' — skipping", name)
            continue

    return entity_ids


async def _llm_confirm_match(
    new_name: str,
    existing_name: str,
    entity_type: str,
) -> bool:
    """Ask the LLM whether *new_name* refers to the same entity as *existing_name*.

    Structured output is enforced and retried.  This is correctness-critical:
    on persistent failure the judgement is *not* silently defaulted to
    "no match" (which would create a duplicate entity).
    :class:`LLMStructuredError` propagates; ``normalize_entities`` catches
    it per-entity and skips that entity.
    """
    from backend.service.structured import chat_structured

    prompt = _ENTITY_MATCH_PROMPT.format(
        new_name=new_name,
        existing_name=existing_name,
        entity_type=entity_type,
    )
    data = await chat_structured(
        [{"role": "user", "content": prompt}],
        json_schema=_MATCH_SCHEMA,
        scenario="entity_normalization",
    )
    return bool(data.get("match", False))


_MATCH_SCHEMA = {
    "type": "object",
    "required": ["match"],
    "properties": {"match": {"type": "boolean"}},
}


async def get_entity_by_name(name: str) -> dict | None:
    """Look up an entity by canonical_name or name.

    Returns the entity row as a dict, or None if not found.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                "SELECT e.id, e.name, e.canonical_name, e.type, e.first_seen_at, "
                "       COUNT(me.memory_id)::int AS memory_count "
                "FROM entities e "
                "LEFT JOIN memory_entities me ON me.entity_id = e.id "
                "WHERE e.canonical_name ILIKE :name OR e.name ILIKE :name "
                "GROUP BY e.id "
                "ORDER BY memory_count DESC "
                "LIMIT 1"
            ),
            {"name": name},
        )
        row = result.fetchone()
        if row is None:
            return None
        return dict(row._mapping)


async def get_entity_relations_for_tool(entity_id: str) -> dict:
    """Return a lightweight relations summary for use by the agent tool.

    Returns related entities and recent memories in dict form (not
    Pydantic models — the tool serialises to JSON).
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        # Related entities
        result = await session.execute(
            text(
                """\
                SELECT e2.id, e2.canonical_name AS name, e2.type,
                       COUNT(DISTINCT me2.memory_id)::int AS memory_count
                FROM memory_entities me1
                JOIN memory_entities me2
                  ON me2.memory_id = me1.memory_id AND me2.entity_id != :eid
                JOIN entities e2 ON e2.id = me2.entity_id
                WHERE me1.entity_id = :eid
                GROUP BY e2.id, e2.canonical_name, e2.type
                ORDER BY memory_count DESC
                LIMIT 10
                """
            ),
            {"eid": entity_id},
        )
        related = [
            {
                "id": str(r.id),
                "name": str(r.name),
                "type": str(r.type),
                "memory_count": r.memory_count,
            }
            for r in result.fetchall()
        ]

        # Recent memories
        result2 = await session.execute(
            text(
                """\
                SELECT m.id, m.summary, m.source_type, m.created_at
                FROM memory_entities me
                JOIN memories m ON m.id = me.memory_id
                WHERE me.entity_id = :eid AND m.deleted_at IS NULL
                ORDER BY m.created_at DESC
                LIMIT 5
                """
            ),
            {"eid": entity_id},
        )
        recent = [
            {
                "id": str(r.id),
                "summary": str(r.summary)[:200],
                "source_type": str(r.source_type),
                "created_at": r.created_at.isoformat() if r.created_at else "",
            }
            for r in result2.fetchall()
        ]

    return {"related_entities": related, "recent_memories": recent}


async def get_memory_entities_batch(memory_ids: list[str]) -> dict[str, list[dict]]:
    """Return a mapping of memory_id → list of linked entity summaries.

    For use by the agent to annotate search results with entity info.
    """
    if not memory_ids:
        return {}

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """\
                SELECT me.memory_id, e.id AS entity_id,
                       e.canonical_name, e.type
                FROM memory_entities me
                JOIN entities e ON e.id = me.entity_id
                WHERE me.memory_id = ANY(:ids)
                """
            ),
            {"ids": memory_ids},
        )
        mapping: dict[str, list[dict]] = {mid: [] for mid in memory_ids}
        for row in result.fetchall():
            mid = str(row.memory_id)
            mapping[mid].append({
                "entity_id": str(row.entity_id),
                "canonical_name": str(row.canonical_name),
                "type": str(row.type),
            })
        return mapping


async def normalize_all_existing() -> dict[str, int]:
    """Backfill: normalise entities from all existing memories.

    Iterates over every non-deleted memory that has entities in its
    JSONB column, runs the normalisation pipeline, and links them.

    Returns:
        Dict with ``memories_processed`` and ``entities_linked`` counts.
    """
    session_factory = get_session_factory()

    async with session_factory() as session:
        result = await session.execute(
            text(
                "SELECT id, entities FROM memories "
                "WHERE deleted_at IS NULL AND entities IS NOT NULL "
                "  AND jsonb_typeof(entities) = 'array' "
                "  AND jsonb_array_length(entities) > 0"
            )
        )
        rows = result.fetchall()

    memories_processed = 0
    entities_linked = 0

    for row in rows:
        memory_id = str(row.id)
        entities = row.entities if isinstance(row.entities, list) else []
        if not entities:
            continue
        ids = await normalize_entities(memory_id, entities)
        memories_processed += 1
        entities_linked += len(ids)

    logger.info(
        "Backfill complete: %d memories processed, %d entities linked",
        memories_processed,
        entities_linked,
    )
    return {"memories_processed": memories_processed, "entities_linked": entities_linked}
