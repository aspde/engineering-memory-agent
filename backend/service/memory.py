"""Memory write service with similarity grading, conflict detection, and merging.

Write path:
  extract_memory() → embed() → similarity_check() → merge or insert
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sqlalchemy import text

from backend.db import get_session_factory
from backend.service.embedding_service import get_embedding_provider
from backend.service.extraction import extract_memory

logger = logging.getLogger(__name__)

# Similarity thresholds for grading
MERGE_THRESHOLD = 0.92       # near-duplicate → merge into existing
CONFLICT_CHECK = 0.75        # close enough to check for contradiction
SUPPLEMENT_THRESHOLD = 0.60  # loosely related → mark as supplement
# below 0.60 → unrelated, insert as new


async def write_memory(
    content: str,
    source_type: str = "conversation",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract memory from *content*, check against existing ones, merge or insert.

    Returns the final memory record (either the newly inserted one or the
    merged existing one).
    """
    provider = get_embedding_provider()
    session_factory = get_session_factory()

    # 1. Run extraction
    extracted = await extract_memory(content)

    # 2. Embed the summary
    vectors = await provider.embed([extracted["summary"]])
    embedding = vectors[0]

    # 3. Check existing memories for similarity
    grade, existing = await _find_similar(embedding, session_factory)

    logger.info(
        "Similarity grade: %s (thresholds: merge=%.2f, conflict=%.2f, supplement=%.2f)",
        grade,
        MERGE_THRESHOLD,
        CONFLICT_CHECK,
        SUPPLEMENT_THRESHOLD,
    )

    # 4. Act on the grade
    if grade >= MERGE_THRESHOLD and existing:
        return await _merge_memory(existing, extracted, embedding, source_type, metadata)

    elif grade >= CONFLICT_CHECK and existing:
        has_conflict = await _detect_conflict(existing, extracted)
        if has_conflict:
            return await _mark_conflict(existing, extracted, embedding, source_type, metadata)
        # Close but not contradictory — supplement
        return await _supplement_memory(existing, extracted, embedding, source_type, metadata)

    elif grade >= SUPPLEMENT_THRESHOLD and existing:
        return await _supplement_memory(existing, extracted, embedding, source_type, metadata)

    # 5. Unrelated — insert as new
    return await _insert_memory(extracted, embedding, source_type, metadata)


async def _find_similar(embedding, session_factory):
    """Find the most similar existing memory and its similarity grade."""
    async with session_factory() as session:
        result = await session.execute(
            text(
                """\
                SELECT id, summary, entities, relations, recall_count, decay_factor,
                       1 - (embedding <=> :vec ::vector) AS similarity
                FROM memories
                WHERE embedding IS NOT NULL
                  AND 1 - (embedding <=> :vec ::vector) > :threshold
                ORDER BY embedding <=> :vec ::vector
                LIMIT 1
                """
            ),
            {"vec": str(embedding), "threshold": SUPPLEMENT_THRESHOLD},
        )
        row = result.fetchone()
        if row is None:
            return 0.0, None
        return row.similarity, dict(row._mapping)


async def _detect_conflict(existing: dict, extracted: dict) -> bool:
    """Ask the LLM whether *extracted* contradicts *existing*.

    Fails safe: if the LLM call fails, assume no conflict rather than
    blocking the write.
    """
    from backend.service.llm_service import get_llm_provider

    try:
        llm = get_llm_provider()
        prompt = _CONFLICT_PROMPT.format(
            existing_summary=existing["summary"],
            new_summary=extracted["summary"],
        )
        response = await llm.chat([{"role": "user", "content": prompt}])
        return response.strip().lower().startswith("yes")
    except Exception:
        logger.warning("LLM conflict detection failed, assuming no conflict")
        return False


_CONFLICT_PROMPT = """\
You are a conflict detector. Compare two summaries and determine if the new one
CONTRADICTS the existing one.

Existing summary: {existing_summary}

New summary: {new_summary}

Does the new summary CONTRADICT the existing one? Reply with ONLY "Yes" or "No"."""


async def _merge_memory(existing, extracted, embedding, source_type, metadata):
    """Merge new memory into existing one — update summary and entities.

    Fails safe: if the LLM merge call fails, returns the existing memory
    unchanged rather than losing data.
    """
    from backend.service.llm_service import get_llm_provider

    try:
        llm = get_llm_provider()
        prompt = _MERGE_PROMPT.format(
            existing_summary=existing["summary"],
            new_summary=extracted["summary"],
        )
        merged_summary = await llm.chat([{"role": "user", "content": prompt}])
        merged_summary = merged_summary.strip()
    except Exception:
        logger.warning("LLM merge failed, keeping existing summary for %s", existing["id"])
        merged_summary = existing["summary"]

    # Merge entities — deduplicate by name, output as a list (not dict)
    seen: dict[str, dict] = {e["name"]: e for e in (existing.get("entities") or [])}
    for e in (extracted.get("entities") or []):
        name = e.get("name", "")
        if name not in seen:
            seen[name] = e
    merged_entities = list(seen.values())

    # Merge relations
    existing_relations = existing.get("relations") or []
    new_relations = extracted.get("relations") or []
    merged_relations = existing_relations + new_relations

    session_factory = get_session_factory()
    async with session_factory() as session:
        await session.execute(
            text(
                """\
                UPDATE memories
                SET summary = :summary,
                    entities = :entities,
                    relations = :relations,
                    embedding = :embedding,
                    meta = :meta
                WHERE id = :id
                """
            ),
            {
                "id": existing["id"],
                "summary": merged_summary.strip(),
                "entities": json.dumps(merged_entities, ensure_ascii=False),
                "relations": json.dumps(merged_relations, ensure_ascii=False),
                "embedding": str(embedding),
                "meta": json.dumps(metadata or {}),
            },
        )
        await session.commit()

    logger.info("Merged into memory %s", existing["id"])
    return {"id": str(existing["id"]), "action": "merged", "summary": merged_summary.strip()}


_MERGE_PROMPT = """\
Combine the following two summaries into a single concise summary.
Preserve all key facts from both. If they describe the same thing, prefer the more detailed version.

Existing summary: {existing_summary}

New summary: {new_summary}

Merged summary:"""


async def _mark_conflict(existing, extracted, embedding, source_type, metadata):
    """Return conflict data without auto-inserting — defers to HITL.

    The caller (agent check_conflict_node) will pause and let the human
    choose: keep_existing, overwrite, merge, or keep_both.
    """
    return {
        "action": "conflict",
        "summary": extracted["summary"],
        "existing_id": str(existing["id"]),
        "existing_summary": existing["summary"],
        "entities": extracted.get("entities", []) or [],
        "relations": extracted.get("relations", []) or [],
        # Deferred insert payload — passed along so the resolver can act
        "_deferred": {
            "extracted": extracted,
            "embedding": str(embedding),
            "source_type": source_type,
            "metadata": (metadata or {}) | {
                "conflicts_with": str(existing["id"]),
                "conflicting_summary": existing["summary"],
            },
        },
    }


async def _supplement_memory(existing, extracted, embedding, source_type, metadata):
    """Insert new memory, linked to existing as a supplement."""
    enriched_meta = (metadata or {}) | {
        "supplements": str(existing["id"]),
        "parent_summary": existing["summary"],
    }
    return await _insert_memory(extracted, embedding, source_type, enriched_meta)


async def _insert_memory(extracted, embedding, source_type, metadata):
    """Insert a fresh memory row."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """\
                INSERT INTO memories (source_type, summary, entities, relations, embedding, meta)
                VALUES (:source_type, :summary, :entities, :relations, :embedding, :meta)
                RETURNING id
                """
            ),
            {
                "source_type": source_type,
                "summary": extracted["summary"],
                "entities": json.dumps(extracted.get("entities") or [], ensure_ascii=False),
                "relations": json.dumps(extracted.get("relations") or [], ensure_ascii=False),
                "embedding": str(embedding),
                "meta": json.dumps(metadata or {}),
            },
        )
        await session.commit()
        new_id = result.fetchone()[0]

    logger.info("Inserted new memory %s (source=%s)", new_id, source_type)
    return {"id": str(new_id), "action": "inserted", "summary": extracted["summary"]}


async def resolve_conflict(
    resolution: str,
    existing_id: str,
    deferred_payload: dict[str, Any],
) -> dict[str, Any]:
    """Resolve a memory conflict per the human's decision.

    Args:
        resolution: One of ``"keep_existing"``, ``"overwrite"``,
            ``"merge"``, or ``"keep_both"``.
        existing_id: The UUID of the conflicting existing memory.
        deferred_payload: The ``_deferred`` dict carried over from the
            ``write_memory`` conflict return value.

    Returns:
        A dict with ``action`` and ``id`` describing what happened.
    """
    extracted = deferred_payload["extracted"]
    embedding = deferred_payload["embedding"]
    source_type = deferred_payload["source_type"]
    metadata = deferred_payload["metadata"]

    session_factory = get_session_factory()

    if resolution == "keep_existing":
        return {"id": existing_id, "action": "conflict_resolved", "resolution": "keep_existing"}

    elif resolution == "overwrite":
        async with session_factory() as session:
            await session.execute(
                text(
                    """\
                    UPDATE memories
                    SET summary = :summary,
                        entities = :entities,
                        relations = :relations,
                        embedding = :embedding,
                        meta = :meta,
                        updated_at = NOW()
                    WHERE id = :id
                    """
                ),
                {
                    "id": existing_id,
                    "summary": extracted["summary"],
                    "entities": json.dumps(extracted.get("entities") or [], ensure_ascii=False),
                    "relations": json.dumps(extracted.get("relations") or [], ensure_ascii=False),
                    "embedding": str(embedding),
                    "meta": json.dumps(metadata or {}),
                },
            )
            await session.commit()
        return {"id": existing_id, "action": "conflict_resolved", "resolution": "overwrite"}

    elif resolution == "merge":
        from backend.service.llm_service import get_llm_provider

        async with session_factory() as session:
            result = await session.execute(
                text("SELECT summary, entities, relations FROM memories WHERE id = :id"),
                {"id": existing_id},
            )
            row = result.fetchone()
            if row:
                existing_summary = row[0]
                existing_entities: list[dict] = (
                    json.loads(row[1]) if isinstance(row[1], str) else (row[1] or [])
                )
                existing_relations: list[dict] = (
                    json.loads(row[2]) if isinstance(row[2], str) else (row[2] or [])
                )
            else:
                existing_summary = extracted["summary"]
                existing_entities = []
                existing_relations = []

        try:
            llm = get_llm_provider()
            prompt = _MERGE_PROMPT.format(
                existing_summary=existing_summary,
                new_summary=extracted["summary"],
            )
            merged_summary = (await llm.chat([{"role": "user", "content": prompt}])).strip()
        except Exception:
            merged_summary = extracted["summary"]

        # Merge entities from both sides, preferring new on name conflict
        seen: dict[str, dict] = {}
        for e in existing_entities:
            name = e.get("name", "")
            if name:
                seen[name] = e
        for e in (extracted.get("entities") or []):
            name = e.get("name", "")
            if name:
                seen[name] = e
        merged_entities = list(seen.values())

        # Merge relations — deduplicate by (subject, predicate, object)
        rel_seen: set[tuple] = set()
        merged_relations: list[dict] = []
        for r in existing_relations + (extracted.get("relations") or []):
            key = (r.get("subject", ""), r.get("predicate", ""), r.get("object", ""))
            if key not in rel_seen:
                rel_seen.add(key)
                merged_relations.append(r)

        async with session_factory() as session:
            await session.execute(
                text(
                    """\
                    UPDATE memories
                    SET summary = :summary,
                        entities = :entities,
                        relations = :relations,
                        embedding = :embedding,
                        meta = :meta,
                        updated_at = NOW()
                    WHERE id = :id
                    """
                ),
                {
                    "id": existing_id,
                    "summary": merged_summary.strip(),
                    "entities": json.dumps(merged_entities, ensure_ascii=False),
                    "relations": json.dumps(merged_relations, ensure_ascii=False),
                    "embedding": str(embedding),
                    "meta": json.dumps(metadata or {}),
                },
            )
            await session.commit()
        return {"id": existing_id, "action": "conflict_resolved", "resolution": "merge"}

    elif resolution == "keep_both":
        return await _insert_memory(extracted, embedding, source_type, metadata)

    else:
        logger.warning("Unknown conflict resolution '%s', defaulting to keep_existing", resolution)
        return {"id": existing_id, "action": "conflict_resolved", "resolution": "keep_existing"}
