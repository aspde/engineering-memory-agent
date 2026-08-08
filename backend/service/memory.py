"""Memory write service with similarity grading, conflict detection, and merging.

Write path:
  extract_memory() → embed() → similarity_check() → merge or insert
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from backend.db import get_session_factory
from backend.service.embedding_service import get_embedding_provider
from backend.service.entity import normalize_entities
from backend.service.extraction import extract_memory
from backend.service.prompts import get_prompt
from backend.shared.config import current_thread_id

logger = logging.getLogger(__name__)

# Similarity thresholds for grading
MERGE_THRESHOLD = 0.92       # near-duplicate → merge into existing
CONFLICT_CHECK = 0.75        # close enough to check for contradiction
SUPPLEMENT_THRESHOLD = 0.60  # loosely related → mark as supplement
# below 0.60 → unrelated, insert as new


def _content_hash(content: str) -> str:
    """SHA-256 of raw content — the exact-duplicate idempotency key.

    Same raw content (same commit, same doc, retried webhook) produces the
    same hash, so re-ingestion can be skipped before any LLM extraction runs.
    """
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def write_memory(
    content: str,
    source_type: str = "conversation",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract memory from *content*, check against existing ones, merge or insert.

    Idempotent per content hash: if *content* was already ingested, skips
    extraction entirely and returns the existing memory with
    ``action="duplicate"``.  Otherwise proceeds through the similarity
    grading pipeline.

    Returns the final memory record (either the newly inserted one or the
    merged existing one).
    """
    provider = get_embedding_provider()
    session_factory = get_session_factory()

    # 0. Idempotency gate — skip exact duplicates before paying for extraction.
    content_hash = _content_hash(content)
    existing_by_hash = await _find_by_content_hash(content_hash, session_factory)
    if existing_by_hash:
        logger.info(
            "Duplicate content hash %s (source=%s) — skipping write",
            content_hash[:12],
            source_type,
        )
        return {
            "id": str(existing_by_hash["id"]),
            "action": "duplicate",
            "summary": existing_by_hash["summary"],
            "entity_ids": [],
        }

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
        return await _merge_memory(existing, extracted, embedding, source_type, metadata, content_hash)

    elif grade >= CONFLICT_CHECK and existing:
        has_conflict = await _detect_conflict(existing, extracted)
        if has_conflict:
            return await _mark_conflict(existing, extracted, embedding, source_type, metadata, content_hash)
        # Close but not contradictory — supplement
        return await _supplement_memory(existing, extracted, embedding, source_type, metadata, content_hash)

    elif grade >= SUPPLEMENT_THRESHOLD and existing:
        return await _supplement_memory(existing, extracted, embedding, source_type, metadata, content_hash)

    # 5. Unrelated — insert as new
    return await _insert_memory(extracted, embedding, source_type, metadata, content_hash)


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
                  AND deleted_at IS NULL
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


async def _find_by_content_hash(content_hash: str, session_factory) -> dict | None:
    """Find a non-deleted memory already stored for this exact content hash."""
    async with session_factory() as session:
        result = await session.execute(
            text(
                """\
                SELECT id, summary FROM memories
                WHERE content_hash = :hash AND deleted_at IS NULL
                LIMIT 1
                """
            ),
            {"hash": content_hash},
        )
        row = result.fetchone()
        return dict(row._mapping) if row else None


async def _detect_conflict(existing: dict, extracted: dict) -> bool:
    """Ask the LLM whether *extracted* contradicts *existing*.

    Structured output is enforced and retried.  This is correctness-critical:
    on persistent failure the conflict is *not* silently assumed away.
    :class:`LLMStructuredError` propagates, so ``write_memory`` fails rather
    than storing a possibly-contradictory memory unmarked.
    """
    from backend.service.structured import chat_structured

    version, prompt = get_prompt("memory.conflict")
    logger.debug("_detect_conflict: using prompt memory.conflict v%s", version)
    prompt = prompt.format(
        existing_summary=existing["summary"],
        new_summary=extracted["summary"],
    )
    data = await chat_structured(
        [{"role": "user", "content": prompt}],
        json_schema=_CONFLICT_SCHEMA,
        scenario="conflict_detection",
    )
    return bool(data.get("conflict", False))


_CONFLICT_SCHEMA = {
    "type": "object",
    "required": ["conflict"],
    "properties": {"conflict": {"type": "boolean"}},
}


def _merge_entities(existing_entities, new_entities) -> list[dict]:
    """Deduplicate entities by name, preferring the new side on name conflict.

    Shared by the ingestion merge path and conflict resolution (ingestion and
    patrol pipelines) so both produce the same entity-merge semantics.
    """
    seen: dict[str, dict] = {}
    for e in existing_entities or []:
        name = e.get("name", "")
        if name:
            seen[name] = e
    for e in new_entities or []:
        name = e.get("name", "")
        if name:
            seen[name] = e
    return list(seen.values())


def _merge_relations(existing_relations, new_relations) -> list[dict]:
    """Deduplicate relations by (from, to, type).

    Extraction emits these keys (see extraction.extract_relations); a
    subject/predicate/object key would collapse every relation to the first.
    Shared by the ingestion merge path and conflict resolution so both
    pipelines deduplicate relations identically.
    """
    seen: set[tuple[str, str, str]] = set()
    merged: list[dict] = []
    for r in (existing_relations or []) + (new_relations or []):
        key = (r.get("from", ""), r.get("to", ""), r.get("type", ""))
        if key not in seen:
            seen.add(key)
            merged.append(r)
    return merged


async def _merge_memory(existing, extracted, embedding, source_type, metadata, content_hash):
    """Merge new memory into existing one — update summary and entities.

    Fails safe: if the LLM merge call fails, returns the existing memory
    unchanged rather than losing data.
    """
    from backend.service.llm_service import get_llm_provider

    try:
        llm = get_llm_provider()
        version, prompt = get_prompt("memory.merge")
        logger.debug("_merge_memory: using prompt memory.merge v%s", version)
        prompt = prompt.format(
            existing_summary=existing["summary"],
            new_summary=extracted["summary"],
        )
        merged_summary = await llm.chat(
            [{"role": "user", "content": prompt}], scenario="memory_merge", temperature=0.3
        )
        merged_summary = merged_summary.strip()
    except Exception:
        logger.warning("LLM merge failed, keeping existing summary for %s", existing["id"])
        merged_summary = existing["summary"]

    # Merge entities — deduplicate by name, output as a list (not dict)
    merged_entities = _merge_entities(existing.get("entities"), extracted.get("entities"))

    # Merge relations — deduplicate by (from, to, type).
    merged_relations = _merge_relations(existing.get("relations"), extracted.get("relations"))

    # Re-embed the merged summary.  The incoming ``embedding`` was computed
    # over the *new* content's summary; storing it against the merged text
    # would leave semantic search ranking this memory by a vector that no
    # longer represents its stored summary.
    vectors = await get_embedding_provider().embed([merged_summary.strip()])
    merged_embedding = vectors[0]

    session_factory = get_session_factory()
    try:
        async with session_factory() as session:
            # Merge metadata — new keys overwrite existing, preserving old keys
            existing_meta: dict[str, Any] = {}
            if existing.get("meta"):
                if isinstance(existing["meta"], str):
                    try:
                        existing_meta = json.loads(existing["meta"])
                    except (json.JSONDecodeError, TypeError):
                        pass
                elif isinstance(existing["meta"], dict):
                    existing_meta = existing["meta"]
            tid = current_thread_id.get("")
            if tid:
                metadata = (metadata or {}) | {"thread_id": tid}
            merged_meta = existing_meta | (metadata or {})

            await session.execute(
                text(
                    """\
                    UPDATE memories
                    SET summary = :summary,
                        entities = :entities,
                        relations = :relations,
                        embedding = :embedding,
                        meta = :meta,
                        content_hash = :content_hash
                    WHERE id = :id
                    """
                ),
                {
                    "id": existing["id"],
                    "summary": merged_summary.strip(),
                    "entities": json.dumps(merged_entities, ensure_ascii=False),
                    "relations": json.dumps(merged_relations, ensure_ascii=False),
                    "embedding": str(merged_embedding),
                    "meta": json.dumps(merged_meta),
                    "content_hash": content_hash,
                },
            )
            await session.commit()
    except IntegrityError:
        # A concurrent identical write already stored this content_hash on
        # another live memory — the merge reassigns the hash here.  Report
        # that stored memory as the duplicate instead of 500ing on the unique
        # index (same recovery as _insert_memory's race branch).
        winner = await _find_by_content_hash(content_hash, session_factory)
        if winner is None:
            raise
        logger.info(
            "Merge lost content-hash race for %s — reusing existing %s",
            content_hash[:12],
            winner["id"],
        )
        return {
            "id": str(winner["id"]),
            "action": "duplicate",
            "summary": winner["summary"],
            "entity_ids": [],
        }

    # Fire-and-forget entity normalisation (best-effort, non-blocking)
    _schedule_normalization(str(existing["id"]), merged_entities)

    logger.info("Merged into memory %s", existing["id"])
    return {
        "id": str(existing["id"]),
        "action": "merged",
        "summary": merged_summary.strip(),
        "entity_ids": [],
    }


async def _mark_conflict(existing, extracted, embedding, source_type, metadata, content_hash):
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
            "content_hash": content_hash,
        },
    }


# Max concurrent entity-normalisation runs.  Each run does embedding +
# vector search + up to 3 LLM round-trips, so an unbounded burst (e.g. 200
# commits ingested at once) would spawn hundreds of concurrent provider
# calls and slam the connection pool.  Excess runs queue on the semaphore
# instead of being dropped — normalisation is best-effort, but skipping it
# silently would leave entities unlinked.  (``asyncio.Semaphore`` binds no
# loop until first use, so the same instance is safe across the pytest
# function-scoped event loops.)
_NORMALIZATION_MAX_CONCURRENCY = 4
_normalization_semaphore = asyncio.Semaphore(_NORMALIZATION_MAX_CONCURRENCY)


def _schedule_normalization(memory_id: str, entities: list[dict]) -> None:
    """Fire-and-forget entity normalisation — never blocks the caller.

    Runs in a background asyncio Task so that memory-write latency is not
    gated on embedding + vector search + LLM round-trips.  Failures are
    logged but never propagated (spec: normalisation is best-effort).

    Concurrency is bounded by ``_normalization_semaphore``: at most
    ``_NORMALIZATION_MAX_CONCURRENCY`` normalisation runs execute at once,
    the rest queue.  Without this, ingesting a large batch of memories at
    once would open one provider call per entity (embedding + search + up
    to 3 LLM judgements) and exhaust the connection pool.
    """

    async def _run() -> None:
        async with _normalization_semaphore:
            try:
                await normalize_entities(memory_id, entities)
            except Exception:
                logger.exception("Entity normalisation failed for memory %s", memory_id)

    try:
        asyncio.create_task(_run())
    except RuntimeError:
        # No running event loop (e.g. synchronous test context) — skip.
        logger.debug("No event loop available; skipping entity normalisation for %s", memory_id)


async def _supplement_memory(existing, extracted, embedding, source_type, metadata, content_hash):
    """Insert new memory, linked to existing as a supplement."""
    enriched_meta = (metadata or {}) | {
        "supplements": str(existing["id"]),
        "parent_summary": existing["summary"],
    }
    return await _insert_memory(extracted, embedding, source_type, enriched_meta, content_hash)


async def _insert_memory(extracted, embedding, source_type, metadata, content_hash):
    """Insert a fresh memory row (idempotent on content_hash)."""
    tid = current_thread_id.get("")
    if tid:
        metadata = (metadata or {}) | {"thread_id": tid}

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """\
                INSERT INTO memories (source_type, summary, entities, relations, embedding, meta, content_hash)
                VALUES (:source_type, :summary, :entities, :relations, :embedding, :meta, :content_hash)
                ON CONFLICT (content_hash) DO NOTHING
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
                "content_hash": content_hash,
            },
        )
        await session.commit()
        row = result.fetchone()
        if row is None:
            # Lost a race to a concurrent identical write — the unique index
            # rejected our row.  Re-fetch and report the stored one.
            winner = await _find_by_content_hash(content_hash, session_factory)
            logger.info("Concurrent duplicate insert for hash %s — reusing existing", content_hash[:12])
            return {
                "id": str(winner["id"]),
                "action": "duplicate",
                "summary": winner["summary"],
                "entity_ids": [],
            }
        new_id = row[0]

    # 6. Fire-and-forget entity normalisation (best-effort, non-blocking)
    _schedule_normalization(str(new_id), extracted.get("entities") or [])

    logger.info("Inserted new memory %s (source=%s)", new_id, source_type)
    return {
        "id": str(new_id),
        "action": "inserted",
        "summary": extracted["summary"],
        "entity_ids": [],
    }


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
    content_hash = deferred_payload.get("content_hash")

    session_factory = get_session_factory()

    if resolution == "keep_existing":
        return {"id": existing_id, "action": "conflict_resolved", "resolution": "keep_existing"}

    elif resolution == "overwrite":
        try:
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
                            content_hash = :content_hash,
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
                        "content_hash": content_hash,
                    },
                )
                await session.commit()
        except IntegrityError:
            # The overwritten content's hash already lives on another live
            # memory (it got ingested separately while this conflict sat in
            # the queue).  The new content IS stored — report that row as the
            # resolution rather than 500ing on the unique index.
            winner = await _find_by_content_hash(content_hash, session_factory)
            if winner is None:
                raise
            logger.info(
                "Overwrite lost content-hash race for %s — content already stored as %s",
                content_hash[:12],
                winner["id"],
            )
            return {
                "id": str(winner["id"]),
                "action": "duplicate",
                "resolution": "overwrite",
                "summary": winner["summary"],
            }
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
            version, prompt = get_prompt("memory.merge")
            logger.debug("resolve_conflict(merge): using prompt memory.merge v%s", version)
            prompt = prompt.format(
                existing_summary=existing_summary,
                new_summary=extracted["summary"],
            )
            merged_summary = (await llm.chat(
                [{"role": "user", "content": prompt}],
                scenario="memory_merge",
                temperature=0.3,
            )).strip()
        except Exception:
            merged_summary = extracted["summary"]

        # Merge entities from both sides, preferring new on name conflict
        merged_entities = _merge_entities(existing_entities, extracted.get("entities"))

        # Merge relations — deduplicate by (from, to, type).
        merged_relations = _merge_relations(existing_relations, extracted.get("relations"))

        # Re-embed the merged summary (same reasoning as _merge_memory): the
        # stored vector must match the stored text.
        vectors = await get_embedding_provider().embed([merged_summary.strip()])
        merged_embedding = vectors[0]

        try:
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
                            content_hash = :content_hash,
                            updated_at = NOW()
                        WHERE id = :id
                        """
                    ),
                    {
                        "id": existing_id,
                        "summary": merged_summary.strip(),
                        "entities": json.dumps(merged_entities, ensure_ascii=False),
                        "relations": json.dumps(merged_relations, ensure_ascii=False),
                        "embedding": str(merged_embedding),
                        "meta": json.dumps(metadata or {}),
                        "content_hash": content_hash,
                    },
                )
                await session.commit()
        except IntegrityError:
            # Same guard as the overwrite branch — the new content's hash
            # already lives on another live memory; report it as the outcome.
            winner = await _find_by_content_hash(content_hash, session_factory)
            if winner is None:
                raise
            logger.info(
                "Merge lost content-hash race for %s — content already stored as %s",
                content_hash[:12],
                winner["id"],
            )
            return {
                "id": str(winner["id"]),
                "action": "duplicate",
                "resolution": "merge",
                "summary": winner["summary"],
            }
        return {"id": existing_id, "action": "conflict_resolved", "resolution": "merge"}

    elif resolution == "keep_both":
        return await _insert_memory(extracted, embedding, source_type, metadata, content_hash)

    else:
        logger.warning("Unknown conflict resolution '%s', defaulting to keep_existing", resolution)
        return {"id": existing_id, "action": "conflict_resolved", "resolution": "keep_existing"}
