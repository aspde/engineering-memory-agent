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
from backend.model.llm import LLMStructuredError
from backend.service.embedding_service import get_embedding_provider
from backend.service.entity import normalize_entities
from backend.service.extraction import extract_memory
from backend.service.prompts import get_prompt
from backend.shared.config import current_thread_id
from backend.shared.resilience import CircuitOpenError

logger = logging.getLogger(__name__)

# Similarity thresholds for grading.
# Calibrated 2026-08-11 against seed-corpus similarity distributions
# (tests/eval/experiments/threshold_calibration.py): LLM-paraphrase pairs of the same
# knowledge (should MERGE) land at cosine 0.842-0.965 (p25 0.878); distinct
# same-category memories (should NOT merge) top out at 0.792.  The old
# MERGE=0.92 sat above the paraphrase p25, so "the same knowledge written by
# a different source" mostly fell into the conflict band and merge rarely
# fired.  0.85 is the natural separation point (above the same-category max,
# below the duplicate p25).  CONFLICT follows down to keep the band.
# Below SUPPLEMENT → unrelated, insert as new.
MERGE_THRESHOLD = 0.85       # near-duplicate → merge into existing
CONFLICT_CHECK = 0.72        # close enough to check for contradiction
SUPPLEMENT_THRESHOLD = 0.60  # loosely related → mark as supplement


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

    Fail-safe: if conflict detection (a correctness-critical structured call)
    exhausts its retries, the new content is *not* dropped and no contradiction
    is assumed — it is conservatively written as a supplement, so ingestion
    and auto-memory never lose content to a detection hiccup.
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

    # 3. Check existing memories for similarity — top-N closest, so a new
    # memory similar to several existing ones can fan out (merge with the
    # closest, record the others as supplements) instead of only ever touching
    # the single nearest row.
    similar = await _find_similar(embedding, session_factory)

    if similar:
        grade, existing = similar[0]
        # Candidates this content also relates to (below merge grade) — their
        # ids are recorded on the new memory's meta as ``supplements`` so a
        # multi-match is visible instead of silently favouring the closest.
        supplement_ids: list[str] = []
        # Design tradeoff: only the *closest* match ever runs _detect_conflict
        # against the new content.  Additional close candidates sitting in the
        # conflict-detection band [CONFLICT_CHECK, MERGE_THRESHOLD) are recorded
        # as supplements below but are NOT individually vetted for a
        # contradiction — a 2nd/3rd near-relative could contradict the new
        # content and still land in the store.  Vet-and-upgrade-to-conflict for
        # every band candidate was considered and rejected: it would add one
        # structured LLM call (with retries) per candidate, i.e. up to N-1 extra
        # calls per write, to catch a rare event.  This is a documented gap, not
        # a silently-ignored flag — so no ``supplements_unchecked`` marker is
        # written (that field was dead: written but never consumed anywhere).
        for sim, row in similar[1:]:
            rid = str(row["id"])
            if sim < SUPPLEMENT_THRESHOLD or rid == str(existing["id"]):
                continue
            supplement_ids.append(rid)
    else:
        grade, existing = 0.0, None
        supplement_ids = []

    logger.info(
        "Similarity grade: %s (thresholds: merge=%.2f, conflict=%.2f, supplement=%.2f)",
        grade,
        MERGE_THRESHOLD,
        CONFLICT_CHECK,
        SUPPLEMENT_THRESHOLD,
    )

    # Fan-out metadata — when the content supplements several related memories,
    # carry their ids on the meta so the merge/supplement writes record them.
    write_meta = metadata or {}
    if supplement_ids:
        write_meta = {**write_meta, "supplements": supplement_ids}

    # 4. Act on the grade
    if grade >= MERGE_THRESHOLD and existing:
        return await _merge_memory(existing, extracted, embedding, source_type, write_meta, content_hash)

    elif grade >= CONFLICT_CHECK and existing:
        try:
            has_conflict = await _detect_conflict(existing, extracted)
        except (LLMStructuredError, CircuitOpenError):
            # Conflict detection failed — either structured output exhausted
            # its retries (LLMStructuredError) or the circuit breaker is open
            # and chat_structured failed fast (CircuitOpenError).  Fail safe:
            # do NOT assume a contradiction — that would either drop the write
            # (ingestion/auto-memory swallow the error) or misroute it to HITL
            # for a non-conflict — and do not write the new content unmarked
            # into the conflicting memory either.
            # Recording it as a supplement conservatively keeps the content
            # without asserting it contradicts anything.
            logger.exception(
                "Conflict detection failed for memory %s — writing as supplement (fail-safe)",
                existing["id"],
            )
            return await _supplement_memory(existing, extracted, embedding, source_type, write_meta, content_hash)
        if has_conflict:
            return await _mark_conflict(existing, extracted, embedding, source_type, write_meta, content_hash)
        # Close but not contradictory — supplement
        return await _supplement_memory(existing, extracted, embedding, source_type, write_meta, content_hash)

    elif grade >= SUPPLEMENT_THRESHOLD and existing:
        return await _supplement_memory(existing, extracted, embedding, source_type, write_meta, content_hash)

    # 5. Unrelated — insert as new
    return await _insert_memory(extracted, embedding, source_type, write_meta, content_hash)


async def _find_similar(embedding, session_factory, limit: int = 3) -> list[tuple[float, dict]]:
    """Find up to *limit* similar existing memories, closest first.

    Returns a list of ``(similarity, row)`` tuples ordered by similarity
    descending — empty when nothing clears the supplement threshold.  The
    first entry is the closest match used for merge/conflict grading; the
    rest let ``write_memory`` fan out, so a new memory similar to several
    existing ones supplements them all instead of only ever touching the
    single nearest row.
    """
    async with session_factory() as session:
        result = await session.execute(
            text(
                """\
                SELECT id, summary, entities, relations, recall_count,
                       meta, content_hash,
                       1 - (embedding <=> :vec ::vector) AS similarity
                FROM memories
                WHERE embedding IS NOT NULL
                  AND deleted_at IS NULL
                  AND 1 - (embedding <=> :vec ::vector) > :threshold
                ORDER BY embedding <=> :vec ::vector
                LIMIT :limit
                """
            ),
            {"vec": str(embedding), "threshold": SUPPLEMENT_THRESHOLD, "limit": limit},
        )
        rows = result.fetchall()
        if not rows:
            return []
        return [(row.similarity, dict(row._mapping)) for row in rows]


async def _find_by_content_hash(content_hash: str, session_factory) -> dict | None:
    """Find a non-deleted memory already stored for this exact content hash.

    Matches both the live ``content_hash`` column and any hash recorded in
    ``meta.prior_hashes`` — merge/overwrite rewrite the live hash but keep the
    superseded ones there, so re-ingesting the original content still hits the
    idempotency gate instead of re-running extract → grade → merge (version
    drift).
    """
    async with session_factory() as session:
        result = await session.execute(
            text(
                """\
                SELECT id, summary FROM memories
                WHERE deleted_at IS NULL
                  AND (content_hash = :hash OR meta->'prior_hashes' ? :hash)
                LIMIT 1
                """
            ),
            {"hash": content_hash},
        )
        row = result.fetchone()
        return dict(row._mapping) if row else None


async def _detect_conflict(existing: dict, extracted: dict) -> bool:
    """Ask the LLM whether *extracted* contradicts *existing*.

    Structured output is enforced and retried.  On persistent failure
    :class:`LLMStructuredError` propagates; when the circuit breaker is open
    :class:`CircuitOpenError` fails fast and propagates too.  ``write_memory``
    catches both and degrades to a supplement write: detection failed ⇒ never
    assume a contradiction (which would drop or misroute the content), but do
    not write the new content unmarked into the conflicting memory either.
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


def _parse_meta(value: Any) -> dict[str, Any]:
    """Tolerate JSONB meta forms (parsed dict, raw JSON string, or None)."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _record_prior_hash(meta: dict[str, Any], content_hash: str | None) -> dict[str, Any]:
    """Return *meta* with *content_hash* appended to ``prior_hashes`` (idempotent).

    Merge/overwrite rewrite a memory's live ``content_hash``; the superseded
    hash is preserved here so ``_find_by_content_hash`` still matches it —
    re-ingesting the original content is gated as a duplicate instead of
    drifting the version through extract → grade → merge again.
    """
    if not content_hash:
        return meta
    prior = meta.get("prior_hashes")
    if isinstance(prior, str):
        prior = [prior]
    prior = [str(h) for h in (prior or [])]
    if content_hash not in prior:
        prior.append(content_hash)
    meta = dict(meta)
    meta["prior_hashes"] = prior
    return meta


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
            existing_meta = _parse_meta(existing.get("meta"))
            tid = current_thread_id.get("")
            if tid:
                metadata = (metadata or {}) | {"thread_id": tid}
            merged_meta = existing_meta | (metadata or {})
            # Preserve the superseded content hash so re-ingestion of the
            # original content is still gated as a duplicate (idempotency).
            merged_meta = _record_prior_hash(merged_meta, existing.get("content_hash"))

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

# Strong references to in-flight normalisation tasks.  ``asyncio.create_task``
# is not enough on its own: the event loop does not keep a strong reference,
# so a fire-and-forget task can be garbage-collected at its first await and
# the memory_entities link sync silently vanishes.  Holding the task here
# extends its lifetime until it finishes, then the ``finally`` below drops it.
_normalization_tasks: set[asyncio.Task] = set()


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
        try:
            async with _normalization_semaphore:
                await normalize_entities(memory_id, entities)
        except Exception:
            logger.exception("Entity normalisation failed for memory %s", memory_id)
        finally:
            _normalization_tasks.discard(asyncio.current_task())

    try:
        task = asyncio.create_task(_run())
    except RuntimeError:
        # No running event loop (e.g. synchronous test context) — skip.
        logger.debug("No event loop available; skipping entity normalisation for %s", memory_id)
        return
    _normalization_tasks.add(task)


async def _supplement_memory(existing, extracted, embedding, source_type, metadata, content_hash):
    """Insert new memory, linked to existing as a supplement."""
    enriched_meta = dict(metadata or {})
    # Fan-out: write_memory may list several related memories under
    # ``supplements`` (a new memory similar to multiple existing ones); the
    # closest / primary parent goes first in the list.
    related = enriched_meta.get("supplements")
    if isinstance(related, str):
        related = [related]
    related = [str(x) for x in (related or [])]
    if str(existing["id"]) not in related:
        related.insert(0, str(existing["id"]))
    enriched_meta["supplements"] = related
    enriched_meta["parent_summary"] = existing["summary"]
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
                ON CONFLICT (content_hash) WHERE deleted_at IS NULL DO NOTHING
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


async def _write_resolved_memory(
    *,
    session_factory,
    existing_id: str,
    peer_id: str | None,
    summary: str,
    entities: list[dict],
    relations: list[dict],
    embedding: list[float],
    meta: dict[str, Any],
    content_hash: str | None,
    resolution: str,
) -> dict[str, Any] | None:
    """Persist a conflict resolution into the surviving memory (A).

    Shared by the overwrite and merge branches of ``resolve_conflict`` — the
    subtle part exists once, not twice.  When *peer_id* is set (a patrol
    contradiction), B must be soft-deleted in the *same transaction* before A
    adopts B's ``content_hash``: the live-content-hash unique index is
    enforced per statement.  The write carries a ``rowcount != 1`` guard
    against A being deleted between the liveness check and the write.  If the
    new content's hash is already live on another memory (it was ingested
    separately while this conflict sat in the queue), the unique index fires
    and this returns the winner instead of 500ing.

    Returns ``None`` on success, or the winner dict when the content-hash
    race was lost.
    """
    try:
        async with session_factory() as session:
            if peer_id:
                await session.execute(
                    text(
                        "UPDATE memories SET deleted_at = NOW() "
                        "WHERE id = :id AND deleted_at IS NULL"
                    ),
                    {"id": peer_id},
                )
            where = "AND deleted_at IS NULL" if peer_id else ""
            result = await session.execute(
                text(
                    f"""\
                    UPDATE memories
                    SET summary = :summary,
                        entities = :entities,
                        relations = :relations,
                        embedding = :embedding,
                        meta = :meta,
                        content_hash = :content_hash,
                        updated_at = NOW()
                    WHERE id = :id {where}
                    """
                ),
                {
                    "id": existing_id,
                    "summary": summary,
                    "entities": json.dumps(entities, ensure_ascii=False),
                    "relations": json.dumps(relations, ensure_ascii=False),
                    "embedding": str(embedding),
                    "meta": json.dumps(meta),
                    "content_hash": content_hash,
                },
            )
            if peer_id and result.rowcount != 1:
                # A deleted between the liveness check and the write — refuse
                # rather than silently resolve against a gone row.
                raise ValueError(
                    f"Surviving memory {existing_id} no longer exists"
                )
            await session.commit()
    except IntegrityError:
        winner = await _find_by_content_hash(content_hash, session_factory)
        if winner is None:
            raise
        logger.info(
            "%s lost content-hash race for %s — content already stored as %s",
            resolution,
            content_hash[:12],
            winner["id"],
        )
        return {
            "id": str(winner["id"]),
            "action": "duplicate",
            "resolution": resolution,
            "summary": winner["summary"],
        }
    return None


async def resolve_conflict(
    resolution: str,
    existing_id: str,
    deferred_payload: dict[str, Any],
    peer_id: str | None = None,
) -> dict[str, Any]:
    """Resolve a memory conflict per the human's decision.

    Args:
        resolution: One of ``"keep_existing"``, ``"overwrite"``,
            ``"merge"``, or ``"keep_both"``.
        existing_id: The UUID of the conflicting existing memory.
        deferred_payload: The ``_deferred`` dict carried over from the
            ``write_memory`` conflict return value.
        peer_id: Set when the conflict is a patrol contradiction — two
            *already-stored* memories (A = ``existing_id`` survives, B =
            ``peer_id`` loses the arbitration).  In that shape the destructive
            options soft-delete B *in the same transaction* as the A rewrite
            (B must be gone before A adopts its ``content_hash``: the
            live-content-hash unique index is enforced per statement), and
            ``keep_both`` leaves both rows untouched — the resolved
            ``pending_conflicts`` row is the keep-both arbitration record.
            ``None`` (the default) keeps the ingestion semantics: one stored
            memory + one deferred new-side.

    Returns:
        A dict with ``action`` and ``id`` describing what happened.
    """
    extracted = deferred_payload["extracted"]
    embedding = deferred_payload["embedding"]
    source_type = deferred_payload["source_type"]
    metadata = deferred_payload["metadata"]
    content_hash = deferred_payload.get("content_hash")

    session_factory = get_session_factory()

    # Patrol shape: the surviving side (A) must still be live before any
    # resolution is applied — a resolution against a deleted row is
    # meaningless (keep_both would bless a dead pair, and the destructive
    # options would report success against a gone row).
    if peer_id:
        async with session_factory() as session:
            row = (
                await session.execute(
                    text(
                        "SELECT id FROM memories "
                        "WHERE id = :id AND deleted_at IS NULL"
                    ),
                    {"id": existing_id},
                )
            ).fetchone()
            if row is None:
                raise ValueError(
                    f"Surviving memory {existing_id} no longer exists"
                )

    if resolution == "keep_existing":
        if peer_id:
            # A survives unchanged; B is dropped.
            async with session_factory() as session:
                await session.execute(
                    text(
                        "UPDATE memories SET deleted_at = NOW() "
                        "WHERE id = :id AND deleted_at IS NULL"
                    ),
                    {"id": peer_id},
                )
                await session.commit()
        return {"id": existing_id, "action": "conflict_resolved", "resolution": "keep_existing"}

    elif resolution == "overwrite":
        # Read the existing meta + content_hash before rewriting, so the
        # resolution preserves the memory's provenance (thread_id, commit_id,
        # source, ...) instead of wiping it, and records the replaced hash in
        # prior_hashes (idempotency gate) — same semantics as _merge_memory.
        async with session_factory() as session:
            row = (
                await session.execute(
                    text("SELECT meta, content_hash FROM memories WHERE id = :id"),
                    {"id": existing_id},
                )
            ).fetchone()
            existing_meta = _parse_meta(row[0]) if row else {}
            old_content_hash = row[1] if row else None

        merged_meta = existing_meta | (metadata or {})
        merged_meta = _record_prior_hash(merged_meta, old_content_hash)

        outcome = await _write_resolved_memory(
            session_factory=session_factory,
            existing_id=existing_id,
            peer_id=peer_id,
            summary=extracted["summary"],
            entities=extracted.get("entities") or [],
            relations=extracted.get("relations") or [],
            embedding=embedding,
            meta=merged_meta,
            content_hash=content_hash,
            resolution="overwrite",
        )
        if outcome is not None:
            return outcome
        # Sync the memory_entities link table for the overwritten content's
        # entities (same fire-and-forget as _insert_memory/_merge_memory).
        _schedule_normalization(str(existing_id), extracted.get("entities") or [])
        return {"id": existing_id, "action": "conflict_resolved", "resolution": "overwrite"}

    elif resolution == "merge":
        from backend.service.llm_service import get_llm_provider

        async with session_factory() as session:
            result = await session.execute(
                text(
                    "SELECT summary, entities, relations, meta, content_hash "
                    "FROM memories WHERE id = :id"
                ),
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
                existing_meta = _parse_meta(row[3])
                old_content_hash = row[4]
            else:
                existing_summary = extracted["summary"]
                existing_entities = []
                existing_relations = []
                existing_meta = {}
                old_content_hash = None

        # Merge metadata — preserve the existing memory's provenance, add the
        # conflict tags, and keep the superseded content hash for the
        # idempotency gate (same semantics as _merge_memory).
        merged_meta = existing_meta | (metadata or {})
        merged_meta = _record_prior_hash(merged_meta, old_content_hash)

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

        outcome = await _write_resolved_memory(
            session_factory=session_factory,
            existing_id=existing_id,
            peer_id=peer_id,
            summary=merged_summary.strip(),
            entities=merged_entities,
            relations=merged_relations,
            embedding=merged_embedding,
            meta=merged_meta,
            content_hash=content_hash,
            resolution="merge",
        )
        if outcome is not None:
            return outcome
        # Sync the memory_entities link table for the merged entities (same
        # fire-and-forget as _merge_memory).
        _schedule_normalization(str(existing_id), merged_entities)
        return {"id": existing_id, "action": "conflict_resolved", "resolution": "merge"}

    elif resolution == "keep_both":
        if peer_id:
            # Both memories stay — the resolved pending_conflicts row written
            # by resolve_pending_conflict is the keep-both arbitration record.
            return {
                "id": existing_id,
                "action": "conflict_resolved",
                "resolution": "keep_both",
            }
        return await _insert_memory(extracted, embedding, source_type, metadata, content_hash)

    else:
        logger.warning("Unknown conflict resolution '%s', defaulting to keep_existing", resolution)
        if peer_id:
            # Same semantics as the explicit keep_existing branch above — an
            # unknown resolution degrades to keep_existing, which in the
            # patrol shape soft-deletes the losing side B.  Without this the
            # conflict would be marked resolved while B stays live, so the
            # pair would re-surface on the next scan.
            async with session_factory() as session:
                await session.execute(
                    text(
                        "UPDATE memories SET deleted_at = NOW() "
                        "WHERE id = :id AND deleted_at IS NULL"
                    ),
                    {"id": peer_id},
                )
                await session.commit()
        return {"id": existing_id, "action": "conflict_resolved", "resolution": "keep_existing"}
