"""Memory & retrieval API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from backend.db import get_session_factory
from backend.service.chunk import chunk_text
from backend.service.conflicts import persist_pending_conflict
from backend.service.memory import write_memory
from backend.service.retrieval import query_memories, retrieve_hybrid, write_chunks

router = APIRouter(prefix="/memory", tags=["memory"])


# ── Request / Response models ──────────────────────────────────────


class IngestRequest(BaseModel):
    document_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    document_id: str
    chunks_written: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    use_llm_rerank: bool = False


class SearchResult(BaseModel):
    content: str
    score: float
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    results: list[SearchResult]


class MemoryWriteRequest(BaseModel):
    content: str
    source_type: str = "api"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryWriteResponse(BaseModel):
    id: str | None = None
    action: str  # "inserted" | "merged" | "conflict"
    summary: str
    entity_ids: list[str] = Field(default_factory=list)
    conflict_id: str | None = None
    existing_id: str | None = None


class MemorySearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    use_llm_rerank: bool = False


class MemorySearchResponse(BaseModel):
    results: list[dict[str, Any]]


class MemoryGetResponse(BaseModel):
    id: str
    source_type: str
    summary: str
    entities: list[dict[str, Any]] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    recall_count: int
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class EntityGraphStats(BaseModel):
    coverage_ratio: float
    growth_rate_7d: float
    density: float
    total_entities: int


class MemoryStatsResponse(BaseModel):
    total_memories: int
    total_chunks: int
    total_conversations: int
    by_source_type: list[dict[str, Any]]
    avg_recall_count: float
    avg_entities_per_memory: float
    avg_relations_per_memory: float
    recent_count_7d: int
    top_entities: list[dict[str, Any]]
    entity_graph: EntityGraphStats | None = None


class MemoryDeleteResponse(BaseModel):
    id: str
    deleted: bool


# ── Routes ─────────────────────────────────────────────────────────


@router.post("/ingest", response_model=IngestResponse)
async def ingest(req: IngestRequest) -> IngestResponse:
    """Chunk + embed + store a document.

    Splits *content* into chunks, computes embeddings, and writes them
    to the ``chunks`` table.
    """
    try:
        chunks = chunk_text(req.content)
        count = await write_chunks(req.document_id, chunks, meta=req.metadata)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestResponse(document_id=req.document_id, chunks_written=count)


@router.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    """Hybrid search over ingested documents.

    Pipeline: dense vector + sparse BM25 union, ranked by reciprocal-rank
    fusion (RRF) of the two lists.  Cross-encoder rerank is skipped by default
    (eval shows it costs ~90x latency without recall gain on the current
    corpus); pass ``use_llm_rerank=True`` for the LLM pointwise variant.

    ``SearchResult.score`` is the RRF fusion normalised to a 0-1 scale
    (1.0 = ranked #1 by both retrievers) — use it for relative ordering,
    not as an absolute similarity threshold.
    """
    try:
        results = await retrieve_hybrid(req.query, top_k=req.top_k, use_llm_rerank=req.use_llm_rerank)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SearchResponse(
        results=[
            SearchResult(content=r.content, score=round(r.score, 4), metadata=r.metadata)
            for r in results
        ]
    )


@router.post("/memories/write", response_model=MemoryWriteResponse)
async def memory_write(req: MemoryWriteRequest) -> MemoryWriteResponse:
    """Extract structured memory from content and persist.

    Performs three-stage extraction (summary → entities → relations),
    then similarity check, conflict detection, and merge-or-insert.
    """
    try:
        result = await write_memory(req.content, source_type=req.source_type, metadata=req.metadata)
        if result.get("action") == "conflict":
            # A plain REST write has no interactive session to pause for HITL —
            # persist the conflict to the pending_conflicts queue so a human can
            # resolve it later (same non-interactive handling as the webhook
            # path).  The conflict result carries no ``id`` (nothing was
            # written), so the response exposes the queue row's ``conflict_id``
            # instead of crashing on a missing ``id`` key.
            pending = await persist_pending_conflict(req.source_type, result)
            return MemoryWriteResponse(
                action="conflict",
                summary=result["summary"],
                conflict_id=pending["id"],
                existing_id=result.get("existing_id"),
            )
        return MemoryWriteResponse(
            id=result["id"],
            action=result["action"],
            summary=result["summary"],
            entity_ids=result.get("entity_ids", []),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/memories/search", response_model=MemorySearchResponse)
async def memory_search(req: MemorySearchRequest) -> MemorySearchResponse:
    """Search structured memories ranked by semantic similarity.

    Recalls are recorded for every surfaced memory (recall_count /
    recalled_at) as metadata for the UI and the patrol archival scan.
    """
    try:
        results = await query_memories(
            req.query, top_k=req.top_k, use_llm_rerank=req.use_llm_rerank
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Convert UUIDs and timestamps to strings for JSON serialization
    clean: list[dict[str, Any]] = []
    for r in results:
        entry: dict[str, Any] = {}
        for k, v in r.items():
            if hasattr(v, "isoformat"):
                entry[k] = v.isoformat()
            elif isinstance(v, dict):
                entry[k] = v  # JSONB fields already parsed
            else:
                entry[k] = str(v) if not isinstance(v, (int, float, str, list, type(None))) else v
        clean.append(entry)

    return MemorySearchResponse(results=clean)


@router.get("/memories/{memory_id}", response_model=MemoryGetResponse)
async def get_memory_by_id(memory_id: str) -> MemoryGetResponse:
    """Fetch a single memory by its primary-key UUID.

    Used by the frontend after a memory source is clicked in the chat
    page — bypasses the vector-search pipeline entirely.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        r = await session.execute(
            text(
                "SELECT id, source_type, summary, entities, relations, "
                "       recall_count, meta, created_at "
                "FROM memories WHERE id = :id AND deleted_at IS NULL"
            ),
            {"id": memory_id},
        )
        row = r.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Memory not found or has been deleted")

        return MemoryGetResponse(
            id=str(row.id),
            source_type=str(row.source_type),
            summary=str(row.summary),
            entities=row.entities or [],
            relations=row.relations or [],
            recall_count=int(row.recall_count),
            meta=row.meta or {},
            created_at=row.created_at.isoformat() if row.created_at else "",
        )


@router.delete("/memories/{memory_id}", response_model=MemoryDeleteResponse)
async def delete_memory(memory_id: str) -> MemoryDeleteResponse:
    """Soft-delete a memory by setting deleted_at = NOW().

    Returns 404 if the memory does not exist or is already deleted.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        r = await session.execute(
            text("SELECT deleted_at FROM memories WHERE id = :id"),
            {"id": memory_id},
        )
        row = r.fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Memory not found")
        if row.deleted_at is not None:
            raise HTTPException(status_code=404, detail="Memory has already been deleted")

        await session.execute(
            text("UPDATE memories SET deleted_at = NOW() WHERE id = :id"),
            {"id": memory_id},
        )
        await session.commit()

    return MemoryDeleteResponse(id=memory_id, deleted=True)


@router.get("/stats", response_model=MemoryStatsResponse)
async def memory_stats() -> MemoryStatsResponse:
    """Return aggregate statistics about the memory store.

    Used by the frontend dashboard to show total counts, source
    distribution, recall stats, and frequently mentioned entities.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        # ── Core counts ──
        r = await session.execute(text("SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL"))
        total_memories = r.fetchone()[0]

        r = await session.execute(text("SELECT COUNT(*) FROM chunks"))
        total_chunks = r.fetchone()[0]

        r = await session.execute(text("SELECT COUNT(*) FROM conversations"))
        total_conversations = r.fetchone()[0]

        # ── Source type distribution ──
        r = await session.execute(text(
            "SELECT source_type, COUNT(*) AS cnt FROM memories "
            "WHERE deleted_at IS NULL "
            "GROUP BY source_type ORDER BY cnt DESC"
        ))
        by_source_type = [{"source_type": row[0], "count": row[1]} for row in r.fetchall()]

        # ── Recall & entity/relation averages
        #     jsonb_array_length returns NULL, not 0, for non-array / NULL columns.
        r = await session.execute(text(
            "SELECT COALESCE(AVG(recall_count), 0) FROM memories WHERE deleted_at IS NULL"
        ))
        avg_recall_count = round(float(r.fetchone()[0]), 4)

        r = await session.execute(text(
            "SELECT COALESCE(AVG("
            "  CASE WHEN jsonb_typeof(entities) = 'array' "
            "       THEN jsonb_array_length(entities) ELSE 0 END"
            "), 0) FROM memories WHERE deleted_at IS NULL"
        ))
        avg_entities_per_memory = round(float(r.fetchone()[0]), 2)

        r = await session.execute(text(
            "SELECT COALESCE(AVG("
            "  CASE WHEN jsonb_typeof(relations) = 'array' "
            "       THEN jsonb_array_length(relations) ELSE 0 END"
            "), 0) FROM memories WHERE deleted_at IS NULL"
        ))
        avg_relations_per_memory = round(float(r.fetchone()[0]), 2)

        # ── Recent 7-day count ──
        r = await session.execute(text(
            "SELECT COUNT(*) FROM memories "
            "WHERE deleted_at IS NULL AND created_at >= NOW() - INTERVAL '7 days'"
        ))
        recent_count_7d = r.fetchone()[0]

        # ── Top 10 entities (skip null / non-array rows defensively) ──
        r = await session.execute(text(
            "SELECT e->>'name' AS name, COUNT(*) AS cnt "
            "FROM memories, jsonb_array_elements(entities) AS e "
            "WHERE deleted_at IS NULL "
            "  AND entities IS NOT NULL "
            "  AND jsonb_typeof(entities) = 'array' "
            "  AND e->>'name' IS NOT NULL "
            "GROUP BY e->>'name' ORDER BY cnt DESC LIMIT 10"
        ))
        top_entities = [{"name": row[0], "count": row[1]} for row in r.fetchall()]

        # ── Entity graph metrics (Phase 1) ──
        entity_graph = None
        try:
            # coverage_ratio: memories linked to entities / total memories
            r = await session.execute(text(
                "SELECT "
                "  COALESCE(COUNT(DISTINCT me.memory_id)::float "
                "    / NULLIF(COUNT(DISTINCT m.id), 0), 0) AS coverage "
                "FROM memories m "
                "LEFT JOIN memory_entities me ON me.memory_id = m.id "
                "WHERE m.deleted_at IS NULL"
            ))
            coverage_ratio = round(float(r.fetchone()[0]), 4)

            # total_entities
            r = await session.execute(text("SELECT COUNT(*) FROM entities"))
            total_entities = r.fetchone()[0]

            # growth_rate_7d: new entities in past 7 days / total entities
            r = await session.execute(text(
                "SELECT COUNT(*) FROM entities "
                "WHERE first_seen_at >= NOW() - INTERVAL '7 days'"
            ))
            new_7d = r.fetchone()[0]
            growth_rate_7d = round(
                new_7d / total_entities if total_entities > 0 else 0.0, 4
            )

            # density: avg entities per memory from the junction table
            r = await session.execute(text(
                "SELECT "
                "  COALESCE("
                "    COUNT(me.entity_id)::float "
                "    / NULLIF(COUNT(DISTINCT me.memory_id), 0), 0) "
                "FROM memory_entities me"
            ))
            density = round(float(r.fetchone()[0]), 2)

            entity_graph = EntityGraphStats(
                coverage_ratio=coverage_ratio,
                growth_rate_7d=growth_rate_7d,
                density=density,
                total_entities=total_entities,
            )
        except Exception:
            pass  # entities table may not exist yet (pre-migration)

    return MemoryStatsResponse(
        total_memories=total_memories,
        total_chunks=total_chunks,
        total_conversations=total_conversations,
        by_source_type=by_source_type,
        avg_recall_count=avg_recall_count,
        avg_entities_per_memory=avg_entities_per_memory,
        avg_relations_per_memory=avg_relations_per_memory,
        recent_count_7d=recent_count_7d,
        top_entities=top_entities,
        entity_graph=entity_graph,
    )
