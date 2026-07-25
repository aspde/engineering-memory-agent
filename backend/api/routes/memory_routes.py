"""Memory & retrieval API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from backend.db import get_session_factory
from backend.service.chunk import chunk_text
from backend.service.memory import write_memory
from backend.service.retrieval import query_memories, retrieve, write_chunks

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
    id: str
    action: str  # "inserted" | "merged" | "conflict"
    summary: str


class MemorySearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=50)
    use_llm_rerank: bool = False


class MemorySearchResponse(BaseModel):
    results: list[dict[str, Any]]


class MemoryStatsResponse(BaseModel):
    total_memories: int
    total_chunks: int
    total_conversations: int
    by_source_type: list[dict[str, Any]]
    avg_decay_factor: float
    avg_entities_per_memory: float
    avg_relations_per_memory: float
    recent_count_7d: int
    top_entities: list[dict[str, Any]]


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
    """Semantic search over ingested documents.

    Pipeline: embed → vector search → rerank.
    """
    try:
        results = await retrieve(req.query, top_k=req.top_k, use_llm_rerank=req.use_llm_rerank)
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
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return MemoryWriteResponse(
        id=result["id"],
        action=result["action"],
        summary=result["summary"],
    )


@router.post("/memories/search", response_model=MemorySearchResponse)
async def memory_search(req: MemorySearchRequest) -> MemorySearchResponse:
    """Search structured memories with decay-weighted ranking.

    Decay factors are updated on recall — frequently retrieved memories
    are boosted over time.
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


@router.get("/stats", response_model=MemoryStatsResponse)
async def memory_stats() -> MemoryStatsResponse:
    """Return aggregate statistics about the memory store.

    Used by the frontend dashboard to show total counts, source
    distribution, decay health, and frequently mentioned entities.
    """
    session_factory = get_session_factory()
    async with session_factory() as session:
        # ── Core counts ──
        r = await session.execute(text("SELECT COUNT(*) FROM memories"))
        total_memories = r.fetchone()[0]

        r = await session.execute(text("SELECT COUNT(*) FROM chunks"))
        total_chunks = r.fetchone()[0]

        r = await session.execute(text("SELECT COUNT(*) FROM conversations"))
        total_conversations = r.fetchone()[0]

        # ── Source type distribution ──
        r = await session.execute(text(
            "SELECT source_type, COUNT(*) AS cnt FROM memories "
            "GROUP BY source_type ORDER BY cnt DESC"
        ))
        by_source_type = [{"source_type": row[0], "count": row[1]} for row in r.fetchall()]

        # ── Decay & entity/relation averages
        #     jsonb_array_length returns NULL, not 0, for non-array / NULL columns.
        r = await session.execute(text(
            "SELECT COALESCE(AVG(decay_factor), 0) FROM memories"
        ))
        avg_decay_factor = round(float(r.fetchone()[0]), 4)

        r = await session.execute(text(
            "SELECT COALESCE(AVG("
            "  CASE WHEN jsonb_typeof(entities) = 'array' "
            "       THEN jsonb_array_length(entities) ELSE 0 END"
            "), 0) FROM memories"
        ))
        avg_entities_per_memory = round(float(r.fetchone()[0]), 2)

        r = await session.execute(text(
            "SELECT COALESCE(AVG("
            "  CASE WHEN jsonb_typeof(relations) = 'array' "
            "       THEN jsonb_array_length(relations) ELSE 0 END"
            "), 0) FROM memories"
        ))
        avg_relations_per_memory = round(float(r.fetchone()[0]), 2)

        # ── Recent 7-day count ──
        r = await session.execute(text(
            "SELECT COUNT(*) FROM memories "
            "WHERE created_at >= NOW() - INTERVAL '7 days'"
        ))
        recent_count_7d = r.fetchone()[0]

        # ── Top 10 entities (skip null / non-array rows defensively) ──
        r = await session.execute(text(
            "SELECT e->>'name' AS name, COUNT(*) AS cnt "
            "FROM memories, jsonb_array_elements(entities) AS e "
            "WHERE entities IS NOT NULL "
            "  AND jsonb_typeof(entities) = 'array' "
            "  AND e->>'name' IS NOT NULL "
            "GROUP BY e->>'name' ORDER BY cnt DESC LIMIT 10"
        ))
        top_entities = [{"name": row[0], "count": row[1]} for row in r.fetchall()]

    return MemoryStatsResponse(
        total_memories=total_memories,
        total_chunks=total_chunks,
        total_conversations=total_conversations,
        by_source_type=by_source_type,
        avg_decay_factor=avg_decay_factor,
        avg_entities_per_memory=avg_entities_per_memory,
        avg_relations_per_memory=avg_relations_per_memory,
        recent_count_7d=recent_count_7d,
        top_entities=top_entities,
    )
