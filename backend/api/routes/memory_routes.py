"""Memory & retrieval API routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

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
