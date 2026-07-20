"""Retrieval pipeline — independent functions, not a black-box chain.

Write path:  embed_query() → write_chunks()
Read path:   embed_query() → vector_search() → rerank() → assemble()
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from backend.db import get_session_factory
from backend.service.decay import search_memories
from backend.service.embedding_service import get_embedding_provider

logger = logging.getLogger(__name__)

_ALLOWED_FILTER_COLS = {"document_id", "chunk_index"}


@dataclass
class RetrievalResult:
    content: str
    score: float
    metadata: dict[str, Any]


# ── Write path ─────────────────────────────────────────────────


async def write_chunks(
    document_id: str,
    chunks: list[str],
    meta: dict[str, Any] | None = None,
) -> int:
    """Embed *chunks* and insert into the chunks table. Returns count written."""
    if not chunks:
        return 0

    provider = get_embedding_provider()
    vectors = await provider.embed(chunks)
    meta_json = json.dumps(meta or {})

    session_factory = get_session_factory()
    async with session_factory() as session:
        # Batch INSERT — single round-trip for all chunks
        values_clauses: list[str] = []
        params: dict[str, Any] = {}
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            key_doc = f"doc_id_{i}"
            key_content = f"content_{i}"
            key_vec = f"vec_{i}"
            key_meta = f"meta_{i}"
            key_idx = f"idx_{i}"
            values_clauses.append(
                f"(:{key_doc}, :{key_content}, :{key_vec} ::vector, :{key_meta} ::jsonb, :{key_idx})"
            )
            params[key_doc] = document_id
            params[key_content] = chunk
            params[key_vec] = str(vec)
            params[key_meta] = meta_json
            params[key_idx] = i

        sql = (
            "INSERT INTO chunks (document_id, content, embedding, metadata, chunk_index) VALUES "
            + ", ".join(values_clauses)
        )
        await session.execute(text(sql), params)
        await session.commit()

    logger.info("Wrote %d chunks for document %s", len(chunks), document_id)
    return len(chunks)


# ── Read path ──────────────────────────────────────────────────


async def vector_search(
    query_vector: list[float],
    top_k: int = 20,
    threshold: float = 0.0,
    *,
    filters: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Cosine-similarity vector recall against the chunks table.

    Optional *filters* (e.g. ``{"document_id": "repo.py"}``) are added as
    ``AND col = :val`` clauses to narrow the search.  Only columns in
    *ALLOWED_FILTER_COLS* are accepted; unknown keys are silently ignored.
    """
    filter_clauses: list[str] = []
    params: dict[str, Any] = {"vec": str(query_vector), "threshold": threshold, "limit": top_k}
    if filters:
        for col, val in filters.items():
            if col not in _ALLOWED_FILTER_COLS:
                logger.warning("Ignoring unknown filter column: %s", col)
                continue
            key = f"f_{col}"
            filter_clauses.append(f"AND {col} = :{key}")
            params[key] = val

    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                f"""\
                SELECT id, content, metadata, chunk_index,
                       1 - (embedding <=> :vec ::vector) AS similarity
                FROM chunks
                WHERE 1 - (embedding <=> :vec ::vector) > :threshold
                {' ' + ' '.join(filter_clauses) if filter_clauses else ''}
                ORDER BY embedding <=> :vec ::vector
                LIMIT :limit
                """
            ),
            params,
        )
        return [dict(r._mapping) for r in result]


async def embed_query(query: str) -> list[float]:
    """Embed a single query string, returning a flat vector."""
    provider = get_embedding_provider()
    vectors = await provider.embed([query])
    return vectors[0]


def assemble(results: list[RetrievalResult], max_items: int = 5) -> str:
    """Join top-N retrieval results into an LLM-ready context string."""
    if not results:
        return ""

    lines: list[str] = []
    for r in results[:max_items]:
        lines.append(f"--- [relevance: {r.score:.2f}] ---\n{r.content}")

    return "\n\n".join(lines)


async def retrieve(
    query: str,
    top_k: int = 5,
    *,
    use_llm_rerank: bool = False,
) -> list[RetrievalResult]:
    """Full read pipeline: embed → vector search → rerank.

    Args:
        query: User query string.
        top_k: Final number of results to return after reranking.
        use_llm_rerank: If True, use LLM-based reranking instead of the
            default cross-encoder.  Slower and costs API tokens, but can
            produce more nuanced relevance judgments.
    """
    from backend.service.rerank import rerank_cross_encoder, rerank_llm

    query_vec = await embed_query(query)
    candidates = await vector_search(query_vec, top_k=max(top_k * 4, 20))

    if not candidates:
        return []

    reranker = rerank_llm if use_llm_rerank else rerank_cross_encoder
    ranked = await reranker(
        query, [c["content"] for c in candidates], top_k=top_k
    )

    return [
        RetrievalResult(
            content=candidates[idx]["content"],
            score=score,
            metadata=candidates[idx].get("metadata") or {},
        )
        for idx, score in ranked
    ]


async def query_memories(
    query: str,
    top_k: int = 5,
    *,
    threshold: float = 0.0,
    use_llm_rerank: bool = False,
) -> list[dict]:
    """Search memories with decay-weighted ranking.

    Full pipeline: embed → decay-weighted vector search → rerank →
    update_decay → return as-ranked list of memory dicts.
    """
    from backend.service.rerank import rerank_cross_encoder, rerank_llm

    provider = get_embedding_provider()
    vectors = await provider.embed([query])
    query_vec = vectors[0]

    candidates = await search_memories(query_vec, top_k=max(top_k * 4, 20), threshold=threshold)

    if not candidates:
        return []

    reranker = rerank_llm if use_llm_rerank else rerank_cross_encoder
    ranked = await reranker(
        query, [c["summary"] for c in candidates], top_k=top_k
    )

    # Re-attach full memory rows in ranked order, and update decay
    from backend.service.decay import update_decay

    result: list[dict] = []
    for idx, score in ranked:
        memory_id = str(candidates[idx]["id"])
        new_decay = await update_decay(memory_id)
        entry = {**candidates[idx], "rerank_score": score, "decay_factor": new_decay}
        result.append(entry)
    return result
