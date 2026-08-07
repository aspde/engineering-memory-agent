"""Retrieval pipeline — independent functions, not a black-box chain.

Write path:  embed_query() → write_chunks()
Read path:   embed_query() → vector_search() → rerank() → assemble()
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import jieba
from sqlalchemy import text

from backend.db import get_session_factory
from backend.service.decay import search_memories
from backend.service.embedding_service import get_embedding_provider

logger = logging.getLogger(__name__)

_ALLOWED_FILTER_COLS = {"document_id", "chunk_index"}

# Rerank floor: scores below this are treated as irrelevant and dropped.
# Shared by every read path so the threshold stays tunable in one place.
_RERANK_FLOOR = 0.15

# CJK stopword set, in two categories:
#   (a) grammatical function words (的/了/是/…), which never discriminate;
#   (b) a few high-frequency tokens whose content word always co-occurs with a
#       more specific term — 支持 (support), 陷入 (fall into), 出过/分了 (aspect
#       fragments) — so dropping them from a query shrinks the Jaccard
#       denominator without losing signal.  Selected empirically on the eval
#       corpus; kept intentionally small — aggressive stopword lists hurt
#       recall on short technical queries.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都",
        "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
        "会", "着", "没有", "看", "好", "自己", "这", "那", "怎么",
        "什么", "哪些", "哪个", "之前", "出过", "几个", "不会", "陷入",
        "支持", "分了",
    }
)


def _tokenize(text: str) -> set[str]:
    """jieba tokenize, drop stopwords and single-char CJK tokens.

    ASCII single-char tokens (e.g. ``c`` in variable names) are kept —
    they carry signal in code queries.
    """
    tokens = jieba.lcut(text)
    return {
        t.strip().lower()
        for t in tokens
        if t.strip()
        and t.strip() not in _STOPWORDS
        and (len(t.strip()) >= 2 or t.strip().isascii())
    }


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
            key_tokens = f"tokens_{i}"
            values_clauses.append(
                f"(:{key_doc}, :{key_content}, :{key_vec} ::vector, "
                f":{key_meta} ::jsonb, :{key_idx}, :{key_tokens} ::text[])"
            )
            params[key_doc] = document_id
            params[key_content] = chunk
            params[key_vec] = str(vec)
            params[key_meta] = meta_json
            params[key_idx] = i
            params[key_tokens] = list(_tokenize(chunk))

        sql = (
            "INSERT INTO chunks (document_id, content, embedding, meta, "
            "chunk_index, tokens) VALUES "
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
                SELECT id, content, meta, chunk_index,
                       1 - (embedding <=> :vec ::vector) AS similarity
                FROM chunks
                WHERE 1 - (embedding <=> :vec ::vector) > :threshold
                {' ' + ' '.join(filter_clauses) if filter_clauses else ''}
                ORDER BY embedding <=> :vec ::vector
                LIMIT :limit"""
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


async def _rerank_and_filter(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int,
    use_llm_rerank: bool,
) -> list[RetrievalResult]:
    """Rerank chunk candidates, drop below-floor results.

    Shared by ``retrieve`` / ``retrieve_hybrid`` / ``retrieve_multi_query``
    — the rerank + floor + assemble tail was copy-pasted in all three.
    """
    from backend.service.rerank import rerank_cross_encoder, rerank_llm

    reranker = rerank_llm if use_llm_rerank else rerank_cross_encoder
    ranked = await reranker(
        query, [c["content"] for c in candidates], top_k=top_k
    )
    return [
        RetrievalResult(
            content=candidates[idx]["content"],
            score=score,
            metadata=candidates[idx].get("meta") or {},
        )
        for idx, score in ranked
        if score >= _RERANK_FLOOR
    ]


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
    query_vec = await embed_query(query)
    candidates = await vector_search(query_vec, top_k=max(top_k * 4, 20))

    if not candidates:
        return []

    return await _rerank_and_filter(query, candidates, top_k, use_llm_rerank)


async def sparse_search(query: str, top_k: int = 20) -> list[dict[str, Any]]:
    """BM25-style keyword search via jieba tokenization + Jaccard similarity.

    Postgres ``simple`` tokenizer can't segment Chinese (treats a whole
    CJK run as one token), so we tokenize the query with jieba and use a
    GIN-indexed ``tokens`` column on ``chunks`` for O(log N) filtering:
    only rows with at least one token overlap are fetched, then Jaccard
    is computed from the DB-side intersection cardinality.

    Requires ``tokens`` to be populated (done at write time by
    ``write_chunks`` and backfilled for pre-existing rows).
    """
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []

    # GIN-indexed overlap filter — the main O(N) → O(log N) win.  Two stages
    # so we never pull full ``content`` for chunks that don't survive scoring:
    #   1. fetch only id + tokens for the overlap set, score Jaccard Python-side
    #      (PG's ``&`` array-intersection operator is int[]-only, so inter
    #      cardinality can't be computed DB-side for text[]);
    #   2. fetch full rows for the surviving top-k by primary key.
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                """
                SELECT id, tokens
                FROM chunks
                WHERE tokens && :q_tokens ::text[]
                """
            ),
            {"q_tokens": list(q_tokens)},
        )
        scored: list[tuple[float, Any]] = []  # (jaccard, chunk_id)
        for row in result:
            cid = row._mapping["id"]
            ct = set(row._mapping.get("tokens") or [])
            if not ct:
                continue
            inter = len(q_tokens & ct)
            if inter == 0:
                continue
            union = len(q_tokens | ct)
            scored.append((inter / union, cid))

        if not scored:
            return []

        scored.sort(key=lambda x: -x[0])
        ids = [cid for _, cid in scored[:top_k]]

        result = await session.execute(
            text(
                """
                SELECT id, content, meta, chunk_index
                FROM chunks
                WHERE id = ANY(:ids)
                """
            ),
            {"ids": ids},
        )
        by_id = {r._mapping["id"]: dict(r._mapping) for r in result}

    return [
        {**by_id[cid], "rank": sim}
        for sim, cid in scored[:top_k]
        if cid in by_id
    ]


async def retrieve_hybrid(
    query: str,
    top_k: int = 5,
    *,
    use_llm_rerank: bool = False,
    skip_rerank: bool = False,
) -> list[RetrievalResult]:
    """Hybrid retrieval: dense vector + sparse BM25 union → rerank.

    Combines semantic recall (``vector_search``) with keyword recall
    (``sparse_search``) to rescue conceptual queries where dense embedding
    misses due to low surface-form overlap.  Both candidate sets are
    unioned (dedup by chunk id), then reranked.

    When ``skip_rerank`` is True, candidates are ranked by the max of
    dense similarity / sparse jaccard — used for A/B comparison to measure
    rerank's contribution.
    """
    query_vec = await embed_query(query)
    dense = await vector_search(query_vec, top_k=max(top_k * 4, 20))
    sparse = await sparse_search(query, top_k=max(top_k * 4, 20))

    # Union by chunk id, keeping BOTH scores so the skip_rerank fallback can
    # take max(dense, sparse) even for chunks found by both retrievers.
    merged: dict[str, dict[str, Any]] = {}
    for r in dense + sparse:
        cid = str(r["id"])
        if cid in merged:
            merged[cid]["rank"] = max(
                float(merged[cid].get("rank") or 0.0),
                float(r.get("rank") or 0.0),
            )
            merged[cid]["similarity"] = max(
                float(merged[cid].get("similarity") or 0.0),
                float(r.get("similarity") or 0.0),
            )
        else:
            merged[cid] = r

    if not merged:
        return []

    candidates = list(merged.values())

    if skip_rerank:
        # Fallback ranking: max(dense similarity, sparse jaccard).
        def _fallback_score(c: dict[str, Any]) -> float:
            return max(
                float(c.get("similarity") or 0.0),
                float(c.get("rank") or 0.0),
            )

        candidates.sort(key=_fallback_score, reverse=True)
        return [
            RetrievalResult(
                content=c["content"],
                score=_fallback_score(c),
                metadata=c.get("meta") or {},
            )
            for c in candidates[:top_k]
        ]

    return await _rerank_and_filter(query, candidates, top_k, use_llm_rerank)


async def retrieve_multi_query(
    query: str,
    top_k: int = 5,
    *,
    use_llm_rerank: bool = False,
) -> list[RetrievalResult]:
    """Multi-query retrieval: LLM rewrite → union → dedup → rerank.

    1. ``rewrite_query`` expands the query into N concrete-term variations.
    2. Variations are embedded in one batched call, then each is searched
       via ``vector_search``.
    3. Results are unioned and deduplicated by chunk id.
    4. The union is reranked (cross-encoder by default) and trimmed.

    Use this for conceptual queries (e.g. "之前出过什么问题") where the
    user's wording won't surface-match stored memories.  Costs one extra
    LLM call (~500ms) for the rewrite; fails safe to single-query if the
    rewrite errors.
    """
    from backend.service.query_rewrite import rewrite_query

    queries = await rewrite_query(query)

    # Batch-embed every variation in one provider call (1 latency hit instead
    # of N sequential embeds), then search each.
    provider = get_embedding_provider()
    vectors = await provider.embed(queries)

    # Multi-query union, dedup by chunk id.
    seen: dict[str, dict[str, Any]] = {}
    for q, vec in zip(queries, vectors):
        results = await vector_search(vec, top_k=max(top_k * 2, 10))
        for r in results:
            cid = str(r["id"])
            if cid not in seen:
                seen[cid] = r

    if not seen:
        return []

    candidates = list(seen.values())[: max(top_k * 4, 20)]
    return await _rerank_and_filter(query, candidates, top_k, use_llm_rerank)


async def query_memories(
    query: str,
    top_k: int = 5,
    *,
    threshold: float = 0.3,
    use_llm_rerank: bool = False,
) -> list[dict]:
    """Search memories with decay-weighted ranking.

    Full pipeline: embed → decay-weighted vector search → rerank →
    update_decay → return as-ranked list of memory dicts.
    """
    from backend.service.rerank import rerank_cross_encoder, rerank_llm

    t0 = time.perf_counter()

    provider = get_embedding_provider()
    vectors = await provider.embed([query])
    query_vec = vectors[0]
    t_embed = time.perf_counter()

    candidates = await search_memories(query_vec, top_k=max(top_k * 4, 20), threshold=threshold)
    t_search = time.perf_counter()

    if not candidates:
        logger.info(
            "query_memories latency: total=%.0fms embed=%.0fms search=%.0fms "
            "rerank=0ms results=0 (no candidates) query=%r",
            (t_search - t0) * 1000, (t_embed - t0) * 1000,
            (t_search - t_embed) * 1000, query[:60],
        )
        return []

    reranker = rerank_llm if use_llm_rerank else rerank_cross_encoder
    ranked = await reranker(
        query, [c["summary"] for c in candidates], top_k=top_k
    )
    t_rerank = time.perf_counter()

    # Re-attach full memory rows in ranked order, and update decay
    # Drop results where the reranker score is below the minimum threshold —
    # this prevents irrelevant results from appearing when no real match exists.
    from backend.service.decay import update_decay

    result: list[dict] = []
    for idx, score in ranked:
        if score < _RERANK_FLOOR:
            continue
        memory_id = str(candidates[idx]["id"])
        new_decay = await update_decay(memory_id)
        entry = {**candidates[idx], "rerank_score": score, "decay_factor": new_decay}
        result.append(entry)

    t_end = time.perf_counter()
    logger.info(
        "query_memories latency: total=%.0fms embed=%.0fms search=%.0fms "
        "rerank=%.0fms decay=%.0fms top_k=%d candidates=%d results=%d "
        "llm_rerank=%s query=%r",
        (t_end - t0) * 1000, (t_embed - t0) * 1000,
        (t_search - t_embed) * 1000, (t_rerank - t_search) * 1000,
        (t_end - t_rerank) * 1000, top_k, len(candidates), len(result),
        use_llm_rerank, query[:60],
    )
    return result
