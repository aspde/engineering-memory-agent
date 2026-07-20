"""Rerank functions — plain functions, no wrapper classes.

- rerank_cross_encoder(): local BGE cross-encoder, zero API cost
- rerank_llm():         LLM-based pointwise scoring via existing LLMProvider
"""

from __future__ import annotations

import asyncio
import logging

from backend.shared.config import config

logger = logging.getLogger(__name__)

# Lazy-loaded cross-encoder model — loaded once on first call
_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        # Use the HF_ENDPOINT from embedding config for consistency
        import os

        os.environ.setdefault("HF_ENDPOINT", config.embedding.hf_endpoint)

        model_name = "BAAI/bge-reranker-v2-m3"
        logger.info("Loading reranker model: %s", model_name)
        _cross_encoder = CrossEncoder(model_name)
    return _cross_encoder


async def rerank_cross_encoder(
    query: str, candidates: list[str], top_k: int = 5
) -> list[tuple[int, float]]:
    """Rerank candidates with a local cross-encoder model.

    Args:
        query: The user query.
        candidates: Candidate text chunks from vector recall.
        top_k: How many to return after reranking.

    Returns:
        List of (original_index, score), sorted by score descending.
    """
    if not candidates:
        return []

    model = _get_cross_encoder()
    pairs = [[query, c] for c in candidates]
    scores = await asyncio.to_thread(model.predict, pairs)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


async def rerank_llm(
    query: str, candidates: list[str], top_k: int = 5
) -> list[tuple[int, float]]:
    """Rerank candidates by asking the LLM to score each one (pointwise).

    Args:
        query: The user query.
        candidates: Candidate text chunks from vector recall.
        top_k: How many to return after reranking.

    Returns:
        List of (original_index, score), sorted by score descending.
    """
    if not candidates:
        return []

    from backend.service.llm_service import get_llm_provider

    llm = get_llm_provider()

    async def _score_one(idx: int, text: str) -> tuple[int, float]:
        prompt = _RERANK_PROMPT.format(query=query, text=text)
        try:
            response = await llm.chat([{"role": "user", "content": prompt}])
            score = float(response.strip())
            return idx, max(0.0, min(1.0, score))
        except (ValueError, TypeError):
            return idx, 0.0

    tasks = [_score_one(i, c) for i, c in enumerate(candidates)]
    results = await asyncio.gather(*tasks)

    ranked = sorted(results, key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


_RERANK_PROMPT = """\
Score how relevant the following text is to the query on a scale from 0 (completely irrelevant) to 1 (perfect match).

Query: {query}

Text: {text}

Reply with ONLY the score as a decimal number (e.g. 0.85). Do not include any other text."""
