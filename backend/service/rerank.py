"""Rerank functions — plain functions, no wrapper classes.

- rerank_cross_encoder(): local BGE cross-encoder, zero API cost
- rerank_llm():         LLM-based pointwise scoring via existing LLMProvider
"""

from __future__ import annotations

import asyncio
import logging
import re

from backend.shared.config import config

logger = logging.getLogger(__name__)

# A bare relevance score is the expected reply, but models sometimes wrap it
# in prose ("Relevance: 0.85") or emit extra tokens.  Extract the first
# numeric token defensively instead of failing the whole candidate to 0.0.
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+")


def _parse_score(raw: str) -> float:
    """Extract a relevance score from a model response, clamped to [0, 1]."""
    if raw is None:
        return 0.0
    try:
        score = float(raw.strip())
    except (ValueError, TypeError):
        match = _NUMBER_RE.search(raw)
        try:
            score = float(match.group())
        except (AttributeError, ValueError, TypeError):
            return 0.0
    return max(0.0, min(1.0, score))

# Lazy-loaded cross-encoder model — loaded once on first call
_cross_encoder = None


def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        import os

        from sentence_transformers import CrossEncoder

        # Use the HF_ENDPOINT from embedding config for consistency
        os.environ.setdefault("HF_ENDPOINT", config.embedding.hf_endpoint)

        # Model is configurable via env var for A/B comparison.
        # Production default: BAAI/bge-reranker-v2-m3 (568M, best quality).
        # Lightweight alternative: BAAI/bge-reranker-base (278M, ~2x faster on CPU).
        model_name = os.environ.get(
            "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        )
        logger.info("Loading reranker model: %s", model_name)
        _cross_encoder = CrossEncoder(
            model_name,
            activation_fn=__import__("torch").nn.Sigmoid(),
        )
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

    # Model loading is lazy and CPU-bound (a 568M weights download+init on
    # first use); run it in the thread pool so the first explicit rerank
    # never freezes the event loop.  Subsequent calls hit the cached
    # instance (one cheap thread-pool hop).
    model = await asyncio.to_thread(_get_cross_encoder)
    pairs = [[query, c] for c in candidates]
    scores = await asyncio.to_thread(model.predict, pairs)

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


async def rerank_llm(
    query: str, candidates: list[str], top_k: int = 5
) -> list[tuple[int, float]]:
    """Rerank candidates by asking the LLM to score each one (pointwise).

    One provider call per candidate; concurrent in-flight calls are capped
    by ``LLM_RERANK_CONCURRENCY`` (default 4) so a 20-40 candidate list
    can't self-inflict a rate-limit storm — the previous ``asyncio.gather``
    fired them all at once.

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

    # Call-local semaphore — bounded concurrency within one rerank call,
    # no coupling between concurrent rerank invocations.
    semaphore = asyncio.Semaphore(config.llm.rerank_concurrency)

    async def _score_one(idx: int, text: str) -> tuple[int, float, bool]:
        async with semaphore:
            prompt = _RERANK_PROMPT.format(query=query, text=text)
            try:
                # Scoring must be deterministic — temperature 0.0.
                response = await llm.chat(
                    [{"role": "user", "content": prompt}],
                    scenario="rerank_llm",
                    temperature=0.0,
                )
            except Exception:
                return idx, 0.0, True  # this candidate's call failed
            return idx, _parse_score(str(response)), False

    tasks = [_score_one(i, c) for i, c in enumerate(candidates)]
    results = await asyncio.gather(*tasks)

    # Channel-failure signal: when EVERY candidate's LLM call failed, the
    # reranker produced no signal at all — not "nothing is relevant".  Return
    # an empty list so callers fall back to the recall ranking instead of an
    # empty result.  Partial failures are treated as honest: the failed
    # candidates are dropped, and the surviving scores are trusted.
    if all(failed for _, _, failed in results):
        logger.warning(
            "LLM rerank: all %d candidate calls failed — returning empty "
            "channel-failure signal",
            len(candidates),
        )
        return []

    ranked = sorted(
        ((idx, score) for idx, score, failed in results if not failed),
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:top_k]


_RERANK_PROMPT = """\
Score how relevant the following text is to the query on a scale from 0 (completely irrelevant) to 1 (perfect match).

Query: {query}

Text: {text}

Reply with ONLY the score as a decimal number (e.g. 0.85). Do not include any other text."""
