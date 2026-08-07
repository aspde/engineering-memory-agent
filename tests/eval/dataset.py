"""Dataset loading, fingerprint matching, and retriever adapters.

The eval pipeline is retriever-agnostic. ``make_chunk_retriever`` and
``make_memory_retriever`` return ``(callable, match_field)`` pairs so the
runner can stay indifferent to whether results come from the chunks table
(``retrieval.retrieve``) or the memories table (``retrieval.query_memories``).

Fingerprint matching is the bridge between ground truth and retrieved
results: a retrieved item is "relevant" iff its ``match_field`` text contains
any of the query's ``relevant_fingerprints``. This avoids UUID coupling and
works identically for both tables.
"""

from __future__ import annotations

import json
import math
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tests.eval.ground_truth import GROUND_TRUTH, GroundTruthItem

SEED_FILE = Path(__file__).parent / "seed_memories.jsonl"

# A retriever callable: (query, top_k) -> list of result dicts.
RetrieverFn = Callable[[str, int], Awaitable[list[dict[str, Any]]]]


@dataclass(frozen=True)
class RetrieverAdapter:
    """Bundles a retriever callable with the field used for fingerprint match.

    ``name`` is used in reports; ``match_field`` is the key in each result
    dict whose value is matched against fingerprints (``content`` for chunks,
    ``summary`` for memories).
    """

    name: str
    fn: RetrieverFn
    match_field: str


# ── Seed corpus ────────────────────────────────────────────────


@dataclass
class SeedMemory:
    id: str
    category: str
    source_type: str
    summary: str
    content: str
    entities: list[dict]
    relations: list[dict]

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SeedMemory":
        return cls(
            id=str(d["id"]),
            category=str(d.get("category", "")),
            source_type=str(d.get("source_type", "")),
            summary=str(d["summary"]),
            content=str(d.get("content", d["summary"])),
            entities=list(d.get("entities", [])),
            relations=list(d.get("relations", [])),
        )


def load_seed_memories(path: Path = SEED_FILE) -> list[SeedMemory]:
    """Load seed memories from JSONL. Raises if file missing or malformed."""
    if not path.exists():
        raise FileNotFoundError(f"seed file not found: {path}")
    seeds: list[SeedMemory] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                seeds.append(SeedMemory.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError) as e:
                raise ValueError(f"{path}:{lineno}: invalid seed entry: {e}") from e
    if not seeds:
        raise ValueError(f"seed file is empty: {path}")
    return seeds


def load_ground_truth() -> list[GroundTruthItem]:
    """Return the labeled set (already validated at module import time)."""
    return list(GROUND_TRUTH)


# ── Validation ────────────────────────────────────────────────


def validate_dataset(
    items: Sequence[GroundTruthItem] | None = None,
    seeds: Sequence[SeedMemory] | None = None,
) -> list[str]:
    """Validate ground truth ↔ seed corpus consistency.

    Returns a list of human-readable warnings (empty == clean).
    Raises ``ValueError`` only for hard failures that would make eval
    results meaningless (e.g., a fingerprint that matches zero seeds).

    Checks:
        1. Every seed_id referenced by ground truth exists in the seed file.
        2. Every fingerprint appears in the summary OR content of *exactly
           one* seed (uniqueness — otherwise relevance is ambiguous).
           Summary is matched by the ``memory`` retriever, content by the
           ``chunk``/``hybrid``/``rewrite`` retrievers, so a fingerprint
           valid for one path must be discoverable here.
        3. Every fingerprint appears in at least one seed's summary or
           content (existence — otherwise the query can never be satisfied).
        4. Every fingerprint's owning seed is in the query's seed_ids
           (cross-reference integrity).
    """
    items = list(items) if items is not None else load_ground_truth()
    seeds = list(seeds) if seeds is not None else load_seed_memories()
    seed_by_id: dict[str, SeedMemory] = {s.id: s for s in seeds}
    warnings: list[str] = []

    # Check 1: seed_id existence
    for it in items:
        for sid in it.seed_ids:
            if sid not in seed_by_id:
                raise ValueError(
                    f"{it.id}: references missing seed_id '{sid}'"
                )

    # Pre-compute fingerprint → owning seed ids.  A fingerprint "belongs" to
    # a seed if it appears in that seed's summary OR content — the two fields
    # the different retriever paths match against (see RetrieverAdapter.match_field).
    fp_to_seed_ids: dict[str, list[str]] = {}
    for s in seeds:
        for it in items:
            for fp in it.relevant_fingerprints:
                if fp in s.summary or fp in s.content:
                    fp_to_seed_ids.setdefault(fp, []).append(s.id)

    # Check 2 & 3: fingerprint existence + uniqueness
    for it in items:
        for fp in it.relevant_fingerprints:
            owners = fp_to_seed_ids.get(fp, [])
            if not owners:
                raise ValueError(
                    f"{it.id}: fingerprint '{fp}' not found in any seed summary"
                )
            if len(owners) > 1:
                warnings.append(
                    f"{it.id}: fingerprint '{fp}' matches multiple seeds "
                    f"{owners}; relevance will be ambiguous"
                )

    # Check 4: cross-reference — fingerprint owner should be in seed_ids
    for it in items:
        for fp in it.relevant_fingerprints:
            owners = fp_to_seed_ids.get(fp, [])
            if owners and not set(owners).issubset(set(it.seed_ids)):
                warnings.append(
                    f"{it.id}: fingerprint '{fp}' is owned by {owners} "
                    f"but seed_ids={it.seed_ids}"
                )

    return warnings


# ── Fingerprint matching ──────────────────────────────────────


def is_relevant(result: dict[str, Any], fingerprints: Iterable[str], match_field: str) -> bool:
    """Return True iff ``result[match_field]`` contains any fingerprint.

    Substring match is intentional: fingerprints are designed as distinctive
    spans (e.g. ``"pgvector 而非 Elasticsearch"``) that uniquely identify a
    memory. Case-sensitive because EMA fingerprints are deliberately cased
    (e.g. ``"BGE-M3"`` vs ``"bge-m3"``).
    """
    text = str(result.get(match_field, ""))
    if not text:
        return False
    return any(fp in text for fp in fingerprints)


def relevance_mask(
    results: Sequence[dict[str, Any]],
    fingerprints: Iterable[str],
    match_field: str,
) -> list[bool]:
    """Map each retrieved result to a binary relevance label."""
    fps = list(fingerprints)
    return [is_relevant(r, fps, match_field) for r in results]


# ── Semantic relevance (supplementary) ──────────────────────────
# Substring fingerprints are lexical anchors; a retriever that returns a
# *paraphrase* of the ground truth (no surface overlap) scores a miss even
# when it is semantically on target.  ``semantic_relevance_mask`` is the
# counterweight: it marks a result relevant when its embedding is within
# ``SEMANTIC_THRESHOLD`` of any target seed summary.  It only *adds* hits
# that substring matching would have missed — the OR combination cannot
# demote an already-relevant result — so it measures the semantic dimension
# of retrieval quality without weakening the lexical guarantees.

SEMANTIC_THRESHOLD = 0.80  # conservative cosine floor for "same meaning"


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


async def semantic_relevance_mask(
    results: Sequence[dict[str, Any]],
    seed_summaries: Sequence[str],
    match_field: str,
    threshold: float = SEMANTIC_THRESHOLD,
) -> list[bool]:
    """Mark each result relevant if it embeds close to any target summary.

    Args:
        results: Retrieved rows (dicts with ``match_field`` text).
        seed_summaries: The ground-truth summaries this query targets
            (i.e. the summaries of ``item.seed_ids``).
        match_field: Key holding the text to compare (``content`` /
            ``summary``).
        threshold: Cosine floor for "semantically same".

    The embedding provider is the same one under evaluation; this is
    acceptable because the gate is conservative and purely additive.
    """
    if not results or not seed_summaries:
        return [False] * len(results)

    from backend.service.embedding_service import get_embedding_provider

    provider = get_embedding_provider()
    target_vecs = await provider.embed(list(seed_summaries))
    texts = [str(r.get(match_field, "") or "") for r in results]
    result_vecs = await provider.embed(texts)

    mask: list[bool] = []
    for rv in result_vecs:
        best = max(_cosine(rv, tv) for tv in target_vecs)
        mask.append(best >= threshold)
    return mask


# ── Retriever adapters ────────────────────────────────────────


def make_chunk_retriever(
    *,
    use_llm_rerank: bool = False,
    threshold: float = 0.0,
) -> RetrieverAdapter:
    """Adapter for ``retrieval.retrieve`` (chunks table).

    Returns results as ``[{"content": str, "score": float, "metadata": dict}]``
    with ``match_field="content"``.
    """
    from backend.service.retrieval import retrieve

    async def _fn(query: str, top_k: int) -> list[dict[str, Any]]:
        results = await retrieve(
            query, top_k=top_k, use_llm_rerank=use_llm_rerank
        )
        # RetrievalResult dataclass → dict; threshold is applied upstream in
        # vector_search, kept here for API symmetry with the memory adapter.
        _ = threshold  # acknowledged, no-op for chunks path
        return [
            {"content": r.content, "score": r.score, "metadata": r.metadata}
            for r in results
        ]

    rerank_tag = "llm" if use_llm_rerank else "ce"
    return RetrieverAdapter(name=f"chunk:{rerank_tag}", fn=_fn, match_field="content")


def make_memory_retriever(
    *,
    use_llm_rerank: bool = False,
    threshold: float = 0.3,
) -> RetrieverAdapter:
    """Adapter for ``retrieval.query_memories`` (memories table).

    Returns results as ``[{"id": str, "summary": str, "rerank_score": float, ...}]``
    with ``match_field="summary"``.
    """
    from backend.service.retrieval import query_memories

    async def _fn(query: str, top_k: int) -> list[dict[str, Any]]:
        return await query_memories(
            query,
            top_k=top_k,
            threshold=threshold,
            use_llm_rerank=use_llm_rerank,
        )

    rerank_tag = "llm" if use_llm_rerank else "ce"
    return RetrieverAdapter(name=f"memory:{rerank_tag}", fn=_fn, match_field="summary")


def make_vector_retriever(*, threshold: float = 0.0) -> RetrieverAdapter:
    """Adapter for raw vector recall (embed_query + vector_search, NO rerank).

    Bypasses cross-encoder rerank entirely. Use this when measuring pure
    BGE-M3 dense recall, or when cross-encoder CPU latency (~50s/query) is
    prohibitive. Returns chunks-table rows with ``match_field="content"``.
    """
    from backend.service.retrieval import embed_query, vector_search

    async def _fn(query: str, top_k: int) -> list[dict[str, Any]]:
        vec = await embed_query(query)
        return await vector_search(vec, top_k=top_k, threshold=threshold)

    return RetrieverAdapter(name="vector:raw", fn=_fn, match_field="content")


def make_hybrid_retriever(
    *, use_llm_rerank: bool = False, skip_rerank: bool = False
) -> RetrieverAdapter:
    """Adapter for ``retrieve_hybrid`` (dense vector + sparse BM25, with rerank).

    Returns chunks-table rows (``content`` field) after dense+sparse union
    and cross-encoder rerank.  Requires the ``tokens`` column on chunks.

    When ``skip_rerank`` is True, candidates are ranked by max(dense
    similarity, sparse jaccard) without cross-encoder — used to measure
    rerank's contribution.
    """
    from backend.service.retrieval import retrieve_hybrid

    async def _fn(query: str, top_k: int) -> list[dict[str, Any]]:
        results = await retrieve_hybrid(
            query,
            top_k=top_k,
            use_llm_rerank=use_llm_rerank,
            skip_rerank=skip_rerank,
        )
        return [
            {"content": r.content, "score": r.score, "meta": r.metadata}
            for r in results
        ]

    if skip_rerank:
        rerank_tag = "norank"
    else:
        rerank_tag = "llm" if use_llm_rerank else "ce"
    return RetrieverAdapter(name=f"hybrid:{rerank_tag}", fn=_fn, match_field="content")


def make_rewrite_retriever(*, use_llm_rerank: bool = False) -> RetrieverAdapter:
    """Adapter for ``retrieve_multi_query`` (LLM rewrite + multi-query union + rerank).

    Returns chunks-table rows (``content`` field).  Costs one extra LLM
    call for rewriting; fails safe to single-query on rewrite error.
    """
    from backend.service.retrieval import retrieve_multi_query

    async def _fn(query: str, top_k: int) -> list[dict[str, Any]]:
        results = await retrieve_multi_query(
            query, top_k=top_k, use_llm_rerank=use_llm_rerank
        )
        return [
            {"content": r.content, "score": r.score, "meta": r.metadata}
            for r in results
        ]

    rerank_tag = "llm" if use_llm_rerank else "ce"
    return RetrieverAdapter(name=f"rewrite:{rerank_tag}", fn=_fn, match_field="content")


# ── Convenience for the runner ────────────────────────────────


def build_adapter(
    retriever: str,
    *,
    use_llm_rerank: bool = False,
    threshold: float | None = None,
) -> RetrieverAdapter:
    """Construct a RetrieverAdapter by name.

    Args:
        retriever: ``"chunk"`` or ``"memory"``.
        use_llm_rerank: route to ``rerank_llm`` instead of cross-encoder.
        threshold: similarity floor. ``None`` uses each path's default
            (0.0 for chunks, 0.3 for memories).
    """
    if retriever == "chunk":
        return make_chunk_retriever(
            use_llm_rerank=use_llm_rerank,
            threshold=threshold if threshold is not None else 0.0,
        )
    if retriever == "memory":
        return make_memory_retriever(
            use_llm_rerank=use_llm_rerank,
            threshold=threshold if threshold is not None else 0.3,
        )
    if retriever == "vector":
        return make_vector_retriever(
            threshold=threshold if threshold is not None else 0.0,
        )
    if retriever == "hybrid":
        return make_hybrid_retriever(use_llm_rerank=use_llm_rerank)
    if retriever == "hybrid_norerank":
        return make_hybrid_retriever(skip_rerank=True)
    if retriever == "rewrite":
        return make_rewrite_retriever(use_llm_rerank=use_llm_rerank)
    raise ValueError(
        f"unknown retriever: {retriever!r} "
        "(expected 'chunk', 'memory', 'vector', 'hybrid', "
        "'hybrid_norerank', or 'rewrite')"
    )
