"""Evaluation runner — runs a config over the labeled set and aggregates.

The runner is the orchestrator: it takes an ``EvalConfig`` (which retriever,
rerank mode, top_k, threshold, optional category filter), executes every
labeled query, computes per-query metrics via :mod:`tests.eval.metrics`, and
rolls up overall / per-category / per-difficulty aggregates.

Key design: metrics are computed on **positional IDs** — the runner maps each
retrieved result to its index ``0..n-1``, and the "relevant" set is the set
of indices whose ``match_field`` matched a fingerprint. This lets the pure
``metrics`` functions stay I/O-free and reusable.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from tests.eval.dataset import (
    RetrieverAdapter,
    build_adapter,
    load_ground_truth,
    load_seed_memories,
    relevance_mask,
    semantic_relevance_mask,
)
from tests.eval.ground_truth import CATEGORIES, DIFFICULTIES, GroundTruthItem
from tests.eval.metrics import compute_all

logger = logging.getLogger(__name__)

# Metrics reported in every aggregate block, in display order.
METRIC_KEYS: tuple[str, ...] = (
    "recall@5",
    "precision@5",
    "hit_rate@5",
    "mrr",
    "ndcg@5",
    "map@5",
)
# Auxiliary (non-metric) keys included in per_query rows.
AUX_KEYS: tuple[str, ...] = ("n_retrieved", "n_relevant", "latency_ms")


@dataclass
class EvalConfig:
    """One eval run configuration.

    ``name`` shows up in reports and A/B comparison tables. ``categories``
    filters the labeled set (None = all categories).
    """

    name: str
    retriever: str = "memory"  # "chunk" | "memory"
    top_k: int = 5
    use_llm_rerank: bool = False
    use_cross_encoder: bool = False
    threshold: float | None = None
    categories: list[str] | None = None

    def adapter(self) -> RetrieverAdapter:
        return build_adapter(
            self.retriever,
            use_llm_rerank=self.use_llm_rerank,
            use_cross_encoder=self.use_cross_encoder,
            threshold=self.threshold,
        )

    @property
    def label(self) -> str:
        """Short human label, e.g. ``memory:norank@k5``.

        ``norank`` is the default read path (no cross-encoder); ``ce`` and
        ``llm`` label the explicit opt-in rerankers.
        """
        rerank = "llm" if self.use_llm_rerank else (
            "ce" if self.use_cross_encoder else "norank"
        )
        return f"{self.retriever}:{rerank}@k{self.top_k}"


@dataclass
class EvalResult:
    """Aggregate result of one ``EvalConfig`` over the labeled set."""

    config: EvalConfig
    per_query: list[dict[str, Any]] = field(default_factory=list)
    overall: dict[str, float] = field(default_factory=dict)
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    n_queries: int = 0
    total_latency_ms: float = 0.0
    errors: list[dict[str, str]] = field(default_factory=list)

    def metric(self, key: str) -> float:
        """Read an overall metric (0.0 if absent)."""
        return float(self.overall.get(key, 0.0))


def _filter_items(
    items: Sequence[GroundTruthItem], categories: list[str] | None
) -> list[GroundTruthItem]:
    if not categories:
        return list(items)
    return [it for it in items if it.category in categories]


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Mean of each metric/aux key across rows. Empty input → all zeros."""
    if not rows:
        return {k: 0.0 for k in METRIC_KEYS + AUX_KEYS}
    out: dict[str, float] = {}
    for k in METRIC_KEYS + AUX_KEYS:
        vals = [float(r.get(k, 0.0)) for r in rows]
        out[k] = sum(vals) / len(vals)
    return out


async def run_eval(
    config: EvalConfig,
    items: Sequence[GroundTruthItem] | None = None,
    *,
    reraise: bool = False,
    semantic_relevance: bool = True,
) -> EvalResult:
    """Execute one config over the labeled set.

    Args:
        config: what to run.
        items: labeled set (defaults to the full ``GROUND_TRUTH``).
        reraise: if True, propagate the first retrieval error instead of
            recording it. Useful in unit tests; False in production runs so
            one bad query doesn't kill the whole eval.
        semantic_relevance: if True (default), relevance is a substring
            match OR an embedding-similarity match against the query's
            target seed summaries (see ``dataset.semantic_relevance_mask``).
            Measures the semantic dimension of retrieval; needs the
            embedding provider.  Pass ``False`` to fall back to the pure,
            deterministic fingerprint matching as a lexical baseline.
            Per-query rows record both channels (``substring_hits`` /
            ``semantic_only_hits``) so the semantic contribution is
            observable either way.

    Returns:
        ``EvalResult`` with per-query rows + aggregates.
    """
    items = list(items) if items is not None else load_ground_truth()
    items = _filter_items(items, config.categories)
    adapter = config.adapter()

    # seed_id → summary, for the semantic relevance pass (only if enabled).
    seed_summary_by_id: dict[str, str] = {}
    if semantic_relevance:
        seed_summary_by_id = {
            s.id: s.summary for s in load_seed_memories()
        }

    result = EvalResult(config=config, n_queries=len(items))
    logger.info("Starting eval: config=%s queries=%d", config.label, len(items))

    for it in items:
        t0 = time.perf_counter()
        try:
            retrieved = await adapter.fn(it.query, config.top_k)
        except Exception as e:
            logger.warning("query %s failed: %s", it.id, e)
            result.errors.append({"id": it.id, "error": str(e)})
            if reraise:
                raise
            # Failed queries still count toward the denominator: aggregate
            # them as zero-recall rows.  Dropping them (the old behaviour)
            # made the overall numbers inflate as errors rose — recall@5
            # looked *better* the more queries blew up.
            result.per_query.append(
                {
                    "id": it.id,
                    "query": it.query,
                    "category": it.category,
                    "difficulty": it.difficulty,
                    "n_retrieved": 0,
                    "n_relevant": 0,
                    "latency_ms": 0.0,
                    "error": str(e),
                    "semantic_relevance": semantic_relevance,
                    "substring_hits": 0,
                    "semantic_only_hits": 0,
                    **compute_all([], set(), k=config.top_k),
                }
            )
            continue
        latency_ms = (time.perf_counter() - t0) * 1000
        result.total_latency_ms += latency_ms

        sub_mask = relevance_mask(retrieved, it.relevant_fingerprints, adapter.match_field)
        sub_hits = {i for i, hit in enumerate(sub_mask) if hit}
        # Semantic supplement: only consulted when substring matching left
        # some results unmatched, and only ever flips False → True.  Hits
        # contributed solely by the semantic channel are tracked separately
        # so the report can show what the lexical baseline missed.
        sem_only: set[int] = set()
        if semantic_relevance and any(not h for h in sub_mask) and retrieved:
            targets = [
                seed_summary_by_id[sid]
                for sid in it.seed_ids
                if sid in seed_summary_by_id
            ]
            if targets:
                sem_mask = await semantic_relevance_mask(
                    retrieved, targets, adapter.match_field
                )
                sem_only = {
                    i for i, hit in enumerate(sem_mask)
                    if hit and not sub_mask[i]
                }
        # Positional IDs: retrieved = [0, 1, ..., n-1]; relevant = matched indices.
        retrieved_ids = list(range(len(retrieved)))
        relevant_ids = sub_hits | sem_only
        metrics = compute_all(retrieved_ids, relevant_ids, k=config.top_k)

        result.per_query.append(
            {
                "id": it.id,
                "query": it.query,
                "category": it.category,
                "difficulty": it.difficulty,
                "n_retrieved": len(retrieved),
                "n_relevant": len(relevant_ids),
                "latency_ms": latency_ms,
                "semantic_relevance": semantic_relevance,
                "substring_hits": len(sub_hits),
                "semantic_only_hits": len(sem_only),
                **metrics,
            }
        )

    # Aggregates — by_category / by_difficulty always emit every bucket (even
    # empty ones, which _aggregate([]) turns into all-zero metric rows) so the
    # report's category/difficulty tables have stable columns.
    result.overall = _aggregate(result.per_query)
    result.by_category = {
        cat: _aggregate([r for r in result.per_query if r.get("category") == cat])
        for cat in CATEGORIES
    }
    result.by_difficulty = {
        diff: _aggregate(
            [r for r in result.per_query if r.get("difficulty") == diff]
        )
        for diff in DIFFICULTIES
    }

    logger.info(
        "Eval complete: config=%s recall@5=%.3f mrr=%.3f ndcg@5=%.3f "
        "queries=%d errors=%d avg_latency=%.0fms",
        config.label,
        result.metric("recall@5"),
        result.metric("mrr"),
        result.metric("ndcg@5"),
        result.n_queries,
        len(result.errors),
        result.overall.get("latency_ms", 0.0),
    )
    return result


async def compare_eval(
    configs: Sequence[EvalConfig],
    items: Sequence[GroundTruthItem] | None = None,
    *,
    reraise: bool = False,
    semantic_relevance: bool = True,
) -> list[EvalResult]:
    """Run multiple configs over the same labeled set, in order.

    Returns results in the same order as ``configs``. The report module uses
    this to produce A/B delta tables. ``reraise`` is forwarded to each
    :func:`run_eval` call (useful in tests; defaults to False for production
    A/B runs where one bad query shouldn't kill the whole comparison).
    ``semantic_relevance`` is likewise forwarded (defaults to True — the
    semantic channel is on unless explicitly disabled).
    """
    items = list(items) if items is not None else load_ground_truth()
    results: list[EvalResult] = []
    for cfg in configs:
        results.append(
            await run_eval(cfg, items, reraise=reraise, semantic_relevance=semantic_relevance)
        )
    return results


def config_from_dict(d: dict[str, Any]) -> EvalConfig:
    """Build an EvalConfig from a plain dict (for JSON-driven runs)."""
    return EvalConfig(
        name=str(d["name"]),
        retriever=str(d.get("retriever", "memory")),
        top_k=int(d.get("top_k", 5)),
        use_llm_rerank=bool(d.get("use_llm_rerank", False)),
        use_cross_encoder=bool(d.get("use_cross_encoder", False)),
        threshold=d.get("threshold"),
        categories=list(d["categories"]) if d.get("categories") else None,
    )


def result_to_dict(result: EvalResult) -> dict[str, Any]:
    """Serialize an EvalResult for JSON reporting."""
    return {
        "config": asdict(result.config),
        "n_queries": result.n_queries,
        "total_latency_ms": result.total_latency_ms,
        "overall": result.overall,
        "by_category": result.by_category,
        "by_difficulty": result.by_difficulty,
        "per_query": result.per_query,
        "errors": result.errors,
    }
