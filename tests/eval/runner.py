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
    relevance_mask,
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
    threshold: float | None = None
    categories: list[str] | None = None

    def adapter(self) -> RetrieverAdapter:
        return build_adapter(
            self.retriever,
            use_llm_rerank=self.use_llm_rerank,
            threshold=self.threshold,
        )

    @property
    def label(self) -> str:
        """Short human label, e.g. ``memory:ce@k5``."""
        rerank = "llm" if self.use_llm_rerank else "ce"
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
) -> EvalResult:
    """Execute one config over the labeled set.

    Args:
        config: what to run.
        items: labeled set (defaults to the full ``GROUND_TRUTH``).
        reraise: if True, propagate the first retrieval error instead of
            recording it. Useful in unit tests; False in production runs so
            one bad query doesn't kill the whole eval.

    Returns:
        ``EvalResult`` with per-query rows + aggregates.
    """
    items = list(items) if items is not None else load_ground_truth()
    items = _filter_items(items, config.categories)
    adapter = config.adapter()

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
            continue
        latency_ms = (time.perf_counter() - t0) * 1000
        result.total_latency_ms += latency_ms

        mask = relevance_mask(retrieved, it.relevant_fingerprints, adapter.match_field)
        # Positional IDs: retrieved = [0, 1, ..., n-1]; relevant = matched indices.
        retrieved_ids = list(range(len(retrieved)))
        relevant_ids = {i for i, hit in enumerate(mask) if hit}
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
) -> list[EvalResult]:
    """Run multiple configs over the same labeled set, in order.

    Returns results in the same order as ``configs``. The report module uses
    this to produce A/B delta tables. ``reraise`` is forwarded to each
    :func:`run_eval` call (useful in tests; defaults to False for production
    A/B runs where one bad query shouldn't kill the whole comparison).
    """
    items = list(items) if items is not None else load_ground_truth()
    results: list[EvalResult] = []
    for cfg in configs:
        results.append(await run_eval(cfg, items, reraise=reraise))
    return results


def config_from_dict(d: dict[str, Any]) -> EvalConfig:
    """Build an EvalConfig from a plain dict (for JSON-driven runs)."""
    return EvalConfig(
        name=str(d["name"]),
        retriever=str(d.get("retriever", "memory")),
        top_k=int(d.get("top_k", 5)),
        use_llm_rerank=bool(d.get("use_llm_rerank", False)),
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
