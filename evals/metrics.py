"""Retrieval evaluation metrics — pure functions, no I/O.

All metrics assume **binary relevance**: an item is either relevant (1) or not
(0). Graded relevance is intentionally unsupported to keep the labeled set
construction cheap (hand-labeling fine-grained grades is not worth it for a
20-30 query eval set).

Conventions:
    - ``retrieved`` is the ranked list of retrieved item identifiers (or any
      hashable tokens) in the order returned by the retriever.
    - ``relevant`` is the set (or list) of identifiers that are truly relevant.
    - ``k`` is the cutoff rank; metrics only look at the top-k of ``retrieved``.
    - All functions return ``0.0`` (never raise) on degenerate inputs, so the
      runner can aggregate without try/except noise.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def _relevant_set(relevant: Iterable[object]) -> set[object]:
    return set(relevant)


def recall_at_k(
    retrieved: Sequence[object], relevant: Iterable[object], k: int = 5
) -> float:
    """Recall@K: fraction of relevant items found in the top-k.

    recall = |relevant ∩ top_k| / |relevant|
    """
    rel = _relevant_set(relevant)
    if not rel or k <= 0:
        return 0.0
    top_k = list(retrieved)[:k]
    hits = len(rel.intersection(top_k))
    return hits / len(rel)


def precision_at_k(
    retrieved: Sequence[object], relevant: Iterable[object], k: int = 5
) -> float:
    """Precision@K: fraction of top-k that is relevant.

    precision = |relevant ∩ top_k| / k
    """
    if k <= 0:
        return 0.0
    rel = _relevant_set(relevant)
    top_k = list(retrieved)[:k]
    hits = len(rel.intersection(top_k))
    return hits / k


def hit_rate_at_k(
    retrieved: Sequence[object], relevant: Iterable[object], k: int = 5
) -> float:
    """HitRate@K: 1.0 if any relevant item appears in top-k, else 0.0.

    Also known as Recall@K with binary relevance and a single relevant item,
    but kept separate because it answers a different question
    ("did the user see at least one good result?") and aggregates differently.
    """
    rel = _relevant_set(relevant)
    if not rel or k <= 0:
        return 0.0
    top_k = set(list(retrieved)[:k])
    return 1.0 if rel.intersection(top_k) else 0.0


def mrr(retrieved: Sequence[object], relevant: Iterable[object]) -> float:
    """Mean Reciprocal Rank: 1/rank of the first relevant item.

    Returns 0.0 if no relevant item is in the list.
    """
    rel = _relevant_set(relevant)
    if not rel:
        return 0.0
    for i, item in enumerate(retrieved, start=1):
        if item in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    retrieved: Sequence[object], relevant: Iterable[object], k: int = 5
) -> float:
    """Normalized Discounted Cumulative Gain @ K with binary relevance.

    DCG@k  = sum_{i=1..k}  rel_i / log2(i + 1)
    IDCG@k = DCG@k of the ideal ranking (all relevant items first)
    NDCG@k = DCG@k / IDCG@k

    With binary relevance, rel_i ∈ {0, 1}, so each hit contributes 1/log2(i+1).
    """
    rel = _relevant_set(relevant)
    if not rel or k <= 0:
        return 0.0

    top_k = list(retrieved)[:k]
    dcg = 0.0
    for i, item in enumerate(top_k, start=1):
        if item in rel:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(
    retrieved: Sequence[object], relevant: Iterable[object], k: int = 5
) -> float:
    """Average Precision @ K for a single query.

    AP@k = (1 / min(|relevant|, k)) * sum_{i=1..k} (Precision@i * rel_i)

    where rel_i is 1 if the i-th retrieved item is relevant, else 0.
    The ``1 / min(|relevant|, k)`` normalization follows the TREC convention
    so AP is comparable across queries with different numbers of relevant
    items and is bounded in [0, 1].
    """
    rel = _relevant_set(relevant)
    if not rel or k <= 0:
        return 0.0

    top_k = list(retrieved)[:k]
    hits = 0
    sum_precise = 0.0
    for i, item in enumerate(top_k, start=1):
        if item in rel:
            hits += 1
            sum_precise += hits / i  # Precision@i at the moment of a hit
    if hits == 0:
        return 0.0
    normalizer = min(len(rel), k)
    return sum_precise / normalizer if normalizer > 0 else 0.0


def map_at_k(
    queries: Sequence[tuple[Sequence[object], Iterable[object]]], k: int = 5
) -> float:
    """Mean Average Precision @ K across multiple queries.

    Args:
        queries: sequence of ``(retrieved, relevant)`` pairs.
        k: cutoff rank.

    Returns:
        Mean of AP@k over all queries. Empty input returns 0.0.
    """
    if not queries:
        return 0.0
    total = 0.0
    for retrieved, relevant in queries:
        total += average_precision_at_k(retrieved, relevant, k)
    return total / len(queries)


METRIC_NAMES: tuple[str, ...] = (
    "recall@5",
    "precision@5",
    "hit_rate@5",
    "mrr",
    "ndcg@5",
    "map@5",
)


def compute_all(
    retrieved: Sequence[object], relevant: Iterable[object], k: int = 5
) -> dict[str, float]:
    """Compute all standard metrics for one query, keyed by ``METRIC_NAMES``.

    Convenience for the runner so it doesn't have to call six functions per
    query. ``k`` is applied to every @K metric; MRR uses the full list.
    """
    rel = _relevant_set(relevant)
    return {
        f"recall@{k}": recall_at_k(retrieved, rel, k),
        f"precision@{k}": precision_at_k(retrieved, rel, k),
        f"hit_rate@{k}": hit_rate_at_k(retrieved, rel, k),
        "mrr": mrr(retrieved, rel),
        f"ndcg@{k}": ndcg_at_k(retrieved, rel, k),
        f"map@{k}": average_precision_at_k(retrieved, rel, k),
    }
