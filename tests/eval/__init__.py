"""EMA retrieval evaluation suite.

Public API:
    metrics      — Recall@K / Precision@K / MRR / NDCG@K / HitRate@K / MAP@K
    ground_truth — labeled query set (30 queries, 5 categories)
    dataset      — fingerprint matching + retriever adapters
    runner       — single-config + A/B comparison runner
    report       — JSON + Markdown report generation
    run_eval     — CLI entry (`python -m tests.eval.run_eval`)
    seed         — seed corpus loader (`python -m tests.eval.seed`)

Design notes:
    - Ground truth uses **content fingerprints** (distinctive substrings of the
      relevant memory's summary/content) rather than UUIDs, so the labeled set
      is portable across DB rebuilds and reproducible in CI.
    - The runner is retriever-agnostic: pass any callable returning
      ``list[dict]`` with a configurable ``match_field`` (``content`` for the
      chunks table, ``summary`` for the memories table).
"""

from tests.eval.metrics import (
    hit_rate_at_k,
    map_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "hit_rate_at_k",
    "map_at_k",
    "mrr",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
