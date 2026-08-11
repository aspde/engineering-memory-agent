"""Unit tests for tests/eval/experiments/hard_negative.py — discrimination scoring.

Pure functions only (no DB / no LLM): the discrimination logic is the part
worth locking down — whether a trap seed ranked above the target correctly
fails the query, whether a target miss with a distractor hit counts as
worse-than-random, and that only ``kind="hard_negative"`` items are loaded.
The live retrieval run (query_memories + BGE-M3) is exercised by running the
module against a seeded eval DB, not in unit tests.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.eval.experiments.hard_negative import aggregate, load_hard_negatives


def _item(
    *,
    target_rank: int | None,
    distractor_rank: int | None,
    item_id: str = "q-test",
) -> dict[str, Any]:
    return {
        "id": item_id,
        "query": "test query",
        "target_seed": "seed-001",
        "distractor_seed": "seed-002",
        "target_rank": target_rank,
        "distractor_rank": distractor_rank,
        "discriminated": target_rank is not None
        and (distractor_rank is None or target_rank < distractor_rank),
    }


class TestAggregate:
    def test_empty(self) -> None:
        agg = aggregate([])
        assert agg["n"] == 0
        assert agg["target_recall@5"] == 0.0
        assert agg["distractor_intrusion@5"] == 0.0
        assert agg["hard_neg_pass@5"] == 0.0
        assert agg["mrr"] == 0.0

    def test_perfect_discrimination(self) -> None:
        # Target ranked 1, trap absent → all pass, mrr = 1/1.
        agg = aggregate([_item(target_rank=1, distractor_rank=None)])
        assert agg["n"] == 1
        assert agg["target_recall@5"] == 1.0
        assert agg["distractor_intrusion@5"] == 0.0
        assert agg["hard_neg_pass@5"] == 1.0
        assert agg["mrr"] == 1.0
        assert agg["worse_than_random"] == 0

    def test_trap_beats_target_fails(self) -> None:
        # Trap ranked above target → discriminated False, counts as worse.
        agg = aggregate([_item(target_rank=2, distractor_rank=1)])
        assert agg["hard_neg_pass@5"] == 0.0
        assert agg["mrr"] == 0.5  # 1/2
        assert agg["worse_than_random"] == 1

    def test_target_miss_with_trap_hit_is_worse_than_random(self) -> None:
        # Target not recalled but trap is → worst case: the retriever surfaced
        # the wrong memory for a query that had a right answer in corpus.
        agg = aggregate([_item(target_rank=None, distractor_rank=1)])
        assert agg["target_recall@5"] == 0.0
        assert agg["distractor_intrusion@5"] == 1.0
        assert agg["hard_neg_pass@5"] == 0.0
        assert agg["worse_than_random"] == 1

    def test_trap_ranked_below_target_passes(self) -> None:
        # Both recalled, but target wins the rank contest → pass.
        agg = aggregate([_item(target_rank=1, distractor_rank=2)])
        assert agg["hard_neg_pass@5"] == 1.0
        assert agg["worse_than_random"] == 0

    def test_mrr_averages_reciprocal_ranks(self) -> None:
        agg = aggregate(
            [
                _item(target_rank=1, distractor_rank=None, item_id="a"),
                _item(target_rank=3, distractor_rank=2, item_id="b"),
                _item(target_rank=None, distractor_rank=1, item_id="c"),
            ]
        )
        # (1/1 + 1/3 + 0) / 3
        assert agg["mrr"] == pytest.approx((1.0 + 1 / 3 + 0.0) / 3)


class TestLoadHardNegatives:
    def test_only_hard_negative_kind_loaded(self) -> None:
        items = load_hard_negatives()
        # query_candidates.jsonl has 27 hard_negative among 108 total.
        assert len(items) >= 27
        assert all(it.get("kind") == "hard_negative" for it in items)
        assert all(it.get("seed_ids") for it in items)
        assert all(it.get("distractor_seed_ids") for it in items)
