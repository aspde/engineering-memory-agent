"""Unit tests for tests/eval/metrics.py — pure-function correctness.

Each metric is exercised against hand-computed cases so a regression in the
formula (e.g. off-by-one in NDCG's log2) surfaces immediately. Degenerate
inputs (empty relevant set, k<=0, no hits) are required to return 0.0 rather
than raise, so the runner can aggregate without try/except noise.
"""

from __future__ import annotations

import math

import pytest

from tests.eval.metrics import (
    METRIC_NAMES,
    average_precision_at_k,
    compute_all,
    hit_rate_at_k,
    map_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        assert recall_at_k(["a", "b", "c", "d", "e"], {"a", "c"}, k=5) == 1.0

    def test_partial_recall(self) -> None:
        # 1 of 2 relevant in top-5
        assert recall_at_k(["a", "b", "c", "d", "e"], {"a", "x"}, k=5) == 0.5

    def test_no_hits(self) -> None:
        assert recall_at_k(["a", "b", "c"], {"x", "y"}, k=5) == 0.0

    def test_relevant_beyond_k(self) -> None:
        # relevant item at position 6, k=5 → not counted
        assert recall_at_k(["a", "b", "c", "d", "e", "f"], {"f"}, k=5) == 0.0

    def test_relevant_within_k_but_list_shorter(self) -> None:
        # list shorter than k, hit present
        assert recall_at_k(["a", "b"], {"b"}, k=5) == 1.0

    def test_empty_relevant(self) -> None:
        assert recall_at_k(["a", "b"], set(), k=5) == 0.0

    def test_k_zero(self) -> None:
        assert recall_at_k(["a"], {"a"}, k=0) == 0.0

    def test_accepts_list_relevant(self) -> None:
        # list should work same as set
        assert recall_at_k(["a", "b"], ["a", "b"], k=5) == 1.0


class TestPrecisionAtK:
    def test_basic(self) -> None:
        # 2 hits / k=5
        assert precision_at_k(["a", "b", "c", "d", "e"], {"a", "c"}, k=5) == 0.4

    def test_all_relevant(self) -> None:
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_k_zero(self) -> None:
        assert precision_at_k(["a"], {"a"}, k=0) == 0.0

    def test_list_shorter_than_k(self) -> None:
        # precision = hits/k, not hits/len(list)
        assert precision_at_k(["a"], {"a"}, k=5) == pytest.approx(0.2)


class TestHitRateAtK:
    def test_hit(self) -> None:
        assert hit_rate_at_k(["a", "b", "c", "d", "e"], {"a"}, k=5) == 1.0

    def test_miss(self) -> None:
        assert hit_rate_at_k(["a", "b", "c", "d", "e"], {"x"}, k=5) == 0.0

    def test_hit_beyond_k(self) -> None:
        assert hit_rate_at_k(["a", "b", "c", "d", "e", "f"], {"f"}, k=5) == 0.0

    def test_empty_relevant(self) -> None:
        assert hit_rate_at_k(["a"], set(), k=5) == 0.0


class TestMRR:
    def test_first_position(self) -> None:
        assert mrr(["a", "b", "c"], {"a"}) == 1.0

    def test_second_position(self) -> None:
        assert mrr(["a", "b", "c"], {"b"}) == 0.5

    def test_third_position(self) -> None:
        assert mrr(["a", "b", "c"], {"c"}) == pytest.approx(1 / 3)

    def test_no_hit(self) -> None:
        assert mrr(["a", "b", "c"], {"x"}) == 0.0

    def test_first_relevant_wins(self) -> None:
        # multiple relevant; MRR uses rank of FIRST hit
        assert mrr(["a", "b", "c"], {"b", "c"}) == 0.5

    def test_empty_relevant(self) -> None:
        assert mrr(["a", "b"], set()) == 0.0

    def test_uses_full_list_not_k(self) -> None:
        # MRR has no k cutoff; a hit at position 10 still counts
        assert mrr(["a"] * 9 + ["b"], {"b"}) == pytest.approx(0.1)


class TestNDCGAtK:
    def test_ideal_ranking(self) -> None:
        # all relevant items first → NDCG = 1.0
        assert ndcg_at_k(["a", "b", "c", "d", "e"], {"a", "b"}, k=5) == 1.0

    def test_known_value_two_relevant(self) -> None:
        # retrieved=[a,b,c,d,e], relevant={a,c}
        # DCG = 1/log2(2) + 1/log2(4) = 1 + 0.5 = 1.5
        # IDCG = 1/log2(2) + 1/log2(3) = 1 + 1/1.58496 ≈ 1.63093
        # NDCG = 1.5 / 1.63093 ≈ 0.9196
        result = ndcg_at_k(["a", "b", "c", "d", "e"], {"a", "c"}, k=5)
        expected = 1.5 / (1.0 + 1 / math.log2(3))
        assert result == pytest.approx(expected, rel=1e-6)

    def test_relevant_at_last_position(self) -> None:
        # retrieved=[a,b,c,d,e], relevant={e}
        # DCG = 1/log2(6)
        # IDCG = 1/log2(2) = 1.0
        result = ndcg_at_k(["a", "b", "c", "d", "e"], {"e"}, k=5)
        assert result == pytest.approx(1 / math.log2(6))

    def test_no_hit(self) -> None:
        assert ndcg_at_k(["a", "b", "c"], {"x"}, k=5) == 0.0

    def test_empty_relevant(self) -> None:
        assert ndcg_at_k(["a"], set(), k=5) == 0.0

    def test_k_zero(self) -> None:
        assert ndcg_at_k(["a"], {"a"}, k=0) == 0.0

    def test_more_relevant_than_k(self) -> None:
        # 3 relevant items but k=2 → IDCG capped at k=2
        # IDCG = 1/log2(2) + 1/log2(3)
        # If both hits in top-2: DCG = IDCG → NDCG = 1.0
        assert ndcg_at_k(["a", "b", "c"], {"a", "b", "c"}, k=2) == 1.0


class TestAveragePrecision:
    def test_two_relevant_spread(self) -> None:
        # retrieved=[a,b,c,d,e], relevant={a,c}
        # hit at 1: P@1 = 1/1 = 1.0
        # hit at 3: P@3 = 2/3
        # AP = (1.0 + 2/3) / min(2,5) = (5/3) / 2 = 5/6
        result = average_precision_at_k(
            ["a", "b", "c", "d", "e"], {"a", "c"}, k=5
        )
        assert result == pytest.approx((1.0 + 2 / 3) / 2)

    def test_single_relevant_at_end(self) -> None:
        # retrieved=[a,b,c,d,e], relevant={e}
        # hit at 5: P@5 = 1/5 = 0.2
        # AP = 0.2 / min(1,5) = 0.2
        assert average_precision_at_k(
            ["a", "b", "c", "d", "e"], {"e"}, k=5
        ) == pytest.approx(0.2)

    def test_no_hit(self) -> None:
        assert average_precision_at_k(["a", "b"], {"x"}, k=5) == 0.0

    def test_empty_relevant(self) -> None:
        assert average_precision_at_k(["a"], set(), k=5) == 0.0

    def test_ideal(self) -> None:
        # all hits first → AP = 1.0
        assert average_precision_at_k(
            ["a", "b", "c"], {"a", "b"}, k=5
        ) == pytest.approx(1.0)

    def test_relevant_beyond_k_not_counted(self) -> None:
        # relevant at position 6, k=5 → no hit in top-k → AP = 0.0
        assert average_precision_at_k(
            ["a", "b", "c", "d", "e", "f"], {"f"}, k=5
        ) == 0.0


class TestMAP:
    def test_mean_of_two_queries(self) -> None:
        # q1: AP = 1.0 (hit at pos 1)
        # q2: AP = 0.5 (hit at pos 2 → P@2 = 1/2 = 0.5, /min(1,5)=0.5)
        # MAP = (1.0 + 0.5) / 2 = 0.75
        queries = [(["a"], {"a"}), (["x", "b"], {"b"})]
        assert map_at_k(queries, k=5) == pytest.approx(0.75)

    def test_empty(self) -> None:
        assert map_at_k([], k=5) == 0.0


class TestComputeAll:
    def test_returns_all_metric_keys(self) -> None:
        result = compute_all(["a", "b", "c"], {"a"}, k=5)
        for name in METRIC_NAMES:
            assert name in result, f"missing metric: {name}"

    def test_values_are_floats(self) -> None:
        result = compute_all(["a", "b", "c"], {"a"}, k=5)
        for v in result.values():
            assert isinstance(v, float)

    def test_consistent_with_individual_functions(self) -> None:
        retrieved = ["a", "b", "c", "d", "e"]
        relevant = {"a", "c"}
        k = 5
        result = compute_all(retrieved, relevant, k=k)
        assert result[f"recall@{k}"] == recall_at_k(retrieved, relevant, k)
        assert result[f"precision@{k}"] == precision_at_k(retrieved, relevant, k)
        assert result[f"hit_rate@{k}"] == hit_rate_at_k(retrieved, relevant, k)
        assert result["mrr"] == mrr(retrieved, relevant)
        assert result[f"ndcg@{k}"] == ndcg_at_k(retrieved, relevant, k)
        assert result[f"map@{k}"] == average_precision_at_k(retrieved, relevant, k)

    def test_empty_relevant_returns_zeros(self) -> None:
        result = compute_all(["a", "b"], set(), k=5)
        assert all(v == 0.0 for v in result.values())
