"""Unit tests for tests/eval/core.py — the shared eval skeleton.

The shared helpers (aggregate / finish / zero_judge_keys / the judge-based
JSON layout / markdown table helpers) are exercised heavily through the
individual runners and reports; these tests pin the core contract directly so
the shared module has its own coverage rather than relying on its consumers.
"""

from __future__ import annotations

import json

import pytest

from tests.eval.core import (
    ANSWER_JUDGE_METRIC_KEYS,
    EvalResult,
    aggregate,
    category_table,
    finish,
    fmt,
    overall_table,
    result_to_json_dict,
    to_json,
    zero_judge_keys,
)


class TestAggregate:
    def test_mean_across_rows(self) -> None:
        rows = [{"a": 1.0, "b": 0.5}, {"a": 0.0, "b": 0.5}]
        out = aggregate(rows, ["a", "b"])
        assert out["a"] == 0.5
        assert out["b"] == 0.5

    def test_missing_key_counts_zero(self) -> None:
        rows = [{"a": 1.0}]
        assert aggregate(rows, ["a", "b"]) == {"a": 1.0, "b": 0.0}

    def test_empty_rows_are_all_zero(self) -> None:
        assert aggregate([], ["a", "b"]) == {"a": 0.0, "b": 0.0}


class TestFinish:
    def test_rolls_up_overall_and_categories(self) -> None:
        rows = [
            {"category": "c1", "m": 1.0},
            {"category": "c1", "m": 0.0},
            {"category": "c2", "m": 1.0},
        ]
        r = finish("suite", "deterministic", rows, [], [], ["m"], ["c1", "c2"])
        assert r.suite == "suite"
        assert r.n_items == 3
        assert r.overall["m"] == pytest.approx(2 / 3)
        assert r.by_category["c1"]["m"] == pytest.approx(0.5)
        assert r.by_category["c2"]["m"] == pytest.approx(1.0)

    def test_empty_category_buckets_present_as_zeros(self) -> None:
        rows = [{"category": "c1", "m": 1.0}]
        r = finish("s", "deterministic", rows, [], [], ["m"], ["c1", "c2", "c3"])
        assert set(r.by_category) == {"c1", "c2", "c3"}
        assert r.by_category["c3"]["m"] == 0.0

    def test_errors_and_judge_errors_passthrough(self) -> None:
        errs = [{"id": "a", "error": "boom"}]
        jerrs = [{"id": "a", "error": "judge down"}]
        r = finish("s", "llm", [], errs, jerrs, ["m"], [])
        assert r.errors == errs
        assert r.judge_errors == jerrs


class TestZeroJudgeKeys:
    def test_zeroes_judge_keys_in_place(self) -> None:
        row = {
            "fact_coverage": 1.0,
            "groundedness": 1.0,
            "hallucination_rate": 1.0,
        }
        zero_judge_keys(row)
        for k in ANSWER_JUDGE_METRIC_KEYS:
            assert row[k] == 0.0

    def test_leaves_det_channel_untouched(self) -> None:
        row = {"fact_coverage": 1.0, "det_fact_coverage": 1.0, "citation_rate": 1.0}
        zero_judge_keys(row)
        assert row["det_fact_coverage"] == 1.0
        assert row["citation_rate"] == 1.0


def _result() -> EvalResult:
    return EvalResult(
        suite="answer",
        judge="deterministic",
        per_query=[{"id": "a1", "category": "factual", "fact_coverage": 1.0}],
        overall={"fact_coverage": 1.0},
        by_category={"factual": {"fact_coverage": 1.0}},
        metric_keys=("fact_coverage",),
        n_items=1,
    )


class TestJsonLayout:
    def test_result_to_json_dict_layout(self) -> None:
        d = result_to_json_dict(_result())
        assert d["suite"] == "answer"
        assert d["metric_keys"] == ["fact_coverage"]
        assert d["n_judge_errors"] == 0
        assert "per_query" in d

    def test_to_json_round_trips(self) -> None:
        payload = json.loads(to_json([_result()]))
        assert "generated_at" in payload
        assert payload["results"][0]["overall"]["fact_coverage"] == 1.0


class TestMarkdownHelpers:
    def test_fmt(self) -> None:
        assert fmt(0.5) == "0.500"
        assert fmt(None) == "—"

    def test_overall_table_has_title_and_metric(self) -> None:
        md = overall_table([_result()], lambda s: "答案")
        assert "### 答案" in md
        assert "| **overall** | 1.000 |" in md

    def test_category_table(self) -> None:
        md = category_table(_result(), headline_exclude=("answer_len",))
        assert "| factual |" in md

    def test_category_table_empty(self) -> None:
        r = EvalResult(suite="s", by_category={}, metric_keys=("m",))
        assert category_table(r) == "_(no category data)_"
