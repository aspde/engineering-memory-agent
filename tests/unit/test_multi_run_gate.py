"""Unit tests for tests/eval/multi_run_gate.py — the multi-run mean ± CI gate.

Pure-function tests over fabricated report data — no LLM, no subprocess.  The
``--n-runs`` subprocess path is exercised by the CI workflow; these tests pin
the statistics (aggregate), the gate decision (gate), the tolerance behaviour,
and report-shape parsing (report_suites / report_judge_mode / the aggregate
report payload).
"""

from __future__ import annotations

import math

import pytest

from tests.eval.multi_run_gate import (
    DEFAULT_TOLERANCE,
    T_VALUES,
    MetricAggregate,
    aggregate,
    build_aggregate_report,
    gate,
    report_judge_mode,
    report_suites,
    t_value,
)


def _report(results: list[dict]) -> dict:
    return {"generated_at": "2026-08-11T00:00:00+00:00", "results": results}


def _suite(name: str, overall: dict, judge: str = "deterministic") -> dict:
    return {"suite": name, "judge": judge, "overall": overall}


def _baseline(overall: dict, judge_mode: str | None = None) -> dict:
    payload: dict = {"schema_version": 1, "overall": overall}
    if judge_mode is not None:
        payload["environment"] = {"judge_mode": judge_mode}
    return payload


class TestAggregate:
    def test_mean_stdev_and_ci_lower_hand_computed(self) -> None:
        # [0.9, 0.9, 0.95] → mean 0.9167, n=3, t(2)=4.303, stdev = sqrt(0.0016667/2).
        reports = [
            _report([_suite("answer", {"fact_coverage": 0.9})]),
            _report([_suite("answer", {"fact_coverage": 0.9})]),
            _report([_suite("answer", {"fact_coverage": 0.95})]),
        ]

        agg = aggregate(reports)["answer"]["fact_coverage"]

        assert agg.mean == pytest.approx(0.9166666666666666)
        assert agg.n == 3
        assert agg.stdev == pytest.approx(math.sqrt(0.0016666666666666666 / 2))
        # ci95_lower = mean - 4.303 * stdev / sqrt(3) = 0.9167 - 0.0717 ≈ 0.8450.
        assert agg.ci95_lower == pytest.approx(0.84495)

    def test_n2_wide_ci_lower(self) -> None:
        # Two runs straddling a value: n=2 → t=12.706 gives a very wide CI.
        # [0.66, 0.68] → mean 0.67, stdev = 0.02/sqrt(2), sem = 0.01.
        reports = [
            _report([_suite("answer", {"fact_coverage": 0.66})]),
            _report([_suite("answer", {"fact_coverage": 0.68})]),
        ]

        agg = aggregate(reports)["answer"]["fact_coverage"]

        assert agg.n == 2
        assert agg.mean == pytest.approx(0.67)
        assert agg.ci95_lower == pytest.approx(0.67 - T_VALUES[2] * 0.01)

    def test_single_run_ci_lower_is_mean(self) -> None:
        # n=1 → no variance estimate: stdev=None, ci95_lower = mean.
        agg = aggregate(
            [_report([_suite("answer", {"fact_coverage": 0.72})])]
        )["answer"]["fact_coverage"]

        assert agg.n == 1
        assert agg.stdev is None
        assert agg.ci95_lower == pytest.approx(0.72)
        assert agg.mean == pytest.approx(0.72)

    def test_aggregates_every_suite_and_metric(self) -> None:
        reports = [
            _report(
                [
                    _suite("tool_selection", {"tool_accuracy": 0.9, "expected_recall": 0.8}),
                    _suite("extraction", {"entity_f1": 0.7}),
                ]
            ),
            _report(
                [
                    _suite("tool_selection", {"tool_accuracy": 1.0, "expected_recall": 0.9}),
                    _suite("extraction", {"entity_f1": 0.9}),
                ]
            ),
        ]

        agg = aggregate(reports)

        assert set(agg["tool_selection"]) == {"tool_accuracy", "expected_recall"}
        assert set(agg["extraction"]) == {"entity_f1"}
        assert agg["tool_selection"]["tool_accuracy"].mean == pytest.approx(0.95)
        # [0.7, 0.9] → mean 0.8, stdev = 0.2/sqrt(2), sem = 0.1.
        assert agg["extraction"]["entity_f1"].ci95_lower == pytest.approx(
            0.8 - T_VALUES[2] * 0.1
        )

    def test_metric_missing_from_some_runs_uses_actual_n(self) -> None:
        reports = [
            _report([_suite("answer", {"fact_coverage": 0.8, "groundedness": 0.9})]),
            _report([_suite("answer", {"fact_coverage": 0.9})]),
        ]

        agg = aggregate(reports)["answer"]

        assert agg["fact_coverage"].n == 2
        assert agg["groundedness"].n == 1
        assert agg["groundedness"].ci95_lower == agg["groundedness"].mean == pytest.approx(0.9)

    def test_baseline_layout_reports_aggregate(self) -> None:
        reports = [
            _baseline({"answer": {"fact_coverage": 0.95}}),
            _baseline({"answer": {"fact_coverage": 0.97}}),
        ]

        agg = aggregate(reports)["answer"]["fact_coverage"]

        assert agg.n == 2
        assert agg.mean == pytest.approx(0.96)


class _Agg:
    """Fabricate a MetricAggregate without going through aggregate()."""

    @staticmethod
    def one(*, mean: float, ci95_lower: float, n: int) -> MetricAggregate:
        return MetricAggregate(mean=mean, stdev=None, ci95_lower=ci95_lower, n=n)


class TestGate:
    def test_ci_lower_above_floor_passes(self) -> None:
        aggs = {"answer": {"groundedness": _Agg.one(mean=0.95, ci95_lower=0.90, n=3)}}

        assert gate(aggs, {"groundedness": 0.90}, 0.03) == []

    def test_ci_lower_below_floor_fails_with_metadata(self) -> None:
        aggs = {"answer": {"groundedness": _Agg.one(mean=0.90, ci95_lower=0.80, n=3)}}

        failures = gate(aggs, {"groundedness": 0.90}, 0.03)  # floor 0.87

        assert len(failures) == 1
        f = failures[0]
        assert f.suite == "answer"
        assert f.metric == "groundedness"
        assert f.threshold == 0.90
        assert f.ci95_lower == pytest.approx(0.80)
        assert f.mean == pytest.approx(0.90)
        assert f.n == 3

    def test_tolerance_absorbs_small_fluctuation(self) -> None:
        # mean 0.70 vs threshold 0.68: the single run would fail, but the CI
        # lower bound 0.66 sits above 0.68 - 0.03 = 0.65 → pass.
        aggs = {"tool_selection": {"tool_accuracy": _Agg.one(mean=0.70, ci95_lower=0.66, n=3)}}

        assert gate(aggs, {"tool_accuracy": 0.68}, 0.03) == []

    def test_without_tolerance_the_same_ci_fails(self) -> None:
        aggs = {"tool_selection": {"tool_accuracy": _Agg.one(mean=0.70, ci95_lower=0.66, n=3)}}

        failures = gate(aggs, {"tool_accuracy": 0.68}, 0.0)  # floor 0.68

        assert len(failures) == 1

    def test_exact_floor_passes(self) -> None:
        aggs = {"answer": {"fact_coverage": _Agg.one(mean=0.90, ci95_lower=0.87, n=3)}}

        assert gate(aggs, {"fact_coverage": 0.90}, 0.03) == []

    def test_n1_degenerates_to_single_value_gate(self) -> None:
        # n=1 → ci95_lower == mean == the observed value; the rule degrades to
        # value < threshold - tolerance (single-run backwards compatibility).
        passing = {"answer": {"groundedness": _Agg.one(mean=0.66, ci95_lower=0.66, n=1)}}
        assert gate(passing, {"groundedness": 0.68}, 0.03) == []  # 0.66 > 0.65

        failing = {"answer": {"groundedness": _Agg.one(mean=0.64, ci95_lower=0.64, n=1)}}
        assert len(gate(failing, {"groundedness": 0.68}, 0.03)) == 1  # 0.64 < 0.65

    def test_gates_each_suite_metric_on_its_own_threshold(self) -> None:
        aggs = {
            "tool_selection": {"tool_accuracy": _Agg.one(mean=0.9, ci95_lower=0.8, n=3)},
            "extraction": {
                "entity_f1": _Agg.one(mean=0.9, ci95_lower=0.8, n=3),
                "relation_f1": _Agg.one(mean=0.2, ci95_lower=0.1, n=3),
            },
        }

        failures = gate(
            aggs, {"tool_accuracy": 0.68, "entity_f1": 0.68, "relation_f1": 0.28}, 0.03
        )

        assert {(f.suite, f.metric) for f in failures} == {("extraction", "relation_f1")}

    def test_metric_without_threshold_is_not_gated(self) -> None:
        aggs = {"answer": {"hallucination_rate": _Agg.one(mean=0.1, ci95_lower=0.2, n=3)}}

        assert gate(aggs, {}, 0.03) == []

    def test_threshold_metric_absent_from_aggregates_never_fails(self) -> None:
        # Mirrors run_llm_eval: a --min-* flag for a metric no run produces
        # warns in the CLI but does not fail the gate.
        aggs = {"answer": {"fact_coverage": _Agg.one(mean=0.9, ci95_lower=0.88, n=3)}}

        assert gate(aggs, {"context_recall": 0.95}, 0.03) == []

    def test_empty_aggregates_pass(self) -> None:
        assert gate({}, {"tool_accuracy": 0.68}, 0.03) == []


class TestTValues:
    def test_table_matches_known_95_percent_two_tailed_values(self) -> None:
        assert T_VALUES[2] == pytest.approx(12.706)
        assert T_VALUES[3] == pytest.approx(4.303)
        assert T_VALUES[4] == pytest.approx(3.182)
        assert T_VALUES[5] == pytest.approx(2.776)
        assert T_VALUES[10] == pytest.approx(2.262)

    def test_n11_plus_uses_conservative_floor(self) -> None:
        assert t_value(11) == pytest.approx(2.228)
        assert t_value(50) == pytest.approx(2.228)

    def test_n_below_2_is_an_error(self) -> None:
        with pytest.raises(ValueError):
            t_value(1)

    def test_default_tolerance(self) -> None:
        assert DEFAULT_TOLERANCE == pytest.approx(0.03)


class TestReportSuites:
    def test_report_json_layout(self) -> None:
        report = _report(
            [
                _suite("tool_selection", {"tool_accuracy": 0.9}),
                _suite("extraction", {"entity_f1": 0.7}),
            ]
        )

        assert report_suites(report) == {
            "tool_selection": {"tool_accuracy": 0.9},
            "extraction": {"entity_f1": 0.7},
        }

    def test_baseline_layout(self) -> None:
        report = _baseline(
            {"answer": {"fact_coverage": 0.958, "groundedness": 1.0}},
            judge_mode="deterministic",
        )

        assert report_suites(report) == {
            "answer": {"fact_coverage": 0.958, "groundedness": 1.0}
        }

    def test_unrecognized_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="neither 'results' nor 'overall'"):
            report_suites({"foo": "bar"})


class TestReportJudgeMode:
    def test_mixed_report_reports_llm_not_tool_selection(self) -> None:
        report = _report(
            [
                _suite("tool_selection", {}, judge="deterministic"),
                _suite("answer", {}, judge="llm"),
                _suite("e2e", {}, judge="llm"),
            ]
        )

        assert report_judge_mode(report) == "llm"

    def test_all_deterministic_report(self) -> None:
        report = _report(
            [
                _suite("tool_selection", {}, judge="deterministic"),
                _suite("answer", {}, judge="deterministic"),
            ]
        )

        assert report_judge_mode(report) == "deterministic"

    def test_tool_selection_only_returns_none(self) -> None:
        report = _report([_suite("tool_selection", {}, judge="deterministic")])

        assert report_judge_mode(report) is None

    def test_baseline_layout_reads_environment_judge_mode(self) -> None:
        assert report_judge_mode(_baseline({}, judge_mode="deterministic")) == "deterministic"
        assert report_judge_mode(_baseline({})) is None


class TestAggregateReport:
    def test_payload_carries_gate_verdict_and_aggregates(self) -> None:
        # tool_selection never touches the judge, so include a judge-using
        # suite (answer) for the judge mode to be recorded as deterministic.
        reports = [
            _report(
                [
                    _suite("tool_selection", {"tool_accuracy": 0.9}),
                    _suite("answer", {"fact_coverage": 1.0}),
                ]
            ),
            _report(
                [
                    _suite("tool_selection", {"tool_accuracy": 0.7}),
                    _suite("answer", {"fact_coverage": 1.0}),
                ]
            ),
        ]
        aggregates = aggregate(reports)
        failures = gate(aggregates, {"tool_accuracy": 0.9}, 0.0)  # ci_lower 0.7 - 12.706*0.1 < 0.9

        payload = build_aggregate_report(
            reports=reports,
            aggregates=aggregates,
            thresholds={"tool_accuracy": 0.9},
            tolerance=0.0,
            failures=failures,
            sources=["a.json", "b.json"],
        )

        assert payload["n_runs"] == 2
        assert payload["judge_mode"] == "deterministic"
        assert payload["gate"]["passed"] is False
        assert payload["gate"]["failures"][0]["suite"] == "tool_selection"
        agg_payload = payload["aggregates"]["tool_selection"]["tool_accuracy"]
        assert set(agg_payload) == {"mean", "stdev", "ci95_lower", "n"}
        assert agg_payload["mean"] == pytest.approx(0.8)
        assert payload["reports"][0]["path"] == "a.json"

    def test_mixed_or_unknown_judge_modes_labeled_mixed(self) -> None:
        reports = [
            _report([_suite("answer", {"fact_coverage": 0.9}, judge="llm")]),
            _baseline({"answer": {"fact_coverage": 0.9}}),  # no judge_mode recorded
        ]
        payload = build_aggregate_report(
            reports=reports,
            aggregates=aggregate(reports),
            thresholds={},
            tolerance=0.03,
            failures=[],
            sources=["a.json", "b.json"],
        )

        assert payload["judge_mode"] == "mixed"
