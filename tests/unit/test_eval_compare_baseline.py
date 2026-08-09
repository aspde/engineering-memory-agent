"""Unit tests for tests/eval/compare_baseline.py — the baseline diff gate.

The compare logic is a pure function over two metric dicts, so it is tested
with fabricated data — no DB, no retrieval, no LLM.
"""

from __future__ import annotations

from tests.eval.compare_baseline import _report_judge_mode, compare


def _report(results: list[dict]) -> dict:
    return {"generated_at": "2026-08-09T00:00:00+00:00", "results": results}


def _suite(name: str, judge: str, overall: dict) -> dict:
    return {"suite": name, "judge": judge, "overall": overall, "errors": [], "judge_errors": []}


class TestReportJudgeMode:
    def test_mixed_report_detects_llm_not_tool_selection(self) -> None:
        """tool_selection is always deterministic, so a mixed run must be
        reported by a judge-using suite — comparing a semantic run against a
        deterministic baseline would otherwise slip past the guard."""
        report = _report(
            [
                _suite("tool_selection", "deterministic", {}),
                _suite("answer", "llm", {}),
                _suite("e2e", "llm", {}),
            ]
        )

        assert _report_judge_mode(report) == "llm"

    def test_all_deterministic_report(self) -> None:
        report = _report(
            [
                _suite("tool_selection", "deterministic", {}),
                _suite("answer", "deterministic", {}),
            ]
        )

        assert _report_judge_mode(report) == "deterministic"

    def test_tool_selection_only_returns_none(self) -> None:
        report = _report([_suite("tool_selection", "deterministic", {})])

        assert _report_judge_mode(report) is None


class TestCompare:
    def test_identical_reports_produce_zero_deltas(self) -> None:
        overall = {"fact_coverage": 0.958, "groundedness": 1.0}
        report = _report([_suite("answer", "deterministic", overall)])
        baseline = {"overall": {"answer": overall}}

        deltas = compare(report, baseline)

        assert deltas["answer"] == {"fact_coverage": 0.0, "groundedness": 0.0}

    def test_improvement_reports_positive_delta(self) -> None:
        report = _report(
            [_suite("answer", "deterministic", {"fact_coverage": 0.98, "groundedness": 1.0})]
        )
        baseline = {"overall": {"answer": {"fact_coverage": 0.958, "groundedness": 1.0}}}

        deltas = compare(report, baseline)

        assert deltas["answer"]["fact_coverage"] == 0.022
        assert deltas["answer"]["groundedness"] == 0.0

    def test_regression_reports_negative_delta(self) -> None:
        report = _report(
            [_suite("answer", "deterministic", {"fact_coverage": 0.80, "groundedness": 0.70})]
        )
        baseline = {"overall": {"answer": {"fact_coverage": 0.958, "groundedness": 1.0}}}

        deltas = compare(report, baseline)

        assert deltas["answer"]["fact_coverage"] == -0.158
        assert deltas["answer"]["groundedness"] == -0.3

    def test_report_metric_absent_from_baseline_is_skipped(self) -> None:
        # A metric the fresh report produced but the baseline never measured
        # cannot be diffed — it is omitted, not treated as a drop.
        report = _report(
            [
                _suite(
                    "answer",
                    "deterministic",
                    {"fact_coverage": 0.958, "new_metric": 0.5},
                )
            ]
        )
        baseline = {"overall": {"answer": {"fact_coverage": 0.958}}}

        deltas = compare(report, baseline)

        assert deltas["answer"] == {"fact_coverage": 0.0}

    def test_suite_missing_from_report_is_marked(self) -> None:
        report = _report([_suite("answer", "deterministic", {"fact_coverage": 0.958})])
        baseline = {
            "overall": {
                "answer": {"fact_coverage": 0.958},
                "e2e": {"context_recall": 1.0},
            }
        }

        deltas = compare(report, baseline)

        assert "__suite_missing__" in deltas["e2e"]
        assert deltas["answer"] == {"fact_coverage": 0.0}

    def test_metric_absent_from_fresh_report_is_skipped(self) -> None:
        # A baseline metric the fresh run no longer produced is skipped, not
        # treated as a drop (the metric is simply not comparable).
        report = _report([_suite("answer", "deterministic", {"fact_coverage": 0.958})])
        baseline = {
            "overall": {"answer": {"fact_coverage": 0.958, "groundedness": 1.0}}
        }

        deltas = compare(report, baseline)

        assert deltas["answer"] == {"fact_coverage": 0.0}
