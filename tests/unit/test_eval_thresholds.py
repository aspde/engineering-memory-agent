"""Unit tests for tests/eval/thresholds.py — the CI regression gate.

The gate is a pure function over an ``overall`` metric dict, so it is tested
with fabricated data — no DB, no retrieval, no LLM.
"""

from __future__ import annotations

from tests.eval.thresholds import check_thresholds, print_threshold_failures


class TestCheckThresholds:
    def test_passes_when_all_metrics_meet_minimums(self) -> None:
        overall = {"recall@5": 0.96, "mrr": 0.91}
        result = check_thresholds(overall, {"recall@5": 0.95, "mrr": 0.90})

        assert result.passed is True
        assert result.failures == []

    def test_boundary_actual_equals_minimum_passes(self) -> None:
        # A metric exactly at the threshold satisfies the gate.
        result = check_thresholds({"recall@5": 0.95}, {"recall@5": 0.95})

        assert result.passed is True
        assert result.failures == []

    def test_fails_when_a_metric_falls_below(self) -> None:
        overall = {"recall@5": 0.94, "mrr": 0.91}
        result = check_thresholds(overall, {"recall@5": 0.95, "mrr": 0.90})

        assert result.passed is False
        assert len(result.failures) == 1
        failure = result.failures[0]
        assert failure["metric"] == "recall@5"
        assert failure["min"] == 0.95
        assert failure["actual"] == 0.94

    def test_reports_every_failing_metric(self) -> None:
        overall = {"recall@5": 0.93, "mrr": 0.85}
        result = check_thresholds(overall, {"recall@5": 0.95, "mrr": 0.90})

        assert result.passed is False
        assert {f["metric"] for f in result.failures} == {"recall@5", "mrr"}

    def test_missing_metric_counts_as_failure(self) -> None:
        # A metric the gate asks for but the run never produced (absent from
        # `overall`) is treated as 0.0 — you cannot gate on what you cannot
        # measure, and 0.0 can never satisfy a positive minimum.
        result = check_thresholds({"recall@5": 0.96}, {"mrr": 0.90})

        assert result.passed is False
        assert result.failures[0]["metric"] == "mrr"
        assert result.failures[0]["actual"] == 0.0

    def test_no_thresholds_always_passes(self) -> None:
        # Empty gate = the existing no-gate default behavior.
        result = check_thresholds({"recall@5": 0.10}, {})

        assert result.passed is True
        assert result.failures == []

    def test_empty_overall_with_thresholds_fails(self) -> None:
        result = check_thresholds({}, {"recall@5": 0.95})

        assert result.passed is False
        assert result.failures[0]["actual"] == 0.0

    def test_exact_threshold_boundary_with_multiple_metrics(self) -> None:
        overall = {"recall@5": 0.95, "mrr": 0.89}
        result = check_thresholds(overall, {"recall@5": 0.95, "mrr": 0.90})

        assert result.passed is False
        assert [f["metric"] for f in result.failures] == ["mrr"]


class TestPrintThresholdFailures:
    def test_prints_nothing_when_passed(self, capsys) -> None:
        print_threshold_failures(check_thresholds({"recall@5": 0.96}, {"recall@5": 0.95}))

        captured = capsys.readouterr()
        assert captured.err == ""

    def test_prints_each_failure_with_actual_and_minimum(self, capsys) -> None:
        result = check_thresholds(
            {"recall@5": 0.93, "mrr": 0.85}, {"recall@5": 0.95, "mrr": 0.90}
        )
        print_threshold_failures(result)

        err = capsys.readouterr().err
        assert "recall@5: 0.930 < required 0.950" in err
        assert "mrr: 0.850 < required 0.900" in err
