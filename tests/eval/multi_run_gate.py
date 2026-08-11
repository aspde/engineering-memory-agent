"""Multi-run mean ± CI gate for the LLM behavior eval.

``run_llm_eval --min-*`` gates on a *single* run's headline metrics.  LLM
output is sampled (even ``--judge deterministic`` leaves the model's
temperature on), so one unlucky draw can swing a metric far beyond run-to-run
noise — relation_recall moved 0.531 → 0.344 between two runs of the same
code — and a single-run floor that sits inside that swing flakes CI for no
real regression.

This module replaces the single-run gate with a *multi-run* gate:

1. Run the eval N times (``--n-runs N``) or point at N existing report JSONs
   (``--reports a.json,b.json``).
2. Aggregate each (suite, metric) across the N runs → mean, sample stdev, and
   the lower bound of the 95% confidence interval of the mean
   (``mean - t(n-1) * stdev / sqrt(n)``).
3. Gate on that **CI lower bound**: a metric fails only when
   ``ci95_lower < threshold - tolerance`` (exit 2).

A single unlucky low value barely moves the CI lower bound; a real quality
drop drags the whole distribution down and does.  ``--tolerance`` (default
0.03 — deliberately wider than ``compare_baseline``'s 0.01) absorbs residual
bias: a small-sample CI is wide and the model is stochastic, so the gate
trips on a sustained drop, not a noise blip.  With one run (n=1) there is no
variance estimate; ``ci95_lower = mean`` and the rule degenerates to the old
single-run ``value < threshold - tolerance``, keeping backwards compatibility.

No scipy: the t values for n=2..10 are a hardcoded table (95% two-tailed),
with a conservative 2.228 floor for n>=11.

Examples:
    # Aggregate two committed reports (n=2) and gate on the CI lower bound
    python -m tests.eval.multi_run_gate --reports \
        tests/eval/reports/llm-eval-report.json,\
        tests/eval/reports/llm-eval-semantic-baseline.json \
        --min-tool-accuracy 0.68 --min-entity-f1 0.68 --min-relation-f1 0.28 \
        --min-fact-coverage 0.90 --min-groundedness 0.95 \
        --min-citation-rate 0.95

    # CI gate: run the eval 3 times, aggregate, gate on the CI lower bound
    python -m tests.eval.multi_run_gate --n-runs 3 --judge deterministic \
        --suite tool_selection,extraction,answer --report-dir llm-eval-reports \
        --min-tool-accuracy 0.68 --min-groundedness 0.95

Exit codes: 0 = gate passed; 2 = at least one metric's CI lower bound fell
below ``threshold - tolerance``.

Mixing judge modes is flagged, not refused: a deterministic (substring) and an
llm (semantic) report score by different rules, so aggregating them is not an
apples-to-apples regression signal — the tool prints a warning (see
``compare_baseline`` for the same caution) but still computes and labels the
aggregate.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# 95% two-tailed Student's t for df = n-1 (hardcoded — no scipy dependency).
T_VALUES: dict[int, float] = {
    2: 12.706,
    3: 4.303,
    4: 3.182,
    5: 2.776,
    6: 2.571,
    7: 2.447,
    8: 2.365,
    9: 2.306,
    10: 2.262,
}
# df = n-1 >= 10: t still sits just above the normal 1.96; 2.228 (the n=11
# value) is conservative and only slightly widens the interval.
T_FLOOR: float = 2.228

DEFAULT_TOLERANCE = 0.03
DEFAULT_SUITE = "tool_selection,extraction,answer"

# The --min-* metrics multi_run_gate can gate on (mirrors run_llm_eval).
_MIN_FLAGS: tuple[tuple[str, str], ...] = (
    ("--min-tool-accuracy", "tool_accuracy"),
    ("--min-expected-recall", "expected_recall"),
    ("--min-entity-f1", "entity_f1"),
    ("--min-relation-f1", "relation_f1"),
    ("--min-fact-coverage", "fact_coverage"),
    ("--min-groundedness", "groundedness"),
    ("--min-citation-rate", "citation_rate"),
    ("--min-context-recall", "context_recall"),
)


@dataclass(frozen=True)
class MetricAggregate:
    """Aggregate of one (suite, metric) across N runs.

    ``ci95_lower`` is the lower bound of the 95% confidence interval of the
    mean.  ``n`` is the number of runs that actually produced the metric (may
    be < the run count when a metric is missing from some reports).
    """

    mean: float
    stdev: float | None
    ci95_lower: float
    n: int


@dataclass(frozen=True)
class GateFailure:
    """One (suite, metric) whose CI lower bound fell below the floor."""

    suite: str
    metric: str
    threshold: float
    ci95_lower: float
    mean: float
    n: int


def t_value(n: int) -> float:
    """95% two-tailed Student's t for df = n-1; T_FLOOR for n >= 11."""
    if n < 2:
        raise ValueError(f"t_value needs n >= 2, got {n}")
    return T_VALUES.get(n, T_FLOOR)


def report_suites(report: dict) -> dict[str, dict[str, float]]:
    """Flatten one report JSON into ``{suite: {metric: value}}``.

    Accepts both the ``run_llm_eval --report-json`` layout (``{"results":
    [{"suite": ..., "overall": {...}}, ...]}``) and the committed-baseline
    layout (``{"overall": {suite: {metric: value}}}``).
    """
    if "results" in report:
        return {r["suite"]: dict(r["overall"]) for r in report["results"]}
    if "overall" in report:
        return {suite: dict(metrics) for suite, metrics in report["overall"].items()}
    raise ValueError("report has neither 'results' nor 'overall'")


def report_judge_mode(report: dict) -> str | None:
    """The judge mode of a run, or None when not recorded.

    Mirrors ``compare_baseline._report_judge_mode``: ``tool_selection`` never
    touches the judge (it is always deterministic), so a mixed run is reported
    by the first judge-using suite.  Baseline-layout reports record the mode in
    ``environment.judge_mode`` (may be absent).
    """
    if "results" in report:
        for r in report.get("results", []):
            if r.get("suite") != "tool_selection" and r.get("judge"):
                return str(r["judge"])
        return None
    return report.get("environment", {}).get("judge_mode")


def load_report(path: str | Path) -> dict:
    """Read one report JSON from disk."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _aggregate_values(values: Sequence[float]) -> MetricAggregate:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        # No variance estimate — ci95_lower is the mean itself, marked with
        # stdev=None so the table reports the honest "can't estimate" case.
        return MetricAggregate(mean=mean, stdev=None, ci95_lower=mean, n=n)
    stdev = statistics.stdev(values)
    margin = t_value(n) * stdev / math.sqrt(n)
    return MetricAggregate(mean=mean, stdev=stdev, ci95_lower=mean - margin, n=n)


def aggregate(reports: Sequence[dict]) -> dict[str, dict[str, MetricAggregate]]:
    """Aggregate N report JSONs into per-(suite, metric) mean ± CI aggregates.

    Returns ``{suite: {metric: MetricAggregate}}``.  ``MetricAggregate.n`` is
    the number of runs that produced that metric; ``ci95_lower`` is the lower
    bound of the 95% confidence interval of the mean across runs.
    """
    collected: dict[str, dict[str, list[float]]] = {}
    for report in reports:
        for suite, metrics in report_suites(report).items():
            for metric, value in metrics.items():
                collected.setdefault(suite, {}).setdefault(metric, []).append(
                    float(value)
                )
    return {
        suite: {metric: _aggregate_values(values) for metric, values in metrics.items()}
        for suite, metrics in collected.items()
    }


def gate(
    aggregates: dict[str, dict[str, MetricAggregate]],
    thresholds: dict[str, float],
    tolerance: float,
) -> list[GateFailure]:
    """Which (suite, metric) fail: ``ci95_lower < threshold - tolerance``.

    Thresholds span suites (tool_accuracy lives only under tool_selection,
    etc.), so each aggregate's metric is gated on its own threshold when one
    exists.  A threshold metric produced by *no* aggregate is not a failure —
    the CLI warns instead, mirroring run_llm_eval's "no suite reports that
    metric" behaviour (you cannot gate on what you cannot measure).
    """
    failures: list[GateFailure] = []
    for suite, metrics in aggregates.items():
        for metric, agg in metrics.items():
            minimum = thresholds.get(metric)
            if minimum is None:
                continue
            floor = minimum - tolerance
            if agg.ci95_lower < floor:
                failures.append(
                    GateFailure(
                        suite=suite,
                        metric=metric,
                        threshold=minimum,
                        ci95_lower=agg.ci95_lower,
                        mean=agg.mean,
                        n=agg.n,
                    )
                )
    return failures


def build_aggregate_report(
    *,
    reports: Sequence[dict],
    aggregates: dict[str, dict[str, MetricAggregate]],
    thresholds: dict[str, float],
    tolerance: float,
    failures: Sequence[GateFailure],
    sources: Sequence[str],
) -> dict[str, Any]:
    """The artifact payload written by --report-dir (CI uploads this)."""
    modes = [report_judge_mode(r) for r in reports]
    labelled = {m if m else "unknown" for m in modes}
    judge_mode = "mixed" if len(labelled) > 1 else next(iter(labelled))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool": "tests.eval.multi_run_gate",
        "n_runs": len(reports),
        "judge_mode": judge_mode,
        "tolerance": tolerance,
        "thresholds": thresholds,
        "reports": [
            {"path": src, "judge": mode, "suites": report_suites(r)}
            for src, mode, r in zip(sources, modes, reports)
        ],
        "aggregates": {
            suite: {metric: asdict(agg) for metric, agg in metrics.items()}
            for suite, metrics in aggregates.items()
        },
        "gate": {
            "passed": not failures,
            "failures": [asdict(f) for f in failures],
        },
    }


def write_json_report(payload: dict[str, Any], path: Path) -> None:
    """Write *payload* as pretty JSON to *path*."""
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def print_aggregates(
    aggregates: dict[str, dict[str, MetricAggregate]],
    thresholds: dict[str, float],
    tolerance: float,
    n_runs: int,
    modes: Sequence[str | None],
) -> None:
    """Print the per-(suite, metric) mean ± CI table (stdout)."""
    labelled = ", ".join(sorted({m if m else "unknown" for m in modes}))
    gate_desc = (
        "no thresholds set"
        if not thresholds
        else ", ".join(f"{k}={v:.3f}" for k, v in sorted(thresholds.items()))
    )
    print(
        f"multi-run gate: {n_runs} run(s), judge modes: {labelled}, "
        f"tolerance={tolerance:.3f}, thresholds: {gate_desc}"
    )
    for suite in sorted(aggregates):
        print(f"[{suite}]")
        for metric in sorted(aggregates[suite]):
            agg = aggregates[suite][metric]
            minimum = thresholds.get(metric)
            stdev_s = "—" if agg.stdev is None else f"{agg.stdev:.4f}"
            n_s = f"{agg.n}/{n_runs}" if agg.n < n_runs else f"{agg.n}"
            head = (
                f"  {metric:22s} mean={agg.mean:.4f}  "
                f"ci95_lower={agg.ci95_lower:.4f}  stdev={stdev_s}  n={n_s}"
            )
            if minimum is None:
                print(f"{head}  (no gate)")
                continue
            floor = minimum - tolerance
            status = "PASS" if agg.ci95_lower >= floor else "FAIL"
            print(f"{head}  min={minimum:.3f}  {status}")


def _warn_mixed_judge_modes(modes: Sequence[str | None]) -> None:
    labelled = {m if m else "unknown" for m in modes}
    if len(labelled) <= 1:
        return
    print(
        f"⚠ reports mix judge modes ({', '.join(sorted(labelled))}): "
        "deterministic (substring) and llm (semantic) scoring are not "
        "comparable — the aggregated mean/CI below blends different grading "
        "rules and is NOT an apples-to-apples regression signal.  Aggregate "
        "same-mode reports (e.g. all --judge deterministic).",
        file=sys.stderr,
    )


def _warn_absent_thresholds(
    aggregates: dict[str, dict[str, MetricAggregate]],
    thresholds: dict[str, float],
) -> None:
    produced = {metric for metrics in aggregates.values() for metric in metrics}
    for metric, minimum in thresholds.items():
        if metric not in produced:
            print(
                f"⚠ --min-{metric} {minimum} ignored: no report produces that "
                "metric",
                file=sys.stderr,
            )


def _run_eval_runs(
    args: argparse.Namespace, report_dir: Path | None
) -> list[tuple[str, dict]]:
    """Run ``run_llm_eval`` ``--n-runs`` times; return ``[(path, report)]``.

    Each run writes its own report JSON (kept under *report_dir* when given,
    else a temporary directory).  A non-zero subprocess exit (execution
    errors / judge-channel failure — run_llm_eval exit 1) aborts the gate: a
    broken run is not a quality signal and must not feed the aggregate.
    """
    runs: list[tuple[str, dict]] = []
    tmp = tempfile.TemporaryDirectory() if report_dir is None else None
    base = report_dir or Path(tmp.name)  # type: ignore[arg-type]
    try:
        for i in range(1, args.n_runs + 1):
            json_path = base / f"llm-eval-run-{i}.json"
            print(f"▶ eval run {i}/{args.n_runs} …", file=sys.stderr)
            cmd = [
                sys.executable,
                "-m",
                "tests.eval.run_llm_eval",
                "--suite",
                args.suite,
                "--judge",
                args.judge,
                "--report-json",
                str(json_path),
            ]
            proc = subprocess.run(cmd, text=True, capture_output=True)
            if proc.stdout:
                print(proc.stdout, end="")
            if proc.stderr:
                print(proc.stderr, end="", file=sys.stderr)
            if proc.returncode != 0:
                raise SystemExit(proc.returncode)
            runs.append((str(json_path), load_report(json_path)))
        return runs
    finally:
        if tmp is not None:
            tmp.cleanup()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.eval.multi_run_gate",
        description=__doc__,
    )
    p.add_argument(
        "--reports",
        default=None,
        help="Comma-separated paths to existing eval report JSONs to aggregate "
        "(run_llm_eval --report-json output, or a baseline JSON in the "
        '{"overall": {...}} layout).  Mutually exclusive with --n-runs.',
    )
    p.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help="Run the eval this many times (subprocess) and aggregate the fresh "
        "reports.  Mutually exclusive with --reports.",
    )
    p.add_argument(
        "--suite",
        default=DEFAULT_SUITE,
        help="Suites to run in --n-runs mode (comma-separated).  Default: "
        f"{DEFAULT_SUITE} (the CI gate set — no DB needed).",
    )
    p.add_argument(
        "--judge",
        choices=("llm", "deterministic"),
        default="deterministic",
        help="Judge mode for --n-runs mode.  Default deterministic — the "
        "reproducible channel CI gates on (see eval.yml).",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="A metric's CI lower bound may sit up to this far below the "
        "threshold without failing the gate (absorbs small-sample CI width "
        f"and residual sampling bias).  Default {DEFAULT_TOLERANCE} — wider "
        "than compare_baseline's 0.01 on purpose.",
    )
    p.add_argument(
        "--report-dir",
        default=None,
        help="Write per-run reports (--n-runs mode) and an aggregate report "
        "JSON (both modes) into this directory; CI uploads it as the report "
        "artifact.",
    )
    for flag, metric in _MIN_FLAGS:
        p.add_argument(
            flag,
            dest=f"min_{metric}",
            type=float,
            default=None,
            help=f"Gate: fail (exit 2) if the 95%% CI lower bound of the mean "
            f"{metric} (across runs) drops below this minus --tolerance.",
        )
    return p


def _build_thresholds(args: argparse.Namespace) -> dict[str, float]:
    """Collect the non-None --min-* flags into a {metric: minimum} map."""
    thresholds: dict[str, float] = {}
    for _, metric in _MIN_FLAGS:
        value = getattr(args, f"min_{metric}", None)
        if value is not None:
            thresholds[metric] = float(value)
    return thresholds


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if (args.reports is not None) == (args.n_runs is not None):
        parser.error("specify exactly one of --reports or --n-runs")
    if args.n_runs is not None and args.n_runs < 1:
        parser.error("--n-runs must be >= 1")

    thresholds = _build_thresholds(args)

    report_dir = Path(args.report_dir) if args.report_dir else None
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)

    if args.reports is not None:
        paths = [p.strip() for p in args.reports.split(",") if p.strip()]
        if not paths:
            parser.error("--reports: no non-empty paths given")
        reports = [load_report(p) for p in paths]
        sources = paths
    else:
        runs = _run_eval_runs(args, report_dir)
        reports = [r for _, r in runs]
        sources = [p for p, _ in runs]

    modes = [report_judge_mode(r) for r in reports]
    _warn_mixed_judge_modes(modes)

    aggregates = aggregate(reports)
    _warn_absent_thresholds(aggregates, thresholds)
    failures = gate(aggregates, thresholds, args.tolerance)

    print_aggregates(aggregates, thresholds, args.tolerance, len(reports), modes)
    if failures:
        for f in failures:
            print(
                f"✗ {f.suite}/{f.metric}: ci95_lower {f.ci95_lower:.4f} < "
                f"min {f.threshold:.3f} - tolerance {args.tolerance:.3f} "
                f"(mean {f.mean:.4f}, n={f.n})",
                file=sys.stderr,
            )
        print(
            f"✗ gate failed: {len(failures)} metric(s) below "
            "threshold - tolerance",
            file=sys.stderr,
        )
    else:
        print("✓ gate passed: no metric's CI lower bound below threshold - tolerance",
              file=sys.stderr)

    if report_dir is not None:
        payload = build_aggregate_report(
            reports=reports,
            aggregates=aggregates,
            thresholds=thresholds,
            tolerance=args.tolerance,
            failures=failures,
            sources=sources,
        )
        out = report_dir / "llm-eval-report.json"
        write_json_report(payload, out)
        print(f"✓ aggregate report → {out}", file=sys.stderr)

    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
