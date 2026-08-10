"""Compare a fresh LLM eval report against the committed baseline.

The eval report file (``tests/eval/reports/llm-eval-report.json``) is overwritten
every run, so on its own it can't answer "did this prompt/model change move
quality?"  This script diffs a fresh report against the committed baseline
(``docs/interview/llm-eval-baseline.json``) and prints per-metric deltas —
the feedback loop that makes an intentional change's effect visible before it
lands.

Usage:
    # After running the eval (or reusing the last report):
    python -m tests.eval.compare_baseline

    # Explicit report path, larger tolerance for LLM-judged runs:
    python -m tests.eval.compare_baseline --report path/to/report.json --tolerance 0.05

Exit code: 0 when no metric dropped beyond tolerance vs baseline; 1 when one
or more did.

Deltas are only meaningful between runs on the same judge mode: a
``deterministic`` baseline compared against an ``llm``-judged report mixes
substring semantics with LLM-judge semantics.  When the judge modes differ
the script reports deltas but also prints a warning so a mixed comparison
isn't mistaken for an apples-to-apples regression.

``--tolerance`` is the smallest downward delta (in absolute metric units) that
counts as a regression.  Even a deterministic judge produces run-to-run
noise (the model itself is sampled at low temperature, so a rerun of the same
corpus moves some metrics by ~±0.001); the default 0.01 swallows that noise
while still tripping on a real regression (e.g. groundedness 0.9 → 0.7).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASELINE_PATH = Path(__file__).resolve().parents[2] / "docs" / "interview" / "llm-eval-baseline.json"


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _report_suites(report: dict) -> dict[str, dict[str, float]]:
    """Flatten the report's per-suite ``overall`` blocks into {suite: {metric: value}}."""
    return {r["suite"]: r["overall"] for r in report["results"]}


def compare(report: dict, baseline: dict) -> dict[str, dict[str, float]]:
    """Return {suite: {metric: delta}} for every metric present in both."""
    deltas: dict[str, dict[str, float]] = {}
    report_suites = _report_suites(report)
    for suite, bl_metrics in baseline["overall"].items():
        if suite not in report_suites:
            deltas[suite] = {"__suite_missing__": 0.0}
            continue
        for metric, bl in bl_metrics.items():
            current = report_suites[suite].get(metric)
            if current is None:
                continue
            deltas.setdefault(suite, {})[metric] = round(current - bl, 4)
    return deltas


def _fmt_delta(d: float) -> str:
    if d > 0:
        return f"+{d:.4f}"
    return f"{d:.4f}"


def _report_judge_mode(report: dict) -> str | None:
    """The judge mode of the run, from the first judge-using suite.

    ``tool_selection`` never touches the judge (it is always deterministic),
    so ``results[0].judge`` would misreport a mixed run as deterministic —
    exactly the case where a semantic (``llm``) report gets wrongly compared
    against a ``deterministic`` baseline.  The ``--judge`` flag is global
    across the judge-using suites, so any non-``tool_selection`` suite's
    judge represents the run.
    """
    for r in report.get("results", []):
        if r.get("suite") != "tool_selection" and r.get("judge"):
            return str(r["judge"])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        default="tests/eval/reports/llm-eval-report.json",
        help="Fresh eval report JSON (default: tests/eval/reports/llm-eval-report.json).",
    )
    parser.add_argument(
        "--baseline",
        default=str(BASELINE_PATH),
        help="Baseline JSON (default: docs/interview/llm-eval-baseline.json).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Smallest downward delta counted as a regression (default 0.01). "
        "Deterministic-judge runs still carry ~+-0.001 run-to-run noise.",
    )
    args = parser.parse_args()

    report = _load(Path(args.report))
    baseline = _load(Path(args.baseline))

    report_judge = _report_judge_mode(report)
    base_judge = baseline.get("environment", {}).get("judge_mode")
    if report_judge and base_judge and report_judge != base_judge:
        print(
            f"✗ judge mode mismatch: report={report_judge} vs baseline={base_judge} — "
            "substring semantics (deterministic) and LLM-judge semantics (llm) "
            "are not comparable; a semantic judge always grades stricter. "
            "Refusing to diff.  Run the eval with --judge "
            f"{base_judge}, or compare against a same-mode baseline "
            "(e.g. --baseline tests/eval/reports/llm-eval-semantic-baseline.json).",
            file=sys.stderr,
        )
        return 1

    deltas = compare(report, baseline)
    if not deltas:
        print("No comparable metrics (baseline or report empty).", file=sys.stderr)
        return 1

    dropped = False
    for suite, metrics in sorted(deltas.items()):
        if "__suite_missing__" in metrics:
            print(f"[{suite}] suite absent from fresh report")
            dropped = True
            continue
        print(f"[{suite}]")
        for metric, delta in sorted(metrics.items()):
            flag = "  ▲ DROP" if delta < -args.tolerance else ""
            if delta < -args.tolerance:
                dropped = True
            print(f"  {metric:22s} {_fmt_delta(delta)}{flag}")

    return 1 if dropped else 0


if __name__ == "__main__":
    sys.exit(main())
