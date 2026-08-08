"""CLI entry: run EMA LLM behavior evaluation.

Covers three agent-behavior dimensions the retrieval eval cannot measure:
tool selection, knowledge extraction, and final-answer groundedness.  Needs
a configured LLM provider (LLM_PROVIDER / LLM_API_KEY / …) because each item
costs real LLM calls.

Examples:
    # Validate the labeled sets (no LLM / no DB needed)
    python -m tests.eval.run_llm_eval --validate-only

    # Cheap smoke run — 3 items per suite, deterministic judging only
    python -m tests.eval.run_llm_eval --sample 3 --judge deterministic

    # Full run with LLM judges and a Markdown report
    python -m tests.eval.run_llm_eval --suite all --report-md docs/interview/llm-eval-report.md

    # Regression gate: fail (exit 2) if headline metrics drop below thresholds
    python -m tests.eval.run_llm_eval --suite all \
        --min-tool-accuracy 0.70 --min-entity-f1 0.60 --min-relation-f1 0.50 \
        --min-fact-coverage 0.60 --min-groundedness 0.80
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from typing import Any

from tests.eval.llm_ground_truth import (
    load_answer_items,
    load_extraction_items,
    load_tool_selection_items,
    validate_llm_dataset,
)
from tests.eval.llm_report import (
    summarize,
    write_json,
    write_markdown,
)
from tests.eval.thresholds import check_thresholds, print_threshold_failures

SUITES: tuple[str, ...] = ("tool_selection", "extraction", "answer")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.eval.run_llm_eval",
        description="Run EMA LLM behavior evaluation "
        "(tool selection / extraction / final answer).",
    )
    p.add_argument(
        "--suite",
        choices=SUITES + ("all",),
        default="all",
        help="Which suite(s) to run. Default: all.",
    )
    p.add_argument(
        "--judge",
        choices=("llm", "deterministic"),
        default="llm",
        help="Answer/summary judging: 'llm' (default) uses an LLM-as-judge "
        "verdict; 'deterministic' uses substring matching only. "
        "Deterministic costs fewer tokens.",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit each suite to this many items (cheap smoke runs).",
    )
    p.add_argument(
        "--report-md",
        default=None,
        help="Write Markdown report to this path.",
    )
    p.add_argument(
        "--report-json",
        default=None,
        help="Write JSON report to this path.",
    )
    p.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate the labeled sets; skip LLM calls.",
    )
    for flag, metric in (
        ("--min-tool-accuracy", "tool_accuracy"),
        ("--min-expected-recall", "expected_recall"),
        ("--min-entity-f1", "entity_f1"),
        ("--min-relation-f1", "relation_f1"),
        ("--min-fact-coverage", "fact_coverage"),
        ("--min-groundedness", "groundedness"),
    ):
        p.add_argument(
            flag,
            dest=f"min_{metric}",
            type=float,
            default=None,
            help=f"Regression gate: fail (exit 2) if overall {metric} drops "
            "below this value. Not set by default — the gate is opt-in.",
        )
    return p


def _build_thresholds(args: argparse.Namespace) -> dict[str, float]:
    """Collect the non-None min-metric CLI flags into a thresholds map."""
    thresholds: dict[str, float] = {}
    for metric in (
        "tool_accuracy",
        "expected_recall",
        "entity_f1",
        "relation_f1",
        "fact_coverage",
        "groundedness",
    ):
        value = getattr(args, f"min_{metric}", None)
        if value is not None:
            thresholds[metric] = float(value)
    return thresholds


def _applicable_thresholds(
    thresholds: dict[str, float], overall: dict[str, float]
) -> dict[str, float]:
    """The subset of *thresholds* checkable against one suite's ``overall``.

    The ``--min-*`` flags span all three suites, but each suite only reports
    its own metric set.  ``check_thresholds`` treats a metric absent from
    ``overall`` as 0.0 (so the CLI cannot silently gate on a metric it cannot
    read); applying every flag to every suite would therefore fail each suite
    on another suite's metrics — e.g. ``--suite all`` with the eval.yml flags
    always exits 2 even at perfect scores.  Restrict the gate to the metrics
    this suite actually produces.
    """
    return {k: v for k, v in thresholds.items() if k in overall}


def _select_suites(args: argparse.Namespace) -> list[str]:
    return list(SUITES) if args.suite == "all" else [args.suite]


def _suite_items(suite: str) -> Sequence[Any]:
    """The labeled set for *suite*, sliced to ``--sample`` when requested."""
    items = {
        "tool_selection": load_tool_selection_items(),
        "extraction": load_extraction_items(),
        "answer": load_answer_items(),
    }[suite]
    return list(items)


async def _run_one(suite: str, args: argparse.Namespace) -> Any:
    from tests.eval.llm_runner import (
        run_answer,
        run_extraction,
        run_tool_selection,
    )

    items = _suite_items(suite)
    if args.sample and args.sample > 0:
        items = items[: args.sample]

    if suite == "tool_selection":
        return await run_tool_selection(items=items)
    if suite == "extraction":
        return await run_extraction(items=items, judge=args.judge)
    return await run_answer(items=items, judge=args.judge)


async def _run(args: argparse.Namespace) -> int:
    warnings = validate_llm_dataset()
    if warnings:
        print("⚠ LLM eval dataset warnings:", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)
    else:
        print("✓ LLM eval datasets validated", file=sys.stderr)

    if args.validate_only:
        return 0

    results = [await _run_one(suite, args) for suite in _select_suites(args)]
    for r in results:
        print(summarize(r))

    if args.report_md:
        path = write_markdown(results, args.report_md)
        print(f"✓ Markdown report → {path}", file=sys.stderr)
    if args.report_json:
        path = write_json(results, args.report_json)
        print(f"✓ JSON report → {path}", file=sys.stderr)

    # Exit 1 if any execution errors (an outage reads as a broken run, not a
    # quality signal).
    rc = 1 if any(r.errors for r in results) else 0

    # Regression gate (opt-in via --min-* flags): a metric below its floor
    # fails the run with exit code 2 (distinct from execution errors).
    thresholds = _build_thresholds(args)
    if thresholds:
        # A flag whose metric no suite in this run produces cannot be gated —
        # warn rather than silently skipping (or failing every suite).
        produced = {m for r in results for m in r.overall}
        for metric, minimum in thresholds.items():
            if metric not in produced:
                print(
                    f"⚠ --min-{metric} {minimum} ignored: no suite in this run "
                    "reports that metric",
                    file=sys.stderr,
                )
        for r in results:
            applicable = _applicable_thresholds(thresholds, r.overall)
            if not applicable:
                continue
            outcome = check_thresholds(r.overall, applicable)
            if not outcome.passed:
                print(
                    f"✗ Threshold check failed for {r.suite}:",
                    file=sys.stderr,
                )
                print_threshold_failures(outcome)
                rc = 2

    return rc


def main() -> None:
    args = _build_parser().parse_args()
    rc = asyncio.run(_run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
