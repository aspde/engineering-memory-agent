"""CLI entry: run EMA task-level end-to-end evaluation.

Drives the *real agent graph* (ReAct loop, real tool execution, HITL gates
auto-approved) on the labeled multi-step task set, and measures task
completion, tool trajectory, loop discipline, and final-answer quality.

Needs a configured LLM provider (LLM_PROVIDER / LLM_API_KEY / …) because
each task costs real LLM calls, plus a seeded corpus and a database — run
``python -m tests.eval.e2e_seed --clear`` first so the facts each task
depends on are searchable.

The LLM-as-judge runs on the dedicated judge provider when ``LLM_JUDGE_*``
is configured (see ``backend.service.llm_service.get_judge_provider``) so
verdicts come from a model independent of the one evaluated.  ``--judge llm``
without ``LLM_JUDGE_*`` is a hard error (self-judging would inflate every
verdict); use ``--judge deterministic`` for judge-free runs.

Examples:
    # Validate the task set (no LLM / no DB needed)
    python -m tests.eval.run_task_eval --validate-only

    # Cheap smoke run — 3 tasks, deterministic judging only
    python -m tests.eval.run_task_eval --sample 3 --judge deterministic

    # Full run with LLM judges and a Markdown report
    python -m tests.eval.run_task_eval --report-md docs/interview/task-eval-report.md

    # Regression gate: fail (exit 2) if completion or groundedness drop
    python -m tests.eval.run_task_eval \
        --min-completed 0.75 --min-groundedness 0.80 --min-within-budget 0.80
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from backend.shared.config import config
from tests.eval.task_ground_truth import load_task_items, validate_task_dataset
from tests.eval.task_report import summarize, write_json, write_markdown
from tests.eval.thresholds import check_thresholds, print_threshold_failures


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.eval.run_task_eval",
        description="Run EMA task-level end-to-end evaluation "
        "(real agent graph on multi-step tasks).",
    )
    p.add_argument(
        "--judge",
        choices=("llm", "deterministic"),
        default="llm",
        help="Final-answer judging: 'llm' (default) uses an LLM-as-judge "
        "verdict against the context the agent actually saw; 'deterministic' "
        "uses substring matching only (cheaper).",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Limit the run to this many tasks (cheap smoke runs).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="ReAct loop budget for the agent graph (default: "
        "config.max_agent_steps).",
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
        help="Only validate the task set; skip LLM calls and graph runs.",
    )
    for flag, metric in (
        ("--min-completed", "completed"),
        ("--min-tool-recall", "tool_recall"),
        ("--min-within-budget", "within_budget"),
        ("--min-fact-coverage", "fact_coverage"),
        ("--min-groundedness", "groundedness"),
        ("--min-citation-rate", "citation_rate"),
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
        "completed",
        "tool_recall",
        "within_budget",
        "fact_coverage",
        "groundedness",
        "citation_rate",
    ):
        value = getattr(args, f"min_{metric}", None)
        if value is not None:
            thresholds[metric] = float(value)
    return thresholds


def _guard_judge_provider(args: argparse.Namespace) -> None:
    """Fail fast when ``--judge llm`` would silently self-judge.

    Same guard as ``run_llm_eval``: without a complete ``LLM_JUDGE_*`` block
    the judge falls back to the primary provider, so the evaluated model
    scores its own output.  ``--judge deterministic`` and ``--validate-only``
    are exempt.
    """
    if args.validate_only or args.judge != "llm":
        return
    missing: list[str] = []
    if not config.llm.judge_provider:
        missing.append("LLM_JUDGE_PROVIDER")
    else:
        if not config.llm.judge_model:
            missing.append("LLM_JUDGE_MODEL")
        if not config.llm.judge_api_key:
            missing.append("LLM_JUDGE_API_KEY")
        if config.llm.judge_provider != "anthropic" and not config.llm.judge_base_url:
            missing.append("LLM_JUDGE_BASE_URL")
    if not missing:
        return
    raise SystemExit(
        "LLM-as-judge (--judge llm) needs a complete dedicated judge provider "
        f"(missing: {', '.join(missing)}).  Set all of LLM_JUDGE_PROVIDER / "
        "LLM_JUDGE_MODEL / LLM_JUDGE_API_KEY (plus LLM_JUDGE_BASE_URL unless "
        "the provider is anthropic) so verdicts come from a model other than "
        "the one evaluated — without it the eval self-judges and every gate "
        "is meaningless.  Run with --judge deterministic to skip LLM judging."
    )


def _judge_channel_degraded(result, *, ratio: float = 0.5) -> bool:
    """True when judge failures cover at least *ratio* of the task rows.

    A judge channel that fails on most tasks is a broken run — the verdicts
    that remain are not a sample of quality.  Treat that as a run failure.
    """
    if not result.judge_errors:
        return False
    return len(result.judge_errors) >= ratio * result.n_items


async def _run(args: argparse.Namespace) -> int:
    warnings = validate_task_dataset()
    if warnings:
        print("⚠ Task dataset warnings:", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)
    else:
        print("✓ Task dataset validated", file=sys.stderr)

    if args.validate_only:
        return 0

    _guard_judge_provider(args)

    from tests.eval.task_runner import run_tasks

    items = load_task_items()
    if args.sample and args.sample > 0:
        items = items[: args.sample]

    result = await run_tasks(
        items=items,
        judge=args.judge,
        max_steps=args.max_steps,
    )
    print(summarize(result))
    if result.judge_errors:
        print(
            f"  ⚠ task: judge degraded on {len(result.judge_errors)}/"
            f"{result.n_items} rows — those verdicts are not LLM-judged",
            file=sys.stderr,
        )

    if args.report_md:
        path = write_markdown([result], args.report_md)
        print(f"✓ Markdown report → {path}", file=sys.stderr)
    if args.report_json:
        path = write_json([result], args.report_json)
        print(f"✓ JSON report → {path}", file=sys.stderr)

    rc = 1 if result.errors else 0
    if _judge_channel_degraded(result):
        print(
            f"✗ task: judge channel failed on {len(result.judge_errors)}/"
            f"{result.n_items} rows — judge metrics not trustworthy, run failed",
            file=sys.stderr,
        )
        rc = 1

    thresholds = _build_thresholds(args)
    if thresholds:
        for metric, minimum in thresholds.items():
            if metric not in result.overall:
                print(
                    f"⚠ --min-{metric} {minimum} ignored: the task run does not "
                    "report that metric",
                    file=sys.stderr,
                )
        applicable = {k: v for k, v in thresholds.items() if k in result.overall}
        if applicable:
            outcome = check_thresholds(result.overall, applicable)
            if not outcome.passed:
                print("✗ Threshold check failed for task:", file=sys.stderr)
                print_threshold_failures(outcome)
                rc = 2

    return rc


def main() -> None:
    args = _build_parser().parse_args()
    rc = asyncio.run(_run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
