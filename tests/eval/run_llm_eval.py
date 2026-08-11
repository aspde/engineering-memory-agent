"""CLI entry: run EMA LLM behavior evaluation.

Covers four agent-behavior dimensions the retrieval eval cannot measure:
tool selection, knowledge extraction, final-answer groundedness, and the
end-to-end chain (real retrieval → answer).  Needs a configured LLM provider
(LLM_PROVIDER / LLM_API_KEY / …) because each item costs real LLM calls.
The e2e suite additionally needs a seeded corpus and a database — run
``python -m tests.eval.e2e_seed --clear`` first.

The LLM-as-judge runs on the dedicated judge provider when ``LLM_JUDGE_*``
is configured (see ``backend.service.llm_service.get_judge_provider``) so
verdicts come from a model independent of the one evaluated — e.g.
GLM-4.7-Flash via the Zhipu OpenAI-compatible endpoint judging DeepSeek's
output.  ``--judge llm`` without ``LLM_JUDGE_*`` is a hard error: the judge
provider would fall back to the primary (self-judging), so the run refuses
to start rather than emit self-preference-inflated verdicts.  Use
``--judge deterministic`` for judge-free runs.

Examples:
    # Validate the labeled sets (no LLM / no DB needed)
    python -m tests.eval.run_llm_eval --validate-only

    # Cheap smoke run — 3 items per suite, deterministic judging only
    python -m tests.eval.run_llm_eval --sample 3 --judge deterministic

    # The three no-DB suites (what the CI llm-eval job runs)
    python -m tests.eval.run_llm_eval --suite tool_selection,extraction,answer

    # Full run incl. end-to-end with LLM judges and a Markdown report
    python -m tests.eval.run_llm_eval --suite all --report-md tests/eval/reports/llm-eval-report.md

    # Regression gate: fail (exit 2) if headline metrics drop below thresholds
    python -m tests.eval.run_llm_eval --suite all \
        --min-tool-accuracy 0.70 --min-entity-f1 0.60 --min-relation-f1 0.50 \
        --min-fact-coverage 0.60 --min-groundedness 0.80 --min-citation-rate 0.80
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from typing import Any

from backend.shared.config import config
from tests.eval.llm_ground_truth import (
    load_answer_items,
    load_e2e_items,
    load_extraction_items,
    load_tool_selection_items,
    validate_llm_dataset,
)
from tests.eval.llm_report import (
    summarize,
    write_json,
    write_markdown,
)
from tests.eval.llm_runner import LlmEvalResult
from tests.eval.thresholds import check_thresholds, print_threshold_failures

SUITES: tuple[str, ...] = ("tool_selection", "extraction", "answer", "e2e")

# Suites whose ``--judge llm`` path actually consults the LLM judge.  A run
# that only exercises tool_selection never touches the judge, so the
# "self-judging" guard below does not apply to it.
JUDGE_USING_SUITES: tuple[str, ...] = ("extraction", "answer", "e2e")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.eval.run_llm_eval",
        description="Run EMA LLM behavior evaluation "
        "(tool selection / extraction / final answer / end-to-end).",
    )
    p.add_argument(
        "--suite",
        default="all",
        help=(
            "Suite(s) to run: comma-separated names or 'all'. "
            f"Valid suites: {', '.join(SUITES)}. "
            "Note: 'e2e' needs a seeded corpus (run `python -m "
            "tests.eval.e2e_seed --clear`) and a database. Default: all."
        ),
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
    p.add_argument(
        "--e2e-mode",
        choices=("memory", "chunk"),
        default="memory",
        help="E2E retrieval path: 'memory' (query_memories, default) or "
        "'chunk' (retrieve_hybrid).",
    )
    p.add_argument(
        "--e2e-top-k",
        type=int,
        default=5,
        help="E2E retrieval top_k (default 5).",
    )
    for flag, metric in (
        ("--min-tool-accuracy", "tool_accuracy"),
        ("--min-expected-recall", "expected_recall"),
        ("--min-entity-f1", "entity_f1"),
        ("--min-relation-f1", "relation_f1"),
        ("--min-fact-coverage", "fact_coverage"),
        ("--min-groundedness", "groundedness"),
        ("--min-citation-rate", "citation_rate"),
        ("--min-context-recall", "context_recall"),
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
        "citation_rate",
        "context_recall",
    ):
        value = getattr(args, f"min_{metric}", None)
        if value is not None:
            thresholds[metric] = float(value)
    return thresholds


def _applicable_thresholds(
    thresholds: dict[str, float], overall: dict[str, float]
) -> dict[str, float]:
    """The subset of *thresholds* checkable against one suite's ``overall``.

    The ``--min-*`` flags span all suites, but each suite only reports its
    own metric set.  ``check_thresholds`` treats a metric absent from
    ``overall`` as 0.0 (so the CLI cannot silently gate on a metric it cannot
    read); applying every flag to every suite would therefore fail each suite
    on another suite's metrics — e.g. ``--suite all`` with the eval.yml flags
    always exits 2 even at perfect scores.  Restrict the gate to the metrics
    this suite actually produces.
    """
    return {k: v for k, v in thresholds.items() if k in overall}


def _select_suites(args: argparse.Namespace) -> list[str]:
    """Resolve ``--suite`` to a list of suite names.

    Accepts ``all`` (every registered suite) or a comma-separated list.
    Unknown names are a hard error — a typo must not silently run nothing.
    """
    raw = args.suite
    if raw == "all":
        return list(SUITES)
    names = [s.strip() for s in raw.split(",") if s.strip()]
    unknown = [s for s in names if s not in SUITES]
    if unknown:
        raise SystemExit(
            f"Unknown suite(s): {', '.join(unknown)} — "
            f"expected one of {', '.join(SUITES)}"
        )
    return names


def _suites_use_judge(suites: Sequence[str]) -> bool:
    """True when any selected suite's ``--judge llm`` path calls the judge."""
    return any(s in JUDGE_USING_SUITES for s in suites)


def _guard_judge_provider(
    args: argparse.Namespace, suites: Sequence[str]
) -> None:
    """Fail fast when ``--judge llm`` would silently self-judge.

    ``get_judge_provider`` falls back to the primary provider when no
    ``LLM_JUDGE_*`` config block is set, so without this guard the evaluated
    model scores its own output — self-preference bias inflating every CI
    gate built on the verdicts.  Refuse to run (raise SystemExit) rather than
    degrade silently.  ``--judge deterministic``, ``--validate-only``, and a
    suite selection that never judges (tool_selection alone) are exempt.

    Mirrors ``backend.shared.config.validate_config``: setting
    ``LLM_JUDGE_PROVIDER`` requires the whole four-field block (model and
    api_key always, plus base_url for non-anthropic providers), because a
    half-set block — e.g. the old ``.env.example`` default of
    ``LLM_JUDGE_PROVIDER=openai`` with an empty key — would otherwise pass
    here and 401 on the first judge call.
    """
    if args.validate_only or args.judge != "llm":
        return
    if not _suites_use_judge(suites):
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


def _judge_channel_degraded(result: LlmEvalResult, *, ratio: float = 0.5) -> bool:
    """True when judge failures cover at least *ratio* of a suite's rows.

    A judge channel that fails on the majority of a suite is a broken run:
    the verdicts that remain are what survived an outage, not a sample of
    quality.  Treat that as a run failure (exit non-zero), not a quality
    signal.
    """
    if not result.judge_errors:
        return False
    return len(result.judge_errors) >= ratio * result.n_items


def _suite_items(suite: str) -> Sequence[Any]:
    """The labeled set for *suite*, sliced to ``--sample`` when requested."""
    items = {
        "tool_selection": load_tool_selection_items(),
        "extraction": load_extraction_items(),
        "answer": load_answer_items(),
        "e2e": load_e2e_items(),
    }[suite]
    return list(items)


async def _run_one(suite: str, args: argparse.Namespace) -> Any:
    from tests.eval.llm_runner import (
        run_answer,
        run_e2e,
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
    if suite == "e2e":
        return await run_e2e(
            items=items,
            judge=args.judge,
            top_k=args.e2e_top_k,
            retrieval_mode=args.e2e_mode,
        )
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

    suites = _select_suites(args)
    _guard_judge_provider(args, suites)

    results = [await _run_one(suite, args) for suite in suites]
    for r in results:
        print(summarize(r))
        if r.judge_errors:
            # Make judge degradation visible in the CI log line — the one-line
            # summarize shows execution errors only, and degraded verdicts must
            # not be mistaken for real scores.
            print(
                f"  ⚠ {r.suite}: judge degraded on {len(r.judge_errors)}/"
                f"{r.n_items} rows — those verdicts are not LLM-judged",
                file=sys.stderr,
            )

    if args.report_md:
        path = write_markdown(results, args.report_md)
        print(f"✓ Markdown report → {path}", file=sys.stderr)
    if args.report_json:
        path = write_json(results, args.report_json)
        print(f"✓ JSON report → {path}", file=sys.stderr)

    # Exit 1 on execution errors (an outage reads as a broken run, not a
    # quality signal) OR a failed judge channel (≥50% of a suite's rows
    # degraded → the judge metrics are not a trustworthy quality signal).
    rc = 1 if any(r.errors for r in results) else 0
    for r in results:
        if _judge_channel_degraded(r):
            print(
                f"✗ {r.suite}: judge channel failed on {len(r.judge_errors)}/"
                f"{r.n_items} rows — judge metrics not trustworthy, run failed",
                file=sys.stderr,
            )
            rc = 1

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
