"""CLI entry: run EMA retrieval evaluation.

Examples:
    # Validate dataset (no DB / no LLM needed)
    python -m tests.eval.run_eval --validate-only

    # Single run: memory retriever, cross-encoder rerank, top-5
    python -m tests.eval.run_eval --retriever memory

    # A/B: cross-encoder vs LLM rerank on the memories table
    python -m tests.eval.run_eval --retriever memory --compare

    # Chunk-table path (needs `python -m tests.eval.seed` first)
    python -m tests.eval.run_eval --retriever chunk --report-md report.md

    # Filter to one category
    python -m tests.eval.run_eval --retriever memory --category 技术决策

    # Save machine-readable JSON for CI gating / trend tracking
    python -m tests.eval.run_eval --retriever memory --report-json report.json

    # Regression gate: fail (exit 2) if overall metrics fall below thresholds
    python -m tests.eval.run_eval --retriever memory --min-recall@5 0.95 --min-mrr 0.90
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from tests.eval.dataset import rerank_tag, validate_dataset
from tests.eval.report import (
    summarize,
    write_json,
    write_markdown,
)
from tests.eval.runner import EvalConfig, compare_eval, run_eval
from tests.eval.thresholds import check_thresholds, print_threshold_failures


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.eval.run_eval",
        description="Run EMA retrieval evaluation (Recall@5 / MRR / NDCG / MAP).",
    )
    p.add_argument(
        "--retriever",
        choices=("memory", "chunk", "vector", "hybrid", "hybrid_norerank", "rewrite"),
        default="memory",
        help="Retrieval path: 'memory' (memories table), 'chunk' (chunks "
        "table), 'vector' (chunks table, NO rerank — fast baseline, "
        "~200ms/query), 'hybrid' (dense+sparse BM25 union), or 'rewrite' "
        "(LLM query rewrite + multi-query union, ~700ms/query). The read "
        "paths skip reranking by default. Default: memory.",
    )
    p.add_argument("--top-k", type=int, default=5, help="K for @K metrics. Default: 5.")
    p.add_argument(
        "--llm-rerank",
        action="store_true",
        help="Use LLM rerank instead of the default no-rerank path. "
        "Costlier; needs LLM provider.",
    )
    p.add_argument(
        "--cross-encoder",
        action="store_true",
        help="Use the local cross-encoder reranker instead of the default "
        "no-rerank path.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Similarity floor. Default: 0.0 for chunks, 0.3 for memories.",
    )
    p.add_argument(
        "--category",
        action="append",
        default=None,
        help="Filter to category (repeatable). Default: all categories.",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="A/B mode: run cross-encoder AND LLM rerank, print delta. "
        "Overrides --llm-rerank.",
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
        help="Only validate the labeled set vs seed corpus; skip retrieval.",
    )
    p.add_argument(
        "--semantic-relevance",
        dest="semantic_relevance",
        action="store_true",
        default=True,
        help="Relevance = substring match OR embedding similarity vs target "
        "seed summaries (default). Pass --no-semantic-relevance for the pure "
        "lexical baseline. Requires the embedding provider.",
    )
    p.add_argument(
        "--no-semantic-relevance",
        dest="semantic_relevance",
        action="store_false",
        help="Disable the semantic channel; score on substring fingerprints only.",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Override the run name in reports. Default: auto-generated label.",
    )
    p.add_argument(
        "--min-recall@5",
        dest="min_recall_at_5",
        type=float,
        default=None,
        help="Regression gate: fail (exit 2) if overall recall@5 drops below "
        "this value. Not set by default — the gate is opt-in.",
    )
    p.add_argument(
        "--min-mrr",
        type=float,
        default=None,
        help="Regression gate: fail (exit 2) if overall MRR drops below this "
        "value. Not set by default — the gate is opt-in.",
    )
    return p


def _build_thresholds(args: argparse.Namespace) -> dict[str, float]:
    """Collect the non-None min-metric CLI flags into a thresholds map.

    Metric keys use the same names as ``EvalResult.overall`` (e.g.
    ``recall@5``, ``mrr``) so the map feeds straight into
    :func:`~tests.eval.thresholds.check_thresholds`.
    """
    thresholds: dict[str, float] = {}
    if args.min_recall_at_5 is not None:
        thresholds["recall@5"] = args.min_recall_at_5
    if args.min_mrr is not None:
        thresholds["mrr"] = args.min_mrr
    return thresholds


def _make_config(
    *,
    name: str | None,
    retriever: str,
    top_k: int,
    use_llm_rerank: bool,
    use_cross_encoder: bool,
    threshold: float | None,
    categories: list[str] | None,
) -> EvalConfig:
    cfg = EvalConfig(
        name=name or f"{retriever}:{rerank_tag(use_llm_rerank, use_cross_encoder)}@k{top_k}",
        retriever=retriever,
        top_k=top_k,
        use_llm_rerank=use_llm_rerank,
        use_cross_encoder=use_cross_encoder,
        threshold=threshold,
        categories=categories,
    )
    return cfg


async def _run(args: argparse.Namespace) -> int:
    # Always validate first — cheap, catches dataset drift before any API calls.
    warnings = validate_dataset()
    if warnings:
        print("⚠ Dataset warnings:", file=sys.stderr)
        for w in warnings:
            print(f"  - {w}", file=sys.stderr)
    else:
        print("✓ Dataset validated (fingerprints unique & resolvable)", file=sys.stderr)

    if args.validate_only:
        return 0

    categories = args.category  # None or list

    if args.compare:
        cfg_a = _make_config(
            # name=None → "{retriever}:{rerank_tag}@k{n}" derived from the flags,
            # so the report label matches the reranker actually wired up.
            name=None,
            retriever=args.retriever,
            top_k=args.top_k,
            use_llm_rerank=False,
            use_cross_encoder=True,
            threshold=args.threshold,
            categories=categories,
        )
        cfg_b = _make_config(
            name=None,
            retriever=args.retriever,
            top_k=args.top_k,
            use_llm_rerank=True,
            use_cross_encoder=False,
            threshold=args.threshold,
            categories=categories,
        )
        results = await compare_eval(
            [cfg_a, cfg_b],
            semantic_relevance=args.semantic_relevance,
        )
    else:
        cfg = _make_config(
            name=args.name,
            retriever=args.retriever,
            top_k=args.top_k,
            use_llm_rerank=args.llm_rerank,
            use_cross_encoder=args.cross_encoder,
            threshold=args.threshold,
            categories=categories,
        )
        results = [await run_eval(cfg, semantic_relevance=args.semantic_relevance)]

    for r in results:
        print(summarize(r))

    if args.report_md:
        path = write_markdown(results, args.report_md)
        print(f"✓ Markdown report → {path}", file=sys.stderr)
    if args.report_json:
        path = write_json(results, args.report_json)
        print(f"✓ JSON report → {path}", file=sys.stderr)

    # Non-zero exit if any retrieval errors occurred (useful for CI).
    rc = 1 if any(r.errors for r in results) else 0

    # Regression gate (opt-in via --min-* flags): when a threshold is set,
    # fail the run if the corresponding overall metric falls below it.  The
    # production-default (memory) config is the one the weekly workflow gates
    # on.  Exit code 2 keeps this distinct from retrieval errors (1).
    thresholds = _build_thresholds(args)
    if thresholds:
        for r in results:
            outcome = check_thresholds(r.overall, thresholds)
            if not outcome.passed:
                print(
                    f"✗ Threshold check failed for {r.config.name}:",
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
