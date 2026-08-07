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
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from tests.eval.dataset import validate_dataset
from tests.eval.report import (
    summarize,
    write_json,
    write_markdown,
)
from tests.eval.runner import EvalConfig, compare_eval, run_eval


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.eval.run_eval",
        description="Run EMA retrieval evaluation (Recall@5 / MRR / NDCG / MAP).",
    )
    p.add_argument(
        "--retriever",
        choices=("memory", "chunk", "vector", "hybrid", "hybrid_norerank", "rewrite"),
        default="memory",
        help="Retrieval path: 'memory' (memories table, with rerank), "
        "'chunk' (chunks table, with rerank), 'vector' (chunks table, "
        "NO rerank — fast baseline, ~200ms/query), 'hybrid' (dense+sparse "
        "BM25 union + rerank), or 'rewrite' (LLM query rewrite + multi-query "
        "union + rerank, ~700ms/query). Default: memory.",
    )
    p.add_argument("--top-k", type=int, default=5, help="K for @K metrics. Default: 5.")
    p.add_argument(
        "--llm-rerank",
        action="store_true",
        help="Use LLM rerank instead of cross-encoder. Costlier; needs LLM provider.",
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
        help="Only validate the labeled set ↔ seed corpus; skip retrieval.",
    )
    p.add_argument(
        "--semantic-relevance",
        action="store_true",
        help="Add an embedding-similarity relevance channel (substring match "
        "OR cosine ≥ 0.80 vs target seed summaries). Measures semantic "
        "retrieval quality; requires the embedding provider.",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Override the run name in reports. Default: auto-generated label.",
    )
    return p


def _make_config(
    *,
    name: str | None,
    retriever: str,
    top_k: int,
    use_llm_rerank: bool,
    threshold: float | None,
    categories: list[str] | None,
) -> EvalConfig:
    cfg = EvalConfig(
        name=name or f"{retriever}:{'llm' if use_llm_rerank else 'ce'}@k{top_k}",
        retriever=retriever,
        top_k=top_k,
        use_llm_rerank=use_llm_rerank,
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
            name=f"{args.retriever}:ce@k{args.top_k}",
            retriever=args.retriever,
            top_k=args.top_k,
            use_llm_rerank=False,
            threshold=args.threshold,
            categories=categories,
        )
        cfg_b = _make_config(
            name=f"{args.retriever}:llm@k{args.top_k}",
            retriever=args.retriever,
            top_k=args.top_k,
            use_llm_rerank=True,
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
    return 1 if any(r.errors for r in results) else 0


def main() -> None:
    args = _build_parser().parse_args()
    rc = asyncio.run(_run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
