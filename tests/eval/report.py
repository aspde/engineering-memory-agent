"""Report generation — Markdown + JSON for human and CI consumption.

Output layout (Markdown):
    1. Header — timestamp, configs, query count, error count
    2. Overall table — one row per config, columns = metrics
    3. Per-category table — recall@5 / mrr by config × category
    4. Per-difficulty table — recall@5 by config × difficulty
    5. A/B delta table — pairwise metric deltas (only if ≥2 configs)
    6. Per-query detail — collapsible, for forensic analysis

JSON mirrors the same structure for programmatic consumption (CI gates,
trend dashboards).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from tests.eval.ground_truth import DIFFICULTIES
from tests.eval.runner import METRIC_KEYS, EvalResult, result_to_dict


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def _delta(a: float, b: float) -> str:
    """Format a-b with sign, e.g. ``+0.123`` / ``-0.045`` / ``+0.000``."""
    return f"{a - b:+.3f}"


def to_json(results: Sequence[EvalResult]) -> str:
    """Serialize a list of EvalResults to a pretty JSON string."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": [result_to_dict(r) for r in results],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _semantic_rescued(r: EvalResult) -> int:
    """Count queries the lexical baseline missed but the semantic channel saved.

    A query is "rescued" when substring matching found nothing
    (``substring_hits == 0``) yet the semantic channel contributed at least
    one hit.  This is the concrete measure of semantic retrieval quality:
    retrieval that only works via surface-form overlap scores 0 here.
    """
    return sum(
        1 for q in r.per_query
        if q.get("substring_hits", 0) == 0 and q.get("semantic_only_hits", 0) > 0
    )


def _overall_table(results: Sequence[EvalResult]) -> str:
    cols = list(METRIC_KEYS) + ["semantic_rescued", "latency_ms", "n_queries"]
    header = "| config | " + " | ".join(cols) + " |"
    sep = "|---" * (len(cols) + 1) + "|"
    lines = [header, sep]
    for r in results:
        cells = [_fmt(r.overall.get(k, 0.0)) for k in METRIC_KEYS]
        cells.append(str(_semantic_rescued(r)))
        cells.append(_fmt(r.overall.get("latency_ms", 0.0), 0))
        cells.append(str(r.n_queries))
        lines.append(f"| {r.config.name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _category_table(results: Sequence[EvalResult], metric: str = "recall@5") -> str:
    # Collect union of categories present across all results, preserving order.
    cats: list[str] = []
    for r in results:
        for c in r.by_category:
            if c not in cats:
                cats.append(c)
    if not cats:
        return "_(no category data)_"
    header = "| config | " + " | ".join(cats) + " |"
    sep = "|---" * (len(cats) + 1) + "|"
    lines = [header, sep]
    for r in results:
        cells = [_fmt(r.by_category.get(c, {}).get(metric, 0.0)) for c in cats]
        lines.append(f"| {r.config.name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _difficulty_table(results: Sequence[EvalResult], metric: str = "recall@5") -> str:
    diffs = list(DIFFICULTIES)
    header = "| config | " + " | ".join(diffs) + " |"
    sep = "|---" * (len(diffs) + 1) + "|"
    lines = [header, sep]
    for r in results:
        cells = [_fmt(r.by_difficulty.get(d, {}).get(metric, 0.0)) for d in diffs]
        lines.append(f"| {r.config.name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _ab_delta_table(results: Sequence[EvalResult]) -> str:
    """Pairwise delta: result[i+1] - result[i] for each metric."""
    if len(results) < 2:
        return ""
    lines: list[str] = []
    for i in range(1, len(results)):
        a, b = results[i - 1], results[i]
        lines.append(f"### Δ {b.config.name} − {a.config.name}\n")
        header = "| metric | " + a.config.name + " | " + b.config.name + " | Δ |"
        sep = "|---|---|---|---|"
        lines.append(header)
        lines.append(sep)
        for k in METRIC_KEYS:
            va = a.overall.get(k, 0.0)
            vb = b.overall.get(k, 0.0)
            lines.append(
                f"| {k} | {_fmt(va)} | {_fmt(vb)} | {_delta(vb, va)} |"
            )
        # latency delta (ms, signed)
        la = a.overall.get("latency_ms", 0.0)
        lb = b.overall.get("latency_ms", 0.0)
        lines.append(
            f"| latency_ms | {_fmt(la, 0)} | {_fmt(lb, 0)} | {lb - la:+.0f} |"
        )
        lines.append("")
    return "\n".join(lines)


def _per_query_detail(results: Sequence[EvalResult]) -> str:
    """Per-query rows for the first result (forensic detail)."""
    if not results:
        return ""
    r = results[0]
    lines = [
        f"<details><summary>Per-query detail ({r.config.name})</summary>",
        "",
        "| id | category | difficulty | n_ret | n_rel | sem | recall@5 | mrr | ndcg@5 | latency_ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for q in r.per_query:
        # sem marker: ✓ = rescued by the semantic channel alone (lexical
        # baseline missed it); — = lexical hit; ✗ = nothing relevant found.
        if q.get("substring_hits", 0) == 0 and q.get("semantic_only_hits", 0) > 0:
            sem_marker = "✓"
        elif q.get("substring_hits", 0) > 0:
            sem_marker = "—"
        else:
            sem_marker = "✗"
        lines.append(
            f"| {q['id']} | {q['category']} | {q['difficulty']} | "
            f"{q['n_retrieved']} | {q['n_relevant']} | {sem_marker} | "
            f"{_fmt(q.get('recall@5', 0.0))} | {_fmt(q.get('mrr', 0.0))} | "
            f"{_fmt(q.get('ndcg@5', 0.0))} | {_fmt(q.get('latency_ms', 0.0), 0)} |"
        )
    lines.append("</details>")
    return "\n".join(lines)


def to_markdown(results: Sequence[EvalResult]) -> str:
    """Render a full Markdown report for one or more EvalResults."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    total_errors = sum(len(r.errors) for r in results)
    # compare_eval runs every config over the same labeled set, so every result
    # shares n_queries. Take the first; fall back to 0 for an empty result list.
    total_queries = results[0].n_queries if results else 0

    # Semantic channel state — inferred from the first result's per-query rows
    # (all configs in one report share the same flag via compare_eval).
    if results and results[0].per_query:
        semantic_on = bool(results[0].per_query[0].get("semantic_relevance", True))
        semantic_line = (
            "enabled (substring OR embedding-similarity)"
            if semantic_on
            else "disabled (substring fingerprints only)"
        )
    else:
        semantic_line = "n/a"

    sections: list[str] = [
        f"# EMA Retrieval Evaluation Report",
        "",
        f"- Generated: {now}",
        f"- Queries: {total_queries}",
        f"- Configs: {len(results)}",
        f"- Errors: {total_errors}",
        f"- Semantic relevance: {semantic_line}",
        "",
        "## Overall",
        "",
        _overall_table(results),
        "",
        "## Recall@5 by category",
        "",
        _category_table(results, "recall@5"),
        "",
        "## MRR by category",
        "",
        _category_table(results, "mrr"),
        "",
        "## Recall@5 by difficulty",
        "",
        _difficulty_table(results, "recall@5"),
        "",
    ]

    delta = _ab_delta_table(results)
    if delta:
        sections.append("## A/B comparison")
        sections.append("")
        sections.append(delta)

    detail = _per_query_detail(results)
    if detail:
        sections.append(detail)
        sections.append("")

    return "\n".join(sections)


def write_json(results: Sequence[EvalResult], path: str) -> str:
    """Write JSON report to ``path``. Returns the path."""
    from pathlib import Path

    p = Path(path)
    p.write_text(to_json(results), encoding="utf-8")
    return str(p)


def write_markdown(results: Sequence[EvalResult], path: str) -> str:
    """Write Markdown report to ``path``. Returns the path."""
    from pathlib import Path

    p = Path(path)
    p.write_text(to_markdown(results), encoding="utf-8")
    return str(p)


def summarize(result: EvalResult) -> str:
    """One-line summary for stdout / CI logs."""
    return (
        f"[{result.config.name}] "
        f"recall@5={_fmt(result.metric('recall@5'))} "
        f"mrr={_fmt(result.metric('mrr'))} "
        f"ndcg@5={_fmt(result.metric('ndcg@5'))} "
        f"map@5={_fmt(result.metric('map@5'))} "
        f"avg_latency={_fmt(result.overall.get('latency_ms', 0.0), 0)}ms "
        f"queries={result.n_queries} errors={len(result.errors)}"
    )
