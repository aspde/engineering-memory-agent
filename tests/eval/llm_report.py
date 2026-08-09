"""Report generation for the LLM behavior eval — Markdown + JSON.

Output layout (Markdown), one section per suite:
    1. Overall table — rows = metric columns for that suite
    2. Per-category table — metric × category
    3. Per-query detail — collapsible, for forensic analysis (answers,
       extracted entities, judge verdicts)
    4. Errors / judge-degradation summary

JSON mirrors the structure for programmatic consumption (CI gates, trend
dashboards).  ``summarize`` is the one-line CI-log view.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any

from tests.eval.llm_runner import LlmEvalResult

SUITE_TITLES: dict[str, str] = {
    "tool_selection": "工具选择",
    "extraction": "知识抽取",
    "answer": "最终答案",
    "e2e": "端到端问答",
}


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def _overall_table(results: Sequence[LlmEvalResult]) -> str:
    sections: list[str] = []
    for r in results:
        title = SUITE_TITLES.get(r.suite, r.suite)
        judge_note = "" if r.judge == "deterministic" else " (LLM judge)"
        sections.append(f"### {title}{judge_note}")
        sections.append("")
        cols = list(r.metric_keys)
        header = "| metric | " + " | ".join(cols) + " |"
        sep = "|---" * (len(cols) + 1) + "|"
        lines = [header, sep]
        cells = [_fmt(r.overall.get(k, 0.0)) for k in cols]
        lines.append(f"| **overall** | " + " | ".join(cells) + " |")
        sections.append("\n".join(lines))
        sections.append("")
    return "\n".join(sections)


def _category_table(result: LlmEvalResult) -> str:
    cats = list(result.by_category)
    if not cats:
        return "_(no category data)_"
    cols = list(result.metric_keys)
    # Keep the table narrow: category breakdown only for the suite's headline
    # metrics (skip auxiliary columns like answer_len / ungrounded_claims).
    headline = [k for k in cols if k not in ("answer_len", "ungrounded_claims")]
    header = "| category | " + " | ".join(headline) + " |"
    sep = "|---" * (len(headline) + 1) + "|"
    lines = [header, sep]
    for cat in cats:
        agg = result.by_category.get(cat, {})
        cells = [_fmt(agg.get(k, 0.0)) for k in headline]
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _per_query_detail(result: LlmEvalResult) -> str:
    lines: list[str] = [
        f"<details><summary>Per-query detail ({result.suite})</summary>",
        "",
    ]
    for q in result.per_query:
        lines.append(f"**{q['id']}** ({q.get('category', '')})")
        if q.get("query"):
            lines.append(f"- query: {q['query']}")
        if q.get("error"):
            lines.append(f"- ⚠ error: {q['error']}")
            continue
        if q.get("called"):
            lines.append(f"- called: {q['called']} "
                         f"(accuracy={_fmt(q.get('tool_accuracy', 0.0))})")
        if "entity_precision" in q:
            lines.append(
                f"- entities: precision={_fmt(q.get('entity_precision', 0.0))} "
                f"recall={_fmt(q.get('entity_recall', 0.0))} "
                f"f1={_fmt(q.get('entity_f1', 0.0))} "
                f"(n={q.get('n_entities', '?')})"
            )
            lines.append(
                f"- relations: precision={_fmt(q.get('relation_precision', 0.0))} "
                f"recall={_fmt(q.get('relation_recall', 0.0))} "
                f"f1={_fmt(q.get('relation_f1', 0.0))} "
                f"(n={q.get('n_relations', '?')})"
            )
            lines.append(
                f"- summary coverage={_fmt(q.get('summary_coverage', 0.0))}"
            )
            if "summary_faithfulness" in q:
                lines.append(
                    f"- judge: faithfulness={_fmt(q.get('summary_faithfulness', 0.0))} "
                    f"completeness={_fmt(q.get('summary_completeness', 0.0))}"
                )
        if "fact_coverage" in q:
            citation = q.get("citation_rate")
            lines.append(
                f"- coverage={_fmt(q.get('fact_coverage', 0.0))} "
                f"grounded={_fmt(q.get('groundedness', 0.0))} "
                f"citation={_fmt(citation)} "
                f"(answer len={q.get('answer_len', '?')})"
            )
            context_recall = q.get("context_recall")
            if context_recall is not None:
                lines.append(
                    f"- context_recall={_fmt(context_recall)} "
                    f"(n_retrieved={q.get('n_retrieved', '?')})"
                )
                if context_recall < 1.0:
                    lines.append(
                        "- ⚠ 检索未召回完整上下文 — 缺失事实在生成层无法补齐"
                    )
            if q.get("answer_preview"):
                lines.append(f"- answer: `{q['answer_preview']}…`")
            if citation is not None and citation == 0.0:
                lines.append(
                    "- ⚠ no source cited — candidate for prompt/eval iteration"
                )
            if q.get("ungrounded_claims"):
                lines.append(f"- ⚠ ungrounded: {q['ungrounded_claims']}")
        if q.get("judge_error"):
            lines.append(f"- ⚠ judge degraded: {q['judge_error']}")
        lines.append("")
    lines.append("</details>")
    return "\n".join(lines)


def to_json(results: Sequence[LlmEvalResult]) -> str:
    """Serialize a list of LlmEvalResults to a pretty JSON string."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results": [
            {
                "suite": r.suite,
                "judge": r.judge,
                "metric_keys": list(r.metric_keys),
                "n_items": r.n_items,
                "overall": r.overall,
                "by_category": r.by_category,
                "n_errors": len(r.errors),
                "n_judge_errors": len(r.judge_errors),
                "errors": r.errors,
                "judge_errors": r.judge_errors,
                "per_query": r.per_query,
            }
            for r in results
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def to_markdown(results: Sequence[LlmEvalResult]) -> str:
    """Render a full Markdown report for one or more suites."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    total_errors = sum(len(r.errors) for r in results)
    total_judge_errors = sum(len(r.judge_errors) for r in results)
    total_items = sum(r.n_items for r in results)

    sections: list[str] = [
        "# EMA LLM Behavior Evaluation Report",
        "",
        f"- Generated: {now}",
        f"- Suites: {len(results)}",
        f"- Items: {total_items}",
        f"- Execution errors: {total_errors}",
        f"- Judge degradations: {total_judge_errors}",
        "",
        "## Overall",
        "",
        _overall_table(results),
        "",
    ]

    for r in results:
        title = SUITE_TITLES.get(r.suite, r.suite)
        sections.append(f"## {title} by category")
        sections.append("")
        sections.append(_category_table(r))
        sections.append("")
        if r.errors:
            sections.append(f"### {title} — execution errors")
            sections.append("")
            for e in r.errors:
                sections.append(f"- `{e['id']}`: {e['error']}")
            sections.append("")
        if r.judge_errors:
            sections.append(f"### {title} — judge degradations")
            sections.append("")
            for e in r.judge_errors:
                sections.append(f"- `{e['id']}`: {e['error']}")
            sections.append("")

    for r in results:
        detail = _per_query_detail(r)
        sections.append(detail)
        sections.append("")

    return "\n".join(sections)


def write_json(results: Sequence[LlmEvalResult], path: str) -> str:
    """Write JSON report to ``path``. Returns the path."""
    from pathlib import Path

    p = Path(path)
    p.write_text(to_json(results), encoding="utf-8")
    return str(p)


def write_markdown(results: Sequence[LlmEvalResult], path: str) -> str:
    """Write Markdown report to ``path``. Returns the path."""
    from pathlib import Path

    p = Path(path)
    p.write_text(to_markdown(results), encoding="utf-8")
    return str(p)


def summarize(result: LlmEvalResult) -> str:
    """One-line summary for stdout / CI logs."""
    if result.suite == "tool_selection":
        return (
            f"[tool_selection] accuracy={_fmt(result.metric('tool_accuracy'))} "
            f"expected_recall={_fmt(result.metric('expected_recall'))} "
            f"unexpected={_fmt(result.metric('unexpected_rate'))} "
            f"items={result.n_items} errors={len(result.errors)}"
        )
    if result.suite == "extraction":
        return (
            f"[extraction] entity_f1={_fmt(result.metric('entity_f1'))} "
            f"relation_f1={_fmt(result.metric('relation_f1'))} "
            f"summary_coverage={_fmt(result.metric('summary_coverage'))} "
            f"items={result.n_items} errors={len(result.errors)}"
        )
    if result.suite == "e2e":
        return (
            f"[e2e] context_recall={_fmt(result.metric('context_recall'))} "
            f"coverage={_fmt(result.metric('fact_coverage'))} "
            f"groundedness={_fmt(result.metric('groundedness'))} "
            f"hallucination={_fmt(result.metric('hallucination_rate'))} "
            f"citation={_fmt(result.metric('citation_rate'))} "
            f"items={result.n_items} errors={len(result.errors)}"
        )
    return (
        f"[answer] coverage={_fmt(result.metric('fact_coverage'))} "
        f"groundedness={_fmt(result.metric('groundedness'))} "
        f"hallucination={_fmt(result.metric('hallucination_rate'))} "
        f"citation={_fmt(result.metric('citation_rate'))} "
        f"items={result.n_items} errors={len(result.errors)}"
    )
