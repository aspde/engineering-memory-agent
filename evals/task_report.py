"""Report generation for the task-level end-to-end eval — Markdown + JSON.

Layout mirrors ``evals.llm_report``:
    1. Overall table — metric columns for the task suite
    2. Per-category table — metric × category
    3. Per-task detail — collapsible, for forensic analysis (trajectory,
       loop steps, answer preview, judge verdicts)
    4. Errors / judge-degradation summary

The JSON layout, table helpers and file writers are shared with the LLM
behaviour report via ``evals.core``.  ``summarize`` is the one-line
CI-log view.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from evals.core import (
    EvalResult,
    category_table,
    overall_table,
    to_json,
    write_text,
)
from evals.core import (
    fmt as _fmt,
)

SUITE_TITLE = "任务级端到端"


def _overall_table(results: Sequence[EvalResult]) -> str:
    """Overall metric table — shared core helper with the single suite title."""
    return overall_table(results, lambda _: SUITE_TITLE)


def _category_table(result: EvalResult) -> str:
    """Category × headline-metric table — shared core helper."""
    return category_table(result, ("n_steps", "answer_len", "ungrounded_claims"))


def _environment_summary(result: EvalResult) -> str:
    """Environment-noise breakdown for one result.

    The 2026-08-24 post-discipline runs lost 14/19 failure cells to provider
    outages and wall-clock timeouts that said nothing about agent behaviour.
    This block separates the two so a future reader does not have to forens
    each per-task cell: ``completed`` averages every row, ``completed_clean``
    averages only rows whose outcome the environment did not disturb.
    """
    classes = {"ok": 0, "provider_error": 0, "timeout": 0}
    for q in result.per_query:
        classes[str(q.get("outcome_class", "ok"))] = (
            classes.get(str(q.get("outcome_class", "ok")), 0) + 1
        )
    polluted = classes["provider_error"] + classes["timeout"]
    if polluted == 0:
        return "_Environment: all 8 runs clean — `completed` is directly comparable._"
    lines = [
        f"_Environment: {result.n_items - polluted}/{result.n_items} runs clean "
        f"({classes['provider_error']} provider errors, {classes['timeout']} timeouts). "
        "`completed` averages every row; `completed_clean` averages the clean subset only._"
    ]
    return "\n".join(lines)


def _per_task_detail(result: EvalResult) -> str:
    lines: list[str] = [
        f"<details><summary>Per-task detail ({SUITE_TITLE})</summary>",
        "",
    ]
    for q in result.per_query:
        lines.append(f"**{q['id']}** ({q.get('category', '')})")
        if q.get("query"):
            lines.append(f"- query: {q['query']}")
        if q.get("error"):
            lines.append(f"- ⚠ error: {q['error']}")
            continue
        lines.append(
            f"- trajectory: {q.get('called', '-')} "
            f"(n={q.get('n_calls', '?')}, steps={q.get('n_steps', '?')}, "
            f"completed={_fmt(q.get('completed', 0.0))}, "
            f"within_budget={_fmt(q.get('within_budget', 0.0))})"
        )
        if q.get("outcome_class") and q.get("outcome_class") != "ok":
            lines.append(f"- ⚠ environment: {q['outcome_class']} (excluded from completed_clean)")
        if "fact_coverage" in q:
            lines.append(
                f"- coverage={_fmt(q.get('fact_coverage', 0.0))} "
                f"grounded={_fmt(q.get('groundedness', 0.0))} "
                f"citation={_fmt(q.get('citation_rate', 0.0))} "
                f"(answer len={q.get('answer_len', '?')})"
            )
            if q.get("answer_preview"):
                lines.append(f"- answer: `{q['answer_preview']}…`")
            if q.get("ungrounded_claims"):
                lines.append(f"- ⚠ ungrounded: {q['ungrounded_claims']}")
        if q.get("judge_error"):
            lines.append(f"- ⚠ judge degraded: {q['judge_error']}")
        lines.append("")
    lines.append("</details>")
    return "\n".join(lines)


def to_markdown(results: Sequence[EvalResult]) -> str:
    """Render a full Markdown report for the task suite."""
    from evals.core import run_provenance

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    total_errors = sum(len(r.errors) for r in results)
    total_judge_errors = sum(len(r.judge_errors) for r in results)
    total_items = sum(r.n_items for r in results)
    prov = run_provenance()

    sections: list[str] = [
        "# EMA Task-Level E2E Evaluation Report",
        "",
        f"- Generated: {now}",
        f"- LLM channel: {prov['provider']} / {prov['model']}",
        f"- Judge: {prov['judge']}",
        f"- Suites: {len(results)}",
        f"- Tasks: {total_items}",
        f"- Execution errors: {total_errors}",
        f"- Judge degradations: {total_judge_errors}",
        "",
        "## Overall",
        "",
        _overall_table(results),
        "",
    ]

    for r in results:
        sections.append(f"## {SUITE_TITLE} by category")
        sections.append("")
        sections.append(_category_table(r))
        sections.append("")
        env_summary = _environment_summary(r)
        sections.append(env_summary)
        sections.append("")
        if r.errors:
            sections.append(f"### {SUITE_TITLE} — execution errors")
            sections.append("")
            for e in r.errors:
                sections.append(f"- `{e['id']}`: {e['error']}")
            sections.append("")
        if r.judge_errors:
            sections.append(f"### {SUITE_TITLE} — judge degradations")
            sections.append("")
            for e in r.judge_errors:
                sections.append(f"- `{e['id']}`: {e['error']}")
            sections.append("")

    for r in results:
        sections.append(_per_task_detail(r))
        sections.append("")

    return "\n".join(sections)


def write_json(results: Sequence[EvalResult], path: str) -> str:
    """Write JSON report to ``path``. Returns the path."""
    return write_text(path, to_json(results))


def write_markdown(results: Sequence[EvalResult], path: str) -> str:
    """Write Markdown report to ``path``. Returns the path."""
    return write_text(path, to_markdown(results))


def summarize(result: EvalResult) -> str:
    """One-line summary for stdout / CI logs."""
    return (
        f"[task] completed={_fmt(result.metric('completed'))} "
        f"completed_clean={_fmt(result.metric('completed_clean'))} "
        f"tool_recall={_fmt(result.metric('tool_recall'))} "
        f"within_budget={_fmt(result.metric('within_budget'))} "
        f"coverage={_fmt(result.metric('fact_coverage'))} "
        f"groundedness={_fmt(result.metric('groundedness'))} "
        f"citation={_fmt(result.metric('citation_rate'))} "
        f"tasks={result.n_items} errors={len(result.errors)}"
    )
