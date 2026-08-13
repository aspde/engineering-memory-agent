"""Shared eval skeleton — the aggregation / result / serialization machinery
the three eval families (retrieval, LLM behaviour, task-level e2e) build on.

Before this module, ``tests.eval.runner`` / ``llm_runner`` / ``task_runner``
each implemented the same ``_aggregate`` mean and ``_finish`` by-category
roll-up, and ``report`` / ``llm_report`` / ``task_report`` each carried an
identical JSON scaffold, ``_fmt`` helper and ``write_*`` wrappers.  The
individual runners now supply only what is genuinely per-suite — their item
types, metric keys, executors, and the per-query detail rendering — and
everything shared lives here.

Public pieces:

- :class:`EvalResult` — the generic result container used by the LLM and task
  runners (``tests.eval.llm_runner.LlmEvalResult`` and
  ``tests.eval.task_runner.TaskEvalResult`` are aliases of it).  The retrieval
  runner keeps its own result type because it carries ``config`` /
  ``by_difficulty`` / ``total_latency_ms`` on top of the shared roll-up.
- :func:`aggregate` / :func:`finish` — the per-key mean and the by-category
  roll-up, including the "failed items count as zero rows" denominator policy.
- :data:`ANSWER_JUDGE_METRIC_KEYS` + :func:`zero_judge_keys` — the shared
  judge-failure policy: zero the judge-owned gate keys instead of re-feeding
  the deterministic substring channel under the same names, so a
  ``--min-groundedness`` gate fails loudly on a judge outage rather than
  silently grading substrings.  The extraction suite is deliberately exempt
  (it leaves its judge-owned summary keys unset, not zeroed — a failed
  judgment is not evidence the summary was bad).
- :func:`to_json` / :func:`write_text` — the judge-based report JSON layout
  and file-write helper shared by the report modules.
- :func:`fmt` / :func:`overall_table` / :func:`category_table` — the markdown
  table helpers shared by ``llm_report`` and ``task_report`` (the retrieval
  report renders config × metric tables instead).
- :func:`load_jsonl_items` — the labeled-set loader shared by the three
  ground-truth modules, which keep their data in ``tests/eval/data/*.jsonl``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# The judge-owned metric keys shared by the answer / e2e / task suites.  On a
# judge failure these are zeroed (never re-fed the deterministic channel) —
# see :func:`zero_judge_keys`.  Kept importable from ``tests.eval.llm_runner``
# for backwards compatibility with older import paths.
ANSWER_JUDGE_METRIC_KEYS: tuple[str, ...] = (
    "fact_coverage",
    "groundedness",
    "hallucination_rate",
)


@dataclass
class EvalResult:
    """Aggregate result of one eval suite over its labeled set.

    Shape shared by the LLM behaviour suites (``tests.eval.llm_runner``) and
    the task-level suite (``tests.eval.task_runner``); the retrieval runner
    uses a separate result type with ``config`` / ``by_difficulty`` on top.
    """

    suite: str
    judge: str = "deterministic"
    per_query: list[dict[str, Any]] = field(default_factory=list)
    overall: dict[str, float] = field(default_factory=dict)
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    metric_keys: tuple[str, ...] = ()
    n_items: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)
    judge_errors: list[dict[str, str]] = field(default_factory=list)

    def metric(self, key: str) -> float:
        """Read an overall metric (0.0 if absent)."""
        return float(self.overall.get(key, 0.0))


def aggregate(
    rows: Sequence[dict[str, Any]], keys: Sequence[str]
) -> dict[str, float]:
    """Mean of each key across *rows*.  Empty input → all zeros.

    Keys absent from every row read as 0.0, so a failed item (recorded as an
    all-zero row) and a metric that no row produced both count against the
    denominator instead of inflating the aggregate.
    """
    if not rows:
        return {k: 0.0 for k in keys}
    return {
        k: sum(float(r.get(k, 0.0)) for r in rows) / len(rows)
        for k in keys
    }


def finish(
    suite: str,
    judge: str,
    rows: list[dict[str, Any]],
    errors: list[dict[str, str]],
    judge_errors: list[dict[str, str]],
    keys: Sequence[str],
    categories: Sequence[str],
) -> EvalResult:
    """Roll up *rows* into an :class:`EvalResult`.

    ``overall`` averages every row; ``by_category`` emits every registered
    category (empty buckets stay present as all-zero rows so the report's
    category table has stable columns).
    """
    return EvalResult(
        suite=suite,
        judge=judge,
        per_query=rows,
        overall=aggregate(rows, keys),
        by_category={
            cat: aggregate([r for r in rows if r.get("category") == cat], keys)
            for cat in categories
        },
        metric_keys=tuple(keys),
        n_items=len(rows),
        errors=errors,
        judge_errors=judge_errors,
    )


def zero_judge_keys(row: dict[str, Any]) -> None:
    """Zero the judge-owned metric keys in *row* (in place).

    The shared answer / e2e / task judge-failure policy: a degraded judgment
    must not be re-graded by the deterministic substring channel under the
    same gate keys — that silently swaps semantics and lets a
    ``--min-groundedness`` gate pass on substrings.  Zeroing makes the gate
    fail loudly; the deterministic values stay under their ``det_*`` keys for
    cross-checking.  The extraction suite does NOT use this — it leaves its
    summary keys unset instead (see ``tests.eval.llm_runner.run_extraction``).
    """
    row.update({k: 0.0 for k in ANSWER_JUDGE_METRIC_KEYS})


# ── JSON serialization (judge-based reports) ─────────────────────────
# The LLM behaviour and task reports serialize identically; the retrieval
# report keeps its own layout because its result carries config/by_difficulty.


def result_to_json_dict(result: EvalResult) -> dict[str, Any]:
    """Serialize one result for the judge-based report JSON."""
    return {
        "suite": result.suite,
        "judge": result.judge,
        "metric_keys": list(result.metric_keys),
        "n_items": result.n_items,
        "overall": result.overall,
        "by_category": result.by_category,
        "n_errors": len(result.errors),
        "n_judge_errors": len(result.judge_errors),
        "errors": result.errors,
        "judge_errors": result.judge_errors,
        "per_query": result.per_query,
    }


def to_json(results: Sequence[EvalResult]) -> str:
    """Serialize a list of results to the judge-based report JSON string."""
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "results": [result_to_json_dict(r) for r in results],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def write_text(path: str, content: str) -> str:
    """Write *content* to *path* (UTF-8).  Returns the path."""
    from pathlib import Path

    p = Path(path)
    p.write_text(content, encoding="utf-8")
    return str(p)


def load_jsonl_items(path: str, cls: type[Any]) -> list[Any]:
    """Load labeled items from a JSONL file under ``tests/eval/data``.

    ``path`` is a filename (e.g. ``"ground_truth.jsonl"``) resolved against
    ``tests/eval/data``.  ``cls`` is the item type: a frozen dataclass
    (constructed as ``cls(**row)``) or a ``dict`` subclass like
    ``GroundTruthItem`` (constructed from the row directly).  Row order is
    preserved so ``--sample N`` and the report keep their stable display
    order.
    """
    from pathlib import Path

    p = Path(__file__).parent / "data" / path
    rows: list[dict[str, Any]] = []
    with open(p, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if isinstance(cls, type) and issubclass(cls, dict):
        return [cls(row) for row in rows]  # type: ignore[arg-type]
    return [cls(**row) for row in rows]


# ── Markdown table helpers (shared by llm_report / task_report) ──────


def fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return "—"
    return f"{v:.{digits}f}"


def overall_table(
    results: Sequence[EvalResult],
    suite_title: Callable[[str], str],
) -> str:
    """One section per result: ``### {title}`` + an overall-metric row.

    ``suite_title`` maps a result's suite to its display title — the per-suite
    title map for the LLM behaviour report, a constant for single-suite
    reports.
    """
    sections: list[str] = []
    for r in results:
        title = suite_title(r.suite)
        judge_note = "" if r.judge == "deterministic" else " (LLM judge)"
        sections.append(f"### {title}{judge_note}")
        sections.append("")
        cols = list(r.metric_keys)
        header = "| metric | " + " | ".join(cols) + " |"
        sep = "|---" * (len(cols) + 1) + "|"
        lines = [header, sep]
        cells = [fmt(r.overall.get(k, 0.0)) for k in cols]
        lines.append("| **overall** | " + " | ".join(cells) + " |")
        sections.append("\n".join(lines))
        sections.append("")
    return "\n".join(sections)


def category_table(
    result: EvalResult,
    headline_exclude: Sequence[str] = (),
) -> str:
    """Category × headline-metric table for one result.

    ``headline_exclude`` drops auxiliary columns (answer_len, n_steps, …) so
    the table stays narrow.
    """
    cats = list(result.by_category)
    if not cats:
        return "_(no category data)_"
    cols = list(result.metric_keys)
    headline = [k for k in cols if k not in headline_exclude]
    header = "| category | " + " | ".join(headline) + " |"
    sep = "|---" * (len(headline) + 1) + "|"
    lines = [header, sep]
    for cat in cats:
        agg = result.by_category.get(cat, {})
        cells = [fmt(agg.get(k, 0.0)) for k in headline]
        lines.append(f"| {cat} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
