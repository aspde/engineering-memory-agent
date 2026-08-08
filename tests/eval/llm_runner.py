"""Orchestration for the LLM behavior eval — one runner per suite.

Mirrors ``tests.eval.runner``: each ``run_*`` function executes every item
in its labeled set through an executor, computes per-item metrics via the
pure functions in ``llm_metrics``, and rolls up overall + per-category
aggregates into a :class:`LlmEvalResult`.

Design notes:

- Runners accept *any* executor callable, so unit tests inject fakes and the
  real LLM is only hit through the default executors (``llm_executors``).
- A failed item is recorded in ``errors`` AND contributes an all-zero row to
  the aggregate — the same "failed queries count toward the denominator"
  policy the retrieval runner adopted, so an outage reads as 0-quality
  instead of quietly dropping items.
- ``judge="llm"`` enables the LLM-as-judge pass (answer suite always; summary
  judgement in the extraction suite).  A judge failure degrades to the
  deterministic metrics for that item and is counted in ``judge_errors`` —
  distinct from ``errors`` so a degraded *judgment* never fails the CI gate
  the way a failed *execution* does.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from tests.eval.llm_ground_truth import (
    ANSWER_CATEGORIES,
    EXTRACTION_CATEGORIES,
    TOOL_SELECTION_CATEGORIES,
    AnswerItem,
    ExtractionItem,
    ToolSelectionItem,
    load_answer_items,
    load_extraction_items,
    load_tool_selection_items,
)
from tests.eval.llm_judge import judge_answer, judge_summary
from tests.eval.llm_metrics import (
    ANSWER_METRIC_KEYS,
    EXTRACTION_METRIC_KEYS,
    TOOL_SELECTION_METRIC_KEYS,
    answer_deterministic_metrics,
    answer_judge_metrics,
    citation_presence,
    entity_metrics,
    relation_metrics,
    summary_keyword_coverage,
    tool_arg_match_rate,
    tool_selection_metrics,
)

# Executor signatures (see llm_executors for the default implementations).
ToolSelector = Callable[[str], Any]
Extractor = Callable[[str], Any]
AnswerGenerator = Callable[[str, str, list[str] | None], Any]


@dataclass
class LlmEvalResult:
    """Aggregate result of one suite over its labeled set."""

    suite: str  # "tool_selection" | "extraction" | "answer"
    judge: str  # "llm" | "deterministic"
    per_query: list[dict[str, Any]]
    overall: dict[str, float]
    by_category: dict[str, dict[str, float]]
    metric_keys: tuple[str, ...]
    n_items: int
    errors: list[dict[str, str]] = field(default_factory=list)
    judge_errors: list[dict[str, str]] = field(default_factory=list)

    def metric(self, key: str) -> float:
        """Read an overall metric (0.0 if absent)."""
        return float(self.overall.get(key, 0.0))


def _aggregate(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> dict[str, float]:
    """Mean of each key across rows. Empty input → all zeros."""
    if not rows:
        return {k: 0.0 for k in keys}
    return {
        k: sum(float(r.get(k, 0.0)) for r in rows) / len(rows)
        for k in keys
    }


def _finish(
    suite: str,
    judge: str,
    rows: list[dict[str, Any]],
    errors: list[dict[str, str]],
    judge_errors: list[dict[str, str]],
    keys: Sequence[str],
    categories: Sequence[str],
) -> LlmEvalResult:
    by_category: dict[str, dict[str, float]] = {
        cat: _aggregate([r for r in rows if r.get("category") == cat], keys)
        for cat in categories
    }
    return LlmEvalResult(
        suite=suite,
        judge=judge,
        per_query=rows,
        overall=_aggregate(rows, keys),
        by_category=by_category,
        metric_keys=tuple(keys),
        n_items=len(rows),
        errors=errors,
        judge_errors=judge_errors,
    )


# ── Tool selection ──────────────────────────────────────────────────


async def run_tool_selection(
    items: Sequence[ToolSelectionItem] | None = None,
    selector: ToolSelector | None = None,
) -> LlmEvalResult:
    """Measure whether the agent picks the right tool(s) per query."""
    items = list(items) if items is not None else load_tool_selection_items()
    select = selector or make_default_selector()
    keys = TOOL_SELECTION_METRIC_KEYS
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    judge_errors: list[dict[str, str]] = []

    for it in items:
        row: dict[str, Any] = {
            "id": it.id,
            "category": it.category,
            "query": it.query[:100],
        }
        try:
            calls = await select(it.query)
            names = [str(c.get("name", "")) for c in calls]
            metrics = tool_selection_metrics(
                names,
                it.expected_tools,
                allowed=it.allowed_tools,
                forbidden=it.forbidden_tools,
            )
            metrics["arg_match_rate"] = tool_arg_match_rate(calls, it.expected_args)
            row.update(metrics)
            row["n_calls"] = len(names)
            row["called"] = ", ".join(n for n in names if n) or "-"
        except Exception as exc:
            errors.append({"id": it.id, "error": str(exc)})
            row["error"] = str(exc)
            row.update({k: 0.0 for k in keys})
        rows.append(row)

    return _finish(
        "tool_selection", "deterministic", rows, errors, judge_errors,
        keys, TOOL_SELECTION_CATEGORIES,
    )


# ── Extraction ──────────────────────────────────────────────────────


async def run_extraction(
    items: Sequence[ExtractionItem] | None = None,
    extractor: Extractor | None = None,
    *,
    judge: str = "llm",
) -> LlmEvalResult:
    """Measure entity/relation extraction accuracy and summary quality."""
    items = list(items) if items is not None else load_extraction_items()
    extract = extractor or make_default_extractor()
    judge_mode = judge if judge in ("llm", "deterministic") else "deterministic"

    keys = list(EXTRACTION_METRIC_KEYS)
    if judge_mode != "llm":
        keys = [k for k in keys if k not in ("summary_faithfulness", "summary_completeness")]

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    judge_errors: list[dict[str, str]] = []

    for it in items:
        row: dict[str, Any] = {"id": it.id, "category": it.category}
        try:
            result = await extract(it.content)
            entities = list(result.get("entities") or [])
            relations = list(result.get("relations") or [])
            summary = str(result.get("summary") or "")
            row.update(entity_metrics(entities, it.expected_entities))
            row.update(relation_metrics(relations, it.expected_relations))
            row["summary_coverage"] = summary_keyword_coverage(
                summary, it.summary_keywords
            )
            row["n_entities"] = len(entities)
            row["n_relations"] = len(relations)

            if judge_mode == "llm":
                try:
                    verdict = await judge_summary(it.content, summary)
                    row["summary_faithfulness"] = verdict["faithfulness"]
                    row["summary_completeness"] = verdict["completeness"]
                except Exception as exc:
                    judge_errors.append({"id": it.id, "error": str(exc)})
                    row["judge_error"] = str(exc)
                    row["summary_faithfulness"] = 0.0
                    row["summary_completeness"] = 0.0
        except Exception as exc:
            errors.append({"id": it.id, "error": str(exc)})
            row["error"] = str(exc)
            row.update({k: 0.0 for k in keys})
        rows.append(row)

    return _finish(
        "extraction", judge_mode, rows, errors, judge_errors,
        keys, EXTRACTION_CATEGORIES,
    )


# ── Final answer ────────────────────────────────────────────────────


async def run_answer(
    items: Sequence[AnswerItem] | None = None,
    generator: AnswerGenerator | None = None,
    *,
    judge: str = "llm",
) -> LlmEvalResult:
    """Measure final-answer fact coverage and groundedness.

    With ``judge="llm"`` (default) the primary metrics come from the LLM
    judge verdict; deterministic substring metrics are always recorded under
    ``det_*`` keys for cross-checking and as the degradation fallback.
    """
    items = list(items) if items is not None else load_answer_items()
    generate = generator or make_default_answer_generator()
    judge_mode = judge if judge in ("llm", "deterministic") else "deterministic"
    keys = ANSWER_METRIC_KEYS

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    judge_errors: list[dict[str, str]] = []

    for it in items:
        row: dict[str, Any] = {
            "id": it.id,
            "category": it.category,
            "query": it.query[:100],
        }
        try:
            answer = await generate(it.query, it.context, list(it.source_ids) or None)
            row["answer_len"] = len(answer)
            row["answer_preview"] = answer[:120]
            # Traceability is deterministic and always measured: did the answer
            # cite at least one of the golden source ids?
            row["citation_rate"] = citation_presence(answer, it.source_ids)

            det = answer_deterministic_metrics(
                answer, it.required_facts, it.prohibited_claims
            )
            row.update({f"det_{k}": v for k, v in det.items()})

            if judge_mode == "llm":
                try:
                    verdict = await judge_answer(
                        it.query, it.context, answer, it.required_facts
                    )
                    row.update(answer_judge_metrics(verdict, it.required_facts))
                    row["ungrounded_claims"] = list(verdict.get("ungrounded_claims") or [])
                except Exception as exc:
                    judge_errors.append({"id": it.id, "error": str(exc)})
                    row["judge_error"] = str(exc)
                    row.update(det)  # degrade to the deterministic channel
            else:
                row.update(det)
        except Exception as exc:
            errors.append({"id": it.id, "error": str(exc)})
            row["error"] = str(exc)
            row.update({k: 0.0 for k in keys})
        rows.append(row)

    return _finish(
        "answer", judge_mode, rows, errors, judge_errors,
        keys, ANSWER_CATEGORIES,
    )


# ── Default executors (lazy import keeps `--validate-only` light) ───


def make_default_selector() -> ToolSelector:
    from tests.eval.llm_executors import make_tool_selector

    return make_tool_selector()


def make_default_extractor() -> Extractor:
    from tests.eval.llm_executors import make_extractor

    return make_extractor()


def make_default_answer_generator() -> AnswerGenerator:
    from tests.eval.llm_executors import make_answer_generator

    return make_answer_generator()
