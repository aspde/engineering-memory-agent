"""Runners for the memory write-path eval — one per suite.

Same contract as ``evals.llm_runner`` (shared ``evals.core`` aggregation,
executor injection, failed item → error + all-zero row), for the three
write-path suites:

- ``run_write_conflict`` — drives the real conflict-detection call and scores
  it against the labeled contradiction verdicts.  Deterministic only: the
  ground truth is a boolean, so there is nothing for an LLM judge to add.
- ``run_write_merge`` — drives the real merge-prompt call; deterministic
  keyword coverage is always recorded, and the LLM judge grades
  faithfulness/completeness when ``judge="llm"``.  A judge failure marks the
  row ``judge_error`` and leaves the judge keys unset (extraction-suite
  policy: a failed judgment is not evidence the merge was bad).
- ``run_auto_gate`` — drives the raising gate core and scores worthy/not as
  binary classification.  Deterministic only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evals.core import EvalResult, finish
from evals.llm_ground_truth import (
    AUTO_GATE_CATEGORIES,
    WRITE_CONFLICT_CATEGORIES,
    WRITE_MERGE_CATEGORIES,
    AutoGateItem,
    WriteConflictItem,
    WriteMergeItem,
    load_auto_gate_items,
    load_write_conflict_items,
    load_write_merge_items,
)
from evals.write_eval_executors import (
    ConflictDetector,
    GateChecker,
    MergeSummarizer,
    make_conflict_detector,
    make_gate_checker,
    make_merge_summarizer,
)
from evals.write_eval_metrics import (
    MERGE_JUDGE_METRIC_KEYS,
    conflict_cell_metrics,
    derive_binary_prf,
    gate_cell_metrics,
    merge_fact_coverage,
    merge_judge_metrics,
)

LlmEvalResult = EvalResult

# Metric keys per suite.  Binary suites record per-item confusion-cell
# indicators; precision/recall/F1/accuracy are derived per aggregation bucket
# (overall / category) from those means — see write_eval_metrics.  The merge
# suite's judge keys appear only when --judge llm is active (same pattern as
# extraction's summary keys).
WRITE_CONFLICT_CELL_KEYS: tuple[str, ...] = (
    "conflict_tp",
    "conflict_fp",
    "conflict_fn",
    "conflict_tn",
)
WRITE_CONFLICT_METRIC_KEYS: tuple[str, ...] = WRITE_CONFLICT_CELL_KEYS + (
    "conflict_precision",
    "conflict_recall",
    "conflict_f1",
    "conflict_accuracy",
    "conflict_false_positive_rate",
    "conflict_false_negative_rate",
)
WRITE_MERGE_METRIC_KEYS: tuple[str, ...] = (
    "merge_fact_coverage",
) + MERGE_JUDGE_METRIC_KEYS
AUTO_GATE_CELL_KEYS: tuple[str, ...] = (
    "worthy_tp",
    "worthy_fp",
    "worthy_fn",
    "worthy_tn",
)
AUTO_GATE_METRIC_KEYS: tuple[str, ...] = AUTO_GATE_CELL_KEYS + (
    "worthy_precision",
    "worthy_recall",
    "worthy_f1",
    "worthy_accuracy",
    "worthy_false_positive_rate",
    "worthy_false_negative_rate",
)


def _derive_binary(result: EvalResult, cell_keys: tuple[str, ...], pos_key: str) -> EvalResult:
    """Derive P/R/F1/accuracy + error rates into *result*'s aggregate maps.

    Runs after ``finish``: the shared aggregation means the 0/1 cell
    indicators; this converts each bucket (overall + every category) into the
    headline metrics via :func:`derive_binary_prf`.  Mutates and returns the
    result.
    """
    result.overall.update(derive_binary_prf(result.overall, pos_key))
    for cat, cells in result.by_category.items():
        cells.update(derive_binary_prf(cells, pos_key))
    # Headline keys a report/gate reads — drop the raw cell counts so tables
    # stay narrow (they remain on per_query rows for forensics).
    result.metric_keys = tuple(
        k for k in (*cell_keys, *(k for k in result.overall if k not in cell_keys))
        if k in result.overall
    )
    return result


async def run_write_conflict(
    items: Sequence[WriteConflictItem] | None = None,
    detector: ConflictDetector | None = None,
) -> LlmEvalResult:
    """Score the conflict gate's contradiction verdicts."""
    items = list(items) if items is not None else load_write_conflict_items()
    detect = detector or make_conflict_detector()

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for it in items:
        row: dict[str, Any] = {
            "id": it.id,
            "category": it.category,
            "existing_preview": it.existing_summary[:80],
            "new_preview": it.new_summary[:80],
        }
        try:
            predicted = bool(await detect(it.existing_summary, it.new_summary))
            row["predicted_conflict"] = predicted
            row.update(conflict_cell_metrics(predicted, it.expected_conflict))
        except Exception as exc:
            errors.append({"id": it.id, "error": str(exc)})
            row["error"] = str(exc)
            row.update({k: 0.0 for k in WRITE_CONFLICT_CELL_KEYS})
        rows.append(row)

    return _derive_binary(
        finish(
            "write_conflict", "deterministic", rows, errors, [],
            WRITE_CONFLICT_METRIC_KEYS, WRITE_CONFLICT_CATEGORIES,
        ),
        WRITE_CONFLICT_CELL_KEYS,
        "conflict",
    )


async def run_write_merge(
    items: Sequence[WriteMergeItem] | None = None,
    summarizer: MergeSummarizer | None = None,
    *,
    judge: str = "llm",
) -> LlmEvalResult:
    """Score merged summaries on fact retention (+ judge when enabled)."""
    items = list(items) if items is not None else load_write_merge_items()
    merge = summarizer or make_merge_summarizer()
    judge_mode = judge if judge in ("llm", "deterministic") else "deterministic"
    keys = list(WRITE_MERGE_METRIC_KEYS)
    if judge_mode != "llm":
        keys = [k for k in keys if k not in MERGE_JUDGE_METRIC_KEYS]

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    judge_errors: list[dict[str, str]] = []

    for it in items:
        row: dict[str, Any] = {
            "id": it.id,
            "category": it.category,
            "existing_preview": it.existing_summary[:80],
            "new_preview": it.new_summary[:80],
        }
        try:
            merged = str(await merge(it.existing_summary, it.new_summary) or "")
            row["merged_len"] = len(merged)
            row["merged_preview"] = merged[:120]
            row["merge_fact_coverage"] = merge_fact_coverage(merged, it.required_facts)

            if judge_mode == "llm":
                from evals.llm_judge import judge_merge

                try:
                    verdict = await judge_merge(
                        it.existing_summary, it.new_summary, merged
                    )
                    row.update(merge_judge_metrics(verdict))
                except Exception as exc:
                    # Extraction-suite policy: a failed judgment leaves the
                    # summary keys unset (aggregated as 0.0 by the denominator
                    # rule) rather than writing a fake 0.0 verdict.
                    judge_errors.append({"id": it.id, "error": str(exc)})
                    row["judge_error"] = str(exc)
        except Exception as exc:
            errors.append({"id": it.id, "error": str(exc)})
            row["error"] = str(exc)
            row.update({k: 0.0 for k in keys})
        rows.append(row)

    return finish(
        "write_merge", judge_mode, rows, errors, judge_errors,
        keys, WRITE_MERGE_CATEGORIES,
    )


async def run_auto_gate(
    items: Sequence[AutoGateItem] | None = None,
    checker: GateChecker | None = None,
) -> LlmEvalResult:
    """Score the auto-memory quality gate as a binary classifier."""
    items = list(items) if items is not None else load_auto_gate_items()
    check = checker or make_gate_checker()

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for it in items:
        row: dict[str, Any] = {
            "id": it.id,
            "category": it.category,
            "content_preview": it.content[:80],
        }
        try:
            predicted = bool(await check(it.content))
            row["predicted_worthy"] = predicted
            row.update(gate_cell_metrics(predicted, it.expected_worthy))
        except Exception as exc:
            errors.append({"id": it.id, "error": str(exc)})
            row["error"] = str(exc)
            row.update({k: 0.0 for k in AUTO_GATE_CELL_KEYS})
        rows.append(row)

    return _derive_binary(
        finish(
            "auto_gate", "deterministic", rows, errors, [],
            AUTO_GATE_METRIC_KEYS, AUTO_GATE_CATEGORIES,
        ),
        AUTO_GATE_CELL_KEYS,
        "worthy",
    )
