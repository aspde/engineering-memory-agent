"""Metrics for the memory write-path eval — pure functions, no I/O.

Follows the ``evals.llm_metrics`` contract (pure, degenerate inputs → 0.0,
never raise) for the three write-path suites:

- **write_conflict** — binary classification of the contradiction call.
- **write_merge** — deterministic keyword coverage over the merged summary
  (``merge_fact_coverage``); the LLM judge supplies faithfulness /
  completeness when ``--judge llm``.
- **auto_gate** — the auto-memory quality gate as a binary classifier.

Binary suites record per-item confusion-cell *indicators* (tp/fp/fn/tn as
0/1) whose means are the cell counts over N; precision / recall / F1 /
accuracy are derived per aggregation bucket from those means via
:func:`derive_binary_prf`.  Averaging per-item precision/recall directly
would be wrong: a true negative's per-item precision is 0/0 → 0.0, and the
mean would punish perfect classification whenever negatives exist.  Deriving
from cell means gives exact micro-P/R/F1 per overall/category bucket.

A false positive in the conflict suite routes a benign supplement to HITL
arbitration; a false negative lets a real contradiction enter the store
unflagged — both directions stay visible as explicit rate columns.
"""

from __future__ import annotations

from typing import Any


def _cell_metrics(predicted_pos: bool, expected_pos: bool, pos_key: str) -> dict[str, float]:
    """Per-item 0/1 confusion-cell indicators (mean over items = count/N)."""
    return {
        f"{pos_key}_tp": 1.0 if predicted_pos and expected_pos else 0.0,
        f"{pos_key}_fp": 1.0 if predicted_pos and not expected_pos else 0.0,
        f"{pos_key}_fn": 1.0 if not predicted_pos and expected_pos else 0.0,
        f"{pos_key}_tn": 1.0 if not predicted_pos and not expected_pos else 0.0,
    }


def conflict_cell_metrics(predicted: bool, expected: bool) -> dict[str, float]:
    """One conflict-detection decision → its confusion-cell indicators."""
    return _cell_metrics(predicted, expected, pos_key="conflict")


def gate_cell_metrics(predicted: bool, expected: bool) -> dict[str, float]:
    """One auto-memory gate decision → its confusion-cell indicators."""
    return _cell_metrics(predicted, expected, pos_key="worthy")


def derive_binary_prf(cells: dict[str, float], pos_key: str) -> dict[str, float]:
    """Derive precision / recall / F1 / accuracy + both error rates from one
    aggregation bucket's confusion-cell means (see module docstring).

    ``cells`` is an ``overall`` or ``by_category[cat]`` mapping produced by
    the shared :func:`evals.core.finish` aggregation.  Missing cells read as
    0.0; empty buckets therefore derive to all-zero metrics (the same
    degenerate convention as ``llm_metrics._prf``).
    """
    tp = float(cells.get(f"{pos_key}_tp", 0.0))
    fp = float(cells.get(f"{pos_key}_fp", 0.0))
    fn = float(cells.get(f"{pos_key}_fn", 0.0))
    tn = float(cells.get(f"{pos_key}_tn", 0.0))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        f"{pos_key}_precision": precision,
        f"{pos_key}_recall": recall,
        f"{pos_key}_f1": f1,
        f"{pos_key}_accuracy": tp + tn,  # (tp + tn) / N — both already /N
        f"{pos_key}_false_positive_rate": fp,
        f"{pos_key}_false_negative_rate": fn,
    }


def merge_fact_coverage(merged_summary: str, required_facts: list[str]) -> float:
    """Fraction of required facts (both sides' key facts) retained in the
    merged summary.

    Deterministic substring check on the raw strings — the same policy as
    ``answer_deterministic_metrics.fact_coverage``, kept under a distinct
    name so the judge-owned faithfulness keys never alias it.
    """
    if not required_facts:
        return 1.0
    text = str(merged_summary or "")
    return sum(1 for f in required_facts if f in text) / len(required_facts)


MERGE_JUDGE_METRIC_KEYS: tuple[str, ...] = (
    "merge_faithfulness",
    "merge_completeness",
)


def merge_judge_metrics(verdict: dict[str, Any]) -> dict[str, float]:
    """Derive merged-summary metrics from an LLM judge verdict.

    The judge grades whether the merge invented content (faithfulness) and
    whether it lost either side's facts (completeness) — the two failure
    modes gap-remediation §8 flags for the merge path.  Same shape as the
    extraction suite's summary verdict.
    """
    return {
        "merge_faithfulness": float(verdict.get("faithfulness", 0.0)),
        "merge_completeness": float(verdict.get("completeness", 0.0)),
    }
