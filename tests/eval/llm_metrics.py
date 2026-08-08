"""Metrics for the LLM behavior eval — pure functions, no I/O.

Mirrors ``tests.eval.metrics`` (pure, degenerate inputs → 0.0, never raise)
but for the three LLM-behavior suites:

- **tool_selection** — binary "called the right tool(s)?" plus partial-credit
  recall over the expected set, an unexpected-call flag, a no-call flag, and
  an optional argument-match check.
- **extraction** — entity / relation precision / recall / F1 with tolerant
  name matching (normalized equality or one-name-contains-the-other), plus
  entity type accuracy and summary keyword coverage.
- **answer** — fact coverage and groundedness, computed either
  deterministically (substring) or from a judge's structured verdict.

Name matching is *containment-tolerant* on purpose: extraction often returns
``"BGE-M3 模型"`` for a golden ``"BGE-M3"``.  Containment only fires when the
shorter name has ≥4 normalized chars so ``"CI"`` never matches
``"CI 构建"``.
"""

from __future__ import annotations

import json
import re
from typing import Any


# ── Shared helpers ──────────────────────────────────────────────────


def normalize_name(name: str) -> str:
    """Lowercase + strip all whitespace."""
    return re.sub(r"\s+", "", str(name).strip().lower())


def names_match(a: str, b: str, min_contain: int = 4) -> bool:
    """True when two entity names refer to the same thing.

    Normalized equality, or one contains the other *and* the shorter side is
    at least ``min_contain`` chars (guards ``"CI"`` ⊂ ``"CI 构建"``).
    """
    na, nb = normalize_name(a), normalize_name(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if len(na) >= min_contain and na in nb:
        return True
    if len(nb) >= min_contain and nb in na:
        return True
    return False


# ── Tool selection ──────────────────────────────────────────────────


def tool_selection_metrics(
    called: list[str],
    expected: list[str],
    *,
    allowed: list[str] | None = None,
    forbidden: list[str] | None = None,
) -> dict[str, float]:
    """Score one tool-selection turn.

    Args:
        called: tool names the model actually called (in order).
        expected: tools that must be called.  Empty ⇒ the model must call
            nothing (a "refrain" item).
        allowed: acceptable substitutes — calls here are not wrong but do not
            satisfy the expected set.
        forbidden: tools that must never be called.

    Returns a dict of 0/1-ish scores:
        tool_accuracy: 1.0 iff the call set is exactly right (expected all
            present, nothing unexpected or forbidden) — or, for an empty
            expected set, iff nothing was called.
        expected_recall: |expected ∩ called| / |expected| (partial credit).
        unexpected_rate: 1.0 if any call fell outside expected ∪ allowed.
        no_call: 1.0 if expected is non-empty and nothing was called.
        arg_match_rate: fraction of ``expected_args`` constraints satisfied
            (see :func:`tool_arg_match_rate`).  Callers pass this in; when no
            args were checked the value is 1.0.
    """
    called_set = set(called)
    exp = set(expected)
    allow = set(allowed or [])
    forb = set(forbidden or [])

    if not exp:
        accuracy = 1.0 if not called_set else 0.0
        expected_recall = 1.0
        unexpected = called_set - allow
        no_call = 0.0
    else:
        unexpected = called_set - exp - allow
        any_forbidden = bool(forb & called_set)
        accuracy = (
            1.0
            if exp.issubset(called_set) and not unexpected and not any_forbidden
            else 0.0
        )
        expected_recall = len(exp & called_set) / len(exp)
        no_call = 1.0 if not called_set else 0.0

    return {
        "tool_accuracy": float(accuracy),
        "expected_recall": float(expected_recall),
        "unexpected_rate": float(1.0 if unexpected else 0.0),
        "no_call": float(no_call),
        "arg_match_rate": 1.0,  # replaced by the caller when args were checked
    }


def tool_arg_match_rate(
    calls: list[dict[str, Any]], expected_args: dict[str, list[str]]
) -> float:
    """Fraction of arg constraints satisfied across the turn's tool calls.

    ``expected_args`` maps a tool name → substrings that must appear in that
    call's JSON-serialized arguments.  A missing call for a constrained tool
    counts as a failed constraint.  No constraints ⇒ 1.0.
    """
    if not expected_args:
        return 1.0
    total = 0
    matched = 0
    for tname, subs in expected_args.items():
        call = next((c for c in calls if str(c.get("name", "")) == tname), None)
        total += len(subs)
        if call is None:
            continue  # a missing call fails every substring constraint
        args_json = json.dumps(call.get("args") or {}, ensure_ascii=False)
        matched += sum(1 for s in subs if s in args_json)
    return matched / total if total else 1.0


# ── Extraction ──────────────────────────────────────────────────────


def _match_pairs(
    predicted: list[dict[str, Any]],
    expected: list[dict[str, Any]],
    *,
    match: Any,
) -> tuple[int, int, int]:
    """Greedy one-to-one matching; returns (tp, fp, fn)."""
    tp = 0
    remaining = list(expected)
    fp = 0
    for p in predicted:
        idx = next(
            (i for i, e in enumerate(remaining) if match(p, e)),
            None,
        )
        if idx is None:
            fp += 1
        else:
            tp += 1
            remaining.pop(idx)
    fn = len(remaining)
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def entity_metrics(
    predicted: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> dict[str, float]:
    """Entity precision / recall / F1 plus type accuracy.

    Names match via :func:`names_match` (containment-tolerant); a predicted
    entity's ``type`` is only compared for entities that matched a golden one.
    """
    if not expected:
        return {
            "entity_precision": 0.0,
            "entity_recall": 0.0,
            "entity_f1": 0.0,
            "entity_type_accuracy": 0.0,
        }

    def _match(p: dict[str, Any], e: dict[str, Any]) -> bool:
        return names_match(str(p.get("name", "")), str(e.get("name", "")))

    tp, fp, fn = _match_pairs(predicted, expected, match=_match)
    precision, recall, f1 = _prf(tp, fp, fn)

    # Type accuracy over the matched pairs — recompute the pairing to recover
    # the (predicted, expected) pairs.
    type_correct = 0
    matched_n = 0
    remaining = list(expected)
    for p in predicted:
        idx = next(
            (i for i, e in enumerate(remaining) if _match(p, e)), None
        )
        if idx is not None:
            matched_n += 1
            if str(p.get("type", "")) == str(remaining[idx].get("type", "")):
                type_correct += 1
            remaining.pop(idx)
    type_acc = type_correct / matched_n if matched_n else 0.0

    return {
        "entity_precision": float(precision),
        "entity_recall": float(recall),
        "entity_f1": float(f1),
        "entity_type_accuracy": float(type_acc),
    }


def relation_metrics(
    predicted: list[dict[str, Any]], expected: list[dict[str, Any]]
) -> dict[str, float]:
    """Relation precision / recall / F1.

    A relation matches when all of ``from`` (name), ``to`` (name) and
    ``type`` match.  Relation types are compared exactly (the extraction
    prompt restricts to an enum, so no tolerance is warranted).
    """
    if not expected:
        return {
            "relation_precision": 0.0,
            "relation_recall": 0.0,
            "relation_f1": 0.0,
        }

    def _match(p: dict[str, Any], e: dict[str, Any]) -> bool:
        return (
            str(p.get("type", "")) == str(e.get("type", ""))
            and names_match(str(p.get("from", "")), str(e.get("from", "")))
            and names_match(str(p.get("to", "")), str(e.get("to", "")))
        )

    tp, fp, fn = _match_pairs(predicted, expected, match=_match)
    precision, recall, f1 = _prf(tp, fp, fn)
    return {
        "relation_precision": float(precision),
        "relation_recall": float(recall),
        "relation_f1": float(f1),
    }


def summary_keyword_coverage(summary: str, keywords: list[str]) -> float:
    """Fraction of golden keywords that appear in the extracted summary.

    Cheap deterministic proxy for summary quality (the LLM judge supplies
    faithfulness / completeness when ``--judge=llm``).
    """
    if not keywords:
        return 1.0
    text = str(summary or "")
    return sum(1 for k in keywords if k in text) / len(keywords)


# ── Final answer ────────────────────────────────────────────────────


def answer_deterministic_metrics(
    answer: str,
    required_facts: list[str],
    prohibited_claims: list[str] | None = None,
) -> dict[str, float]:
    """Substring-based fact coverage and groundedness.

    groundedness is 0.0 when any prohibited claim appears in the answer —
    a string the model must *not* say (usually a tempting hallucination).
    """
    text = str(answer or "")
    if required_facts:
        coverage = sum(1 for f in required_facts if f in text) / len(required_facts)
    else:
        coverage = 1.0
    grounded = 1.0 if not any(p in text for p in (prohibited_claims or [])) else 0.0
    return {
        "fact_coverage": float(coverage),
        "groundedness": float(grounded),
        "hallucination_rate": float(1.0 - grounded),
    }


def answer_judge_metrics(
    verdict: dict[str, Any], required_facts: list[str]
) -> dict[str, float]:
    """Derive metrics from an LLM judge verdict (see ``llm_judge``).

    ``verdict`` carries ``covered_facts`` (verbatim entries of
    ``required_facts`` the judge found in the answer), ``grounded`` (bool)
    and ``ungrounded_claims`` (list).
    """
    covered = set(verdict.get("covered_facts") or [])
    if required_facts:
        coverage = sum(1 for f in required_facts if f in covered) / len(required_facts)
    else:
        coverage = 1.0
    grounded = 1.0 if verdict.get("grounded") else 0.0
    n_ungrounded = len(verdict.get("ungrounded_claims") or [])
    return {
        "fact_coverage": float(coverage),
        "groundedness": float(grounded),
        "hallucination_rate": float(1.0 if n_ungrounded else 0.0),
        "ungrounded_claims": float(n_ungrounded),
    }


# Source IDs are matched tolerantly: the model may cite the full id as shown,
# only its 8-char tail, or drop a ``mem-``/``doc-`` prefix.  An opaque short id
# is never present by accident, so a match is a strong signal of an actual
# citation.
_CITATION_PREFIXES = ("mem-", "doc-", "记忆", "文档")


def citation_presence(answer: str, source_ids: list[str]) -> float:
    """1.0 iff *answer* cites at least one of *source_ids* inline.

    Traceability check for the final-answer suite: when the golden context
    carries source ids (the answer suite always provides them), a grounded
    answer should reference one.  Empty ``source_ids`` ⇒ 1.0 (nothing to
    cite, vacuously compliant).
    """
    if not source_ids:
        return 1.0
    text = str(answer or "")
    for sid in source_ids:
        sid = str(sid).strip()
        if not sid:
            continue
        variants = {sid}
        if len(sid) > 8:
            variants.add(sid[-8:])
        for prefix in _CITATION_PREFIXES:
            if sid.startswith(prefix):
                variants.add(sid[len(prefix):])
        if any(v and v in text for v in variants):
            return 1.0
    return 0.0


# ── Per-suite metric key order (report column order) ────────────────

TOOL_SELECTION_METRIC_KEYS: tuple[str, ...] = (
    "tool_accuracy",
    "expected_recall",
    "unexpected_rate",
    "no_call",
    "arg_match_rate",
)

EXTRACTION_METRIC_KEYS: tuple[str, ...] = (
    "entity_precision",
    "entity_recall",
    "entity_f1",
    "entity_type_accuracy",
    "relation_precision",
    "relation_recall",
    "relation_f1",
    "summary_coverage",
    # present only when the summary judge is enabled:
    "summary_faithfulness",
    "summary_completeness",
)

ANSWER_METRIC_KEYS: tuple[str, ...] = (
    "fact_coverage",
    "groundedness",
    "hallucination_rate",
    "citation_rate",
)
