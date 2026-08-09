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
  judgement in the extraction suite).  A judge failure marks the row
  ``judge_error`` and is counted in ``judge_errors`` — distinct from ``errors``
  so a degraded *judgment* and a failed *execution* stay distinguishable.
  In the answer / e2e suites a judge failure *zeroes* the judge-owned metric
  keys (``fact_coverage`` / ``groundedness`` / ``hallucination_rate``) instead
  of substituting the deterministic channel under the same names, so the CI
  gate on ``--min-groundedness`` etc. fails loudly rather than silently
  grading substring matches; the deterministic values stay available under
  ``det_*`` for cross-checking.  In the extraction suite a failed judgment
  leaves the summary keys unset (aggregated as 0.0) rather than writing a
  fake 0.0 verdict.
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
    E2EItem,
    ExtractionItem,
    ToolSelectionItem,
    load_answer_items,
    load_e2e_items,
    load_extraction_items,
    load_tool_selection_items,
)
from tests.eval.llm_judge import judge_answer, judge_summary
from tests.eval.llm_metrics import (
    ANSWER_METRIC_KEYS,
    E2E_METRIC_KEYS,
    EXTRACTION_METRIC_KEYS,
    TOOL_SELECTION_METRIC_KEYS,
    answer_deterministic_metrics,
    answer_judge_metrics,
    citation_presence,
    context_recall,
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
# E2E runner: query → E2EOutcome (answer + what the model saw).
E2ERunner = Callable[[str], Any]

# The final-answer metrics the LLM judge produces (answer + e2e suites).  On
# a judge failure these are zeroed explicitly rather than re-fed from the
# deterministic channel (``answer_deterministic_metrics``), which returns the
# *same key names* with a *different semantic* — substring matching.  Reusing
# them would make ``--min-groundedness`` silently grade substrings.  The
# deterministic values remain available under ``det_*`` keys.
ANSWER_JUDGE_METRIC_KEYS: tuple[str, ...] = (
    "fact_coverage",
    "groundedness",
    "hallucination_rate",
)


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
                    # Do NOT write summary_faithfulness / summary_completeness
                    # = 0.0: a failed judgment is not evidence the summary was
                    # bad, and a silent 0.0 would drag the mean with no gate to
                    # catch it.  The row stays marked judge_error; the aggregate
                    # counts the missing key as 0.0 (denominator policy), so a
                    # judge outage still reads as degraded, not perfect.
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
                    # A judge failure must NOT reuse the deterministic channel
                    # under the judge metric keys — that swaps a different
                    # semantic (substring matching) into the same gate key.
                    # Zero the judge-owned keys so --min-fact-coverage /
                    # --min-groundedness fail loudly; the deterministic values
                    # stay under det_* for cross-checking.
                    row.update({k: 0.0 for k in ANSWER_JUDGE_METRIC_KEYS})
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


# ── End-to-end (E2E) ────────────────────────────────────────────────


async def run_e2e(
    items: Sequence[E2EItem] | None = None,
    runner: E2ERunner | None = None,
    *,
    judge: str = "llm",
    top_k: int = 5,
    retrieval_mode: str = "memory",
) -> LlmEvalResult:
    """Measure the full chain: real retrieval → answer → groundedness.

    Unlike ``run_answer`` (golden context), each item's context comes from
    real retrieval of the seeded corpus.  Two distinct signals per item:

    - ``context_recall`` — the retrieval-side bound: did the model's context
      contain each required fact?  Deterministic, always measured.  A low
      value pins the blame on retrieval (the answer can't cover what it
      never saw).
    - answer metrics — fact coverage / groundedness, deterministic and/or
      LLM-judged *against the retrieved context* (the faithful production
      grounding contract).  Citation is checked against the source ids
      retrieval actually surfaced, not golden ids.

    ``runner``, when injected, answers every item.  Otherwise each item is
    dispatched to a runner built for its ``retrieval_mode`` (memory →
    ``query_memories``, chunk → ``retrieve_hybrid``); ``retrieval_mode``
    remains the fallback for items without an explicit mode.
    A failed item is recorded in ``errors`` and counts as an all-zero row.
    A judge failure zeroes the judge-owned metric keys (``fact_coverage`` /
    ``groundedness`` / ``hallucination_rate``) and is counted in
    ``judge_errors`` — see the module docstring.
    """
    items = list(items) if items is not None else load_e2e_items()
    judge_mode = judge if judge in ("llm", "deterministic") else "deterministic"
    keys = E2E_METRIC_KEYS

    # Per-item runner dispatch.  An injected ``runner`` answers every item;
    # otherwise each item runs through a runner built for its
    # ``retrieval_mode`` (memory → query_memories, chunk → retrieve_hybrid),
    # cached one per mode.  A single module-level runner would route the whole
    # mixed set through one retrieval path — e.g. the default memory mode
    # silently zeroing every chunk item's context_recall, which is exactly the
    # CI e2e-eval failure this dispatch fixes.
    runners: dict[str, E2ERunner] = {}

    def _runner_for(item: E2EItem) -> E2ERunner:
        if runner is not None:
            return runner
        mode = item.retrieval_mode or retrieval_mode
        r = runners.get(mode)
        if r is None:
            r = make_default_e2e_runner(top_k=top_k, retrieval_mode=mode)
            runners[mode] = r
        return r

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
            outcome = await _runner_for(it)(it.query)
            row["n_retrieved"] = outcome.n_retrieved
            row["retrieved_preview"] = (
                ", ".join(str(s)[:8] for s in outcome.retrieved_source_ids[:5]) or "-"
            )
            # Retrieval-side bound (deterministic): what did the model see?
            row["context_recall"] = context_recall(
                it.required_facts, outcome.context_text
            )
            # Citation against the ids retrieval actually surfaced.
            row["citation_rate"] = citation_presence(
                outcome.answer, outcome.retrieved_source_ids
            )

            row["answer_len"] = len(outcome.answer)
            row["answer_preview"] = outcome.answer[:120]

            det = answer_deterministic_metrics(
                outcome.answer, it.required_facts, it.prohibited_claims
            )
            row.update({f"det_{k}": v for k, v in det.items()})

            if judge_mode == "llm":
                try:
                    verdict = await judge_answer(
                        it.query, outcome.context_text, outcome.answer,
                        it.required_facts,
                    )
                    row.update(answer_judge_metrics(verdict, it.required_facts))
                    row["ungrounded_claims"] = list(verdict.get("ungrounded_claims") or [])
                except Exception as exc:
                    judge_errors.append({"id": it.id, "error": str(exc)})
                    row["judge_error"] = str(exc)
                    # Same as run_answer: zero the judge-owned metric keys
                    # instead of re-feeding the deterministic channel into the
                    # gate keys (substring semantics under an LLM-judge name).
                    row.update({k: 0.0 for k in ANSWER_JUDGE_METRIC_KEYS})
            else:
                row.update(det)
        except Exception as exc:
            errors.append({"id": it.id, "error": str(exc)})
            row["error"] = str(exc)
            row.update({k: 0.0 for k in keys})
        rows.append(row)

    return _finish(
        "e2e", judge_mode, rows, errors, judge_errors,
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


def make_default_e2e_runner(
    *, top_k: int = 5, retrieval_mode: str = "memory"
) -> E2ERunner:
    from tests.eval.llm_executors import make_e2e_runner

    return make_e2e_runner(top_k=top_k, retrieval_mode=retrieval_mode)
