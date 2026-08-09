"""Orchestration for the task-level end-to-end eval — one runner.

Mirrors ``tests.eval.llm_runner.run_e2e``: each task in the labeled set is
executed through an (injectable) executor, per-task metrics come from the
pure functions in ``task_metrics``, and everything rolls up into a
:class:`TaskEvalResult`.

Design notes (same policy as the LLM behavior eval):

- The runner accepts *any* executor callable (``TaskRunner``), so unit tests
  inject fakes and the real agent graph is only hit through
  ``tests.eval.task_executors.make_task_runner``.
- A failed task is recorded in ``errors`` AND contributes an all-zero row —
  the "failed queries count toward the denominator" policy, so an outage
  reads as 0-quality instead of quietly dropping tasks.
- ``judge="llm"`` runs the LLM-as-judge pass over the final answer against
  the context the agent actually saw (the tool displays).  A judge failure
  marks the row ``judge_error`` and zeroes the judge-owned metric keys
  (``fact_coverage`` / ``groundedness`` / ``hallucination_rate``) rather than
  substituting the deterministic channel under the same names — the CI gate
  on ``--min-groundedness`` must fail loudly, not silently grade substring
  matches.  The deterministic values stay under ``det_*``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from tests.eval.llm_judge import judge_answer
from tests.eval.llm_metrics import (
    answer_deterministic_metrics,
    answer_judge_metrics,
    citation_presence,
)
from tests.eval.llm_runner import ANSWER_JUDGE_METRIC_KEYS
from tests.eval.task_executors import TaskOutcome, make_task_runner
from tests.eval.task_ground_truth import (
    TASK_CATEGORIES,
    TaskItem,
    load_task_items,
)
from tests.eval.task_metrics import TASK_METRIC_KEYS, task_completion_metrics

# query → TaskOutcome (trajectory + answer + what the model saw).
TaskExecutor = Callable[[str], Awaitable[Any]]


@dataclass
class TaskEvalResult:
    """Aggregate result of the task suite over its labeled set."""

    suite: str  # "task"
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
    rows: list[dict[str, Any]],
    errors: list[dict[str, str]],
    judge_errors: list[dict[str, str]],
    keys: Sequence[str],
    judge: str,
) -> TaskEvalResult:
    by_category: dict[str, dict[str, float]] = {
        cat: _aggregate([r for r in rows if r.get("category") == cat], keys)
        for cat in TASK_CATEGORIES
    }
    return TaskEvalResult(
        suite="task",
        judge=judge,
        per_query=rows,
        overall=_aggregate(rows, keys),
        by_category=by_category,
        metric_keys=tuple(keys),
        n_items=len(rows),
        errors=errors,
        judge_errors=judge_errors,
    )


async def run_tasks(
    items: Sequence[TaskItem] | None = None,
    executor: TaskExecutor | None = None,
    *,
    judge: str = "llm",
    max_steps: int | None = None,
) -> TaskEvalResult:
    """Run every task through the agent and aggregate the metrics.

    Args:
        items: the labeled task set (default: ``load_task_items()``).
        executor: a ``TaskRunner`` (default: ``make_task_runner()`` — the real
            agent graph with auto-approved HITL).  Tests inject fakes.
        judge: ``"llm"`` (default) LLM-judges answer groundedness against the
            tool context the agent saw; ``"deterministic"`` uses substring
            matching only (cheaper, judge-free).
        max_steps: forwarded to the default executor's ReAct loop budget
            (ignored when ``executor`` is injected).
    """
    items = list(items) if items is not None else load_task_items()
    if executor is None:
        executor = make_task_runner(max_steps=max_steps)
    run = executor
    judge_mode = judge if judge in ("llm", "deterministic") else "deterministic"

    # Metric keys stay the same in both judge modes (the answer / e2e
    # convention): ``fact_coverage`` / ``groundedness`` / ``hallucination_rate``
    # come from the LLM judge when ``judge="llm"`` and from substring matching
    # when ``judge="deterministic"`` — a deterministic run is a cheaper, lower-
    # fidelity version of the same keys, not a different metric.  The
    # deterministic values are always recorded under ``det_*`` for cross-check.
    keys = list(TASK_METRIC_KEYS)

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
            outcome: TaskOutcome = await run(it.query)
            row["n_steps"] = outcome.n_steps
            row["called"] = ", ".join(
                str(c.get("name", "")) for c in outcome.tool_calls
            ) or "-"
            row["n_calls"] = len(outcome.tool_calls)

            called = [str(c.get("name", "")) for c in outcome.tool_calls]
            row.update(
                task_completion_metrics(
                    called,
                    it.expected_tools,
                    outcome.answer,
                    allowed=it.allowed_tools,
                    forbidden=it.forbidden_tools,
                    within_budget=outcome.within_budget,
                    had_error=outcome.had_error,
                )
            )

            row["answer_len"] = len(outcome.answer)
            row["answer_preview"] = outcome.answer[:120]
            # Traceability: did the answer cite a source id the agent actually
            # saw?  Empty source_ids (no retrieval tool ran) ⇒ vacuous 1.0.
            row["citation_rate"] = citation_presence(outcome.answer, outcome.source_ids)

            det = answer_deterministic_metrics(
                outcome.answer, it.required_facts, it.prohibited_claims
            )
            row.update({f"det_{k}": v for k, v in det.items()})

            if judge_mode == "llm":
                try:
                    verdict = await judge_answer(
                        it.query,
                        outcome.context_text,
                        outcome.answer,
                        it.required_facts,
                    )
                    row.update(answer_judge_metrics(verdict, it.required_facts))
                    row["ungrounded_claims"] = list(verdict.get("ungrounded_claims") or [])
                except Exception as exc:
                    judge_errors.append({"id": it.id, "error": str(exc)})
                    row["judge_error"] = str(exc)
                    # Judge failure must NOT reuse the deterministic channel
                    # under the judge metric keys (different semantic) — zero
                    # them so --min-fact-coverage / --min-groundedness fail
                    # loudly; the deterministic values stay under det_*.
                    row.update({k: 0.0 for k in ANSWER_JUDGE_METRIC_KEYS})
            else:
                row.update(det)
        except Exception as exc:
            errors.append({"id": it.id, "error": str(exc)})
            row["error"] = str(exc)
            row.update({k: 0.0 for k in keys})
        rows.append(row)

    return _finish(rows, errors, judge_errors, keys, judge_mode)


# ── Default executor (lazy import keeps `--validate-only` light) ───────


def make_default_task_executor() -> TaskExecutor:
    return make_task_runner()
