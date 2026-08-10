"""Metrics for the task-level end-to-end eval — pure functions, no I/O.

Mirrors ``tests.eval.llm_metrics`` (pure, degenerate inputs → 0.0, never
raise) but for the *task* dimension.  The task eval drives the real agent
graph, so a row is a whole trajectory, not a single decision.  Two distinct
signals per task:

- **completion** — did the run finish the way the task demands?  ``completed``
  is strict (all expected tools called + a substantive answer + no error),
  ``tool_recall`` gives partial credit over the expected set, and
  ``unexpected_rate`` flags calls outside the acceptable set.
- **loop discipline** — ``within_budget``: did the agent reach the final
  answer before ``max_steps`` forced termination?  A task can be completed
  while still being wasteful; the two metrics separate the cases.

Final-answer quality reuses the answer metrics from ``llm_metrics``
(``answer_deterministic_metrics`` / ``answer_judge_metrics`` /
``citation_presence``) so the task report's coverage / groundedness /
citation columns mean the same thing as in the answer and e2e suites.
"""

from __future__ import annotations

from typing import Any

from tests.eval.llm_metrics import answer_deterministic_metrics

# The two apology strings the agent's failure path streams (backend/agent/nodes.py).
# A final answer that is exactly one of these is an error stub, not a
# completed task — even though the graph "ended" normally.
APOLOGY_MARKERS: tuple[str, ...] = (
    "抱歉，当前回答生成失败，请稍后重试。",
    "抱歉，生成回复时出现错误，请稍后重试。",
)

# A substantive answer is more than this many characters of non-whitespace
# (guards against an empty / one-word "answer" counting as completion).
_MIN_ANSWER_CHARS = 8


def is_apology_stub(answer: str) -> bool:
    """True when *answer* is one of the agent's provider-failure stubs."""
    return str(answer or "").strip() in APOLOGY_MARKERS


def task_completion_metrics(
    called: list[str],
    expected: list[str],
    answer: str,
    *,
    allowed: list[str] | None = None,
    forbidden: list[str] | None = None,
    within_budget: bool = True,
    had_error: bool = False,
) -> dict[str, float]:
    """Score one completed task trajectory.

    Args:
        called: tool names the agent actually called, in order.
        expected: tools that must be called.  Empty ⇒ the agent must call
            nothing (a refrain task).
        answer: the final answer text.
        allowed: acceptable substitutes — calls here are not wrong but do not
            satisfy ``expected``.
        forbidden: tools that must never be called.
        within_budget: False when the run was force-terminated by ``max_steps``
            (the route sent it to the final-answer node before it finished on
            its own) or aborted by the per-task wall-clock timeout.  Reported
            separately — a task completed inefficiently is still completed.
        had_error: True when the run hit a provider error / timeout / graph
            exception (the answer, if any, is an apology stub).

    Returns:
        ``completed`` — 1.0 iff the run produced a substantive, non-apology
        answer with no error AND called exactly the expected tools (all
        present, nothing unexpected or forbidden; a refrain task must call
        nothing).
        ``tool_recall`` — |expected ∩ called| / |expected|.
        ``unexpected_rate`` — 1.0 if any call fell outside expected ∪ allowed.
        ``within_budget`` — 1.0 as passed (the loop-discipline signal).
    """
    called_set = set(called)
    exp = set(expected)
    allow = set(allowed or [])
    forb = set(forbidden or [])

    trajectory_ok = exp.issubset(called_set) and not (called_set - exp - allow) and not (
        forb & called_set
    )
    # Refrain task: calling nothing is the correct trajectory.
    if not exp:
        trajectory_ok = not called_set

    # A forbidden call is always unexpected, even when the (contradictory)
    # dataset also lists the tool under allowed — forbidden wins, matching the
    # ``forb & called_set`` completion rule above.
    unexpected = (called_set - exp - allow) | (forb & called_set)

    substantive = (
        not had_error
        and not is_apology_stub(answer)
        and len(str(answer or "").strip()) >= _MIN_ANSWER_CHARS
    )

    if not exp:
        tool_recall = 1.0
    else:
        tool_recall = len(exp & called_set) / len(exp)

    return {
        "completed": float(1.0 if (trajectory_ok and substantive) else 0.0),
        "tool_recall": float(tool_recall),
        "unexpected_rate": float(1.0 if unexpected else 0.0),
        "within_budget": float(1.0 if within_budget else 0.0),
    }


def answer_metrics(
    answer: str,
    required_facts: list[str],
    prohibited_claims: list[str] | None = None,
) -> dict[str, float]:
    """Deterministic final-answer quality — delegates to the shared answer
    metrics so the task report's coverage / groundedness columns match the
    answer and e2e suites.
    """
    return answer_deterministic_metrics(answer, required_facts, prohibited_claims)


# Column order for the task report / aggregate.  ``n_steps`` is informational
# (mean iterations per task), everything else is 0/1 or a fraction.
TASK_METRIC_KEYS: tuple[str, ...] = (
    "completed",
    "tool_recall",
    "unexpected_rate",
    "within_budget",
    "n_steps",
    "fact_coverage",
    "groundedness",
    "hallucination_rate",
    "citation_rate",
)
