"""Metrics for the task-level end-to-end eval — pure functions, no I/O.

Mirrors ``evals.llm_metrics`` (pure, degenerate inputs → 0.0, never
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

from collections.abc import Sequence
from typing import Any

# The two apology strings the agent's failure path streams (backend/agent/nodes.py).
# A final answer that is exactly one of these is an error stub, not a
# completed task — even though the graph "ended" normally.
APOLOGY_MARKERS: tuple[str, ...] = (
    "抱歉，当前回答生成失败，请稍后重试。",
    "抱歉，生成回复时出现错误，请稍后重试。",
)

# Environment classification for one task row.  Provider outages (apology
# stubs, provider exceptions) and wall-clock timeouts pollute the answer
# metrics without saying anything about agent behaviour — the 2026-08-24
# post-discipline runs lost 14/19 failure cells to them.  ``outcome_class``
# records which rows are trustworthy behaviour evidence so the report can
# aggregate ``completed_clean`` over the clean subset only.
OUTCOME_CLASSES: tuple[str, ...] = (
    "ok",              # substantive answer, no provider failure
    "provider_error",  # apology stub / provider exception mid-run
    "timeout",         # wall-clock timeout (AGENT_TIMEOUT) forced an abort
)

# A substantive answer is more than this many characters of non-whitespace
# (guards against an empty / one-word "answer" counting as completion).
_MIN_ANSWER_CHARS = 8


def is_apology_stub(answer: str) -> bool:
    """True when *answer* is one of the agent's provider-failure stubs."""
    return str(answer or "").strip() in APOLOGY_MARKERS


def outcome_class(
    answer: str,
    *,
    had_error: bool,
    error: str = "",
    within_budget: bool = True,
) -> str:
    """Classify one task run's outcome for report-level noise separation.

    Precedence: a wall-clock timeout (``error == "timeout"``, set by the
    task executor) is ``timeout`` even though the run also reports
    ``had_error``; an apology stub or a graph/provider error is
    ``provider_error``; everything else is ``ok``.  ``within_budget`` does
    NOT affect the class — running out of ReAct steps is agent behaviour,
    not environment noise.
    """
    if str(error or "").strip() == "timeout":
        return "timeout"
    if had_error or is_apology_stub(answer):
        return "provider_error"
    return "ok"


def clean_completed_mean(rows: Sequence[dict[str, Any]]) -> float:
    """Mean ``completed`` over the *clean* rows only (``outcome_class=="ok"``).

    The masked-mean companion to :func:`outcome_class`: provider outages and
    timeouts pollute answer metrics without saying anything about agent
    behaviour, so the report carries ``completed_clean`` next to ``completed``
    — the strict score over the environment-trustworthy subset.  The mean
    divides by the number of clean rows (polluted rows are dropped from the
    denominator, not averaged in as zeros — they already score ``completed``
    0.0, so a plain mean over all rows would equal ``completed`` and carry no
    extra signal).  Rows without an ``outcome_class`` key are treated as
    clean (only an explicit pollution mark excludes).  No clean rows (fully
    polluted run) → 0.0; disambiguated by the report's environment-error
    count.
    """
    clean = [r for r in rows if r.get("outcome_class", "ok") == "ok"]
    if not clean:
        return 0.0
    return sum(float(r.get("completed", 0.0)) for r in clean) / len(clean)


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


# Column order for the task report / aggregate.  ``n_steps`` is informational
# (mean iterations per task), everything else is 0/1 or a fraction.
# ``completed_clean`` is the masked mean of ``completed`` over the rows whose
# ``outcome_class`` is ``ok`` (see ``clean_completed_mean``) — the strict
# completion signal isolated from provider outages and wall-clock timeouts.
TASK_METRIC_KEYS: tuple[str, ...] = (
    "completed",
    "completed_clean",
    "tool_recall",
    "unexpected_rate",
    "within_budget",
    "n_steps",
    "fact_coverage",
    "groundedness",
    "hallucination_rate",
    "citation_rate",
)
