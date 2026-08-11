"""Labeled task set for the task-level end-to-end eval — one suite.

The retrieval eval (``tests.eval.ground_truth``) measures which memories a
search returns, and the LLM behavior eval (``tests.eval.llm_ground_truth``)
measures single dimensions of agent behavior (tool choice, extraction,
final answer).  Neither drives the *agent graph*: a query through
``run_llm_eval --suite e2e`` still runs one retrieval call and one answer
call, not the ReAct loop with real tool execution and HITL gates.

This module carries the labeled **tasks** for that missing dimension — the
full agent (``build_agent_graph``) must *complete* a multi-step request:

- **task-001 / 006 / 004** — a single retrieval feeding a grounded answer
  (baseline for the loop working at all);
- **task-002 / 008** — facts that live in *two different stores* (memories
  table + chunks table), so a correct answer structurally requires calling
  both ``search_memories_tool`` and ``retrieve_chunks_tool``;
- **task-003** — retrieve *then persist* (``write_memory_tool``);
- **task-005** — retrieve *then notify* (``notify_feishu_tool``, gated by
  the chat approval set);
- **task-007** — a refrain task: the agent must call nothing.

Each ``required_fact`` must be a substring of one of the e2e seed
``source_content`` strings (``tests.eval.e2e_seed`` seeds exactly those into
the memories / chunks tables), so the fact is *reachable* by the tool that
would surface it.  Validation enforces this — a fact that is not in the seed
corpus can never score.

The eval drive path is ``tests.eval.run_task_eval`` (CLI) with the corpus
seeded by ``python -m tests.eval.e2e_seed --clear`` first.

Write-tasks (task-003) write a real memory row during the run — a documented
side effect, tagged by its content so it is identifiable and removable.

The tasks themselves live in ``tests/eval/data/tasks.jsonl``; this module
keeps the item type, constants, loader and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tests.eval.core import load_jsonl_items


TASK_CATEGORIES: tuple[str, ...] = (
    "factual",
    "multi_retrieve",
    "write",
    "conceptual",
    "notify",
    "no_tool",
)


@dataclass(frozen=True)
class TaskItem:
    """One multi-step task the agent must complete end to end.

    Unlike ``ToolSelectionItem`` (single tool decision) or ``E2EItem``
    (single retrieval → answer), a task asserts the *trajectory*: which
    tools must be called (``expected_tools``), and what the final answer
    must cover (``required_facts``) and must not claim
    (``prohibited_claims``).
    """

    id: str
    query: str
    expected_tools: list[str]  # tools the agent MUST call to complete the task
    category: str
    required_facts: list[str]  # substrings the final answer should contain
    prohibited_claims: list[str] = field(default_factory=list)
    #: Acceptable substitutes — calls here are not wrong but do not satisfy
    #: ``expected_tools`` (near-miss trajectory).
    allowed_tools: list[str] = field(default_factory=list)
    #: Tools that must never be called for this task.
    forbidden_tools: list[str] = field(default_factory=list)
    notes: str = ""


# Loaded once at import.  Module-level name so tests can read/replace it.
TASK_ITEMS: list[TaskItem] = load_jsonl_items("tasks.jsonl", TaskItem)


def load_task_items() -> list[TaskItem]:
    return list(TASK_ITEMS)


def _normalize_name(name: str) -> str:
    import re

    return re.sub(r"\s+", "", str(name).strip().lower())


def _valid_tool_names() -> set[str]:
    """Names of the tools registered in ``backend.agent.tools.ALL_TOOLS``."""
    from backend.agent.tools import ALL_TOOLS

    return {t.name for t in ALL_TOOLS}


def validate_task_dataset(items: list[TaskItem] | None = None) -> list[str]:
    """Check internal consistency of the task labeled set.

    Args:
        items: the set to validate (default: the module-level ``TASK_ITEMS``).
            Tests pass a custom list; the CLI and run_task_eval use the default.

    Returns a list of human-readable warnings (empty == clean).  Raises
    ``ValueError`` only for hard failures that would make task-eval results
    meaningless:

    - unique ids; non-empty queries;
    - valid category; every expected/allowed/forbidden tool name exists in
      ``ALL_TOOLS``;
    - a non-refrain task carries at least one required fact, and every
      required fact is a substring of some e2e seed ``source_content`` —
      otherwise it can never be retrieved and the fact_coverage metric is
      structurally capped below 1.0.
    """
    warnings: list[str] = []
    tool_names = _valid_tool_names()

    # The facts the tasks must surface come from the e2e seed corpus.  A fact
    # absent from every source_content can never be retrieved, so flag it now
    # (the e2e-seed + task-eval run is what guarantees reachability at run
    # time).
    from tests.eval.llm_ground_truth import load_e2e_items

    seed_texts = {it.source_content for it in load_e2e_items()}
    items = list(items) if items is not None else list(TASK_ITEMS)

    seen: set[str] = set()
    for it in items:
        if it.id in seen:
            raise ValueError(f"duplicate task item id: {it.id}")
        seen.add(it.id)
        if not it.query.strip():
            raise ValueError(f"{it.id}: empty query")
        if it.category not in TASK_CATEGORIES:
            raise ValueError(
                f"{it.id}: unknown category {it.category!r} "
                f"(expected one of {TASK_CATEGORIES})"
            )
        for tool in it.expected_tools + it.allowed_tools + it.forbidden_tools:
            if tool not in tool_names:
                raise ValueError(f"{it.id}: unknown tool {tool!r} (not in ALL_TOOLS)")
        if it.category != "no_tool" and not it.required_facts:
            raise ValueError(f"{it.id}: required_facts is empty for a tool task")
        for fact in it.required_facts:
            if not str(fact).strip():
                raise ValueError(f"{it.id}: empty required fact")
            if not any(str(fact) in text for text in seed_texts):
                warnings.append(
                    f"{it.id}: required fact {fact!r} is not in any e2e seed "
                    "source_content — run `python -m tests.eval.e2e_seed` and "
                    "the fact_coverage metric is capped below 1.0"
                )
        if not it.expected_tools and it.forbidden_tools:
            warnings.append(
                f"{it.id}: expected_tools empty but forbidden_tools set — "
                "forbidden checks are vacuous unless the agent calls something"
            )

    return warnings
