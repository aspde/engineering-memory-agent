"""Labeled datasets for the LLM behavior eval — three suites.

The retrieval eval (``tests.eval.ground_truth``) measures *which memories a
search returns*; it cannot answer "did the agent call the right tool", "did
extraction produce correct entities", or "is the final answer grounded in
the retrieved context".  This module carries the labeled sets for those
three LLM-behavior dimensions:

- **tool_selection** — a query + the tool(s) the agent *must* call, the ones
  it must *not* call, and acceptable alternatives.  Driven through the real
  ``call_llm_node`` (real system prompt + real tool schemas) so the eval
  measures production behavior, not a stripped-down harness.
- **extraction** — a source text + the entities / relations / summary
  keywords a correct ``extract_memory`` run should produce.  Entity and
  relation types follow the enum in ``backend.service.extraction``.
- **answer** — a query + a golden retrieved *context* (the only fact source
  the model may use) + ``required_facts`` the answer must cover and
  ``prohibited_claims`` it must not make.  Measures groundedness and
  coverage of the final-answer path in isolation.

The sets are small by design: each item costs 1-6 real LLM calls per run, so
a full pass is ~60-90 calls — cheap enough for a weekly scheduled job.  Keep
them hand-authored (never generated) so drift is human-reviewable.

The items themselves live in ``tests/eval/data/*.jsonl`` (one file per suite:
``tool_selection.jsonl`` / ``extraction.jsonl`` / ``answer.jsonl`` /
``e2e.jsonl``); this module keeps the item types, constants, loaders and the
dataset validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from tests.eval.core import load_jsonl_items


# ── Per-suite categories ───────────────────────────────────────────

TOOL_SELECTION_CATEGORIES: tuple[str, ...] = (
    "memory_search",
    "doc_search",
    "write",
    "ingest",
    "entity",
    "extract",
    "notify",
    "no_tool",
)

EXTRACTION_CATEGORIES: tuple[str, ...] = (
    "code_decision",
    "incident",
    "architecture",
    "process",
)

ANSWER_CATEGORIES: tuple[str, ...] = (
    "factual",
    "causal",
    "instruction",
    "negation",
)

# Entity / relation type enums — must match backend.service.extraction's
# schemas (duplicated here so the dataset module stays decoupled from the
# production prompt internals; validation flags drift below).
ENTITY_TYPES: tuple[str, ...] = (
    "person",
    "project",
    "technology",
    "decision",
    "event",
    "file",
    "concept",
)
RELATION_TYPES: tuple[str, ...] = (
    "depends_on",
    "causes",
    "part_of",
    "contradicts",
    "supersedes",
    "relates_to",
)

# Contexts over 800 chars get truncated by the agent's tool-result cap
# (backend/agent/nodes.py ``_truncate_tool_content``); the answer suite injects
# context directly, but keeping items under the cap keeps the eval honest to
# what the model would actually see through the retrieval path.
ANSWER_CONTEXT_HARD_CAP = 2000
ANSWER_CONTEXT_SOFT_CAP = 800


# ── Tool selection ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolSelectionItem:
    id: str
    query: str
    expected_tools: list[str]
    category: str
    #: Tools the agent must never call for this query.
    forbidden_tools: list[str] = field(default_factory=list)
    #: Tools that are acceptable substitutes for the expected ones (near-miss
    #: calls do not count as wrong, but do not satisfy the expected set).
    allowed_tools: list[str] = field(default_factory=list)
    #: Optional arg-precision check: tool name → substrings that must appear
    #: in that call's JSON-serialized arguments (e.g. the search query).
    expected_args: dict[str, list[str]] = field(default_factory=dict)
    notes: str = ""


# Loaded once at import.  Module-level names so tests can monkeypatch a bad
# item into the list and re-run validation.
TOOL_SELECTION_ITEMS: list[ToolSelectionItem] = load_jsonl_items(
    "tool_selection.jsonl", ToolSelectionItem
)


# ── Extraction ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExtractionItem:
    id: str
    content: str
    expected_entities: list[dict[str, str]]  # [{"name", "type"}]
    expected_relations: list[dict[str, str]]  # [{"from", "to", "type"}]
    category: str
    summary_keywords: list[str]
    notes: str = ""


EXTRACTION_ITEMS: list[ExtractionItem] = load_jsonl_items(
    "extraction.jsonl", ExtractionItem
)


# ── Final answer ────────────────────────────────────────────────────


@dataclass(frozen=True)
class AnswerItem:
    id: str
    query: str
    context: str  # the only fact source the answer may use
    required_facts: list[str]
    category: str
    prohibited_claims: list[str] = field(default_factory=list)
    #: Source IDs the model is shown alongside the context (the answer suite
    #: presents them the way a real memory-search display does).  A grounded
    #: answer should cite at least one — measured by ``citation_presence``.
    source_ids: list[str] = field(default_factory=list)
    notes: str = ""


ANSWER_ITEMS: list[AnswerItem] = load_jsonl_items("answer.jsonl", AnswerItem)


# ── End-to-end (E2E) ────────────────────────────────────────────────
# Unlike the answer suite (which injects golden context), e2e items drive the
# real retrieval chain: the query is issued against the seeded corpus, the
# retrieved results become the model's context, and the answer is judged
# against *what was actually retrieved*.  ``context_recall`` (see
# llm_metrics) then separates retrieval losses from generation losses.
#
# ``source_content`` is the knowledge that MUST be retrievable — it is seeded
# into the memories/chunks table by ``tests.eval.e2e_seed`` before the run.
# Every ``required_fact`` must be a substring of ``source_content`` (enforced
# by validation) or the item can never score context_recall = 1.0.

E2E_RETRIEVAL_MODES: tuple[str, ...] = ("memory", "chunk")


@dataclass(frozen=True)
class E2EItem:
    id: str
    query: str
    source_content: str
    required_facts: list[str]
    category: str
    prohibited_claims: list[str] = field(default_factory=list)
    #: "memory" → query_memories (default read path); "chunk" →
    #: retrieve_hybrid (document chunks).
    retrieval_mode: str = "memory"
    notes: str = ""


E2E_ITEMS: list[E2EItem] = load_jsonl_items("e2e.jsonl", E2EItem)


# ── Loading & validation ───────────────────────────────────────────


def load_tool_selection_items() -> list[ToolSelectionItem]:
    return list(TOOL_SELECTION_ITEMS)


def load_extraction_items() -> list[ExtractionItem]:
    return list(EXTRACTION_ITEMS)


def load_answer_items() -> list[AnswerItem]:
    return list(ANSWER_ITEMS)


def load_e2e_items() -> list[E2EItem]:
    return list(E2E_ITEMS)


def _normalize_name(name: str) -> str:
    """Lowercase + strip all whitespace — the same normalization the metrics use."""
    return re.sub(r"\s+", "", str(name).strip().lower())


def _valid_tool_names() -> set[str]:
    """Names of the tools registered in ``backend.agent.tools.ALL_TOOLS``."""
    from backend.agent.tools import ALL_TOOLS

    return {t.name for t in ALL_TOOLS}


def validate_llm_dataset() -> list[str]:
    """Check internal consistency of all three labeled sets.

    Returns a list of human-readable warnings (empty == clean).  Raises
    ``ValueError`` only for hard failures that would make eval results
    meaningless (e.g. a required tool name that does not exist).

    Checks:
        - tool_selection: unique ids; non-empty queries; every tool name in
          expected/allowed/forbidden/expected_args exists in ``ALL_TOOLS``.
        - extraction: unique ids; non-empty content; entity names/types valid;
          every relation's endpoints exist among the golden entities and use
          a valid relation type.
        - answer: unique ids; non-empty query/context/facts; context under the
          hard cap.
    """
    warnings: list[str] = []
    tool_names = _valid_tool_names()

    # ── tool_selection ──
    seen: set[str] = set()
    for it in TOOL_SELECTION_ITEMS:
        if it.id in seen:
            raise ValueError(f"duplicate tool_selection item id: {it.id}")
        seen.add(it.id)
        if not it.query.strip():
            raise ValueError(f"{it.id}: empty query")
        if it.category not in TOOL_SELECTION_CATEGORIES:
            raise ValueError(
                f"{it.id}: unknown category {it.category!r} "
                f"(expected one of {TOOL_SELECTION_CATEGORIES})"
            )
        for tool in it.expected_tools + it.allowed_tools + it.forbidden_tools:
            if tool not in tool_names:
                raise ValueError(f"{it.id}: unknown tool {tool!r} (not in ALL_TOOLS)")
        for tool in it.expected_args:
            if tool not in tool_names:
                raise ValueError(f"{it.id}: expected_args tool {tool!r} not in ALL_TOOLS")
        if not it.expected_tools and it.forbidden_tools:
            warnings.append(
                f"{it.id}: expected_tools empty but forbidden_tools set — "
                "forbidden checks are vacuous unless the model calls something"
            )

    # ── extraction ──
    seen.clear()
    for it in EXTRACTION_ITEMS:
        if it.id in seen:
            raise ValueError(f"duplicate extraction item id: {it.id}")
        seen.add(it.id)
        if not it.content.strip():
            raise ValueError(f"{it.id}: empty content")
        if it.category not in EXTRACTION_CATEGORIES:
            raise ValueError(
                f"{it.id}: unknown category {it.category!r} "
                f"(expected one of {EXTRACTION_CATEGORIES})"
            )
        if not it.expected_entities:
            raise ValueError(f"{it.id}: expected_entities is empty")
        entity_names: list[str] = []
        for e in it.expected_entities:
            name = str(e.get("name", "")).strip()
            etype = str(e.get("type", "")).strip()
            if not name:
                raise ValueError(f"{it.id}: entity with empty name")
            if etype not in ENTITY_TYPES:
                raise ValueError(f"{it.id}: entity {name!r} has invalid type {etype!r}")
            entity_names.append(name)
        norm_names = [_normalize_name(n) for n in entity_names]
        for r in it.expected_relations:
            rtype = str(r.get("type", "")).strip()
            if rtype not in RELATION_TYPES:
                raise ValueError(
                    f"{it.id}: relation has invalid type {rtype!r} "
                    f"(expected one of {RELATION_TYPES})"
                )
            for endpoint in (str(r.get("from", "")), str(r.get("to", ""))):
                if _normalize_name(endpoint) not in norm_names:
                    raise ValueError(
                        f"{it.id}: relation endpoint {endpoint!r} is not a golden "
                        "entity — extraction filters such relations out, so this "
                        "golden label can never match"
                    )
        if not it.summary_keywords:
            raise ValueError(f"{it.id}: summary_keywords is empty")

    # ── answer ──
    seen.clear()
    for it in ANSWER_ITEMS:
        if it.id in seen:
            raise ValueError(f"duplicate answer item id: {it.id}")
        seen.add(it.id)
        if not it.query.strip():
            raise ValueError(f"{it.id}: empty query")
        if not it.context.strip():
            raise ValueError(f"{it.id}: empty context")
        if not it.required_facts:
            raise ValueError(f"{it.id}: required_facts is empty")
        if len(it.context) > ANSWER_CONTEXT_HARD_CAP:
            raise ValueError(
                f"{it.id}: context {len(it.context)} chars exceeds hard cap "
                f"{ANSWER_CONTEXT_HARD_CAP}"
            )
        if len(it.context) > ANSWER_CONTEXT_SOFT_CAP:
            warnings.append(
                f"{it.id}: context {len(it.context)} chars exceeds the agent's "
                f"{ANSWER_CONTEXT_SOFT_CAP}-char tool-result cap — the model would "
                "only see the truncated head through the real retrieval path"
            )
        if it.category not in ANSWER_CATEGORIES:
            raise ValueError(
                f"{it.id}: unknown category {it.category!r} "
                f"(expected one of {ANSWER_CATEGORIES})"
            )
        if not it.source_ids:
            warnings.append(
                f"{it.id}: no source_ids — the citation_rate metric is vacuous "
                "(1.0 for every answer) without something to cite"
            )
        for sid in it.source_ids:
            if not str(sid).strip():
                raise ValueError(f"{it.id}: empty source id in source_ids")

    # ── e2e ──
    seen.clear()
    for it in E2E_ITEMS:
        if it.id in seen:
            raise ValueError(f"duplicate e2e item id: {it.id}")
        seen.add(it.id)
        if not it.query.strip():
            raise ValueError(f"{it.id}: empty query")
        if not it.source_content.strip():
            raise ValueError(f"{it.id}: empty source_content")
        if not it.required_facts:
            raise ValueError(f"{it.id}: required_facts is empty")
        if it.category not in ANSWER_CATEGORIES:
            raise ValueError(
                f"{it.id}: unknown category {it.category!r} "
                f"(expected one of {ANSWER_CATEGORIES})"
            )
        if it.retrieval_mode not in E2E_RETRIEVAL_MODES:
            raise ValueError(
                f"{it.id}: unknown retrieval_mode {it.retrieval_mode!r} "
                f"(expected one of {E2E_RETRIEVAL_MODES})"
            )
        for fact in it.required_facts:
            if not str(fact).strip():
                raise ValueError(f"{it.id}: empty required fact")
            if str(fact) not in it.source_content:
                raise ValueError(
                    f"{it.id}: required fact {fact!r} is not a substring of "
                    "source_content — context_recall can never reach 1.0"
                )

    return warnings
