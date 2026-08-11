"""Memory extraction — three independent stages: summary → entities → relations.

Each function calls LLM independently so results can be tested in isolation.
No chaining, no framework wrapping.
"""

from __future__ import annotations

import asyncio
import json
import logging

from backend.model.llm import LLMStructuredError
from backend.service.llm_service import get_llm_provider
from backend.service.prompts import get_prompt
from backend.service.structured import chat_structured
from backend.shared.config import config
from backend.shared.resilience import CircuitOpenError
from backend.shared.runtime_metrics import inc_structured_failures

logger = logging.getLogger(__name__)

# Entity / relation type enums — the single source of truth shared by the
# JSON schemas (chat_structured validation) and the function-calling tool
# schemas (generation-time constraint).  Keeping one list means an enum edit
# can't leave the two channels disagreeing.
_ENTITY_TYPES: list[str] = [
    "person", "project", "technology", "decision", "event", "file", "concept",
]
_RELATION_TYPES: list[str] = [
    "depends_on", "causes", "part_of", "contradicts", "supersedes", "relates_to",
]

# ── Stage 1: Summary ─────────────────────────────────────────────


async def extract_summary(content: str) -> str:
    """Condense *content* into a single concise summary paragraph.

    Fails safe: if the LLM call fails, returns the first 200 characters
    of the original content as a fallback summary.
    """
    try:
        llm = get_llm_provider()
        version, prompt = get_prompt("extraction.summary")
        logger.debug("extract_summary: using prompt extraction.summary v%s", version)
        msg = prompt.format(content=content)
        # Summarisation should stay faithful to the source — low temperature.
        summary = await llm.chat(
            [{"role": "user", "content": msg}],
            scenario="extraction_summary",
            temperature=0.3,
        )
        return summary.strip()
    except Exception:
        logger.exception("LLM summary extraction failed, using raw content fallback")
        return content.strip()[:200]

# ── Stage 2: Entities ─────────────────────────────────────────────


async def extract_entities(content_or_summary: str) -> list[dict]:
    """Extract named entities from *content_or_summary*.

    Works with either raw content or a summary — the prompt adapts
    automatically.  Types: person, project, technology, decision, event,
    file, concept.

    Preferred channel: function calling (``extract_entities`` tool, enum
    constraints at generation time) on OpenAI-compatible providers, falling
    back to ``chat_structured`` (JSON-schema enforcement + retry).  If both
    fail, degrades loudly — ERROR log + failure counter — and returns an
    empty list (enrichment, not correctness-critical, so the memory write
    proceeds).
    """
    try:
        llm = get_llm_provider()
        via_tool = await _entities_via_tool(llm, content_or_summary)
        if via_tool is not None:
            return via_tool

        version, prompt = get_prompt("extraction.entities")
        logger.debug("extract_entities: using prompt extraction.entities v%s", version)
        msg = prompt.format(input_text=content_or_summary)
        data = await chat_structured(
            [{"role": "user", "content": msg}],
            json_schema=_ENTITIES_SCHEMA,
            scenario="extraction_entities",
        )
        return data if isinstance(data, list) else []
    except (LLMStructuredError, CircuitOpenError):
        # Structured output exhausted its retries (LLMStructuredError) or the
        # circuit breaker is open and chat_structured failed fast
        # (CircuitOpenError).  Enrichment is not correctness-critical — degrade
        # to [] so the memory write proceeds rather than dropping the content.
        inc_structured_failures("extraction_entities")
        logger.error(
            "Entity extraction degraded to [] after retries (content=%r)",
            content_or_summary[:200],
        )
        return []


_ENTITIES_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["name", "type"],
        "properties": {
            "name": {"type": "string", "minLength": 1},
            "type": {"type": "string", "enum": _ENTITY_TYPES},
        },
    },
}


# ── Function-calling channel ─────────────────────────────────────────
# Preferred path for OpenAI-compatible providers (DeepSeek in production):
# the tool schema carries the same enum/required constraints as the JSON
# schema, but they hold *at generation time* — the model can't emit an
# out-of-enum type that the post-hoc validator then has to reject and
# retry.  The call is made without forcing ``tool_choice``: our DeepSeek
# endpoint's thinking mode rejects a forced choice with a 400
# ("Thinking mode does not support this tool_choice"), but reliably calls
# the tool on its own when it is offered.  When the model returns plain
# content instead of a tool call (or the provider errors), the caller
# falls back to ``chat_structured`` — graceful degradation, never a write
# failure.  Anthropic skips this channel entirely: its ``chat_json``
# already implements schema-constrained output via a forced ``tool_use``
# block, so offering a second tool would be a wasted call.

_ENTITIES_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_entities",
        "description": (
            "Extract the named entities explicitly mentioned in the text — "
            "technologies, people, projects, files, events, decisions, and "
            "concepts.  Only include entities that are concrete and specific; "
            "do not split compound phrases into parts and do not extract "
            "generic pronouns or vague references.  Each entity needs a name "
            "and one of the enumerated types."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "type"],
                        "properties": {
                            "name": {"type": "string", "minLength": 1},
                            "type": {"type": "string", "enum": _ENTITY_TYPES},
                        },
                    },
                },
            },
            "required": ["entities"],
        },
    },
}

_RELATIONS_TOOL = {
    "type": "function",
    "function": {
        "name": "extract_relations",
        "description": (
            "Extract relationships between entities, based only on what the "
            "summary explicitly states or unambiguously implies.  Do not "
            "invent links between entities the summary does not connect.  "
            "Each relation needs 'from' and 'to' entity names and one of the "
            "enumerated types."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["from", "to", "type"],
                        "properties": {
                            "from": {"type": "string", "minLength": 1},
                            "to": {"type": "string", "minLength": 1},
                            "type": {"type": "string", "enum": _RELATION_TYPES},
                        },
                    },
                },
            },
            "required": ["relations"],
        },
    },
}


def _is_openai_compatible(llm) -> bool:
    """True when the provider supports OpenAI-style function calling.

    ``chat_structured`` is the universal fallback; the dedicated extraction
    tool is an OpenAI-compatible enhancement (generation-time enum), so it
    must only be offered where it works.  ``FallbackLLMProvider`` delegates
    to a primary — use the primary's family for the decision.
    """
    name = getattr(llm, "PROVIDER_NAME", "")
    if name == "openai-compatible":
        return True
    primary = getattr(llm, "_primary", None)
    if primary is not None:
        return getattr(primary, "PROVIDER_NAME", "") == "openai-compatible"
    return False


async def _entities_via_tool(
    llm, content_or_summary: str
) -> list[dict] | None:
    """Extract entities via the ``extract_entities`` function tool.

    Returns ``None`` (not an empty list) when the channel produced nothing
    usable — the caller distinguishes "no result" from "channel failed" by
    falling back to ``chat_structured`` on ``None``.
    """
    if not _is_openai_compatible(llm):
        return None
    try:
        version, prompt = get_prompt("extraction.entities")
        msg = prompt.format(input_text=content_or_summary)
        resp = await llm.chat_raw(
            [{"role": "user", "content": msg}],
            tools=[_ENTITIES_TOOL],
            scenario="extraction_entities_tool",
            temperature=config.llm.structured_temperature,
        )
        for tc in resp.get("tool_calls") or []:
            args = tc.get("args")
            if not isinstance(args, dict) or not isinstance(args.get("entities"), list):
                continue
            return [
                e for e in args["entities"]
                if isinstance(e, dict) and str(e.get("name", "")).strip()
            ]
        return None
    except Exception:
        logger.warning(
            "Function-calling entity extraction failed — falling back to chat_structured",
            exc_info=True,
        )
        return None


# ── Stage 3: Relations ────────────────────────────────────────────


def _filter_relations(
    relations: list[dict], entity_names: list[str]
) -> list[dict]:
    """Drop relations whose endpoints aren't in *entity_names*.

    Guards against hallucinated endpoints — both channels (function tool and
    chat_structured) pass through here so the rule can't drift apart.
    """
    valid_names = set(entity_names)
    filtered = [
        r
        for r in relations
        if r.get("from") in valid_names and r.get("to") in valid_names
    ]
    if len(filtered) != len(relations):
        logger.warning(
            "Dropped %d relation(s) referencing unknown entities",
            len(relations) - len(filtered),
        )
    return filtered


async def _relations_via_tool(
    llm, summary: str, entity_names: list[str]
) -> list[dict] | None:
    """Extract relations via the ``extract_relations`` function tool.

    Returns ``None`` when the channel produced nothing usable — the caller
    falls back to ``chat_structured``.
    """
    if not _is_openai_compatible(llm):
        return None
    try:
        version, prompt = get_prompt("extraction.relations")
        msg = prompt.format(
            summary=summary, entities=json.dumps(entity_names, ensure_ascii=False)
        )
        resp = await llm.chat_raw(
            [{"role": "user", "content": msg}],
            tools=[_RELATIONS_TOOL],
            scenario="extraction_relations_tool",
            temperature=config.llm.structured_temperature,
        )
        for tc in resp.get("tool_calls") or []:
            args = tc.get("args")
            if not isinstance(args, dict) or not isinstance(args.get("relations"), list):
                continue
            return [
                r for r in args["relations"]
                if isinstance(r, dict)
                and str(r.get("from", "")).strip()
                and str(r.get("to", "")).strip()
            ]
        return None
    except Exception:
        logger.warning(
            "Function-calling relation extraction failed — falling back to chat_structured",
            exc_info=True,
        )
        return None


async def extract_relations(
    summary: str, entities: list[dict]
) -> list[dict]:
    """Extract relationships between *entities* from *summary*.

    Returns a list of {{from, to, type}} dicts.
    Types: depends_on, causes, part_of, contradicts, supersedes, relates_to.

    Preferred channel: function calling (``extract_relations`` tool) on
    OpenAI-compatible providers, falling back to ``chat_structured`` (JSON
    schema + retry).  Relations whose ``from``/``to`` reference names outside
    *entities* are dropped (guards against hallucinated endpoints).  On
    persistent failure, degrades loudly — ERROR log + failure counter — and
    returns an empty list (enrichment, not correctness-critical).
    """
    if len(entities) < 2:
        return []

    entity_names = [e["name"] for e in entities]
    try:
        llm = get_llm_provider()
        via_tool = await _relations_via_tool(llm, summary, entity_names)
        if via_tool is not None:
            return _filter_relations(via_tool, entity_names)

        version, prompt = get_prompt("extraction.relations")
        logger.debug("extract_relations: using prompt extraction.relations v%s", version)
        msg = prompt.format(
            summary=summary, entities=json.dumps(entity_names, ensure_ascii=False)
        )
        data = await chat_structured(
            [{"role": "user", "content": msg}],
            json_schema=_RELATIONS_SCHEMA,
            scenario="extraction_relations",
        )
        relations = data if isinstance(data, list) else []
        return _filter_relations(relations, entity_names)
    except (LLMStructuredError, CircuitOpenError):
        # Same fail-safe as entities: structured output exhausted its retries
        # (LLMStructuredError) or the circuit breaker is open and
        # chat_structured failed fast (CircuitOpenError).  Degrade to [] so the
        # memory write proceeds.
        inc_structured_failures("extraction_relations")
        logger.error(
            "Relation extraction degraded to [] after retries (summary=%r)",
            summary[:200],
        )
        return []


_RELATIONS_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "required": ["from", "to", "type"],
        "properties": {
            "from": {"type": "string", "minLength": 1},
            "to": {"type": "string", "minLength": 1},
            "type": {"type": "string", "enum": _RELATION_TYPES},
        },
    },
}

# ── Orchestration (still a plain function, not a chain) ────────────


async def extract_memory(content: str) -> dict:
    """Run extraction stages with maximum parallelism.

    Summary and entities can start simultaneously (both only need content).
    Relations need entities, so it runs after.
    """
    # Stage 1 + 2 in parallel — both only depend on raw content
    summary, entities = await asyncio.gather(
        extract_summary(content),
        extract_entities(content),
    )

    # Stage 3 — depends on entities from stage 2
    relations = await extract_relations(summary, entities) if entities else []

    return {
        "summary": summary,
        "entities": entities,
        "relations": relations,
    }
