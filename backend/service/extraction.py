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
from backend.service.structured import chat_structured
from backend.shared.metrics import record_structured_failure

logger = logging.getLogger(__name__)

# ── Stage 1: Summary ─────────────────────────────────────────────


async def extract_summary(content: str) -> str:
    """Condense *content* into a single concise summary paragraph.

    Fails safe: if the LLM call fails, returns the first 200 characters
    of the original content as a fallback summary.
    """
    try:
        llm = get_llm_provider()
        msg = _SUMMARY_PROMPT.format(content=content)
        summary = await llm.chat([{"role": "user", "content": msg}], scenario="extraction_summary")
        return summary.strip()
    except Exception:
        logger.exception("LLM summary extraction failed, using raw content fallback")
        return content.strip()[:200]


_SUMMARY_PROMPT = """\
Summarize the following content in one concise paragraph (2-5 sentences).
Focus on key facts, decisions, and actionable information.
Avoid fluff — only write what someone searching for this information would want to find.
Respond in the same language as the input content.

Content:
{content}

Summary:"""

# ── Stage 2: Entities ─────────────────────────────────────────────


async def extract_entities(content_or_summary: str) -> list[dict]:
    """Extract named entities from *content_or_summary*.

    Works with either raw content or a summary — the prompt adapts
    automatically.  Types: person, project, technology, decision, event,
    file, concept.

    Structured output is enforced and retried via ``chat_structured``.  If
    the LLM still cannot produce schema-valid JSON, degrades loudly — ERROR
    log + failure counter — and returns an empty list (enrichment, not
    correctness-critical, so the memory write proceeds).
    """
    try:
        msg = _ENTITIES_PROMPT.format(input_text=content_or_summary)
        data = await chat_structured(
            [{"role": "user", "content": msg}],
            json_schema=_ENTITIES_SCHEMA,
            scenario="extraction_entities",
        )
        return data if isinstance(data, list) else []
    except LLMStructuredError:
        record_structured_failure("extraction_entities")
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
            "type": {
                "type": "string",
                "enum": [
                    "person",
                    "project",
                    "technology",
                    "decision",
                    "event",
                    "file",
                    "concept",
                ],
            },
        },
    },
}

_ENTITIES_PROMPT = """\
Extract named entities from the following text.
Return ONLY a JSON array of objects with "name" and "type" fields.
Types must be one of: person, project, technology, decision, event, file, concept.
Use the same language as the input text for entity names.

Text:
{input_text}

Example output:
[{{"name": "PostgreSQL", "type": "technology"}}, {{"name": "migration plan", "type": "decision"}}]"""


# ── Stage 3: Relations ────────────────────────────────────────────


async def extract_relations(
    summary: str, entities: list[dict]
) -> list[dict]:
    """Extract relationships between *entities* from *summary*.

    Returns a list of {{from, to, type}} dicts.
    Types: depends_on, causes, part_of, contradicts, supersedes, relates_to.

    Structured output is enforced and retried via ``chat_structured``.
    Relations whose ``from``/``to`` reference names outside *entities* are
    dropped (guards against hallucinated endpoints).  On persistent failure,
    degrades loudly — ERROR log + failure counter — and returns an empty
    list (enrichment, not correctness-critical).
    """
    if len(entities) < 2:
        return []

    entity_names = [e["name"] for e in entities]
    try:
        msg = _RELATIONS_PROMPT.format(
            summary=summary, entities=json.dumps(entity_names, ensure_ascii=False)
        )
        data = await chat_structured(
            [{"role": "user", "content": msg}],
            json_schema=_RELATIONS_SCHEMA,
            scenario="extraction_relations",
        )
        relations = data if isinstance(data, list) else []

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
    except LLMStructuredError:
        record_structured_failure("extraction_relations")
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
            "type": {
                "type": "string",
                "enum": [
                    "depends_on",
                    "causes",
                    "part_of",
                    "contradicts",
                    "supersedes",
                    "relates_to",
                ],
            },
        },
    },
}

_RELATIONS_PROMPT = """\
Identify relationships between the following entities based on the summary.
Return ONLY a JSON array of objects with "from", "to", and "type" fields.
"from" and "to" must be entity names from the provided list.
Types must be one of: depends_on, causes, part_of, contradicts, supersedes, relates_to.

Summary:
{summary}

Entities: {entities}

Example output:
[{{"from": "PostgreSQL", "to": "pgvector", "type": "depends_on"}}, {{"from": "migration", "to": "downtime", "type": "causes"}}]"""

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
