"""Memory extraction — three independent stages: summary → entities → relations.

Each function calls LLM independently so results can be tested in isolation.
No chaining, no framework wrapping.
"""

from __future__ import annotations

import asyncio
import json
import logging

from backend.service.llm_service import get_llm_provider

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
        summary = await llm.chat([{"role": "user", "content": msg}])
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

    Fails safe: returns an empty list on LLM or parse errors.
    """
    try:
        llm = get_llm_provider()
        msg = _ENTITIES_PROMPT.format(input_text=content_or_summary)
        raw = await llm.chat([{"role": "user", "content": msg}])
        raw = raw.strip()

        try:
            entities = json.loads(raw)
            if isinstance(entities, list):
                return entities
        except json.JSONDecodeError:
            logger.warning("Failed to parse entities JSON, returning empty list")

        return []
    except Exception:
        logger.exception("LLM call failed during entity extraction")
        return []


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

    Fails safe: returns an empty list on LLM or parse errors.
    """
    if len(entities) < 2:
        return []

    entity_names = [e["name"] for e in entities]
    try:
        llm = get_llm_provider()
        msg = _RELATIONS_PROMPT.format(
            summary=summary, entities=json.dumps(entity_names, ensure_ascii=False)
        )
        raw = await llm.chat([{"role": "user", "content": msg}])
        raw = raw.strip()

        try:
            relations = json.loads(raw)
            if isinstance(relations, list):
                return relations
        except json.JSONDecodeError:
            logger.warning("Failed to parse relations JSON, returning empty list")

        return []
    except Exception:
        logger.exception("LLM call failed during relation extraction")
        return []


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
