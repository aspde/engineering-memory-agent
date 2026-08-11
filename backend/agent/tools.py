"""Agent tool definitions — thin wrappers around existing backend services.

Each tool is an async function decorated with ``@tool`` so LangGraph's
``ToolNode`` can auto-generate schemas and execute tool calls.  Tools
return formatted strings because that's what the LLM reads via
``ToolMessage.content``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from langchain_core.tools import tool
from pydantic import Field

from backend.agent.tool_envelope import build_tool_envelope
from backend.service.chunk import chunk_code, chunk_text
from backend.service.entity import (
    get_entity_by_name,
    get_entity_relations_for_tool,
    get_memory_entities_batch,
)
from backend.service.extraction import extract_memory
from backend.service.ingestion import ingest_repo
from backend.service.memory import write_memory
from backend.service.retrieval import query_memories, retrieve_hybrid, write_chunks


# ── Retrieval tools ──────────────────────────────────────────────────


@tool
async def search_memories_tool(
    query: str,
    top_k: int = Field(default=5, ge=1, le=20),
) -> str:
    """Search long-term engineering memories for knowledge, decisions,
    lessons learned, and past context.

    Memories come from multiple sources: manual conversations, PingCode
    work items (pingcode / pingcode_bug), CI/CD builds (ci_build /
    ci_regression), 飞书 discussions (feishu), Git commits, and document
    ingestion.  This tool searches across ALL sources by default.

    Use this when the user asks about project history, technical
    decisions, architecture, past discussions, or anything that might
    have been recorded as a memory — regardless of which source it
    came from.

    Args:
        query: Natural-language search query.
        top_k: Number of results (1-20).
    """
    results = await query_memories(query, top_k=min(top_k, 20))
    if not results:
        return "No relevant memories found."

    # Annotate with normalized entity links
    memory_ids = [str(r["id"]) for r in results]
    entity_map: dict[str, list[dict]] = {}
    try:
        entity_map = await get_memory_entities_batch(memory_ids)
    except Exception:
        pass  # entities table may not exist pre-migration

    lines = [f"Found {len(results)} relevant memories:"]
    sources: list[dict[str, Any]] = []
    for i, r in enumerate(results):
        score = r.get("rerank_score", r.get("similarity", 0))
        mid = str(r["id"])
        entities = entity_map.get(mid, [])
        entity_names = [e["canonical_name"] for e in entities]

        # Expose the stable memory short ID in the display text so the LLM
        # can cite it inline (per the agent.system citation guidance).  The
        # full ID stays in the sources envelope for the UI.  Recall stats are
        # shown so the model can judge staleness (patrol archival) from the
        # raw access history.
        short_id = mid[:8]
        recalls = int(r.get("recall_count", 0) or 0)
        recalled_at = r.get("recalled_at")
        last_recalled = recalled_at.isoformat()[:10] if recalled_at else "never"
        line = (
            f"[{i + 1}] (memory: {short_id}, relevance: {score:.2f}, "
            f"recalls: {recalls}, last_recalled: {last_recalled}) "
            f"{r['summary']}"
        )
        if entity_names:
            line += f"  [entities: {', '.join(entity_names)}]"
        lines.append(line)

        sources.append({
            "id": mid,
            "type": "memory",
            "summary": str(r["summary"])[:200],
            "relevance": round(float(score), 4),
            "entities": entities,
        })
    return build_tool_envelope("\n".join(lines), sources)


@tool
async def retrieve_chunks_tool(
    query: str,
    top_k: int = Field(default=5, ge=1, le=20),
) -> str:
    """Hybrid search over ingested document chunks (dense vector + BM25 keyword).

    Combines BGE-M3 semantic recall with Postgres tsvector keyword recall
    for better coverage on conceptual queries.  Use this as the default
    document search, or when memory search doesn't return enough context.

    Args:
        query: Natural-language search query.
        top_k: Number of results (1-20).
    """
    results = await retrieve_hybrid(query, top_k=min(top_k, 20))
    if not results:
        return "No relevant document chunks found."

    lines = [f"Found {len(results)} relevant chunks:"]
    sources: list[dict[str, Any]] = []
    for i, r in enumerate(results):
        meta = r.metadata or {}
        doc = meta.get("document_id") or ""
        if doc:
            lines.append(f"[{i + 1}] (relevance: {r.score:.2f}, document: {doc}) {r.content}")
        else:
            lines.append(f"[{i + 1}] (relevance: {r.score:.2f}) {r.content}")
        sources.append({
            "document_id": str(doc),
            "chunk_index": meta.get("chunk_index", i),
            "type": "chunk",
            "snippet": r.content[:200],
            "relevance": round(float(r.score), 4),
            "metadata": meta,
        })
    return build_tool_envelope("\n".join(lines), sources)


@tool
async def query_rewrite_and_search_tool(
    query: str,
    top_k: int = Field(default=5, ge=1, le=20),
) -> str:
    """Multi-query retrieval: LLM rewrites the query into variations,
    then unions and reranks results from all variations.

    Use this for conceptual or abstract queries where the user's wording
    may not match the stored memory's wording — e.g. "之前出过什么问题",
    "会不会陷入死循环", "同名实体怎么归一化".  The LLM expands such
    queries into concrete terms (component names, error types) that
    surface in the knowledge base.

    Costs one extra LLM call (~500ms) for rewriting.  For specific
    technical queries (e.g. "pgvector 向量检索"), prefer
    retrieve_chunks_tool instead.

    Args:
        query: Natural-language search query (especially conceptual ones).
        top_k: Number of results (1-20).
    """
    from backend.service.retrieval import retrieve_multi_query

    results = await retrieve_multi_query(query, top_k=min(top_k, 20))
    if not results:
        return "No relevant document chunks found (even after query rewriting)."

    lines = [f"Found {len(results)} relevant chunks (query rewritten):"]
    sources: list[dict[str, Any]] = []
    for i, r in enumerate(results):
        meta = r.metadata or {}
        doc = meta.get("document_id") or ""
        if doc:
            lines.append(f"[{i + 1}] (relevance: {r.score:.2f}, document: {doc}) {r.content}")
        else:
            lines.append(f"[{i + 1}] (relevance: {r.score:.2f}) {r.content}")
        sources.append({
            "document_id": str(doc),
            "chunk_index": meta.get("chunk_index", i),
            "type": "chunk",
            "snippet": r.content[:200],
            "relevance": round(float(r.score), 4),
            "metadata": meta,
        })
    return build_tool_envelope("\n".join(lines), sources)


# ── Write tools ──────────────────────────────────────────────────────


@tool
async def write_memory_tool(
    content: str,
    source_type: str = "conversation",
    metadata: dict[str, Any] | None = None,
) -> str:
    """Write a new long-term memory from conversation or content.

    The system will extract a summary, entities, and relations,
    then check for duplicates before inserting.  Returns the action
    taken (inserted, merged, or conflict).

    Use this when the user explicitly asks to remember something, or
    when important knowledge emerges in conversation.

    Args:
        content: The text to extract a memory from.
        source_type: Where the content came from (conversation, doc, etc.).
        metadata: Optional extra metadata to store with the memory.
    """
    result = await write_memory(content, source_type=source_type, metadata=metadata)
    if result.get("action") == "conflict":
        return json.dumps(
            {
                "action": result["action"],
                "summary": result["summary"],
                "existing_id": result["existing_id"],
                "existing_summary": result["existing_summary"],
                "entities": result.get("entities", []),
                "relations": result.get("relations", []),
                "_deferred": result.get("_deferred"),
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {
            "id": result["id"],
            "action": result["action"],
            "summary": result["summary"],
        },
        ensure_ascii=False,
    )


@tool
async def extract_memory_tool(content: str) -> str:
    """Extract structured knowledge (summary, entities, relations) from text.

    Does NOT persist — only extracts.  Use this when the user wants to
    preview what would be remembered, or to analyse text for entities
    and relationships.

    Args:
        content: The text to extract structured knowledge from.
    """
    result = await extract_memory(content)
    return json.dumps(
        {
            "summary": result["summary"],
            "entities": result["entities"],
            "relations": result["relations"],
        },
        ensure_ascii=False,
    )


# ── Entity tools ─────────────────────────────────────────────────────


@tool
async def query_entity_tool(entity_name: str) -> str:
    """Look up a normalized entity by name and return its profile, related
    entities, and recent memories.

    Entities are extracted from all sources — conversations, PingCode work items,
    CI builds, 飞书 discussions, and Git commits.  This tool shows the full
    picture: which memories mention this entity, which other entities it
    relates to, and where those memories came from.

    Use this when the user asks about a specific technology, person,
    project, or concept — e.g. \"what do we know about PostgreSQL?\" or
    \"show me everything related to Kafka\".

    Args:
        entity_name: The entity name to look up (fuzzy match against
            canonical_name and name).
    """
    entity = await get_entity_by_name(entity_name)
    if entity is None:
        return json.dumps(
            {"found": False, "message": f"No entity found matching '{entity_name}'."},
            ensure_ascii=False,
        )

    relations = await get_entity_relations_for_tool(str(entity["id"]))

    return json.dumps(
        {
            "found": True,
            "entity": {
                "id": str(entity["id"]),
                "name": str(entity["name"]),
                "canonical_name": str(entity["canonical_name"]),
                "type": str(entity["type"]),
                "memory_count": entity["memory_count"],
            },
            "related_entities": relations["related_entities"],
            "recent_memories": relations["recent_memories"],
        },
        ensure_ascii=False,
        default=str,
    )


# ── Ingestion tools ──────────────────────────────────────────────────


@tool
async def ingest_git_repo_tool(
    repo_path: str,
    max_commits: int = Field(default=50, ge=1, le=200),
    branch: str | None = None,
) -> str:
    """Ingest a local Git repository's commit history as memories.

    Each commit becomes a structured memory with author, message, and
    diff context.  Use this when the user wants to ingest a codebase's
    history for future retrieval.

    Args:
        repo_path: Absolute path to the local Git repository.
        max_commits: How many recent commits to process (1-200).
        branch: Branch name (default: HEAD).
    """
    results = await ingest_repo(
        repo_path, max_commits=min(max_commits, 200), branch=branch
    )
    if not results:
        return "No commits were ingested (repository may be empty or inaccessible)."

    lines = [f"Ingested {len(results)} commits as memories:"]
    for r in results:
        lines.append(f"  [{r['action']}] {r['summary'][:120]}")
    return "\n".join(lines)


@tool
async def ingest_document_tool(
    document_id: str,
    content: str,
    language: str = "text",
    metadata: dict[str, Any] | None = None,
) -> str:
    """Chunk, embed, and store a document for later retrieval.

    Splits content into chunks — prose documents use recursive separator
    splitting, Python code uses AST-aware function/class boundaries so a
    200-line function is never cut in half.

    Use this when the user asks to index a document, code file, or any
    text for future search.

    Args:
        document_id: Unique identifier (e.g. filename or path).
        content: Full text content of the document.
        language: "python" for AST-aware code chunking, "text" (default)
            for recursive separator splitting.
        metadata: Optional metadata (source, author, etc.).
    """
    meta = (metadata or {}) | {"language": language}

    # Chunking is CPU-bound (recursive separator splitting, or AST parsing
    # for Python) — offload it to the thread pool so the event loop isn't
    # blocked while the LLM is waiting on the tool result.
    if language == "python":
        chunks = await asyncio.to_thread(chunk_code, content)
    else:
        chunks = await asyncio.to_thread(chunk_text, content)

    count = await write_chunks(document_id, chunks, meta=meta)
    return f"Ingested {count} chunks from document '{document_id}'."


# ── Notification tool ─────────────────────────────────────────────────


@tool
async def notify_feishu_tool(
    message: str,
    msg_type: str = "text",
    title: str | None = None,
) -> str:
    """Send a notification to the team's 飞书 (Feishu/Lark) group via bot webhook.

    Use this to push patrol findings, event alerts, or important discoveries
    to the team's 飞书 chat.  The message is formatted as plain text by default;
    set msg_type="interactive" for a rich card with a title.

    Args:
        message: Notification text (plain text or markdown-like).
        msg_type: ``"text"`` (default) or ``"interactive"`` for a card.
        title: Card title (only used when msg_type="interactive").
    """
    import httpx

    from backend.shared.config import config

    webhook_url = config.feishu_webhook_url
    if not webhook_url:
        return json.dumps(
            {"ok": False, "error": "FEISHU_WEBHOOK_URL is not configured"},
            ensure_ascii=False,
        )

    # Build Feishu bot webhook payload
    if msg_type == "interactive":
        payload: dict[str, Any] = {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {"tag": "plain_text", "content": title or "EMA 巡检通知"},
                    "template": "blue",
                },
                "elements": [
                    {"tag": "markdown", "content": message},
                ],
            },
        }
    else:
        payload = {
            "msg_type": "text",
            "content": {"text": message},
        }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(webhook_url, json=payload)
            resp.raise_for_status()
        feishu_result = resp.json()
        return json.dumps(
            {
                "ok": True,
                "msg_type": msg_type,
                "message_preview": message[:200],
                "feishu_status": feishu_result.get("code", -1),
            },
            ensure_ascii=False,
        )
    except httpx.TimeoutException:
        import logging
        logging.getLogger(__name__).error("Feishu webhook timed out")
        return json.dumps(
            {"ok": False, "error": "Webhook request timed out"},
            ensure_ascii=False,
        )
    except Exception as exc:
        import logging
        logging.getLogger(__name__).error("Feishu webhook failed: %s", exc)
        return json.dumps(
            {"ok": False, "error": str(exc)},
            ensure_ascii=False,
        )


# ── Tool roster ──────────────────────────────────────────────────────
# Register all tools in the order they should appear to the LLM.

ALL_TOOLS: list = [
    search_memories_tool,
    query_entity_tool,
    retrieve_chunks_tool,
    query_rewrite_and_search_tool,
    write_memory_tool,
    extract_memory_tool,
    ingest_git_repo_tool,
    ingest_document_tool,
    notify_feishu_tool,
]
