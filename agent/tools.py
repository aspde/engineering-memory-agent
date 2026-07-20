"""Agent tool definitions — thin wrappers around existing backend services.

Each tool is an async function decorated with ``@tool`` so LangGraph's
``ToolNode`` can auto-generate schemas and execute tool calls.  Tools
return formatted strings because that's what the LLM reads via
``ToolMessage.content``.
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool

from backend.service.chunk import chunk_text
from backend.service.extraction import extract_memory
from backend.service.ingestion import ingest_repo
from backend.service.memory import write_memory
from backend.service.retrieval import query_memories, retrieve, write_chunks


# ── Retrieval tools ──────────────────────────────────────────────────


@tool
async def search_memories_tool(
    query: str,
    top_k: int = 5,
    use_llm_rerank: bool = False,
) -> str:
    """Search long-term engineering memories for knowledge, decisions,
    lessons learned, and past context.

    Use this when the user asks about project history, technical
    decisions, architecture, past discussions, or anything that might
    have been recorded as a memory.

    Args:
        query: Natural-language search query.
        top_k: Number of results (1-20).
        use_llm_rerank: Set to True for LLM-based relevance scoring.
    """
    results = await query_memories(
        query, top_k=min(top_k, 20), use_llm_rerank=use_llm_rerank
    )
    if not results:
        return "No relevant memories found."

    lines = [f"Found {len(results)} relevant memories:"]
    for i, r in enumerate(results):
        score = r.get("rerank_score", r.get("weighted_score", 0))
        decay = r.get("decay_factor", 1.0)
        lines.append(
            f"[{i + 1}] (relevance: {score:.2f}, decay: {decay:.2f}) "
            f"{r['summary']}"
        )
    return "\n".join(lines)


@tool
async def retrieve_chunks_tool(
    query: str,
    top_k: int = 5,
    use_llm_rerank: bool = False,
) -> str:
    """Semantic search over ingested document chunks (code, docs, etc.).

    Use this when the user asks about content that was ingested as
    documents, or when memory search doesn't return enough context.

    Args:
        query: Natural-language search query.
        top_k: Number of results (1-20).
        use_llm_rerank: Set to True for LLM-based relevance scoring.
    """
    results = await retrieve(
        query, top_k=min(top_k, 20), use_llm_rerank=use_llm_rerank
    )
    if not results:
        return "No relevant document chunks found."

    lines = [f"Found {len(results)} relevant chunks:"]
    for i, r in enumerate(results):
        lines.append(f"[{i + 1}] (relevance: {r.score:.2f}) {r.content}")
    return "\n".join(lines)


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


# ── Ingestion tools ──────────────────────────────────────────────────


@tool
async def ingest_git_repo_tool(
    repo_path: str,
    max_commits: int = 50,
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
    metadata: dict[str, Any] | None = None,
) -> str:
    """Chunk, embed, and store a document for later retrieval.

    Splits content into overlapping chunks, computes embeddings, and
    stores them in the vector database.  Use this when the user asks
    to index a document, code file, or any text for future search.

    Args:
        document_id: Unique identifier for the document (e.g. filename).
        content: Full text content of the document.
        metadata: Optional metadata (source, language, author, etc.).
    """
    chunks = chunk_text(content)
    count = await write_chunks(document_id, chunks, meta=metadata)
    return f"Ingested {count} chunks from document '{document_id}'."


# ── Tool roster ──────────────────────────────────────────────────────
# Register all tools in the order they should appear to the LLM.

ALL_TOOLS: list = [
    search_memories_tool,
    retrieve_chunks_tool,
    write_memory_tool,
    extract_memory_tool,
    ingest_git_repo_tool,
    ingest_document_tool,
]
