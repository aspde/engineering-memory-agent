"""LangGraph agent state schema.

Uses MessagesState pattern — messages accumulate with ID-based
deduplication via the ``add_messages`` reducer.
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State carried between agent nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    """Conversation history with automatic ID-based dedup."""

    retrieved_chunks: list[dict[str, Any]]
    """Results from chunk retrieval (populated by retrieve_chunks_tool)."""

    retrieved_memories: list[dict[str, Any]]
    """Results from memory search (populated by search_memories_tool)."""

    context_assembled: str
    """Assembled context string consumed by generate_final_node."""

    final_response: str | None
    """Final answer set by generate_final_node."""

    error: str | None
    """Error state for graceful degradation."""
