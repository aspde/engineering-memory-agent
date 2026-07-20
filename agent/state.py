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

    final_response: str | None
    """Final answer set by generate_final_node.  Must be Optional because
    the initial state has no answer yet — only the terminal node sets it."""

    error: str | None
    """Error state for graceful degradation.  Set by any node that catches
    an unrecoverable exception so the caller can inspect what went wrong."""

    pending_approval: dict[str, Any] | None
    """Non-None when the graph is paused waiting for human approval.
    Set by check_approval_node before ``interrupt()``; cleared on resume."""
