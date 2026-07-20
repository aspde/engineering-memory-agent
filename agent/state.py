"""LangGraph agent state schema.

Uses MessagesState pattern — messages accumulate with ID-based
deduplication via the ``add_messages`` reducer.

Frozen — do not add new fields without a concrete consumer node.
"""

from __future__ import annotations

from typing import Annotated

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State carried between agent nodes."""

    messages: Annotated[list[BaseMessage], add_messages]
    """Conversation history with automatic ID-based dedup."""

    final_response: str | None
    """Final answer set by generate_final_node."""

    error: str | None
    """Error state for graceful degradation."""
