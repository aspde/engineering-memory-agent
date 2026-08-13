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

    step_count: int | None
    """Number of ``call_llm`` invocations in the current conversation turn.
    Used with ``MAX_AGENT_STEPS`` to bound the ReAct loop.  ``None`` until
    the first ``call_llm`` pass.  Restarts from 0 when a fresh HumanMessage
    arrives (a new turn); loopbacks within a turn keep counting."""

    final_response: str | None
    """Final answer set by generate_final_node.  Must be Optional because
    the initial state has no answer yet — only the terminal node sets it."""

    final_prompt: list[dict[str, str]] | None
    """Prompt messages built by generate_final_node for the final answer.

    Kept for audit / replay (e.g. the resume path replays a completed
    answer).  Live streaming no longer reads this field: the nodes stream
    LLM tokens directly through the graph's ``custom`` stream
    (``get_stream_writer``)."""

    error: str | None
    """Error state for graceful degradation.  Set by any node that catches
    an unrecoverable exception so the caller can inspect what went wrong."""

    pending_approval: dict[str, Any] | None
    """Non-None when the graph is paused waiting for human approval.
    Set by check_approval_node before ``interrupt()``; cleared on resume."""

    force_write: bool | None
    """True when this turn's user message must be written to the memory store
    (the frontend's 强制写入记忆 checkbox).  ``call_llm_node`` injects a
    ``write_memory_tool`` call into the AIMessage it returns and clears the
    flag, so the write runs through the normal ReAct pipeline (ToolNode +
    check_conflict) exactly once.  ``None``/``False`` = no forced write."""
