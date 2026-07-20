"""Agent chat API routes — supports HITL interrupt/resume."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from fastapi import APIRouter
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field

from backend.service.agent_service import get_agent_for_thread

router = APIRouter(prefix="/agent", tags=["agent"])


# ── Request / Response models ────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=10000)
    thread_id: str = Field(default_factory=lambda: str(uuid4()))
    resume_data: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    thread_id: str
    status: str  # "completed" | "interrupted" | "error"
    response: str = ""
    interrupt: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)


# ── Routes ───────────────────────────────────────────────────────────


@router.post("/chat", response_model=ChatResponse)
async def agent_chat(req: ChatRequest) -> ChatResponse:
    """Send a message to the EMA agent and receive a response.

    The agent autonomously decides which tools to call (memory search,
    document retrieval, ingestion, etc.) based on the message content.

    Provide a *thread_id* to continue an existing conversation.
    When the agent pauses for human approval, the response has
    ``status="interrupted"`` with an ``interrupt`` payload.  Send a new
    request with the same *thread_id* and ``resume_data`` (e.g.
    ``{"approved": true}`` or ``{"approved": false, "reason": "..."}``)
    to resume.
    """
    agent = get_agent_for_thread()
    config = {"configurable": {"thread_id": req.thread_id}}

    try:
        if req.resume_data is not None:
            result = await agent.ainvoke(
                Command(resume=req.resume_data),
                config=config,
            )
        else:
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=req.message)]},
                config=config,
            )
    except Exception as exc:
        return ChatResponse(
            thread_id=req.thread_id,
            status="error",
            response=f"Agent error: {exc}",
        )

    # Check for interrupt first
    interrupts = result.get("__interrupt__")
    if interrupts:
        interrupt_payload = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
        return ChatResponse(
            thread_id=req.thread_id,
            status="interrupted",
            interrupt=interrupt_payload,
        )

    # Normal completion
    final_response: str = result.get("final_response", "") or ""
    if not final_response:
        for m in reversed(result.get("messages", [])):
            if (
                hasattr(m, "content")
                and m.content
                and not getattr(m, "tool_calls", None)
            ):
                final_response = str(m.content)
                break

    # Extract tool call traces and sources in a single pass
    tool_call_traces: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for m in result.get("messages", []):
        if not isinstance(m, ToolMessage):
            continue
        tool_name = getattr(m, "name", "unknown")
        tool_call_traces.append({
            "tool": tool_name,
            "content": str(m.content)[:300],
        })
        if tool_name == "search_memories_tool":
            source_type = "memory"
        elif tool_name == "retrieve_chunks_tool":
            source_type = "chunk"
        else:
            source_type = "unknown"
        content = str(m.content)[:200] if m.content else ""
        if content:
            sources.append({"type": source_type, "snippet": content})

    return ChatResponse(
        thread_id=req.thread_id,
        status="completed",
        response=final_response,
        tool_calls=tool_call_traces,
        sources=sources,
    )
