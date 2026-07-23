"""Agent chat API routes — supports HITL interrupt/resume, streaming,
and conversation history persistence."""

from __future__ import annotations

import json
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field
from sqlalchemy import text

from backend.db import get_session_factory
from backend.service.agent_service import get_agent_for_thread
from backend.service.llm_service import get_llm_provider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agent", tags=["agent"])


# ── Conversation persistence ──────────────────────────────────────────


async def _upsert_conversation(thread_id: str, title: str = "") -> None:
    """Insert or update a conversation row with *title*."""
    try:
        async with get_session_factory()() as session:
            await session.execute(
                text(
                    "INSERT INTO conversations (thread_id, title, updated_at) "
                    "VALUES (:tid, :title, now()) "
                    "ON CONFLICT (thread_id) DO UPDATE SET "
                    "title = COALESCE(NULLIF(:title, ''), conversations.title), "
                    "updated_at = now()"
                ),
                {"tid": thread_id, "title": title},
            )
            await session.commit()
    except Exception:
        logger.warning("Failed to upsert conversation", exc_info=True)


# ── Helpers ────────────────────────────────────────────────────────────


async def _stream_final_answer(
    agent, config: dict, final_prompt: list[dict[str, str]]
):
    """Stream tokens from *final_prompt* via SSE, then persist into the graph."""
    provider = get_llm_provider()
    full_text = ""
    async for token in provider.chat_stream(final_prompt):
        full_text += token
        yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"

    await agent.aupdate_state(
        config,
        {"final_response": full_text, "messages": [AIMessage(content=full_text)]},
    )


def _extract_tool_traces(
    messages: list,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract tool call traces and sources from a message list."""
    tool_call_traces: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for m in messages:
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
    return tool_call_traces, sources


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


class ThreadInfo(BaseModel):
    thread_id: str
    title: str


class ThreadMessagesResponse(BaseModel):
    thread_id: str
    messages: list[dict[str, Any]] = Field(default_factory=list)


# ── Conversation history routes ──────────────────────────────────────


@router.get("/threads", response_model=list[ThreadInfo])
async def list_threads() -> list[ThreadInfo]:
    """Return conversation history from the ``conversations`` table."""
    try:
        async with get_session_factory()() as session:
            rows = await session.execute(
                text(
                    "SELECT thread_id, title FROM conversations "
                    "ORDER BY updated_at DESC LIMIT 50"
                )
            )
            return [
                ThreadInfo(thread_id=row[0], title=row[1] or row[0][:8])
                for row in rows.fetchall()
            ]
    except Exception:
        logger.warning("Failed to query conversations", exc_info=True)
        return []


@router.get("/thread/{thread_id}", response_model=ThreadMessagesResponse)
async def get_thread_messages(thread_id: str) -> ThreadMessagesResponse:
    """Return the message history for a given *thread_id* from checkpoint state."""
    agent = get_agent_for_thread()
    try:
        state = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    except Exception:
        return ThreadMessagesResponse(thread_id=thread_id, messages=[])

    if not state or not state.values:
        return ThreadMessagesResponse(thread_id=thread_id, messages=[])

    messages: list[dict[str, Any]] = []
    for m in state.values.get("messages", []):
        role: str = "assistant"
        if isinstance(m, HumanMessage):
            role = "user"
        elif isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, ToolMessage):
            role = "system"
        msg_dict: dict[str, Any] = {
            "role": role,
            "content": str(m.content) if m.content else "",
        }
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            msg_dict["_meta"] = {
                "tool_calls": [
                    {"tool": tc.get("name", ""), "content": str(tc.get("args", ""))[:200]}
                    for tc in m.tool_calls
                ]
            }
        messages.append(msg_dict)

    return ThreadMessagesResponse(thread_id=thread_id, messages=messages)


# ── Chat routes ──────────────────────────────────────────────────────


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

    # Record this conversation as active
    title = req.message[:80] if req.message else ""
    if req.resume_data is None and title:
        await _upsert_conversation(req.thread_id, title)

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
    tool_call_traces, sources = _extract_tool_traces(result.get("messages", []))

    return ChatResponse(
        thread_id=req.thread_id,
        status="completed",
        response=final_response,
        tool_calls=tool_call_traces,
        sources=sources,
    )


@router.post("/chat/stream")
async def agent_chat_stream(req: ChatRequest):
    """Send a message and stream the response via Server-Sent Events.

    Yields ``data: {"type":"node","node":"..."}`` for each graph node
    that completes, ``data: {"type":"token","content":"..."}`` for each
    token of the final answer, and ``data: {"type":"interrupt",...}``
    when the agent pauses for human approval.

    The client can use ``event:`` lines to route different event types.
    """
    agent = get_agent_for_thread()
    config = {"configurable": {"thread_id": req.thread_id}}

    # Record this conversation as active
    title = req.message[:80] if req.message else ""
    if req.resume_data is None and title:
        await _upsert_conversation(req.thread_id, title)

    async def _stream():
        try:
            if req.resume_data is not None:
                # Resume: use ainvoke (interrupt/resume doesn't stream nodes cleanly)
                result = await agent.ainvoke(
                    Command(resume=req.resume_data),
                    config=config,
                )
                # Check interrupt after resume
                interrupts = result.get("__interrupt__")
                if interrupts:
                    payload = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
                    yield f"data: {json.dumps({'type': 'interrupt', 'data': payload}, ensure_ascii=False)}\n\n"
                    return

                # Stream the final answer from the state
                final_prompt = result.get("final_prompt")
                if final_prompt:
                    async for sse_line in _stream_final_answer(agent, config, final_prompt):
                        yield sse_line

                yield f"data: {json.dumps({'type': 'done'})}\n\n"
                return

            # New message: stream through the graph
            async for _, event_data in agent.astream(
                {"messages": [HumanMessage(content=req.message)]},
                config=config,
                stream_mode="updates",
                subgraphs=True,
            ):
                # Check for interrupts
                if event_data and "__interrupt__" in event_data:
                    interrupts = event_data["__interrupt__"]
                    payload = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
                    yield f"data: {json.dumps({'type': 'interrupt', 'data': payload}, ensure_ascii=False)}\n\n"
                    return

                # Node completion events
                for node_name, node_state in (event_data or {}).items():
                    yield f"data: {json.dumps({'type': 'node', 'node': node_name}, ensure_ascii=False)}\n\n"

                    if node_name == "generate_final":
                        final_prompt = node_state.get("final_prompt")
                        if final_prompt:
                            async for sse_line in _stream_final_answer(agent, config, final_prompt):
                                yield sse_line

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as exc:
            logger.exception("Streaming error")
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
