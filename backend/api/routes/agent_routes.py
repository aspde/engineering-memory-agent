"""Agent chat API routes — supports HITL interrupt/resume, streaming,
and conversation history persistence."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command
from pydantic import BaseModel, Field
from sqlalchemy import text

from backend.db import get_session_factory
from backend.service.agent_service import get_agent_for_thread
from backend.shared.config import config, current_thread_id
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


async def _is_disconnected(request: Request) -> bool:
    """True if the SSE client has gone away — stop streaming to save tokens.

    ``Request.is_disconnected`` is the FastAPI/Starlette way to detect a
    closed client connection mid-stream; it may raise on some ASGI servers,
    so a failure is treated as "still connected" rather than aborting.
    """
    try:
        return await request.is_disconnected()
    except Exception:
        return False


async def _stream_final_answer(
    request: Request, final_response: str
):
    """Replay *final_response* as SSE tokens — used on the resume path only.

    New-message runs stream the final answer live (LLM tokens pushed through
    the graph's ``custom`` stream).  A resumed run uses ``ainvoke``, which
    has no custom stream, so the already-generated answer is replayed here
    as small chunks so the frontend still shows the token animation.
    Checks per-chunk that the client is still connected and stops early if
    they closed the tab (avoids pushing tokens to a dead socket).
    """
    # Stream the response in small chunks so the frontend sees a
    # token-by-token animation (same UX as before, but deterministic).
    chunk_size = 4
    for i in range(0, len(final_response), chunk_size):
        if await _is_disconnected(request):
            logger.info("SSE client disconnected during final-answer stream — aborting")
            return
        chunk = final_response[i:i + chunk_size]
        yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"


async def _mark_interrupted_thread(agent, thread_id: str) -> None:
    """Note a timeout on the thread so a retry reads as a continuation.

    ``asyncio.timeout`` cancels the run mid-flight; LangGraph checkpoints
    per node, so the thread keeps the user's message plus whatever tool
    chatter completed before cancellation.  Without a marker, a retry of
    the same question appends a *second* identical user turn the model
    sees as an independent ask.  Appending a SystemMessage lets the model
    treat the retry as the interrupted turn continuing.  Best-effort: a
    state update failure must not break the timeout response itself.
    """
    try:
        run_config = {"configurable": {"thread_id": thread_id}}
        state = await agent.aget_state(run_config)
        if state and state.values:
            await agent.aupdate_state(
                run_config,
                {
                    "messages": [
                        SystemMessage(
                            content=(
                                "The previous agent run was interrupted by a timeout "
                                "before producing a final answer. The user may retry the "
                                "same question — treat it as a continuation of the "
                                "interrupted turn."
                            )
                        )
                    ]
                },
            )
    except Exception:
        logger.warning("Failed to mark interrupted thread %s", thread_id, exc_info=True)


# Read-only tools whose results belong in the sources panel, not the
# tool-call panel (the user doesn't approve reads, and their output is
# already reflected in the assistant's answer + source references).
_READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "search_memories_tool",
    "retrieve_chunks_tool",
})

# Write tools whose output is already confirmed by the approval flow —
# showing them in the tool-call panel adds noise, not value.
_SILENT_TOOLS: frozenset[str] = frozenset({
    "write_memory_tool",
    "extract_memory_tool",
})


def _extract_tool_traces(
    messages: list,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract tool call traces and sources from a message list.

    Read-only tools (search / retrieve) are surfaced as **sources** only.
    Write tools (write / ingest / extract) appear as **tool-call traces**
    so the user can see what was modified.
    """
    tool_call_traces: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        tool_name = getattr(m, "name", "unknown")

        # Silent tools — confirmed by approval, no need to show.
        if tool_name in _SILENT_TOOLS:
            continue

        # Chunk retrieval sources have no stable IDs — skip entirely.
        if tool_name == "retrieve_chunks_tool":
            continue

        raw = str(m.content) if m.content else ""

        # Try to extract structured sources from JSON envelope
        parsed_sources: list[dict[str, Any]] | None = None
        try:
            data = json.loads(raw)
            if isinstance(data, dict) and "sources" in data:
                parsed_sources = data["sources"]
                display = data.get("display", raw)
            else:
                display = raw
        except (json.JSONDecodeError, TypeError):
            display = raw

        # Only write / ingest tools go to the tool-call panel.  Read
        # results (search_memories_tool) are shown as clickable sources.
        if tool_name not in _READ_ONLY_TOOLS:
            tool_call_traces.append({
                "tool": tool_name,
                "content": display[:300],
            })

        if parsed_sources is not None:
            sources.extend(parsed_sources)
        elif tool_name == "search_memories_tool":
            if raw.strip() and raw.strip() != "No relevant memories found.":
                sources.append({"type": "memory", "snippet": raw[:200]})
        elif tool_name not in _READ_ONLY_TOOLS and raw.strip():
            sources.append({"type": "unknown", "snippet": raw[:200]})

    # Deduplicate sources by id (same memory may be returned by multiple
    # search calls in a single ReAct loop).  Sources without an id are
    # kept as-is (legacy / fallback entries).
    seen_ids: set[str] = set()
    unique_sources: list[dict[str, Any]] = []
    for s in sources:
        sid = s.get("id")
        if sid is not None:
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
        unique_sources.append(s)
    return tool_call_traces, unique_sources


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


class ThreadDeleteResponse(BaseModel):
    thread_id: str
    deleted: bool


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

    # Build displayed messages in "turns": each HumanMessage starts a new
    # turn.  Tool-call traces and sources extracted from ToolMessages
    # within a turn are attached to that turn's final assistant message.
    displayed: list[dict[str, Any]] = []
    raw_messages = state.values.get("messages", [])

    # Collect raw messages belonging to the current turn.
    turn_buf: list = []

    def _flush_turn() -> None:
        """Process the accumulated *turn_buf* into *displayed*."""
        if not turn_buf:
            return

        # Extract tool traces + sources only from this turn's ToolMessages.
        turn_traces, turn_sources = _extract_tool_traces(turn_buf)

        for m in turn_buf:
            # SystemMessages are agent instructions, not user-visible content.
            if isinstance(m, SystemMessage):
                continue
            if isinstance(m, ToolMessage):
                continue
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                continue

            role = "user" if isinstance(m, HumanMessage) else "assistant"
            msg_dict: dict[str, Any] = {
                "role": role,
                "content": str(m.content) if m.content else "",
            }
            # Deduplicate consecutive assistant messages within the turn.
            if role == "assistant" and displayed and displayed[-1]["role"] == "assistant":
                displayed[-1] = msg_dict
            else:
                displayed.append(msg_dict)

        # Attach this turn's traces & sources to its last assistant msg.
        if turn_traces or turn_sources:
            for i in range(len(displayed) - 1, -1, -1):
                if displayed[i]["role"] == "assistant":
                    if turn_traces:
                        displayed[i]["tool_calls"] = turn_traces
                    if turn_sources:
                        displayed[i]["sources"] = turn_sources
                    break

        turn_buf.clear()

    for m in raw_messages:
        if isinstance(m, HumanMessage) and turn_buf:
            # A new user message starts a fresh turn.
            _flush_turn()
        turn_buf.append(m)

    _flush_turn()
    messages = displayed

    return ThreadMessagesResponse(thread_id=thread_id, messages=messages)


@router.delete("/thread/{thread_id}", response_model=ThreadDeleteResponse)
async def delete_thread(thread_id: str) -> ThreadDeleteResponse:
    """Delete a conversation and its checkpoint data.

    Returns 404 if the conversation does not exist.
    """
    session_factory = get_session_factory()

    # Verify the conversation exists (outside the write-session to
    # avoid __aexit__ errors swallowing HTTPException on 404).
    async with session_factory() as ro:
        r = await ro.execute(
            text("SELECT 1 FROM conversations WHERE thread_id = :tid"),
            {"tid": thread_id},
        )
        if r.fetchone() is None:
            raise HTTPException(status_code=404, detail="Conversation not found")

    try:
        async with session_factory() as session:
            # Delete checkpoint data — these tables may not exist when
            # the checkpointer fell back to InMemorySaver (e.g. database
            # unreachable, or Windows ProactorEventLoop incompatibility).
            for tbl in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                try:
                    await session.execute(
                        text(f"DELETE FROM {tbl} WHERE thread_id = :tid"),
                        {"tid": thread_id},
                    )
                except Exception:
                    # Table may not exist (e.g. InMemorySaver on Windows).
                    # Roll back the aborted sub-transaction so subsequent
                    # statements in this session aren't blocked.
                    await session.rollback()

            # Delete the conversation record
            await session.execute(
                text("DELETE FROM conversations WHERE thread_id = :tid"),
                {"tid": thread_id},
            )
            await session.commit()
    except Exception as exc:
        logger.warning("Failed to delete thread %s: %s", thread_id, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ThreadDeleteResponse(thread_id=thread_id, deleted=True)


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
    run_config = {"configurable": {"thread_id": req.thread_id}}

    # Tag memories written during this turn with the conversation thread.
    current_thread_id.set(req.thread_id)

    # Record this conversation as active
    title = req.message[:80] if req.message else ""
    if req.resume_data is None and title:
        await _upsert_conversation(req.thread_id, title)

    t0 = time.perf_counter()
    try:
        async with asyncio.timeout(config.agent_timeout):
            if req.resume_data is not None:
                result = await agent.ainvoke(
                    Command(resume=req.resume_data),
                    config=run_config,
                )
            else:
                result = await agent.ainvoke(
                    {"messages": [HumanMessage(content=req.message)]},
                    config=run_config,
                )
    except TimeoutError:
        t1 = time.perf_counter()
        logger.warning(
            "agent_chat timed out after %ds (%.0fms) thread_id=%s",
            config.agent_timeout, (t1 - t0) * 1000, req.thread_id,
        )
        await _mark_interrupted_thread(agent, req.thread_id)
        return ChatResponse(
            thread_id=req.thread_id,
            status="error",
            response=(
                f"Agent 处理超时（超过 {config.agent_timeout} 秒），已停止本轮处理，"
                "请重试或简化问题。"
            ),
        )
    except Exception as exc:
        t1 = time.perf_counter()
        logger.warning("agent_chat failed in %.0fms: %s", (t1 - t0) * 1000, exc)
        return ChatResponse(
            thread_id=req.thread_id,
            status="error",
            response=f"Agent error: {exc}",
        )

    # Check for interrupt first
    interrupts = result.get("__interrupt__")
    if interrupts:
        t1 = time.perf_counter()
        logger.info(
            "agent_chat interrupted in %.0fms thread_id=%s",
            (t1 - t0) * 1000, req.thread_id,
        )
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

    # Count AIMessages with tool_calls to estimate ReAct steps
    tool_call_count = sum(
        1 for m in result.get("messages", [])
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
    )

    t1 = time.perf_counter()
    logger.info(
        "agent_chat completed in %.0fms tool_calls=%d traces=%d thread_id=%s msg=%r",
        (t1 - t0) * 1000, tool_call_count, len(tool_call_traces),
        req.thread_id, (req.message or "")[:60],
    )

    return ChatResponse(
        thread_id=req.thread_id,
        status="completed",
        response=final_response,
        tool_calls=tool_call_traces,
        sources=sources,
    )


@router.post("/chat/stream")
async def agent_chat_stream(req: ChatRequest, request: Request):
    """Send a message and stream the response via Server-Sent Events.

    Yields ``data: {"type":"node","node":"..."}`` for each graph node
    that completes, ``data: {"type":"token","content":"..."}`` for each
    token of the final answer, and ``data: {"type":"interrupt",...}``
    when the agent pauses for human approval.

    The client can use ``event:`` lines to route different event types.
    The whole run is bounded by ``AGENT_TIMEOUT``; a disconnected SSE
    client aborts the stream so later tool/LLM steps don't burn tokens.
    """
    agent = get_agent_for_thread()
    run_config = {"configurable": {"thread_id": req.thread_id}}

    # Tag memories written during this turn with the conversation thread.
    current_thread_id.set(req.thread_id)

    # Record this conversation as active
    title = req.message[:80] if req.message else ""
    if req.resume_data is None and title:
        await _upsert_conversation(req.thread_id, title)

    async def _stream():
        try:
            async with asyncio.timeout(config.agent_timeout):
                if req.resume_data is not None:
                    # Resume: use ainvoke (interrupt/resume doesn't stream nodes cleanly)
                    result = await agent.ainvoke(
                        Command(resume=req.resume_data),
                        config=run_config,
                    )
                    # Check interrupt after resume
                    interrupts = result.get("__interrupt__")
                    if interrupts:
                        payload = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
                        yield f"data: {json.dumps({'type': 'interrupt', 'data': payload}, ensure_ascii=False)}\n\n"
                        return

                    # Client gone while the resume run was in flight — don't push to a dead socket.
                    if await _is_disconnected(request):
                        return

                    # Stream the final answer from the state
                    final_response = result.get("final_response")
                    if final_response:
                        async for sse_line in _stream_final_answer(request, final_response):
                            yield sse_line

                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    return

                # New message: stream through the graph.
                # Two modes:
                #   updates — node-completion events, interrupts
                #   custom  — live LLM tokens pushed from inside the nodes
                #             via get_stream_writer() (real streaming; the
                #             final answer is generated, not replayed).
                # With subgraphs=True + a mode list, events arrive as
                # (namespace, mode, data) tuples.
                async for _, mode, event_data in agent.astream(
                    {"messages": [HumanMessage(content=req.message)]},
                    config=run_config,
                    stream_mode=["updates", "custom"],
                    subgraphs=True,
                ):
                    # Client disconnected — abort so later tool/LLM steps
                    # don't burn tokens for nobody.
                    if await _is_disconnected(request):
                        logger.info("SSE client disconnected mid-run — aborting agent stream")
                        return

                    if mode == "custom":
                        # Token delta emitted by call_llm_node /
                        # generate_final_node while the LLM generates.
                        if isinstance(event_data, dict) and event_data.get("type") == "token":
                            yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                        continue

                    # mode == "updates": node completions + interrupts.
                    # Check for interrupts
                    if event_data and "__interrupt__" in event_data:
                        interrupts = event_data["__interrupt__"]
                        payload = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
                        yield f"data: {json.dumps({'type': 'interrupt', 'data': payload}, ensure_ascii=False)}\n\n"
                        return

                    # Node completion events.  The final answer's tokens
                    # already arrived via the custom stream while the LLM
                    # was generating — nothing to replay here.
                    for node_name, node_state in (event_data or {}).items():
                        yield f"data: {json.dumps({'type': 'node', 'node': node_name}, ensure_ascii=False)}\n\n"

                # Send tool traces fetched from the final graph state
                try:
                    final_state = await agent.aget_state(run_config)
                    if final_state and final_state.values:
                        tool_traces, sources = _extract_tool_traces(
                            final_state.values.get("messages", [])
                        )
                        if tool_traces or sources:
                            yield f"data: {json.dumps({'type': 'meta', 'tool_calls': tool_traces, 'sources': sources}, ensure_ascii=False)}\n\n"
                except Exception:
                    pass

                yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except TimeoutError:
            logger.warning(
                "agent_chat_stream timed out after %ds thread_id=%s",
                config.agent_timeout, req.thread_id,
            )
            await _mark_interrupted_thread(agent, req.thread_id)
            yield f"data: {json.dumps({'type': 'error', 'message': f'Agent 处理超时（超过 {config.agent_timeout} 秒），已停止本轮处理，请重试或简化问题。'}, ensure_ascii=False)}\n\n"
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


# ── Cost / observability endpoints ───────────────────────────────────


@router.get("/usage")
async def get_token_usage_endpoint() -> dict[str, Any]:
    """Return cumulative LLM token usage broken down by scenario.

    Scenarios are tagged at each LLM call site via the ``scenario=``
    kwarg and include:
      - ``agent_chat`` — ReAct LLM calls (with tools)
      - ``agent_final`` — final answer generation
      - ``extraction_summary`` / ``extraction_entities`` / ``extraction_relations``
      - ``conflict_detection`` — memory conflict check (0.75-0.92 band)
      - ``memory_merge`` — summary merge (≥0.92 band)
      - ``entity_normalization`` — LLM entity-match judgement
      - ``rerank_llm`` — LLM-based rerank (when ``use_llm_rerank=True``)

    To start a fresh measurement window during an interview demo, call
    ``POST /api/agent/usage/reset`` to clear counters.

    ``structured_failures`` counts structured-output calls (entity/relation
    extraction) that exhausted their retries and degraded to empty — so
    silent degradation is observable.
    """
    from backend.shared.metrics import get_structured_failures, get_token_usage

    usage = get_token_usage()
    total = sum(usage.values())
    return {
        "total_tokens": total,
        "by_scenario": usage,
        "scenarios": len(usage),
        "structured_failures": get_structured_failures(),
    }


@router.post("/usage/reset")
async def reset_token_usage_endpoint() -> dict[str, Any]:
    """Reset all usage counters (tokens + structured-output failures)."""
    from backend.shared.metrics import reset_structured_failures, reset_token_usage

    reset_token_usage()
    reset_structured_failures()
    return {"reset": True}
