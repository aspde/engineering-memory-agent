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

from backend.agent.nodes import CHAT_APPROVAL_TOOLS
from backend.agent.tool_envelope import parse_tool_envelope
from backend.db import get_session_factory
from backend.service.agent_service import (
    CHAT_LLM_TOOLS,
    _release_agent_slot,
    _try_acquire_agent_slot,
    get_agent_for_thread,
)
from backend.shared.config import config, current_thread_id, current_trace_id

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
    "query_rewrite_and_search_tool",
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

        raw = str(m.content) if m.content else ""

        # Try to extract structured sources from the tool-result JSON envelope
        # (shared parser — backend.agent.tool_envelope).  Non-envelope results
        # (plain text, write/ingest/entity JSON) fall back to the raw text.
        parsed_sources: list[dict[str, Any]] | None = None
        envelope = parse_tool_envelope(raw)
        if envelope is not None:
            parsed_sources = envelope.get("sources")
            display = str(envelope.get("display") or raw)
        else:
            display = raw

        # Only write / ingest tools go to the tool-call panel.  Read
        # results (search_memories_tool / retrieve_chunks_tool /
        # query_rewrite_and_search_tool) are shown as sources — chunk
        # results carry document_id so the answer's inline citations are
        # verifiable in the sources panel.
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

    # Deduplicate sources: memories by id (same memory returned by multiple
    # search calls in a single ReAct loop), chunks by (document_id,
    # chunk_index) since they carry no stable id.  Sources with neither key
    # are kept as-is (legacy / fallback entries).
    seen_ids: set[str] = set()
    seen_chunks: set[tuple[str, int]] = set()
    unique_sources: list[dict[str, Any]] = []
    for s in sources:
        sid = s.get("id")
        if sid is not None:
            if sid in seen_ids:
                continue
            seen_ids.add(sid)
        if s.get("type") == "chunk":
            doc = s.get("document_id")
            cidx = s.get("chunk_index")
            if doc is not None and cidx is not None:
                try:
                    key = (str(doc), int(cidx))
                except (TypeError, ValueError):
                    key = None
                if key is not None:
                    if key in seen_chunks:
                        continue
                    seen_chunks.add(key)
        unique_sources.append(s)
    return tool_call_traces, unique_sources


def _extract_write_result(messages: list) -> dict[str, str] | None:
    """Return ``{"action", "summary"}`` from this turn's last
    ``write_memory_tool`` result, or ``None`` when this turn wrote nothing.

    Drives the frontend's force-write toast (inserted / merged / conflict).
    Only the current turn counts — a previous turn's write must not re-trigger
    a toast on a later reply.  A rejected/cancelled write is a non-JSON
    ``[REJECTED]``/``[CANCELLED]`` ToolMessage and yields ``None``.
    """
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return None
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "write_memory_tool":
            try:
                data = json.loads(str(m.content))
            except (TypeError, ValueError):
                return None
            if not isinstance(data, dict):
                return None  # valid JSON but not a write result (e.g. a list)
            action = data.get("action")
            if not isinstance(action, str):
                return None
            summary = data.get("summary", "")
            return {"action": action, "summary": str(summary) if summary else ""}
    return None


# ── Request / Response models ────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str = Field(default="", max_length=10000)
    thread_id: str = Field(default_factory=lambda: str(uuid4()))
    resume_data: dict[str, Any] | None = None
    force_write: bool = False
    """User checked 记住这条 — the turn's message is written to the
    memory store regardless of the model's judgement (the write is injected
    into the agent run, not a separate REST call)."""


class ChatResponse(BaseModel):
    thread_id: str
    status: str  # "completed" | "interrupted" | "error"
    response: str = ""
    interrupt: dict[str, Any] | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    memory_write: dict[str, Any] | None = None
    """``{"action", "summary"}`` when this turn's force-write actually wrote
    (inserted / merged / conflict), else ``None``."""


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
    agent = get_agent_for_thread(
        approval_required_tools=CHAT_APPROVAL_TOOLS,
        llm_tools=CHAT_LLM_TOOLS,
    )
    run_config = {"configurable": {"thread_id": req.thread_id}}

    # Tag memories written during this turn with the conversation thread.
    # Concurrency cap: beyond MAX_AGENT_CONCURRENCY simultaneous runs the
    # request is refused (503), not queued — a queued run would sit behind
    # long ReAct loops that each hold LLM slots for up to AGENT_TIMEOUT.
    # Checked before any contextvar/DB setup so a refused request costs
    # nothing (no conversation upsert, no trace/thread stamps).
    if not _try_acquire_agent_slot():
        logger.warning(
            "agent_chat refused — concurrency cap reached (max=%d) thread_id=%s",
            config.max_agent_concurrency, req.thread_id,
        )
        raise HTTPException(
            status_code=503,
            detail="系统繁忙，当前同时处理的会话数已达上限，请稍后重试。",
        )

    current_thread_id.set(req.thread_id)

    # Link every LLM call in this run to one trace id (usage observability).
    current_trace_id.set(str(uuid4()))

    # Record this conversation as active
    title = req.message[:80] if req.message else ""
    if req.resume_data is None and title:
        await _upsert_conversation(req.thread_id, title)

    t0 = time.perf_counter()
    try:
        async with asyncio.timeout(config.agent_timeout):
            if req.resume_data is not None:
                result = await agent.ainvoke(  # type: ignore[call-overload]
                    Command(resume=req.resume_data),
                    config=run_config,
                )
            else:
                result = await agent.ainvoke(  # type: ignore[call-overload]
                    {
                        "messages": [HumanMessage(content=req.message)],
                        "force_write": req.force_write,
                    },
                    config=run_config,
                )
    except TimeoutError:
        t1 = time.perf_counter()
        logger.warning(
            "agent_chat timed out after %ds (%.0fms) thread_id=%s",
            config.agent_timeout, (t1 - t0) * 1000, req.thread_id,
        )
        await _mark_interrupted_thread(agent, req.thread_id)
        # 504 (not 200): the whole ReAct run exceeded its deadline.  The body
        # keeps FastAPI's {"detail": ...} shape so the frontend's error handler
        # shows the Chinese message without leaking internals.
        raise HTTPException(
            status_code=504,
            detail=(
                f"Agent 处理超时（超过 {config.agent_timeout} 秒），已停止本轮处理，"
                "请重试或简化问题。"
            ),
        ) from None
    except Exception:
        # Internal exception text (provider keys, DB URLs, stack details) must
        # not leak to the client — log it server-side and return a generic
        # message.  The streaming path streams its own error text; this is the
        # non-streaming fallback.
        t1 = time.perf_counter()
        logger.exception(
            "agent_chat failed in %.0fms thread_id=%s msg=%r",
            (t1 - t0) * 1000,
            req.thread_id,
            (req.message or "")[:60],
        )
        # 502 (not 200): the agent run failed server-side; exception text stays
        # in the log, the client gets a generic message via the error body.
        raise HTTPException(
            status_code=502,
            detail="Agent 处理出错，请稍后重试或简化问题。",
        ) from None
    finally:
        # Release the concurrency slot on every exit path — success, timeout,
        # or internal error.  The graph run is complete at this point; the
        # interrupt/response processing below is pure CPU and holds no slot.
        _release_agent_slot()

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
    memory_write = _extract_write_result(result.get("messages", []))

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

    # Observe the ReAct loop length — the live-traffic view of the over-call
    # signal the offline task eval measures (agent_steps histogram).
    try:
        from backend.shared.runtime_metrics import observe_agent_steps

        observe_agent_steps(result.get("step_count") or 0)
    except Exception:
        pass

    return ChatResponse(
        thread_id=req.thread_id,
        status="completed",
        response=final_response,
        tool_calls=tool_call_traces,
        sources=sources,
        memory_write=memory_write,
    )


@router.post("/chat/stream")
async def agent_chat_stream(req: ChatRequest, request: Request):
    """Send a message and stream the response via Server-Sent Events.

    Yields ``data: {"type":"node","node":"..."}`` for each graph node
    that completes, ``data: {"type":"token","content":"..."}`` for each
    token of the final answer, and ``data: {"type":"interrupt",...}``
    when the agent pauses for human approval.

    A ``resume_data`` body continues an interrupted run through the same
    stream pipeline: the resumed nodes emit live token deltas via the
    graph's ``custom`` stream, so resume feels identical to the first send.

    The client can use ``event:`` lines to route different event types.
    The whole run is bounded by ``AGENT_TIMEOUT``; a disconnected SSE
    client aborts the stream so later tool/LLM steps don't burn tokens.
    """
    agent = get_agent_for_thread(
        approval_required_tools=CHAT_APPROVAL_TOOLS,
        llm_tools=CHAT_LLM_TOOLS,
    )
    run_config = {"configurable": {"thread_id": req.thread_id}}

    # Tag memories written during this turn with the conversation thread.
    # Concurrency cap (same policy as the non-streaming /chat): acquire a slot
    # before the stream starts so the 503 is a plain HTTP error, not an SSE
    # event.  The slot is held for the whole stream and released in _stream's
    # finally — a disconnected client closes the generator, which runs it.
    # Checked before any contextvar/DB setup so a refused request costs
    # nothing (no conversation upsert, no trace/thread stamps).
    if not _try_acquire_agent_slot():
        logger.warning(
            "agent_chat_stream refused — concurrency cap reached (max=%d) thread_id=%s",
            config.max_agent_concurrency, req.thread_id,
        )
        raise HTTPException(
            status_code=503,
            detail="系统繁忙，当前同时处理的会话数已达上限，请稍后重试。",
        )

    current_thread_id.set(req.thread_id)

    # Link every LLM call in this run to one trace id (usage observability).
    current_trace_id.set(str(uuid4()))

    # Record this conversation as active
    title = req.message[:80] if req.message else ""
    if req.resume_data is None and title:
        await _upsert_conversation(req.thread_id, title)

    async def _stream():
        try:
            async with asyncio.timeout(config.agent_timeout):
                if req.resume_data is not None:
                    # Resume: continue the interrupted run through the same
                    # stream pipeline as a new message.  Nodes on the resume
                    # path (check_approval / check_conflict → tools →
                    # call_llm → generate_final) push LLM tokens through the
                    # graph's custom stream via get_stream_writer(), so the
                    # client sees live token deltas — not a replayed answer.
                    graph_input: dict | Command = Command(resume=req.resume_data)
                else:
                    graph_input = {
                        "messages": [HumanMessage(content=req.message)],
                        "force_write": req.force_write,
                    }

                # Stream through the graph.  Two modes:
                #   updates — node-completion events, interrupts
                #   custom  — live LLM tokens pushed from inside the nodes
                #             via get_stream_writer() (real streaming; the
                #             final answer is generated, not replayed).
                # With subgraphs=True + a mode list, events arrive as
                # (namespace, mode, data) tuples.
                stream = agent.astream(  # type: ignore[call-overload]
                    graph_input,
                    config=run_config,
                    stream_mode=["updates", "custom"],
                    subgraphs=True,
                )
                try:
                    async for _, mode, event_data in stream:
                        # Client disconnected — abort so later tool/LLM steps
                        # don't burn tokens for nobody.
                        if await _is_disconnected(request):
                            logger.info(
                                "SSE client disconnected mid-run — aborting agent stream"
                            )
                            # The checkpoint keeps the user's message; mark the
                            # thread (same marker the timeout path writes) so a
                            # retry reads as a continuation, not a second
                            # identical user turn.
                            await _mark_interrupted_thread(agent, req.thread_id)
                            return

                        if mode == "custom":
                            # Token delta emitted by call_llm_node /
                            # generate_final_node while the LLM generates.
                            if isinstance(event_data, dict) and event_data.get("type") == "token":
                                yield f"data: {json.dumps(event_data, ensure_ascii=False)}\n\n"
                            elif isinstance(event_data, dict) and event_data.get("type") == "error":
                                # A node failed after streaming partial tokens — it
                                # deliberately did NOT append the error to the answer,
                                # so surface it as its own error event (the client
                                # renders it separately instead of gluing it on).
                                yield f"data: {json.dumps({'type': 'error', 'message': event_data.get('message', '生成出错，请稍后重试。')}, ensure_ascii=False)}\n\n"
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
                finally:
                    # Close the graph stream on every exit — normal completion,
                    # interrupt, client disconnect, or an internal error.  An
                    # abandoned ``astream`` leaves LangGraph's streaming task
                    # dangling ("Task was destroyed") and can leave a half-run
                    # checkpoint that a retry reads as a duplicate user turn.
                    try:
                        await stream.aclose()
                    except Exception:
                        logger.debug("agent stream close failed", exc_info=True)

                # Fetch the final graph state once: tool traces for the meta
                # event, plus the ReAct loop length (agent_steps histogram) —
                # the same over-call signal the task eval measures on live
                # traffic.  One checkpointer read serves both.
                try:
                    final_state = await agent.aget_state(run_config)  # type: ignore[arg-type]
                    if final_state and final_state.values:
                        messages = final_state.values.get("messages", [])
                        tool_traces, sources = _extract_tool_traces(messages)
                        memory_write = _extract_write_result(messages)
                        if tool_traces or sources or memory_write:
                            yield f"data: {json.dumps({'type': 'meta', 'tool_calls': tool_traces, 'sources': sources, 'memory_write': memory_write}, ensure_ascii=False)}\n\n"

                        from backend.shared.runtime_metrics import observe_agent_steps

                        observe_agent_steps(final_state.values.get("step_count") or 0)
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
        except Exception:
            # Internal exception text must not leak to the SSE client — log
            # the traceback and emit a generic error event.
            logger.exception("Streaming error")
            yield f"data: {json.dumps({'type': 'error', 'message': '流式响应出错，请稍后重试。'}, ensure_ascii=False)}\n\n"
        finally:
            # Release the concurrency slot when the stream ends — completion,
            # timeout, client disconnect (generator close), or error all land
            # here exactly once.
            _release_agent_slot()

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
