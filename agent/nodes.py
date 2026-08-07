"""Agent graph nodes — call_llm_node, check_approval_node, and generate_final_node.

Plain functions with no class wrappers, matching the project's
service-layer conventions.  Each node receives ``AgentState`` and
returns a partial state dict or ``Command`` for dynamic routing.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.types import Command, interrupt

from agent.state import AgentState
from backend.service.llm_service import get_llm_provider
from backend.service.memory import resolve_conflict

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are EMA, the Engineering Memory Agent for development teams.

You have access to tools for:
- Searching long-term memories (from conversations, PingCode work items,
  CI builds, 飞书 discussions, Git history, and manual ingestion)
- Searching document chunks (code, documentation)
- Writing new memories from conversations or content
- Extracting structured knowledge from text
- Ingesting git repository history
- Ingesting documents into the knowledge base

Memories in your knowledge base come from multiple sources:
- Manual: conversations, documents uploaded by the team
- PingCode: bug root causes, fixes, and work item resolutions
- CI/CD: build failures, test regressions, duration anomalies
- 飞书: technical discussions and decisions from chat threads
- Git: commit history and code changes
You search across ALL sources by default — the user does not need to specify.

When the user asks a question:
1. Search relevant memories and documents first
2. Synthesize information from retrieved context
3. Answer clearly and concisely — do not list or enumerate sources, they are shown separately in the UI
4. If a search returned no results, simply ignore it — do not mention empty searches

When the user asks about a specific external item (a PingCode work item like
"#1234", a CI build, a 飞书 discussion):
- Search for memories related to that item first
- If found, answer from the memory
- If NOT found, say "该 issue/事件 尚未被摄入 EMA，我目前没有关于它的记忆。"
  rather than a generic "I don't know"

When the user asks to ingest or index content, use the appropriate tools.
Always prefer searching over guessing.

When the user tells you to remember something, or shares facts/decisions/knowledge:
- Call write_memory_tool IMMEDIATELY with the user's exact words as content.
- Do NOT pre-check for conflicts yourself. The tool has built-in conflict detection
  and will pause for human review if a contradiction is found.
- Do NOT ask the user whether to overwrite or merge — that is handled by the tool.

Always respond in Chinese (简体中文). All your answers, explanations,
and tool interactions should use Chinese unless the user explicitly
requests another language."""


# ── Helper: message conversion ───────────────────────────────────────


def _stream_writer():
    """Return the LangGraph custom-stream writer, or a no-op outside a run.

    ``get_stream_writer()`` raises outside a graph execution context (e.g.
    when a node is invoked directly in a unit test).  Inside a run it
    forwards token deltas to ``stream_mode="custom"`` consumers; outside,
    streaming is simply disabled.
    """
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _payload: None


def _to_openai_tools(tools: list) -> list[dict[str, Any]]:
    """Convert LangChain tool objects to OpenAI function-calling schemas."""
    schemas: list[dict[str, Any]] = []
    for t in tools:
        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
            },
        }
        if t.args_schema and hasattr(t.args_schema, "model_json_schema"):
            try:
                json_schema = t.args_schema.model_json_schema()
                schema["function"]["parameters"] = json_schema
            except Exception:
                logger.warning(
                    "Failed to serialize tool schema for %s, falling back to empty parameters",
                    t.name,
                )
                schema["function"]["parameters"] = {"type": "object", "properties": {}}
        else:
            schema["function"]["parameters"] = {"type": "object", "properties": {}}
        schemas.append(schema)
    return schemas


def _messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, object]]:
    """Convert LangChain messages to OpenAI-compatible dicts.

    Preserves ``tool_calls`` on AIMessages and ``tool_call_id`` on
    ToolMessages so the LLM can track the ReAct conversation loop.

    Orphaned tool_calls (no ToolMessage response) are stripped to
    prevent OpenAI 400 errors when resuming an interrupted conversation.
    """
    # Collect all tool_call_ids that have a ToolMessage response
    responded_ids: set[str] = set()
    for m in messages:
        if isinstance(m, ToolMessage) and m.tool_call_id:
            responded_ids.add(m.tool_call_id)

    dicts: list[dict[str, object]] = []
    for m in messages:
        # 1. Determine role
        if isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, ToolMessage):
            role = "tool"
        elif isinstance(m, HumanMessage):
            role = "user"
        else:
            logger.debug("Skipping unknown message type: %s", type(m).__name__)
            continue

        # 2. Extract content
        content: str = ""
        if isinstance(m.content, str):
            content = m.content
        elif isinstance(m.content, list):
            parts: list[str] = []
            for block in m.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif hasattr(block, "text"):
                    parts.append(str(block.text))  # type: ignore[union-attr]
            content = " ".join(parts)

        entry: dict[str, object] = {"role": role, "content": content or ""}

        # 3. Preserve tool_calls on assistant messages (OpenAI API requirement)
        #    but only those that have a corresponding ToolMessage response —
        #    otherwise OpenAI rejects with "insufficient tool messages" 400.
        if isinstance(m, AIMessage) and m.tool_calls:
            answered = [
                tc for tc in m.tool_calls
                if tc.get("id") in responded_ids
            ]
            if answered:
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"], ensure_ascii=False),
                        },
                    }
                    for tc in answered
                ]
            elif not content.strip():
                # Orphaned tool_calls with no content — supply a placeholder
                # so the API sees a valid assistant message.
                entry["content"] = "(tool call was interrupted)"

        # 4. Preserve tool_call_id on tool messages (OpenAI API requirement)
        if isinstance(m, ToolMessage) and m.tool_call_id:
            entry["tool_call_id"] = m.tool_call_id

        dicts.append(entry)
    return dicts


def _has_tool_results_this_turn(messages: list[BaseMessage]) -> bool:
    """True if a ToolMessage appeared since the most recent HumanMessage.

    A tool result in the current turn means the final-answer synthesis must
    account for it (the synthesis prompt folds tool output into the system
    context); a plain-chat turn has none, so the last ``call_llm`` output is
    already a complete answer.  Only the *current* turn counts — tool
    results from earlier turns in the same thread don't force synthesis.
    """
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return False
        if isinstance(m, ToolMessage):
            return True
    return False


# ── Nodes ────────────────────────────────────────────────────────────


async def call_llm_node(state: AgentState, *, tools: list) -> dict[str, Any]:
    """Send messages + tool definitions to the LLM, return an AIMessage.

    The *tools* parameter is injected at graph-construction time via
    ``functools.partial`` so the node has access to tool schemas without
    global state.

    Streaming: the LLM call is streamed, and text deltas are forwarded to
    the SSE client live through the LangGraph custom stream (``stream_mode=
    "custom"``).  Tool calls are accumulated and emitted at the end of the
    turn.  Under a non-streaming invocation (``ainvoke``, resume) the stream
    writer is a no-op and behaviour is unchanged.
    """
    provider = get_llm_provider()

    # Prepend system prompt if this is the first call
    messages = list(state["messages"])
    has_system = any(isinstance(m, SystemMessage) for m in messages)
    if not has_system:
        messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

    tool_schemas = _to_openai_tools(tools)
    dicts = _messages_to_dicts(messages)

    writer = _stream_writer()
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] | None = None
    try:
        async for event in provider.chat_raw_stream(
            messages=dicts, tools=tool_schemas, scenario="agent_chat"
        ):
            if event.get("type") == "content":
                text = str(event.get("text", ""))
                content_parts.append(text)
                writer({"type": "token", "content": text})
            elif event.get("type") == "tool_calls":
                tool_calls = event.get("tool_calls")
    except Exception as exc:
        logger.exception("LLM call failed in call_llm_node")
        return {
            "error": str(exc),
            "messages": [AIMessage(content=f"LLM call failed: {exc}")],
        }

    content = "".join(content_parts)

    if tool_calls:
        # Build AIMessage with tool_calls — LangChain will handle parsing
        lc_tool_calls: list[dict[str, Any]] = []
        for tc in tool_calls:
            lc_tool_calls.append({
                "id": str(tc["id"]),
                "name": str(tc["name"]),
                "args": tc["args"] if isinstance(tc["args"], dict) else {},
                "type": "tool_call",
            })
        aimessage = AIMessage(content=content or "", tool_calls=lc_tool_calls)  # type: ignore[arg-type]
    else:
        aimessage = AIMessage(content=content)

    return {"messages": [aimessage], "step_count": state.get("step_count", 0) + 1}


async def generate_final_node(state: AgentState) -> dict[str, Any]:
    """Produce the final answer — reuse ``call_llm``'s output when it is
    already complete, otherwise synthesize one with an LLM call.

    Reads tool-call results from the conversation history (ToolMessages)
    rather than from discrete state fields, so every tool's output is
    automatically included regardless of which tool was called.

    Plain-chat shortcut: when the last message is an AIMessage with text,
    no ``tool_calls``, and no ToolMessage appeared since the latest
    HumanMessage, that output is already a complete answer and is returned
    directly — no second LLM call.  Only tool turns hit the LLM here
    (once), so the response is persisted in the checkpointer state and the
    API streaming layer reads it token-by-token.
    """
    # ── Plain-chat shortcut ─────────────────────────────────────────
    # When call_llm_node's output is already a complete answer — the last
    # message is an AIMessage with text and no tool_calls, and no tool
    # results appeared this turn — reuse it instead of synthesizing again.
    # Without this every plain chat turn paid two LLM calls (chat_raw for
    # tool detection + chat for the final answer), discarding the first.
    messages = state["messages"]
    last = messages[-1] if messages else None
    if (
        isinstance(last, AIMessage)
        and not getattr(last, "tool_calls", None)
        and last.content
        and not _has_tool_results_this_turn(messages)
    ):
        logger.info(
            "Reusing call_llm output as final answer (no tool results this turn)"
        )
        return {"final_prompt": None, "final_response": str(last.content)}

    # ── Harvest context from ToolMessages in conversation history ──
    context_parts: list[str] = []
    for m in state["messages"]:
        if not isinstance(m, ToolMessage):
            continue
        tool_name = getattr(m, "name", "unknown")
        # Chunk retrieval is noisy; the LLM's answer should rely on
        # memory search results (which have stable IDs and summaries).
        if tool_name == "retrieve_chunks_tool":
            continue
        raw = str(m.content) if m.content else ""
        if not raw.strip():
            continue
        # If the tool returned a JSON envelope with a "display" field,
        # use only the display text for LLM context (hiding structured metadata).
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "display" in parsed:
                content = str(parsed["display"])
            else:
                content = raw
        except (json.JSONDecodeError, TypeError):
            content = raw
        context_parts.append(f"### {tool_name}\n{content}")

    context_str = "\n\n".join(context_parts) if context_parts else ""

    # Build the final prompt — single system message (Anthropic only
    # accepts one top-level system param, so context is folded in).
    system_content = (
        "You are EMA, the Engineering Memory Agent. "
        "Answer the user's question based on the conversation "
        "and the retrieved context below. "
        "Be concise.  Do NOT list or enumerate the sources in your response — "
        "they are already displayed separately in the UI."
    )
    if context_str:
        system_content += f"\n\nContext:\n{context_str}"

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content},
    ]

    # Include the conversation history (skip tool & system messages)
    for m in state["messages"]:
        if isinstance(m, (ToolMessage, SystemMessage)):
            continue
        role = "assistant" if isinstance(m, AIMessage) else "user"
        content = m.content if isinstance(m.content, str) else str(m.content)
        if role == "assistant" and content == "":
            continue  # skip tool_call-only AIMessages
        messages.append({"role": role, "content": content})

    # ── Call LLM here (once) so the response is persisted ──
    # Streamed: text deltas are forwarded to the SSE client live via the
    # LangGraph custom stream.  The full text is aggregated so the persisted
    # final_response matches what the client saw.
    provider = get_llm_provider()
    writer = _stream_writer()
    response_parts: list[str] = []
    try:
        async for token in provider.chat_stream(messages, scenario="agent_final"):
            response_parts.append(token)
            writer({"type": "token", "content": token})
        response = "".join(response_parts)
    except Exception as exc:
        logger.exception("Final answer LLM call failed")
        response = f"抱歉，生成回复时出现错误: {exc}"

    aimessage = AIMessage(content=response)

    return {
        "final_prompt": messages,
        "final_response": response,
        "messages": [aimessage],
    }


# ── Tools requiring human approval before execution ────────────────────
# Search/retrieval tools are safe — they only read.  Write tools must
# be approved because they modify the memory store.

APPROVAL_REQUIRED_TOOLS: frozenset[str] = frozenset({
    "write_memory_tool",
    "ingest_git_repo_tool",
    "ingest_document_tool",
})


async def check_approval_node(
    state: AgentState,
) -> Command[Literal["tools", "call_llm"]]:
    """Gate sensitive tool calls before they reach ``ToolNode``.

    Inspects the last AIMessage's ``tool_calls`` and classifies each as
    *safe* (search / retrieval) or *sensitive* (write / ingest).

    - All-safe → passes through to ``tools`` directly (no interrupt).
    - Any sensitive → ``interrupt()`` pauses the graph, surfacing the
      tool name and arguments for human review.  On resume the
      ``Command(resume=...)`` value decides:

        * ``{"approved": true}``  → route to ``tools`` (execute).
        * ``{"approved": false}`` → inject a rejection ``ToolMessage``
          and route back to ``call_llm`` so the LLM can explain why it
          skipped the action.
    """
    # Locate the AIMessage with tool_calls that tools_condition just matched
    messages = state["messages"]
    last_ai: AIMessage | None = None
    for m in reversed(messages):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            last_ai = m
            break

    if last_ai is None:
        logger.warning(
            "tools_condition matched but no AIMessage with tool_calls found — "
            "injecting empty assistant message to break potential loop"
        )
        return Command(
            goto="call_llm",
            update={"messages": [AIMessage(content="(internal: no tool calls to process)")]},
        )

    tool_calls = last_ai.tool_calls

    # Separate safe from sensitive
    safe: list[dict] = []
    sensitive: list[dict] = []
    for tc in tool_calls:
        name = str(tc.get("name", tc.get("function", {}).get("name", "")))
        if name in APPROVAL_REQUIRED_TOOLS:
            sensitive.append(dict(tc))
        else:
            safe.append(dict(tc))

    # If everything is safe, go straight to ToolNode
    if not sensitive:
        return Command(goto="tools")

    # Build approval payloads for all sensitive calls
    calls_payload: list[dict[str, Any]] = []
    for call in sensitive:
        tname = str(call.get("name", call.get("function", {}).get("name", "unknown")))
        targs = call.get("args", call.get("function", {}).get("args", {}))
        if isinstance(targs, str):
            try:
                targs = json.loads(targs)
            except json.JSONDecodeError:
                targs = {"raw": targs}

        entry: dict[str, Any] = {
            "tool_name": tname,
            "tool_args": targs,
        }
        if tname == "write_memory_tool":
            entry["summary"] = str(targs.get("content", ""))[:200]
        elif tname == "ingest_git_repo_tool":
            entry["summary"] = f"Repo: {targs.get('repo_path', '?')}"
        elif tname == "ingest_document_tool":
            entry["summary"] = f"Doc: {targs.get('document_id', '?')}"

        calls_payload.append(entry)

    if len(calls_payload) == 1:
        payload: dict[str, Any] = dict(calls_payload[0])
    else:
        payload = {"type": "batch", "calls": calls_payload}

    logger.info("Requesting human approval for %d tool(s)", len(calls_payload))

    # Pause and wait for human decision
    decision: dict[str, Any] = interrupt(payload)

    if len(calls_payload) == 1:
        approved = decision.get("approved") is True
    else:
        # Batch mode: check if ALL calls were approved
        batch_decisions = decision.get("calls", [])
        approved = bool(batch_decisions) and all(
            d.get("approved") is True for d in batch_decisions
        )

    if approved:
        logger.info("All %d tool(s) approved, executing", len(calls_payload))
        return Command(goto="tools", update={"pending_approval": None})

    # Some or all rejected — inject a ToolMessage for every tool_call in this turn
    reason = decision.get("reason", "Tool call was rejected by the user.")
    rejection_msgs: list[ToolMessage] = []
    for call in sensitive + safe:
        cid = str(call.get("id", call.get("function", {}).get("id", "")))
        tname = str(call.get("name", call.get("function", {}).get("name", "")))
        if call in sensitive:
            # Check if this specific call was rejected in batch mode
            call_rejected = True
            if len(calls_payload) > 1 and "calls" in decision:
                batch_decisions = decision["calls"]
                call_name = str(call.get("name", call.get("function", {}).get("name", "")))
                for bd in batch_decisions:
                    if bd.get("tool_name") == call_name:
                        call_rejected = bd.get("approved") is not True
                        break
            content = f"[REJECTED] {reason}" if call_rejected else "[APPROVED]"
        else:
            content = f"[CANCELLED] A related write operation was rejected by the user."
        rejection_msgs.append(
            ToolMessage(content=content, tool_call_id=cid, name=tname)
        )

    # Log with names from calls_payload (always available, regardless of path)
    rejected_names = ", ".join(c["tool_name"] for c in calls_payload)
    logger.info("Tool(s) rejected [%s]: %s", rejected_names, reason)
    return Command(
        goto="call_llm",
        update={
            "messages": rejection_msgs,
            "pending_approval": None,
        },
    )


async def check_conflict_node(
    state: AgentState,
) -> Command[Literal["call_llm"]]:
    """Inspect ``write_memory_tool`` results — if conflict, pause for human.

    This node runs **after** ``tools`` and before ``call_llm``.  It scans
    the last ToolMessage from ``write_memory_tool`` for an ``action:
    "conflict"`` result.  When one is found it calls ``interrupt()`` with
    the conflict details so the human can choose:

    - ``keep_existing`` — discard the new memory
    - ``overwrite``     — replace existing with new
    - ``merge``         — LLM-merges both into existing
    - ``keep_both``     — insert new alongside existing

    On resume the human's ``resolution`` is applied via
    ``resolve_conflict()`` and a note is injected as a ToolMessage so the
    LLM is aware of what happened.
    """
    messages = state["messages"]

    # Find the last write_memory_tool result
    for m in reversed(messages):
        if (
            isinstance(m, ToolMessage)
            and getattr(m, "name", "") == "write_memory_tool"
        ):
            break
    else:
        return Command(goto="call_llm")

    content = str(m.content)  # type: ignore[possibly-used-before-assignment]
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return Command(goto="call_llm")

    if result.get("action") != "conflict":
        return Command(goto="call_llm")

    # ── Conflict detected — pause for human ──
    conflicting_summary = str(result.get("existing_summary", "")[:300])
    new_summary = str(result.get("summary", "")[:300])

    payload: dict[str, Any] = {
        "type": "conflict",
        "new_summary": new_summary,
        "existing_id": str(result.get("existing_id", "")),
        "existing_summary": conflicting_summary,
        "options": ["keep_existing", "overwrite", "merge", "keep_both"],
        "deferred": result.get("_deferred"),
    }

    logger.info(
        "Conflict detected — pausing for human. New='%s' vs Existing='%s'",
        new_summary[:80],
        conflicting_summary[:80],
    )

    decision: dict[str, Any] = interrupt(payload)

    resolution = decision.get("resolution", "keep_existing")
    deferred = result.get("_deferred") or {}
    existing_id = str(result.get("existing_id", ""))

    try:
        outcome = await resolve_conflict(resolution, existing_id, deferred)
    except Exception as exc:
        logger.exception("Conflict resolution failed")
        outcome = {"id": existing_id, "action": "conflict_resolved", "resolution": "keep_existing"}

    action_label = _RESOLUTION_LABELS.get(resolution, resolution)
    logger.info("Conflict resolved: %s → %s (%s)", existing_id, resolution, outcome.get("id", "?"))

    # Replace the conflict ToolMessage with a resolution note (same id so
    # add_messages deduplicates it) — avoids two ToolMessages with the same
    # tool_call_id, which OpenAI-compatible APIs reject.
    note = ToolMessage(
        content=(
            f"Conflict resolved — {action_label}. "
            f"Memory id: {outcome.get('id', '?')}."
        ),
        tool_call_id=str(getattr(m, "tool_call_id", "")),
        name="write_memory_tool",
        id=getattr(m, "id", None),  # same id → replaces conflict msg
    )
    return Command(
        goto="call_llm",
        update={"messages": [note], "pending_approval": None},
    )


_RESOLUTION_LABELS: dict[str, str] = {
    "keep_existing": "kept the existing memory",
    "overwrite": "overwrote the existing memory with the new one",
    "merge": "merged both memories together",
    "keep_both": "kept both memories",
}
