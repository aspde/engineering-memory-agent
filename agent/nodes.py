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
from langgraph.types import Command, interrupt

from agent.state import AgentState
from backend.service.llm_service import get_llm_provider
from backend.service.memory import resolve_conflict

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are EMA, the Engineering Memory Agent for development teams.

You have access to tools for:
- Searching long-term memories (past decisions, lessons, architecture)
- Searching document chunks (code, documentation)
- Writing new memories from conversations or content
- Extracting structured knowledge from text
- Ingesting git repository history
- Ingesting documents into the knowledge base

When the user asks a question:
1. Search relevant memories and documents first
2. Synthesize information from retrieved context
3. Answer clearly with references to sources when available

When the user asks to ingest or index content, use the appropriate tools.
Always prefer searching over guessing."""


# ── Helper: message conversion ───────────────────────────────────────


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
                schema["function"]["parameters"] = {"type": "object", "properties": {}}
        else:
            schema["function"]["parameters"] = {"type": "object", "properties": {}}
        schemas.append(schema)
    return schemas


def _messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, object]]:
    """Convert LangChain messages to OpenAI-compatible dicts.

    Preserves ``tool_calls`` on AIMessages and ``tool_call_id`` on
    ToolMessages so the LLM can track the ReAct conversation loop.
    """
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
        if isinstance(m, AIMessage) and m.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"], ensure_ascii=False),
                    },
                }
                for tc in m.tool_calls
            ]

        # 4. Preserve tool_call_id on tool messages (OpenAI API requirement)
        if isinstance(m, ToolMessage) and m.tool_call_id:
            entry["tool_call_id"] = m.tool_call_id

        dicts.append(entry)
    return dicts


# ── Nodes ────────────────────────────────────────────────────────────


async def call_llm_node(state: AgentState, *, tools: list) -> dict[str, Any]:
    """Send messages + tool definitions to the LLM, return an AIMessage.

    The *tools* parameter is injected at graph-construction time via
    ``functools.partial`` so the node has access to tool schemas without
    global state.
    """
    provider = get_llm_provider()

    # Prepend system prompt if this is the first call
    messages = list(state["messages"])
    has_system = any(isinstance(m, SystemMessage) for m in messages)
    if not has_system:
        messages.insert(0, SystemMessage(content=SYSTEM_PROMPT))

    tool_schemas = _to_openai_tools(tools)
    dicts = _messages_to_dicts(messages)

    try:
        raw = await provider.chat_raw(messages=dicts, tools=tool_schemas)
    except Exception as exc:
        logger.exception("LLM call failed in call_llm_node")
        return {
            "error": str(exc),
            "messages": [AIMessage(content=f"LLM call failed: {exc}")],
        }

    content = str(raw.get("content", ""))
    tool_calls = raw.get("tool_calls")

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

    return {"messages": [aimessage]}


async def generate_final_node(state: AgentState) -> dict[str, Any]:
    """Assemble context from retrieved results and build the final prompt.

    Reads tool-call results from the conversation history (ToolMessages)
    rather than from discrete state fields, so every tool's output is
    automatically included regardless of which tool was called.

    The LLM call is deferred to the API streaming layer so the response
    can be streamed token-by-token to the client.  The assembled prompt
    is stored in ``final_prompt``.
    """
    # ── Harvest context from ToolMessages in conversation history ──
    context_parts: list[str] = []
    for m in state["messages"]:
        if not isinstance(m, ToolMessage):
            continue
        tool_name = getattr(m, "name", "unknown")
        content = str(m.content) if m.content else ""
        if not content.strip():
            continue
        context_parts.append(f"### {tool_name}\n{content}")

    context_str = "\n\n".join(context_parts) if context_parts else ""

    # Build the final prompt
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "You are EMA, the Engineering Memory Agent. "
                "Answer the user's question based on the conversation "
                "and any retrieved context provided below. "
                "Be concise and cite sources when available."
            ),
        },
    ]

    if context_str:
        messages.append({"role": "system", "content": f"Context:\n{context_str}"})

    # Include the conversation history (skip tool & system messages)
    for m in state["messages"]:
        if isinstance(m, (ToolMessage, SystemMessage)):
            continue
        role = "assistant" if isinstance(m, AIMessage) else "user"
        content = m.content if isinstance(m.content, str) else str(m.content)
        if role == "assistant" and content == "":
            continue  # skip tool_call-only AIMessages
        messages.append({"role": role, "content": content})

    return {"final_prompt": messages}


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
        return Command(goto="call_llm")

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

    # Build the approval payload for the first sensitive call
    # (LLM typically emits one tool_call per turn; if multiple, pause on the first)
    call = sensitive[0]
    tool_name = str(call.get("name", call.get("function", {}).get("name", "unknown")))
    tool_args = call.get("args", call.get("function", {}).get("args", {}))
    if isinstance(tool_args, str):
        try:
            tool_args = json.loads(tool_args)
        except json.JSONDecodeError:
            tool_args = {"raw": tool_args}

    payload: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_args": tool_args,
    }

    # Build a human-readable summary
    if tool_name == "write_memory_tool":
        payload["summary"] = str(tool_args.get("content", ""))[:200]
    elif tool_name == "ingest_git_repo_tool":
        payload["summary"] = f"Repo: {tool_args.get('repo_path', '?')}"
    elif tool_name == "ingest_document_tool":
        payload["summary"] = f"Doc: {tool_args.get('document_id', '?')}"

    logger.info("Requesting human approval for %s", tool_name)

    # Pause and wait for human decision
    decision: dict[str, Any] = interrupt(payload)

    if decision.get("approved") is True:
        logger.info("Tool %s approved, executing", tool_name)
        return Command(goto="tools", update={"pending_approval": None})

    # Rejected — inject a ToolMessage for each rejected call so the LLM
    # can explain why the action was skipped.
    reason = decision.get("reason", "Tool call was rejected by the user.")
    rejection_msgs: list[ToolMessage] = []
    for call in sensitive:
        cid = str(call.get("id", ""))
        tname = str(call.get("name", ""))
        rejection_msgs.append(
            ToolMessage(content=f"[REJECTED] {reason}", tool_call_id=cid, name=tname)
        )

    logger.info("Tool %s rejected: %s", tool_name, reason)
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
    note = ToolMessage(
        content=(
            f"Conflict resolved — {action_label}. "
            f"Memory id: {outcome.get('id', '?')}."
        ),
        tool_call_id=str(getattr(m, "tool_call_id", "")),
        name="write_memory_tool",
    )
    logger.info("Conflict resolved: %s → %s", existing_id, resolution)

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
