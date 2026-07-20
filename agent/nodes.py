"""Agent graph nodes — call_llm_node and generate_final_node.

Plain functions with no class wrappers, matching the project's
service-layer conventions.  Each node receives ``AgentState`` and
returns a partial state dict.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agent.state import AgentState
from backend.service.llm_service import get_llm_provider

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
    """Assemble context from retrieved results and produce the final answer.

    Reads tool-call results from the conversation history (ToolMessages)
    rather than from discrete state fields, so every tool's output is
    automatically included regardless of which tool was called.
    """
    provider = get_llm_provider()

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

    try:
        response = await provider.chat(messages)
    except Exception as exc:
        logger.exception("LLM call failed in generate_final_node")
        return {
            "final_response": f"Sorry, I encountered an error: {exc}",
            "error": str(exc),
        }

    return {
        "final_response": response,
        "messages": [AIMessage(content=response)],
    }
