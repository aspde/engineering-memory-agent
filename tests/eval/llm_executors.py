"""Default executors for the LLM behavior eval — wrap production code.

The eval runners are executor-agnostic (they accept any callable), which is
what makes them unit-testable with fakes.  The *default* executors here are
the faithful path: they drive the same code a real user request drives, so
the eval measures production behavior rather than a simplified harness.

- ``make_tool_selector`` — runs ``agent.nodes.call_llm_node`` with the real
  tool roster (real system prompt, real schema serialization, real message
  conversion, real streaming provider call).  Only the *decision* is
  measured — tools are never actually executed.
- ``make_extractor`` — runs ``backend.service.extraction.extract_memory``
  (summary + entities + relations with their production prompts).
- ``make_answer_generator`` — builds the final-answer prompt exactly like
  ``generate_final_node`` does (the ``agent.system`` template with the
  context folded through its ``{context}`` placeholder) and streams the
  answer.  It drives the LLM directly rather than the node because the node
  also runs best-effort auto-memory (a real side effect) and swallows
  provider exceptions into an apology string — the eval needs clean
  exception propagation so failures land in the error rows.

``make_answer_generator`` accepts ``provider=None`` for test injection; the
other two are tested by patching their production functions directly.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import HumanMessage

from agent.nodes import call_llm_node
from agent.tools import ALL_TOOLS

# query → list of {"name": str, "args": dict} tool-call decisions
ToolSelector = Callable[[str], Awaitable[list[dict[str, Any]]]]
# content → {"summary": str, "entities": list, "relations": list}
Extractor = Callable[[str], Awaitable[dict[str, Any]]]
# (query, context, source_ids) → answer text
AnswerGenerator = Callable[[str, str, list[str] | None], Awaitable[str]]


def make_tool_selector(tools: list | None = None) -> ToolSelector:
    """Executor that measures the agent's tool-selection decision.

    Drives ``call_llm_node`` with the given tool roster (default: the full
    ``ALL_TOOLS``) and returns the tool calls the model chose.
    """
    roster = list(tools) if tools is not None else ALL_TOOLS

    async def _select(query: str) -> list[dict[str, Any]]:
        state: dict[str, Any] = {"messages": [HumanMessage(content=query)]}
        out = await call_llm_node(state, tools=roster)
        if out.get("error"):
            raise RuntimeError(str(out["error"]))
        ai = out["messages"][0]
        calls = getattr(ai, "tool_calls", None) or []
        return [
            {"name": str(tc.get("name", "")), "args": tc.get("args", {})}
            for tc in calls
        ]

    return _select


def make_extractor() -> Extractor:
    """Executor that runs the real memory-extraction pipeline."""

    async def _extract(content: str) -> dict[str, Any]:
        from backend.service.extraction import extract_memory

        return await extract_memory(content)

    return _extract


def make_answer_generator(provider: Any | None = None) -> AnswerGenerator:
    """Executor that generates a final answer from a golden context.

    Prompt assembly mirrors ``agent.nodes.generate_final_node``: the
    ``agent.system`` template receives the context through its ``{context}``
    placeholder, wrapped in the same ``<memory source=...>`` framing a real
    retrieval ToolMessage would produce, and the answer is streamed via the
    provider's ``chat_stream``.

    ``source_ids`` are rendered in the context block the way a real
    ``search_memories_tool`` display exposes the memory short ID
    (``memory: <id>``), so the model has the same citation material the
    production path provides.
    """
    from backend.service.llm_service import get_llm_provider
    from backend.service.prompts import get_prompt

    async def _generate(query: str, context: str, source_ids: list[str] | None = None) -> str:
        llm = provider if provider is not None else get_llm_provider()
        version, system_text = get_prompt("agent.system")
        if context:
            ids = [str(s) for s in (source_ids or []) if str(s).strip()]
            label = ", ".join(f"memory: {sid}" for sid in ids) if ids else "memory"
            block = (
                f"\n\nContext:\n<memory source=\"search_memories_tool\">\n"
                f"[{label}]\n{context}\n</memory>"
            )
        else:
            block = ""
        system_content = system_text.format(context=block)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": query},
        ]
        parts: list[str] = []
        async for token in llm.chat_stream(messages, scenario="eval_answer"):
            parts.append(str(token))
        return "".join(parts)

    return _generate
