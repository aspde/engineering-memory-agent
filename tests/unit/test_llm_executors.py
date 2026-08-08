"""Unit tests for tests/eval/llm_executors.py — default executors.

These wrap production functions; each test patches its production seam so
the default path is exercised without a real LLM:

- ``make_tool_selector`` → patch ``agent.nodes.call_llm_node``.
- ``make_extractor`` → patch ``backend.service.extraction.extract_memory``.
- ``make_answer_generator`` → inject a fake provider (no patching needed —
  this is why the factory accepts ``provider``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from langchain_core.messages import AIMessage

import tests.eval.llm_executors as exec_mod
from tests.eval.llm_executors import (
    make_answer_generator,
    make_extractor,
    make_tool_selector,
)


class TestMakeToolSelector:
    @pytest.mark.asyncio
    async def test_extracts_tool_calls_from_node_output(self, monkeypatch) -> None:
        async def fake_call_llm_node(state, *, tools):
            ai = AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "c1",
                        "name": "search_memories_tool",
                        "args": {"query": "泄漏"},
                        "type": "tool_call",
                    }
                ],
            )
            return {"messages": [ai]}

        monkeypatch.setattr(exec_mod, "call_llm_node", fake_call_llm_node)
        selector = make_tool_selector(tools=[])
        calls = await selector("之前泄漏怎么解决的")
        assert calls == [{"name": "search_memories_tool", "args": {"query": "泄漏"}}]

    @pytest.mark.asyncio
    async def test_node_error_propagates(self, monkeypatch) -> None:
        async def fake_call_llm_node(state, *, tools):
            return {"error": "boom", "messages": [AIMessage(content="error")]}

        monkeypatch.setattr(exec_mod, "call_llm_node", fake_call_llm_node)
        with pytest.raises(RuntimeError, match="boom"):
            await make_tool_selector()("q")


class TestMakeExtractor:
    @pytest.mark.asyncio
    async def test_delegates_to_extract_memory(self, monkeypatch) -> None:
        canned = {"summary": "s", "entities": [], "relations": []}

        async def fake_extract_memory(content: str):
            return canned

        monkeypatch.setattr(
            "backend.service.extraction.extract_memory", fake_extract_memory
        )
        extractor = make_extractor()
        assert await extractor("some content") is canned


class FakeStreamingProvider:
    """Minimal LLMProvider stub exposing only ``chat_stream``."""

    def __init__(self, tokens: list[str]):
        self._tokens = tokens
        self.calls: list[list[dict[str, str]]] = []

    async def chat_stream(self, messages: list[dict[str, str]], **kw) -> AsyncIterator[str]:
        self.calls.append(messages)
        for t in self._tokens:
            yield t


class TestMakeAnswerGenerator:
    @pytest.mark.asyncio
    async def test_builds_system_context_prompt_and_streams(self) -> None:
        provider = FakeStreamingProvider(["答案是", " pgvector"])
        generator = make_answer_generator(provider=provider)

        answer = await generator(
            "选型是什么", "用 pgvector 而非 Elasticsearch 做向量检索"
        )
        assert answer == "答案是 pgvector"

        system = provider.calls[0][0]
        assert system["role"] == "system"
        assert "EMA" in system["content"]  # production agent.system template
        assert "<memory source=\"search_memories_tool\">" in system["content"]
        assert "pgvector 而非 Elasticsearch" in system["content"]
        assert provider.calls[0][1] == {"role": "user", "content": "选型是什么"}
