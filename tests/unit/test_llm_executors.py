"""Unit tests for tests/eval/llm_executors.py — default executors.

These wrap production functions; each test patches its production seam so
the default path is exercised without a real LLM:

- ``make_tool_selector`` → patch ``backend.agent.nodes.call_llm_node``.
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
    make_e2e_runner,
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

    @pytest.mark.asyncio
    async def test_renders_source_ids_for_citation(self) -> None:
        provider = FakeStreamingProvider(["答案是", " pgvector"])
        generator = make_answer_generator(provider=provider)

        await generator(
            "选型是什么", "用 pgvector 而非 Elasticsearch 做向量检索", ["a1b2c3d4"]
        )
        system = provider.calls[0][0]
        # Source ids are rendered the way a real search display exposes the
        # memory short ID, so the model can cite them inline.
        assert "[memory: a1b2c3d4]" in system["content"]


class TestMakeE2ERunner:
    @pytest.mark.asyncio
    async def test_memory_mode_wraps_context_and_exposes_short_id(self, monkeypatch) -> None:
        async def fake_query_memories(query, top_k=5):
            return [
                {
                    "id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                    "summary": "用 pgvector 而非 Elasticsearch 做向量检索",
                    "rerank_score": 0.9,
                    "decay_factor": 1.0,
                }
            ]

        monkeypatch.setattr(
            "backend.service.retrieval.query_memories", fake_query_memories
        )
        provider = FakeStreamingProvider(["答案是 pgvector"])
        runner = make_e2e_runner(top_k=5, retrieval_mode="memory", provider=provider)

        outcome = await runner("选型是什么")

        assert outcome.answer == "答案是 pgvector"
        assert outcome.n_retrieved == 1
        assert outcome.retrieved_source_ids == ["f47ac10b-58cc-4372-a567-0e02b2c3d479"]
        system = provider.calls[0][0]
        assert system["role"] == "system"
        assert '<memory source="search_memories_tool">' in system["content"]
        # The short id (first 8 chars of the UUID) is exposed the way the
        # production search_memories_tool displays it, so the model has the
        # same citation material the production agent provides.
        assert "memory: f47ac10b" in system["content"]
        assert "用 pgvector 而非 Elasticsearch" in system["content"]
        assert "<doc" not in system["content"]
        assert provider.calls[0][1] == {"role": "user", "content": "选型是什么"}

    @pytest.mark.asyncio
    async def test_chunk_mode_wraps_as_doc(self, monkeypatch) -> None:
        from backend.service.retrieval import RetrievalResult

        async def fake_retrieve_hybrid(query, top_k=5):
            return [
                RetrievalResult(
                    content="文档分块策略：用 AST 解析按函数边界切分，chunk_code 保证函数不被切坏",
                    score=0.85,
                    metadata={"document_id": "ema-e2e-seed"},
                )
            ]

        monkeypatch.setattr(
            "backend.service.retrieval.retrieve_hybrid", fake_retrieve_hybrid
        )
        provider = FakeStreamingProvider(["分块"])
        runner = make_e2e_runner(top_k=5, retrieval_mode="chunk", provider=provider)

        outcome = await runner("文档分块怎么做")

        assert outcome.n_retrieved == 1
        assert outcome.retrieved_source_ids == ["ema-e2e-seed"]
        system = provider.calls[0][0]
        assert '<doc source="retrieve_chunks_tool">' in system["content"]
        assert "document: ema-e2e-seed" in system["content"]
        assert "<memory" not in system["content"]

    @pytest.mark.asyncio
    async def test_empty_retrieval_builds_no_context(self, monkeypatch) -> None:
        async def fake_query_memories(query, top_k=5):
            return []

        monkeypatch.setattr(
            "backend.service.retrieval.query_memories", fake_query_memories
        )
        provider = FakeStreamingProvider(["没有找到"])
        outcome = await make_e2e_runner(provider=provider)("q")

        assert outcome.n_retrieved == 0
        assert outcome.context_text == ""
        assert outcome.retrieved_source_ids == []
        system = provider.calls[0][0]
        assert "<memory" not in system["content"]
        assert "Context:" not in system["content"]
