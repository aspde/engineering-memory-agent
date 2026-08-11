"""Tests for automatic knowledge capture (B3) — enabled by default, best-effort.

Auto memory is enabled by default; when enabled, substantive user turns are
extracted and written unless the agent already wrote this turn.  Failures are
logged and swallowed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from backend.agent.state import AgentState
from backend.shared import config as config_mod
from tests.support.process_state import wait_auto_memory_tasks


@pytest.fixture(autouse=True)
def _llm_gate_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run most auto-memory tests through the keyword-heuristic gate.

    AUTO_MEMORY_LLM_GATE now defaults on; these tests exercise the zero-cost
    keyword mode (gate off) so they stay hermetic.  ``TestAutoMemoryLlmGate``
    re-enables the gate and mocks the judge instead.
    """
    monkeypatch.setattr(config_mod.config, "auto_memory_llm_gate", False)


def _make_state(messages: list) -> AgentState:
    return AgentState(
        messages=messages,
        final_response=None,
        final_prompt=None,
        error=None,
        pending_approval=None,
    )


def _set_auto_memory(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.setattr(config_mod.config, "auto_memory_enabled", enabled)


def _mock_services(
    monkeypatch: pytest.MonkeyPatch,
    *,
    summary: str = "A substantive summary of the user's technical decision.",
    entities: list | None = None,
    write_result: dict | None = None,
    extract_raises: Exception | None = None,
    write_raises: Exception | None = None,
) -> tuple[AsyncMock, AsyncMock]:
    """Patch backend.agent.nodes.extract_memory / write_memory; return their mocks."""
    import backend.agent.nodes as mod

    mock_extract = AsyncMock(
        return_value={"summary": summary, "entities": entities or [], "relations": []}
    )
    if extract_raises:
        mock_extract.side_effect = extract_raises

    mock_write = AsyncMock(
        return_value=write_result or {"id": "mem-1", "action": "inserted"}
    )
    if write_raises:
        mock_write.side_effect = write_raises

    monkeypatch.setattr(mod, "extract_memory", mock_extract)
    monkeypatch.setattr(mod, "write_memory", mock_write)
    return mock_extract, mock_write


class TestAutoMemoryGate:
    """The config gate — enabled by default; disabled means no-op."""

    @pytest.mark.asyncio
    async def test_enabled_by_default(self, monkeypatch) -> None:
        import backend.agent.nodes as mod

        mock_extract, mock_write = _mock_services(monkeypatch)
        assert config_mod.config.auto_memory_enabled is True
        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="记住：用 PostgreSQL 存向量")])
        )
        mock_extract.assert_awaited_once()
        mock_write.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disabled_does_not_extract_or_write(self, monkeypatch) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, False)
        mock_extract, mock_write = _mock_services(monkeypatch)

        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="记住：用 PostgreSQL 存向量")])
        )
        mock_extract.assert_not_awaited()
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_memory_pipeline_disabled_blocks_auto_memory(self, monkeypatch) -> None:
        """MEMORY_ENABLED=false turns the agent into pure chat — auto capture
        must not keep extracting and writing memories behind the scenes."""
        import backend.agent.nodes as mod

        # auto_memory_enabled is on (the default), but the whole memory
        # pipeline is off.
        assert config_mod.config.auto_memory_enabled is True
        monkeypatch.setattr(config_mod.config, "memory_enabled", False)
        mock_extract, mock_write = _mock_services(monkeypatch)

        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="记住：用 PostgreSQL 存向量")])
        )
        mock_extract.assert_not_awaited()
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_enabled_writes_substantive_message(self, monkeypatch) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(monkeypatch)

        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="记住：用 PostgreSQL 存向量")])
        )
        mock_extract.assert_awaited_once()
        mock_write.assert_awaited_once()
        content = mock_write.await_args.args[0]
        assert "PostgreSQL" in content
        assert mock_write.await_args.kwargs["source_type"] == "conversation"


class TestAutoMemorySubstance:
    """Substantive-knowledge heuristic gates the write."""

    @pytest.mark.asyncio
    async def test_short_or_empty_summary_is_not_written(self, monkeypatch) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(monkeypatch, summary="ok", entities=[])

        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="部署了新的 Kafka 限流中间件")])
        )
        mock_extract.assert_awaited_once()
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_entities_alone_can_trigger_write(self, monkeypatch) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(
            monkeypatch, summary="x", entities=[{"name": "pgvector", "type": "technology"}]
        )

        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="pgvector 是向量检索插件")])
        )
        mock_write.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_degraded_truncated_summary_is_not_written(self, monkeypatch) -> None:
        """LLM-outage extraction (verbatim-truncated summary + no entities)
        must not be written — it is a failure artifact, not knowledge.

        Regression: with the LLM down, extract_summary falls back to the
        first 200 chars of the source verbatim and entity extraction
        degrades to [] — a combination that cleared the old length-only
        substance gate and polluted the memory store with raw truncations.
        """
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        user_content = "记住：用 PostgreSQL 存向量，且连接池超时配置是 30 秒"
        mock_extract, mock_write = _mock_services(
            monkeypatch, summary=user_content[:200], entities=[]
        )

        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content=user_content)])
        )
        mock_extract.assert_awaited_once()
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_real_llm_outage_truncated_extraction_not_written(self, monkeypatch) -> None:
        """Real extraction pipeline under an LLM outage: extract_summary falls
        back to a verbatim truncation and entity extraction degrades to [] —
        _has_substance rejects the artifact and no memory is written.

        Unlike the other substance tests (which mock extract_memory), this
        runs the real extraction chain, so it guards the *combination* of the
        degradation fallback and the substance gate.
        """
        import backend.agent.nodes as mod
        from backend.service.extraction import extract_memory

        _set_auto_memory(monkeypatch, True)
        # A declarative statement that passes the keyword-heuristic gate
        # (this file's default) and is long enough to be truncated.
        user_content = (
            "本次巡检发现记忆检索流水线包含四个阶段：先对查询文本执行 BGE-M3 编码得到 1024 维向量，"
            "再用 pgvector 的 HNSW 索引做余弦相似度召回，得到候选后用衰变公式 R=e^(-t/S) 加权排序，"
            "最后把命中摘要拼装为上下文供大模型综合回答，整条链路在单个请求内完成，平均延迟约八百毫秒，"
            "其中向量编码约占一半耗时，属于当前检索性能的主要瓶颈，后续可考虑更轻量的编码模型。"
            "此外记忆去重依赖内容哈希和余弦阈值分档，冲突检测失败时降级为补充写入，"
            "这些决策都需要在评估数据集中反复校准才能保证召回率稳定。"
        )
        assert len(user_content) > 200

        class _DownLLM:
            async def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003, ANN002
                raise RuntimeError("LLM down")
            async def chat_json(self, messages, **kwargs):  # noqa: ANN001, ANN003, ANN002
                raise RuntimeError("LLM down")

        down = _DownLLM()
        monkeypatch.setattr(
            "backend.service.extraction.get_llm_provider", lambda: down
        )
        monkeypatch.setattr(
            "backend.service.structured.get_llm_provider", lambda: down
        )
        # One attempt, no backoff — keep the degraded extraction fast.
        monkeypatch.setattr(config_mod.config.llm, "structured_max_attempts", 1)
        monkeypatch.setattr(config_mod.config.llm, "structured_backoff", 0)

        mock_write = AsyncMock()
        monkeypatch.setattr(mod, "write_memory", mock_write)

        # The real extraction degrades: verbatim truncation + no entities.
        extracted = await extract_memory(user_content)
        assert extracted["summary"] == user_content[:200]
        assert extracted["entities"] == []
        assert extracted["relations"] == []
        assert mod._has_substance(extracted, user_content) is False

        # Full auto-memory path: no memory is written for the artifact.
        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content=user_content)])
        )
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_human_message_is_noop(self, monkeypatch) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(monkeypatch)

        await mod._maybe_auto_memory(_make_state([]))
        mock_extract.assert_not_awaited()
        mock_write.assert_not_awaited()


class TestAutoMemorySuppression:
    """When the agent already wrote a memory this turn, auto memory stands down."""

    @pytest.mark.asyncio
    async def test_write_tool_call_this_turn_suppresses_auto_write(
        self, monkeypatch
    ) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(monkeypatch)

        state = _make_state(
            [
                HumanMessage(content="记住：用 PostgreSQL 存向量"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "c1", "name": "write_memory_tool", "args": {}, "type": "tool_call"}
                    ],
                ),
            ]
        )
        await mod._maybe_auto_memory(state)
        mock_extract.assert_not_awaited()
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_write_tool_result_this_turn_suppresses_auto_write(
        self, monkeypatch
    ) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(monkeypatch)

        state = _make_state(
            [
                HumanMessage(content="记住：用 PostgreSQL 存向量"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "c1", "name": "write_memory_tool", "args": {}, "type": "tool_call"}
                    ],
                ),
                ToolMessage(
                    content='{"id": "m1", "action": "inserted"}',
                    tool_call_id="c1",
                    name="write_memory_tool",
                ),
            ]
        )
        await mod._maybe_auto_memory(state)
        mock_extract.assert_not_awaited()
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_previous_turn_write_does_not_suppress(self, monkeypatch) -> None:
        """A write in an earlier turn doesn't suppress this turn's auto memory."""
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(monkeypatch)

        state = _make_state(
            [
                HumanMessage(content="记住：A"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "c1", "name": "write_memory_tool", "args": {}, "type": "tool_call"}
                    ],
                ),
                ToolMessage(
                    content='{"id": "m1", "action": "inserted"}',
                    tool_call_id="c1",
                    name="write_memory_tool",
                ),
                AIMessage(content="记住了"),
                HumanMessage(content="今天上线了新的限流中间件"),
                AIMessage(content="好的"),
            ]
        )
        await mod._maybe_auto_memory(state)
        mock_extract.assert_awaited_once()
        mock_write.assert_awaited_once()


class TestAutoMemoryFailures:
    """Failures are logged, never raised — the chat response is unaffected."""

    @pytest.mark.asyncio
    async def test_extraction_failure_is_swallowed(self, monkeypatch) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(
            monkeypatch, extract_raises=RuntimeError("LLM down")
        )

        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="记住：用 PostgreSQL 存向量")])
        )
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_write_failure_is_swallowed(self, monkeypatch) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        _, mock_write = _mock_services(monkeypatch, write_raises=RuntimeError("db down"))

        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="记住：用 PostgreSQL 存向量")])
        )
        mock_write.assert_awaited_once()


class TestAutoMemoryWiring:
    """generate_final_node invokes auto memory on both exit paths."""

    @pytest.mark.asyncio
    async def test_plain_chat_path_auto_writes_when_enabled(self, monkeypatch) -> None:
        import backend.agent.nodes as mod
        from tests._fake_llm import text_stream

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(monkeypatch)

        mock_provider = AsyncMock()
        mock_provider.chat.return_value = "should not be used"
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        state = _make_state(
            [
                HumanMessage(content="记住：用 PostgreSQL 存向量"),
                AIMessage(content="已记住"),
            ]
        )
        result = await mod.generate_final_node(state)
        assert result["final_response"] == "已记住"
        await wait_auto_memory_tasks()
        mock_write.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tool_path_auto_writes_when_enabled(self, monkeypatch) -> None:
        import backend.agent.nodes as mod
        from tests._fake_llm import text_stream

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(monkeypatch)

        mock_provider = AsyncMock()
        mock_provider.chat_stream = text_stream("Final.")
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        state = _make_state(
            [
                HumanMessage(content="我们验证了 pgvector 索引方案"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "c1", "name": "search_memories_tool", "args": {}, "type": "tool_call"}
                    ],
                ),
                ToolMessage(content="Found 1", tool_call_id="c1", name="search_memories_tool"),
            ]
        )
        result = await mod.generate_final_node(state)
        assert result["final_response"] == "Final."
        await wait_auto_memory_tasks()
        mock_write.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_plain_chat_path_does_not_write_when_disabled(self, monkeypatch) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, False)
        mock_extract, mock_write = _mock_services(monkeypatch)

        mock_provider = AsyncMock()
        mock_provider.chat.return_value = "should not be used"
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        state = _make_state(
            [
                HumanMessage(content="记住：用 PostgreSQL 存向量"),
                AIMessage(content="已记住"),
            ]
        )
        result = await mod.generate_final_node(state)
        assert result["final_response"] == "已记住"
        await wait_auto_memory_tasks()
        mock_write.assert_not_awaited()


class TestAutoMemoryQualityGate:
    """Declarative-knowledge gate in keyword-heuristic mode
    (AUTO_MEMORY_LLM_GATE=false) — questions, requests, and filler skip
    before any LLM call is made."""

    @pytest.mark.asyncio
    async def test_question_turn_is_not_extracted(self, monkeypatch) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(monkeypatch)

        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="PostgreSQL 怎么优化性能？")])
        )
        mock_extract.assert_not_awaited()
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_request_turn_is_not_extracted(self, monkeypatch) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(monkeypatch)

        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="帮我查一下 pgvector 的索引配置")])
        )
        mock_extract.assert_not_awaited()
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_short_fragment_is_not_extracted(self, monkeypatch) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(monkeypatch)

        await mod._maybe_auto_memory(_make_state([HumanMessage(content="pgvector")]))
        mock_extract.assert_not_awaited()
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_chatty_filler_is_not_extracted(self, monkeypatch) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(monkeypatch)

        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="好的，谢谢！")])
        )
        mock_extract.assert_not_awaited()
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_symbol_noise_is_not_extracted(self, monkeypatch) -> None:
        """Emoji/symbol runs pass the raw-length gate but carry zero knowledge."""
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(monkeypatch)

        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="🎉" * 12)])
        )
        mock_extract.assert_not_awaited()
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_long_acknowledgement_is_not_extracted(self, monkeypatch) -> None:
        """A longer polite acknowledgement exceeds the raw length gate but has
        no informative content after stripping chatty words."""
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(monkeypatch)

        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="好的，明白了，收到，谢谢！")])
        )
        mock_extract.assert_not_awaited()
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_declarative_statement_is_extracted(self, monkeypatch) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(monkeypatch)

        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="生产环境禁止用 root 跑 PostgreSQL")])
        )
        mock_extract.assert_awaited_once()
        mock_write.assert_awaited_once()


class TestAutoMemoryLlmGate:
    """AUTO_MEMORY_LLM_GATE=true (the default) adds one LLM call to the
    quality gate — a fast-path-passing turn is judged again before extraction."""

    def test_gate_on_by_default(self, monkeypatch) -> None:
        """The LLM gate is the production default; false opts back to keywords."""
        monkeypatch.delenv("AUTO_MEMORY_LLM_GATE", raising=False)
        from backend.shared.config import AppConfig

        assert AppConfig().auto_memory_llm_gate is True

    @pytest.mark.asyncio
    async def test_gate_worthy_true_still_extracts(self, monkeypatch) -> None:
        import backend.agent.nodes as mod
        from backend.service import structured as structured_mod

        _set_auto_memory(monkeypatch, True)
        monkeypatch.setattr(config_mod.config, "auto_memory_llm_gate", True)
        mock_extract, mock_write = _mock_services(monkeypatch)
        mock_provider = AsyncMock()
        mock_provider.chat_json.return_value = '{"worthy": true}'
        monkeypatch.setattr(structured_mod, "get_llm_provider", lambda: mock_provider)

        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="部署了新的 Kafka 限流中间件")])
        )
        mock_extract.assert_awaited_once()
        mock_write.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gate_worthy_false_skips(self, monkeypatch) -> None:
        import backend.agent.nodes as mod
        from backend.service import structured as structured_mod

        _set_auto_memory(monkeypatch, True)
        monkeypatch.setattr(config_mod.config, "auto_memory_llm_gate", True)
        mock_extract, mock_write = _mock_services(monkeypatch)
        mock_provider = AsyncMock()
        mock_provider.chat_json.return_value = '{"worthy": false}'
        monkeypatch.setattr(structured_mod, "get_llm_provider", lambda: mock_provider)

        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="部署了新的 Kafka 限流中间件")])
        )
        mock_extract.assert_not_awaited()
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gate_failure_allows_through(self, monkeypatch) -> None:
        """A gate outage must not drop a heuristic-passing turn — the later
        substance check still guards the write."""
        import backend.agent.nodes as mod
        from backend.service import structured as structured_mod

        _set_auto_memory(monkeypatch, True)
        monkeypatch.setattr(config_mod.config, "auto_memory_llm_gate", True)
        mock_extract, mock_write = _mock_services(monkeypatch)
        mock_provider = AsyncMock()
        mock_provider.chat_json.side_effect = RuntimeError("LLM down")
        monkeypatch.setattr(structured_mod, "get_llm_provider", lambda: mock_provider)

        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="部署了新的 Kafka 限流中间件")])
        )
        mock_extract.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gate_escapes_braces_in_content(self, monkeypatch) -> None:
        """A message containing braces (code snippets) must not crash the
        gate — .format() interpolates them as literal text, so the gate still
        judges instead of failing open on a KeyError."""
        import backend.agent.nodes as mod
        from backend.service import structured as structured_mod

        _set_auto_memory(monkeypatch, True)
        monkeypatch.setattr(config_mod.config, "auto_memory_llm_gate", True)
        mock_extract, mock_write = _mock_services(monkeypatch)
        mock_provider = AsyncMock()
        mock_provider.chat_json.return_value = '{"worthy": false}'
        monkeypatch.setattr(structured_mod, "get_llm_provider", lambda: mock_provider)

        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="用 {ctx} 变量会导致泄漏")])
        )
        # The gate actually judged (one chat_json call) and returned false —
        # extraction is skipped.  A KeyError from .format() would instead have
        # failed the gate open and reached extraction.
        mock_provider.chat_json.assert_awaited_once()
        sent = mock_provider.chat_json.call_args.args[0][0]["content"]
        assert "{{ctx}}" in sent  # content braces escaped for .format()
        assert '{"worthy": true}' in sent  # template's JSON example stays intact
        mock_extract.assert_not_awaited()
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_complex_declarative_with_question_marker_captured(
        self, monkeypatch
    ) -> None:
        """The fast path does NOT kill a declarative statement that carries a
        question marker — the LLM gate judges it worthy and capture proceeds.

        Under the old keyword heuristic (gate off) this message would have
        been rejected for containing 为什么/？ — a false negative this gate
        fixes."""
        import backend.agent.nodes as mod
        from backend.service import structured as structured_mod

        _set_auto_memory(monkeypatch, True)
        monkeypatch.setattr(config_mod.config, "auto_memory_llm_gate", True)
        mock_extract, mock_write = _mock_services(monkeypatch)
        mock_provider = AsyncMock()
        mock_provider.chat_json.return_value = '{"worthy": true}'
        monkeypatch.setattr(structured_mod, "get_llm_provider", lambda: mock_provider)

        await mod._maybe_auto_memory(
            _make_state(
                [HumanMessage(content="为什么 pgvector 索引会失效？后来查明是没建索引导致全表扫描")]
            )
        )
        mock_extract.assert_awaited_once()
        mock_write.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_long_chatty_filler_rejected_by_gate(self, monkeypatch) -> None:
        """Long filler that passes the fast path is rejected by the LLM gate.

        "今天天气不错…" carries no question marker, so the keyword heuristic
        let it through to extraction — a false positive the gate fixes."""
        import backend.agent.nodes as mod
        from backend.service import structured as structured_mod

        _set_auto_memory(monkeypatch, True)
        monkeypatch.setattr(config_mod.config, "auto_memory_llm_gate", True)
        mock_extract, mock_write = _mock_services(monkeypatch)
        mock_provider = AsyncMock()
        mock_provider.chat_json.return_value = '{"worthy": false}'
        monkeypatch.setattr(structured_mod, "get_llm_provider", lambda: mock_provider)

        await mod._maybe_auto_memory(
            _make_state(
                [HumanMessage(content="今天天气真不错啊，阳光很温暖，我心情很好")]
            )
        )
        mock_extract.assert_not_awaited()
        mock_write.assert_not_awaited()


class TestAutoMemoryThrottle:
    """Frequency control — per-thread interval/cap, repeat skip, global window."""

    @pytest.mark.asyncio
    async def test_repeat_content_in_same_thread_is_skipped(self, monkeypatch) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(monkeypatch)

        config_mod.current_thread_id.set("t-a")
        state = _make_state([HumanMessage(content="部署了新的 Kafka 限流中间件")])
        await mod._maybe_auto_memory(state)
        mock_write.assert_awaited_once()

        mock_extract.reset_mock()
        mock_write.reset_mock()
        await mod._maybe_auto_memory(state)
        mock_extract.assert_not_awaited()
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_min_interval_throttles_rapid_writes(self, monkeypatch) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(monkeypatch)

        config_mod.current_thread_id.set("t-a")
        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="部署了新的 Kafka 限流中间件")])
        )
        mock_write.assert_awaited_once()

        mock_extract.reset_mock()
        mock_write.reset_mock()
        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="回滚了 Kafka 的配置")])
        )
        mock_extract.assert_not_awaited()
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_interval_disabled_when_zero(self, monkeypatch) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        monkeypatch.setattr(config_mod.config, "auto_memory_min_interval", 0)
        mock_extract, mock_write = _mock_services(monkeypatch)

        config_mod.current_thread_id.set("t-a")
        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="部署了新的 Kafka 限流中间件")])
        )
        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="回滚了 Kafka 的配置")])
        )
        assert mock_write.await_count == 2

    @pytest.mark.asyncio
    async def test_per_thread_cap_limits_writes(self, monkeypatch) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        monkeypatch.setattr(config_mod.config, "auto_memory_max_per_thread", 2)
        monkeypatch.setattr(config_mod.config, "auto_memory_min_interval", 0)
        mock_extract, mock_write = _mock_services(monkeypatch)

        config_mod.current_thread_id.set("t-a")
        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="部署了新的 Kafka 限流中间件")])
        )
        assert mock_write.await_count == 1
        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="回滚了 Kafka 的配置")])
        )
        assert mock_write.await_count == 2
        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="升级了 pgvector 索引")])
        )
        assert mock_write.await_count == 2  # capped at the per-thread limit

    @pytest.mark.asyncio
    async def test_global_window_cap_across_threads(self, monkeypatch) -> None:
        import backend.agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        monkeypatch.setattr(config_mod.config, "auto_memory_max_per_window", 2)
        mock_extract, mock_write = _mock_services(monkeypatch)

        for tid, content in (
            ("t-a", "部署了新的限流中间件"),
            ("t-b", "回滚了 Kafka 的配置"),
            ("t-c", "升级了 pgvector 索引"),
        ):
            config_mod.current_thread_id.set(tid)
            await mod._maybe_auto_memory(
                _make_state([HumanMessage(content=content)])
            )
        assert mock_write.await_count == 2  # third crosses the global window cap


class TestAutoMemoryBackgrounding:
    """generate_final_node must NOT block on the auto-memory capture.

    A substantive turn's capture costs 4-7 LLM calls plus embedding + DB
    writes.  Running it synchronously after the answer stream finished held
    the request open for seconds and held the agent-concurrency slot.  It now
    runs as a fire-and-forget task: the node returns before the capture
    finishes, and the capture still completes.
    """

    @pytest.mark.asyncio
    async def test_node_returns_before_slow_capture_completes(self, monkeypatch) -> None:
        import asyncio

        import backend.agent.nodes as mod
        from tests._fake_llm import text_stream

        _set_auto_memory(monkeypatch, True)

        mock_extract = AsyncMock(
            return_value={"summary": "A substantive technical decision.", "entities": [], "relations": []}
        )

        async def _slow_write(content, source_type="conversation", metadata=None):
            await asyncio.sleep(0.3)
            return {"id": "mem-1", "action": "inserted"}

        mock_write = AsyncMock(side_effect=_slow_write)
        monkeypatch.setattr(mod, "extract_memory", mock_extract)
        monkeypatch.setattr(mod, "write_memory", mock_write)

        mock_provider = AsyncMock()
        mock_provider.chat.return_value = "已记住"
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        state = _make_state(
            [
                HumanMessage(content="记住：用 PostgreSQL 存向量"),
                AIMessage(content="已记住"),
            ]
        )

        result = await mod.generate_final_node(state)
        assert result["final_response"] == "已记住"
        # The capture is still in flight — the node returned before it
        # finished, instead of awaiting all of it synchronously.
        assert mod._auto_memory_tasks
        await wait_auto_memory_tasks()
        mock_write.assert_awaited_once()
