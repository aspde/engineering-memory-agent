"""Tests for automatic knowledge capture (B3) — enabled by default, best-effort.

Auto memory is enabled by default; when enabled, substantive user turns are
extracted and written unless the agent already wrote this turn.  Failures are
logged and swallowed.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.state import AgentState
from backend.shared import config as config_mod


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
    """Patch agent.nodes.extract_memory / write_memory; return their mocks."""
    import agent.nodes as mod

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
        import agent.nodes as mod

        mock_extract, mock_write = _mock_services(monkeypatch)
        assert config_mod.config.auto_memory_enabled is True
        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="记住：用 PostgreSQL 存向量")])
        )
        mock_extract.assert_awaited_once()
        mock_write.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disabled_does_not_extract_or_write(self, monkeypatch) -> None:
        import agent.nodes as mod

        _set_auto_memory(monkeypatch, False)
        mock_extract, mock_write = _mock_services(monkeypatch)

        await mod._maybe_auto_memory(
            _make_state([HumanMessage(content="记住：用 PostgreSQL 存向量")])
        )
        mock_extract.assert_not_awaited()
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_enabled_writes_substantive_message(self, monkeypatch) -> None:
        import agent.nodes as mod

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
        import agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(monkeypatch, summary="ok", entities=[])

        await mod._maybe_auto_memory(_make_state([HumanMessage(content="你好")]))
        mock_extract.assert_awaited_once()
        mock_write.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_entities_alone_can_trigger_write(self, monkeypatch) -> None:
        import agent.nodes as mod

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(
            monkeypatch, summary="x", entities=[{"name": "pgvector", "type": "technology"}]
        )

        await mod._maybe_auto_memory(_make_state([HumanMessage(content="pgvector")]))
        mock_write.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_human_message_is_noop(self, monkeypatch) -> None:
        import agent.nodes as mod

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
        import agent.nodes as mod

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
        import agent.nodes as mod

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
        import agent.nodes as mod

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
                HumanMessage(content="今天上线了新的限流器"),
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
        import agent.nodes as mod

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
        import agent.nodes as mod

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
        import agent.nodes as mod
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
        mock_write.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tool_path_auto_writes_when_enabled(self, monkeypatch) -> None:
        import agent.nodes as mod
        from tests._fake_llm import text_stream

        _set_auto_memory(monkeypatch, True)
        mock_extract, mock_write = _mock_services(monkeypatch)

        mock_provider = AsyncMock()
        mock_provider.chat_stream = text_stream("Final.")
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        state = _make_state(
            [
                HumanMessage(content="搜索一下"),
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
        mock_write.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_plain_chat_path_does_not_write_when_disabled(self, monkeypatch) -> None:
        import agent.nodes as mod

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
        mock_write.assert_not_awaited()
