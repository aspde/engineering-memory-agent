"""Tests for conversation compaction (B4) — opt-in, fails safe.

When enabled, messages older than the context window are folded into one
running-summary SystemMessage; when disabled, the existing truncation
behaviour is unchanged.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from backend.shared import config as config_mod


def _set_compaction(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.setattr(config_mod.config, "conversation_compaction_enabled", enabled)


class TestCompactionDisabled:
    """Default off — truncation behaviour is identical to before."""

    @pytest.mark.asyncio
    async def test_returns_messages_unchanged_without_llm_call(self, monkeypatch) -> None:
        import agent.nodes as mod

        _set_compaction(monkeypatch, False)
        mock_provider = AsyncMock()
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        messages = [HumanMessage(content=f"m{i}") for i in range(30)]
        result = await mod._maybe_compact(messages)
        assert result is messages
        mock_provider.chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_bounded_messages_equals_window_when_disabled(self, monkeypatch) -> None:
        import agent.nodes as mod

        _set_compaction(monkeypatch, False)

        messages = [HumanMessage(content=f"m{i}") for i in range(30)]
        result = await mod._bounded_messages(messages)
        expected = mod._window_messages(messages)
        assert result == expected
        assert result[-1].content == "m29"


class TestCompactionEnabled:
    """When enabled, overflow messages are folded into a summary."""

    @pytest.mark.asyncio
    async def test_early_messages_folded_into_summary(self, monkeypatch) -> None:
        import agent.nodes as mod

        _set_compaction(monkeypatch, True)
        mock_provider = AsyncMock()
        mock_provider.chat.return_value = "Early part: user asked about X, then Y."
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        # 14 filler messages push the first ones out of a 12-message window.
        messages = [
            HumanMessage(content=f"old detail {i}") for i in range(14)
        ] + [HumanMessage(content="current question")]

        result = await mod._maybe_compact(messages)

        # One summary SystemMessage prepended, tail preserved.
        assert isinstance(result[0], SystemMessage)
        assert "Early part" in result[0].content
        assert result[-1].content == "current question"
        mock_provider.chat.assert_awaited_once()
        # The LLM saw the compaction prompt (scenario conversation_compaction).
        kwargs = mock_provider.chat.await_args.kwargs
        assert kwargs["scenario"] == "conversation_compaction"

    @pytest.mark.asyncio
    async def test_within_window_does_not_summarize(self, monkeypatch) -> None:
        import agent.nodes as mod

        _set_compaction(monkeypatch, True)
        mock_provider = AsyncMock()
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        messages = [HumanMessage(content=f"m{i}") for i in range(5)]
        result = await mod._maybe_compact(messages)
        assert result is messages
        mock_provider.chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_summarize_failure_falls_back_to_truncation(self, monkeypatch) -> None:
        import agent.nodes as mod

        _set_compaction(monkeypatch, True)
        mock_provider = AsyncMock()
        mock_provider.chat.side_effect = RuntimeError("LLM down")
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        messages = [HumanMessage(content=f"m{i}") for i in range(30)]
        result = await mod._maybe_compact(messages)
        # Falls back to the original list; windowing still truncates later.
        assert result is messages


class TestCompactionThroughNodes:
    """Compaction integrates with call_llm and generate_final."""

    @pytest.mark.asyncio
    async def test_call_llm_sends_summary_when_enabled(self, monkeypatch) -> None:
        from tests._fake_llm import content_stream, sequential_stream

        import agent.nodes as mod

        _set_compaction(monkeypatch, True)
        mock_provider = AsyncMock()
        mock_provider.chat_raw_stream = sequential_stream(content_stream("ok"))
        # The compaction LLM call (provider.chat) is separate from the stream.
        mock_provider.chat.return_value = "SUMMARY-OF-EARLY-MESSAGES"
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        from agent.tools import ALL_TOOLS

        history = [HumanMessage(content=f"old message {i}") for i in range(30)]
        await mod.call_llm_node(_agent_state(history), tools=ALL_TOOLS)

        sent = mock_provider.chat_raw_stream.call_args.kwargs["messages"]
        contents = [m["content"] for m in sent if m["role"] == "system"]
        assert any("SUMMARY-OF-EARLY-MESSAGES" in c for c in contents)

    @pytest.mark.asyncio
    async def test_generate_final_folds_summary_into_context(self, monkeypatch) -> None:
        from tests._fake_llm import text_stream

        import agent.nodes as mod

        _set_compaction(monkeypatch, True)
        mock_provider = AsyncMock()
        mock_provider.chat_stream = text_stream("Final.")
        mock_provider.chat.return_value = "SUMMARY-OF-EARLY-HISTORY"
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        messages = [
            HumanMessage(content=f"early {i}") for i in range(14)
        ] + [
            HumanMessage(content="current question"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "search_memories_tool", "args": {}, "type": "tool_call"}
                ],
            ),
            ToolMessage(content="Found 1", tool_call_id="c1", name="search_memories_tool"),
        ]
        result = await mod.generate_final_node(_agent_state(messages))
        system = next(p for p in result["final_prompt"] if p["role"] == "system")
        assert "SUMMARY-OF-EARLY-HISTORY" in system["content"]
        assert "<summary>" in system["content"]

    @pytest.mark.asyncio
    async def test_generate_final_no_summary_when_disabled(self, monkeypatch) -> None:
        from tests._fake_llm import text_stream

        import agent.nodes as mod

        _set_compaction(monkeypatch, False)
        mock_provider = AsyncMock()
        mock_provider.chat_stream = text_stream("Final.")
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        messages = [
            HumanMessage(content=f"early {i}") for i in range(14)
        ] + [
            HumanMessage(content="current question"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "search_memories_tool", "args": {}, "type": "tool_call"}
                ],
            ),
            ToolMessage(content="Found 1", tool_call_id="c1", name="search_memories_tool"),
        ]
        result = await mod.generate_final_node(_agent_state(messages))
        system = next(p for p in result["final_prompt"] if p["role"] == "system")
        assert "<summary>" not in system["content"]
        assert "SUMMARY-OF-EARLY-HISTORY" not in system["content"]


def _agent_state(messages: list) -> dict:
    from agent.state import AgentState

    return AgentState(
        messages=messages,
        final_response=None,
        final_prompt=None,
        error=None,
        pending_approval=None,
    )
