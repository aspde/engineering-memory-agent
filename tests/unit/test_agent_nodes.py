"""Tests for agent node functions — mock LLM provider."""

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.nodes import _messages_to_dicts, _to_openai_tools
from agent.state import AgentState


class TestMessageConversion:
    def test_messages_to_dicts_human(self) -> None:
        dicts = _messages_to_dicts([HumanMessage(content="hello")])
        assert dicts == [{"role": "user", "content": "hello"}]

    def test_messages_to_dicts_mixed(self) -> None:
        dicts = _messages_to_dicts([
            HumanMessage(content="hi"),
            AIMessage(content="hey"),
        ])
        assert len(dicts) == 2
        assert dicts[0]["role"] == "user"
        assert dicts[1]["role"] == "assistant"

    def test_messages_to_dicts_preserves_tool_calls(self) -> None:
        """AIMessage with tool_calls must serialize them for the API."""
        dicts = _messages_to_dicts([
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "call_1", "name": "search_memories_tool", "args": {"query": "test"}, "type": "tool_call"}
                ],
            ),
            ToolMessage(content="Found 3 results", tool_call_id="call_1"),
        ])
        assert len(dicts) == 2
        assert dicts[0]["role"] == "assistant"
        assert "tool_calls" in dicts[0]
        assert dicts[1]["role"] == "tool"
        assert dicts[1]["tool_call_id"] == "call_1"

    def test_to_openai_tools_returns_schemas(self) -> None:
        from agent.tools import search_memories_tool

        schemas = _to_openai_tools([search_memories_tool])
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "search_memories_tool"


def _make_state(messages=None, final_response=None, error=None):
    return AgentState(
        messages=messages or [],
        final_response=final_response,
        error=error,
    )


class TestCallLLMNode:
    @pytest.mark.asyncio
    async def test_returns_aimessage_no_tools(self, monkeypatch) -> None:
        """When LLM returns plain text, node produces an AIMessage."""
        mock_provider = AsyncMock()
        mock_provider.chat_raw.return_value = {"content": "Hello, how can I help?"}

        import agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        from agent.tools import ALL_TOOLS

        result = await mod.call_llm_node(
            _make_state([HumanMessage(content="hi")]), tools=ALL_TOOLS
        )
        messages = result["messages"]
        assert len(messages) == 1
        assert isinstance(messages[0], AIMessage)
        assert "how can I help" in messages[0].content

    @pytest.mark.asyncio
    async def test_prepends_system_prompt_once(self, monkeypatch) -> None:
        """First call adds SystemMessage; second call should not duplicate."""
        mock_provider = AsyncMock()
        mock_provider.chat_raw.return_value = {"content": "ok"}

        import agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        from agent.tools import ALL_TOOLS

        result = await mod.call_llm_node(
            _make_state([HumanMessage(content="hi")]), tools=ALL_TOOLS
        )
        call_args = mock_provider.chat_raw.call_args
        sent_messages = call_args.kwargs["messages"]
        assert sent_messages[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_handles_llm_error_gracefully(self, monkeypatch) -> None:
        mock_provider = AsyncMock()
        mock_provider.chat_raw.side_effect = RuntimeError("API down")

        import agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        from agent.tools import ALL_TOOLS

        result = await mod.call_llm_node(
            _make_state([HumanMessage(content="hi")]), tools=ALL_TOOLS
        )
        assert result["error"] is not None
        assert "API down" in result["error"]


class TestGenerateFinalNode:
    @pytest.mark.asyncio
    async def test_produces_final_response(self, monkeypatch) -> None:
        mock_provider = AsyncMock()
        mock_provider.chat.return_value = "Here is the final answer."

        import agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        state = _make_state(
            messages=[
                HumanMessage(content="What is EMA?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "call_1", "name": "search_memories_tool",
                         "args": {"query": "EMA"}, "type": "tool_call"}
                    ],
                ),
                ToolMessage(
                    content="Found 1 memory: EMA is an Engineering Memory Agent.",
                    tool_call_id="call_1",
                ),
            ],
        )

        result = await mod.generate_final_node(state)
        assert result["final_response"] == "Here is the final answer."

    @pytest.mark.asyncio
    async def test_handles_error(self, monkeypatch) -> None:
        mock_provider = AsyncMock()
        mock_provider.chat.side_effect = RuntimeError("LLM timeout")

        import agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        result = await mod.generate_final_node(_make_state())
        assert "error" in result["final_response"].lower()
        assert result["error"] is not None
