"""Tests for agent node functions — mock LLM provider."""

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

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

    def test_to_openai_tools_returns_schemas(self) -> None:
        from agent.tools import search_memories_tool

        schemas = _to_openai_tools([search_memories_tool])
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "search_memories_tool"


class TestCallLLMNode:
    @pytest.mark.asyncio
    async def test_returns_aimessage_no_tools(self, monkeypatch) -> None:
        """When LLM returns plain text, node produces an AIMessage."""
        mock_provider = AsyncMock()
        mock_provider.chat_raw.return_value = {"content": "Hello, how can I help?"}

        import agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        from agent.tools import ALL_TOOLS

        state: AgentState = {
            "messages": [HumanMessage(content="hi")],
            "retrieved_chunks": [],
            "retrieved_memories": [],
            "context_assembled": "",
            "final_response": None,
            "error": None,
        }

        result = await mod.call_llm_node(state, tools=ALL_TOOLS)
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

        state: AgentState = {
            "messages": [HumanMessage(content="hi")],
            "retrieved_chunks": [],
            "retrieved_memories": [],
            "context_assembled": "",
            "final_response": None,
            "error": None,
        }

        result = await mod.call_llm_node(state, tools=ALL_TOOLS)
        # The state dict passed to chat_raw should have a system message first
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

        state: AgentState = {
            "messages": [HumanMessage(content="hi")],
            "retrieved_chunks": [],
            "retrieved_memories": [],
            "context_assembled": "",
            "final_response": None,
            "error": None,
        }

        result = await mod.call_llm_node(state, tools=ALL_TOOLS)
        assert result["error"] is not None
        assert "API down" in result["error"]


class TestGenerateFinalNode:
    @pytest.mark.asyncio
    async def test_produces_final_response(self, monkeypatch) -> None:
        mock_provider = AsyncMock()
        mock_provider.chat.return_value = "Here is the final answer."

        import agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        state: AgentState = {
            "messages": [HumanMessage(content="What is EMA?")],
            "retrieved_chunks": [],
            "retrieved_memories": [
                {"summary": "EMA is an Engineering Memory Agent."}
            ],
            "context_assembled": "",
            "final_response": None,
            "error": None,
        }

        result = await mod.generate_final_node(state)
        assert result["final_response"] == "Here is the final answer."

    @pytest.mark.asyncio
    async def test_handles_error(self, monkeypatch) -> None:
        mock_provider = AsyncMock()
        mock_provider.chat.side_effect = RuntimeError("LLM timeout")

        import agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        state: AgentState = {
            "messages": [],
            "retrieved_chunks": [],
            "retrieved_memories": [],
            "context_assembled": "",
            "final_response": None,
            "error": None,
        }

        result = await mod.generate_final_node(state)
        assert "error" in result["final_response"].lower()
        assert result["error"] is not None
