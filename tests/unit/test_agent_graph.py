"""Tests for the compiled agent graph structure and routing."""

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph

from agent.graph import build_agent_graph


def _make_fake_tool() -> object:
    from langchain_core.tools import tool

    @tool
    async def fake_search(query: str) -> str:
        """Search for something."""
        return f"Found: {query}"

    return fake_search


class TestGraphStructure:
    def test_graph_compiles_with_tools(self) -> None:
        tools = [_make_fake_tool()]
        graph = build_agent_graph(tools)
        assert isinstance(graph, CompiledStateGraph)

    def test_graph_has_expected_nodes(self) -> None:
        tools = [_make_fake_tool()]
        graph = build_agent_graph(tools)
        nodes = list(graph.get_graph().nodes.keys())
        assert "call_llm" in nodes
        assert "tools" in nodes
        assert "generate_final" in nodes

    def test_graph_compiles_with_empty_tools(self) -> None:
        graph = build_agent_graph([])
        assert isinstance(graph, CompiledStateGraph)


class TestGraphRouting:
    @pytest.mark.asyncio
    async def test_direct_answer_path_no_tools(self, monkeypatch) -> None:
        """When LLM returns text (no tool_calls), route to generate_final."""
        mock_provider = AsyncMock()
        mock_provider.chat_raw.return_value = {"content": "I can answer that directly."}
        mock_provider.chat.return_value = "Here is the answer."

        import agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        graph = build_agent_graph([], checkpointer=None)

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="What is Python?")]},
            {"configurable": {"thread_id": "test-1"}},
        )

        assert result["final_response"] == "Here is the answer."

    @pytest.mark.asyncio
    async def test_tool_calling_path(self, monkeypatch) -> None:
        """When LLM returns tool_calls, execute tools and loop back."""
        tools = [_make_fake_tool()]

        # First call_llm: LLM returns a tool_call
        # Second call_llm (after tool): LLM returns final text
        mock_provider = AsyncMock()
        mock_provider.chat_raw.side_effect = [
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "fake_search",
                        "args": {"query": "Python info"},
                    }
                ],
            },
            {"content": "Based on search results, Python is a programming language."},
        ]
        mock_provider.chat.return_value = "Python is a programming language."

        import agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        graph = build_agent_graph(tools, checkpointer=None)

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Tell me about Python")]},
            {"configurable": {"thread_id": "test-2"}},
        )

        # The graph should have completed (via generate_final or direct call_llm)
        assert "final_response" in result
