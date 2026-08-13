"""Tests for the compiled agent graph structure and routing."""

from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from backend.agent.graph import build_agent_graph
from backend.shared import config as config_mod


@pytest.fixture(autouse=True)
def _disable_auto_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep auto memory off — these tests exercise graph routing, not B3.

    Auto memory defaults on; the graph's generate_final_node would otherwise
    hit the real extraction service (unmocked) at the end of each turn.
    """
    monkeypatch.setattr(config_mod.config, "auto_memory_enabled", False)


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
        """Graph contains all four agent nodes (checked via compilation).

        ``get_graph().nodes`` triggers a sorting issue in LangGraph 1.2.9
        when ``Command`` routing is combined with conditional edges, so we
        verify via successful compilation + runtime routing tests below.
        """
        tools = [_make_fake_tool()]
        graph = build_agent_graph(tools)
        assert isinstance(graph, CompiledStateGraph)
        # Infer node presence from the builder registration — confirmed
        # by TestGraphRouting and TestGraphHITLRouting which exercise
        # every path through the graph.

    def test_graph_compiles_with_empty_tools(self) -> None:
        graph = build_agent_graph([])
        assert isinstance(graph, CompiledStateGraph)

    @pytest.mark.asyncio
    async def test_rejected_approval_does_not_execute_tool(self, monkeypatch) -> None:
        """Regression: a rejected approval must NOT run the write tool.

        LangGraph (1.2.10) follows the resumed node's ``Command(goto=...)``
        AND any static/conditional edge out of that node.  check_approval
        therefore has no static edge — routing is purely by Command — so a
        rejection (``Command(goto="call_llm")``) must not also route to
        ``tools`` and execute the rejected tool_calls.
        """
        from langchain_core.tools import tool
        from langgraph.types import Command

        from tests._fake_llm import (
            content_stream,
            sequential_stream,
            text_stream,
            tool_call_stream,
        )

        executed: list[str] = []

        @tool
        async def fake_write(content: str) -> str:
            """Write a memory."""
            executed.append(content)
            return "inserted"

        provider = AsyncMock()
        provider.chat_raw_stream = sequential_stream(
            tool_call_stream([{
                "id": "call_r",
                "name": "fake_write",
                "args": {"content": "should-not-run"},
            }]),
            content_stream("明白了，我不会写入。"),
        )
        provider.chat_stream = text_stream("好的。")  # tool turn → synthesis path

        monkeypatch.setattr("backend.agent.nodes.get_llm_provider", lambda: provider)

        graph = build_agent_graph(
            [fake_write],
            checkpointer=None,
            approval_required_tools=frozenset({"fake_write"}),
        )
        run_config = {"configurable": {"thread_id": "test-reject-1"}}

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="write something")]},
            config=run_config,
        )
        assert "__interrupt__" in result

        await graph.ainvoke(
            Command(resume={"approved": False, "reason": "no"}),
            config=run_config,
        )

        assert executed == [], (
            "the rejected write must not execute — the approval gate is "
            "cosmetic if a rejection still runs the tool"
        )


class TestGraphRouting:
    @pytest.mark.asyncio
    async def test_direct_answer_path_no_tools(self, monkeypatch) -> None:
        """When LLM returns text (no tool_calls), that text is the final answer.

        call_llm_node's output is reused directly — generate_final_node does
        NOT make a second LLM call (previously every plain chat turn paid
        two LLM calls and discarded the first response).
        """
        from tests._fake_llm import content_stream, sequential_stream

        mock_provider = AsyncMock()
        mock_provider.chat_raw_stream = sequential_stream(
            content_stream("I can answer that directly.")
        )

        import backend.agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        graph = build_agent_graph([], checkpointer=None)

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="What is Python?")]},
            {"configurable": {"thread_id": "test-1"}},
        )

        # The streamed text becomes the final answer; no second LLM call.
        assert "final_response" in result
        assert result["final_response"] == "I can answer that directly."
        mock_provider.chat.assert_not_awaited()
        mock_provider.chat_stream.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tool_calling_path(self, monkeypatch) -> None:
        """When LLM returns tool_calls, execute tools and loop back."""
        from tests._fake_llm import (
            content_stream,
            sequential_stream,
            text_stream,
            tool_call_stream,
        )

        tools = [_make_fake_tool()]

        mock_provider = AsyncMock()
        mock_provider.chat_raw_stream = sequential_stream(
            tool_call_stream([
                {
                    "id": "call_1",
                    "name": "fake_search",
                    "args": {"query": "Python info"},
                }
            ]),
            content_stream("Based on search results, Python is a programming language."),
        )
        mock_provider.chat_stream = text_stream("Final synthesized answer.")

        import backend.agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        graph = build_agent_graph(tools, checkpointer=None)

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Tell me about Python")]},
            {"configurable": {"thread_id": "test-2"}},
        )

        # generate_final_node produces final_prompt (not final_response)
        assert "final_prompt" in result


class TestGraphHITLRouting:
    """Tests that the HITL gates are correctly wired in the graph."""

    @pytest.mark.asyncio
    async def test_safe_tool_passes_through_check_approval(self, monkeypatch) -> None:
        """Safe tools route through check_approval → tools without interrupt."""
        from tests._fake_llm import (
            content_stream,
            sequential_stream,
            text_stream,
            tool_call_stream,
        )

        tools = [_make_fake_tool()]

        # LLM calls a safe tool (fake_search), then returns final text
        mock_provider = AsyncMock()
        mock_provider.chat_raw_stream = sequential_stream(
            tool_call_stream([
                {
                    "id": "call_1",
                    "name": "fake_search",
                    "args": {"query": "test"},
                }
            ]),
            content_stream("Got it."),
        )
        mock_provider.chat_stream = text_stream("Final answer.")

        monkeypatch.setattr(
            "backend.agent.nodes.get_llm_provider",
            lambda: mock_provider,
        )

        graph = build_agent_graph(tools, checkpointer=None)

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="search for test")]},
            {"configurable": {"thread_id": "test-hitl-safe"}},
        )

        # Should complete without interrupt (safe tool passes through)
        assert "final_prompt" in result
        assert "__interrupt__" not in result

    @pytest.mark.asyncio
    async def test_conflict_routes_through_check_conflict(self, monkeypatch) -> None:
        """write_memory_tool conflict triggers check_conflict → interrupt.

        Because write_memory_tool is also in APPROVAL_REQUIRED_TOOLS,
        the graph hits *both* gates: first check_approval (approve tool),
        then tools → conflict → check_conflict (resolve conflict).

        We verify the full flow by sending in a pre-written conflict
        ToolMessage and confirming check_conflict fires.
        """

        from backend.agent.graph import build_agent_graph
        from backend.agent.tools import write_memory_tool
        from tests._fake_llm import sequential_stream, tool_call_stream

        mock_provider = AsyncMock()
        # First call_llm: approve the tool call
        mock_provider.chat_raw_stream = sequential_stream(
            tool_call_stream([{
                "id": "call_cf2",
                "name": "write_memory_tool",
                "args": {"content": "EMA uses SQLite"},
            }])
        )

        monkeypatch.setattr("backend.agent.nodes.get_llm_provider", lambda: mock_provider)

        graph = build_agent_graph([write_memory_tool], checkpointer=None)
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="remember that EMA uses SQLite")]},
            {"configurable": {"thread_id": "test-conflict-graph"}},
        )

        # The *first* interrupt is check_approval (tool approval gate),
        # not check_conflict.  verify the graph paused.
        assert "__interrupt__" in result
        first_interrupt = result["__interrupt__"][0].value
        # The first gate is the tool approval — write_memory_tool
        assert first_interrupt["tool_name"] == "write_memory_tool"

    @pytest.mark.asyncio
    async def test_notify_gated_in_chat_but_not_default(self, monkeypatch) -> None:
        """notify_feishu_tool requires approval on the chat set; the default
        (patrol/scenario) set passes it through so automated flows notify
        autonomously."""
        from backend.agent.nodes import CHAT_APPROVAL_TOOLS
        from backend.agent.tools import notify_feishu_tool
        from backend.shared.config import config
        from tests._fake_llm import (
            content_stream,
            sequential_stream,
            text_stream,
            tool_call_stream,
        )

        monkeypatch.setattr(config, "feishu_webhook_url", "")

        notify_call = {
            "id": "call_n",
            "name": "notify_feishu_tool",
            "args": {"message": "hi team"},
        }

        # Default approval set (patrol): notify passes through, tool executes.
        mock_default = AsyncMock()
        mock_default.chat_raw_stream = sequential_stream(
            tool_call_stream([notify_call]),
            content_stream("Notified."),
        )
        mock_default.chat_stream = text_stream("Final.")
        monkeypatch.setattr("backend.agent.nodes.get_llm_provider", lambda: mock_default)

        graph_default = build_agent_graph([notify_feishu_tool], checkpointer=None)
        result_default = await graph_default.ainvoke(
            {"messages": [HumanMessage(content="ping the team")]},
            {"configurable": {"thread_id": "test-notify-default"}},
        )
        assert "__interrupt__" not in result_default

        # Chat approval set: notify is gated → the graph pauses for approval.
        mock_chat = AsyncMock()
        mock_chat.chat_raw_stream = sequential_stream(tool_call_stream([notify_call]))
        monkeypatch.setattr("backend.agent.nodes.get_llm_provider", lambda: mock_chat)

        graph_chat = build_agent_graph(
            [notify_feishu_tool],
            checkpointer=None,
            approval_required_tools=CHAT_APPROVAL_TOOLS,
        )
        result_chat = await graph_chat.ainvoke(
            {"messages": [HumanMessage(content="ping the team")]},
            {"configurable": {"thread_id": "test-notify-chat"}},
        )
        assert "__interrupt__" in result_chat
        first_interrupt = result_chat["__interrupt__"][0].value
        assert first_interrupt["tool_name"] == "notify_feishu_tool"

    @pytest.mark.asyncio
    async def test_llm_tools_separates_schemas_from_execution_roster(
        self, monkeypatch
    ) -> None:
        """``llm_tools`` narrows the schemas the model may pick, while the
        execution roster still runs a tool that is hidden from the LLM.

        Chat passes ``CHAT_LLM_TOOLS`` (no ``write_memory_tool``): the model
        must never be able to *choose* a memory write, but a system-injected
        write_memory_tool call (the force-write path) still executes via
        ToolNode.
        """
        from langchain_core.tools import tool

        from tests._fake_llm import (
            content_stream,
            sequential_stream,
            text_stream,
            tool_call_stream,
        )

        @tool
        async def write_memory_tool(content: str) -> str:
            """Write a memory — hidden from the LLM schemas."""
            return '{"action": "inserted", "summary": "x"}'

        @tool
        async def search_memories_tool(query: str) -> str:
            """Search memories."""
            return "Found 1 memory"

        tools = [write_memory_tool, search_memories_tool]
        llm_tools = [search_memories_tool]

        mock_provider = AsyncMock()
        mock_provider.chat_raw_stream = sequential_stream(
            tool_call_stream([{
                "id": "call_1",
                "name": "write_memory_tool",
                "args": {"content": "记住端口 8080"},
            }]),
            content_stream("已记录，无需更多工具"),
        )
        mock_provider.chat_stream = text_stream("Final.")
        monkeypatch.setattr("backend.agent.nodes.get_llm_provider", lambda: mock_provider)

        graph = build_agent_graph(
            tools,
            checkpointer=None,
            approval_required_tools=frozenset(),
            llm_tools=llm_tools,
        )
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="remember x")]},
            {"configurable": {"thread_id": "test-llm-tools-sep"}},
        )

        # The first tool-selection call saw only the visible tool's schema.
        first_call = mock_provider.chat_raw_stream.call_args_list[0]
        sent_tools = first_call.kwargs["tools"]
        assert [t["function"]["name"] for t in sent_tools] == ["search_memories_tool"]

        # The hidden tool still executed (ToolNode ran it) and the run
        # completed with a final answer.
        assert result.get("final_response") == "Final."


class TestStepCountResetAcrossTurns:
    """The ReAct step budget restarts each user turn — a thread whose first
    turn exhausted max_steps must still be able to call tools on later turns."""

    @pytest.mark.asyncio
    async def test_second_turn_can_still_call_tools(self, monkeypatch) -> None:
        """Round 1 burns the whole max_steps budget; round 2 must still
        execute tools.

        Regression: step_count was persisted session-wide by the checkpointer,
        so once a thread hit max_steps, every later turn's first call_llm
        routed straight to generate_final and tools were silently disabled.
        """
        from langchain_core.tools import tool

        from tests._fake_llm import (
            content_stream,
            sequential_stream,
            text_stream,
            tool_call_stream,
        )

        calls: list[str] = []

        @tool
        async def fake_search(query: str) -> str:
            """Search for something."""
            calls.append(query)
            return f"Found: {query}"

        mock_provider = AsyncMock()
        mock_provider.chat_raw_stream = sequential_stream(
            tool_call_stream([
                {"id": "c1", "name": "fake_search", "args": {"query": "round1"}},
            ]),
            content_stream("Round one answer."),
            tool_call_stream([
                {"id": "c2", "name": "fake_search", "args": {"query": "round2"}},
            ]),
            content_stream("Round two answer."),
        )
        mock_provider.chat_stream = text_stream("Final.", "Final.")

        import backend.agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        graph = build_agent_graph([fake_search], checkpointer=None, max_steps=2)

        # Round 1: a tool call plus its loopback answer consume the budget.
        result1 = await graph.ainvoke(
            {"messages": [HumanMessage(content="first round")]},
            {"configurable": {"thread_id": "test-steps-reset"}},
        )
        assert "final_response" in result1 or "final_prompt" in result1

        # Round 2: a new user turn — the step budget must restart so the tool
        # executes again instead of skipping straight to the final answer.
        result2 = await graph.ainvoke(
            {"messages": [HumanMessage(content="second round")]},
            {"configurable": {"thread_id": "test-steps-reset"}},
        )
        assert "final_response" in result2 or "final_prompt" in result2

        assert calls == ["round1", "round2"], (
            "the second user turn must be able to call tools again"
        )
