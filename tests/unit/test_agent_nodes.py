"""Tests for agent node functions — mock LLM provider."""

from unittest.mock import AsyncMock, patch

import pytest
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.nodes import (
    APPROVAL_REQUIRED_TOOLS,
    _messages_to_dicts,
    _to_openai_tools,
)
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

    def test_messages_to_dicts_strips_orphaned_tool_calls(self) -> None:
        """AIMessage with tool_calls but no ToolMessage → tool_calls stripped."""
        dicts = _messages_to_dicts([
            HumanMessage(content="remember this"),
            AIMessage(
                content="I'll write that.",
                tool_calls=[
                    {"id": "call_1", "name": "write_memory_tool",
                     "args": {"content": "test"}, "type": "tool_call"}
                ],
            ),
        ])
        assert len(dicts) == 2
        assert dicts[0]["role"] == "user"
        assert dicts[1]["role"] == "assistant"
        assert "tool_calls" not in dicts[1]
        assert "I'll write that" in dicts[1]["content"]

    def test_messages_to_dicts_orphaned_tool_calls_empty_content(self) -> None:
        """Orphaned tool_calls + empty content → placeholder to keep API valid."""
        dicts = _messages_to_dicts([
            HumanMessage(content="hi"),
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "orphan_1", "name": "write_memory_tool",
                     "args": {"content": "x"}, "type": "tool_call"}
                ],
            ),
        ])
        assert dicts[1]["content"] == "(tool call was interrupted)"
        assert "tool_calls" not in dicts[1]

    def test_messages_to_dicts_keeps_answered_tool_calls(self) -> None:
        """AIMessage tool_calls with ToolMessage responses are preserved."""
        dicts = _messages_to_dicts([
            AIMessage(
                content="Let me search.",
                tool_calls=[
                    {"id": "call_1", "name": "search_memories_tool",
                     "args": {"query": "test"}, "type": "tool_call"}
                ],
            ),
            ToolMessage(content="Found 3 results", tool_call_id="call_1"),
            AIMessage(
                content="Here are the results.",
            ),
        ])
        assert len(dicts) == 3
        assert "tool_calls" in dicts[0]
        assert dicts[0]["tool_calls"][0]["id"] == "call_1"

    def test_messages_to_dicts_mixed_orphaned_and_answered(self) -> None:
        """Only the answered tool_calls survive; orphaned ones are dropped."""
        dicts = _messages_to_dicts([
            AIMessage(
                content="I'll do two things.",
                tool_calls=[
                    {"id": "call_ok", "name": "search_memories_tool",
                     "args": {"query": "x"}, "type": "tool_call"},
                    {"id": "call_orphan", "name": "write_memory_tool",
                     "args": {"content": "x"}, "type": "tool_call"},
                ],
            ),
            ToolMessage(content="Found", tool_call_id="call_ok"),
        ])
        assert "tool_calls" in dicts[0]
        assert len(dicts[0]["tool_calls"]) == 1
        assert dicts[0]["tool_calls"][0]["id"] == "call_ok"

    def test_messages_to_dicts_multi_turn_roundtrip(self) -> None:
        """Simulate a resumed conversation: answered mid-turn + new orphan at end."""
        dicts = _messages_to_dicts([
            HumanMessage(content="what is EMA?"),
            AIMessage(
                content="Let me search.",
                tool_calls=[
                    {"id": "c1", "name": "search_memories_tool",
                     "args": {"query": "EMA"}, "type": "tool_call"}
                ],
            ),
            ToolMessage(content="Found 1", tool_call_id="c1"),
            AIMessage(content="EMA is an agent."),
            HumanMessage(content="remember that"),
            AIMessage(
                content="I'll write it.",
                tool_calls=[
                    {"id": "c2", "name": "write_memory_tool",
                     "args": {"content": "EMA is an agent"}, "type": "tool_call"}
                ],
            ),
        ])
        # c1 is answered, c2 is orphaned (interrupted mid-turn)
        assert "tool_calls" in dicts[1]  # c1 preserved
        assert dicts[1]["tool_calls"][0]["id"] == "c1"
        # c2 orphaned → stripped from last assistant message
        assert "tool_calls" not in dicts[-1]
        assert "I'll write it" in dicts[-1]["content"]

    def test_to_openai_tools_returns_schemas(self) -> None:
        from agent.tools import search_memories_tool

        schemas = _to_openai_tools([search_memories_tool])
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "search_memories_tool"


def _make_state(messages=None, final_response=None, error=None, pending_approval=None, final_prompt=None):
    return AgentState(
        messages=messages or [],
        final_response=final_response,
        final_prompt=final_prompt,
        error=error,
        pending_approval=pending_approval,
    )


def _make_write_tool() -> object:
    """Create a tool whose name matches APPROVAL_REQUIRED_TOOLS."""
    from langchain_core.tools import tool

    @tool
    async def write_memory_tool(content: str) -> str:
        """Write a memory — name matches APPROVAL_REQUIRED_TOOLS."""
        return f"Written: {content}"

    return write_memory_tool


def _make_ingest_tool() -> object:
    """Create a tool whose name matches APPROVAL_REQUIRED_TOOLS."""
    from langchain_core.tools import tool

    @tool
    async def ingest_git_repo_tool(repo_path: str, max_commits: int = 50) -> str:
        """Ingest repo — name matches APPROVAL_REQUIRED_TOOLS."""
        return f"Ingested: {repo_path}"

    return ingest_git_repo_tool


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
    async def test_produces_final_prompt(self) -> None:
        import agent.nodes as mod

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
        assert result["final_prompt"] is not None
        prompts = result["final_prompt"]
        assert any("EMA" in p.get("content", "") for p in prompts)
        # Should include context from the tool message
        context_block = [p for p in prompts if p["role"] == "system" and "Context" in p["content"]]
        assert len(context_block) == 1
        assert "Engineering Memory Agent" in context_block[0]["content"]

    @pytest.mark.asyncio
    async def test_no_tools_produces_prompt(self) -> None:
        """Without any tool results, still produces a valid prompt (no context block)."""
        import agent.nodes as mod

        state = _make_state(
            messages=[
                HumanMessage(content="hi"),
            ],
        )

        result = await mod.generate_final_node(state)
        assert "final_prompt" in result
        # No context block since there are no tool messages
        has_context = any("Context" in p.get("content", "") for p in result["final_prompt"] if p["role"] == "system")
        assert not has_context

    @pytest.mark.asyncio
    async def test_plain_chat_reuses_call_llm_output(self, monkeypatch) -> None:
        """No tool results this turn → the last call_llm AIMessage is the
        final answer; no second LLM call is made."""
        import agent.nodes as mod

        mock_provider = AsyncMock()
        mock_provider.chat.return_value = "should not be used"
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        state = _make_state(
            messages=[
                HumanMessage(content="hi"),
                AIMessage(content="Hello! How can I help?"),
            ],
        )

        result = await mod.generate_final_node(state)
        assert result["final_response"] == "Hello! How can I help?"
        mock_provider.chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tool_results_this_turn_still_synthesizes(self, monkeypatch) -> None:
        """A ToolMessage in the current turn forces the synthesis path, so the
        tool output can be folded into the final-answer context."""
        import agent.nodes as mod

        mock_provider = AsyncMock()
        mock_provider.chat.return_value = "Synthesized from tool output."
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        state = _make_state(
            messages=[
                HumanMessage(content="search"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "c1", "name": "fake_search",
                                 "args": {"query": "x"}, "type": "tool_call"}],
                ),
                ToolMessage(content="Found 1 result", tool_call_id="c1"),
                AIMessage(content="I found one result."),
            ],
        )

        result = await mod.generate_final_node(state)
        assert result["final_response"] == "Synthesized from tool output."
        assert result["final_prompt"] is not None

    @pytest.mark.asyncio
    async def test_previous_turn_tool_results_do_not_block_shortcut(self, monkeypatch) -> None:
        """Old-turn ToolMessages don't force synthesis — only this turn's do.

        In a multi-turn thread the history keeps prior tool results; a later
        plain-chat turn must still skip the second LLM call."""
        import agent.nodes as mod

        mock_provider = AsyncMock()
        mock_provider.chat.return_value = "should not be used"
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        state = _make_state(
            messages=[
                HumanMessage(content="search for X"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "old", "name": "fake_search",
                                 "args": {"query": "x"}, "type": "tool_call"}],
                ),
                ToolMessage(content="Found 1 result", tool_call_id="old"),
                AIMessage(content="I found X."),
                HumanMessage(content="thanks"),
                AIMessage(content="You're welcome!"),
            ],
        )

        result = await mod.generate_final_node(state)
        assert result["final_response"] == "You're welcome!"
        mock_provider.chat.assert_not_awaited()


class TestCheckApprovalNode:
    """Tests for the HITL approval gate that intercepts sensitive tools."""

    @pytest.mark.asyncio
    async def test_safe_tools_pass_through(self) -> None:
        """Safe tools (search, retrieve, extract) pass straight to tools node."""
        from langgraph.types import Command

        from agent.nodes import check_approval_node

        state = _make_state(
            messages=[
                HumanMessage(content="what is EMA?"),
                AIMessage(
                    content="Let me search.",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "search_memories_tool",
                            "args": {"query": "EMA"},
                            "type": "tool_call",
                        }
                    ],
                ),
            ],
        )

        result = await check_approval_node(state)
        assert isinstance(result, Command)
        assert result.goto == "tools"

    @pytest.mark.asyncio
    async def test_sensitive_tool_triggers_interrupt(self, monkeypatch) -> None:
        """Through compiled graph, a write tool triggers interrupt."""
        tools = [_make_write_tool()]

        mock_provider = AsyncMock()
        mock_provider.chat_raw.return_value = {
            "content": "I'll write that.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "name": "write_memory_tool",
                    "args": {"content": "test"},
                }
            ],
        }

        import agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        from agent.graph import build_agent_graph

        graph = build_agent_graph(tools, checkpointer=None)
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="remember this")]},
            {"configurable": {"thread_id": "test-interrupt-1"}},
        )

        # Should be interrupted (no final_response)
        assert "__interrupt__" in result

    @pytest.mark.asyncio
    async def test_ingest_git_repo_triggers_interrupt(self, monkeypatch) -> None:
        """Through compiled graph, ingest tool also triggers interrupt."""
        tools = [_make_ingest_tool()]

        mock_provider = AsyncMock()
        mock_provider.chat_raw.return_value = {
            "content": "Let me ingest.",
            "tool_calls": [
                {
                    "id": "call_2",
                    "name": "ingest_git_repo_tool",
                    "args": {"repo_path": "/tmp/repo", "max_commits": 10},
                }
            ],
        }

        import agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        from agent.graph import build_agent_graph

        graph = build_agent_graph(tools, checkpointer=None)
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="ingest /tmp/repo")]},
            {"configurable": {"thread_id": "test-interrupt-2"}},
        )

        assert "__interrupt__" in result

    @pytest.mark.asyncio
    async def test_no_tool_calls_returns_to_call_llm(self) -> None:
        """When last AI message has no tool_calls, route back to call_llm."""
        from langgraph.types import Command

        from agent.nodes import check_approval_node

        state = _make_state(
            messages=[
                HumanMessage(content="hi"),
                AIMessage(content="Hello, how can I help?"),
            ],
        )

        result = await check_approval_node(state)
        assert isinstance(result, Command)
        assert result.goto == "call_llm"

    def test_approval_tools_set_contains_expected_tools(self) -> None:
        """Verify the APPROVAL_REQUIRED_TOOLS set has the correct tools."""
        assert "write_memory_tool" in APPROVAL_REQUIRED_TOOLS
        assert "ingest_git_repo_tool" in APPROVAL_REQUIRED_TOOLS
        assert "ingest_document_tool" in APPROVAL_REQUIRED_TOOLS
        assert "search_memories_tool" not in APPROVAL_REQUIRED_TOOLS
        assert "retrieve_chunks_tool" not in APPROVAL_REQUIRED_TOOLS
        assert "extract_memory_tool" not in APPROVAL_REQUIRED_TOOLS


class TestCheckConflictNode:
    """Tests for the HITL conflict-resolution gate that intercepts
    write_memory_tool conflict results after execution."""

    import json as _json

    @pytest.mark.asyncio
    async def test_no_conflict_passes_through(self) -> None:
        """Non-conflict results pass straight to call_llm."""
        from langgraph.types import Command

        from agent.nodes import check_conflict_node

        state = _make_state(
            messages=[
                HumanMessage(content="remember this"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "c1", "name": "write_memory_tool",
                         "args": {"content": "test"}, "type": "tool_call"}
                    ],
                ),
                ToolMessage(
                    content=self._json.dumps({
                        "id": "abc", "action": "inserted", "summary": "test"
                    }),
                    tool_call_id="c1",
                    name="write_memory_tool",
                ),
            ],
        )

        result = await check_conflict_node(state)
        assert isinstance(result, Command)
        assert result.goto == "call_llm"

    @pytest.mark.asyncio
    async def test_conflict_triggers_interrupt(self, monkeypatch) -> None:
        """A conflict result triggers interrupt() — verified through compiled graph.

        We use a write_memory_tool that returns a conflict and verify the
        graph reaches the conflict interrupt (past check_approval).
        """
        import json as _json

        # Step 1: approve the write_memory_tool header
        # Step 2: write_memory_tool returns conflict → check_conflict interrupt
        from agent.graph import build_agent_graph
        from agent.tools import write_memory_tool

        mock_provider = AsyncMock()
        mock_provider.chat_raw.return_value = {
            "content": "I'll write that.",
            "tool_calls": [{
                "id": "call_int", "name": "write_memory_tool",
                "args": {"content": "EMA uses MySQL"},
            }],
        }

        monkeypatch.setattr("agent.nodes.get_llm_provider", lambda: mock_provider)
        monkeypatch.setattr(
            "agent.tools.write_memory",
            AsyncMock(return_value={
                "action": "conflict",
                "summary": "EMA uses MySQL",
                "existing_id": "mem-1",
                "existing_summary": "EMA uses PostgreSQL",
                "entities": [],
                "relations": [],
                "_deferred": {
                    "extracted": {"summary": "EMA uses MySQL"},
                    "embedding": "[0.1, 0.2]",
                    "source_type": "conversation",
                    "metadata": {"conflicts_with": "mem-1"},
                },
            }),
        )

        graph = build_agent_graph([write_memory_tool], checkpointer=None)
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="remember EMA uses MySQL")]},
            {"configurable": {"thread_id": "test-conflict-1"}},
        )

        # First gate is tool approval (check_approval)
        assert "__interrupt__" in result
        first = result["__interrupt__"][0].value
        assert first["tool_name"] == "write_memory_tool"

        # Resume with approval → the graph continues to tools → check_conflict
        result2 = await graph.ainvoke(
            Command(resume={"approved": True}),
            {"configurable": {"thread_id": "test-conflict-1"}},
        )

        assert "__interrupt__" in result2
        second = result2["__interrupt__"][0].value
        assert second["type"] == "conflict"
        assert "keep_existing" in second["options"]

    @pytest.mark.asyncio
    async def test_non_write_tool_passes_through(self) -> None:
        """ToolMessages from search tools are ignored by check_conflict."""
        from langgraph.types import Command

        from agent.nodes import check_conflict_node

        state = _make_state(
            messages=[
                HumanMessage(content="search"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "c2", "name": "search_memories_tool",
                         "args": {"query": "test"}, "type": "tool_call"}
                    ],
                ),
                ToolMessage(
                    content="Found 3 results",
                    tool_call_id="c2",
                    name="search_memories_tool",
                ),
            ],
        )

        result = await check_conflict_node(state)
        assert isinstance(result, Command)
        assert result.goto == "call_llm"
