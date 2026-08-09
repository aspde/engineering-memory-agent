"""Tests for agent node functions — mock LLM provider."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langgraph.types import Command
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent.nodes import (
    APPROVAL_REQUIRED_TOOLS,
    _MAX_TOOL_CONTENT_CHARS,
    _estimate_tokens,
    _heuristic_tokens,
    _messages_to_dicts,
    _to_openai_tools,
    _truncate_tool_content,
    _window_messages,
)
from agent.state import AgentState
from backend.shared import config as config_mod


@pytest.fixture(autouse=True)
def _disable_auto_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep auto memory off — these tests exercise the graph nodes, not B3.

    Auto memory defaults on; generate_final_node would otherwise hit the
    real extraction service (unmocked) at the end of each turn.
    """
    monkeypatch.setattr(config_mod.config, "auto_memory_enabled", False)


@pytest.fixture(autouse=True)
def _heuristic_token_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the heuristic token estimate — windowing assertions pin exact
    token counts that tiktoken (when available) would shift."""
    import agent.nodes as mod

    monkeypatch.setattr(mod, "_get_tokenizer", lambda: None)


def _disable_compaction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable conversation compaction for tests that assert pure windowing."""
    monkeypatch.setattr(config_mod.config, "conversation_compaction_enabled", False)


def _set_budget(monkeypatch: pytest.MonkeyPatch, tokens: int) -> None:
    """Set the context-window token budget — what drives windowing."""
    monkeypatch.setattr(config_mod.config, "context_token_budget", tokens)


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
        assert dicts[1]["role"] == "tool"
        assert dicts[1]["tool_call_id"] == "call_1"

    def test_messages_to_dicts_drops_orphaned_tool_result(self) -> None:
        """A ToolMessage whose tool-call AIMessage was windowed/compacted out
        must not be emitted — OpenAI-compatible APIs reject a ``role=tool``
        message whose ``tool_call_id`` has no preceding assistant tool_calls.
        The window boundary can land between the two (large prose + tool_calls
        on the AIMessage), leaving only the ToolMessage in the suffix."""
        dicts = _messages_to_dicts([
            ToolMessage(content="Found 3 results", tool_call_id="call_1"),
            HumanMessage(content="thanks"),
            AIMessage(content="you're welcome"),
        ])
        # The orphaned result is dropped; the rest serializes normally.
        assert [d["role"] for d in dicts] == ["user", "assistant"]

    def test_messages_to_dicts_keeps_tool_result_when_parent_retained(self) -> None:
        """A retained tool-call AIMessage keeps its ToolMessage result."""
        dicts = _messages_to_dicts([
            AIMessage(
                content="Let me search.",
                tool_calls=[
                    {"id": "call_1", "name": "search_memories_tool",
                     "args": {"query": "test"}, "type": "tool_call"}
                ],
            ),
            ToolMessage(content="Found 3 results", tool_call_id="call_1"),
        ])
        assert dicts[1]["role"] == "tool"
        assert dicts[1]["tool_call_id"] == "call_1"

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

    def test_to_openai_tools_cached_per_tool_set(self, monkeypatch) -> None:
        """Serialisation runs once per tool name — later calls hit the cache.

        A tool unique to this test keeps the module-level cache key from
        colliding with other tests' schemas.
        """
        from langchain_core.tools import tool

        @tool
        async def unique_schema_tool(query: str) -> str:
            """A tool only this test uses."""
            return "ok"

        args_schema = MagicMock()
        args_schema.model_json_schema.return_value = {"type": "object", "properties": {}}
        monkeypatch.setattr(unique_schema_tool, "args_schema", args_schema)

        first = _to_openai_tools([unique_schema_tool])
        second = _to_openai_tools([unique_schema_tool])
        assert args_schema.model_json_schema.call_count == 1
        assert first == second
        assert first is not second  # shallow copy — callers can't mutate the cache


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
        from tests._fake_llm import content_stream, sequential_stream

        mock_provider = AsyncMock()
        mock_provider.chat_raw_stream = sequential_stream(
            content_stream("Hello, how can I help?")
        )

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
        from tests._fake_llm import content_stream, sequential_stream

        mock_provider = AsyncMock()
        mock_provider.chat_raw_stream = sequential_stream(content_stream("ok"))

        import agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        from agent.tools import ALL_TOOLS

        result = await mod.call_llm_node(
            _make_state([HumanMessage(content="hi")]), tools=ALL_TOOLS
        )
        call_args = mock_provider.chat_raw_stream.call_args
        sent_messages = call_args.kwargs["messages"]
        assert sent_messages[0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_handles_llm_error_gracefully(self, monkeypatch) -> None:
        from tests._fake_llm import raise_stream, sequential_stream

        mock_provider = AsyncMock()
        mock_provider.chat_raw_stream = sequential_stream(
            raise_stream(RuntimeError("API down"))
        )

        import agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        from agent.tools import ALL_TOOLS

        result = await mod.call_llm_node(
            _make_state([HumanMessage(content="hi")]), tools=ALL_TOOLS
        )
        assert result["error"] is not None
        assert "API down" in result["error"]

    @pytest.mark.asyncio
    async def test_streams_error_text_on_llm_failure(self, monkeypatch) -> None:
        """A failed LLM call must reach the client through the stream writer.

        Regression guard: the error text becomes this turn's final_response,
        and the SSE path only renders what the nodes stream — so an
        unstreamed error would leave the assistant message empty on provider
        failure.
        """
        from tests._fake_llm import raise_stream, sequential_stream

        mock_provider = AsyncMock()
        mock_provider.chat_raw_stream = sequential_stream(
            raise_stream(RuntimeError("API down"))
        )

        import agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        emitted: list[dict[str, str]] = []
        monkeypatch.setattr(mod, "_stream_writer", lambda: emitted.append)

        await mod.call_llm_node(_make_state([HumanMessage(content="hi")]), tools=[])

        error_tokens = [e for e in emitted if e.get("type") == "token"]
        assert error_tokens, "the error must be streamed as a token"
        # The exception text must not reach the client — only a generic
        # user-facing message is streamed.
        assert all("API down" not in e.get("content", "") for e in error_tokens)

    @pytest.mark.asyncio
    async def test_step_count_resets_on_new_user_turn(self, monkeypatch) -> None:
        """A fresh HumanMessage restarts the ReAct step budget from 0.

        Regression: the checkpointer persisted step_count across turns, so a
        thread whose first turn exhausted MAX_AGENT_STEPS routed every later
        turn's first call_llm straight to generate_final — tools silently
        disabled.  A new turn is exactly when the latest message is a
        HumanMessage.
        """
        from tests._fake_llm import content_stream, sequential_stream

        mock_provider = AsyncMock()
        mock_provider.chat_raw_stream = sequential_stream(content_stream("ok"))

        import agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        from agent.tools import ALL_TOOLS

        # Second user turn — the prior turn already exhausted the budget.
        result = await mod.call_llm_node(
            AgentState(
                messages=[
                    HumanMessage(content="first turn"),
                    AIMessage(content="first answer"),
                    HumanMessage(content="second turn"),
                ],
                final_response=None,
                final_prompt=None,
                error=None,
                pending_approval=None,
                step_count=5,
            ),
            tools=ALL_TOOLS,
        )
        assert result["step_count"] == 1

    @pytest.mark.asyncio
    async def test_step_count_continues_on_loopback(self, monkeypatch) -> None:
        """Loopbacks within a turn keep the current count — no reset.

        An approval rejection (latest message is a ToolMessage, not a
        HumanMessage) is the same turn's loop; resetting there would let a
        ReAct loop spin past max_steps.
        """
        from tests._fake_llm import content_stream, sequential_stream

        mock_provider = AsyncMock()
        mock_provider.chat_raw_stream = sequential_stream(content_stream("ok"))

        import agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        from agent.tools import ALL_TOOLS

        result = await mod.call_llm_node(
            AgentState(
                messages=[
                    HumanMessage(content="turn"),
                    AIMessage(content="", tool_calls=[
                        {"id": "c1", "name": "write_memory_tool",
                         "args": {"content": "x"}, "type": "tool_call"}]),
                    ToolMessage(content="[REJECTED] no", tool_call_id="c1"),
                ],
                final_response=None,
                final_prompt=None,
                error=None,
                pending_approval=None,
                step_count=2,
            ),
            tools=ALL_TOOLS,
        )
        assert result["step_count"] == 3

    @pytest.mark.asyncio
    async def test_llm_error_stub_is_marked_and_never_resent(self, monkeypatch) -> None:
        """A failed call_llm's error stub must not be re-sent to the model.

        Regression: the error AIMessage was appended to the checkpoint and
        later replayed to the LLM as assistant history.  It is now marked and
        skipped by serialization (and by synthesis/compaction transcripts).
        """
        from tests._fake_llm import raise_stream, sequential_stream

        import agent.nodes as mod

        mock_provider = AsyncMock()
        mock_provider.chat_raw_stream = sequential_stream(
            raise_stream(RuntimeError("API down"))
        )
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        result = await mod.call_llm_node(
            AgentState(
                messages=[HumanMessage(content="hi")],
                final_response=None,
                final_prompt=None,
                error=None,
                pending_approval=None,
                step_count=None,
            ),
            tools=[],
        )
        stub = result["messages"][0]
        assert stub.additional_kwargs.get(mod._LLM_ERROR_MARKER) is True
        # The stub is dropped from the tool-selection history sent to the LLM.
        assert mod._messages_to_dicts([stub]) == []


class TestGenerateFinalNode:
    @pytest.mark.asyncio
    async def test_produces_final_prompt(self, monkeypatch) -> None:
        import agent.nodes as mod
        from tests._fake_llm import text_stream

        # The synthesis path calls the real provider; on CI the empty
        # LLM_API_KEY makes AsyncOpenAI raise at construction.  Feed it a
        # fake streaming provider so no LLM is touched.
        mock_provider = AsyncMock()
        mock_provider.chat_stream = text_stream("Final answer.")
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
        assert result["final_prompt"] is not None
        prompts = result["final_prompt"]
        assert any("EMA" in p.get("content", "") for p in prompts)
        # Should include context from the tool message
        context_block = [p for p in prompts if p["role"] == "system" and "Context" in p["content"]]
        assert len(context_block) == 1
        assert "Engineering Memory Agent" in context_block[0]["content"]

    @pytest.mark.asyncio
    async def test_streams_error_text_when_final_call_fails(self, monkeypatch) -> None:
        """Synthesis failure must stream the error text to the client.

        The final answer's tokens normally arrive via the custom stream; on
        failure the error text must be pushed the same way or the assistant
        message renders empty.
        """
        from tests._fake_llm import raise_stream, sequential_stream

        mock_provider = AsyncMock()
        mock_provider.chat_stream = sequential_stream(
            raise_stream(RuntimeError("synthesis down"))
        )

        import agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        emitted: list[dict[str, str]] = []
        monkeypatch.setattr(mod, "_stream_writer", lambda: emitted.append)

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

        assert "生成回复时出现错误" in result["final_response"]
        assert any(
            e.get("type") == "token" and "生成回复时出现错误" in e.get("content", "")
            for e in emitted
        )
        # The exception text stays server-side.
        assert "synthesis down" not in result["final_response"]

    @pytest.mark.asyncio
    async def test_harvest_parses_envelope_before_truncating(self, monkeypatch) -> None:
        """A large JSON-envelope tool result still yields its clean display text.

        Regression: the harvest loop used to truncate the raw tool content to
        800 chars *before* json.loads.  Search tools return a
        ``{"display", "sources"}`` envelope whose ``sources`` alone exceeds
        the cap, so truncate-first corrupted the JSON and the LLM received
        truncated raw JSON instead of the display text — exactly the large
        results the truncation exists for.
        """
        import json

        from tests._fake_llm import text_stream

        import agent.nodes as mod

        mock_provider = AsyncMock()
        mock_provider.chat_stream = text_stream("Final.")
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        envelope = json.dumps(
            {
                "display": "CLEAN-DISPLAY-TEXT",
                "sources": [{"snippet": "y" * 900}],  # pushes raw over the cap
            },
            ensure_ascii=False,
        )
        assert len(envelope) > mod._MAX_TOOL_CONTENT_CHARS

        state = _make_state(
            messages=[
                HumanMessage(content="ask"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "c1", "name": "search_memories_tool",
                         "args": {"query": "x"}, "type": "tool_call"}
                    ],
                ),
                ToolMessage(content=envelope, tool_call_id="c1"),
            ],
        )
        result = await mod.generate_final_node(state)
        system = next(p for p in result["final_prompt"] if p["role"] == "system")
        assert "CLEAN-DISPLAY-TEXT" in system["content"]
        # Neither the truncation marker nor raw truncated JSON reached the LLM.
        assert "…[truncated]" not in system["content"]
        assert "snippet" not in system["content"]

    @pytest.mark.asyncio
    async def test_harvest_windows_tool_results(self, monkeypatch) -> None:
        """The synthesis prompt harvests only the recent window of ToolMessages.

        Regression: the harvest loop walked the *full* message history even
        though the role loop below was windowed, so a long thread resubmitted
        every tool result from every turn into each synthesis prompt.
        """
        from tests._fake_llm import text_stream

        import agent.nodes as mod

        mock_provider = AsyncMock()
        mock_provider.chat_stream = text_stream("Final.")
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        def _tool_msg(tool_call_id: str, content: str) -> ToolMessage:
            return ToolMessage(content=content, tool_call_id=tool_call_id)

        _disable_compaction(monkeypatch)  # assert pure windowing, not summary
        # A small token budget pushes the old turn's tool result out of the
        # window while keeping the current turn's.
        _set_budget(monkeypatch, 20)

        messages: list = []
        # An old turn's tool result, then enough filler to push it out of the
        # token-budget window, then the current turn's tool result.
        messages += [
            AIMessage(content="", tool_calls=[
                {"id": "old", "name": "search_memories_tool",
                 "args": {"query": "x"}, "type": "tool_call"}]),
            _tool_msg("old", "OLD-MEMORY-RESULT"),
        ]
        messages += [HumanMessage(content=f"filler {i}") for i in range(14)]
        messages += [
            HumanMessage(content="current question"),
            AIMessage(content="", tool_calls=[
                {"id": "new", "name": "search_memories_tool",
                 "args": {"query": "y"}, "type": "tool_call"}]),
            _tool_msg("new", "RECENT-MEMORY-RESULT"),
        ]

        result = await mod.generate_final_node(_make_state(messages=messages))
        system = next(p for p in result["final_prompt"] if p["role"] == "system")
        assert "RECENT-MEMORY-RESULT" in system["content"]
        assert "OLD-MEMORY-RESULT" not in system["content"]

    @pytest.mark.asyncio
    async def test_harvest_includes_chunk_retrieval_results(self, monkeypatch) -> None:
        """A retrieve_chunks_tool result must reach the synthesis prompt.

        Regression: generate_final_node skipped retrieve_chunks_tool results
        while folding in the identical query_rewrite_and_search_tool, so a
        turn whose last tool call was a chunk search synthesized its answer
        without the chunks it had retrieved.
        """
        import agent.nodes as mod
        from tests._fake_llm import text_stream

        from agent.tool_envelope import build_tool_envelope

        mock_provider = AsyncMock()
        mock_provider.chat_stream = text_stream("Final.")
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)
        _disable_compaction(monkeypatch)

        state = _make_state(
            messages=[
                HumanMessage(content="chunk question"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "c1", "name": "retrieve_chunks_tool",
                         "args": {"query": "x"}, "type": "tool_call"}
                    ],
                ),
                ToolMessage(
                    content=build_tool_envelope(
                        "CHUNK-RETRIEVED-CONTENT", [{"document_id": "doc.md"}]
                    ),
                    tool_call_id="c1",
                    name="retrieve_chunks_tool",
                ),
            ],
        )

        result = await mod.generate_final_node(state)
        system = next(p for p in result["final_prompt"] if p["role"] == "system")
        assert "CHUNK-RETRIEVED-CONTENT" in system["content"]
        # Tagged as doc-source data, consistent with _CONTEXT_DOC_TOOLS.
        assert 'source="retrieve_chunks_tool"' in system["content"]

    @pytest.mark.asyncio
    async def test_no_tools_produces_prompt(self, monkeypatch) -> None:
        """Without any tool results, still produces a valid prompt (no context block)."""
        import agent.nodes as mod
        from tests._fake_llm import text_stream

        # Same fake-provider requirement as test_produces_final_prompt: the
        # empty LLM_API_KEY on CI would make AsyncOpenAI raise.
        mock_provider = AsyncMock()
        mock_provider.chat_stream = text_stream("Final answer.")
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

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
        from tests._fake_llm import text_stream

        import agent.nodes as mod

        mock_provider = AsyncMock()
        mock_provider.chat_stream = text_stream("Synthesized from tool output.")
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
        from tests._fake_llm import sequential_stream, tool_call_stream

        tools = [_make_write_tool()]

        mock_provider = AsyncMock()
        mock_provider.chat_raw_stream = sequential_stream(
            tool_call_stream([
                {
                    "id": "call_1",
                    "name": "write_memory_tool",
                    "args": {"content": "test"},
                }
            ])
        )

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
        from tests._fake_llm import sequential_stream, tool_call_stream

        tools = [_make_ingest_tool()]

        mock_provider = AsyncMock()
        mock_provider.chat_raw_stream = sequential_stream(
            tool_call_stream([
                {
                    "id": "call_2",
                    "name": "ingest_git_repo_tool",
                    "args": {"repo_path": "/tmp/repo", "max_commits": 10},
                }
            ])
        )

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

    @pytest.mark.asyncio
    async def test_batch_partial_approval_executes_only_approved(self, monkeypatch) -> None:
        """Partial batch approval runs the approved subset, rejecting the rest.

        Two sensitive calls to the SAME tool plus one safe call: approving only
        the first write must route to ToolNode with exactly that call (matched
        by id, not name) plus the safe call, and inject one [REJECTED]
        ToolMessage for the second write — not a fake [APPROVED] for it.
        """
        from langgraph.types import Command

        from agent.nodes import check_approval_node

        state = _make_state(
            messages=[
                HumanMessage(content="remember A and B"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "call_a", "name": "write_memory_tool",
                         "args": {"content": "A"}, "type": "tool_call"},
                        {"id": "call_b", "name": "write_memory_tool",
                         "args": {"content": "B"}, "type": "tool_call"},
                        {"id": "call_c", "name": "search_memories_tool",
                         "args": {"query": "x"}, "type": "tool_call"},
                    ],
                ),
            ],
        )

        import agent.nodes as mod
        monkeypatch.setattr(
            mod,
            "interrupt",
            lambda payload: {
                "calls": [
                    {"id": "call_a", "tool_name": "write_memory_tool", "approved": True},
                    {"id": "call_b", "tool_name": "write_memory_tool", "approved": False},
                ]
            },
        )

        result = await check_approval_node(state)
        assert isinstance(result, Command)
        assert result.goto == "tools"

        msgs = result.update["messages"]
        rejected = [m for m in msgs if isinstance(m, ToolMessage)]
        assert len(rejected) == 1
        assert rejected[0].tool_call_id == "call_b"
        assert "[REJECTED]" in rejected[0].content

        exec_ai = next(m for m in msgs if isinstance(m, AIMessage))
        # Approved sensitive call (call_a) plus the safe call (call_c);
        # call_b must NOT be executed.
        assert [tc["id"] for tc in exec_ai.tool_calls] == ["call_a", "call_c"]

    @pytest.mark.asyncio
    async def test_batch_all_rejected_injects_rejection_per_call(self, monkeypatch) -> None:
        """Rejecting every call in a batch injects a [REJECTED] ToolMessage per
        sensitive call, matched by id so same-named calls both get rejected."""
        from langgraph.types import Command

        from agent.nodes import check_approval_node

        state = _make_state(
            messages=[
                HumanMessage(content="remember A and B"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "call_a", "name": "write_memory_tool",
                         "args": {"content": "A"}, "type": "tool_call"},
                        {"id": "call_b", "name": "write_memory_tool",
                         "args": {"content": "B"}, "type": "tool_call"},
                    ],
                ),
            ],
        )

        import agent.nodes as mod
        monkeypatch.setattr(
            mod,
            "interrupt",
            lambda payload: {
                "calls": [
                    {"id": "call_a", "tool_name": "write_memory_tool", "approved": False},
                    {"id": "call_b", "tool_name": "write_memory_tool", "approved": False},
                ],
                "reason": "用户拒绝了工具调用。",
            },
        )

        result = await check_approval_node(state)
        assert isinstance(result, Command)
        assert result.goto == "call_llm"

        msgs = result.update["messages"]
        rejected = [m for m in msgs if isinstance(m, ToolMessage)]
        assert len(rejected) == 2
        assert {m.tool_call_id for m in rejected} == {"call_a", "call_b"}
        assert all("[REJECTED]" in m.content for m in rejected)


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
        from tests._fake_llm import sequential_stream, tool_call_stream

        from agent.graph import build_agent_graph
        from agent.tools import write_memory_tool

        mock_provider = AsyncMock()
        mock_provider.chat_raw_stream = sequential_stream(
            tool_call_stream([{
                "id": "call_int", "name": "write_memory_tool",
                "args": {"content": "EMA uses MySQL"},
            }])
        )

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

    @pytest.mark.asyncio
    async def test_ignores_conflict_from_previous_turn(self) -> None:
        """A previous turn's write-conflict result must not pause this turn.

        Regression: check_conflict scanned the whole history, so a conflict
        ToolMessage left over from an earlier turn (after an interrupt /
        rollback) triggered a spurious conflict pause on a turn that never
        wrote anything.
        """
        from langgraph.types import Command

        from agent.nodes import check_conflict_node

        state = _make_state(
            messages=[
                HumanMessage(content="remember X"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {"id": "old", "name": "write_memory_tool",
                         "args": {"content": "x"}, "type": "tool_call"}
                    ],
                ),
                ToolMessage(
                    content=self._json.dumps({
                        "action": "conflict",
                        "summary": "x",
                        "existing_id": "mem-1",
                        "existing_summary": "y",
                    }),
                    tool_call_id="old",
                    name="write_memory_tool",
                ),
                AIMessage(content="已处理"),
                HumanMessage(content="plain question"),
            ],
        )

        result = await check_conflict_node(state)
        assert isinstance(result, Command)
        assert result.goto == "call_llm"


class TestContextBounding:
    """Windowed history + tool-content truncation bound the LLM context."""

    def test_heuristic_tokens_cjk_and_ascii(self) -> None:
        # ASCII ≈ 4 chars/token.
        assert _heuristic_tokens("x" * 20) == 5
        assert _heuristic_tokens("short") == 2  # ceil(5/4)
        # CJK ≈ 1 token/char.
        assert _heuristic_tokens("中文") == 2
        assert _heuristic_tokens("") == 0

    def test_heuristic_tokens_weights_symbols(self) -> None:
        # Code/JSON punctuation is token-dense — weighted ~2 chars/token.
        assert _heuristic_tokens("{" * 20) == 10
        # A symbols-heavy string estimates higher than the same-length alnum.
        assert _heuristic_tokens("path/to:foo") > _heuristic_tokens("pathtofoo")
        # Symbols count heavier than whitespace of the same length.
        assert _heuristic_tokens("a!b!c!d!e") > _heuristic_tokens("a b c d e")

    def test_estimate_tokens_delegates_to_tokenizer(self, monkeypatch) -> None:
        """With a tokenizer available, _estimate_tokens uses its real count."""
        import agent.nodes as mod

        class _FakeEncoder:
            def encode(self, text: str) -> list[int]:
                return list(range(11))  # 11 tokens

        monkeypatch.setattr(mod, "_get_tokenizer", lambda: _FakeEncoder())
        assert mod._estimate_tokens("anything") == 11

    def test_estimate_tokens_falls_back_to_heuristic(self) -> None:
        """Without a tokenizer (offline), _estimate_tokens equals the heuristic."""
        assert _estimate_tokens("x" * 20) == _heuristic_tokens("x" * 20) == 5
        assert _estimate_tokens("中文") == 2

    def test_window_keeps_most_recent_messages_by_budget(self) -> None:
        # Each message is ~5 tokens ("x"*20 → 5 ASCII tokens); a budget of
        # 25 keeps exactly the newest 5.
        messages = [HumanMessage(content="x" * 20) for _ in range(20)]
        windowed = _window_messages(messages, max_tokens=25)
        assert len(windowed) == 5
        # The oldest messages are dropped; the tail is kept (by reference).
        assert windowed[0] is messages[15]
        assert windowed[-1] is messages[-1]

    def test_window_pins_system_prompt(self) -> None:
        messages = [SystemMessage(content="SYS")] + [
            HumanMessage(content="x" * 20) for _ in range(20)
        ]
        windowed = _window_messages(messages, max_tokens=25)
        assert windowed[0].content == "SYS"
        assert len(windowed) == 6  # pinned system + 5 recent
        assert windowed[-1] is messages[-1]

    def test_window_by_token_not_message_count(self) -> None:
        # The same budget keeps more short messages than long ones — the
        # window is sized by tokens, not a fixed message count.
        short = _window_messages(
            [HumanMessage(content="hi") for _ in range(30)], max_tokens=30
        )
        long = _window_messages(
            [HumanMessage(content="x" * 120) for _ in range(30)], max_tokens=30
        )
        assert len(short) == 30  # 30 × 1 token fits
        assert len(long) == 1  # one 30-token message fits, the next doesn't

    def test_window_keeps_newest_message_when_oversized(self) -> None:
        messages = [HumanMessage(content="x" * 2000), HumanMessage(content="tail")]
        windowed = _window_messages(messages, max_tokens=10)
        assert len(windowed) == 1
        assert windowed[0].content == "tail"

    def test_window_passthrough_when_within_budget(self) -> None:
        messages = [HumanMessage(content="a"), HumanMessage(content="b")]
        assert _window_messages(messages) is messages

    def test_truncate_caps_long_tool_content(self) -> None:
        truncated = _truncate_tool_content("x" * 2000)
        assert truncated.endswith("…[truncated]")
        # 800 chars + "\n" + marker
        assert len(truncated) == 800 + 1 + len("…[truncated]")

    def test_truncate_leaves_short_content_untouched(self) -> None:
        assert _truncate_tool_content("short") == "short"

    def test_messages_to_dicts_truncates_tool_content(self) -> None:
        dicts = _messages_to_dicts([
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "search_memories_tool",
                     "args": {"query": "x"}, "type": "tool_call"}
                ],
            ),
            ToolMessage(content="y" * 5000, tool_call_id="c1"),
        ])
        tool_msg = dicts[1]
        assert "…[truncated]" in tool_msg["content"]
        assert len(tool_msg["content"]) == 800 + 1 + len("…[truncated]")

    def test_messages_to_dicts_unwraps_envelope_before_truncating(self) -> None:
        """A large envelope (sources alone exceed the cap) is sent to the
        model as its complete display text, never as a half-cut JSON blob.
        The ReAct loop re-reads the converted message as assistant history,
        so truncating the raw envelope would poison the next turn's context.
        The display here is short, so unwrapping means no truncation at all —
        the fix is that the JSON is gone and the clean text survived."""
        envelope = json.dumps(
            {
                "display": "这里是一段简短的检索摘要",
                "sources": [{"id": f"src-{i}", "summary": "x" * 200} for i in range(50)],
            },
            ensure_ascii=False,
        )
        assert len(envelope) > _MAX_TOOL_CONTENT_CHARS

        dicts = _messages_to_dicts([
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "search_memories_tool",
                     "args": {"query": "x"}, "type": "tool_call"}
                ],
            ),
            ToolMessage(content=envelope, tool_call_id="c1"),
        ])
        tool_msg = dicts[1]
        # The complete display text survived — no truncation marker, no JSON
        # braces, no sources content.
        assert tool_msg["content"] == "这里是一段简短的检索摘要"
        assert "src-" not in tool_msg["content"]
        assert "…[truncated]" not in tool_msg["content"]

    def test_messages_to_dicts_truncates_unwrapped_display(self) -> None:
        """When the display itself is oversized, truncation applies to the
        unwrapped text — the model still reads display content, never JSON."""
        envelope = json.dumps(
            {"display": "摘" * 2000, "sources": [{"id": "s1"}]},
            ensure_ascii=False,
        )
        dicts = _messages_to_dicts([
            AIMessage(
                content="",
                tool_calls=[
                    {"id": "c1", "name": "search_memories_tool",
                     "args": {"query": "x"}, "type": "tool_call"}
                ],
            ),
            ToolMessage(content=envelope, tool_call_id="c1"),
        ])
        tool_msg = dicts[1]
        # Truncated display text — no JSON braces, no "sources".
        assert "…[truncated]" in tool_msg["content"]
        assert '"display"' not in tool_msg["content"]
        assert "sources" not in tool_msg["content"]
        assert len(tool_msg["content"]) == 800 + 1 + len("…[truncated]")

    @pytest.mark.asyncio
    async def test_call_llm_sends_windowed_history(self, monkeypatch) -> None:
        """Long history is windowed (system + most recent 12) before the LLM."""
        from tests._fake_llm import content_stream, sequential_stream

        mock_provider = AsyncMock()
        mock_provider.chat_raw_stream = sequential_stream(content_stream("ok"))

        import agent.nodes as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        from agent.tools import ALL_TOOLS

        _disable_compaction(monkeypatch)  # assert pure windowing, not summary
        # Each "old message N" is ~4 estimated tokens; a 48-token budget keeps
        # exactly the newest 12.
        _set_budget(monkeypatch, 48)
        history = [HumanMessage(content=f"old message {i}") for i in range(30)]
        await mod.call_llm_node(_make_state(history), tools=ALL_TOOLS)

        sent = mock_provider.chat_raw_stream.call_args.kwargs["messages"]
        assert len(sent) == 1 + 12  # system prompt + windowed tail
        assert sent[0]["role"] == "system"
        assert sent[-1]["content"] == "old message 29"

    @pytest.mark.asyncio
    async def test_generate_final_windows_history(self, monkeypatch) -> None:
        """The synthesis prompt carries a windowed history, not all of it."""
        from tests._fake_llm import text_stream

        import agent.nodes as mod

        mock_provider = AsyncMock()
        mock_provider.chat_stream = text_stream("Final.")
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_provider)

        _disable_compaction(monkeypatch)  # assert pure windowing, not summary
        # Each "m{i}" is ~1 estimated token; a 12-token budget keeps 12.
        _set_budget(monkeypatch, 12)
        state = _make_state(
            messages=[
                HumanMessage(content="search"),
                AIMessage(
                    content="",
                    tool_calls=[{"id": "c1", "name": "s",
                                 "args": {}, "type": "tool_call"}],
                ),
                ToolMessage(content="result", tool_call_id="c1"),
            ]
            + [HumanMessage(content=f"m{i}") for i in range(28)],
        )

        result = await mod.generate_final_node(state)
        prompt = result["final_prompt"]
        history_msgs = [p for p in prompt if p["role"] in ("user", "assistant")]
        assert len(history_msgs) <= 12
        assert history_msgs[-1]["content"] == "m27"
