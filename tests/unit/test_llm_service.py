"""Unit tests for LLM service."""

from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.model.llm import LLMProvider
from backend.service.usage import pending_rows
from backend.shared.metrics import pop_scenario
from tests.support.process_state import reset_circuit_breakers, reset_usage_buffer


def _total_recorded_tokens() -> int:
    """Sum ``total_tokens`` across buffered usage observations.

    The in-memory per-scenario counters behind ``/api/agent/usage`` were
    removed as redundant; the usage recording buffer (drained into
    ``llm_usage``) is now the single recording path, so token assertions in
    this module read it.
    """
    return sum(r.get("total_tokens") or 0 for r in pending_rows())


@pytest.fixture(autouse=True)
def _clean_usage_buffer() -> None:
    """Isolate the usage recording buffer between tests (mirrors test_usage.py)."""
    reset_usage_buffer()
    yield
    reset_usage_buffer()


class FakeLLMProvider(LLMProvider):
    """Stub provider returning canned responses."""

    def __init__(self, model: str = "fake-model") -> None:
        self._model = model

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        return f"[{self._model}] echo: {messages[-1]['content']}"

    def chat_sync(self, messages: list[dict[str, str]], **kwargs) -> str:
        return f"[{self._model}] echo: {messages[-1]['content']}"

    async def chat_raw(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, object]] | None = None,
        **kwargs,
    ) -> dict[str, object]:
        """Return canned response; if tools are provided, simulate a tool call."""
        result: dict[str, object] = {"content": ""}
        if tools:
            result["content"] = f"[{self._model}] tool call"
            result["tool_calls"] = [
                {
                    "id": "call_fake_1",
                    "name": tools[0]["function"]["name"],  # type: ignore[index]
                    "args": {"query": messages[-1]["content"]},
                }
            ]
        else:
            result["content"] = f"[{self._model}] echo: {messages[-1]['content']}"
        return result

    @property
    def model(self) -> str:
        return self._model


class TestFakeLLMProvider:
    """Unit tests with stub — no API key needed."""

    @pytest.mark.asyncio
    async def test_chat_returns_string(self) -> None:
        provider = FakeLLMProvider()
        result = await provider.chat([{"role": "user", "content": "hello"}])
        assert isinstance(result, str)
        assert "hello" in result

    def test_chat_sync_returns_string(self) -> None:
        provider = FakeLLMProvider(model="test")
        result = provider.chat_sync([{"role": "user", "content": "world"}])
        assert "test" in result
        assert "world" in result

    def test_model_property(self) -> None:
        provider = FakeLLMProvider(model="gpt-99")
        assert provider.model == "gpt-99"


# ── Token usage accounting ────────────────────────────────────────────


class TestPopScenario:
    """The ``scenario`` kwarg must be popped before sending to the SDK."""

    def test_pops_existing_scenario(self) -> None:
        kwargs: dict = {"scenario": "agent_chat", "temperature": 0.5}
        assert pop_scenario(kwargs) == "agent_chat"
        assert "scenario" not in kwargs
        assert kwargs == {"temperature": 0.5}

    def test_defaults_to_default_when_missing(self) -> None:
        kwargs: dict = {"temperature": 0.5}
        assert pop_scenario(kwargs) == "default"

    def test_defaults_for_empty_string(self) -> None:
        kwargs: dict = {"scenario": "", "temperature": 0.5}
        assert pop_scenario(kwargs) == "default"

    def test_defaults_for_non_string(self) -> None:
        kwargs: dict = {"scenario": None, "temperature": 0.5}
        assert pop_scenario(kwargs) == "default"


# ── chat_json — structured-output constraint per provider ─────────────


class TestOpenAICompatibleChatJson:
    """OpenAI-compatible provider must pass response_format=json_object
    and return the raw JSON content string."""

    @pytest.mark.asyncio
    async def test_passes_response_format_and_returns_content(self) -> None:
        from backend.service.llm_service import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key="test-key", base_url="https://example.com/v1", model="test-model"
        )
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content='{"ok": true}'))]
        resp.usage = None
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = resp
        provider._async_client = mock_client  # type: ignore[assignment]

        raw = await provider.chat_json(
            [{"role": "user", "content": "hi"}], json_schema={"type": "object"}
        )
        assert raw == '{"ok": true}'
        kwargs = mock_client.chat.completions.create.await_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_empty_content_falls_back_without_response_format(self) -> None:
        """An OpenAI-compatible proxy that returns empty content under
        response_format (a lazy implementation, seen on some gateways) must
        be retried once without the constraint — empty output is a dead end
        for the schema validator downstream."""
        from backend.service.llm_service import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key="test-key", base_url="https://example.com/v1", model="test-model"
        )
        calls: list[dict] = []

        def _create(**kwargs):
            calls.append(kwargs)
            resp = MagicMock()
            resp.usage = None
            if len(calls) == 1:
                # First attempt (response_format) → empty content.
                resp.choices = [MagicMock(message=MagicMock(content=""))]
            else:
                resp.choices = [MagicMock(message=MagicMock(content='{"ok": true}'))]
            return resp

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = _create
        provider._async_client = mock_client  # type: ignore[assignment]

        raw = await provider.chat_json(
            [{"role": "user", "content": "hi"}], json_schema={"type": "object"}
        )
        assert raw == '{"ok": true}'
        assert len(calls) == 2
        # First call carries the constraint; the fallback drops it.
        assert calls[0]["response_format"] == {"type": "json_object"}
        assert "response_format" not in calls[1]

    @pytest.mark.asyncio
    async def test_populated_malformed_content_is_not_retried(self) -> None:
        """Only *empty* content triggers the fallback — a populated (even if
        unparseable) response goes back to the caller for the validator to
        reject, exactly once."""
        from backend.service.llm_service import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key="test-key", base_url="https://example.com/v1", model="test-model"
        )
        resp = MagicMock()
        resp.usage = None
        resp.choices = [MagicMock(message=MagicMock(content="not json at all"))]
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = resp
        provider._async_client = mock_client  # type: ignore[assignment]

        raw = await provider.chat_json(
            [{"role": "user", "content": "hi"}], json_schema={"type": "object"}
        )
        assert raw == "not json at all"
        assert mock_client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_empty_choices_raises_instead_of_index_error(self) -> None:
        """A provider response with an empty ``choices`` list must be handled
        as a recorded error, not crash with an unrecorded IndexError that
        skips usage/error accounting."""
        from backend.service.llm_service import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key="test-key", base_url="https://example.com/v1", model="test-model"
        )
        resp = MagicMock()
        resp.usage = None
        resp.choices = []
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = resp
        provider._async_client = mock_client  # type: ignore[assignment]

        with pytest.raises(RuntimeError):
            await provider.chat_json(
                [{"role": "user", "content": "hi"}], json_schema={"type": "object"}
            )
        assert mock_client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_empty_content_fallback_records_first_response_usage(self) -> None:
        """When the empty response triggers the no-response_format fallback,
        the first (empty) response's token usage must be recorded too — it
        consumed tokens even though its content was empty."""
        from backend.service.llm_service import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key="test-key", base_url="https://example.com/v1", model="test-model"
        )
        calls: list[dict] = []

        def _create(**kwargs):
            calls.append(kwargs)
            resp = MagicMock()
            if len(calls) == 1:
                # First attempt (response_format) → empty content, but it
                # still reported usage.
                resp.usage = SimpleNamespace(total_tokens=10)
                resp.choices = [MagicMock(message=MagicMock(content=""))]
            else:
                resp.usage = SimpleNamespace(total_tokens=20)
                resp.choices = [MagicMock(message=MagicMock(content='{"ok": true}'))]
            return resp

        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = _create
        provider._async_client = mock_client  # type: ignore[assignment]

        raw = await provider.chat_json(
            [{"role": "user", "content": "hi"}],
            json_schema={"type": "object"},
            scenario="agent_chat",
        )
        assert raw == '{"ok": true}'
        assert len(calls) == 2
        # Both the empty first response (10) and the fallback (20) are counted.
        assert _total_recorded_tokens() == 30


class TestOpenAICompatibleChatRaw:
    """OpenAI-compatible chat_raw must surface tool_calls as
    {"id", "name", "args"} — and degrade malformed or empty arguments
    instead of crashing the call, mirroring the streaming path."""

    @staticmethod
    def _make_provider(messages_with_tool_calls: list):
        from backend.service.llm_service import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key="test-key", base_url="https://example.com/v1", model="test-model"
        )
        msg = MagicMock()
        msg.content = ""
        msg.tool_calls = messages_with_tool_calls
        resp = MagicMock()
        resp.choices = [MagicMock(message=msg)]
        resp.usage = None
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = resp
        provider._async_client = mock_client  # type: ignore[assignment]
        return provider

    @pytest.mark.asyncio
    async def test_parses_valid_tool_arguments(self) -> None:
        tc = MagicMock()
        tc.id = "call_1"
        tc.function.name = "search_memories_tool"
        tc.function.arguments = '{"query": "pgvector"}'
        provider = self._make_provider([tc])

        result = await provider.chat_raw(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "search_memories_tool"}}],
        )
        assert result["tool_calls"] == [
            {"id": "call_1", "name": "search_memories_tool", "args": {"query": "pgvector"}}
        ]

    @pytest.mark.asyncio
    async def test_malformed_arguments_degrades_to_raw(self) -> None:
        tc = MagicMock()
        tc.id = "call_2"
        tc.function.name = "search_memories_tool"
        tc.function.arguments = "{not json"
        provider = self._make_provider([tc])

        result = await provider.chat_raw([{"role": "user", "content": "hi"}])
        assert result["tool_calls"] == [
            {"id": "call_2", "name": "search_memories_tool", "args": {"raw": "{not json"}}
        ]

    @pytest.mark.asyncio
    async def test_empty_arguments_become_empty_dict(self) -> None:
        tc = MagicMock()
        tc.id = "call_3"
        tc.function.name = "extract_memory_tool"
        tc.function.arguments = ""
        provider = self._make_provider([tc])

        result = await provider.chat_raw([{"role": "user", "content": "hi"}])
        assert result["tool_calls"] == [
            {"id": "call_3", "name": "extract_memory_tool", "args": {}}
        ]


class TestAnthropicChatJson:
    """Anthropic has no response_format; chat_json must drive a forced
    tool_use whose input_schema wraps the caller schema in an envelope,
    then unwrap it back to a plain JSON string."""

    @pytest.mark.asyncio
    async def test_forced_tool_and_envelope_unwrap(self, monkeypatch) -> None:
        import anthropic

        from backend.service.llm_service import AnthropicProvider

        captured: dict = {}

        class _ToolUseBlock:
            type = "tool_use"
            input: ClassVar[dict] = {"result": [{"from": "a", "to": "b", "type": "depends_on"}]}

        class _FakeMessages:
            async def create(self, **kwargs):  # type: ignore[no-untyped-def]
                captured.update(kwargs)
                resp = MagicMock()
                resp.content = [_ToolUseBlock()]
                resp.usage = None
                return resp

        class _FakeAsyncAnthropic:
            def __init__(self, *args, **kwargs) -> None:
                self.messages = _FakeMessages()

        monkeypatch.setattr(anthropic, "AsyncAnthropic", _FakeAsyncAnthropic)

        provider = AnthropicProvider(api_key="test-key", model="claude-test")
        raw = await provider.chat_json(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            json_schema={"type": "array"},
        )
        assert raw == '[{"from": "a", "to": "b", "type": "depends_on"}]'
        assert captured["tool_choice"] == {"type": "tool", "name": "emit_json"}
        # system promoted to a top-level block AND cached — chat_json now goes
        # through the same caching as chat_raw (fix: it previously sent the
        # bare string with no cache breakpoint).
        assert captured["system"] == [
            {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
        ]
        # messages converted to Anthropic shape (system pulled out, user kept)
        assert captured["messages"] == [{"role": "user", "content": "hi"}]
        # envelope still wraps the schema; the single emit_json tool is passed
        # through _to_anthropic_tools and carries the cache breakpoint
        assert [t["name"] for t in captured["tools"]] == ["emit_json"]
        assert captured["tools"][0]["input_schema"]["properties"]["result"] == {"type": "array"}  # type: ignore[index]
        assert captured["tools"][0]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_chat_json_converts_tool_result_messages(self, monkeypatch) -> None:
        """chat_json must translate OpenAI-format messages (assistant
        tool_calls + tool results) into Anthropic tool_use/tool_result shapes —
        a bare ``role: tool`` message would be rejected by the Anthropic API
        (fix: chat_json previously forwarded user_messages unconverted)."""
        import anthropic

        from backend.service.llm_service import AnthropicProvider

        captured: dict = {}

        class _ToolUseBlock:
            type = "tool_use"
            input: ClassVar[dict] = {"result": {"ok": True}}

        class _FakeMessages:
            async def create(self, **kwargs):  # type: ignore[no-untyped-def]
                captured.update(kwargs)
                resp = MagicMock()
                resp.content = [_ToolUseBlock()]
                resp.usage = None
                return resp

        class _FakeAsyncAnthropic:
            def __init__(self, *args, **kwargs) -> None:
                self.messages = _FakeMessages()

        monkeypatch.setattr(anthropic, "AsyncAnthropic", _FakeAsyncAnthropic)

        # prompt_caching=False keeps this test focused on message conversion.
        provider = AnthropicProvider(
            api_key="test-key", model="claude-test", prompt_caching=False
        )
        messages: list[dict[str, object]] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "summarize"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_memories_tool",
                            "arguments": '{"query": "pgvector"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "Found 2", "tool_call_id": "call_1"},
        ]

        raw = await provider.chat_json(messages, json_schema={"type": "object"})
        assert raw == '{"ok": true}'
        assert captured["messages"][0] == {"role": "user", "content": "summarize"}
        assert captured["messages"][1]["role"] == "assistant"
        assert captured["messages"][1]["content"][0] == {
            "type": "tool_use",
            "id": "call_1",
            "name": "search_memories_tool",
            "input": {"query": "pgvector"},
        }
        assert captured["messages"][2] == {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "Found 2"}
            ],
        }


class TestAnthropicChatRaw:
    """Anthropic has no OpenAI-style tool/function-calling format: chat_raw
    must translate OpenAI-format tools and messages into Anthropic's
    tool_use / tool_result shapes, and surface tool_use blocks back as
    OpenAI-style tool_calls."""

    @pytest.mark.asyncio
    async def test_translates_tools_and_tool_messages(self, monkeypatch) -> None:
        import anthropic

        from backend.service.llm_service import AnthropicProvider

        captured: dict = {}

        class _ToolUseBlock:
            type = "tool_use"
            id = "toolu_1"
            name = "search_memories_tool"
            input: ClassVar[dict] = {"query": "pgvector"}

        class _FakeMessages:
            async def create(self, **kwargs):  # type: ignore[no-untyped-def]
                captured.update(kwargs)
                resp = MagicMock()
                resp.content = [_ToolUseBlock()]
                resp.usage = None
                return resp

        class _FakeAsyncAnthropic:
            def __init__(self, *args, **kwargs) -> None:
                self.messages = _FakeMessages()

        monkeypatch.setattr(anthropic, "AsyncAnthropic", _FakeAsyncAnthropic)

        # prompt_caching=False keeps this test focused on the tool/message
        # translation; cache breakpoints are covered by TestAnthropicPromptCaching.
        provider = AnthropicProvider(
            api_key="test-key", model="claude-test", prompt_caching=False
        )

        openai_tools: list[dict[str, object]] = [
            {
                "type": "function",
                "function": {
                    "name": "search_memories_tool",
                    "description": "Search memories",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        messages: list[dict[str, object]] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_memories_tool",
                            "arguments": '{"query": "pgvector"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "content": "Found 2 relevant memories",
                "tool_call_id": "call_1",
            },
        ]

        result = await provider.chat_raw(messages, tools=openai_tools)

        # OpenAI schema → Anthropic input_schema
        assert captured["tools"] == [
            {
                "name": "search_memories_tool",
                "description": "Search memories",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            }
        ]
        # system promoted to top-level param; messages converted
        assert captured["system"] == "sys"
        assert captured["messages"][0] == {"role": "user", "content": "hi"}
        assert captured["messages"][1]["role"] == "assistant"
        assert captured["messages"][1]["content"][0] == {
            "type": "tool_use",
            "id": "call_1",
            "name": "search_memories_tool",
            "input": {"query": "pgvector"},
        }
        assert captured["messages"][2] == {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "Found 2 relevant memories",
                }
            ],
        }
        # tool_use block surfaced as OpenAI-style tool_calls for the agent
        assert result["content"] == ""
        assert result["tool_calls"] == [
            {"id": "toolu_1", "name": "search_memories_tool", "args": {"query": "pgvector"}}
        ]

    @pytest.mark.asyncio
    async def test_parallel_tool_results_coalesced_into_single_user_message(
        self, monkeypatch
    ) -> None:
        """Two tool results for one assistant turn (parallel tool calls) must
        be sent as a single user message with both tool_result blocks —
        Anthropic rejects consecutive same-role messages."""
        import anthropic

        from backend.service.llm_service import AnthropicProvider

        captured: dict = {}

        class _FakeMessages:
            async def create(self, **kwargs):  # type: ignore[no-untyped-def]
                captured.update(kwargs)
                resp = MagicMock()
                resp.content = []
                resp.usage = None
                return resp

        class _FakeAsyncAnthropic:
            def __init__(self, *args, **kwargs) -> None:
                self.messages = _FakeMessages()

        monkeypatch.setattr(anthropic, "AsyncAnthropic", _FakeAsyncAnthropic)

        provider = AnthropicProvider(api_key="test-key", model="claude-test")

        messages: list[dict[str, object]] = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search_memories_tool",
                            "arguments": '{"query": "a"}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "search_memories_tool",
                            "arguments": '{"query": "b"}',
                        },
                    },
                ],
            },
            {"role": "tool", "content": "Result A", "tool_call_id": "call_1"},
            {"role": "tool", "content": "Result B", "tool_call_id": "call_2"},
        ]

        await provider.chat_raw(messages)

        # user + assistant turn + one coalesced user result message
        assert len(captured["messages"]) == 3
        assert captured["messages"][2] == {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "Result A"},
                {"type": "tool_result", "tool_use_id": "call_2", "content": "Result B"},
            ],
        }

    @pytest.mark.asyncio
    async def test_plain_messages_without_tools(self, monkeypatch) -> None:
        import anthropic

        from backend.service.llm_service import AnthropicProvider

        captured: dict = {}

        class _FakeMessages:
            async def create(self, **kwargs):  # type: ignore[no-untyped-def]
                captured.update(kwargs)
                resp = MagicMock()
                resp.content = [SimpleNamespace(type="text", text="hello back")]
                resp.usage = None
                return resp

        class _FakeAsyncAnthropic:
            def __init__(self, *args, **kwargs) -> None:
                self.messages = _FakeMessages()

        monkeypatch.setattr(anthropic, "AsyncAnthropic", _FakeAsyncAnthropic)

        provider = AnthropicProvider(api_key="test-key", model="claude-test")
        result = await provider.chat_raw(
            [{"role": "user", "content": "hello"}]
        )
        assert result["content"] == "hello back"
        assert "tool_calls" not in result
        assert captured["messages"] == [{"role": "user", "content": "hello"}]
        assert "tools" not in captured

    def test_to_anthropic_tools_passes_through_anthropic_shape(self) -> None:
        from backend.service.llm_service import AnthropicProvider

        already = [
            {"name": "t", "description": "d", "input_schema": {"type": "object"}}
        ]
        assert AnthropicProvider._to_anthropic_tools(already) == already


class TestAnthropicPromptCaching:
    """Prompt caching — cache_control breakpoints on the system block and the
    last tool schema.  OpenAI-compatible providers cache server-side
    automatically, so this is an Anthropic-only mechanism."""

    @pytest.mark.asyncio
    async def test_system_block_and_last_tool_cached_by_default(
        self, monkeypatch
    ) -> None:
        import anthropic

        from backend.service.llm_service import AnthropicProvider

        captured: dict = {}

        class _FakeMessages:
            async def create(self, **kwargs):  # type: ignore[no-untyped-def]
                captured.update(kwargs)
                resp = MagicMock()
                resp.content = []
                resp.usage = None
                return resp

        class _FakeAsyncAnthropic:
            def __init__(self, *args, **kwargs) -> None:
                self.messages = _FakeMessages()

        monkeypatch.setattr(anthropic, "AsyncAnthropic", _FakeAsyncAnthropic)

        provider = AnthropicProvider(api_key="test-key", model="claude-test")

        openai_tools: list[dict[str, object]] = [
            {
                "type": "function",
                "function": {
                    "name": "search_memories_tool",
                    "description": "Search",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_memory_tool",
                    "description": "Write",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        await provider.chat_raw(
            [
                {"role": "system", "content": "You are EMA."},
                {"role": "user", "content": "hi"},
            ],
            tools=openai_tools,
        )

        # system promoted to a cached content block
        assert captured["system"] == [
            {
                "type": "text",
                "text": "You are EMA.",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        # cache breakpoint sits on the LAST tool only (Anthropic caches the
        # whole prefix up to the marker)
        assert captured["tools"][0].get("cache_control") is None
        assert captured["tools"][1]["cache_control"] == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_caching_disabled_keeps_bare_string_system(
        self, monkeypatch
    ) -> None:
        import anthropic

        from backend.service.llm_service import AnthropicProvider

        captured: dict = {}

        class _FakeMessages:
            async def create(self, **kwargs):  # type: ignore[no-untyped-def]
                captured.update(kwargs)
                resp = MagicMock()
                resp.content = [SimpleNamespace(type="text", text="ok")]
                resp.usage = None
                return resp

        class _FakeAsyncAnthropic:
            def __init__(self, *args, **kwargs) -> None:
                self.messages = _FakeMessages()

        monkeypatch.setattr(anthropic, "AsyncAnthropic", _FakeAsyncAnthropic)

        provider = AnthropicProvider(
            api_key="test-key", model="claude-test", prompt_caching=False
        )
        await provider.chat(
            [
                {"role": "system", "content": "You are EMA."},
                {"role": "user", "content": "hi"},
            ]
        )
        assert captured["system"] == "You are EMA."

    @pytest.mark.asyncio
    async def test_empty_system_stays_empty_string(self, monkeypatch) -> None:
        import anthropic

        from backend.service.llm_service import AnthropicProvider

        captured: dict = {}

        class _FakeMessages:
            async def create(self, **kwargs):  # type: ignore[no-untyped-def]
                captured.update(kwargs)
                resp = MagicMock()
                resp.content = [SimpleNamespace(type="text", text="ok")]
                resp.usage = None
                return resp

        class _FakeAsyncAnthropic:
            def __init__(self, *args, **kwargs) -> None:
                self.messages = _FakeMessages()

        monkeypatch.setattr(anthropic, "AsyncAnthropic", _FakeAsyncAnthropic)

        provider = AnthropicProvider(api_key="test-key", model="claude-test")
        await provider.chat([{"role": "user", "content": "hi"}])
        assert captured["system"] == ""


# ── Streaming token accounting ─────────────────────────────────────────


class _FakeAsyncStream:
    """Minimal async-iterable stand-in for an OpenAI ``AsyncStream``.

    Mirrors the SDK surface the provider reads after iteration: an optional
    ``usage`` attribute on the stream object (newer SDKs) and per-chunk
    ``usage`` on the final chunk.
    """

    def __init__(self, chunks: list, usage=None) -> None:
        self._chunks = list(chunks)
        self.usage = usage

    def __aiter__(self):
        return self._iter()

    async def _iter(self):
        for chunk in self._chunks:
            yield chunk


def _openai_chunk(text: str, usage=None):
    delta = MagicMock(content=text, tool_calls=None)
    chunk = MagicMock(choices=[MagicMock(delta=delta)])
    chunk.usage = usage
    return chunk


class TestStreamingUsage:
    """Streaming paths must record token usage per scenario so
    ``agent_chat`` / ``agent_final`` are not permanently zero."""

    @pytest.mark.asyncio
    async def test_openai_stream_object_usage_recorded(self) -> None:
        from backend.service.llm_service import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key="k", base_url="https://example.com/v1", model="m"
        )
        stream = _FakeAsyncStream(
            [_openai_chunk("hello", usage=None)],
            usage=SimpleNamespace(total_tokens=42),
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=stream)
        provider._async_client = mock_client  # type: ignore[assignment]

        events = [
            event
            async for event in provider.chat_raw_stream(
                [{"role": "user", "content": "hi"}], scenario="agent_chat"
            )
        ]

        assert events == [{"type": "content", "text": "hello"}]
        assert _total_recorded_tokens() == 42

    @pytest.mark.asyncio
    async def test_openai_final_chunk_usage_fallback(self) -> None:
        """SDKs that only carry usage on the last chunk still record it."""
        from backend.service.llm_service import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key="k", base_url="https://example.com/v1", model="m"
        )
        stream = _FakeAsyncStream(
            [
                _openai_chunk("Hel", usage=None),
                _openai_chunk("lo", usage=SimpleNamespace(total_tokens=55)),
            ],
            usage=None,  # no stream-level usage — rely on the final chunk
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=stream)
        provider._async_client = mock_client  # type: ignore[assignment]

        text = "".join(
            [
                event["text"]
                async for event in provider.chat_raw_stream(
                    [{"role": "user", "content": "hi"}], scenario="agent_chat"
                )
                if event.get("type") == "content"
            ]
        )

        assert text == "Hello"
        assert _total_recorded_tokens() == 55

    @pytest.mark.asyncio
    async def test_openai_chat_stream_records_usage(self) -> None:
        """``chat_stream`` (used by agent_final) delegates to the raw stream
        and therefore records usage too."""
        from backend.service.llm_service import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key="k", base_url="https://example.com/v1", model="m"
        )
        stream = _FakeAsyncStream(
            [_openai_chunk("done", usage=None)],
            usage=SimpleNamespace(total_tokens=30),
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=stream)
        provider._async_client = mock_client  # type: ignore[assignment]

        text = "".join(
            [
                token
                async for token in provider.chat_stream(
                    [{"role": "user", "content": "hi"}], scenario="agent_final"
                )
            ]
        )

        assert text == "done"
        assert _total_recorded_tokens() == 30

    @pytest.mark.asyncio
    async def test_openai_no_usage_is_silent(self) -> None:
        """A provider stream that reports no usage must not fail the stream."""
        from backend.service.llm_service import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key="k", base_url="https://example.com/v1", model="m"
        )
        stream = _FakeAsyncStream(
            [_openai_chunk("x", usage=None)], usage=None
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=stream)
        provider._async_client = mock_client  # type: ignore[assignment]

        events = [
            event
            async for event in provider.chat_raw_stream(
                [{"role": "user", "content": "hi"}], scenario="agent_chat"
            )
        ]

        assert events == [{"type": "content", "text": "x"}]
        assert _total_recorded_tokens() == 0

    @pytest.mark.asyncio
    async def test_anthropic_stream_usage_recorded(self) -> None:
        from backend.service.llm_service import AnthropicProvider

        provider = AnthropicProvider(api_key="k", model="claude-test")

        class _TextEvent:
            type = "content_block_delta"
            index = 0
            delta = SimpleNamespace(type="text_delta", text="hello")

        class _StopEvent:
            type = "message_stop"

        class _FakeStream:
            current_message_snapshot = SimpleNamespace(
                usage=SimpleNamespace(input_tokens=100, output_tokens=50)
            )

            def __aiter__(self):
                return self._iter()

            async def _iter(self):
                yield _TextEvent()
                yield _StopEvent()

        manager = MagicMock()
        manager.__aenter__ = AsyncMock(return_value=_FakeStream())
        manager.__aexit__ = AsyncMock(return_value=False)
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = manager
        provider._async_client = mock_client  # type: ignore[assignment]

        events = [
            event
            async for event in provider.chat_raw_stream(
                [{"role": "user", "content": "hi"}], scenario="agent_chat"
            )
        ]

        assert events == [{"type": "content", "text": "hello"}]
        assert _total_recorded_tokens() == 150

    @pytest.mark.asyncio
    async def test_anthropic_no_usage_is_silent(self) -> None:
        from backend.service.llm_service import AnthropicProvider

        provider = AnthropicProvider(api_key="k", model="claude-test")

        class _StopEvent:
            type = "message_stop"

        class _FakeStream:
            # snapshot that never materialised (raises like the SDK property)
            @property
            def current_message_snapshot(self):
                raise AssertionError("snapshot is None")

            def __aiter__(self):
                return self._iter()

            async def _iter(self):
                yield _StopEvent()

        manager = MagicMock()
        manager.__aenter__ = AsyncMock(return_value=_FakeStream())
        manager.__aexit__ = AsyncMock(return_value=False)
        mock_client = MagicMock()
        mock_client.messages.stream.return_value = manager
        provider._async_client = mock_client  # type: ignore[assignment]

        events = [
            event
            async for event in provider.chat_raw_stream(
                [{"role": "user", "content": "hi"}], scenario="agent_chat"
            )
        ]

        assert events == []
        assert _total_recorded_tokens() == 0


class TestStreamingBreaker:
    """The OpenAI stream path must record breaker success only once the stream
    is fully consumed — a stream that connects but dies mid-iteration is a
    breaker *failure*, not a success."""

    @pytest.fixture(autouse=True)
    def _isolate_breakers(self):
        """Fresh breaker registry per test — test_llm_service.py otherwise
        shares one module-global breaker per endpoint|model name, and the
        threshold/state from a neighbouring test would leak in."""
        reset_circuit_breakers()
        yield
        reset_circuit_breakers()

    @pytest.mark.asyncio
    async def test_mid_stream_failure_trips_breaker(self, monkeypatch) -> None:
        import httpx

        from backend.service.llm_service import OpenAICompatibleProvider
        from backend.shared import resilience
        from backend.shared.config import config

        monkeypatch.setattr(config.resilience, "circuit_breaker_threshold", 1)
        provider = OpenAICompatibleProvider(
            api_key="k", base_url="https://example.com/v1", model="m"
        )
        breaker = resilience.get_circuit_breaker(provider._breaker_name)

        class _BurstStream(_FakeAsyncStream):
            async def _iter(self):
                yield _openai_chunk("prefix", usage=None)
                raise httpx.ConnectError("connection lost mid-stream")

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_BurstStream([]))
        provider._async_client = mock_client  # type: ignore[assignment]

        collected: list = []
        with pytest.raises(httpx.ConnectError):
            async for event in provider.chat_raw_stream(
                [{"role": "user", "content": "hi"}], scenario="agent_chat"
            ):
                collected.append(event)

        # The prefix already delivered is preserved; the breaker learned that
        # this provider died mid-stream.
        assert collected == [{"type": "content", "text": "prefix"}]
        assert breaker.is_open

    @pytest.mark.asyncio
    async def test_successful_stream_keeps_breaker_closed(self) -> None:
        from backend.service.llm_service import OpenAICompatibleProvider
        from backend.shared import resilience

        provider = OpenAICompatibleProvider(
            api_key="k", base_url="https://example.com/v1", model="m"
        )
        breaker = resilience.get_circuit_breaker(provider._breaker_name)
        stream = _FakeAsyncStream(
            [_openai_chunk("hello", usage=None)],
            usage=SimpleNamespace(total_tokens=5),
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=stream)
        provider._async_client = mock_client  # type: ignore[assignment]

        events = [
            event
            async for event in provider.chat_raw_stream(
                [{"role": "user", "content": "hi"}], scenario="agent_chat"
            )
        ]

        assert events == [{"type": "content", "text": "hello"}]
        assert not breaker.is_open  # success recorded after full consumption


class TestSdkRetryDisabled:
    """The SDK clients must be built with max_retries=0 — resilience.py is the
    single retry owner, so a transient failure is never retried twice
    (SDK default 2 × tenacity max_attempts)."""

    def test_openai_clients_built_without_sdk_retry(self, monkeypatch) -> None:
        import openai

        from backend.service.llm_service import OpenAICompatibleProvider

        captured: dict = {}

        class _FakeAsyncOpenAI:
            def __init__(self, *args, **kwargs) -> None:
                captured["async"] = kwargs

        class _FakeOpenAI:
            def __init__(self, *args, **kwargs) -> None:
                captured["sync"] = kwargs

        monkeypatch.setattr(openai, "AsyncOpenAI", _FakeAsyncOpenAI)
        monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)

        OpenAICompatibleProvider(api_key="k", base_url="https://example.com/v1", model="m")
        assert captured["async"]["max_retries"] == 0
        assert captured["sync"]["max_retries"] == 0

    def test_anthropic_clients_built_without_sdk_retry(self, monkeypatch) -> None:
        import anthropic

        from backend.service.llm_service import AnthropicProvider

        captured: dict = {}

        class _FakeAsyncAnthropic:
            def __init__(self, *args, **kwargs) -> None:
                captured["async"] = kwargs

        class _FakeAnthropic:
            def __init__(self, *args, **kwargs) -> None:
                captured["sync"] = kwargs

        monkeypatch.setattr(anthropic, "AsyncAnthropic", _FakeAsyncAnthropic)
        monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)

        AnthropicProvider(api_key="k", model="claude-test")
        assert captured["async"]["max_retries"] == 0
        assert captured["sync"]["max_retries"] == 0


class TestAttemptAccounting:
    """Transient provider failures swallowed by tenacity must be visible in
    ``llm_usage`` accounting: the ``attempts`` column counts how many times the
    provider was actually hit before this row's outcome — a success after two
    retried 429s records attempts=3, never a silent 1."""

    class _RateLimit(Exception):
        status_code = 429

    @pytest.fixture(autouse=True)
    def _isolate_breakers(self) -> None:
        """Fresh breaker registry per test so the retried-failure counters in
        this class never leak into (or out of) other breaker tests."""
        reset_circuit_breakers()
        yield
        reset_circuit_breakers()

    @staticmethod
    def _make_openai_provider(monkeypatch: pytest.MonkeyPatch):
        """Zero-out tenacity's backoff so the retry tests run instantly."""
        from backend.service.llm_service import OpenAICompatibleProvider
        from backend.shared.config import config

        monkeypatch.setattr(config.resilience, "backoff_base", 0.0)
        monkeypatch.setattr(config.resilience, "backoff_max", 0.0)
        return OpenAICompatibleProvider(
            api_key="k", base_url="https://example.com/v1", model="test-model"
        )

    @pytest.mark.asyncio
    async def test_success_after_retry_records_attempts(self, monkeypatch) -> None:
        provider = self._make_openai_provider(monkeypatch)
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="hi back"))]
        resp.usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2)
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = [
            self._RateLimit("slow down"), resp
        ]
        provider._async_client = mock_client  # type: ignore[assignment]

        out = await provider.chat(
            [{"role": "user", "content": "hi"}], scenario="agent_chat"
        )
        assert out == "hi back"
        # one 429 swallowed by tenacity, then the real success
        assert mock_client.chat.completions.create.await_count == 2
        rows = pending_rows()
        assert len(rows) == 1
        assert rows[0]["status"] == "success"
        assert rows[0]["attempts"] == 2

    @pytest.mark.asyncio
    async def test_clean_success_records_attempts_one(self, monkeypatch) -> None:
        provider = self._make_openai_provider(monkeypatch)
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="hi back"))]
        resp.usage = None
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = resp
        provider._async_client = mock_client  # type: ignore[assignment]

        await provider.chat([{"role": "user", "content": "hi"}], scenario="agent_chat")
        rows = pending_rows()
        assert len(rows) == 1
        assert rows[0]["attempts"] == 1

    @pytest.mark.asyncio
    async def test_final_failure_records_total_attempts(self, monkeypatch) -> None:
        """A call that exhausts tenacity's retries records how many times the
        provider was actually hit before giving up (max_attempts=3 → 3)."""
        provider = self._make_openai_provider(monkeypatch)
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = self._RateLimit(
            "always 429"
        )
        provider._async_client = mock_client  # type: ignore[assignment]

        with pytest.raises(self._RateLimit):
            await provider.chat(
                [{"role": "user", "content": "hi"}], scenario="agent_chat"
            )
        assert mock_client.chat.completions.create.await_count == 3
        rows = pending_rows()
        assert len(rows) == 1
        assert rows[0]["status"] == "error"
        assert rows[0]["attempts"] == 3

    @pytest.mark.asyncio
    async def test_stream_connect_retry_records_attempts(self, monkeypatch) -> None:
        """The streaming path counts connection-establishment retries: a 429
        at connect time is retried before the stream exists, and the eventual
        success row must reflect both attempts."""
        provider = self._make_openai_provider(monkeypatch)
        stream = _FakeAsyncStream(
            [_openai_chunk("hello", usage=None)],
            usage=SimpleNamespace(total_tokens=42),
        )
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=[self._RateLimit("slow down"), stream]
        )
        provider._async_client = mock_client  # type: ignore[assignment]

        events = [
            event
            async for event in provider.chat_raw_stream(
                [{"role": "user", "content": "hi"}], scenario="agent_chat"
            )
        ]
        assert events == [{"type": "content", "text": "hello"}]
        rows = pending_rows()
        assert len(rows) == 1
        assert rows[0]["status"] == "success"
        assert rows[0]["attempts"] == 2


class TestGetJudgeProvider:
    """get_judge_provider — dedicated judge model vs primary fallback.

    The judge provider must be independent of the primary singleton so the
    eval's LLM-as-judge runs on a different model from the one evaluated.
    """

    def _reset_judge(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import backend.service.llm_service as ls

        monkeypatch.setattr(ls, "_judge_provider", None)

    def test_falls_back_to_primary_when_not_configured(self, monkeypatch) -> None:
        import backend.service.llm_service as ls

        self._reset_judge(monkeypatch)
        monkeypatch.setattr(ls.config.llm, "judge_provider", "")
        monkeypatch.setattr(ls.config.llm, "judge_model", "")
        primary = FakeLLMProvider("primary")
        monkeypatch.setattr(ls, "get_llm_provider", lambda: primary)

        assert ls.get_judge_provider() is primary

    def test_builds_independent_instance_when_configured(self, monkeypatch) -> None:
        import backend.service.llm_service as ls

        self._reset_judge(monkeypatch)
        monkeypatch.setattr(ls.config.llm, "judge_provider", "openai")
        monkeypatch.setattr(ls.config.llm, "judge_model", "glm-4.7-flash")
        monkeypatch.setattr(ls.config.llm, "judge_api_key", "k")
        monkeypatch.setattr(
            ls.config.llm,
            "judge_base_url",
            "https://open.bigmodel.cn/api/paas/v4",
        )
        judge = FakeLLMProvider("judge")
        seen: dict[str, str] = {}

        def _fake_build(provider, api_key, base_url, model, **kw):
            seen.update(
                provider=provider,
                api_key=api_key,
                base_url=base_url,
                model=model,
            )
            return judge

        monkeypatch.setattr(ls, "_build_provider", _fake_build)

        assert ls.get_judge_provider() is judge
        assert seen["provider"] == "openai"
        assert seen["model"] == "glm-4.7-flash"
        assert seen["api_key"] == "k"
        assert seen["base_url"] == "https://open.bigmodel.cn/api/paas/v4"

    def test_result_is_cached(self, monkeypatch) -> None:
        import backend.service.llm_service as ls

        self._reset_judge(monkeypatch)
        monkeypatch.setattr(ls.config.llm, "judge_provider", "openai")
        monkeypatch.setattr(ls.config.llm, "judge_model", "glm-4.7-flash")
        monkeypatch.setattr(ls.config.llm, "judge_api_key", "k")
        monkeypatch.setattr(ls.config.llm, "judge_base_url", "b")
        calls = {"n": 0}

        def _fake_build(*a, **kw):
            calls["n"] += 1
            return FakeLLMProvider("judge")

        monkeypatch.setattr(ls, "_build_provider", _fake_build)

        first = ls.get_judge_provider()
        second = ls.get_judge_provider()
        assert first is second
        assert calls["n"] == 1

    def test_fallback_result_is_cached(self, monkeypatch) -> None:
        """When no dedicated judge is configured the primary fallback is
        cached in ``_judge_provider`` too, so the fallback decision (and the
        primary singleton lookup) runs once, not on every judge call."""
        import backend.service.llm_service as ls

        self._reset_judge(monkeypatch)
        monkeypatch.setattr(ls.config.llm, "judge_provider", "")
        monkeypatch.setattr(ls.config.llm, "judge_model", "")
        primary = FakeLLMProvider("primary")
        calls = {"n": 0}

        def _fake_get():
            calls["n"] += 1
            return primary

        monkeypatch.setattr(ls, "get_llm_provider", _fake_get)

        first = ls.get_judge_provider()
        second = ls.get_judge_provider()
        assert first is primary
        assert second is primary
        assert calls["n"] == 1

    def test_judge_build_passes_prompt_caching(self, monkeypatch) -> None:
        """An Anthropic judge must be built with the configured
        prompt_caching_enabled, not silently inheriting the provider default
        True when PROMPT_CACHING_ENABLED=false."""
        import backend.service.llm_service as ls

        self._reset_judge(monkeypatch)
        monkeypatch.setattr(ls.config.llm, "judge_provider", "anthropic")
        monkeypatch.setattr(ls.config.llm, "judge_model", "claude-haiku")
        monkeypatch.setattr(ls.config.llm, "judge_api_key", "k")
        monkeypatch.setattr(ls.config.llm, "judge_base_url", "")
        monkeypatch.setattr(ls.config.llm, "prompt_caching_enabled", False)
        seen: dict = {}

        def _fake_build(*args, **kwargs):
            seen.update(kwargs)
            return FakeLLMProvider("judge")

        monkeypatch.setattr(ls, "_build_provider", _fake_build)

        ls.get_judge_provider()
        assert seen["prompt_caching"] is False
