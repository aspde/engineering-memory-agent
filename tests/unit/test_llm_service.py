"""Unit tests for LLM service."""

from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.model.llm import LLMProvider
from backend.shared.metrics import (
    _extract_total_tokens,
    get_token_usage,
    pop_scenario,
    record_usage,
    reset_token_usage,
)


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


class TestTokenUsageAccounting:
    """Tests for the module-level token counters used in cost monitoring."""

    def setup_method(self) -> None:
        reset_token_usage()

    def teardown_method(self) -> None:
        reset_token_usage()

    def test_extract_total_tokens_openai_shape(self) -> None:
        """OpenAI CompletionUsage exposes ``total_tokens``."""
        usage = SimpleNamespace(total_tokens=1234, prompt_tokens=1000, completion_tokens=234)
        assert _extract_total_tokens(usage) == 1234

    def test_extract_total_tokens_anthropic_shape(self) -> None:
        """Anthropic Usage exposes ``input_tokens`` + ``output_tokens``."""
        usage = SimpleNamespace(input_tokens=800, output_tokens=200)
        assert _extract_total_tokens(usage) == 1000

    def test_extract_total_tokens_dict_shape(self) -> None:
        assert _extract_total_tokens({"total_tokens": 42}) == 42
        assert _extract_total_tokens({"input_tokens": 30, "output_tokens": 12}) == 42

    def test_extract_total_tokens_none(self) -> None:
        assert _extract_total_tokens(None) == 0

    def test_record_usage_accumulates_per_scenario(self) -> None:
        record_usage("agent_chat", SimpleNamespace(total_tokens=100))
        record_usage("agent_chat", SimpleNamespace(total_tokens=250))
        record_usage("conflict_detection", {"total_tokens": 50})

        snapshot = get_token_usage()
        assert snapshot["agent_chat"] == 350
        assert snapshot["conflict_detection"] == 50

    def test_record_usage_ignores_zero_and_none(self) -> None:
        record_usage("empty", None)
        record_usage("zero", SimpleNamespace(total_tokens=0))
        assert get_token_usage() == {}

    def test_reset_clears_all_counters(self) -> None:
        record_usage("agent_chat", SimpleNamespace(total_tokens=999))
        assert get_token_usage() == {"agent_chat": 999}
        reset_token_usage()
        assert get_token_usage() == {}

    def test_get_token_usage_returns_snapshot_copy(self) -> None:
        """Mutating the returned dict must not affect internal state."""
        record_usage("x", SimpleNamespace(total_tokens=10))
        snap = get_token_usage()
        snap["x"] = 99999
        snap["injected"] = 1
        assert get_token_usage() == {"x": 10}


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
            input = {"result": [{"from": "a", "to": "b", "type": "depends_on"}]}

        class _FakeMessages:
            async def create(self, **kwargs):  # type: ignore[no-untyped-def]
                captured.update(kwargs)
                resp = MagicMock()
                resp.content = [_ToolUseBlock()]
                resp.usage = None
                return resp

        class _FakeAsyncAnthropic:
            def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
                self.messages = _FakeMessages()

        monkeypatch.setattr(anthropic, "AsyncAnthropic", _FakeAsyncAnthropic)

        provider = AnthropicProvider(api_key="test-key", model="claude-test")
        raw = await provider.chat_json(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
            json_schema={"type": "array"},
        )
        assert raw == '[{"from": "a", "to": "b", "type": "depends_on"}]'
        assert captured["tool_choice"] == {"type": "tool", "name": "emit_json"}
        # system message promoted to top-level param; envelope wraps the schema
        assert captured["system"] == "sys"
        assert captured["tools"][0]["input_schema"]["properties"]["result"] == {"type": "array"}  # type: ignore[index]


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
            input = {"query": "pgvector"}

        class _FakeMessages:
            async def create(self, **kwargs):  # type: ignore[no-untyped-def]
                captured.update(kwargs)
                resp = MagicMock()
                resp.content = [_ToolUseBlock()]
                resp.usage = None
                return resp

        class _FakeAsyncAnthropic:
            def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
                self.messages = _FakeMessages()

        monkeypatch.setattr(anthropic, "AsyncAnthropic", _FakeAsyncAnthropic)

        provider = AnthropicProvider(api_key="test-key", model="claude-test")

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
            def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
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
            def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
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

    def setup_method(self) -> None:
        reset_token_usage()

    def teardown_method(self) -> None:
        reset_token_usage()

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
        assert get_token_usage()["agent_chat"] == 42

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
        assert get_token_usage()["agent_chat"] == 55

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
        assert get_token_usage()["agent_final"] == 30

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
        assert get_token_usage() == {}

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
        assert get_token_usage()["agent_chat"] == 150

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
        assert get_token_usage() == {}
