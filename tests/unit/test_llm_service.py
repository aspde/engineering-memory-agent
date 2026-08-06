"""Unit tests for LLM service."""

from collections.abc import AsyncIterator
from types import SimpleNamespace

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
