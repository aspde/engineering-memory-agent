"""Unit tests for LLM service."""

import pytest

from backend.model.llm import LLMProvider


class FakeLLMProvider(LLMProvider):
    """Stub provider returning canned responses."""

    def __init__(self, model: str = "fake-model") -> None:
        self._model = model

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        return f"[{self._model}] echo: {messages[-1]['content']}"

    def chat_sync(self, messages: list[dict[str, str]], **kwargs) -> str:
        return f"[{self._model}] echo: {messages[-1]['content']}"

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
