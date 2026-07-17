"""Tests for LLM service."""

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


@pytest.mark.integration
@pytest.mark.skipif(
    condition=True,
    reason="Set REAL_LLM_TEST=1 to run against the configured LLM provider.",
)
class TestRealLLMProvider:
    """Integration tests — requires valid API key."""

    @pytest.fixture(scope="class")
    def provider(self) -> LLMProvider:
        from backend.service.llm_service import get_llm_provider

        return get_llm_provider()

    @pytest.mark.asyncio
    async def test_chat_basic(self, provider: LLMProvider) -> None:
        result = await provider.chat(
            [{"role": "user", "content": "Say 'OK' in one word."}]
        )
        assert len(result) > 0
