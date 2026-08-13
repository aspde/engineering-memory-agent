"""Integration tests for LLM service — requires valid API key."""

import os

import pytest

from backend.model.llm import LLMProvider


@pytest.mark.skipif(
    condition=os.getenv("REAL_LLM_TEST") != "1",
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
