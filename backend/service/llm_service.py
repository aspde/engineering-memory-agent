"""LLM service — factory for LLM providers."""

from __future__ import annotations

import logging

from openai import AsyncOpenAI, OpenAI

from backend.model.llm import LLMProvider
from backend.shared.config import config

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """Generic provider for any OpenAI-compatible API (DeepSeek, OpenAI, etc.)."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self._model = model
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._sync_client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info("LLM provider ready: %s @ %s", model, base_url)

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def chat_sync(self, messages: list[dict[str, str]], **kwargs) -> str:
        response = self._sync_client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )
        return response.choices[0].message.content or ""

    @property
    def model(self) -> str:
        return self._model


_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    """Return a singleton LLM provider based on config."""
    global _provider
    if _provider is not None:
        return _provider

    _provider = OpenAICompatibleProvider(
        api_key=config.llm.api_key,
        base_url=config.llm.base_url,
        model=config.llm.model,
    )
    return _provider
