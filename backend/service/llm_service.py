"""LLM service — factory for LLM providers."""

from __future__ import annotations

import logging

from backend.model.llm import LLMProvider
from backend.shared.config import config

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """Generic provider for any OpenAI-compatible API (DeepSeek, OpenAI, etc.)."""

    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        from openai import AsyncOpenAI, OpenAI

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


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic Claude API."""

    def __init__(self, api_key: str, model: str) -> None:
        from anthropic import Anthropic, AsyncAnthropic

        self._model = model
        self._async_client = AsyncAnthropic(api_key=api_key)
        self._sync_client = Anthropic(api_key=api_key)
        logger.info("Anthropic provider ready: %s", model)

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        system, user_messages = self._split_messages(messages)
        response = await self._async_client.messages.create(
            model=self._model,
            max_tokens=kwargs.pop("max_tokens", 4096),
            system=system,
            messages=user_messages,  # type: ignore[arg-type]
            **kwargs,
        )
        return self._extract_text(response.content)

    def chat_sync(self, messages: list[dict[str, str]], **kwargs) -> str:
        system, user_messages = self._split_messages(messages)
        response = self._sync_client.messages.create(
            model=self._model,
            max_tokens=kwargs.pop("max_tokens", 4096),
            system=system,
            messages=user_messages,  # type: ignore[arg-type]
            **kwargs,
        )
        return self._extract_text(response.content)

    @staticmethod
    def _split_messages(
        messages: list[dict[str, str]],
    ) -> tuple[str, list[dict[str, str]]]:
        """Anthropic requires system prompt as a top-level param, not a message."""
        system = ""
        if messages and messages[0].get("role") == "system":
            system = messages[0]["content"]
            messages = messages[1:]
        return system, messages

    @staticmethod
    def _extract_text(content: list) -> str:
        """Extract text from Anthropic content blocks."""
        parts: list[str] = []
        for block in content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    @property
    def model(self) -> str:
        return self._model


_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    """Return a singleton LLM provider based on config."""
    global _provider
    if _provider is not None:
        return _provider

    provider_name = config.llm.provider

    if provider_name == "anthropic":
        _provider = AnthropicProvider(
            api_key=config.llm.api_key,
            model=config.llm.model,
        )
    else:
        _provider = OpenAICompatibleProvider(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            model=config.llm.model,
        )
    return _provider
