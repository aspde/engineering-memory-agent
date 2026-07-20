"""LLM service — factory for LLM providers."""

from __future__ import annotations

import logging

from backend.model.llm import LLMProvider
from backend.shared.config import config

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """Generic provider for any OpenAI-compatible API (DeepSeek, OpenAI, etc.)."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 60,
    ) -> None:
        from openai import AsyncOpenAI, OpenAI

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._async_client = AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout
        )
        self._sync_client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout
        )
        logger.info("LLM provider ready: %s @ %s", model, base_url)

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        kwargs.setdefault("temperature", self._temperature)
        kwargs.setdefault("max_tokens", self._max_tokens)
        response = await self._async_client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            **kwargs,
        )
        return response.choices[0].message.content or ""

    async def chat_raw(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, object]] | None = None,
        **kwargs,
    ) -> dict[str, object]:
        kwargs.setdefault("temperature", self._temperature)
        kwargs.setdefault("max_tokens", self._max_tokens)
        create_kwargs: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            **kwargs,
        }
        if tools:
            create_kwargs["tools"] = tools

        response = await self._async_client.chat.completions.create(**create_kwargs)  # type: ignore[arg-type]
        msg = response.choices[0].message
        result: dict[str, object] = {"content": msg.content or ""}

        if msg.tool_calls:
            import json

            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                }
                for tc in msg.tool_calls
            ]
        return result

    def chat_sync(self, messages: list[dict[str, str]], **kwargs) -> str:
        kwargs.setdefault("temperature", self._temperature)
        kwargs.setdefault("max_tokens", self._max_tokens)
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

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        timeout: int = 60,
    ) -> None:
        from anthropic import Anthropic, AsyncAnthropic

        self._model = model
        self._max_tokens = max_tokens
        self._async_client = AsyncAnthropic(
            api_key=api_key, timeout=timeout
        )
        self._sync_client = Anthropic(api_key=api_key, timeout=timeout)
        logger.info("Anthropic provider ready: %s", model)

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        system, user_messages = self._split_messages(messages)
        kwargs.setdefault("max_tokens", self._max_tokens)
        response = await self._async_client.messages.create(
            model=self._model,
            system=system,
            messages=user_messages,  # type: ignore[arg-type]
            **kwargs,
        )
        return self._extract_text(response.content)

    async def chat_raw(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, object]] | None = None,
        **kwargs,
    ) -> dict[str, object]:
        system, user_messages = self._split_messages(messages)
        kwargs.setdefault("max_tokens", self._max_tokens)
        create_kwargs: dict[str, object] = {
            "model": self._model,
            "system": system,
            "messages": user_messages,
            **kwargs,
        }
        if tools:
            create_kwargs["tools"] = tools

        response = await self._async_client.messages.create(**create_kwargs)  # type: ignore[arg-type]
        content_blocks: list[object] = response.content  # type: ignore[assignment]
        result: dict[str, object] = {"content": self._extract_text(content_blocks)}

        tool_calls: list[dict[str, object]] = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "args": block.get("input", {}),
                    }
                )
            elif hasattr(block, "type") and getattr(block, "type", "") == "tool_use":
                tool_calls.append(
                    {
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "args": getattr(block, "input", {}),
                    }
                )
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    def chat_sync(self, messages: list[dict[str, str]], **kwargs) -> str:
        system, user_messages = self._split_messages(messages)
        kwargs.setdefault("max_tokens", self._max_tokens)
        response = self._sync_client.messages.create(
            model=self._model,
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

    if config.llm.provider == "anthropic":
        _provider = AnthropicProvider(
            api_key=config.llm.api_key,
            model=config.llm.model,
            max_tokens=config.llm.max_tokens,
            timeout=config.llm.timeout,
        )
    elif config.llm.provider in ("deepseek", "openai"):
        _provider = OpenAICompatibleProvider(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            timeout=config.llm.timeout,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm.provider}")
    return _provider
