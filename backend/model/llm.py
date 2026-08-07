"""LLM provider abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any


class LLMStructuredError(RuntimeError):
    """Raised when a structured LLM call cannot produce schema-valid JSON.

    Raised by :func:`backend.service.structured.chat_structured` after
    bounded retries are exhausted.  Correctness-critical call sites
    propagate it (the write/operation fails); enrichment call sites catch
    it and degrade loudly (ERROR log + failure counter).
    """


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Send messages and return the response text."""
        ...

    @abstractmethod
    def chat_sync(self, messages: list[dict[str, str]], **kwargs) -> str:
        """Synchronous version of chat."""
        ...

    @abstractmethod
    async def chat_raw(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Send messages with optional tool definitions, return structured response.

        Returns:
            ``{"content": str, "tool_calls": [...] | None}`` where each
            tool_call is ``{"id": str, "name": str, "args": dict}``.
        """
        ...

    async def chat_raw_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream a raw response with optional tool definitions.

        Yields ``{"type": "content", "text": str}`` deltas as the model
        produces them, then a single trailing ``{"type": "tool_calls",
        "tool_calls": [...]}`` event when the turn ends with tool calls.

        Default implementation defers to one non-streaming
        :meth:`chat_raw` call and emits its result in one shot — real
        providers (OpenAI-compatible, Anthropic) override with true token
        streaming; lightweight fakes in tests rely on the default.
        """
        raw = await self.chat_raw(messages, tools=tools, **kwargs)
        if raw.get("content"):
            yield {"type": "content", "text": raw["content"]}
        if raw.get("tool_calls"):
            yield {"type": "tool_calls", "tool_calls": raw["tool_calls"]}

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream the response text token-by-token.

        Default implementation defers to one non-streaming :meth:`chat`
        call and yields the full text once; real providers override with
        true token streaming.
        """
        text = await self.chat(messages, **kwargs)
        yield text

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict[str, Any] | None = None,
        **kwargs,
    ) -> str:
        """Constrained chat: reply is guaranteed to be valid JSON text only.

        Providers implement this natively (``response_format`` for
        OpenAI-compatible APIs, forced ``tool_use`` for Anthropic).  Returns
        the raw JSON string; parsing, schema validation and retry live in
        :func:`backend.service.structured.chat_structured`.

        Non-abstract so lightweight fakes (``tests/unit/test_llm_service.py``)
        and ``AsyncMock`` providers in tests keep working without it.
        """
        raise NotImplementedError("chat_json not implemented by this provider")

    @property
    @abstractmethod
    def model(self) -> str:
        """Model name."""
        ...
