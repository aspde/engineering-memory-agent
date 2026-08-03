"""LLM provider abstract interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


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

    @property
    @abstractmethod
    def model(self) -> str:
        """Model name."""
        ...
