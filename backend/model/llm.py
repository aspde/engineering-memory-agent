"""LLM provider abstract interface."""

from abc import ABC, abstractmethod


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

    @property
    @abstractmethod
    def model(self) -> str:
        """Model name."""
        ...
