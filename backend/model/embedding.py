"""Embedding provider abstract interface."""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract base for embedding providers."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode texts to vectors."""
        ...

    @abstractmethod
    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        """Encode texts synchronously."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Output vector dimension."""
        ...
