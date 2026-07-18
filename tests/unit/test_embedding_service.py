"""Unit tests for embedding service."""

import pytest

from backend.model.embedding import EmbeddingProvider


class FakeEmbeddingProvider(EmbeddingProvider):
    """Stub provider returning fixed-size vectors."""

    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dimension for _ in texts]

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * self._dimension for _ in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


class TestFakeEmbeddingProvider:
    """Tests with stub — no model download needed."""

    @pytest.mark.asyncio
    async def test_embed_returns_correct_shape(self) -> None:
        provider = FakeEmbeddingProvider(dimension=8)
        texts = ["hello", "world"]
        result = await provider.embed(texts)
        assert len(result) == 2
        assert len(result[0]) == 8

    @pytest.mark.asyncio
    async def test_embed_empty_list(self) -> None:
        provider = FakeEmbeddingProvider()
        result = await provider.embed([])
        assert result == []

    def test_embed_sync_returns_correct_shape(self) -> None:
        provider = FakeEmbeddingProvider(dimension=4)
        result = provider.embed_sync(["a", "b", "c"])
        assert len(result) == 3
        assert all(len(v) == 4 for v in result)

    def test_dimension_property(self) -> None:
        provider = FakeEmbeddingProvider(dimension=256)
        assert provider.dimension == 256
