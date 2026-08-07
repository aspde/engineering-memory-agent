"""Unit tests for embedding service."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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


# ── OpenAI embedding provider tests ───────────────────────────────────


def _make_embedding_response(embeddings: list[list[float]]) -> MagicMock:
    """Build a mock OpenAI embeddings response with the given vectors."""
    data = [
        MagicMock(embedding=vec) for vec in embeddings
    ]
    return MagicMock(data=data)


class TestOpenAIEmbeddingProvider:
    """Tests for OpenAIEmbeddingProvider — all SDK calls mocked."""

    def test_dimension_known_model(self) -> None:
        """Known models resolve to their documented dimensions."""
        from backend.service.embedding_service import OpenAIEmbeddingProvider

        p = OpenAIEmbeddingProvider(
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            model="text-embedding-3-small",
        )
        assert p.dimension == 1536

        p2 = OpenAIEmbeddingProvider(
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            model="text-embedding-3-large",
        )
        assert p2.dimension == 3072

        p3 = OpenAIEmbeddingProvider(
            api_key="sk-test",
            base_url="https://api.openai.com/v1",
            model="text-embedding-ada-002",
        )
        assert p3.dimension == 1536

    def test_dimension_unknown_model_warns_and_defaults(self, caplog) -> None:
        """Unknown models log a warning and fall back to 1536."""
        from backend.service.embedding_service import OpenAIEmbeddingProvider

        with caplog.at_level("WARNING"):
            p = OpenAIEmbeddingProvider(
                api_key="sk-test",
                base_url="",
                model="some-future-model",
            )
        assert p.dimension == 1536
        assert "Unknown dimension" in caplog.text

    @pytest.mark.asyncio
    async def test_embed_batches_correctly(self) -> None:
        """Multiple texts are split into batches and results concatenated."""
        from backend.service.embedding_service import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            api_key="sk-test",
            base_url="",
            model="text-embedding-3-small",
            batch_size=3,
        )

        # Build 5 texts → 2 batches (3 + 2).
        texts = ["a", "b", "c", "d", "e"]
        mock_batch_1 = _make_embedding_response([
            [1.0], [2.0], [3.0],
        ])
        mock_batch_2 = _make_embedding_response([
            [4.0], [5.0],
        ])
        provider._async_client.embeddings.create = AsyncMock(
            side_effect=[mock_batch_1, mock_batch_2],
        )

        result = await provider.embed(texts)
        assert len(result) == 5
        assert result == [[1.0], [2.0], [3.0], [4.0], [5.0]]
        assert provider._async_client.embeddings.create.call_count == 2

    def test_embed_sync_batches_correctly(self) -> None:
        """Sync path batch-splitting matches async."""
        from backend.service.embedding_service import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            api_key="sk-test",
            base_url="",
            model="text-embedding-3-small",
            batch_size=2,
        )

        texts = ["x", "y", "z"]  # 2 batches (2 + 1)
        provider._client.embeddings.create = MagicMock(
            side_effect=[
                _make_embedding_response([[0.1], [0.2]]),
                _make_embedding_response([[0.3]]),
            ],
        )

        result = provider.embed_sync(texts)
        assert result == [[0.1], [0.2], [0.3]]
        assert provider._client.embeddings.create.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_empty_list_returns_empty(self) -> None:
        from backend.service.embedding_service import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            api_key="sk-test",
            base_url="",
        )
        # Empty list — no batches, no API call.
        result = await provider.embed([])
        assert result == []

    def test_strips_trailing_slash_from_base_url(self) -> None:
        from backend.service.embedding_service import OpenAIEmbeddingProvider

        p = OpenAIEmbeddingProvider(
            api_key="sk-test",
            base_url="https://api.openai.com/v1/",
            model="text-embedding-3-small",
        )
        # openai SDK stores base_url as a URL object — compare strings.
        assert str(p._async_client.base_url).rstrip("/") == "https://api.openai.com/v1"

    def test_empty_base_url_defaults_to_openai(self) -> None:
        from backend.service.embedding_service import OpenAIEmbeddingProvider

        p = OpenAIEmbeddingProvider(
            api_key="sk-test",
            base_url="",
            model="text-embedding-3-small",
        )
        assert str(p._async_client.base_url).rstrip("/") == "https://api.openai.com/v1"

    def test_constructor_passes_timeout_to_sdk(self, monkeypatch) -> None:
        """Both SDK clients inherit the configured request timeout.

        Without this the openai SDK defaults to a 600s timeout, so a hung
        embedding API call would pin the request for ten minutes.
        """
        import openai

        from backend.service.embedding_service import OpenAIEmbeddingProvider

        mock_async = MagicMock()
        mock_sync = MagicMock()
        monkeypatch.setattr(openai, "AsyncOpenAI", mock_async)
        monkeypatch.setattr(openai, "OpenAI", mock_sync)

        OpenAIEmbeddingProvider(
            api_key="sk-test",
            base_url="",
            model="text-embedding-3-small",
            timeout=42,
        )
        assert mock_async.call_args.kwargs["timeout"] == 42
        assert mock_sync.call_args.kwargs["timeout"] == 42
