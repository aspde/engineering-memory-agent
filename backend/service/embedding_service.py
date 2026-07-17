"""Embedding service with factory function."""

from __future__ import annotations

import asyncio
import logging

from backend.model.embedding import EmbeddingProvider
from backend.shared.config import config

logger = logging.getLogger(__name__)


class BGEEmbeddingProvider(EmbeddingProvider):
    """BGE-M3 via sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", model_name)
        self._model = SentenceTransformer(model_name)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = await asyncio.to_thread(
            self._model.encode, texts, normalize_embeddings=True
        )
        return embeddings.tolist()

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension()


_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    """Return a singleton embedding provider based on config."""
    global _provider
    if _provider is not None:
        return _provider

    if config.embedding.provider == "local":
        _provider = BGEEmbeddingProvider(config.embedding.model)
    else:
        raise ValueError(f"Unsupported embedding provider: {config.embedding.provider}")

    return _provider
