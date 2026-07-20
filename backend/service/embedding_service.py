"""Embedding service with factory function."""

from __future__ import annotations

import asyncio
import logging
import os

from backend.model.embedding import EmbeddingProvider
from backend.shared.config import config

logger = logging.getLogger(__name__)


class BGEEmbeddingProvider(EmbeddingProvider):
    """BGE-M3 via sentence-transformers."""

    def __init__(
        self,
        model_name: str,
        normalize: bool = True,
        batch_size: int = 32,
        hf_endpoint: str = "https://hf-mirror.com",
    ) -> None:
        from sentence_transformers import SentenceTransformer

        # Force offline — the model is cached locally.  Both HF_HUB_OFFLINE
        # (huggingface_hub) and TRANSFORMERS_OFFLINE (transformers) must be
        # set; otherwise tokenizer loading calls hf.co/api/models/….
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

        logger.info("Loading embedding model: %s (offline)", model_name)
        self._model = SentenceTransformer(model_name, local_files_only=True)
        self._normalize = normalize
        self._batch_size = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = await asyncio.to_thread(
            self._model.encode,
            texts,
            normalize_embeddings=self._normalize,
            batch_size=self._batch_size,
        )
        return embeddings.tolist()

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=self._normalize,
            batch_size=self._batch_size,
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        # sentence-transformers renamed this method in v4; fall back for compat.
        if hasattr(self._model, "get_embedding_dimension"):
            return self._model.get_embedding_dimension()
        return self._model.get_sentence_embedding_dimension()


_provider: EmbeddingProvider | None = None


def get_embedding_provider() -> EmbeddingProvider:
    """Return a singleton embedding provider based on config."""
    global _provider
    if _provider is not None:
        return _provider

    if config.embedding.provider == "local":
        _provider = BGEEmbeddingProvider(
            model_name=config.embedding.model,
            normalize=config.embedding.normalize,
            batch_size=config.embedding.batch_size,
            hf_endpoint=config.embedding.hf_endpoint,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {config.embedding.provider}")

    return _provider
