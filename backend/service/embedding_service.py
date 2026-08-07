"""Embedding service with factory function."""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any

# ── Force offline BEFORE any HF/transformers imports ──────────────────
# Must be at module top-level: when `from backend.service.embedding_service
# import ...` is executed (by pytest, agent_service eager init, or any
# caller), these are set before SentenceTransformer / AutoTokenizer sees
# the module for the first time.
for _k, _v in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}.items():
    os.environ[_k] = _v  # force-override — setdefault may leave stale values

from backend.model.embedding import EmbeddingProvider  # noqa: E402
from backend.shared.config import config  # noqa: E402
from backend.shared.resilience import (  # noqa: E402
    call_with_resilience,
    call_with_resilience_sync,
)

logger = logging.getLogger(__name__)

# ── Known dimensions for common embedding models ──────────────────────
# Used to resolve dimension without making an API call.
_KNOWN_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-m3": 1024,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


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

        os.environ.setdefault("HF_ENDPOINT", hf_endpoint)

        logger.info("Loading embedding model: %s (offline)", model_name)
        self._model = SentenceTransformer(model_name, local_files_only=True)
        self._normalize = normalize
        self._batch_size = batch_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        t0 = time.perf_counter()
        embeddings = await asyncio.to_thread(
            self._model.encode,
            texts,
            normalize_embeddings=self._normalize,
            batch_size=self._batch_size,
        )
        t1 = time.perf_counter()
        n = len(texts)
        logger.info(
            "BGE embed latency: total=%.0fms per_item=%.2fms count=%d batch_size=%d",
            (t1 - t0) * 1000, (t1 - t0) * 1000 / max(n, 1),
            n, self._batch_size,
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


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI / OpenAI-compatible embedding API (text-embedding-3-*, etc.).

    Uses the ``openai`` SDK for both async and sync paths.  Batch
    requests are sent in groups of *batch_size* to stay under provider
    rate limits.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        timeout: int = 60,
    ) -> None:
        from openai import AsyncOpenAI, OpenAI

        base = base_url.rstrip("/") if base_url else "https://api.openai.com/v1"
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base, timeout=timeout)
        self._client = OpenAI(api_key=api_key, base_url=base, timeout=timeout)
        self._model = model
        self._batch_size = batch_size

        # Resolve dimension from known models; warn + default for unknowns.
        self._dimension = _KNOWN_DIMENSIONS.get(model)
        if self._dimension is None:
            logger.warning(
                "Unknown dimension for model %r — defaulting to 1536. "
                "Add the model to _KNOWN_DIMENSIONS in %s.",
                model,
                __file__,
            )
            self._dimension = 1536

        logger.info(
            "OpenAI embedding provider ready: model=%r dimension=%d batch_size=%d",
            model,
            self._dimension,
            batch_size,
        )

    # ── async ─────────────────────────────────────────────────────────

    async def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        t0 = time.perf_counter()
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]

            async def _op(batch: list[str] = batch) -> Any:
                return await self._async_client.embeddings.create(
                    model=self._model,
                    input=batch,
                )

            response = await call_with_resilience("embedding:openai", _op)
            all_embeddings.extend(
                [d.embedding for d in response.data]
            )
        t1 = time.perf_counter()
        n = len(texts)
        logger.info(
            "OpenAI embed latency: total=%.0fms per_item=%.2fms count=%d model=%s",
            (t1 - t0) * 1000, (t1 - t0) * 1000 / max(n, 1), n, self._model,
        )
        return all_embeddings

    # ── sync ──────────────────────────────────────────────────────────

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]

            def _op(batch: list[str] = batch) -> Any:
                return self._client.embeddings.create(
                    model=self._model,
                    input=batch,
                )

            response = call_with_resilience_sync("embedding:openai", _op)
            all_embeddings.extend(
                [d.embedding for d in response.data]
            )
        return all_embeddings

    @property
    def dimension(self) -> int:
        return self._dimension


_provider: EmbeddingProvider | None = None
_lock: threading.Lock = threading.Lock()


def get_embedding_provider() -> EmbeddingProvider:
    """Return a singleton embedding provider based on config.

    Thread-safe: the lock prevents the background warmup and the first
    real request from racing to initialise the provider.
    """
    global _provider
    if _provider is not None:
        return _provider

    with _lock:
        if _provider is not None:  # double-check after acquiring lock
            return _provider

        provider_name = config.embedding.provider
        if provider_name == "local":
            _provider = BGEEmbeddingProvider(
                model_name=config.embedding.model,
                normalize=config.embedding.normalize,
                batch_size=config.embedding.batch_size,
                hf_endpoint=config.embedding.hf_endpoint,
            )
        elif provider_name == "openai":
            _provider = OpenAIEmbeddingProvider(
                api_key=config.embedding.api_key,
                base_url=config.embedding.base_url,
                model=config.embedding.model,
                batch_size=config.embedding.batch_size,
                timeout=config.embedding.timeout,
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {provider_name!r}")

        return _provider
