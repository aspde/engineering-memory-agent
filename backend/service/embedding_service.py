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
from backend.shared.config import EMBEDDING_DIMENSIONS, config  # noqa: E402
from backend.shared.resilience import (  # noqa: E402
    call_with_resilience,
    call_with_resilience_sync,
)

logger = logging.getLogger(__name__)

# ── Circuit-breaker naming ───────────────────────────────────────────
# Embedding breakers are keyed by ``base_url|model`` so a fallback provider
# never shares the primary's breaker (mirrors ``llm_service.breaker_name``).
# This is the single naming source: the provider computes its instance name
# from it, and ``embedding_breaker_name`` reads the configured primary through
# it, so a naming change lands in one place instead of being kept "in
# lockstep" across modules.


def breaker_name(base_url: str, model: str) -> str:
    """Circuit-breaker name for an OpenAI-compatible embedding endpoint.

    Keyed by ``base_url|model`` (e.g. ``embedding:https://api.openai.com/v1|
    text-embedding-3-small``) so the primary and its fallback each get their
    own breaker.  The base URL is normalised the same way the provider's
    ``__init__`` normalises it (trailing slash stripped, empty → the OpenAI
    default), keeping ``/health`` and the provider in lockstep.
    """
    base = base_url.rstrip("/") if base_url else "https://api.openai.com/v1"
    return f"embedding:{base}|{model}"


def embedding_breaker_name() -> str:
    """Circuit-breaker name for the configured primary embedding provider.

    Mirrors the per-instance names the provider classes compute in
    ``__init__`` (``base_url|model``) so ``/health`` can report the primary's
    breaker state without reaching into provider internals.  Only meaningful
    for the remote ``openai`` provider — the local BGE provider makes no
    remote calls and has no breaker.
    """
    return breaker_name(config.embedding.base_url, config.embedding.model)

# ── Dimension resolution ───────────────────────────────────────────────
# The model→dimension map lives in ``config.EmbeddingConfig`` (single source
# of truth); the schema and every provider read it from there.  OpenAI has no
# API to introspect its dimension; the local BGE provider reports the real
# dimension of the loaded model, and ``get_embedding_provider()`` warns when
# the two disagree (a wrong guess would otherwise fail every embedding write).


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
        self._base_url = base
        self._async_client = AsyncOpenAI(api_key=api_key, base_url=base, timeout=timeout)
        self._client = OpenAI(api_key=api_key, base_url=base, timeout=timeout)
        self._model = model
        self._batch_size = batch_size

        # Dimension follows this provider's own model, from the shared config
        # map (OpenAI has no API to introspect it).  Unknown models warn +
        # default to 1536.
        self._dimension = EMBEDDING_DIMENSIONS.get(model, 1536)
        if model not in EMBEDDING_DIMENSIONS:
            logger.warning(
                "Unknown dimension for model %r — defaulting to %d. "
                "Add the model to EMBEDDING_DIMENSIONS in %s.",
                model,
                self._dimension,
                config.__module__,
            )

        logger.info(
            "OpenAI embedding provider ready: model=%r dimension=%d batch_size=%d",
            model,
            self._dimension,
            batch_size,
        )

    def _breaker_name(self) -> str:
        """Per-endpoint circuit-breaker name — keyed by ``base_url|model``.

        A primary provider and its fallback must not share one breaker:
        tripping the primary (which is exactly when failover engages) would
        otherwise fast-fail the healthy fallback too (see the failover tests).
        Mirrors ``llm_service.breaker_name``.  The ``_base_url`` getattr
        fallback keeps the ``object.__new__`` construction in
        ``test_resilience`` (which sets ``_model`` only) working.
        """
        base = getattr(self, "_base_url", "")
        return breaker_name(base, self._model)

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

            response = await call_with_resilience(self._breaker_name(), _op)
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

            response = call_with_resilience_sync(self._breaker_name(), _op)
            all_embeddings.extend(
                [d.embedding for d in response.data]
            )
        return all_embeddings

    @property
    def dimension(self) -> int:
        return self._dimension


class FallbackEmbeddingProvider(EmbeddingProvider):
    """Route embedding calls to a secondary provider when the primary fails.

    Mirrors ``FallbackLLMProvider`` for LLMs.  The primary already retries
    and circuit-breaks internally (``resilience.py``); this wrapper adds
    cross-provider failover.  Any exception out of the primary — a retryable
    error after its retries are exhausted, an open circuit breaker, or a
    local-model failure such as a corrupt BGE checkpoint — retries the call
    once against the fallback.

    Dimension constraint: the pgvector columns are built for
    ``config.embedding.dimension`` (== the primary's dimension, checked in
    ``get_embedding_provider``).  A fallback whose dimension differs produces
    vectors the schema rejects on every failover write, so a mismatch is
    warned at construction rather than silently stored.
    """

    def __init__(
        self, primary: EmbeddingProvider, fallback: EmbeddingProvider
    ) -> None:
        self._primary = primary
        self._fallback = fallback
        if fallback.dimension != primary.dimension:
            logger.warning(
                "Embedding failover dimension mismatch: primary=%d fallback=%d — "
                "a failover write will produce vectors the pgvector schema rejects. "
                "Align EMBEDDING_FALLBACK_MODEL with EMBEDDING_MODEL (or re-embed "
                "and resize the columns before enabling failover).",
                primary.dimension,
                fallback.dimension,
            )
        logger.info(
            "Embedding failover active: primary=%s (dim %d) -> fallback=%s (dim %d)",
            getattr(primary, "_model", None) or type(primary).__name__,
            primary.dimension,
            getattr(fallback, "_model", None) or type(fallback).__name__,
            fallback.dimension,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            return await self._primary.embed(texts)
        except Exception as exc:
            logger.warning(
                "Primary embedding failed (%s) — failing over to %s",
                exc,
                getattr(self._fallback, "_model", None) or type(self._fallback).__name__,
            )
            return await self._fallback.embed(texts)

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        try:
            return self._primary.embed_sync(texts)
        except Exception as exc:
            logger.warning(
                "Primary embedding failed (%s) — failing over to %s",
                exc,
                getattr(self._fallback, "_model", None) or type(self._fallback).__name__,
            )
            return self._fallback.embed_sync(texts)

    @property
    def dimension(self) -> int:
        """Report the primary's dimension — the schema dimension."""
        return self._primary.dimension


_provider: EmbeddingProvider | None = None
_lock: threading.Lock = threading.Lock()


def _build_embedding_provider(
    provider_name: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    batch_size: int,
    timeout: int,
    normalize: bool = True,
    hf_endpoint: str = "https://hf-mirror.com",
) -> EmbeddingProvider:
    """Construct one embedding provider from explicit settings (primary or fallback)."""
    if provider_name == "local":
        return BGEEmbeddingProvider(
            model_name=model,
            normalize=normalize,
            batch_size=batch_size,
            hf_endpoint=hf_endpoint,
        )
    if provider_name == "openai":
        return OpenAIEmbeddingProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            batch_size=batch_size,
            timeout=timeout,
        )
    raise ValueError(f"Unsupported embedding provider: {provider_name!r}")


def _get_or_create_provider() -> EmbeddingProvider:
    """Build and cache the singleton provider (thread-safe).

    Single source of truth for construction, shared by the sync
    (:func:`get_embedding_provider`) and async (:func:`get_embedding_provider_async`)
    entry points so the two paths can never double-build or cache separately.
    The ``threading.Lock`` is only contended while a build is in flight; after
    startup (the warmup is awaited in ``main.py`` lifespan before the app
    serves requests) it is uncontended on the request path.

    When ``EMBEDDING_FALLBACK_PROVIDER`` is set, returns a
    :class:`FallbackEmbeddingProvider` wrapping the primary and the configured
    fallback; otherwise the primary provider alone (default).
    """
    global _provider
    if _provider is not None:
        return _provider

    with _lock:
        if _provider is not None:  # double-check after acquiring lock
            return _provider

        embedding = config.embedding
        provider = _build_embedding_provider(
            embedding.provider,
            api_key=embedding.api_key,
            base_url=embedding.base_url,
            model=embedding.model,
            batch_size=embedding.batch_size,
            timeout=embedding.timeout,
            normalize=embedding.normalize,
            hf_endpoint=embedding.hf_endpoint,
        )

        if provider.dimension != embedding.dimension:
            logger.warning(
                "Embedding provider reports dimension %d but config/schema "
                "assumes %d (model=%r) — the pgvector columns were built as "
                "vector(%d), so embedding writes will fail. Align "
                "EMBEDDING_MODEL with the schema, or switch models and let "
                "init_db resize the columns, then re-embed with "
                "`python -m scripts.reembed_embeddings`.",
                provider.dimension,
                embedding.dimension,
                embedding.model,
                embedding.dimension,
            )

        if embedding.fallback_provider:
            fallback = _build_embedding_provider(
                embedding.fallback_provider,
                api_key=embedding.fallback_api_key,
                base_url=embedding.fallback_base_url,
                model=embedding.fallback_model,
                batch_size=embedding.fallback_batch_size,
                timeout=embedding.fallback_timeout,
                normalize=embedding.normalize,
                hf_endpoint=embedding.hf_endpoint,
            )
            provider = FallbackEmbeddingProvider(provider, fallback)

        _provider = provider
        return _provider


def get_embedding_provider() -> EmbeddingProvider:
    """Return the singleton embedding provider (sync, for scripts/threads).

    Thread-safe and idempotent: concurrent callers share a single build and a
    single cached instance.  In the app, prefer :func:`get_embedding_provider_async`
    in event-loop contexts so a first-time load (BGE-M3 takes 30s+) never
    blocks the event loop.
    """
    return _get_or_create_provider()


async def get_embedding_provider_async() -> EmbeddingProvider:
    """Return the singleton embedding provider without blocking the event loop.

    The heavy construction (loading BGE-M3 can take 30s+) runs in a background
    thread; the event loop awaits it instead of blocking on the
    ``threading.Lock``, so an embedding request or health probe arriving
    mid-load cannot freeze the loop.  Shares the singleton and build lock with
    :func:`get_embedding_provider` — whichever path loads first wins and
    everyone else returns the same cached instance.
    """
    if _provider is not None:
        return _provider
    return await asyncio.to_thread(_get_or_create_provider)
