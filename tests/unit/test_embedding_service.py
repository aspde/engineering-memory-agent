"""Unit tests for embedding service."""

from __future__ import annotations

import asyncio
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


# ── Fallback embedding provider tests ────────────────────────────────


class _FailingProvider(FakeEmbeddingProvider):
    """A primary that always raises — simulates a down/corrupt provider."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("model checkpoint corrupt")

    def embed_sync(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("model checkpoint corrupt")


class TestFallbackEmbeddingProvider:
    """Cross-provider failover — primary failure delegates to the fallback."""

    def test_fallback_used_on_primary_failure(self) -> None:
        from backend.service.embedding_service import FallbackEmbeddingProvider

        primary = _FailingProvider(dimension=4)
        fallback = FakeEmbeddingProvider(dimension=4)
        provider = FallbackEmbeddingProvider(primary, fallback)

        result = asyncio.run(provider.embed(["a", "b"]))
        assert result == [[0.1] * 4, [0.1] * 4]

    def test_fallback_not_called_on_primary_success(self) -> None:
        from backend.service.embedding_service import FallbackEmbeddingProvider

        primary = FakeEmbeddingProvider(dimension=4)
        fallback = FakeEmbeddingProvider(dimension=4)
        # Track whether the fallback is ever touched.
        fallback.embed = AsyncMock(side_effect=AssertionError("fallback used on success"))
        provider = FallbackEmbeddingProvider(primary, fallback)

        result = asyncio.run(provider.embed(["a"]))
        assert result == [[0.1] * 4]

    def test_primary_success_sync(self) -> None:
        from backend.service.embedding_service import FallbackEmbeddingProvider

        primary = FakeEmbeddingProvider(dimension=4)
        fallback = FakeEmbeddingProvider(dimension=4)
        fallback.embed_sync = MagicMock(side_effect=AssertionError("fallback used on success"))
        provider = FallbackEmbeddingProvider(primary, fallback)

        assert provider.embed_sync(["a"]) == [[0.1] * 4]

    def test_fallback_used_on_primary_failure_sync(self) -> None:
        from backend.service.embedding_service import FallbackEmbeddingProvider

        provider = FallbackEmbeddingProvider(
            _FailingProvider(dimension=4), FakeEmbeddingProvider(dimension=4)
        )
        assert provider.embed_sync(["a"]) == [[0.1] * 4]

    def test_fallback_failure_propagates(self) -> None:
        from backend.service.embedding_service import FallbackEmbeddingProvider

        provider = FallbackEmbeddingProvider(
            _FailingProvider(dimension=4), _FailingProvider(dimension=4)
        )
        with pytest.raises(RuntimeError):
            asyncio.run(provider.embed(["a"]))

    def test_dimension_reports_primary(self) -> None:
        from backend.service.embedding_service import FallbackEmbeddingProvider

        provider = FallbackEmbeddingProvider(
            FakeEmbeddingProvider(dimension=8), FakeEmbeddingProvider(dimension=8)
        )
        assert provider.dimension == 8

    def test_dimension_mismatch_warns(self, caplog) -> None:
        """A fallback whose dimension differs from the primary is warned —
        its vectors would be rejected by the pgvector schema."""
        from backend.service.embedding_service import FallbackEmbeddingProvider

        with caplog.at_level("WARNING"):
            FallbackEmbeddingProvider(
                FakeEmbeddingProvider(dimension=4), FakeEmbeddingProvider(dimension=8)
            )
        assert "dimension mismatch" in caplog.text


# ── Breaker isolation between primary and fallback ───────────────────
# Regression: both OpenAIEmbeddingProvider paths used to hard-code the same
# ``"embedding:openai"`` breaker name, so an open primary breaker fast-failed
# the healthy fallback too — exactly when failover is supposed to engage.


class TestOpenAIBreakerIsolation:
    """Primary and fallback OpenAI embedding providers get distinct breakers."""

    @staticmethod
    def _status_error(status: int) -> RuntimeError:
        err = RuntimeError(f"status {status}")
        err.status_code = status  # is_retryable classifies by status_code
        return err

    @pytest.mark.asyncio
    async def test_primary_breaker_open_does_not_block_fallback(
        self, monkeypatch,
    ) -> None:
        from backend.shared import resilience
        from backend.shared.config import config
        from backend.shared.resilience import get_circuit_breaker

        from backend.service.embedding_service import (
            FallbackEmbeddingProvider,
            OpenAIEmbeddingProvider,
        )

        try:
            primary = OpenAIEmbeddingProvider(
                api_key="sk-test",
                base_url="https://api.primary.example/v1",
                model="text-embedding-3-small",
            )
            fallback = OpenAIEmbeddingProvider(
                api_key="sk-test",
                base_url="https://api.fallback.example/v1",
                model="text-embedding-3-small",
            )

            assert primary._breaker_name() != fallback._breaker_name()

            # One retryable failure trips the primary's breaker (no retries).
            monkeypatch.setattr(config.resilience, "circuit_breaker_threshold", 1)
            monkeypatch.setattr(config.resilience, "max_attempts", 1)
            monkeypatch.setattr(config.resilience, "backoff_base", 0.01)
            monkeypatch.setattr(config.resilience, "backoff_max", 0.05)

            primary._async_client.embeddings.create = AsyncMock(
                side_effect=self._status_error(429)
            )
            with pytest.raises(RuntimeError):
                await primary.embed(["text"])
            assert get_circuit_breaker(primary._breaker_name()).is_open

            # The fallback carries its own closed breaker, so failover still
            # succeeds — with the old shared "embedding:openai" name this call
            # would have raised CircuitOpenError instead.
            fallback._async_client.embeddings.create = AsyncMock(
                return_value=_make_embedding_response([[0.5, 0.5]])
            )
            result = await FallbackEmbeddingProvider(primary, fallback).embed(
                ["text"]
            )
            assert result == [[0.5, 0.5]]
            assert not get_circuit_breaker(fallback._breaker_name()).is_open
        finally:
            resilience.reset_circuit_breakers()


# ── Provider loading path: concurrent, idempotent, loop-friendly ─────
# Regression: the singleton getter held a threading.Lock while loading the
# model; a request arriving mid-warmup blocked the *event loop* on
# Lock.acquire() and froze it for 30s+.  The async getter must await the load
# in a background thread (loop stays responsive) and both getters must build
# exactly once under concurrency.


class TestEmbeddingProviderLoadingPath:
    """Loading the singleton is idempotent under concurrency."""

    @staticmethod
    def _fake_build(calls: list, dimension: int, delay: float = 0.05):
        import time

        def _build(*args, **kwargs):
            time.sleep(delay)  # simulate a slow BGE-M3 load
            calls.append(1)
            return FakeEmbeddingProvider(dimension=dimension)

        return _build

    def test_sync_getter_concurrent_load_builds_once(self, monkeypatch) -> None:
        import threading

        import backend.service.embedding_service as mod
        from backend.shared.config import config

        monkeypatch.setattr(mod, "_provider", None)
        monkeypatch.setattr(config.embedding, "fallback_provider", "")
        calls: list = []
        monkeypatch.setattr(
            mod,
            "_build_embedding_provider",
            self._fake_build(calls, dimension=config.embedding.dimension),
        )

        results: list = []
        errors: list = []

        def _call() -> None:
            try:
                results.append(mod.get_embedding_provider())
            except Exception as exc:  # pragma: no cover - assertion below
                errors.append(exc)

        threads = [threading.Thread(target=_call) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(calls) == 1  # built exactly once under contention
        assert all(r is results[0] for r in results)  # one shared instance

    @pytest.mark.asyncio
    async def test_async_getter_awaits_load_without_blocking_loop(
        self, monkeypatch,
    ) -> None:
        import backend.service.embedding_service as mod
        from backend.shared.config import config

        monkeypatch.setattr(mod, "_provider", None)
        monkeypatch.setattr(config.embedding, "fallback_provider", "")
        calls: list = []
        monkeypatch.setattr(
            mod,
            "_build_embedding_provider",
            self._fake_build(
                calls, dimension=config.embedding.dimension, delay=0.15
            ),
        )

        # A heartbeat task proves the event loop keeps scheduling while the
        # model "loads": if the loader blocked the loop (threading.Lock held
        # on the event loop thread), the heartbeat would never tick.
        heartbeat: list[int] = []

        async def _beat() -> None:
            for _ in range(50):
                await asyncio.sleep(0.01)
                heartbeat.append(1)

        beat_task = asyncio.create_task(_beat())
        providers = await asyncio.gather(
            *[mod.get_embedding_provider_async() for _ in range(5)]
        )
        await beat_task

        assert len(calls) == 1  # concurrent async loaders share one build
        assert all(p is providers[0] for p in providers)
        assert len(heartbeat) > 0  # the event loop stayed responsive


# ── Semaphore permit safety under cancellation ───────────────────────
# Regression: ``_encode_with_semaphore`` used to ``await to_thread(acquire)``
# *outside* the try/finally.  A coroutine cancelled exactly there let the
# background thread take a ``BoundedSemaphore`` permit that nobody ever
# released — after enough cancellations the semaphore drained and every
# embed blocked forever (memory writes and searches both depend on it).
# The acquire and the encode now share one thread-pool task, so the ``with``
# block always pairs the release with the permit.


class TestSemaphoreCancellationSafety:
    @pytest.mark.asyncio
    async def test_cancellation_does_not_leak_semaphore_permit(self) -> None:
        import time
        import threading

        import backend.service.embedding_service as mod

        class _SlowSemaphore:
            """A semaphore whose acquire sleeps, opening a cancellation window
            while the thread-pool task is between acquiring and releasing."""

            def __init__(self) -> None:
                self._inner = threading.BoundedSemaphore(1)
                self.acquire_calls = 0
                self.releases = 0

            def acquire(self, blocking: bool = True) -> bool:
                self.acquire_calls += 1
                time.sleep(0.2)  # leave the await cancellable mid-acquire
                return self._inner.acquire(blocking)

            def release(self) -> None:
                self.releases += 1
                self._inner.release()

            def __enter__(self) -> "_SlowSemaphore":
                self.acquire()
                return self

            def __exit__(self, *exc) -> None:
                self.release()

        provider = object.__new__(mod.BGEEmbeddingProvider)
        provider._embed_semaphore = _SlowSemaphore()
        provider._normalize = True
        provider._batch_size = 32
        provider._model = MagicMock()
        provider._model.encode.return_value = [[0.1, 0.2]]

        task = asyncio.create_task(provider._encode_with_semaphore(["x"]))
        await asyncio.sleep(0.05)  # the thread is inside the slow acquire
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The thread-pool task is not cancellable: it finishes the critical
        # section and the ``with`` block releases the permit.
        await asyncio.sleep(0.3)
        assert provider._embed_semaphore.acquire_calls == 1
        assert provider._embed_semaphore.releases == 1, (
            "cancellation leaked the semaphore permit"
        )

