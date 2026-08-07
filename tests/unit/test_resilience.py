"""Unit tests for provider resilience — transport retry + circuit breaker.

Covers ``backend/shared/resilience.py`` and its wiring into the LLM and
embedding providers.  All external SDK calls are mocked (no network).

Providers are built with ``object.__new__`` + hand-assigned mocks because
constructing the real SDK clients costs ~3s each — the tests target the
resilience wiring, not client construction.  Retry settings come from
``config.resilience`` (monkeypatched to near-zero backoff) and the
circuit-breaker registry is reset between tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anthropic
import httpx
import openai
import pytest

from backend.service.embedding_service import OpenAIEmbeddingProvider
from backend.service.llm_service import AnthropicProvider, OpenAICompatibleProvider
from backend.shared import resilience
from backend.shared.config import config
from backend.shared.resilience import CircuitBreaker, CircuitOpenError, is_retryable


# ── Helpers ──────────────────────────────────────────────────────────


def _api_error(exc_cls, status: int, message: str = "err"):
    """Build an SDK status error carrying a real numeric ``status_code``."""
    resp = MagicMock()
    resp.request = MagicMock()
    resp.status_code = status
    return exc_cls(message, response=resp, body=None)


def _chat_response(content: str = "ok"):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content))]
    resp.usage = None
    return resp


def _openai_provider() -> OpenAICompatibleProvider:
    """Provider instance without the slow real-SDK client construction."""
    p = object.__new__(OpenAICompatibleProvider)
    p._model = "m"
    p._temperature = 0.7
    p._max_tokens = 4096
    p._async_client = MagicMock()
    p._sync_client = MagicMock()
    return p


def _anthropic_provider() -> AnthropicProvider:
    p = object.__new__(AnthropicProvider)
    p._model = "claude-test"
    p._max_tokens = 4096
    p._async_client = MagicMock()
    p._sync_client = MagicMock()
    return p


def _embedding_provider() -> OpenAIEmbeddingProvider:
    p = object.__new__(OpenAIEmbeddingProvider)
    p._model = "text-embedding-3-small"
    p._batch_size = 100
    p._async_client = MagicMock()
    p._client = MagicMock()
    return p


class _StatusError(RuntimeError):
    """Minimal status-carrying error for testing ``is_retryable``."""

    def __init__(self, status: int) -> None:
        super().__init__(f"err {status}")
        self.status_code = status


# ── Shared fixtures ──────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_breakers():
    resilience.reset_circuit_breakers()
    yield
    resilience.reset_circuit_breakers()


@pytest.fixture(autouse=True)
def _fast_retry(monkeypatch):
    """Near-zero backoff so retry tests don't sleep for seconds."""
    monkeypatch.setattr(config.resilience, "backoff_base", 0.01)
    monkeypatch.setattr(config.resilience, "backoff_max", 0.05)


# ── is_retryable classification ──────────────────────────────────────


class TestIsRetryable:
    def test_status_429_and_5xx_retryable(self) -> None:
        assert is_retryable(_StatusError(429))
        assert is_retryable(_StatusError(500))
        assert is_retryable(_StatusError(503))

    def test_client_4xx_not_retryable(self) -> None:
        assert not is_retryable(_StatusError(400))
        assert not is_retryable(_StatusError(401))
        assert not is_retryable(_StatusError(403))
        assert not is_retryable(_StatusError(404))

    def test_real_sdk_status_errors_classified_by_status(self) -> None:
        assert is_retryable(_api_error(openai.RateLimitError, 429))
        assert is_retryable(_api_error(openai.InternalServerError, 500))
        assert not is_retryable(_api_error(openai.BadRequestError, 400))

    def test_sdk_timeout_connection_errors_retryable(self) -> None:
        assert is_retryable(openai.APITimeoutError(request=MagicMock()))
        assert is_retryable(openai.APIConnectionError(request=MagicMock()))
        assert is_retryable(anthropic.APITimeoutError(request=MagicMock()))
        assert is_retryable(anthropic.APIConnectionError(request=MagicMock()))

    def test_httpx_and_builtin_transport_errors_retryable(self) -> None:
        assert is_retryable(httpx.TimeoutException("t"))
        assert is_retryable(httpx.ConnectError("c"))
        assert is_retryable(TimeoutError("t"))
        assert is_retryable(ConnectionError("c"))

    def test_unrelated_errors_not_retryable(self) -> None:
        assert not is_retryable(ValueError("nope"))
        assert not is_retryable(RuntimeError("nope"))


# ── CircuitBreaker unit semantics ────────────────────────────────────


class TestCircuitBreakerUnit:
    def test_starts_closed(self) -> None:
        cb = CircuitBreaker("t", failure_threshold=3, cooldown_seconds=60)
        assert not cb.is_open
        cb.before_call()  # no raise

    def test_trips_at_threshold(self) -> None:
        cb = CircuitBreaker("t", failure_threshold=3, cooldown_seconds=60)
        cb.record_failure()
        assert not cb.is_open
        cb.record_failure()
        assert not cb.is_open
        cb.record_failure()
        assert cb.is_open
        with pytest.raises(CircuitOpenError):
            cb.before_call()

    def test_success_resets_failure_count(self) -> None:
        cb = CircuitBreaker("t", failure_threshold=3, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        cb.record_failure()
        assert not cb.is_open  # never reached threshold consecutively

    def test_recovers_after_cooldown(self) -> None:
        import time

        cb = CircuitBreaker("t", failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        assert cb.is_open
        time.sleep(0.02)
        cb.before_call()  # cooldown elapsed → auto reset, probe admitted
        assert not cb.is_open


# ── Retry behavior through the providers ─────────────────────────────


class TestProviderRetry:
    @pytest.mark.asyncio
    async def test_retries_transient_429_then_succeeds(self) -> None:
        provider = _openai_provider()
        create = AsyncMock(
            side_effect=[_api_error(openai.RateLimitError, 429), _chat_response("ok")]
        )
        provider._async_client.chat.completions.create = create

        result = await provider.chat([{"role": "user", "content": "hi"}])
        assert result == "ok"
        assert create.await_count == 2

    @pytest.mark.asyncio
    async def test_retries_429_then_5xx_then_succeeds(self) -> None:
        provider = _openai_provider()
        create = AsyncMock(
            side_effect=[
                _api_error(openai.RateLimitError, 429),
                _api_error(openai.InternalServerError, 500),
                _chat_response("ok"),
            ]
        )
        provider._async_client.chat.completions.create = create

        result = await provider.chat([{"role": "user", "content": "hi"}])
        assert result == "ok"
        assert create.await_count == 3

    @pytest.mark.asyncio
    async def test_exhausts_retries_and_raises(self, monkeypatch) -> None:
        monkeypatch.setattr(config.resilience, "max_attempts", 2)
        provider = _openai_provider()
        create = AsyncMock(side_effect=_api_error(openai.RateLimitError, 429))
        provider._async_client.chat.completions.create = create

        with pytest.raises(openai.RateLimitError):
            await provider.chat([{"role": "user", "content": "hi"}])
        assert create.await_count == 2  # max_attempts total, no more

    @pytest.mark.asyncio
    async def test_non_retryable_raises_immediately(self) -> None:
        provider = _openai_provider()
        create = AsyncMock(side_effect=_api_error(openai.BadRequestError, 400))
        provider._async_client.chat.completions.create = create

        with pytest.raises(openai.BadRequestError):
            await provider.chat([{"role": "user", "content": "hi"}])
        assert create.await_count == 1

    @pytest.mark.asyncio
    async def test_anthropic_provider_retries_connection_error(self) -> None:
        provider = _anthropic_provider()
        resp = MagicMock()
        resp.content = [MagicMock(type="text", text="hello")]
        resp.usage = None
        create = AsyncMock(
            side_effect=[httpx.ConnectError("down"), httpx.ConnectError("down"), resp]
        )
        provider._async_client.messages.create = create

        result = await provider.chat([{"role": "user", "content": "hi"}])
        assert result == "hello"
        assert create.await_count == 3

    @pytest.mark.asyncio
    async def test_embedding_provider_retries(self) -> None:
        provider = _embedding_provider()
        data = MagicMock(data=[MagicMock(embedding=[0.1, 0.2])])
        create = AsyncMock(side_effect=[_api_error(openai.RateLimitError, 429), data])
        provider._async_client.embeddings.create = create

        result = await provider.embed(["text"])
        assert result == [[0.1, 0.2]]
        assert create.await_count == 2

    def test_sync_path_retries(self) -> None:
        provider = _openai_provider()
        create = MagicMock(
            side_effect=[_api_error(openai.RateLimitError, 429), _chat_response("ok")]
        )
        provider._sync_client.chat.completions.create = create

        result = provider.chat_sync([{"role": "user", "content": "hi"}])
        assert result == "ok"
        assert create.call_count == 2


# ── Circuit breaker wiring through the providers ─────────────────────


class TestCircuitBreakerWiring:
    @pytest.mark.asyncio
    async def test_fails_fast_when_open(self, monkeypatch) -> None:
        monkeypatch.setattr(config.resilience, "circuit_breaker_threshold", 3)
        provider = _openai_provider()
        create = AsyncMock(side_effect=_api_error(openai.RateLimitError, 429))
        provider._async_client.chat.completions.create = create

        for _ in range(3):  # 3 retryable failures → breaker trips
            with pytest.raises(openai.RateLimitError):
                await provider.chat([{"role": "user", "content": "hi"}])
        calls_before = create.await_count

        with pytest.raises(CircuitOpenError):
            await provider.chat([{"role": "user", "content": "hi"}])
        assert create.await_count == calls_before  # no provider call while open

    @pytest.mark.asyncio
    async def test_recovers_and_probe_succeeds_after_cooldown(
        self, monkeypatch
    ) -> None:
        import time

        monkeypatch.setattr(config.resilience, "circuit_breaker_threshold", 1)
        monkeypatch.setattr(config.resilience, "circuit_breaker_cooldown", 0.01)
        provider = _openai_provider()
        create = AsyncMock(side_effect=_api_error(openai.RateLimitError, 429))
        provider._async_client.chat.completions.create = create

        with pytest.raises(openai.RateLimitError):
            await provider.chat([{"role": "user", "content": "hi"}])  # trip
        with pytest.raises(CircuitOpenError):
            await provider.chat([{"role": "user", "content": "hi"}])

        time.sleep(0.02)  # cooldown elapses → recovery probe admitted
        create.side_effect = None
        create.return_value = _chat_response("recovered")
        result = await provider.chat([{"role": "user", "content": "hi"}])
        assert result == "recovered"
        assert not resilience.get_circuit_breaker("llm:openai").is_open

    @pytest.mark.asyncio
    async def test_chat_json_fails_fast_when_breaker_open(self, monkeypatch) -> None:
        monkeypatch.setattr(config.resilience, "circuit_breaker_threshold", 1)
        provider = _openai_provider()
        create = AsyncMock(side_effect=_api_error(openai.RateLimitError, 429))
        provider._async_client.chat.completions.create = create

        with pytest.raises(openai.RateLimitError):
            await provider.chat([{"role": "user", "content": "hi"}])  # trip

        provider._async_client.chat.completions.create = AsyncMock(
            return_value=_chat_response('{"ok": true}')
        )
        with pytest.raises(CircuitOpenError):
            await provider.chat_json(
                [{"role": "user", "content": "hi"}], json_schema={"type": "object"}
            )
        # breaker-only path: no SDK call attempted while open
        assert provider._async_client.chat.completions.create.await_count == 0

    @pytest.mark.asyncio
    async def test_chat_json_passes_through_when_closed(self) -> None:
        provider = _openai_provider()
        create = AsyncMock(return_value=_chat_response('{"ok": true}'))
        provider._async_client.chat.completions.create = create

        raw = await provider.chat_json(
            [{"role": "user", "content": "hi"}], json_schema={"type": "object"}
        )
        assert raw == '{"ok": true}'
        assert create.await_count == 1


# ── Metrics interaction: usage recorded only on success ──────────────


class TestUsageOnRetry:
    @pytest.mark.asyncio
    async def test_usage_recorded_once_after_retry_succeeds(self) -> None:
        from backend.shared import metrics

        metrics.reset_token_usage()
        provider = _openai_provider()
        resp = _chat_response("ok")
        resp.usage = SimpleNamespace(total_tokens=7)
        create = AsyncMock(side_effect=[_api_error(openai.RateLimitError, 429), resp])
        provider._async_client.chat.completions.create = create

        await provider.chat([{"role": "user", "content": "hi"}], scenario="agent_final")
        # One successful response → tokens counted once, despite 2 SDK calls.
        assert metrics.get_token_usage() == {"agent_final": 7}
