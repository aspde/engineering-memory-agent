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
    p._base_url = "https://test"
    p._breaker_name = f"llm:openai:{p._base_url}|{p._model}"
    p._async_client = MagicMock()
    p._sync_client = MagicMock()
    return p


def _anthropic_provider() -> AnthropicProvider:
    p = object.__new__(AnthropicProvider)
    p._model = "claude-test"
    p._max_tokens = 4096
    p._prompt_caching = False  # retry tests exercise transport, not caching
    p._breaker_name = f"llm:anthropic:{p._model}"
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


def _wait_cooldown_elapsed(breaker: CircuitBreaker, margin: float = 0.02) -> None:
    """Block until the breaker's cooldown has definitely elapsed.

    A bare ``time.sleep(0.02)`` against a 0.01s cooldown races on Windows
    (timer granularity ≈ 15.6ms leaves ~5ms of margin; under full-suite load
    the probe call can land before the deadline and flake).  Polling on the
    breaker's monotonic ``_open_until`` is deterministic regardless of timer
    resolution.
    """
    import time

    deadline = breaker._open_until + margin
    while time.monotonic() < deadline:
        time.sleep(0.005)


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
        probe_token = cb.before_call()  # cooldown elapsed → the single recovery probe is admitted
        assert probe_token is not None
        # HALF-OPEN: the probe is in flight, so calls still fail fast…
        assert cb.is_open
        with pytest.raises(CircuitOpenError):
            cb.before_call()  # only one probe — every other caller fails fast
        cb.record_success(probe_token)  # probe succeeded → CLOSED
        assert not cb.is_open

    def test_probe_failure_reopens_immediately(self) -> None:
        """One failed probe re-opens the breaker even below the failure
        threshold — a failed probe means the provider is still down."""
        import time

        cb = CircuitBreaker("t", failure_threshold=3, cooldown_seconds=0.01)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()  # trip OPEN at the threshold
        time.sleep(0.02)
        cb.before_call()  # admit the recovery probe
        assert cb.is_open
        cb.record_failure()  # probe fails → back to OPEN
        assert cb.is_open
        with pytest.raises(CircuitOpenError):
            cb.before_call()

    def test_stale_probe_does_not_wedge(self) -> None:
        """A probe that never resolves (e.g. its task was cancelled) must not
        wedge the breaker — after one cooldown window a fresh probe may be
        admitted instead of failing fast forever."""
        import time

        cb = CircuitBreaker("t", failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        cb.before_call()  # admit the probe
        time.sleep(0.02)  # probe never resolves → becomes stale
        assert not cb.is_open
        cb.before_call()  # a fresh probe is admitted, not a CircuitOpenError
        assert cb.is_open

    def test_stale_inflight_success_does_not_close_open_breaker(self) -> None:
        """Regression: a call admitted *before* the breaker tripped and still
        in flight when the breaker opens must not close it when it finally
        succeeds.  Otherwise the breaker oscillates OPEN→CLOSED→OPEN — the
        stale success re-closes it, a fresh burst of failures re-opens it —
        while hammering a provider that is still down."""
        cb = CircuitBreaker("t", failure_threshold=1, cooldown_seconds=60)
        cb.before_call()  # call A admitted while CLOSED (token None)
        cb.record_failure()  # another call trips the breaker → OPEN
        assert cb.is_open
        # Call A — in flight since before the trip — eventually succeeds.
        cb.record_success()
        assert cb.is_open  # must NOT re-close the open breaker
        with pytest.raises(CircuitOpenError):
            cb.before_call()  # still failing fast

    def test_wrong_probe_token_does_not_close_half_open(self) -> None:
        """A success carrying a token that does not match the admitted probe —
        e.g. a stale caller racing with a stale token — must not close the
        breaker; only the current probe's token may."""
        import time

        cb = CircuitBreaker("t", failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        probe_token = cb.before_call()  # the admitted probe
        assert probe_token is not None
        cb.record_success(probe_token + 999)  # stale caller's wrong token
        assert cb.is_open  # only the real probe's token may close it
        cb.record_success(probe_token)
        assert not cb.is_open

    def test_only_admitted_probe_success_closes_half_open(self) -> None:
        """The single half-open recovery probe — identified by the token
        ``before_call`` returns — is the only call that can close the breaker;
        a stale success from a call admitted before the trip (no token) must
        leave it HALF-OPEN."""
        import time

        cb = CircuitBreaker("t", failure_threshold=1, cooldown_seconds=0.01)
        cb.before_call()  # call A admitted while CLOSED (token None)
        cb.record_failure()  # trip → OPEN
        time.sleep(0.02)  # cooldown elapses
        probe_token = cb.before_call()  # probe B admitted (token set)
        assert probe_token is not None
        assert cb.is_open  # HALF-OPEN: probe B in flight
        cb.record_success()  # stale call A succeeds with no token
        assert cb.is_open  # must stay HALF-OPEN — probe B still the decider
        cb.record_success(probe_token)  # the admitted probe succeeds → CLOSED
        assert not cb.is_open

    def test_stale_probe_success_with_old_token_does_not_close(self) -> None:
        """A probe from an earlier half-open cycle that went stale must not
        close the breaker when it eventually resolves — only the *current*
        probe's token may close it."""
        import time

        cb = CircuitBreaker("t", failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()
        time.sleep(0.02)
        old_token = cb.before_call()  # probe 1 admitted
        time.sleep(0.02)  # probe 1 never resolves → stale
        new_token = cb.before_call()  # fresh probe 2 admitted
        assert new_token != old_token
        cb.record_success(old_token)  # stale probe 1 eventually succeeds
        assert cb.is_open  # must stay HALF-OPEN — probe 2 still in flight
        cb.record_success(new_token)  # probe 2 succeeds → CLOSED
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
        monkeypatch.setattr(config.resilience, "circuit_breaker_threshold", 1)
        monkeypatch.setattr(config.resilience, "circuit_breaker_cooldown", 0.01)
        provider = _openai_provider()
        create = AsyncMock(side_effect=_api_error(openai.RateLimitError, 429))
        provider._async_client.chat.completions.create = create

        with pytest.raises(openai.RateLimitError):
            await provider.chat([{"role": "user", "content": "hi"}])  # trip
        with pytest.raises(CircuitOpenError):
            await provider.chat([{"role": "user", "content": "hi"}])

        _wait_cooldown_elapsed(
            resilience.get_circuit_breaker(provider._breaker_name)
        )  # cooldown elapses → recovery probe admitted
        create.side_effect = None
        create.return_value = _chat_response("recovered")
        result = await provider.chat([{"role": "user", "content": "hi"}])
        assert result == "recovered"
        assert not resilience.get_circuit_breaker(provider._breaker_name).is_open

    @pytest.mark.asyncio
    async def test_probe_failure_reopens_breaker(self, monkeypatch) -> None:
        monkeypatch.setattr(config.resilience, "circuit_breaker_threshold", 1)
        monkeypatch.setattr(config.resilience, "circuit_breaker_cooldown", 0.01)
        monkeypatch.setattr(config.resilience, "max_attempts", 1)
        provider = _openai_provider()
        create = AsyncMock(side_effect=_api_error(openai.RateLimitError, 429))
        provider._async_client.chat.completions.create = create

        with pytest.raises(openai.RateLimitError):
            await provider.chat([{"role": "user", "content": "hi"}])  # trip
        _wait_cooldown_elapsed(
            resilience.get_circuit_breaker(provider._breaker_name)
        )  # cooldown elapses → the recovery probe is admitted
        with pytest.raises(openai.RateLimitError):
            await provider.chat([{"role": "user", "content": "hi"}])  # probe fails
        # Probe failure re-opened the breaker → the next call fails fast with
        # no SDK request (a single probe, not a stampede).
        with pytest.raises(CircuitOpenError):
            await provider.chat([{"role": "user", "content": "hi"}])
        assert create.await_count == 2  # trip + probe only

    @pytest.mark.asyncio
    async def test_non_retryable_probe_failure_settles_probe(
        self, monkeypatch
    ) -> None:
        """A recovery probe that fails with a *non-retryable* error (e.g. a
        401 auth problem, which the retry loop never touches) must still
        settle the probe: the caller sees the real 401, and the breaker
        re-opens instead of staying wedged HALF-OPEN failing everyone fast
        for the rest of the cooldown."""
        monkeypatch.setattr(config.resilience, "circuit_breaker_threshold", 1)
        monkeypatch.setattr(config.resilience, "circuit_breaker_cooldown", 0.01)
        monkeypatch.setattr(config.resilience, "max_attempts", 1)
        provider = _openai_provider()
        create = AsyncMock(side_effect=_api_error(openai.RateLimitError, 429))
        provider._async_client.chat.completions.create = create

        with pytest.raises(openai.RateLimitError):
            await provider.chat([{"role": "user", "content": "hi"}])  # trip

        _wait_cooldown_elapsed(
            resilience.get_circuit_breaker(provider._breaker_name)
        )  # cooldown elapses → the recovery probe is admitted
        create.side_effect = _api_error(openai.AuthenticationError, 401)
        # The probe's own caller sees the real 401 (not a CircuitOpenError)…
        with pytest.raises(openai.AuthenticationError):
            await provider.chat([{"role": "user", "content": "hi"}])

        breaker = resilience.get_circuit_breaker(provider._breaker_name)
        # …and the probe is settled: no longer "probe in flight", re-opened.
        assert not breaker._probing
        assert breaker.is_open
        with pytest.raises(CircuitOpenError):
            await provider.chat([{"role": "user", "content": "hi"}])  # fast fail
        assert create.await_count == 2  # trip + probe only

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


# ── Streaming connection retry ──────────────────────────────────────
# chat_raw_stream now routes connection establishment through the same
# tenacity retry as the non-streaming paths (a 429/5xx at create time is
# retried before any token is consumed).


class _FakeStreamChunk:
    def __init__(self, text: str) -> None:
        self.choices = [
            SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))
        ]
        self.usage = None


class _FakeStream:
    def __init__(self, chunks: list) -> None:
        self._chunks = list(chunks)
        self.usage = None

    def __aiter__(self):
        async def _iter():
            for c in self._chunks:
                yield c

        return _iter()


class TestStreamingConnectionRetry:
    @pytest.mark.asyncio
    async def test_openai_stream_retries_connection_429(self) -> None:
        provider = _openai_provider()
        create = AsyncMock(
            side_effect=[
                _api_error(openai.RateLimitError, 429),
                _FakeStream([_FakeStreamChunk("hello")]),
            ]
        )
        provider._async_client.chat.completions.create = create

        events = [
            event
            async for event in provider.chat_raw_stream(
                [{"role": "user", "content": "hi"}], scenario="agent_chat"
            )
        ]

        assert events == [{"type": "content", "text": "hello"}]
        # 429 at connection time was retried, then the stream succeeded.
        assert create.await_count == 2


# ── Cross-provider failover ─────────────────────────────────────────
# FallbackLLMProvider: retryable primary failures (or an open breaker) are
# retried once against the fallback; non-retryable 4xx propagate untouched.


def _fake_llm_provider(name: str):
    """A duck-typed provider surface for failover tests."""
    return SimpleNamespace(
        model=name,
        PROVIDER_NAME=name,
        chat=AsyncMock(),
        chat_raw=AsyncMock(),
        chat_sync=MagicMock(),
        chat_json=AsyncMock(),
        chat_raw_stream=AsyncMock(),
        chat_stream=AsyncMock(),
    )


class TestProviderFailover:
    @pytest.mark.asyncio
    async def test_fallback_used_when_primary_raises_retryable(self) -> None:
        from backend.service.llm_service import FallbackLLMProvider

        primary = _fake_llm_provider("primary")
        fallback = _fake_llm_provider("fallback")
        fallback.chat.return_value = "fallback-ok"
        primary.chat.side_effect = _api_error(openai.RateLimitError, 429)
        wrapped = FallbackLLMProvider(primary, fallback)

        result = await wrapped.chat(
            [{"role": "user", "content": "hi"}], scenario="agent_chat"
        )

        assert result == "fallback-ok"
        primary.chat.assert_awaited_once()
        fallback.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fallback_not_used_for_non_retryable(self) -> None:
        from backend.service.llm_service import FallbackLLMProvider

        primary = _fake_llm_provider("primary")
        fallback = _fake_llm_provider("fallback")
        primary.chat.side_effect = _api_error(openai.BadRequestError, 400)
        wrapped = FallbackLLMProvider(primary, fallback)

        with pytest.raises(openai.BadRequestError):
            await wrapped.chat([{"role": "user", "content": "hi"}])
        fallback.chat.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_fallback_used_when_primary_breaker_open(self) -> None:
        from backend.service.llm_service import FallbackLLMProvider

        primary = _fake_llm_provider("primary")
        fallback = _fake_llm_provider("fallback")
        fallback.chat.return_value = "fallback-ok"
        primary.chat.side_effect = CircuitOpenError("breaker open")
        wrapped = FallbackLLMProvider(primary, fallback)

        result = await wrapped.chat([{"role": "user", "content": "hi"}])

        assert result == "fallback-ok"
        fallback.chat.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_same_family_fallback_uses_distinct_breakers(
        self, monkeypatch
    ) -> None:
        """A deepseek→openai failover must not share one circuit breaker.

        Regression: OpenAICompatibleProvider hard-coded a single ``llm:openai``
        breaker, so the primary and its fallback hit the *same* process-wide
        breaker.  Tripping the primary (which is exactly when failover is
        supposed to engage) opened the breaker for the fallback too, fast-
        failing every failover call during the cooldown window.
        """
        from backend.service.llm_service import OpenAICompatibleProvider

        monkeypatch.setattr(config.resilience, "circuit_breaker_threshold", 1)
        monkeypatch.setattr(config.resilience, "circuit_breaker_cooldown", 30)

        primary = object.__new__(OpenAICompatibleProvider)
        primary._model = "deepseek-chat"
        primary._temperature = 0.7
        primary._max_tokens = 4096
        primary._base_url = "https://api.deepseek.com"
        primary._breaker_name = f"llm:openai:{primary._base_url}|{primary._model}"
        primary._async_client = MagicMock()
        primary._sync_client = MagicMock()

        fallback = object.__new__(OpenAICompatibleProvider)
        fallback._model = "gpt-4o-mini"
        fallback._temperature = 0.7
        fallback._max_tokens = 4096
        fallback._base_url = "https://api.openai.com/v1"
        fallback._breaker_name = f"llm:openai:{fallback._base_url}|{fallback._model}"
        fallback._async_client = MagicMock()
        fallback._sync_client = MagicMock()

        assert primary._breaker_name != fallback._breaker_name

        # Trip the primary's breaker…
        primary._async_client.chat.completions.create = AsyncMock(
            side_effect=_api_error(openai.RateLimitError, 429)
        )
        with pytest.raises(openai.RateLimitError):
            await primary.chat([{"role": "user", "content": "hi"}])
        assert resilience.get_circuit_breaker(primary._breaker_name).is_open

        # …and the fallback's breaker stays closed, so a failover call (which
        # consults the fallback's breaker before routing) is admitted.
        assert not resilience.get_circuit_breaker(fallback._breaker_name).is_open

    @pytest.mark.asyncio
    async def test_stream_falls_back_before_first_token(self) -> None:
        from backend.service.llm_service import FallbackLLMProvider

        primary = _fake_llm_provider("primary")
        fallback = _fake_llm_provider("fallback")

        async def _primary_stream(*args, **kwargs):
            raise openai.APIConnectionError(request=MagicMock())
            yield  # pragma: no cover

        async def _fallback_stream(*args, **kwargs):
            yield {"type": "content", "text": "fb"}

        primary.chat_raw_stream = _primary_stream
        fallback.chat_raw_stream = _fallback_stream
        wrapped = FallbackLLMProvider(primary, fallback)

        events = [
            event
            async for event in wrapped.chat_raw_stream(
                [{"role": "user", "content": "hi"}]
            )
        ]

        assert events == [{"type": "content", "text": "fb"}]

    @pytest.mark.asyncio
    async def test_stream_does_not_fail_over_mid_stream(self) -> None:
        from backend.service.llm_service import FallbackLLMProvider

        primary = _fake_llm_provider("primary")
        fallback = _fake_llm_provider("fallback")

        async def _primary_stream(*args, **kwargs):
            yield {"type": "content", "text": "prefix"}
            raise openai.APIConnectionError(request=MagicMock())

        def _should_not_be_called(*args, **kwargs):
            raise AssertionError("fallback must not be used mid-stream")

        primary.chat_raw_stream = _primary_stream
        fallback.chat_raw_stream = _should_not_be_called
        wrapped = FallbackLLMProvider(primary, fallback)

        collected: list = []
        with pytest.raises(openai.APIConnectionError):
            async for event in wrapped.chat_raw_stream(
                [{"role": "user", "content": "hi"}]
            ):
                collected.append(event)
        # The prefix was already delivered; the failure propagates untouched.
        assert collected == [{"type": "content", "text": "prefix"}]


class TestProviderFailoverFactory:
    """``get_llm_provider`` wires the fallback wrapper only when configured."""

    def test_no_fallback_by_default(self, monkeypatch) -> None:
        import backend.service.llm_service as mod

        monkeypatch.setattr(mod, "_provider", None)
        monkeypatch.setattr(config.llm, "fallback_provider", "")
        fake_primary = SimpleNamespace(model="primary", PROVIDER_NAME="primary")
        monkeypatch.setattr(mod, "_build_provider", lambda *a, **k: fake_primary)

        assert mod.get_llm_provider() is fake_primary

    def test_wraps_fallback_when_configured(self, monkeypatch) -> None:
        import backend.service.llm_service as mod

        monkeypatch.setattr(mod, "_provider", None)
        monkeypatch.setattr(config.llm, "fallback_provider", "deepseek")
        fake_primary = SimpleNamespace(model="primary", PROVIDER_NAME="primary")
        fake_fallback = SimpleNamespace(model="fb", PROVIDER_NAME="fallback")

        def _build(provider, api_key, base_url, model, **kwargs):
            # The primary build passes temperature/prompt_caching; the
            # fallback build passes only max_tokens/timeout.  (Both may carry
            # the same provider name, e.g. deepseek → deepseek.)
            if "temperature" in kwargs or "prompt_caching" in kwargs:
                return fake_primary
            return fake_fallback

        monkeypatch.setattr(mod, "_build_provider", _build)

        provider = mod.get_llm_provider()
        assert isinstance(provider, mod.FallbackLLMProvider)
        assert provider._primary is fake_primary
        assert provider._fallback is fake_fallback
