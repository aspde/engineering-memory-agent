"""Unit tests for backend/shared/runtime_metrics.py — Prometheus health metrics.

Covers the record helpers, the circuit-breaker state mirroring through the
real ``CircuitBreaker``, the HTTP middleware, and the ``GET /metrics``
endpoint on the real app.  Uses the real ``prometheus_client`` (reset via
``reset_runtime_metrics``) so the tests assert the actual text exposition —
no mock of the metric library.
"""

from __future__ import annotations

import pytest

from backend.shared import config as config_mod
from backend.shared.runtime_metrics import (
    AGENT_SLOTS_IN_USE,
    inc_agent_slots_rejected,
    inc_circuit_breaker_opens,
    inc_circuit_breaker_rejections,
    observe_agent_steps,
    record_http_request,
    record_llm_call,
    render_metrics,
    reset_runtime_metrics,
    set_agent_slots_in_use,
    set_circuit_breaker_state,
)


@pytest.fixture(autouse=True)
def _fresh_metrics() -> None:
    """Every test starts from a clean metric set."""
    reset_runtime_metrics()
    yield
    reset_runtime_metrics()


def _sample_lines() -> str:
    """The rendered exposition with TYPE/HELP/_created lines stripped."""
    return "\n".join(
        line
        for line in render_metrics().splitlines()
        if not line.startswith("#") and "_created" not in line
    )


class TestLlmMetrics:
    def test_llm_call_counter_histogram_tokens(self) -> None:
        record_llm_call(
            scenario="agent_chat",
            status="success",
            latency_ms=1200,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        out = _sample_lines()
        assert (
            'ema_llm_calls_total{scenario="agent_chat",status="success"} 1.0' in out
        )
        assert (
            'ema_llm_calls_total{scenario="agent_chat",status="error"}'
            not in out
        )
        assert (
            'ema_llm_duration_seconds_count{scenario="agent_chat"} 1.0' in out
        )
        assert 'ema_llm_tokens_total{kind="input",scenario="agent_chat"} 100.0' in out
        assert 'ema_llm_tokens_total{kind="output",scenario="agent_chat"} 50.0' in out
        assert 'ema_llm_tokens_total{kind="total",scenario="agent_chat"} 150.0' in out

    def test_error_calls_counted_separately(self) -> None:
        record_llm_call(
            scenario="rerank", status="error", latency_ms=500,
            input_tokens=0, output_tokens=0, total_tokens=0,
        )
        out = _sample_lines()
        assert 'ema_llm_calls_total{scenario="rerank",status="error"} 1.0' in out

    def test_disabled_metrics_record_nothing(self, monkeypatch) -> None:
        monkeypatch.setattr(config_mod.config, "metrics_enabled", False)
        record_llm_call(
            scenario="agent_chat", status="success", latency_ms=10,
            input_tokens=1, output_tokens=1, total_tokens=2,
        )
        record_http_request("GET", "/health", 200, 0.01)
        out = _sample_lines()
        # Zero-valued gauges/histograms still emit sample lines — what must
        # NOT appear is any recorded count for the disabled calls.
        assert "ema_llm_calls_total" not in out
        assert "ema_llm_tokens_total" not in out
        assert "ema_http_requests_total" not in out


class TestHttpMetrics:
    def test_http_request_and_duration_recorded(self) -> None:
        record_http_request("POST", "/api/agent/chat", 200, 0.5)
        record_http_request("GET", "/metrics", 200, 0.001)
        out = _sample_lines()
        assert 'ema_http_requests_total{method="POST",path="/api/agent/chat",status="200"} 1.0' in out
        assert 'ema_http_requests_total{method="GET",path="/metrics",status="200"} 1.0' in out
        assert 'ema_http_request_duration_seconds_count{method="POST",path="/api/agent/chat"} 1.0' in out

    def test_error_status_recorded(self) -> None:
        record_http_request("GET", "/api/agent/thread/x", 404, 0.01)
        out = _sample_lines()
        assert 'ema_http_requests_total{method="GET",path="/api/agent/thread/x",status="404"} 1.0' in out


class TestCircuitBreakerMetrics:
    def test_state_and_counters_through_real_breaker(self) -> None:
        from backend.shared.resilience import CircuitBreaker, CircuitOpenError

        cb = CircuitBreaker("llm:test", failure_threshold=2, cooldown_seconds=0.05)
        cb.before_call()
        cb.record_success()
        # A success closes the breaker → gauge reads 0, no opens/rejections.
        out = _sample_lines()
        assert 'ema_circuit_breaker_state{name="llm:test"} 0.0' in out
        assert "ema_circuit_breaker_opens_total" not in out
        assert "ema_circuit_breaker_rejections_total" not in out

        # Two consecutive failures trip the breaker open.
        cb.record_failure()
        cb.record_failure()
        out = _sample_lines()
        assert 'ema_circuit_breaker_state{name="llm:test"} 1.0' in out
        assert 'ema_circuit_breaker_opens_total{name="llm:test"} 1.0' in out

        # A call while open fails fast → rejection counted.
        with pytest.raises(CircuitOpenError):
            cb.before_call()
        assert 'ema_circuit_breaker_rejections_total{name="llm:test"} 1.0' in _sample_lines()

        # A success closes the breaker → gauge back to 0.
        cb.record_success()
        assert 'ema_circuit_breaker_state{name="llm:test"} 0.0' in _sample_lines()

    def test_direct_helpers(self) -> None:
        set_circuit_breaker_state("embedding:openai", True)
        inc_circuit_breaker_opens("embedding:openai")
        inc_circuit_breaker_rejections("embedding:openai")
        out = _sample_lines()
        assert 'ema_circuit_breaker_state{name="embedding:openai"} 1.0' in out
        assert 'ema_circuit_breaker_opens_total{name="embedding:openai"} 1.0' in out
        assert 'ema_circuit_breaker_rejections_total{name="embedding:openai"} 1.0' in out


class TestAgentMetrics:
    def test_slots_and_steps(self) -> None:
        set_agent_slots_in_use(2)
        assert 'ema_agent_slots_in_use 2.0' in _sample_lines()
        inc_agent_slots_rejected()
        assert 'ema_agent_slots_rejected_total 1.0' in _sample_lines()
        observe_agent_steps(3)
        out = _sample_lines()
        assert 'ema_agent_steps_count 1.0' in out
        assert 'ema_agent_steps_sum 3.0' in out


class TestMetricsMiddleware:
    @pytest.mark.asyncio
    async def test_middleware_records_route_path(self) -> None:
        """Requests through the middleware are counted with their route path."""
        from httpx import ASGITransport, AsyncClient

        from backend.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            await client.get("/health")
            await client.get("/health")
        out = _sample_lines()
        assert 'ema_http_requests_total{method="GET",path="/health",status="200"} 2.0' in out

    @pytest.mark.asyncio
    async def test_middleware_records_500_when_app_raises(self) -> None:
        """An exception in the app is recorded as 500, then re-raised."""
        from httpx import ASGITransport, AsyncClient

        from backend.shared.runtime_metrics import MetricsMiddleware

        async def _exploding(scope, receive, send) -> None:
            raise RuntimeError("boom")

        transport = ASGITransport(app=MetricsMiddleware(_exploding))
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with pytest.raises(RuntimeError):
                await client.get("/explode")
        out = _sample_lines()
        assert (
            'ema_http_requests_total{method="GET",path="/explode",status="500"} 1.0'
            in out
        )

    @pytest.mark.asyncio
    async def test_metrics_endpoint_serves_text_format(self) -> None:
        from httpx import ASGITransport, AsyncClient

        from backend.main import app

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            record_llm_call(
                scenario="agent_chat", status="success", latency_ms=100,
                input_tokens=10, output_tokens=5, total_tokens=15,
            )
            r = await client.get("/metrics")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        assert "ema_llm_calls_total{scenario=\"agent_chat\",status=\"success\"} 1.0" in r.text

    @pytest.mark.asyncio
    async def test_metrics_endpoint_404_when_disabled(self, monkeypatch) -> None:
        from httpx import ASGITransport, AsyncClient

        from backend.main import app

        monkeypatch.setattr(config_mod.config, "metrics_enabled", False)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.get("/metrics")
        assert r.status_code == 404


class TestReset:
    def test_reset_clears_samples(self) -> None:
        record_llm_call(
            scenario="agent_chat", status="success", latency_ms=10,
            input_tokens=1, output_tokens=1, total_tokens=2,
        )
        set_circuit_breaker_state("llm:x", True)
        assert "ema_llm_calls_total{scenario=\"agent_chat\",status=\"success\"}" in _sample_lines()
        reset_runtime_metrics()
        out = _sample_lines()
        # Recorded count series are gone after reset.
        assert "ema_llm_calls_total" not in out
        assert "ema_llm_tokens_total" not in out
        # Gauges go back to zero, not left at their last value.
        AGENT_SLOTS_IN_USE.set(3.0)
        reset_runtime_metrics()
        assert "ema_agent_slots_in_use 0.0" in _sample_lines()
