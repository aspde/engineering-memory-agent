"""Tests for LLM health alerting (``backend/service/alerts.py``).

Covers the three check dimensions (error rate, structured-failure growth,
circuit breaker), the cooldown suppression, and the opt-in 飞书 push.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import text

from backend.db import get_session_factory
from backend.service import alerts
from backend.shared.metrics import (
    record_structured_failure,
    reset_structured_failures,
)


@pytest.fixture(autouse=True)
def _clean_alert_state():
    """Isolate alert cooldown + structured-failure baselines between tests."""
    alerts.reset_alert_state()
    reset_structured_failures()
    yield
    alerts.reset_alert_state()
    reset_structured_failures()


@pytest.fixture(autouse=True)
async def _clean_llm_usage_table():
    """Isolate DB-backed error-rate checks."""
    async with get_session_factory()() as session:
        await session.execute(text("DELETE FROM llm_usage"))
        await session.commit()
    yield
    async with get_session_factory()() as session:
        await session.execute(text("DELETE FROM llm_usage"))
        await session.commit()


async def _insert_calls(statuses: list[str], attempts: list[int] | None = None) -> None:
    """Insert llm_usage rows with the given statuses (window: now).

    ``attempts`` mirrors the attempts column (provider hits before outcome);
    when given, rows carry it (``attempts > 1`` = retried before succeeding).
    """
    attempts = attempts or [1] * len(statuses)
    values = ", ".join(
        f"('agent_chat', 'p', 'm', '{s}', {a})" for s, a in zip(statuses, attempts)
    )
    async with get_session_factory()() as session:
        await session.execute(
            text(
                "INSERT INTO llm_usage (scenario, provider, model, status, attempts) "
                f"VALUES {values}"
            )
        )
        await session.commit()


class TestErrorRateAlert:
    @pytest.mark.asyncio
    async def test_high_error_rate_fires(self, monkeypatch) -> None:
        monkeypatch.setattr(alerts.config, "alert_error_rate_threshold", 0.3)
        await _insert_calls(["error", "error", "success", "success", "success"])

        fired = await alerts.check_alerts()
        assert any(a["key"] == "llm_error_rate" for a in fired)

    @pytest.mark.asyncio
    async def test_low_error_rate_silent(self, monkeypatch) -> None:
        monkeypatch.setattr(alerts.config, "alert_error_rate_threshold", 0.9)
        await _insert_calls(["error", "success", "success", "success", "success"])

        fired = await alerts.check_alerts()
        assert not any(a["key"] == "llm_error_rate" for a in fired)

    @pytest.mark.asyncio
    async def test_min_calls_guard_silent(self, monkeypatch) -> None:
        """A tiny sample (2 calls) must not fire even at a 100% error rate."""
        monkeypatch.setattr(alerts.config, "alert_error_rate_threshold", 0.1)
        await _insert_calls(["error", "success"])

        fired = await alerts.check_alerts()
        assert not any(a["key"] == "llm_error_rate" for a in fired)

    @pytest.mark.asyncio
    async def test_cooldown_suppresses_repeat(self, monkeypatch) -> None:
        monkeypatch.setattr(alerts.config, "alert_error_rate_threshold", 0.3)
        await _insert_calls(["error", "error", "success", "success", "success"])

        first = await alerts.check_alerts()
        second = await alerts.check_alerts()
        assert any(a["key"] == "llm_error_rate" for a in first)
        assert not any(a["key"] == "llm_error_rate" for a in second)

    @pytest.mark.asyncio
    async def test_empty_table_silent(self, monkeypatch) -> None:
        monkeypatch.setattr(alerts.config, "alert_error_rate_threshold", 0.0)
        fired = await alerts.check_alerts()
        assert not any(a["key"] == "llm_error_rate" for a in fired)


class TestRetryJitterAlert:
    """attempts>1 successes — the "provider was unstable but recovered" signal
    the error-rate check cannot see (no error status was ever recorded)."""

    @pytest.mark.asyncio
    async def test_high_retry_share_fires(self, monkeypatch) -> None:
        monkeypatch.setattr(alerts.config, "alert_error_rate_threshold", 0.9)
        # 4/6 calls retried before succeeding (attempts=3), 2 clean.
        await _insert_calls(
            ["success"] * 6, attempts=[3, 3, 3, 3, 1, 1]
        )

        fired = await alerts.check_alerts()
        assert any(a["key"] == "llm_retry_jitter" for a in fired)

    @pytest.mark.asyncio
    async def test_low_retry_share_silent(self, monkeypatch) -> None:
        monkeypatch.setattr(alerts.config, "alert_error_rate_threshold", 0.9)
        # Only 1 retried call of 10 — below _MIN_RETRY_CALLS and the ratio.
        await _insert_calls(["success"] * 10, attempts=[3] + [1] * 9)

        fired = await alerts.check_alerts()
        assert not any(a["key"] == "llm_retry_jitter" for a in fired)

    @pytest.mark.asyncio
    async def test_final_failures_not_counted_as_jitter(self, monkeypatch) -> None:
        """A call that retried but ultimately failed is an error row, not a
        jitter row — the error-rate check owns it."""
        monkeypatch.setattr(alerts.config, "alert_error_rate_threshold", 0.9)
        # 4 failures with attempts=3, 4 clean successes: 0 jitter, low error.
        await _insert_calls(
            ["error"] * 4 + ["success"] * 4, attempts=[3] * 4 + [1] * 4
        )

        fired = await alerts.check_alerts()
        assert not any(a["key"] == "llm_retry_jitter" for a in fired)
        assert not any(a["key"] == "llm_error_rate" for a in fired)


class TestStructuredFailureAlert:
    @pytest.mark.asyncio
    async def test_growth_over_threshold_fires(self) -> None:
        for _ in range(6):
            record_structured_failure("extraction_entities")

        fired = await alerts.check_alerts()
        assert any(
            a["key"] == "structured_failure:extraction_entities" for a in fired
        )

    @pytest.mark.asyncio
    async def test_small_growth_silent(self) -> None:
        record_structured_failure("extraction_entities")  # 1 < threshold 5

        fired = await alerts.check_alerts()
        assert not any(a["key"].startswith("structured_failure") for a in fired)


class TestCircuitAlert:
    @pytest.mark.asyncio
    async def test_open_breaker_fires(self, monkeypatch) -> None:
        mock_breaker = MagicMock()
        mock_breaker.is_open = True
        monkeypatch.setattr(alerts, "get_circuit_breaker", lambda name: mock_breaker)

        fired = await alerts.check_alerts()
        assert any(a["key"] == "llm_circuit_open" for a in fired)

    @pytest.mark.asyncio
    async def test_closed_breaker_silent(self, monkeypatch) -> None:
        mock_breaker = MagicMock()
        mock_breaker.is_open = False
        monkeypatch.setattr(alerts, "get_circuit_breaker", lambda name: mock_breaker)

        fired = await alerts.check_alerts()
        assert not any(a["key"] == "llm_circuit_open" for a in fired)


class TestFallbackJudgeCircuitAlert:
    """Fallback and judge breakers are checked when those providers are set."""

    @pytest.mark.asyncio
    async def test_open_fallback_breaker_fires(self, monkeypatch) -> None:
        mock_breaker = MagicMock()
        mock_breaker.is_open = True
        monkeypatch.setattr(alerts.config.llm, "fallback_provider", "deepseek")
        monkeypatch.setattr(alerts.config.llm, "fallback_base_url", "https://fb.test/v1")
        monkeypatch.setattr(alerts.config.llm, "fallback_model", "fb-model")
        monkeypatch.setattr(alerts.config.llm, "judge_provider", "")

        names: list[str] = []

        def _get_breaker(name: str) -> MagicMock:
            names.append(name)
            return mock_breaker

        monkeypatch.setattr(alerts, "get_circuit_breaker", _get_breaker)

        fired = await alerts.check_alerts()
        assert any(a["key"] == "llm_circuit_open:fallback" for a in fired)
        # The fallback breaker is read through llm_service's single naming
        # source (was a duplicated "kept in lockstep" derivation).
        from backend.service.llm_service import breaker_name

        assert names and names[-1] == breaker_name(
            "deepseek", "https://fb.test/v1", "fb-model"
        )

    @pytest.mark.asyncio
    async def test_open_judge_breaker_fires(self, monkeypatch) -> None:
        mock_breaker = MagicMock()
        mock_breaker.is_open = True
        monkeypatch.setattr(alerts.config.llm, "judge_provider", "anthropic")
        monkeypatch.setattr(alerts.config.llm, "judge_model", "claude-judge")
        monkeypatch.setattr(alerts.config.llm, "judge_api_key", "k")
        monkeypatch.setattr(alerts.config.llm, "fallback_provider", "")

        names: list[str] = []

        def _get_breaker(name: str) -> MagicMock:
            names.append(name)
            return mock_breaker

        monkeypatch.setattr(alerts, "get_circuit_breaker", _get_breaker)

        fired = await alerts.check_alerts()
        assert any(a["key"] == "llm_circuit_open:judge" for a in fired)
        from backend.service.llm_service import breaker_name

        assert names and names[-1] == breaker_name("anthropic", "", "claude-judge")

    def test_breaker_name_follows_llm_service_naming(self) -> None:
        """alerts.py derives breaker names from llm_service's public helper —
        a naming change there propagates to the alert checks automatically."""
        from backend.service.llm_service import breaker_name

        assert breaker_name("anthropic", "", "claude-x") == "llm:anthropic:claude-x"
        assert (
            breaker_name("deepseek", "https://api.test/v1", "m")
            == "llm:openai:https://api.test/v1|m"
        )


class TestObservabilityDegraded:
    """A failing llm_usage DB query must alert, not silently read as 0 rows."""

    @pytest.mark.asyncio
    async def test_db_failure_fires_observability_degraded(self, monkeypatch) -> None:
        async def _boom() -> dict:
            raise RuntimeError("db down")

        monkeypatch.setattr(alerts, "_error_window_stats", _boom)

        fired = await alerts.check_alerts()
        assert any(a["key"] == "observability_degraded" for a in fired)
        assert not any(a["key"] == "llm_error_rate" for a in fired)


class TestFeishuNotification:
    """飞书 push only happens when explicitly opted in (default off)."""

    @pytest.mark.asyncio
    async def test_pushes_when_enabled(self, monkeypatch) -> None:
        monkeypatch.setattr(alerts.config, "alert_feishu_enabled", True)
        monkeypatch.setattr(alerts.config, "feishu_webhook_url", "https://feishu.test/hook")
        mock_breaker = MagicMock()
        mock_breaker.is_open = True
        monkeypatch.setattr(alerts, "get_circuit_breaker", lambda name: mock_breaker)

        class _FakeResp:
            def raise_for_status(self) -> None:
                pass

        class _FakeClient:
            def __init__(self) -> None:
                self.called: tuple | None = None

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args) -> bool:
                return False

            async def post(self, url, json):
                self.called = (url, json)
                return _FakeResp()

        fake = _FakeClient()
        monkeypatch.setattr("httpx.AsyncClient", lambda **kwargs: fake)

        fired = await alerts.check_alerts()
        assert any(a["key"] == "llm_circuit_open" for a in fired)
        assert fake.called is not None
        assert fake.called[0] == "https://feishu.test/hook"
        assert "llm_circuit_open" in fake.called[1]["content"]["text"]

    @pytest.mark.asyncio
    async def test_no_push_when_disabled(self, monkeypatch) -> None:
        monkeypatch.setattr(alerts.config, "alert_feishu_enabled", False)
        monkeypatch.setattr(alerts.config, "feishu_webhook_url", "https://feishu.test/hook")
        mock_breaker = MagicMock()
        mock_breaker.is_open = True
        monkeypatch.setattr(alerts, "get_circuit_breaker", lambda name: mock_breaker)

        with patch("httpx.AsyncClient") as mock_client:
            await alerts.check_alerts()
        mock_client.assert_not_called()
