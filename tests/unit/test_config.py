"""Tests for startup configuration validation (``validate_config``)."""

from __future__ import annotations

import pytest

from backend.shared import config as config_mod


def _valid_patrol(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the schedule fields to legal values (isolate from .env drift)."""
    cfg = config_mod.config
    monkeypatch.setattr(cfg, "patrol_daily_hour", 8)
    monkeypatch.setattr(cfg, "patrol_weekly_day", 1)
    monkeypatch.setattr(cfg, "patrol_weekly_hour", 9)


class TestValidateConfig:
    def test_default_config_is_valid(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.llm, "api_key", "")
        assert config_mod.validate_config() == []

    def test_patrol_hour_out_of_range(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config, "patrol_daily_hour", 25)

        problems = config_mod.validate_config()
        assert any("PATROL_DAILY_HOUR" in p for p in problems)

    def test_patrol_weekly_day_out_of_range(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config, "patrol_weekly_day", 7)

        problems = config_mod.validate_config()
        assert any("PATROL_WEEKLY_DAY" in p for p in problems)

    def test_llm_api_key_required_outside_test(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "development")
        monkeypatch.setattr(config_mod.config.llm, "api_key", "")

        problems = config_mod.validate_config()
        assert any("LLM_API_KEY" in p for p in problems)

    def test_llm_api_key_exempt_in_test_env(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.llm, "api_key", "")

        assert config_mod.validate_config() == []

    def test_retry_max_attempts_below_one(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.resilience, "max_attempts", 0)

        problems = config_mod.validate_config()
        assert any("LLM_RETRY_MAX_ATTEMPTS" in p for p in problems)

    def test_agent_timeout_must_be_positive(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config, "agent_timeout", 0)

        problems = config_mod.validate_config()
        assert any("AGENT_TIMEOUT" in p for p in problems)
