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

    def test_rerank_concurrency_below_one(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.llm, "rerank_concurrency", 0)

        problems = config_mod.validate_config()
        assert any("LLM_RERANK_CONCURRENCY" in p for p in problems)

    def test_fallback_requires_model_and_key(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.llm, "fallback_provider", "deepseek")
        monkeypatch.setattr(config_mod.config.llm, "fallback_model", "")
        monkeypatch.setattr(config_mod.config.llm, "fallback_api_key", "")
        monkeypatch.setattr(config_mod.config.llm, "fallback_base_url", "")

        problems = config_mod.validate_config()
        assert any("LLM_FALLBACK_MODEL" in p for p in problems)
        assert any("LLM_FALLBACK_API_KEY" in p for p in problems)
        assert any("LLM_FALLBACK_BASE_URL" in p for p in problems)

    def test_fallback_anthropic_needs_no_base_url(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.llm, "fallback_provider", "anthropic")
        monkeypatch.setattr(config_mod.config.llm, "fallback_model", "claude-haiku")
        monkeypatch.setattr(config_mod.config.llm, "fallback_api_key", "k")
        monkeypatch.setattr(config_mod.config.llm, "fallback_base_url", "")

        assert config_mod.validate_config() == []

    def test_usage_sample_rate_out_of_range(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config, "usage_sample_rate", 1.5)

        problems = config_mod.validate_config()
        assert any("USAGE_SAMPLE_RATE" in p for p in problems)

    def test_usage_sample_retention_below_one(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config, "usage_sample_retention_days", 0)

        problems = config_mod.validate_config()
        assert any("USAGE_SAMPLE_RETENTION_DAYS" in p for p in problems)

    def test_alert_error_rate_threshold_out_of_range(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config, "alert_error_rate_threshold", -0.1)

        problems = config_mod.validate_config()
        assert any("ALERT_ERROR_RATE_THRESHOLD" in p for p in problems)

    def test_alert_check_interval_below_one(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config, "alert_check_interval_seconds", 0)

        problems = config_mod.validate_config()
        assert any("ALERT_CHECK_INTERVAL_SECONDS" in p for p in problems)

    def test_agent_concurrency_below_one(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config, "max_agent_concurrency", 0)

        problems = config_mod.validate_config()
        assert any("MAX_AGENT_CONCURRENCY" in p for p in problems)

    def test_embedding_fallback_requires_model(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.embedding, "fallback_provider", "openai")
        monkeypatch.setattr(config_mod.config.embedding, "fallback_model", "")

        problems = config_mod.validate_config()
        assert any("EMBEDDING_FALLBACK_MODEL" in p for p in problems)

    def test_embedding_openai_fallback_requires_key(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.embedding, "fallback_provider", "openai")
        monkeypatch.setattr(config_mod.config.embedding, "fallback_model", "text-embedding-3-small")
        monkeypatch.setattr(config_mod.config.embedding, "fallback_api_key", "")

        problems = config_mod.validate_config()
        assert any("EMBEDDING_FALLBACK_API_KEY" in p for p in problems)

    def test_embedding_fallback_complete_is_valid(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.embedding, "fallback_provider", "openai")
        monkeypatch.setattr(config_mod.config.embedding, "fallback_model", "text-embedding-3-small")
        monkeypatch.setattr(config_mod.config.embedding, "fallback_api_key", "k")

        assert config_mod.validate_config() == []
