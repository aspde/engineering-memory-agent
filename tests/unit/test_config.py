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

    def test_patrol_max_tokens_must_be_positive(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config, "patrol_max_tokens", 0)

        problems = config_mod.validate_config()
        assert any("PATROL_MAX_TOKENS" in p for p in problems)

    def test_repo_allow_root_must_exist(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(
            config_mod.config, "repo_allow_roots", ("/nonexistent/root",)
        )

        problems = config_mod.validate_config()
        assert any("REPO_ALLOW_ROOT" in p for p in problems)

    def test_repo_allow_root_valid_dir_passes(self, monkeypatch, tmp_path) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config, "repo_allow_roots", (str(tmp_path),))

        assert config_mod.validate_config() == []

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

    def test_judge_requires_model_key_and_base_url(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.llm, "judge_provider", "openai")
        monkeypatch.setattr(config_mod.config.llm, "judge_model", "")
        monkeypatch.setattr(config_mod.config.llm, "judge_api_key", "")
        monkeypatch.setattr(config_mod.config.llm, "judge_base_url", "")

        problems = config_mod.validate_config()
        assert any("LLM_JUDGE_MODEL" in p for p in problems)
        assert any("LLM_JUDGE_API_KEY" in p for p in problems)
        assert any("LLM_JUDGE_BASE_URL" in p for p in problems)

    def test_judge_anthropic_needs_no_base_url(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.llm, "judge_provider", "anthropic")
        monkeypatch.setattr(config_mod.config.llm, "judge_model", "claude-haiku")
        monkeypatch.setattr(config_mod.config.llm, "judge_api_key", "k")
        monkeypatch.setattr(config_mod.config.llm, "judge_base_url", "")

        assert config_mod.validate_config() == []

    def test_judge_unset_is_valid(self, monkeypatch) -> None:
        """No judge config at all is valid — the judge falls back to primary."""
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.llm, "judge_provider", "")
        monkeypatch.setattr(config_mod.config.llm, "judge_model", "")
        monkeypatch.setattr(config_mod.config.llm, "judge_api_key", "")
        monkeypatch.setattr(config_mod.config.llm, "judge_base_url", "")

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

    def test_usage_flush_interval_below_one(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config, "usage_flush_interval_seconds", 0)

        problems = config_mod.validate_config()
        assert any("USAGE_FLUSH_INTERVAL_SECONDS" in p for p in problems)

    def test_usage_buffer_max_below_one(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config, "usage_buffer_max", 0)

        problems = config_mod.validate_config()
        assert any("USAGE_BUFFER_MAX" in p for p in problems)

    def test_temperature_out_of_range(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.llm, "temperature", 2.5)

        problems = config_mod.validate_config()
        assert any("LLM_TEMPERATURE" in p for p in problems)

    def test_structured_backoff_negative(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.llm, "structured_backoff", -0.1)

        problems = config_mod.validate_config()
        assert any("LLM_STRUCTURED_BACKOFF" in p for p in problems)

    def test_backoff_base_may_exceed_max(self, monkeypatch) -> None:
        """base > max is not a config error — tenacity caps every wait at
        backoff_max, so a larger base just means a flat max backoff."""
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.resilience, "backoff_base", 10.0)
        monkeypatch.setattr(config_mod.config.resilience, "backoff_max", 5.0)

        assert config_mod.validate_config() == []

    def test_structured_temperature_out_of_range(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.llm, "structured_temperature", 2.5)

        problems = config_mod.validate_config()
        assert any("LLM_STRUCTURED_TEMPERATURE" in p for p in problems)

    def test_llm_provider_enum_invalid(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.llm, "provider", "deepseekk")

        problems = config_mod.validate_config()
        assert any("LLM_PROVIDER" in p for p in problems)

    def test_fallback_provider_enum_invalid(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.llm, "fallback_provider", "gpt")
        monkeypatch.setattr(config_mod.config.llm, "fallback_model", "m")
        monkeypatch.setattr(config_mod.config.llm, "fallback_api_key", "k")

        problems = config_mod.validate_config()
        assert any("LLM_FALLBACK_PROVIDER" in p for p in problems)

    def test_judge_provider_enum_invalid(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.llm, "judge_provider", "gemini")
        monkeypatch.setattr(config_mod.config.llm, "judge_model", "m")
        monkeypatch.setattr(config_mod.config.llm, "judge_api_key", "k")

        problems = config_mod.validate_config()
        assert any("LLM_JUDGE_PROVIDER" in p for p in problems)

    def test_embedding_provider_enum_invalid(self, monkeypatch) -> None:
        _valid_patrol(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")
        monkeypatch.setattr(config_mod.config.embedding, "provider", "voyage")

        problems = config_mod.validate_config()
        assert any("EMBEDDING_PROVIDER" in p for p in problems)


class TestBreadthLayerFlags:
    """ADR-011: breadth layers (connectors/webhooks, scenarios, patrol) are
    off by default and only active when explicitly enabled or in APP_ENV=test.
    """

    def _reset_flags(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pin the three raw flags to their off state, isolating from .env drift."""
        monkeypatch.setattr(config_mod.config, "connectors_enabled", False)
        monkeypatch.setattr(config_mod.config, "scenarios_enabled", False)
        monkeypatch.setattr(config_mod.config, "patrol_enabled", False)

    def test_all_off_in_development(self, monkeypatch) -> None:
        self._reset_flags(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "development")

        assert config_mod.config.connectors_active is False
        assert config_mod.config.scenarios_active is False
        assert config_mod.config.patrol_active is False

    def test_all_active_in_test_env(self, monkeypatch) -> None:
        self._reset_flags(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "test")

        assert config_mod.config.connectors_active is True
        assert config_mod.config.scenarios_active is True
        assert config_mod.config.patrol_active is True

    def test_explicit_enable_in_development(self, monkeypatch) -> None:
        self._reset_flags(monkeypatch)
        monkeypatch.setattr(config_mod.config, "app_env", "development")
        monkeypatch.setattr(config_mod.config, "scenarios_enabled", True)

        assert config_mod.config.scenarios_active is True
        assert config_mod.config.connectors_active is False
