"""Unified configuration from environment variables."""

from __future__ import annotations

import contextvars
import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# Context variable to pass the current conversation thread_id through
# the agent → tool → service call chain without threading it through
# every function signature.  Set by the API layer before invoking the
# agent graph, read by the memory-write path to tag new memories with
# their originating conversation.
current_thread_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_thread_id", default=""
)


# Known embedding-vector dimensions by model name — the single source of
# truth for both the DB schema (which pgvector dimension to create) and the
# embedding providers (OpenAI has no way to introspect its dimension).
# Unknown models default to 1536 with a loud warning (see the embedding
# provider in ``embedding_service.py``); the local BGE provider reports its
# actual dimension at runtime, so a wrong guess surfaces as a mismatch
# warning in ``get_embedding_provider()`` rather than a silent write failure.
EMBEDDING_DIMENSIONS: dict[str, int] = {
    "BAAI/bge-m3": 1024,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


@dataclass
class EmbeddingConfig:
    provider: str = field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "local"))
    api_key: str = field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("EMBEDDING_BASE_URL", ""))
    model: str = field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"))
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    )
    timeout: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_TIMEOUT", "60"))
    )
    normalize: bool = field(
        default_factory=lambda: os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true"
    )
    hf_endpoint: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_HF_ENDPOINT", "https://hf-mirror.com")
    )

    @property
    def dimension(self) -> int:
        """Embedding dimension used to build the pgvector schema."""
        return EMBEDDING_DIMENSIONS.get(self.model, 1536)


@dataclass
class LLMConfig:
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "deepseek"))
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "https://api.deepseek.com"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "deepseek-v4-pro"))
    temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "4096"))
    )
    timeout: int = field(
        default_factory=lambda: int(os.getenv("LLM_TIMEOUT", "60"))
    )
    # Structured-output calls (chat_structured) retry this many times with
    # linear backoff before raising LLMStructuredError.
    structured_max_attempts: int = field(
        default_factory=lambda: int(os.getenv("LLM_STRUCTURED_MAX_ATTEMPTS", "3"))
    )
    structured_backoff: float = field(
        default_factory=lambda: float(os.getenv("LLM_STRUCTURED_BACKOFF", "0.5"))
    )
    # LLM-based rerank (rerank_llm) sends one provider call per candidate;
    # this caps concurrent in-flight calls so a 20-40 candidate list can't
    # self-inflict a rate-limit storm on the provider.
    rerank_concurrency: int = field(
        default_factory=lambda: int(os.getenv("LLM_RERANK_CONCURRENCY", "4"))
    )


@dataclass
class ResilienceConfig:
    """Transport retry + circuit breaker settings for external AI providers.

    Shared by LLM and embedding provider calls (see
    ``backend/shared/resilience.py``).  ``max_attempts`` counts the total
    attempts including the first, so 3 means one initial try + two retries.
    """

    max_attempts: int = field(
        default_factory=lambda: int(os.getenv("LLM_RETRY_MAX_ATTEMPTS", "3"))
    )
    backoff_base: float = field(
        default_factory=lambda: float(os.getenv("LLM_RETRY_BACKOFF_BASE", "1.0"))
    )
    backoff_max: float = field(
        default_factory=lambda: float(os.getenv("LLM_RETRY_BACKOFF_MAX", "8.0"))
    )
    circuit_breaker_threshold: int = field(
        default_factory=lambda: int(os.getenv("LLM_CIRCUIT_BREAKER_THRESHOLD", "5"))
    )
    circuit_breaker_cooldown: float = field(
        default_factory=lambda: float(os.getenv("LLM_CIRCUIT_BREAKER_COOLDOWN", "30.0"))
    )


@dataclass
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)
    database_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "postgresql://ema:ema123@localhost:5432/ema_dev")
    )
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    max_agent_steps: int = field(
        default_factory=lambda: int(os.getenv("MAX_AGENT_STEPS", "5"))
    )
    # Per-turn total deadline (seconds) for the whole ReAct run — one request
    # through to the final answer.  Guards against a slow provider / many
    # tool steps hanging the request far beyond max_agent_steps × LLM_TIMEOUT.
    agent_timeout: int = field(
        default_factory=lambda: int(os.getenv("AGENT_TIMEOUT", "180"))
    )
    # Patrol runs are background tasks that scan the whole memory store, so
    # they get their own (longer) deadline rather than borrowing the
    # interactive AGENT_TIMEOUT.  A patrol that exceeds it is marked failed.
    patrol_timeout: int = field(
        default_factory=lambda: int(os.getenv("PATROL_TIMEOUT", "600"))
    )
    memory_enabled: bool = field(
        default_factory=lambda: os.getenv("MEMORY_ENABLED", "true").lower() == "true"
    )
    # ── Phase 3: proactive agent ───────────────────────────────────
    patrol_enabled: bool = field(
        default_factory=lambda: os.getenv("PATROL_ENABLED", "true").lower() == "true"
    )
    patrol_daily_hour: int = field(
        default_factory=lambda: int(os.getenv("PATROL_DAILY_HOUR", "8"))
    )
    patrol_weekly_enabled: bool = field(
        default_factory=lambda: os.getenv("PATROL_WEEKLY_ENABLED", "true").lower() == "true"
    )
    patrol_weekly_day: int = field(
        default_factory=lambda: int(os.getenv("PATROL_WEEKLY_DAY", "1"))
    )
    patrol_weekly_hour: int = field(
        default_factory=lambda: int(os.getenv("PATROL_WEEKLY_HOUR", "9"))
    )
    feishu_webhook_url: str = field(
        default_factory=lambda: os.getenv("FEISHU_WEBHOOK_URL", "")
    )


config = AppConfig()


class ConfigError(RuntimeError):
    """Raised when :func:`validate_config` finds invalid configuration."""


def validate_config() -> list[str]:
    """Return a list of config problems, empty when the config is valid.

    Startup validation (see ``backend/main.py`` lifespan): catches mistakes
    that otherwise surface late — a bogus ``PATROL_DAILY_HOUR=25`` silently
    producing an invalid scheduler schedule, or a missing ``LLM_API_KEY``
    failing only on the first request.  Structural checks (ranges, positive
    bounds) always run; the API-key check is skipped when ``APP_ENV=test``
    (tests use fake providers).
    """
    problems: list[str] = []

    # Patrol schedule ranges — out-of-range hours crash scheduler.next_run
    # (datetime.replace) only when the loop is already running.
    for env, value in (
        ("PATROL_DAILY_HOUR", config.patrol_daily_hour),
        ("PATROL_WEEKLY_HOUR", config.patrol_weekly_hour),
    ):
        if not 0 <= value <= 23:
            problems.append(f"{env}={value} must be in 0..23")
    if not 0 <= config.patrol_weekly_day <= 6:
        problems.append(
            f"PATROL_WEEKLY_DAY={config.patrol_weekly_day} must be in 0..6 (0=Mon)"
        )

    # Positive / non-negative bounds for resilience & structured retry params.
    if config.resilience.max_attempts < 1:
        problems.append(f"LLM_RETRY_MAX_ATTEMPTS={config.resilience.max_attempts} must be >= 1")
    if config.resilience.backoff_base < 0:
        problems.append(f"LLM_RETRY_BACKOFF_BASE={config.resilience.backoff_base} must be >= 0")
    if config.resilience.backoff_max < 0:
        problems.append(f"LLM_RETRY_BACKOFF_MAX={config.resilience.backoff_max} must be >= 0")
    if config.resilience.circuit_breaker_threshold < 1:
        problems.append(
            f"LLM_CIRCUIT_BREAKER_THRESHOLD={config.resilience.circuit_breaker_threshold} must be >= 1"
        )
    if config.resilience.circuit_breaker_cooldown <= 0:
        problems.append(
            f"LLM_CIRCUIT_BREAKER_COOLDOWN={config.resilience.circuit_breaker_cooldown} must be > 0"
        )
    if config.llm.structured_max_attempts < 1:
        problems.append(
            f"LLM_STRUCTURED_MAX_ATTEMPTS={config.llm.structured_max_attempts} must be >= 1"
        )
    if config.llm.rerank_concurrency < 1:
        problems.append(
            f"LLM_RERANK_CONCURRENCY={config.llm.rerank_concurrency} must be >= 1"
        )
    if config.agent_timeout <= 0:
        problems.append(f"AGENT_TIMEOUT={config.agent_timeout} must be > 0")
    if config.patrol_timeout <= 0:
        problems.append(f"PATROL_TIMEOUT={config.patrol_timeout} must be > 0")

    # LLM API key — real providers all need one; only tests are exempt.
    if config.app_env != "test" and not config.llm.api_key:
        problems.append(
            "LLM_API_KEY is empty — set it in .env (or run tests with APP_ENV=test)"
        )

    return problems
