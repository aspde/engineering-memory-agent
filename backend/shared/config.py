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

# Context variable for the current trace — one id per agent run (a chat
# request, a patrol, …).  The LLM provider layer reads it to stamp every
# usage row / structured log line with the run it belonged to, so a trace
# can be replayed end-to-end.  Empty for background tasks that don't set
# one (connectors, webhook handlers) — their rows still carry scenario.
current_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "current_trace_id", default=""
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
    # Default temperature for structured-output calls (chat_structured) —
    # extraction / conflict judgements should be deterministic, so this is
    # low rather than the general LLM_TEMPERATURE.  Callers override per-call
    # by passing ``temperature=`` in kwargs (chat_structured forwards them).
    structured_temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_STRUCTURED_TEMPERATURE", "0.1"))
    )
    # LLM-based rerank (rerank_llm) sends one provider call per candidate;
    # this caps concurrent in-flight calls so a 20-40 candidate list can't
    # self-inflict a rate-limit storm on the provider.
    rerank_concurrency: int = field(
        default_factory=lambda: int(os.getenv("LLM_RERANK_CONCURRENCY", "4"))
    )
    # Anthropic prompt caching: adds cache_control breakpoints on the system
    # block and the last tool schema so repeated agent turns reuse a cached
    # prefix.  OpenAI-compatible providers (OpenAI/DeepSeek) cache repeated
    # input prefixes automatically server-side — no API parameter involved.
    prompt_caching_enabled: bool = field(
        default_factory=lambda: os.getenv("PROMPT_CACHING_ENABLED", "true").lower() == "true"
    )
    # Optional cross-provider failover: when ``LLM_FALLBACK_PROVIDER`` is set
    # and the primary's call fails with a retryable error (or its circuit
    # breaker is open), the call is retried once against the fallback.  All
    # ``LLM_FALLBACK_*`` fields are required when ``LLM_FALLBACK_PROVIDER`` is
    # non-empty (see ``get_llm_provider`` in ``llm_service.py``).
    fallback_provider: str = field(
        default_factory=lambda: os.getenv("LLM_FALLBACK_PROVIDER", "")
    )
    fallback_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_FALLBACK_API_KEY", "")
    )
    fallback_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_FALLBACK_BASE_URL", "")
    )
    fallback_model: str = field(
        default_factory=lambda: os.getenv("LLM_FALLBACK_MODEL", "")
    )
    fallback_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_FALLBACK_MAX_TOKENS", "4096"))
    )
    fallback_timeout: int = field(
        default_factory=lambda: int(os.getenv("LLM_FALLBACK_TIMEOUT", "60"))
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
    # When enabled, generate_final_node automatically extracts substantive
    # user turns into memories (unless the agent already wrote this turn).
    # Enabled by default — auto memory is the intended steady-state; set
    # AUTO_MEMORY_ENABLED=false to opt out.
    auto_memory_enabled: bool = field(
        default_factory=lambda: os.getenv("AUTO_MEMORY_ENABLED", "true").lower() == "true"
    )
    # Auto-memory frequency control — each capture costs 3 LLM extractions
    # (summary + entities + relations) plus embedding and a similarity scan,
    # so capture is throttled: a per-thread minimum interval, a per-thread
    # lifetime cap, and a process-wide rolling-window cap (see the throttle
    # helpers in ``agent.nodes``).
    auto_memory_min_interval: int = field(
        default_factory=lambda: int(os.getenv("AUTO_MEMORY_MIN_INTERVAL", "60"))
    )
    auto_memory_max_per_thread: int = field(
        default_factory=lambda: int(os.getenv("AUTO_MEMORY_MAX_PER_THREAD", "10"))
    )
    auto_memory_max_per_window: int = field(
        default_factory=lambda: int(os.getenv("AUTO_MEMORY_MAX_PER_WINDOW", "30"))
    )
    # When enabled, messages older than the context window are folded into a
    # running-summary SystemMessage instead of being dropped.  Enabled by
    # default — long threads are summarised rather than truncated; set
    # CONVERSATION_COMPACTION_ENABLED=false to restore plain truncation.
    conversation_compaction_enabled: bool = field(
        default_factory=lambda: os.getenv(
            "CONVERSATION_COMPACTION_ENABLED", "true"
        ).lower() == "true"
    )
    # Target token budget for the agent context window (conversation history +
    # tool results) sent to the LLM each turn.  The windowing/compaction logic
    # in ``agent.nodes`` retains history under this budget using a rough
    # CJK-aware token estimate — an approximate ceiling, not an exact one.
    context_token_budget: int = field(
        default_factory=lambda: int(os.getenv("CONTEXT_TOKEN_BUDGET", "12000"))
    )
    # ── LLM usage tracing / cost persistence ────────────────────────
    # Every LLM call is recorded (in-memory buffer → batch INSERT into the
    # ``llm_usage`` table by a background flusher).  ``usage_enabled=false``
    # disables persistence entirely (the buffer stops recording); the
    # in-memory counters behind ``/api/agent/usage`` are unaffected.
    usage_enabled: bool = field(
        default_factory=lambda: os.getenv("USAGE_ENABLED", "true").lower() == "true"
    )
    # How often the flusher drains the recording buffer into the DB.
    usage_flush_interval_seconds: int = field(
        default_factory=lambda: int(os.getenv("USAGE_FLUSH_INTERVAL_SECONDS", "10"))
    )
    # Buffer cap — on overflow the oldest rows are dropped with a warning
    # (observability must never back-pressure the LLM hot path).
    usage_buffer_max: int = field(
        default_factory=lambda: int(os.getenv("USAGE_BUFFER_MAX", "5000"))
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
    if config.context_token_budget < 1:
        problems.append(
            f"CONTEXT_TOKEN_BUDGET={config.context_token_budget} must be >= 1"
        )

    # LLM failover: an incomplete fallback config only fails on the first
    # request, so validate it here instead (same fail-fast policy as the
    # LLM_API_KEY check below).
    if config.llm.fallback_provider:
        if not config.llm.fallback_model:
            problems.append(
                "LLM_FALLBACK_MODEL is required when LLM_FALLBACK_PROVIDER is set"
            )
        if not config.llm.fallback_api_key:
            problems.append(
                "LLM_FALLBACK_API_KEY is required when LLM_FALLBACK_PROVIDER is set"
            )
        if (
            config.llm.fallback_provider != "anthropic"
            and not config.llm.fallback_base_url
        ):
            problems.append(
                "LLM_FALLBACK_BASE_URL is required when LLM_FALLBACK_PROVIDER "
                "is an OpenAI-compatible provider"
            )

    # LLM API key — real providers all need one; only tests are exempt.
    if config.app_env != "test" and not config.llm.api_key:
        problems.append(
            "LLM_API_KEY is empty — set it in .env (or run tests with APP_ENV=test)"
        )

    return problems
