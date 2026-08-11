"""Unified configuration from environment variables."""

from __future__ import annotations

import contextvars
import os
from dataclasses import dataclass, field
from pathlib import Path

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
    # CPU embedding 并发控制（local provider）。BGE-M3 推理是 CPU 密集，
    # torch/OpenMP 默认抢占全部核；不限制时并发请求的 embed 同时涌入互相
    # 抢核，造成延迟长尾（冷路径压测实测 10 并发 P95 19s，单条 embed
    # 366→1120ms）。``max_concurrency`` 限制同时进行的推理任务数，
    # ``torch_threads`` 限制单任务的内部线程数——两者乘积约等于核数时
    # 零超卖。调整后需重跑冷压测验证 P95（tests/perf/locustfile.py 默认
    # 冷池）。
    max_concurrency: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_MAX_CONCURRENCY", "2"))
    )
    torch_threads: int = field(
        default_factory=lambda: int(
            os.getenv("EMBEDDING_TORCH_THREADS", str(max(1, (os.cpu_count() or 4) // 2)))
        )
    )
    # Optional cross-provider failover: when ``EMBEDDING_FALLBACK_PROVIDER`` is
    # set and the primary's call fails (a retryable error after its retries,
    # an open circuit breaker, or a local-model failure such as a corrupt BGE
    # checkpoint), the call is retried once against the fallback.  The fallback
    # model's vector dimension must match the primary's (== the pgvector schema
    # dimension) or failover writes will be rejected — see
    # ``FallbackEmbeddingProvider`` in ``embedding_service.py``.
    fallback_provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_FALLBACK_PROVIDER", "")
    )
    fallback_api_key: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_FALLBACK_API_KEY", "")
    )
    fallback_base_url: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_FALLBACK_BASE_URL", "")
    )
    fallback_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_FALLBACK_MODEL", "")
    )
    fallback_batch_size: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_FALLBACK_BATCH_SIZE", "32"))
    )
    fallback_timeout: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_FALLBACK_TIMEOUT", "60"))
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
    # Dedicated judge provider for the LLM behavior eval (tests/eval).  When
    # set, the eval's LLM-as-judge runs on this independent provider so the
    # verdict comes from a different model than the one being evaluated —
    # avoiding the self-preference bias of same-model judging.  All
    # ``LLM_JUDGE_*`` fields empty means the judge falls back to the primary
    # provider (get_judge_provider in llm_service.py) — behaviour unchanged
    # for setups without a second provider.
    judge_provider: str = field(
        default_factory=lambda: os.getenv("LLM_JUDGE_PROVIDER", "")
    )
    judge_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_JUDGE_API_KEY", "")
    )
    judge_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_JUDGE_BASE_URL", "")
    )
    judge_model: str = field(
        default_factory=lambda: os.getenv("LLM_JUDGE_MODEL", "")
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
    # Max simultaneous interactive agent runs (chat / chat-stream).  Each run
    # holds LLM slots for its whole ReAct loop (up to AGENT_TIMEOUT), so an
    # unbounded number of concurrent sessions would together saturate the
    # provider rate limit and trip the circuit breaker for everyone.  Beyond
    # this cap the chat endpoint answers 503 (refuse, not queue) — see
    # ``agent_service``'s slot counter.
    max_agent_concurrency: int = field(
        default_factory=lambda: int(os.getenv("MAX_AGENT_CONCURRENCY", "4"))
    )
    # Patrol runs are background tasks that scan the whole memory store, so
    # they get their own (longer) deadline rather than borrowing the
    # interactive AGENT_TIMEOUT.  A patrol that exceeds it is marked failed.
    patrol_timeout: int = field(
        default_factory=lambda: int(os.getenv("PATROL_TIMEOUT", "600"))
    )
    # Scenario runs invoke the full agent (recursion_limit=50) for their whole
    # compose chain, so a stuck scenario can hang the request indefinitely.
    # This bounds each run — beyond SCENARIO_TIMEOUT_SECONDS the scenario
    # endpoint answers 504 (see scenario_routes) instead of leaving the
    # request open forever.
    scenario_timeout: int = field(
        default_factory=lambda: int(os.getenv("SCENARIO_TIMEOUT_SECONDS", "300"))
    )
    # Max simultaneous scenario runs.  Each run holds agent/LLM slots for its
    # whole compose chain (up to SCENARIO_TIMEOUT_SECONDS), so an unbounded
    # number of concurrent scenarios would together saturate the provider rate
    # limit and can only be stopped by a restart.  Beyond this cap the scenario
    # endpoint answers 503 (refuse, not queue) — see scenario_routes' slot
    # counter in ``backend.service.scenarios``.
    max_scenario_concurrency: int = field(
        default_factory=lambda: int(os.getenv("MAX_SCENARIO_CONCURRENCY", "2"))
    )
    memory_enabled: bool = field(
        default_factory=lambda: os.getenv("MEMORY_ENABLED", "true").lower() == "true"
    )
    # Git repository ingestion allow-list — only repositories under one of
    # these roots may be read by ``ingest_git_repo_tool`` (comma-separated
    # ``REPO_ALLOW_ROOT``).  Empty (the default) fails closed: every ingest is
    # rejected with a clear message until at least one root is configured.
    # This is the sandbox that keeps a prompt-injected tool call — or a
    # mis-guided agent — from reading an arbitrary local git repository.  The
    # tool is approval-gated, but the HITL gate must not be the only defense.
    repo_allow_roots: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            p.strip()
            for p in os.getenv("REPO_ALLOW_ROOT", "").split(",")
            if p.strip()
        )
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
    # so capture is throttled by a per-thread minimum interval.  The per-thread
    # lifetime cap and process-wide window cap this once had were cut: the
    # interval alone bounds the steady-state rate, and ``write_memory``'s
    # content-hash idempotency already makes exact repeats a no-op (see
    # ``backend.agent.nodes``).
    auto_memory_min_interval: int = field(
        default_factory=lambda: int(os.getenv("AUTO_MEMORY_MIN_INTERVAL", "60"))
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
    # in ``backend.agent.nodes`` retains history under this budget using a rough
    # CJK-aware token estimate — an approximate ceiling, not an exact one.
    context_token_budget: int = field(
        default_factory=lambda: int(os.getenv("CONTEXT_TOKEN_BUDGET", "12000"))
    )
    # ── LLM usage tracing / cost persistence ────────────────────────
    # Every LLM call is recorded (in-memory buffer → batch INSERT into the
    # ``llm_usage`` table by a background flusher).  ``usage_enabled=false``
    # disables persistence entirely (the buffer stops recording).
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
    # Fraction of successful LLM calls whose prompt/response text is sampled
    # into llm_usage (for post-hoc quality analysis).  Error calls are always
    # sampled regardless of this rate.  0 disables success-path sampling.
    usage_sample_rate: float = field(
        default_factory=lambda: float(os.getenv("USAGE_SAMPLE_RATE", "0.05"))
    )
    # Sampled prompt/response text is kept this many days, then nulled by the
    # usage flusher (metadata rows stay for summaries; only the text columns
    # are released).  Bounds the sample columns so they don't accumulate
    # unboundedly on a long-running deployment.
    usage_sample_retention_days: int = field(
        default_factory=lambda: int(os.getenv("USAGE_SAMPLE_RETENTION_DAYS", "30"))
    )
    # ── Runtime health metrics (Prometheus) ──────────────────────────
    # In-memory time series (request latency / status, LLM call count /
    # latency / tokens, circuit-breaker state, agent concurrency, ReAct step
    # distribution) exposed at ``GET /metrics`` for Prometheus scraping.
    # Independent of ``usage_enabled`` — this is process-local health
    # observability, not the persisted cost rows.  Disabling it drops the
    # /metrics endpoint's data without touching the llm_usage pipeline.
    metrics_enabled: bool = field(
        default_factory=lambda: os.getenv("METRICS_ENABLED", "true").lower() == "true"
    )
    # ── API rate limiting (per-key token bucket) ─────────────────────
    # Caps request volume per API key so an unbounded / abusive caller can't
    # burn unlimited LLM tokens (each chat round costs real tokens).  Two
    # tiers: ``chat`` (agent chat + scenario runs, LLM-dense) and ``general``
    # (every other /api route).  Buckets are process-local — same single-
    # instance assumption as the circuit breakers / agent concurrency counter
    # (see deployment.md 单实例部署约束).  See backend/api/ratelimit.py.
    rate_limit_enabled: bool = field(
        default_factory=lambda: os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    )
    rate_limit_chat_requests: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_CHAT_REQUESTS", "30"))
    )
    rate_limit_chat_window_seconds: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_CHAT_WINDOW_SECONDS", "60"))
    )
    rate_limit_general_requests: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_GENERAL_REQUESTS", "120"))
    )
    rate_limit_general_window_seconds: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_GENERAL_WINDOW_SECONDS", "60"))
    )
    # ── Phase 2/3 breadth layers — off by default (ADR-011) ────────
    # Connectors/webhooks, vertical scenarios and the patrol scheduler all
    # serve data sources EMA does not have yet (the production store is
    # empty); their API routes and background work are compiled into the
    # process but only activated when the corresponding flag is set.
    # ``*_active`` folds in the ``APP_ENV=test`` exemption so the API suite
    # keeps exercising the breadth routes (same convention as auth / rate
    # limiting — see backend/api/router.py for the route mounting).
    connectors_enabled: bool = field(
        default_factory=lambda: os.getenv("CONNECTORS_ENABLED", "false").lower() == "true"
    )
    scenarios_enabled: bool = field(
        default_factory=lambda: os.getenv("SCENARIOS_ENABLED", "false").lower() == "true"
    )
    # ── Phase 3: proactive agent ───────────────────────────────────
    patrol_enabled: bool = field(
        default_factory=lambda: os.getenv("PATROL_ENABLED", "false").lower() == "true"
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
    # Output ceiling for the patrol *final synthesis* call (and the repair
    # retry), independent of the interactive LLM_MAX_TOKENS.  A patrol report
    # is generated in one LLM call and must arrive as complete JSON; once the
    # model spends part of its 4096-token budget on internal reasoning the
    # final message gets truncated and the run fails contract validation
    # (see ``_validate_findings`` in patrol.py).  PATROL_MAX_TOKENS gives the
    # report headroom; the daily/weekly prompts cap per-category entry counts
    # so the report stays small enough to fit.
    patrol_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("PATROL_MAX_TOKENS", "8000"))
    )
    feishu_webhook_url: str = field(
        default_factory=lambda: os.getenv("FEISHU_WEBHOOK_URL", "")
    )

    # ── Breadth-layer activation (route mounting / scheduler start) ──
    # A layer is *active* when explicitly enabled or when running the API
    # test suite (``APP_ENV=test``) — tests exercise every router regardless
    # of the flags.  Production default: all three off (ADR-011).
    @property
    def connectors_active(self) -> bool:
        return self.app_env == "test" or self.connectors_enabled

    @property
    def scenarios_active(self) -> bool:
        return self.app_env == "test" or self.scenarios_enabled

    @property
    def patrol_active(self) -> bool:
        return self.app_env == "test" or self.patrol_enabled


config = AppConfig()

# Valid provider names — validate_config fails a typo at startup instead of
# on the first call.  LLM providers are the OpenAI-compatible trio (DeepSeek
# is OpenAI-compatible); embeddings are local BGE or an OpenAI-compatible API.
_LLM_PROVIDERS = ("deepseek", "openai", "anthropic")
_EMBEDDING_PROVIDERS = ("local", "openai")


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
    # No ``backoff_base <= backoff_max`` check: tenacity's
    # ``wait_exponential_jitter`` caps every wait at ``backoff_max``
    # (``max(0, min(initial * exp, max))``), so base > max is harmless — it
    # just means every retry already waits the full ``backoff_max``.

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
    if config.llm.structured_backoff < 0:
        problems.append(
            f"LLM_STRUCTURED_BACKOFF={config.llm.structured_backoff} must be >= 0"
        )
    if config.llm.rerank_concurrency < 1:
        problems.append(
            f"LLM_RERANK_CONCURRENCY={config.llm.rerank_concurrency} must be >= 1"
        )
    # General chat temperature is a ratio bounded well above the common 0..1
    # range (some providers accept up to 2); out-of-range silently clamps.
    if not 0 <= config.llm.temperature <= 2:
        problems.append(
            f"LLM_TEMPERATURE={config.llm.temperature} must be in 0..2"
        )
    # Structured calls use their own (lower) temperature, and that same value
    # is what the eval judge provider is built with — so it gets the same
    # 0..2 bound as the general chat temperature.
    if not 0 <= config.llm.structured_temperature <= 2:
        problems.append(
            f"LLM_STRUCTURED_TEMPERATURE={config.llm.structured_temperature} "
            "must be in 0..2"
        )
    if config.agent_timeout <= 0:
        problems.append(f"AGENT_TIMEOUT={config.agent_timeout} must be > 0")
    if config.max_agent_concurrency < 1:
        problems.append(
            f"MAX_AGENT_CONCURRENCY={config.max_agent_concurrency} must be >= 1"
        )
    if config.patrol_timeout <= 0:
        problems.append(f"PATROL_TIMEOUT={config.patrol_timeout} must be > 0")
    if config.patrol_max_tokens < 1:
        problems.append(
            f"PATROL_MAX_TOKENS={config.patrol_max_tokens} must be >= 1"
        )
    if config.scenario_timeout <= 0:
        problems.append(
            f"SCENARIO_TIMEOUT_SECONDS={config.scenario_timeout} must be > 0"
        )
    if config.max_scenario_concurrency < 1:
        problems.append(
            f"MAX_SCENARIO_CONCURRENCY={config.max_scenario_concurrency} must be >= 1"
        )
    if config.context_token_budget < 1:
        problems.append(
            f"CONTEXT_TOKEN_BUDGET={config.context_token_budget} must be >= 1"
        )

    # Rate-limit quotas: a non-positive requests count admits nothing, and a
    # zero window would make ``requests / window`` divide by zero in the
    # limiter's token bucket.
    if config.rate_limit_chat_requests < 1:
        problems.append(
            f"RATE_LIMIT_CHAT_REQUESTS={config.rate_limit_chat_requests} must be >= 1"
        )
    if config.rate_limit_chat_window_seconds < 1:
        problems.append(
            f"RATE_LIMIT_CHAT_WINDOW_SECONDS={config.rate_limit_chat_window_seconds} "
            "must be >= 1"
        )
    if config.rate_limit_general_requests < 1:
        problems.append(
            f"RATE_LIMIT_GENERAL_REQUESTS={config.rate_limit_general_requests} must be >= 1"
        )
    if config.rate_limit_general_window_seconds < 1:
        problems.append(
            f"RATE_LIMIT_GENERAL_WINDOW_SECONDS={config.rate_limit_general_window_seconds} "
            "must be >= 1"
        )

    # Sampling rate and alert thresholds are ratios — out-of-range values
    # would silently disable sampling or trip alerts on every cycle.
    if not 0 <= config.usage_sample_rate <= 1:
        problems.append(
            f"USAGE_SAMPLE_RATE={config.usage_sample_rate} must be in 0..1"
        )
    if config.usage_sample_retention_days < 1:
        problems.append(
            f"USAGE_SAMPLE_RETENTION_DAYS={config.usage_sample_retention_days} "
            "must be >= 1"
        )
    # Flusher cadence and buffer cap: 0 interval spins asyncio.sleep(0) burning
    # CPU in the flusher loop, and a non-positive buffer cap makes record_call
    # pop from an empty buffer.
    if config.usage_flush_interval_seconds < 1:
        problems.append(
            f"USAGE_FLUSH_INTERVAL_SECONDS={config.usage_flush_interval_seconds} "
            "must be >= 1"
        )
    if config.usage_buffer_max < 1:
        problems.append(
            f"USAGE_BUFFER_MAX={config.usage_buffer_max} must be >= 1"
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

    # Embedding failover: same fail-fast policy — an incomplete fallback
    # config only surfaces on the first embedding call otherwise.
    if config.embedding.fallback_provider:
        if not config.embedding.fallback_model:
            problems.append(
                "EMBEDDING_FALLBACK_MODEL is required when "
                "EMBEDDING_FALLBACK_PROVIDER is set"
            )
        if (
            config.embedding.fallback_provider == "openai"
            and not config.embedding.fallback_api_key
        ):
            problems.append(
                "EMBEDDING_FALLBACK_API_KEY is required for an openai "
                "embedding fallback"
            )

    # Judge provider: same fail-fast policy as LLM failover — a half-set
    # ``LLM_JUDGE_*`` config would otherwise only surface on the first eval
    # judge call, silently degrading the eval to self-judging.
    if config.llm.judge_provider:
        if not config.llm.judge_model:
            problems.append(
                "LLM_JUDGE_MODEL is required when LLM_JUDGE_PROVIDER is set"
            )
        if not config.llm.judge_api_key:
            problems.append(
                "LLM_JUDGE_API_KEY is required when LLM_JUDGE_PROVIDER is set"
            )
        if (
            config.llm.judge_provider != "anthropic"
            and not config.llm.judge_base_url
        ):
            problems.append(
                "LLM_JUDGE_BASE_URL is required when LLM_JUDGE_PROVIDER "
                "is an OpenAI-compatible provider"
            )

    # Provider names — a typo (e.g. ``LLM_PROVIDER=deepseekk``) must fail at
    # startup, not on the first provider call.  Fallback/judge may be empty
    # (feature off) but must be a valid name when set.
    if config.llm.provider not in _LLM_PROVIDERS:
        problems.append(
            f"LLM_PROVIDER={config.llm.provider!r} must be one of "
            f"{', '.join(_LLM_PROVIDERS)}"
        )
    if config.llm.fallback_provider and config.llm.fallback_provider not in _LLM_PROVIDERS:
        problems.append(
            f"LLM_FALLBACK_PROVIDER={config.llm.fallback_provider!r} must be one of "
            f"{', '.join(_LLM_PROVIDERS)}"
        )
    if config.llm.judge_provider and config.llm.judge_provider not in _LLM_PROVIDERS:
        problems.append(
            f"LLM_JUDGE_PROVIDER={config.llm.judge_provider!r} must be one of "
            f"{', '.join(_LLM_PROVIDERS)}"
        )
    if config.embedding.provider not in _EMBEDDING_PROVIDERS:
        problems.append(
            f"EMBEDDING_PROVIDER={config.embedding.provider!r} must be one of "
            f"{', '.join(_EMBEDDING_PROVIDERS)}"
        )

    # LLM API key — real providers all need one; only tests are exempt.
    if config.app_env != "test" and not config.llm.api_key:
        problems.append(
            "LLM_API_KEY is empty — set it in .env (or run tests with APP_ENV=test)"
        )

    # Git-ingestion sandbox — a configured allow-root that does not exist on
    # this host means every ingest would be rejected (or, worse, silently
    # resolve against a mount that isn't there).  Warn early instead of
    # failing the first tool call.
    if config.repo_allow_roots:
        for root in config.repo_allow_roots:
            if not Path(root).expanduser().is_dir():
                problems.append(
                    f"REPO_ALLOW_ROOT entry {root!r} is not an existing directory"
                )

    return problems
