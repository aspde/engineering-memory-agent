"""Tests for LLM usage tracing (``backend/service/usage.py``).

Covers the recording buffer (``begin_call`` / ``record_call``), the batch
flush into ``llm_usage``, the query summaries, cost estimation, and
provider-level instrumentation (a real provider class records one
observation per LLM call).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.service import usage
from backend.service.usage import (
    begin_call,
    estimate_cost,
    flush_usage_buffer,
    get_daily_summary,
    get_model_summary,
    get_samples,
    get_scenario_summary,
    get_thread_usage,
    get_trace,
    pending_count,
    pending_rows,
    record_call,
    reset_usage_buffer,
)
from backend.shared.config import current_thread_id, current_trace_id


@pytest.fixture(autouse=True)
def _clean_buffer():
    reset_usage_buffer()
    yield
    reset_usage_buffer()


@pytest.fixture(autouse=True)
async def _clean_llm_usage_table():
    """Isolate DB-backed tests: the test DB persists across a session, so
    ``llm_usage`` rows from one test would otherwise leak into the next."""
    from sqlalchemy import text

    from backend.db import get_session_factory

    async def _truncate() -> None:
        async with get_session_factory()() as session:
            await session.execute(text("DELETE FROM llm_usage"))
            await session.commit()

    await _truncate()
    yield
    await _truncate()


def _ctx(trace: str = "", thread: str = "") -> dict:
    """Return a recording context with the given trace/thread in place."""
    t_trace = current_trace_id.set(trace)
    t_thread = current_thread_id.set(thread)
    ctx = begin_call([{"role": "user", "content": "hello world"}])
    current_trace_id.reset(t_trace)
    current_thread_id.reset(t_thread)
    return ctx


# ── Cost estimation ────────────────────────────────────────────────────


class TestEstimateCost:
    def test_known_model_matched_by_substring(self) -> None:
        # deepseek rule: 0.27 in / 1.10 out per 1M tokens
        cost = estimate_cost("deepseek-chat", 1_000_000, 0)
        assert cost == pytest.approx(0.27, abs=1e-6)

    def test_unknown_model_falls_back_to_default(self) -> None:
        # _DEFAULT_PRICE = (1.0, 3.0)
        cost = estimate_cost("custom-model", 1_000_000, 1_000_000)
        assert cost == pytest.approx(4.0, abs=1e-6)

    def test_zero_tokens_is_zero(self) -> None:
        assert estimate_cost("deepseek-chat", 0, 0) == 0.0

    def test_cache_read_discounted_cache_creation_full(self) -> None:
        # claude-sonnet input price $3.0 / 1M; cache reads at 10%, creation
        # at full input price.
        cost = estimate_cost(
            "claude-sonnet",
            0,
            0,
            cache_read_tokens=1_000_000,
            cache_creation_tokens=500_000,
        )
        assert cost == pytest.approx(1.8, abs=1e-6)

    def test_cache_fields_ignored_when_zero(self) -> None:
        assert estimate_cost("deepseek-chat", 0, 0) == estimate_cost(
            "deepseek-chat", 0, 0, cache_read_tokens=0, cache_creation_tokens=0
        )

    def test_cache_read_factor_dispatches_by_provider(self) -> None:
        # OpenAI-family cache reads bill at 0.5x input price (gpt-4o in $2.5/1M).
        openai_cost = estimate_cost(
            "gpt-4o", 0, 0, cache_read_tokens=1_000_000,
            provider="openai-compatible",
        )
        assert openai_cost == pytest.approx(1.25, abs=1e-6)
        # DeepSeek cache hits bill at 0.1x input price (deepseek in $0.27/1M),
        # even though its stored provider column is "openai-compatible".
        deepseek_cost = estimate_cost(
            "deepseek-chat", 0, 0, cache_read_tokens=1_000_000,
            provider="openai-compatible",
        )
        assert deepseek_cost == pytest.approx(0.027, abs=1e-6)
        # Anthropic prompt-cache reads bill at 0.1x input price.
        anthropic_cost = estimate_cost(
            "claude-sonnet", 0, 0, cache_read_tokens=1_000_000,
            provider="anthropic",
        )
        assert anthropic_cost == pytest.approx(0.3, abs=1e-6)


# ── Recording context ──────────────────────────────────────────────────


class TestBeginCall:
    def test_reads_trace_and_thread_from_contextvars(self) -> None:
        t_trace = current_trace_id.set("trace-abc")
        t_thread = current_thread_id.set("thread-9")
        try:
            ctx = begin_call([{"role": "user", "content": "hello"}])
        finally:
            current_trace_id.reset(t_trace)
            current_thread_id.reset(t_thread)
        assert ctx["trace_id"] == "trace-abc"
        assert ctx["thread_id"] == "thread-9"
        assert "t0" in ctx
        assert ctx["prompt_chars"] == len("hello")

    def test_empty_contextvars_yield_empty_ids(self) -> None:
        ctx = _ctx("", "")
        assert ctx["trace_id"] == ""
        assert ctx["thread_id"] == ""


# ── record_call ────────────────────────────────────────────────────────


class TestRecordCall:
    def test_records_success_with_openai_usage_shape(self) -> None:
        ctx = _ctx("trace-1", "thread-1")
        record_call(
            ctx,
            model="deepseek-chat",
            provider="openai-compatible",
            scenario="agent_chat",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            response_text="answer",
        )
        rows = pending_rows()
        assert len(rows) == 1
        row = rows[0]
        assert row["trace_id"] == "trace-1"
        assert row["thread_id"] == "thread-1"
        assert row["scenario"] == "agent_chat"
        assert row["model"] == "deepseek-chat"
        assert row["input_tokens"] == 10
        assert row["output_tokens"] == 5
        assert row["total_tokens"] == 15
        assert row["status"] == "success"
        assert row["error"] is None
        assert row["response_chars"] == len("answer")
        assert row["prompt_chars"] == len("hello world")
        assert row["latency_ms"] >= 0

    def test_records_anthropic_usage_shape(self) -> None:
        ctx = _ctx("t", "th")
        record_call(
            ctx,
            model="claude-sonnet",
            provider="anthropic",
            scenario="conflict_detection",
            usage=SimpleNamespace(input_tokens=20, output_tokens=3),
        )
        row = pending_rows()[0]
        assert row["input_tokens"] == 20
        assert row["output_tokens"] == 3
        assert row["total_tokens"] == 23

    def test_records_dict_usage_shape(self) -> None:
        ctx = _ctx("t", "th")
        record_call(
            ctx,
            model="m",
            provider="p",
            scenario="s",
            usage={"prompt_tokens": 4, "completion_tokens": 2},
        )
        row = pending_rows()[0]
        assert row["total_tokens"] == 6

    def test_records_without_usage(self) -> None:
        ctx = _ctx("t", "th")
        record_call(ctx, model="m", provider="p", scenario="s", usage=None)
        row = pending_rows()[0]
        assert row["total_tokens"] == 0
        assert row["status"] == "success"

    def test_record_call_attempts_defaults_to_none(self) -> None:
        """Callers that don't report attempts keep the column NULL — historical
        rows (and non-retryable call paths) must not get a fabricated value."""
        ctx = _ctx("t", "th")
        record_call(ctx, model="m", provider="p", scenario="s", usage=None)
        assert pending_rows()[0]["attempts"] is None

    def test_record_call_stores_attempts(self) -> None:
        """record_call must persist how many times the provider was hit before
        this outcome (3 = two tenacity retries swallowed before success)."""
        ctx = _ctx("t", "th")
        record_call(
            ctx,
            model="m",
            provider="p",
            scenario="agent_chat",
            usage=None,
            attempts=3,
        )
        row = pending_rows()[0]
        assert row["attempts"] == 3

    def test_records_error_status(self) -> None:
        ctx = _ctx("t", "th")
        record_call(
            ctx,
            model="m",
            provider="p",
            scenario="agent_chat",
            status="error",
            error="boom",
        )
        row = pending_rows()[0]
        assert row["status"] == "error"
        assert row["error"] == "boom"
        assert row["total_tokens"] == 0

    def test_error_text_redacted_before_storage(self) -> None:
        """Provider error strings that echo a secret are masked on the way in."""
        ctx = _ctx("t", "th")
        record_call(
            ctx,
            model="m",
            provider="p",
            scenario="agent_chat",
            status="error",
            error="Authorization: Bearer sk-abc123XYZ987 boom",
        )
        row = pending_rows()[0]
        assert "sk-abc123XYZ987" not in row["error"]
        assert "Bearer ***" in row["error"]
        assert "boom" in row["error"]  # the non-secret tail survives

    def test_disabled_when_usage_enabled_false(self, monkeypatch) -> None:
        monkeypatch.setattr(usage.config, "usage_enabled", False)
        ctx = _ctx("t", "th")
        record_call(ctx, model="m", provider="p", scenario="s", usage=None)
        assert pending_count() == 0

    def test_buffer_caps_at_max_and_drops_oldest(self, monkeypatch) -> None:
        monkeypatch.setattr(usage.config, "usage_buffer_max", 2)
        for i in range(4):
            record_call(_ctx(f"t{i}", "th"), model="m", provider="p", scenario="s")
        rows = pending_rows()
        assert len(rows) == 2
        assert rows[0]["trace_id"] == "t2"
        assert rows[1]["trace_id"] == "t3"


# ── Prompt-cache token accounting ─────────────────────────────────────
# Cached input tokens (read from / created into a provider cache) are tracked
# separately so cost summaries can apply the discounted cache-read rate.


class TestCacheTokenAccounting:
    def test_openai_cached_tokens_recorded(self) -> None:
        ctx = _ctx("trace-1", "thread-1")
        record_call(
            ctx,
            model="gpt-4o",
            provider="openai-compatible",
            scenario="agent_chat",
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                prompt_tokens_details=SimpleNamespace(cached_tokens=60),
            ),
        )
        row = pending_rows()[0]
        # input is the full-price portion: prompt_tokens minus its cached subset
        assert row["input_tokens"] == 40
        assert row["output_tokens"] == 20
        assert row["total_tokens"] == 120
        assert row["cache_read_tokens"] == 60
        assert row["cache_creation_tokens"] == 0

    def test_anthropic_cached_tokens_recorded(self) -> None:
        ctx = _ctx("t", "th")
        record_call(
            ctx,
            model="claude-sonnet",
            provider="anthropic",
            scenario="agent_chat",
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                cache_read_input_tokens=80,
                cache_creation_input_tokens=30,
            ),
        )
        row = pending_rows()[0]
        assert row["input_tokens"] == 10
        assert row["output_tokens"] == 5
        assert row["total_tokens"] == 125  # input + output + both cache buckets
        assert row["cache_read_tokens"] == 80
        assert row["cache_creation_tokens"] == 30

    def test_dict_cache_hit_tokens_recorded(self) -> None:
        # DeepSeek-style usage: prompt_tokens = cache_hit + cache_miss.
        ctx = _ctx("t", "th")
        record_call(
            ctx,
            model="deepseek-chat",
            provider="openai-compatible",
            scenario="agent_chat",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "prompt_cache_hit_tokens": 60,
                "prompt_cache_miss_tokens": 40,
            },
        )
        row = pending_rows()[0]
        assert row["input_tokens"] == 40  # cache-miss (full-price) tokens
        assert row["total_tokens"] == 120
        assert row["cache_read_tokens"] == 60
        assert row["cache_creation_tokens"] == 0

    def test_object_cache_hit_tokens_recorded(self) -> None:
        # DeepSeek's SDK returns an *object* shape whose cache hits live at the
        # top level (prompt_cache_hit_tokens) with empty prompt_tokens_details —
        # the object branch must not report every hit at full price.
        ctx = _ctx("t", "th")
        record_call(
            ctx,
            model="deepseek-chat",
            provider="openai-compatible",
            scenario="agent_chat",
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=20,
                prompt_cache_hit_tokens=60,
                prompt_cache_miss_tokens=40,
                prompt_tokens_details=None,
            ),
        )
        row = pending_rows()[0]
        assert row["input_tokens"] == 40  # cache-miss (full-price) tokens
        assert row["total_tokens"] == 120
        assert row["cache_read_tokens"] == 60
        assert row["cache_creation_tokens"] == 0

    def test_dict_alias_fields_not_double_counted(self) -> None:
        # A proxy returning both prompt_tokens and input_tokens must not have
        # its input summed twice (the old dict branch did).
        ctx = _ctx("t", "th")
        record_call(
            ctx,
            model="m",
            provider="p",
            scenario="s",
            usage={
                "prompt_tokens": 10,
                "input_tokens": 99,
                "completion_tokens": 5,
                "output_tokens": 88,
            },
        )
        row = pending_rows()[0]
        assert row["input_tokens"] == 10
        assert row["output_tokens"] == 5
        assert row["total_tokens"] == 15

    def test_total_only_object_still_counted(self) -> None:
        ctx = _ctx("t", "th")
        record_call(
            ctx,
            model="m",
            provider="p",
            scenario="s",
            usage=SimpleNamespace(total_tokens=42),
        )
        row = pending_rows()[0]
        assert row["total_tokens"] == 42
        assert row["input_tokens"] == 0


# ── Sample redaction ──────────────────────────────────────────────────


class TestRedaction:
    def test_masks_json_api_key(self) -> None:
        out = usage._redact('{"api_key": "sk-abc123XYZ987"}')
        assert "sk-abc123XYZ987" not in out
        assert 'api_key": "***"' in out

    def test_masks_authorization_bearer(self) -> None:
        out = usage._redact("Authorization: Bearer sk-abc123XYZ987")
        assert "sk-abc123XYZ987" not in out
        assert "Bearer ***" in out

    def test_masks_bare_bearer_token(self) -> None:
        out = usage._redact("use Bearer sk-abc123XYZ987 for auth")
        assert "sk-abc123XYZ987" not in out

    def test_masks_key_value_pair(self) -> None:
        assert usage._redact("token=abcdefgh12345678") == "token=***"

    def test_masks_token_prefix_in_plain_text(self) -> None:
        assert "sk-abc123XYZ987" not in usage._redact("key is sk-abc123XYZ987 ok")

    def test_masks_private_key_block(self) -> None:
        text = "-----BEGIN PRIVATE KEY-----\nZm9vYmFyCg==\n-----END PRIVATE KEY-----"
        out = usage._redact(text)
        assert "PRIVATE KEY" not in out
        assert "Zm9vYmFy" not in out

    def test_leaves_plain_text_unchanged(self) -> None:
        assert usage._redact("ordinary prompt text") == "ordinary prompt text"

    def test_empty_string(self) -> None:
        assert usage._redact("") == ""

    def test_record_call_stores_redacted_samples(self, monkeypatch) -> None:
        monkeypatch.setattr(usage.config, "usage_sample_rate", 1.0)
        record_call(
            _ctx("t", "th"),
            model="m", provider="p", scenario="s",
            response_text="Answer with api_key=sk-abc123XYZ987 inside",
        )
        row = pending_rows()[0]
        assert row["response_sample"] is not None
        assert "sk-abc123XYZ987" not in row["response_sample"]
        assert "api_key=***" in row["response_sample"]


# ── Flush + query (test DB) ────────────────────────────────────────────


class TestFlushAndQuery:
    @pytest.mark.asyncio
    async def test_flush_then_query_aggregates(self) -> None:
        t1 = current_trace_id.set("trace-1")
        current_thread_id.set("thread-1")
        ctx1 = begin_call([{"role": "user", "content": "a"}])
        record_call(
            ctx1,
            model="deepseek-chat",
            provider="openai-compatible",
            scenario="agent_chat",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        current_trace_id.reset(t1)

        t2 = current_trace_id.set("trace-2")
        current_thread_id.set("thread-1")
        ctx2 = begin_call([{"role": "user", "content": "b"}])
        record_call(
            ctx2,
            model="claude-sonnet",
            provider="anthropic",
            scenario="conflict_detection",
            usage=SimpleNamespace(input_tokens=20, output_tokens=3),
        )
        current_trace_id.reset(t2)

        assert await flush_usage_buffer() == 2
        assert pending_count() == 0

        daily = await get_daily_summary(1)
        assert daily, "daily summary should contain today"
        assert daily[0]["calls"] == 2
        assert daily[0]["total_tokens"] == 38
        assert daily[0]["error_calls"] == 0

        scenarios = {s["scenario"]: s for s in await get_scenario_summary(1)}
        assert scenarios["agent_chat"]["calls"] == 1
        assert scenarios["agent_chat"]["total_tokens"] == 15
        assert scenarios["conflict_detection"]["total_tokens"] == 23

        models = {m["model"]: m for m in await get_model_summary(1)}
        assert models["openai-compatible/deepseek-chat"]["calls"] == 1
        assert models["anthropic/claude-sonnet"]["calls"] == 1
        assert "est_cost" in models["openai-compatible/deepseek-chat"]

        thread_calls = await get_thread_usage("thread-1")
        assert len(thread_calls) == 2

        trace = await get_trace("trace-1")
        assert len(trace) == 1
        assert trace[0]["scenario"] == "agent_chat"
        assert trace[0]["thread_id"] == "thread-1"

    @pytest.mark.asyncio
    async def test_trace_returns_calls_in_insertion_order(self) -> None:
        t = current_trace_id.set("trace-ordered")
        try:
            record_call(
                begin_call([{"role": "user", "content": "1"}]),
                model="m", provider="p", scenario="agent_chat",
            )
            record_call(
                begin_call([{"role": "user", "content": "2"}]),
                model="m", provider="p", scenario="agent_final",
            )
        finally:
            current_trace_id.reset(t)
        await flush_usage_buffer()

        trace = await get_trace("trace-ordered")
        assert [c["scenario"] for c in trace] == ["agent_chat", "agent_final"]

    @pytest.mark.asyncio
    async def test_trace_respects_limit(self) -> None:
        t = current_trace_id.set("trace-limit")
        try:
            for i in range(3):
                record_call(
                    begin_call([{"role": "user", "content": str(i)}]),
                    model="m", provider="p", scenario="agent_chat",
                )
        finally:
            current_trace_id.reset(t)
        await flush_usage_buffer()

        assert len(await get_trace("trace-limit")) == 3
        assert len(await get_trace("trace-limit", limit=2)) == 2

    @pytest.mark.asyncio
    async def test_empty_window_returns_empty(self) -> None:
        assert await get_daily_summary(1) == []
        assert await get_scenario_summary(1) == []
        assert await get_model_summary(1) == []
        assert await get_trace("nope") == []
        assert await get_thread_usage("nope") == []

    @pytest.mark.asyncio
    async def test_flush_returns_zero_when_buffer_empty(self) -> None:
        assert await flush_usage_buffer() == 0

    @pytest.mark.asyncio
    async def test_flush_persists_attempts(self) -> None:
        """The ``attempts`` column survives the buffer → llm_usage roundtrip."""
        from sqlalchemy import text

        from backend.db import get_session_factory

        record_call(
            _ctx("t", "th"),
            model="m", provider="p", scenario="s",
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            attempts=3,
        )
        await flush_usage_buffer()

        async with get_session_factory()() as session:
            result = await session.execute(text("SELECT attempts FROM llm_usage"))
            assert [tuple(r) for r in result] == [(3,)]

    @pytest.mark.asyncio
    async def test_summary_reports_cache_tokens(self) -> None:
        record_call(
            _ctx("t", "th"),
            model="claude-sonnet",
            provider="anthropic",
            scenario="agent_chat",
            usage=SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                cache_read_input_tokens=80,
                cache_creation_input_tokens=30,
            ),
        )
        await flush_usage_buffer()

        daily = await get_daily_summary(1)
        assert daily[0]["cache_read_tokens"] == 80
        assert daily[0]["cache_creation_tokens"] == 30

        scenarios = {s["scenario"]: s for s in await get_scenario_summary(1)}
        assert scenarios["agent_chat"]["cache_read_tokens"] == 80
        assert scenarios["agent_chat"]["cache_creation_tokens"] == 30

    @pytest.mark.asyncio
    async def test_model_summary_includes_estimated_cost(self) -> None:
        record_call(
            _ctx("t", "th"),
            model="deepseek-chat",
            provider="openai-compatible",
            scenario="agent_chat",
            usage=SimpleNamespace(prompt_tokens=1_000_000, completion_tokens=0),
        )
        await flush_usage_buffer()

        models = {m["model"]: m for m in await get_model_summary(1)}
        # deepseek input price: $0.27 / 1M tokens
        assert models["openai-compatible/deepseek-chat"]["est_cost"] == pytest.approx(
            0.27, abs=0.001
        )


class TestFlushRetry:
    """A failed flush requeues its rows for retry, capped per row."""

    class _BoomSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> bool:
            return False

        async def execute(self, *args, **kwargs):
            raise RuntimeError("db down")

        async def commit(self):
            raise RuntimeError("db down")

    @pytest.mark.asyncio
    async def test_failed_flush_requeues_rows(self, monkeypatch) -> None:
        record_call(_ctx("t1", "th"), model="m", provider="p", scenario="s")
        real_factory = usage.get_session_factory
        monkeypatch.setattr(usage, "get_session_factory", lambda: self._BoomSession)

        assert await usage.flush_usage_buffer() == 0
        # rows came back to the head of the buffer, attempt counter incremented
        rows = pending_rows()
        assert len(rows) == 1
        assert rows[0]["trace_id"] == "t1"
        assert rows[0]["_flush_attempts"] == 1
        assert usage._consecutive_flush_failures == 1

        # a subsequent success drains them (the internal key is ignored by INSERT)
        monkeypatch.setattr(usage, "get_session_factory", real_factory)
        assert await usage.flush_usage_buffer() == 1
        assert pending_count() == 0
        assert usage._consecutive_flush_failures == 0

    @pytest.mark.asyncio
    async def test_rows_dropped_after_max_attempts(self, monkeypatch) -> None:
        record_call(_ctx("t1", "th"), model="m", provider="p", scenario="s")
        monkeypatch.setattr(usage, "get_session_factory", lambda: self._BoomSession)

        for _ in range(usage._FLUSH_MAX_ATTEMPTS):
            await usage.flush_usage_buffer()
        # past the cap the row is dropped rather than requeued forever
        assert pending_count() == 0

    @pytest.mark.asyncio
    async def test_failed_flush_counts_consecutive_failures(self, monkeypatch) -> None:
        record_call(_ctx("t1", "th"), model="m", provider="p", scenario="s")
        monkeypatch.setattr(usage, "get_session_factory", lambda: self._BoomSession)

        await usage.flush_usage_buffer()
        assert usage._consecutive_flush_failures == 1
        await usage.flush_usage_buffer()
        assert usage._consecutive_flush_failures == 2

    class _CancelSession(_BoomSession):
        """A session that is cancelled mid-await (shutdown path)."""

        async def execute(self, *args, **kwargs):
            raise asyncio.CancelledError

        async def commit(self):
            raise asyncio.CancelledError

    @pytest.mark.asyncio
    async def test_cancelled_flush_requeues_and_propagates(self, monkeypatch) -> None:
        """A cancelled flush requeues its drained rows, then re-raises — the
        rows are never silently discarded (except Exception misses CancelledError)."""
        record_call(_ctx("t1", "th"), model="m", provider="p", scenario="s")
        monkeypatch.setattr(usage, "get_session_factory", lambda: self._CancelSession)

        with pytest.raises(asyncio.CancelledError):
            await usage.flush_usage_buffer()

        rows = pending_rows()
        assert len(rows) == 1
        assert rows[0]["trace_id"] == "t1"
        assert rows[0]["_flush_attempts"] == 1

    @pytest.mark.asyncio
    async def test_empty_flush_does_not_reset_consecutive_failures(self) -> None:
        """An empty buffer is not a successful drain — it must not clear the
        backoff counter (only a non-empty successful flush does)."""
        usage._consecutive_flush_failures = 3
        assert await usage.flush_usage_buffer() == 0
        assert usage._consecutive_flush_failures == 3


# ── Sampled prompt/response text ───────────────────────────────────────


class TestSampleColumns:
    """prompt/response sampling: error calls always, success at the rate."""

    def test_success_sampled_at_rate_one(self, monkeypatch) -> None:
        monkeypatch.setattr(usage.config, "usage_sample_rate", 1.0)
        record_call(
            _ctx("t", "th"), model="m", provider="p", scenario="s",
            response_text="answer text",
        )
        row = pending_rows()[0]
        assert row["prompt_sample"] == "hello world"
        assert row["response_sample"] == "answer text"

    def test_success_not_sampled_at_rate_zero(self, monkeypatch) -> None:
        monkeypatch.setattr(usage.config, "usage_sample_rate", 0.0)
        record_call(
            _ctx("t", "th"), model="m", provider="p", scenario="s",
            response_text="answer text",
        )
        row = pending_rows()[0]
        assert row["prompt_sample"] is None
        assert row["response_sample"] is None

    def test_error_always_sampled_even_at_rate_zero(self, monkeypatch) -> None:
        monkeypatch.setattr(usage.config, "usage_sample_rate", 0.0)
        record_call(
            _ctx("t", "th"), model="m", provider="p", scenario="s",
            status="error", error="boom", response_text="partial",
        )
        row = pending_rows()[0]
        assert row["prompt_sample"] == "hello world"
        assert row["response_sample"] == "partial"

    def test_prompt_text_truncated_to_sample_cap(self) -> None:
        long_text = "x" * (usage._SAMPLE_MAX_CHARS + 500)
        ctx = begin_call([{"role": "user", "content": long_text}])
        assert len(ctx["prompt_text"]) == usage._SAMPLE_MAX_CHARS
        # prompt_chars still reflects the full transcript, not the cap
        assert ctx["prompt_chars"] == len(long_text)

    def test_response_sample_truncated(self, monkeypatch) -> None:
        monkeypatch.setattr(usage.config, "usage_sample_rate", 1.0)
        long_resp = "y" * (usage._SAMPLE_MAX_CHARS + 100)
        record_call(_ctx("t", "th"), model="m", provider="p", scenario="s", response_text=long_resp)
        row = pending_rows()[0]
        assert len(row["response_sample"]) == usage._SAMPLE_MAX_CHARS


class TestSamplesQuery:
    """get_samples returns only calls that carried sampled text."""

    @pytest.mark.asyncio
    async def test_returns_sampled_calls_and_filters(self, monkeypatch) -> None:
        monkeypatch.setattr(usage.config, "usage_sample_rate", 1.0)
        t = current_trace_id.set("trace-samples")
        record_call(
            begin_call([{"role": "user", "content": "prompt A"}]),
            model="m", provider="p", scenario="agent_chat", response_text="resp A",
        )
        current_trace_id.reset(t)
        record_call(
            _ctx("", ""),
            model="m", provider="p", scenario="conflict_detection",
            status="error", error="boom", response_text="err resp",
        )
        await flush_usage_buffer()

        samples = await get_samples()
        assert len(samples) == 2
        assert any(s["response_sample"] == "resp A" for s in samples)
        assert any(s["response_sample"] == "err resp" for s in samples)

        errs = await get_samples(status="error")
        assert len(errs) == 1 and errs[0]["status"] == "error"

        by_scenario = await get_samples(scenario="agent_chat")
        assert len(by_scenario) == 1 and by_scenario[0]["scenario"] == "agent_chat"

        by_trace = await get_samples(trace_id="trace-samples")
        assert len(by_trace) == 1
        assert by_trace[0]["prompt_sample"] == "prompt A"

    @pytest.mark.asyncio
    async def test_excludes_unsampled_calls(self, monkeypatch) -> None:
        monkeypatch.setattr(usage.config, "usage_sample_rate", 0.0)
        record_call(
            _ctx("t", "th"), model="m", provider="p", scenario="s",
            response_text="not sampled",
        )
        await flush_usage_buffer()
        assert await get_samples() == []


class TestSamplePurge:
    """Sampled text older than the retention window is nulled; the metadata
    row (tokens / latency / status) stays for usage summaries."""

    @pytest.mark.asyncio
    async def test_purges_expired_keeps_fresh(self, monkeypatch) -> None:
        from sqlalchemy import text

        from backend.db import get_session_factory

        monkeypatch.setattr(usage.config, "usage_sample_retention_days", 30)
        async with get_session_factory()() as session:
            await session.execute(
                text(
                    "INSERT INTO llm_usage (scenario, provider, model, status, "
                    "prompt_sample, response_sample, created_at) VALUES "
                    "('s','p','m','success','OLD_PROMPT','OLD_RESP', "
                    "now() - make_interval(days => 60)), "
                    "('s','p','m','success','NEW_PROMPT','NEW_RESP', now())"
                )
            )
            await session.commit()

        purged = await usage._purge_expired_samples()
        assert purged == 1  # exactly the expired row, not the fresh one

        async with get_session_factory()() as session:
            result = await session.execute(
                text(
                    "SELECT prompt_sample, response_sample FROM llm_usage "
                    "ORDER BY created_at DESC"
                )
            )
            rows = [tuple(r) for r in result]
        # Fresh row keeps its sample; the expired row has both columns nulled
        # but is still present (metadata survives).
        assert rows[0] == ("NEW_PROMPT", "NEW_RESP")
        assert rows[1] == (None, None)


# ── Provider instrumentation ───────────────────────────────────────────


class TestProviderInstrumentation:
    @pytest.mark.asyncio
    async def test_openai_chat_records_one_row(self) -> None:
        from backend.service.llm_service import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key="k", base_url="https://example.com/v1", model="test-model"
        )
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="hi back"))]
        resp.usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2)
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = resp
        provider._async_client = mock_client  # type: ignore[assignment]

        out = await provider.chat(
            [{"role": "user", "content": "hi"}], scenario="agent_chat"
        )

        assert out == "hi back"
        rows = pending_rows()
        assert len(rows) == 1
        assert rows[0]["scenario"] == "agent_chat"
        assert rows[0]["model"] == "test-model"
        assert rows[0]["provider"] == "openai-compatible"
        assert rows[0]["input_tokens"] == 3
        assert rows[0]["output_tokens"] == 2
        assert rows[0]["status"] == "success"

    @pytest.mark.asyncio
    async def test_openai_chat_records_error_and_propagates(self) -> None:
        from backend.service.llm_service import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key="k", base_url="https://example.com/v1", model="test-model"
        )
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("boom")
        provider._async_client = mock_client  # type: ignore[assignment]

        with pytest.raises(RuntimeError):
            await provider.chat(
                [{"role": "user", "content": "hi"}], scenario="agent_chat"
            )

        rows = pending_rows()
        assert len(rows) == 1
        assert rows[0]["status"] == "error"
        assert "boom" in rows[0]["error"]

    @pytest.mark.asyncio
    async def test_openai_stream_records_after_full_consumption(self) -> None:
        from backend.service.llm_service import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            api_key="k", base_url="https://example.com/v1", model="test-model"
        )

        delta = MagicMock(content="hello", tool_calls=None)
        chunk = MagicMock(choices=[MagicMock(delta=delta)])
        chunk.usage = None

        class _FakeStream:
            def __init__(self) -> None:
                self.usage = SimpleNamespace(total_tokens=42)
                self._chunks = [chunk]

            def __aiter__(self):
                return self._iter()

            async def _iter(self):
                for c in self._chunks:
                    yield c

        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=_FakeStream())
        provider._async_client = mock_client  # type: ignore[assignment]

        events = [
            event
            async for event in provider.chat_raw_stream(
                [{"role": "user", "content": "hi"}], scenario="agent_chat"
            )
        ]
        assert events == [{"type": "content", "text": "hello"}]

        rows = pending_rows()
        assert len(rows) == 1
        assert rows[0]["scenario"] == "agent_chat"
        assert rows[0]["total_tokens"] == 42
        assert rows[0]["status"] == "success"

    @pytest.mark.asyncio
    async def test_anthropic_chat_json_records_row(self, monkeypatch) -> None:
        import anthropic

        from backend.service.llm_service import AnthropicProvider

        class _ToolUseBlock:
            type = "tool_use"
            input = {"result": {"ok": True}}

        class _FakeMessages:
            async def create(self, **kwargs):  # type: ignore[no-untyped-def]
                resp = MagicMock()
                resp.content = [_ToolUseBlock()]
                resp.usage = SimpleNamespace(input_tokens=7, output_tokens=1)
                return resp

        class _FakeAsyncAnthropic:
            def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
                self.messages = _FakeMessages()

        monkeypatch.setattr(anthropic, "AsyncAnthropic", _FakeAsyncAnthropic)

        provider = AnthropicProvider(api_key="k", model="claude-test")
        raw = await provider.chat_json(
            [{"role": "user", "content": "hi"}],
            json_schema={"type": "object"},
            scenario="extraction_entities",
        )
        assert raw == '{"ok": true}'

        rows = pending_rows()
        assert len(rows) == 1
        assert rows[0]["provider"] == "anthropic"
        assert rows[0]["scenario"] == "extraction_entities"
        assert rows[0]["input_tokens"] == 7
        assert rows[0]["output_tokens"] == 1
