"""Tests for LLM usage tracing (``backend/service/usage.py``).

Covers the recording buffer (``begin_call`` / ``record_call``), the batch
flush into ``llm_usage``, the query summaries, cost estimation, and
provider-level instrumentation (a real provider class records one
observation per LLM call).
"""

from __future__ import annotations

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
