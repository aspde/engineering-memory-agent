"""API tests for the LLM usage tracing endpoints (``/api/usage/*``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from sqlalchemy import text

from backend.db import get_session_factory
from backend.service.usage import (
    begin_call,
    flush_usage_buffer,
    record_call,
)
from tests.support.process_state import reset_usage_buffer
from backend.shared.config import current_thread_id, current_trace_id


@pytest.fixture(autouse=True)
async def _clean_usage_table():
    async with get_session_factory()() as session:
        await session.execute(text("DELETE FROM llm_usage"))
        await session.commit()
    reset_usage_buffer()
    yield
    reset_usage_buffer()


async def _seed_one_call(
    trace: str = "trace-1", thread: str = "thread-1"
) -> None:
    t_trace = current_trace_id.set(trace)
    t_thread = current_thread_id.set(thread)
    ctx = begin_call([{"role": "user", "content": "hello"}])
    record_call(
        ctx,
        model="deepseek-chat",
        provider="openai-compatible",
        scenario="agent_chat",
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
    )
    current_trace_id.reset(t_trace)
    current_thread_id.reset(t_thread)
    await flush_usage_buffer()


@pytest.mark.asyncio
async def test_summary_endpoint(async_client) -> None:
    await _seed_one_call()
    resp = await async_client.get("/api/usage/summary?days=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["days"] == 1
    assert data["total_calls"] == 1
    assert data["total_tokens"] == 120
    assert data["items"][0]["total_tokens"] == 120


@pytest.mark.asyncio
async def test_scenarios_endpoint(async_client) -> None:
    await _seed_one_call()
    resp = await async_client.get("/api/usage/scenarios?days=1")
    assert resp.status_code == 200
    scenarios = {s["scenario"]: s for s in resp.json()["items"]}
    assert scenarios["agent_chat"]["calls"] == 1


@pytest.mark.asyncio
async def test_models_endpoint(async_client) -> None:
    await _seed_one_call()
    resp = await async_client.get("/api/usage/models?days=1")
    assert resp.status_code == 200
    models = {m["model"]: m for m in resp.json()["items"]}
    assert models["openai-compatible/deepseek-chat"]["calls"] == 1


@pytest.mark.asyncio
async def test_trace_endpoint(async_client) -> None:
    await _seed_one_call(trace="trace-abc")
    resp = await async_client.get("/api/usage/trace/trace-abc")
    assert resp.status_code == 200
    calls = resp.json()["calls"]
    assert len(calls) == 1
    assert calls[0]["scenario"] == "agent_chat"
    assert calls[0]["thread_id"] == "thread-1"


@pytest.mark.asyncio
async def test_trace_endpoint_honors_limit(async_client) -> None:
    t_trace = current_trace_id.set("trace-limit")
    try:
        for i in range(3):
            ctx = begin_call([{"role": "user", "content": str(i)}])
            record_call(
                ctx,
                model="m",
                provider="p",
                scenario="agent_chat",
            )
    finally:
        current_trace_id.reset(t_trace)
    await flush_usage_buffer()

    full = (await async_client.get("/api/usage/trace/trace-limit")).json()["calls"]
    assert len(full) == 3
    limited = (
        await async_client.get("/api/usage/trace/trace-limit?limit=2")
    ).json()["calls"]
    assert len(limited) == 2


@pytest.mark.asyncio
async def test_usage_read_endpoints_require_api_key_in_production(
    monkeypatch, async_client
) -> None:
    """Samples / trace / threads sit behind the router-level Bearer-key guard
    outside APP_ENV=test (fail closed) — no per-route mechanism is needed."""
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("EMA_API_KEY", "secret-key")

    for path in (
        "/api/usage/samples",
        "/api/usage/trace/some-trace",
        "/api/usage/threads/some-thread",
    ):
        resp = await async_client.get(path)
        assert resp.status_code == 401, path
        assert resp.json()["detail"] == "Invalid or missing API key"

    resp = await async_client.get(
        "/api/usage/samples", headers={"Authorization": "Bearer secret-key"}
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_read_endpoints_redact_error_text(monkeypatch, async_client) -> None:
    """Error text is re-redacted on read for samples / trace / threads."""
    from backend.service import usage

    monkeypatch.setattr(usage.config, "usage_sample_rate", 1.0)
    t_trace = current_trace_id.set("trace-secret")
    t_thread = current_thread_id.set("thread-secret")
    try:
        ctx = begin_call([{"role": "user", "content": "hi"}])
        record_call(
            ctx,
            model="m",
            provider="p",
            scenario="agent_chat",
            status="error",
            error="boom",
            response_text="err",
        )
    finally:
        current_trace_id.reset(t_trace)
        current_thread_id.reset(t_thread)
    await flush_usage_buffer()

    # Simulate a row persisted before write-time redaction existed.
    async with get_session_factory()() as session:
        await session.execute(
            text(
                "UPDATE llm_usage SET error = 'api_key=sk-abc123XYZ987 exploded' "
                "WHERE trace_id = 'trace-secret'"
            )
        )
        await session.commit()

    samples = (await async_client.get("/api/usage/samples?status=error")).json()["samples"]
    assert len(samples) == 1
    assert "sk-abc123XYZ987" not in samples[0]["error"]
    assert "api_key=***" in samples[0]["error"]

    trace = (await async_client.get("/api/usage/trace/trace-secret")).json()["calls"]
    assert trace and "sk-abc123XYZ987" not in trace[0]["error"]

    thread = (await async_client.get("/api/usage/threads/thread-secret")).json()["calls"]
    assert thread and "sk-abc123XYZ987" not in thread[0]["error"]


@pytest.mark.asyncio
async def test_thread_endpoint(async_client) -> None:
    await _seed_one_call(thread="thread-xyz")
    resp = await async_client.get("/api/usage/threads/thread-xyz")
    assert resp.status_code == 200
    calls = resp.json()["calls"]
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_empty_trace_returns_empty(async_client) -> None:
    resp = await async_client.get("/api/usage/trace/does-not-exist")
    assert resp.status_code == 200
    assert resp.json()["calls"] == []


@pytest.mark.asyncio
async def test_samples_endpoint_returns_sampled_calls(monkeypatch, async_client) -> None:
    from backend.service import usage

    monkeypatch.setattr(usage.config, "usage_sample_rate", 1.0)
    t_trace = current_trace_id.set("trace-sample")
    try:
        ctx = begin_call([{"role": "user", "content": "hello"}])
        record_call(
            ctx,
            model="deepseek-chat",
            provider="openai-compatible",
            scenario="agent_chat",
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            response_text="answer body",
        )
    finally:
        current_trace_id.reset(t_trace)
    await flush_usage_buffer()

    resp = await async_client.get("/api/usage/samples")
    assert resp.status_code == 200
    samples = resp.json()["samples"]
    assert len(samples) == 1
    assert samples[0]["response_sample"] == "answer body"
    assert samples[0]["prompt_sample"] == "hello"


@pytest.mark.asyncio
async def test_samples_endpoint_filters_by_status(monkeypatch, async_client) -> None:
    from backend.service import usage

    # rate 0 → only error calls carry samples
    monkeypatch.setattr(usage.config, "usage_sample_rate", 0.0)
    ctx = begin_call([{"role": "user", "content": "hi"}])
    record_call(
        ctx,
        model="m",
        provider="p",
        scenario="agent_chat",
        status="error",
        error="boom",
        response_text="err body",
    )
    await flush_usage_buffer()

    err_resp = await async_client.get("/api/usage/samples?status=error")
    assert err_resp.status_code == 200
    assert len(err_resp.json()["samples"]) == 1

    ok_resp = await async_client.get("/api/usage/samples?status=success")
    assert ok_resp.status_code == 200
    assert ok_resp.json()["samples"] == []


@pytest.mark.asyncio
async def test_samples_endpoint_redacts_on_read(monkeypatch, async_client) -> None:
    """Read-time redaction: a pre-redaction row still never surfaces a secret."""
    from backend.service import usage

    monkeypatch.setattr(usage.config, "usage_sample_rate", 1.0)
    t_trace = current_trace_id.set("trace-redact")
    try:
        ctx = begin_call([{"role": "user", "content": "hi"}])
        record_call(
            ctx,
            model="m",
            provider="p",
            scenario="agent_chat",
            response_text="answer body",
        )
    finally:
        current_trace_id.reset(t_trace)
    await flush_usage_buffer()

    # Simulate a row persisted before write-time redaction existed.
    async with get_session_factory()() as session:
        await session.execute(
            text(
                "UPDATE llm_usage SET response_sample = "
                "'Authorization: Bearer sk-abc123XYZ987' "
                "WHERE trace_id = 'trace-redact'"
            )
        )
        await session.commit()

    resp = await async_client.get("/api/usage/samples")
    assert resp.status_code == 200
    samples = resp.json()["samples"]
    assert len(samples) == 1
    assert "sk-abc123XYZ987" not in samples[0]["response_sample"]
    assert "Bearer ***" in samples[0]["response_sample"]


@pytest.mark.asyncio
async def test_summary_exposes_cache_totals(async_client) -> None:
    t_trace = current_trace_id.set("trace-cache")
    try:
        ctx = begin_call([{"role": "user", "content": "hi"}])
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
    finally:
        current_trace_id.reset(t_trace)
    await flush_usage_buffer()

    resp = await async_client.get("/api/usage/scenarios?days=1")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_cache_read_tokens"] == 80
    assert data["total_cache_creation_tokens"] == 30
    item = data["items"][0]
    assert item["cache_read_tokens"] == 80
    assert item["cache_creation_tokens"] == 30
