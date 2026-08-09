"""Tests for checkpointer fallback behaviour.

``_setup_checkpointer`` must fall back to ``InMemorySaver`` when the
psycopg pool cannot be established — not hang startup.  On Windows the
psycopg 3 async driver cannot run on uvicorn's default
``ProactorEventLoop`` and retries the connection forever instead of
raising; the bounded ``asyncio.wait_for`` around ``_pool.wait()`` turns
that infinite retry into a ``TimeoutError`` so the fallback branch runs.
"""

from __future__ import annotations

import asyncio

import pytest

# psycopg 3 的异步驱动在 Windows 开发机上不安装（ProactorEventLoop 下不可用），
# 该测试要 monkeypatch 它的 AsyncConnectionPool 才能跑——本平台优雅跳过，
# CI（Linux）上照常执行。这也与测试目的一致：它测的正是 Windows 才会触发的
# 降级路径，降级行为在安装驱动的平台验证。
pytest.importorskip("psycopg_pool")

from backend.service.agent_service import _setup_checkpointer


@pytest.mark.asyncio
async def test_checkpointer_falls_back_when_pool_wait_times_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A psycopg pool that never connects (Windows ProactorEventLoop
    behaviour) must time out and degrade to InMemorySaver."""

    class _StuckPool:
        async def open(self) -> None:
            return None

        async def wait(self) -> None:
            await asyncio.sleep(3600)  # never connects, never raises

    async def _fake_open(self) -> None:
        return None

    # Force the timeout path: replace asyncio.wait_for so the stuck wait
    # really is bounded (the test must not block for the real 10s window
    # repeatedly), then assert the fallback ran.
    real_wait_for = asyncio.wait_for

    async def _bounded_wait_for(awaitable, timeout):
        return await real_wait_for(awaitable, timeout=0.1)

    monkeypatch.setattr(
        "psycopg_pool.AsyncConnectionPool",
        lambda **kwargs: _StuckPool(),
    )
    monkeypatch.setattr("asyncio.wait_for", _bounded_wait_for)

    # _setup_checkpointer writes _checkpointer in the module; import it
    # through the module so the assignment is observable.
    import backend.service.agent_service as svc

    await _setup_checkpointer()
    assert isinstance(svc._checkpointer, svc.InMemorySaver)
