"""Tests for rerank functions — LLM pointwise rerank concurrency bound."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


class TestRerankLlmConcurrency:
    """rerank_llm must not fire every candidate LLM call at once."""

    @pytest.mark.asyncio
    async def test_concurrent_llm_calls_capped_by_semaphore(self, monkeypatch) -> None:
        from backend.service import rerank as mod
        from backend.shared.config import config

        monkeypatch.setattr(config.llm, "rerank_concurrency", 2)

        active = 0
        peak = 0

        async def _chat(messages, **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return "0.5"

        provider = MagicMock()
        provider.chat = AsyncMock(side_effect=_chat)
        monkeypatch.setattr(
            "backend.service.llm_service.get_llm_provider", lambda: provider
        )

        candidates = [f"candidate {i}" for i in range(6)]
        ranked = await mod.rerank_llm("query", candidates, top_k=5)

        assert len(ranked) == 5
        # Without the semaphore, all 6 would be in flight at once (peak=6).
        assert peak <= 2
        assert provider.chat.await_count == 6

    @pytest.mark.asyncio
    async def test_empty_candidates(self) -> None:
        from backend.service import rerank as mod

        assert await mod.rerank_llm("query", [], top_k=5) == []

    @pytest.mark.asyncio
    async def test_llm_failure_scores_zero(self, monkeypatch) -> None:
        from backend.service import rerank as mod

        provider = MagicMock()
        provider.chat = AsyncMock(side_effect=RuntimeError("provider down"))
        monkeypatch.setattr(
            "backend.service.llm_service.get_llm_provider", lambda: provider
        )

        ranked = await mod.rerank_llm("query", ["a", "b"], top_k=5)
        # Sorted by score — both 0.0, stable sort keeps input order.
        assert ranked == [(0, 0.0), (1, 0.0)]


class TestCrossEncoderFirstLoad:
    """The lazy cross-encoder load must not freeze the event loop.

    Loading a 568M model synchronously on first use would block the async
    request that triggers an explicit rerank; the loader runs via
    ``asyncio.to_thread`` instead.
    """

    @pytest.mark.asyncio
    async def test_load_offloaded_to_thread_pool(self, monkeypatch) -> None:
        from backend.service import rerank as mod

        offloaded: list = []

        async def fake_to_thread(fn, *args, **kwargs):  # noqa: ANN001, ANN003, ANN002
            offloaded.append(fn)
            return fn(*args, **kwargs)

        class FakeModel:
            def predict(self, pairs):  # noqa: ANN001
                return [0.9, 0.1]

        monkeypatch.setattr(mod.asyncio, "to_thread", fake_to_thread)
        monkeypatch.setattr(mod, "_get_cross_encoder", lambda: FakeModel())

        ranked = await mod.rerank_cross_encoder("q", ["a", "b"], top_k=2)

        assert ranked == [(0, 0.9), (1, 0.1)]
        # The model loader was invoked through the thread pool, not inline.
        assert mod._get_cross_encoder in offloaded

    @pytest.mark.asyncio
    async def test_empty_candidates(self) -> None:
        from backend.service import rerank as mod

        assert await mod.rerank_cross_encoder("q", [], top_k=5) == []
