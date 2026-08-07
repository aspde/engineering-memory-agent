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
