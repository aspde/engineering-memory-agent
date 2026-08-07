"""Tests for the memory write-path's correctness-critical structured call.

``_detect_conflict`` is correctness-critical: after retries are exhausted it
must raise (so the write fails) rather than silently assume no conflict.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from backend.model.llm import LLMStructuredError
from backend.service.memory import _detect_conflict


class TestDetectConflict:
    @pytest.mark.asyncio
    async def test_returns_true_when_conflict(self, monkeypatch) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat_json.return_value = json.dumps({"conflict": True})
        monkeypatch.setattr("backend.service.structured.get_llm_provider", lambda: mock_llm)

        assert await _detect_conflict({"summary": "old"}, {"summary": "new"}) is True

    @pytest.mark.asyncio
    async def test_returns_false_when_no_conflict(self, monkeypatch) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat_json.return_value = json.dumps({"conflict": False})
        monkeypatch.setattr("backend.service.structured.get_llm_provider", lambda: mock_llm)

        assert await _detect_conflict({"summary": "old"}, {"summary": "new"}) is False

    @pytest.mark.asyncio
    async def test_propagates_on_exhaustion(self, monkeypatch) -> None:
        """Never silently assumes no-conflict when structured output fails."""

        async def _raise(*args, **kwargs):
            raise LLMStructuredError("no schema-valid JSON after retries")

        monkeypatch.setattr("backend.service.structured.chat_structured", _raise)

        with pytest.raises(LLMStructuredError):
            await _detect_conflict({"summary": "old"}, {"summary": "new"})
