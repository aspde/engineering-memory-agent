"""Verify that write_memory schedules entity normalisation after insert/merge."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest


class TestScheduleNormalization:
    @pytest.mark.asyncio
    async def test_schedule_normalization_creates_task(self):
        """_schedule_normalization creates a background task that runs normalize_entities."""
        from backend.service.memory import _schedule_normalization

        with patch("backend.service.memory.normalize_entities") as mock_norm:
            mock_norm.return_value = ["e-1"]

            _schedule_normalization("mem-1", [{"name": "Test", "type": "concept"}])

            # Allow the background task to run
            await asyncio.sleep(0.1)

            mock_norm.assert_called_once()
            args = mock_norm.call_args[0]
            assert args[0] == "mem-1"
            assert args[1] == [{"name": "Test", "type": "concept"}]

    @pytest.mark.asyncio
    async def test_schedule_normalization_swallows_errors(self):
        """When normalize_entities raises, the error is logged not propagated."""
        from backend.service.memory import _schedule_normalization

        with patch("backend.service.memory.normalize_entities") as mock_norm:
            mock_norm.side_effect = RuntimeError("Boom")

            # Should not raise
            _schedule_normalization("mem-2", [{"name": "Bad", "type": "concept"}])

            await asyncio.sleep(0.1)

            mock_norm.assert_called_once()
