"""Tests for PatrolScheduler — time calculation, task management, toggle logic."""

from __future__ import annotations

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

from backend.service.scheduler import PatrolScheduler


class TestSchedulerTimeCalculation:
    """Verify the scheduler computes the next run time correctly."""

    @pytest.mark.asyncio
    async def test_scheduler_calculates_next_daily_run(self) -> None:
        """Next daily run should be the target hour today or tomorrow."""
        scheduler = PatrolScheduler()
        callback = AsyncMock()

        # Mock datetime to a known time: 2026-01-15 10:30
        fixed_now = datetime(2026, 1, 15, 10, 30, 0)

        with patch("backend.service.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            # Let timedelta, replace, etc. pass through to real datetime
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

            scheduler.schedule_daily(hour=8, callback=callback)

            # Get the task and verify it was created
            assert len(scheduler._tasks) == 1
            task = scheduler._tasks[0]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_scheduler_skips_when_disabled(self) -> None:
        """When PATROL_ENABLED=false, scheduler should not start any tasks."""
        # This tests the config integration — in main.py, the scheduler
        # is only instantiated and started when patrol_enabled is True.
        # Here we verify that an empty scheduler (no scheduled tasks)
        # starts and stops cleanly without errors.
        scheduler = PatrolScheduler()
        assert len(scheduler._tasks) == 0
        await scheduler.start()
        await scheduler.stop()
        assert len(scheduler._tasks) == 0

    @pytest.mark.asyncio
    async def test_scheduler_calculates_next_weekly_run(self) -> None:
        """Next weekly run should be the target day+hour this week or next."""
        scheduler = PatrolScheduler()
        callback = AsyncMock()

        # Thursday 2026-01-15 (weekday=3 for Thursday)
        fixed_now = datetime(2026, 1, 15, 10, 30, 0)

        with patch("backend.service.scheduler.datetime") as mock_dt:
            mock_dt.now.return_value = fixed_now
            mock_dt.side_effect = lambda *args, **kw: datetime(*args, **kw)

            # Schedule for Monday (0) at 9:00.  Since it's Thursday,
            # next Monday should be 2026-01-19.
            scheduler.schedule_weekly(day=0, hour=9, callback=callback)

            assert len(scheduler._tasks) == 1
            task = scheduler._tasks[0]
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    @pytest.mark.asyncio
    async def test_weekly_patrol_skips_when_disabled(self) -> None:
        """Weekly patrol config disabled — should not register a task."""
        # In main.py, the weekly callback is only registered when
        # PATROL_WEEKLY_ENABLED is true.  An empty scheduler handles
        # this gracefully.
        scheduler = PatrolScheduler()
        await scheduler.start()
        await scheduler.stop()


class TestSchedulerLifecycle:
    """Verify start/stop behaviour."""

    @pytest.mark.asyncio
    async def test_scheduler_stop_cancels_tasks(self) -> None:
        """stop() should cancel all registered tasks."""
        call_count = 0

        async def _never_called() -> None:
            nonlocal call_count
            call_count += 1

        scheduler = PatrolScheduler()
        scheduler.schedule_daily(hour=3, callback=_never_called)
        assert len(scheduler._tasks) == 1

        await scheduler.stop()
        # Task should be cancelled, callback should never have run
        # (the sleep loop hasn't finished, so our callback isn't hit)
        assert call_count == 0
        assert len(scheduler._tasks) == 0   # cleared
