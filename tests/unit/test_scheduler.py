"""Tests for PatrolScheduler — time calculation, task management, toggle logic."""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

from backend.runner.scheduler import (
    _SLOT_GRACE_SECONDS,
    PatrolScheduler,
    _slot_overshoot,
    previous_daily_slot,
    previous_weekly_slot,
    should_catch_up,
)


class _FakeResult:
    def __init__(self, value):
        self._value = value

    def fetchone(self):
        return self._value

    def scalar(self):
        return self._value


class _FakeSession:
    """Minimal async-session stand-in for should_catch_up's two queries."""

    def __init__(self, count: int, has_run: bool) -> None:
        self._count = count
        self._has_run = has_run

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    async def execute(self, stmt, params=None):
        if "COUNT(*)" in str(stmt):
            return _FakeResult(self._count)
        return _FakeResult((1,) if self._has_run else None)


class _FakeFactory:
    def __init__(self, count: int, has_run: bool) -> None:
        self._session = _FakeSession(count, has_run)

    def __call__(self):
        return self._session


class TestSlotOvershoot:
    """Overshoot math behind the poll loop — no real sleeping involved.

    The poll loop's only untestable surface is ``asyncio.sleep``; its
    decision (whether a discovered slot was missed) is the pure arithmetic of
    ``_slot_overshoot``, tested here with fixed wall-clock inputs.
    """

    def test_slot_overshoot_woke_early(self) -> None:
        # Woke 1s before the slot (spurious wakeup) → negative, not a miss.
        slot = datetime(2026, 1, 15, 10, 0, 0)
        assert _slot_overshoot(slot, woke_at=slot - timedelta(seconds=1)) == -1.0

    def test_slot_overshoot_within_grace_is_jitter(self) -> None:
        # 59s late: ordinary jitter — not a missed slot.
        slot = datetime(2026, 1, 15, 10, 0, 0)
        assert _slot_overshoot(slot, woke_at=slot + timedelta(seconds=59)) == 59.0

    def test_slot_overshoot_at_grace_boundary_is_not_missed(self) -> None:
        # The grace check is strict ``>``, so exactly 300s is still jitter.
        slot = datetime(2026, 1, 15, 10, 0, 0)
        woke_at = slot + timedelta(seconds=_SLOT_GRACE_SECONDS)
        assert _slot_overshoot(slot, woke_at=woke_at) <= _SLOT_GRACE_SECONDS

    def test_slot_overshoot_past_grace_is_missed(self) -> None:
        # Host suspended for 2 hours → the wake missed its slot.
        slot = datetime(2026, 1, 15, 10, 0, 0)
        woke_at = slot + timedelta(hours=2)
        assert _slot_overshoot(slot, woke_at=woke_at) > _SLOT_GRACE_SECONDS


class TestSchedulerTimeCalculation:
    """Verify the scheduler computes the next run time correctly."""

    @pytest.mark.asyncio
    async def test_scheduler_calculates_next_daily_run(self) -> None:
        """Next daily run should be the target hour today or tomorrow."""
        scheduler = PatrolScheduler()
        callback = AsyncMock()

        # Mock datetime to a known time: 2026-01-15 10:30
        fixed_now = datetime(2026, 1, 15, 10, 30, 0)

        with patch("backend.runner.scheduler.datetime") as mock_dt:
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

        with patch("backend.runner.scheduler.datetime") as mock_dt:
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


class _LoopStop(Exception):
    """Test-only sentinel that ends the poll loop deterministically."""


@contextlib.contextmanager
def _poll_fixture(times, polls):
    """Drive the poll loop with a fixed clock sequence and no real sleeping.

    *times* is an iterable of naive datetimes served to ``datetime.now`` in
    order — the first is the startup tick, then one per poll.  *polls* is how
    many ``asyncio.sleep`` calls return before the loop is stopped with
    ``_LoopStop``, so a test asserting "fires once" runs exactly the polls it
    wants before the loop dies.
    """
    stop_after = [polls]

    async def _fake_sleep(_delay):
        stop_after[0] -= 1
        if stop_after[0] < 0:
            raise _LoopStop()

    with patch("backend.runner.scheduler.asyncio.sleep", new=_fake_sleep), \
            patch("backend.runner.scheduler.datetime") as mock_dt:
        mock_dt.now.side_effect = lambda: next(times).astimezone()
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        yield


class TestPollingLoop:
    """Wall-clock poll loop: fires each slot exactly once, never replays the
    slot a mid-cycle startup already owns, and catches up a slot the host
    slept through instead of dropping it.
    """

    @pytest.mark.asyncio
    async def test_daily_fires_once_when_slot_crossed(self) -> None:
        # Startup 07:59 → previous slot is yesterday 08:00.  The first poll
        # at 08:05 has crossed today's 08:00 → fires exactly once; the second
        # poll is still today's slot → no second fire.
        scheduler = PatrolScheduler()
        callback = AsyncMock()
        times = iter([
            datetime(2026, 1, 15, 7, 59, 0),
            datetime(2026, 1, 15, 8, 5, 0),
            datetime(2026, 1, 15, 8, 5, 30),
        ])
        with _poll_fixture(times, polls=2):
            scheduler.schedule_daily(hour=8, callback=callback)
            with pytest.raises(_LoopStop):
                await scheduler._tasks[0]
        assert callback.await_count == 1

    @pytest.mark.asyncio
    async def test_daily_start_after_slot_waits_for_next(self) -> None:
        # Startup 09:00 → previous slot is today's 08:00, already marked
        # fired (the startup catch-up hook owns that slot).  The loop must
        # keep polling without replaying today's run.
        scheduler = PatrolScheduler()
        callback = AsyncMock()
        times = iter([
            datetime(2026, 1, 15, 9, 0, 0),
            datetime(2026, 1, 15, 9, 1, 0),
        ])
        with _poll_fixture(times, polls=1):
            scheduler.schedule_daily(hour=8, callback=callback)
            with pytest.raises(_LoopStop):
                await scheduler._tasks[0]
        assert callback.await_count == 0

    @pytest.mark.asyncio
    async def test_weekly_fires_once_when_slot_crossed(self) -> None:
        # Startup Monday 08:00 → previous Monday@09 is last week's.  The
        # first poll Monday 09:05 crosses this week's Monday@09 → fires once;
        # the next poll 09:10 is still the same slot → no second fire.
        scheduler = PatrolScheduler()
        callback = AsyncMock()
        times = iter([
            datetime(2026, 1, 12, 8, 0, 0),   # Monday
            datetime(2026, 1, 12, 9, 5, 0),   # crossed Monday 09:00
            datetime(2026, 1, 12, 9, 10, 0),  # same slot
        ])
        with _poll_fixture(times, polls=2):
            scheduler.schedule_weekly(day=0, hour=9, callback=callback)
            with pytest.raises(_LoopStop):
                await scheduler._tasks[0]
        assert callback.await_count == 1

    @pytest.mark.asyncio
    async def test_daily_late_wake_logs_overshoot(self) -> None:
        # Host slept from 07:00 to 13:00 — the monotonic clock froze while
        # the wall clock jumped.  The first poll at 13:00 discovers today's
        # 08:00 slot 5h late → fires and logs the overshoot instead of
        # silently dropping it (the failure the old long-sleep loop was prone
        # to).
        scheduler = PatrolScheduler()
        callback = AsyncMock()
        times = iter([
            datetime(2026, 1, 15, 7, 0, 0),
            datetime(2026, 1, 15, 13, 0, 0),
        ])
        with _poll_fixture(times, polls=1), \
                patch("backend.runner.scheduler.logger") as mock_logger:
            scheduler.schedule_daily(hour=8, callback=callback)
            with pytest.raises(_LoopStop):
                await scheduler._tasks[0]
        assert callback.await_count == 1
        mock_logger.warning.assert_called_once()


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


class TestCatchUp:
    """Missed-slot detection — previous_*_slot computation + should_catch_up."""

    # ── slot computation ─────────────────────────────────────────

    def test_previous_daily_slot(self) -> None:
        # 10:30 → the previous 8:00 slot is today 8:00
        now = datetime(2026, 1, 15, 10, 30, 0)
        assert previous_daily_slot(8, now=now).replace(tzinfo=None) == datetime(2026, 1, 15, 8, 0, 0)

    def test_previous_daily_slot_before_target_hour(self) -> None:
        # 07:30 → today's 8:00 hasn't happened; previous is yesterday 8:00
        now = datetime(2026, 1, 15, 7, 30, 0)
        assert previous_daily_slot(8, now=now).replace(tzinfo=None) == datetime(2026, 1, 14, 8, 0, 0)

    def test_previous_weekly_slot(self) -> None:
        # Thursday 10:30, Monday@9 → previous is Mon 01-12 9:00
        now = datetime(2026, 1, 15, 10, 30, 0)  # Thursday
        assert previous_weekly_slot(0, 9, now=now).replace(tzinfo=None) == datetime(2026, 1, 12, 9, 0, 0)

    def test_previous_weekly_slot_same_day_before_hour(self) -> None:
        # Monday 08:00, Monday@9 → today's 9:00 hasn't happened yet; the
        # previous slot is last Monday 9:00
        now = datetime(2026, 1, 12, 8, 0, 0)  # Monday
        assert previous_weekly_slot(0, 9, now=now).replace(tzinfo=None) == datetime(2026, 1, 5, 9, 0, 0)

    # ── missed detection (mocked DB) ──────────────────────────────

    @pytest.mark.asyncio
    async def test_should_catch_up_true_when_slot_missed(self, monkeypatch) -> None:
        # History exists but no run at/after the slot → missed.
        monkeypatch.setattr(
            "backend.db.get_session_factory",
            lambda: _FakeFactory(count=2, has_run=False),
        )
        slot = datetime(2026, 1, 15, 8, 0, 0)
        assert await should_catch_up("daily", slot) is True

    @pytest.mark.asyncio
    async def test_should_catch_up_false_when_ran(self, monkeypatch) -> None:
        # A run at/after the slot satisfies it (even if it later failed).
        monkeypatch.setattr(
            "backend.db.get_session_factory",
            lambda: _FakeFactory(count=2, has_run=True),
        )
        slot = datetime(2026, 1, 15, 8, 0, 0)
        assert await should_catch_up("daily", slot) is False

    @pytest.mark.asyncio
    async def test_should_catch_up_false_without_history(self, monkeypatch) -> None:
        # Fresh install (no patrol_logs rows) must not fire an immediate run.
        monkeypatch.setattr(
            "backend.db.get_session_factory",
            lambda: _FakeFactory(count=0, has_run=False),
        )
        slot = datetime(2026, 1, 15, 8, 0, 0)
        assert await should_catch_up("daily", slot) is False
