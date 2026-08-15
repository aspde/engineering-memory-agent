"""Patrol Scheduler — lightweight asyncio-based cron for proactive agent patrols.

Uses ``asyncio.sleep`` loops rather than external task queues (APScheduler,
Celery, etc.).  Schedules are simple — daily at a fixed hour, weekly at a
fixed day+hour — so a persistent loop is sufficient.  The loop skips a run
when the server is down at the scheduled time; :func:`should_catch_up` +
the startup hook in ``backend.main`` detect that miss and fire one
catch-up run so a restart doesn't silently drop the patrol for a cycle.

Usage::

    scheduler = PatrolScheduler()
    scheduler.schedule_daily(hour=8, callback=run_daily)
    scheduler.schedule_weekly(day=1, hour=9, callback=run_weekly)
    await scheduler.start()
    # ... server running ...
    await scheduler.stop()
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

ScheduleCallback = Callable[[], Awaitable[None]]

# Sleep-precision allowance: a wake that overshoots its slot by more than
# this (seconds) is a missed slot, not scheduler jitter.  Only used for the
# catch-up log line below.
_SLOT_GRACE_SECONDS = 300


async def _sleep_until(next_run: datetime, what: str) -> float:
    """Sleep until *next_run*; return the overshoot seconds when woken.

    A single long ``asyncio.sleep`` stalls while the host suspends /
    hibernates (Windows freezes the event loop's monotonic clock), so a wake
    can land hours past the slot and the loop would silently push the run to
    the next cycle.  Logging the overshoot surfaces the miss; the caller
    still fires the callback — a late scan beats a dropped one, and
    ``run_patrol``'s overlap guard dedups against a concurrent run.
    """
    now = datetime.now().astimezone()
    wait = (next_run - now).total_seconds()
    if wait > 0:
        await asyncio.sleep(wait)
    overshoot = (datetime.now().astimezone() - next_run).total_seconds()
    if overshoot > _SLOT_GRACE_SECONDS:
        logger.warning(
            "%s woke %.0fs past its %s slot — firing catch-up",
            what,
            overshoot,
            next_run.isoformat(),
        )
    return overshoot


class PatrolScheduler:
    """Manages recurring patrol tasks via asyncio background loops.

    Each scheduled task runs in its own ``asyncio.Task`` with an independent
    sleep loop.  ``stop()`` cancels all tasks and waits for them to finish.
    """

    def __init__(self) -> None:
        self._tasks: list[asyncio.Task[None]] = []

    def schedule_daily(self, hour: int, callback: ScheduleCallback) -> None:
        """Run *callback* once per day at the given *hour* (0-23)."""

        async def _loop() -> None:
            logger.info("Daily patrol scheduled at %02d:00 each day", hour)
            while True:
                now = datetime.now().astimezone()
                next_run = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(days=1)
                logger.debug(
                    "Daily patrol next run: %s (in %.0f seconds)",
                    next_run.isoformat(),
                    (next_run - now).total_seconds(),
                )
                await _sleep_until(next_run, "Daily patrol")
                try:
                    await callback()
                except Exception:
                    logger.exception("Daily patrol callback failed")

        self._tasks.append(asyncio.create_task(_loop()))

    def schedule_weekly(
        self,
        day: int,
        hour: int,
        callback: ScheduleCallback,
        *,
        name: str = "Weekly patrol",
    ) -> None:
        """Run *callback* once per week on the given *day* (0=Mon, 6=Sun)
        at the given *hour* (0-23).

        *name* labels the task in the startup log — callers register more
        than one weekly task (patrol scan, tech-debt radar), and without a
        name they all log as "Weekly patrol", which reads like a duplicate
        registration.
        """

        async def _loop() -> None:
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            label = days[day] if 0 <= day <= 6 else f"day={day}"
            logger.info("%s scheduled on %s at %02d:00", name, label, hour)
            while True:
                now = datetime.now().astimezone()
                next_run = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                # Advance to the target day-of-week
                days_ahead = day - now.weekday()
                if days_ahead < 0 or (days_ahead == 0 and next_run <= now):
                    days_ahead += 7
                next_run += timedelta(days=days_ahead)
                logger.debug(
                    "%s next run: %s (in %.0f seconds)",
                    name,
                    next_run.isoformat(),
                    (next_run - now).total_seconds(),
                )
                await _sleep_until(next_run, name)
                try:
                    await callback()
                except Exception:
                    logger.exception("Weekly patrol callback failed")

        self._tasks.append(asyncio.create_task(_loop()))

    async def start(self) -> None:
        """Start all registered scheduled tasks (they begin sleeping)."""
        if not self._tasks:
            logger.info("PatrolScheduler started with no tasks registered")
        else:
            logger.info("PatrolScheduler started with %d task(s)", len(self._tasks))

    async def stop(self) -> None:
        """Cancel all scheduled tasks and wait for them to complete."""
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            results = await asyncio.gather(*self._tasks, return_exceptions=True)
            for i, r in enumerate(results):
                if isinstance(r, asyncio.CancelledError):
                    continue
                if isinstance(r, Exception):
                    logger.warning("Scheduler task %d raised on shutdown: %s", i, r)
        self._tasks.clear()
        logger.info("PatrolScheduler stopped")


# ── Catch-up ───────────────────────────────────────────────────────────
# The scheduler loop can't fire a run it wasn't alive for.  On startup
# (backend.main) we compare each schedule's most recent slot against the
# patrol_logs: if the patrol has history but nothing started at/after the
# slot, that run was missed and a single catch-up fires.  The history guard
# keeps a fresh install — where "no run since slot" is simply the initial
# state — from running an immediate patrol on first startup.


def previous_daily_slot(hour: int, *, now: datetime | None = None) -> datetime:
    """The most recent daily schedule slot at *hour* strictly in the past.

    Returned as an aware local datetime (the scheduler loop's wall-clock
    semantics, with the local offset attached).  DST is deliberately not
    tracked across the boundary — the offset is the one at call time, so a
    transition within the next day shifts a slot by an hour at worst, which
    the catch-up logic tolerates.
    """
    now = (now or datetime.now()).astimezone()
    slot = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if slot >= now:
        slot -= timedelta(days=1)
    return slot


def previous_weekly_slot(
    day: int, hour: int, *, now: datetime | None = None
) -> datetime:
    """The most recent weekly schedule slot (*day*, *hour*) strictly in the past."""
    now = (now or datetime.now()).astimezone()
    slot = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    days_ahead = day - now.weekday()
    slot += timedelta(days=days_ahead)
    if slot >= now:
        slot -= timedelta(days=7)
    return slot


async def should_catch_up(patrol_type: str, slot: datetime) -> bool:
    """True if the scheduled *slot* was missed and a catch-up run is warranted.

    A slot counts as missed when the patrol has prior history (at least one
    ``patrol_logs`` row) and no run started at/after *slot*.  A run that
    started within the slot — even one that later failed — satisfies it: this
    fills genuinely-missed slots, it does not retry failures (a separate
    concern handled by the next scheduled slot / stale-row marking).

    *slot* is a local wall-clock time, naive or aware; it is normalized to an
    aware instant with the offset at call time (``slot.astimezone()``) so the
    TIMESTAMPTZ comparison in Postgres is against the same instant.  A DST
    transition within a day of the slot shifts it by an hour at worst —
    acceptable for missed-slot detection.
    """
    from sqlalchemy import text

    from backend.db import get_session_factory

    slot = slot.astimezone()

    session_factory = get_session_factory()
    async with session_factory() as session:
        count = await session.execute(
            text("SELECT COUNT(*) FROM patrol_logs WHERE patrol_type = :type"),
            {"type": patrol_type},
        )
        if (count.scalar() or 0) == 0:
            return False
        ran = await session.execute(
            text(
                """SELECT 1 FROM patrol_logs
                   WHERE patrol_type = :type AND started_at >= :since
                   LIMIT 1"""
            ),
            {"type": patrol_type, "since": slot},
        )
        return ran.fetchone() is None
