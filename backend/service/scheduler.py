"""Patrol Scheduler — lightweight asyncio-based cron for proactive agent patrols.

Uses ``asyncio.sleep`` loops rather than external task queues (APScheduler,
Celery, etc.).  Schedules are simple — daily at a fixed hour, weekly at a
fixed day+hour — so a persistent loop is sufficient.  Restarting the server
may skip at most one patrol run; this is an acceptable trade-off.

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
                now = datetime.now()
                next_run = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                if next_run <= now:
                    next_run += timedelta(days=1)
                wait_seconds = (next_run - now).total_seconds()
                logger.debug(
                    "Daily patrol next run: %s (in %.0f seconds)",
                    next_run.isoformat(),
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)
                try:
                    await callback()
                except Exception:
                    logger.exception("Daily patrol callback failed")

        self._tasks.append(asyncio.create_task(_loop()))

    def schedule_weekly(
        self, day: int, hour: int, callback: ScheduleCallback
    ) -> None:
        """Run *callback* once per week on the given *day* (0=Mon, 6=Sun)
        at the given *hour* (0-23)."""

        async def _loop() -> None:
            days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            label = days[day] if 0 <= day <= 6 else f"day={day}"
            logger.info("Weekly patrol scheduled on %s at %02d:00", label, hour)
            while True:
                now = datetime.now()
                next_run = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                # Advance to the target day-of-week
                days_ahead = day - now.weekday()
                if days_ahead < 0 or (days_ahead == 0 and next_run <= now):
                    days_ahead += 7
                next_run += timedelta(days=days_ahead)
                wait_seconds = (next_run - now).total_seconds()
                logger.debug(
                    "Weekly patrol next run: %s (in %.0f seconds)",
                    next_run.isoformat(),
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)
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
