"""Patrol Scheduler — lightweight wall-clock poller for proactive patrols.

Runs short ``asyncio.sleep`` polls rather than external task queues
(APScheduler, Celery, etc.).  Schedules are simple — daily at a fixed hour,
weekly at a fixed day+hour — so a persistent polling loop is sufficient.
Polling the wall clock every ``_POLL_INTERVAL_SECONDS`` keeps a slot from
being silently lost to host suspend/hibernate: the monotonic clock that a
long ``asyncio.sleep`` waits on freezes during standby, so a wake after
sleep could otherwise land hours past the slot and push the run to the next
cycle.  A slot missed while the server is down (restart / deploy / crash)
is caught up once by :func:`should_catch_up` + the startup hook in
``backend.main``.

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

# Sleep-precision allowance: a slot discovered more than this many seconds
# late was missed — host suspend/hibernate, a long event-loop stall — not
# ordinary poll jitter.  The poll loop logs that case; the threshold lives
# here so ``_slot_overshoot``'s tests can assert the boundary.
_SLOT_GRACE_SECONDS = 300

# Poll cadence for the scheduler loops.  Each loop wakes this often and
# compares the wall clock against its slot.  A short interval bounds how
# late a slot can be discovered after the host wakes from standby (one
# interval) while keeping the per-tick cost trivial — a couple of
# ``datetime.now`` calls a minute.
_POLL_INTERVAL_SECONDS = 60


def _slot_overshoot(slot: datetime, *, woke_at: datetime | None = None) -> float:
    """Seconds the wake landed past *slot* (negative when it woke early).

    An overshoot beyond ``_SLOT_GRACE_SECONDS`` means the slot was missed —
    host suspend/hibernate, a long event-loop stall — rather than ordinary
    poll jitter.  The caller logs that case but still fires the callback,
    since a late scan beats a dropped one.
    """
    woke_at = (woke_at or datetime.now()).astimezone()
    return (woke_at - slot.astimezone()).total_seconds()


class PatrolScheduler:
    """Manages recurring patrol tasks via asyncio background loops.

    Each scheduled task runs in its own ``asyncio.Task`` with an independent
    sleep loop.  ``stop()`` cancels all tasks and waits for them to finish.
    """

    def __init__(self) -> None:
        self._tasks: list[asyncio.Task[None]] = []

    def _spawn_poll_loop(
        self,
        *,
        label: str,
        start_log: str,
        slot_at: Callable[[datetime], datetime],
        callback: ScheduleCallback,
    ) -> None:
        """Register one wall-clock poll loop as a background task.

        Every loop this scheduler spawns polls the wall clock every
        ``_POLL_INTERVAL_SECONDS`` and fires when the clock crosses a new
        slot, instead of sleeping straight to the slot: a long
        ``asyncio.sleep`` waits on the monotonic clock, which freezes while
        the host suspends/hibernates, so a wake after standby could land
        hours past the slot and push the run to the next cycle.  Polling
        discovers a slot within one interval of the host waking — a late
        scan instead of a dropped one.

        *slot_at* maps a wake time to the most recent schedule slot strictly
        before it; *label* prefixes the overshoot / failure log lines.
        """

        async def _loop() -> None:
            logger.info(start_log)
            # Seed the fired marker with the most recent slot so a mid-day
            # startup waits for the next slot instead of replaying the one
            # the startup catch-up hook already handled.
            fired_for = slot_at(datetime.now().astimezone())
            while True:
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)
                now = datetime.now().astimezone()
                slot = slot_at(now)
                if slot == fired_for:
                    continue
                fired_for = slot
                overshoot = _slot_overshoot(slot, woke_at=now)
                if overshoot > _SLOT_GRACE_SECONDS:
                    logger.warning(
                        "%s woke %.0fs past its %s slot — "
                        "firing catch-up",
                        label,
                        overshoot,
                        slot.isoformat(),
                    )
                try:
                    await callback()
                except Exception:
                    logger.exception("%s callback failed", label)

        self._tasks.append(asyncio.create_task(_loop()))

    def schedule_daily(self, hour: int, callback: ScheduleCallback) -> None:
        """Run *callback* once per day at the given *hour* (0-23).

        Polls the wall clock — see :meth:`_spawn_poll_loop` for why polling
        beats sleeping straight to the slot.
        """
        self._spawn_poll_loop(
            label="Daily patrol",
            start_log=f"Daily patrol scheduled at {hour:02d}:00 each day",
            slot_at=lambda now: previous_daily_slot(hour, now=now),
            callback=callback,
        )

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

        Polls the wall clock like :meth:`schedule_daily` so a week's slot is
        discovered within one poll interval of the host waking from standby,
        not hours late.

        *name* labels the task in the startup log — callers register more
        than one weekly task (patrol scan, tech-debt radar), and without a
        name they all log as "Weekly patrol", which reads like a duplicate
        registration.
        """
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        label = days[day] if 0 <= day <= 6 else f"day={day}"
        self._spawn_poll_loop(
            label=name,
            start_log=f"{name} scheduled on {label} at {hour:02d}:00",
            slot_at=lambda now: previous_weekly_slot(day, hour, now=now),
            callback=callback,
        )

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
