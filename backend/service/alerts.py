"""LLM health alerting — error rate / structured failures / circuit breaker.

The ``llm_usage`` table records every LLM call and the metrics module counts
structured-output degradations, but until this module nothing *reacts* when
those signals turn bad.  :func:`check_alerts` inspects a recent window and
surfaces a problem as a WARNING log (always) and, when
``ALERT_FEISHU_ENABLED=true``, a 飞书 webhook message.

Checks (independent):

1. **LLM error rate** — error calls / total calls in the last window (10 min)
   at/above ``ALERT_ERROR_RATE_THRESHOLD``.  A minimum-calls guard prevents a
   tiny sample (e.g. 1/2) from firing.
2. **Structured-output failures** — the in-memory failure counters in
   ``metrics.py`` grew by 5+ since the previous check.  These are extraction
   degradations that ``chat_structured`` logged as failures.
3. **Circuit breaker** — the primary LLM provider's breaker is open (provider
   failing fast).

Cooldown: an alert *kind* is notified at most once per cooldown window (1 h),
so a persistent condition is reported once instead of spamming every cycle.
Log-only by default (``ALERTS_ENABLED=true``, no side effect); the 飞书
notification is opt-in and reuses ``FEISHU_WEBHOOK_URL`` — external
notifications are never on without explicit configuration.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

from sqlalchemy import text

from backend.db import get_session_factory
from backend.service.llm_service import primary_breaker_name
from backend.shared.config import config
from backend.shared.metrics import get_structured_failures
from backend.shared.resilience import get_circuit_breaker

logger = logging.getLogger(__name__)

# ── Alert thresholds (window / guards / cooldown) ────────────────────
# Hard-coded rather than config: they tune alert *cadence*, not behaviour,
# and the defaults are sane for an engineering tool.  Config carries the
# user-facing knobs (error-rate threshold, check interval, 飞书 on/off).
_ERROR_WINDOW_SECONDS = 600  # look-back window for the error-rate check
_MIN_CALLS = 5               # below this many calls in the window, no alert
_STRUCTURED_FAILURE_THRESHOLD = 5  # per-scenario growth between checks
_ALERT_COOLDOWN_SECONDS = 3600     # min gap between notifications per kind

# ── In-memory state (same pattern as metrics.py / usage.py) ───────────

_alerts_lock = threading.Lock()
_last_fired: dict[str, float] = {}  # alert key -> monotonic ts of last notify
_prev_structured_failures: dict[str, int] = {}  # scenario -> last-seen count


def reset_alert_state() -> None:
    """Drop cooldown + counter baselines — tests use this for isolation."""
    global _prev_structured_failures
    with _alerts_lock:
        _last_fired.clear()
        _prev_structured_failures = {}


def _cooldown_ok(key: str) -> bool:
    """True if *key* is not currently in cooldown; records the fire time."""
    now = time.monotonic()
    with _alerts_lock:
        last = _last_fired.get(key)
        if last is not None and now - last < _ALERT_COOLDOWN_SECONDS:
            return False
        _last_fired[key] = now
        return True


# ── Individual checks ─────────────────────────────────────────────────


async def _error_window_stats() -> dict[str, Any]:
    """Calls / errors / error_rate in the last ``_ERROR_WINDOW_SECONDS``."""
    async with get_session_factory()() as session:
        result = await session.execute(
            text(
                """\
                SELECT COUNT(*) AS calls,
                       COUNT(*) FILTER (WHERE status = 'error') AS errors
                FROM llm_usage
                WHERE created_at >= now() - make_interval(secs => :window)
                """
            ),
            {"window": _ERROR_WINDOW_SECONDS},
        )
        row = result.fetchone()
    calls = int(row.calls or 0)
    errors = int(row.errors or 0)
    return {
        "calls": calls,
        "errors": errors,
        "error_rate": errors / calls if calls else 0.0,
    }


def _structured_failure_growth() -> list[tuple[str, int]]:
    """(scenario, growth) pairs since the previous check, threshold-passing."""
    current = get_structured_failures()
    grown: list[tuple[str, int]] = []
    for scenario, count in current.items():
        prev = _prev_structured_failures.get(scenario, 0)
        growth = count - prev
        _prev_structured_failures[scenario] = count
        if growth >= _STRUCTURED_FAILURE_THRESHOLD:
            grown.append((scenario, growth))
    # Scenarios that stopped failing keep a stale baseline — harmless, they
    # simply stop contributing growth until their counter moves again.
    return grown


# ── Alert firing + notification ───────────────────────────────────────


async def _notify_feishu(alert: dict[str, Any]) -> None:
    """Best-effort 飞书 webhook push (only called when opt-in + configured)."""
    import httpx

    message = (
        f"[EMA ALERT] {alert['severity'].upper()}: {alert['key']} — "
        f"{alert['detail']}"
    )
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                config.feishu_webhook_url,
                json={"msg_type": "text", "content": {"text": message}},
            )
            resp.raise_for_status()
        logger.info("Alert %s pushed to 飞书", alert["key"])
    except Exception:
        logger.exception("Failed to push alert %s to 飞书", alert["key"])


async def check_alerts() -> list[dict[str, Any]]:
    """Run all checks; notify for any that crossed a threshold + cooldown.

    Returns the alerts fired this cycle (empty when nothing crossed a
    threshold or everything is in cooldown).  Best-effort — a failing check
    logs and is skipped; it must never break the caller's loop.
    """
    fired: list[dict[str, Any]] = []

    # 1. LLM error rate (DB-backed window).
    try:
        stats = await _error_window_stats()
        if (
            stats["calls"] >= _MIN_CALLS
            and stats["error_rate"] >= config.alert_error_rate_threshold
        ):
            alert = {
                "key": "llm_error_rate",
                "severity": "warning",
                "detail": (
                    f"LLM error rate {stats['error_rate']:.0%} "
                    f"({stats['errors']}/{stats['calls']} calls) in the last "
                    f"{_ERROR_WINDOW_SECONDS // 60} min"
                ),
            }
            if _cooldown_ok(alert["key"]):
                fired.append(alert)
    except Exception:
        logger.exception("Error-rate alert check failed")

    # 2. Structured-output failures (in-memory counters, growth since last).
    try:
        for scenario, growth in _structured_failure_growth():
            alert = {
                "key": f"structured_failure:{scenario}",
                "severity": "warning",
                "detail": (
                    f"structured output degraded {growth}x for scenario="
                    f"{scenario} since the last check"
                ),
            }
            if _cooldown_ok(alert["key"]):
                fired.append(alert)
    except Exception:
        logger.exception("Structured-failure alert check failed")

    # 3. Primary LLM circuit breaker open.
    try:
        if get_circuit_breaker(primary_breaker_name()).is_open:
            alert = {
                "key": "llm_circuit_open",
                "severity": "critical",
                "detail": "primary LLM provider circuit breaker is open — "
                "provider calls are failing fast",
            }
            if _cooldown_ok(alert["key"]):
                fired.append(alert)
    except Exception:
        logger.exception("Circuit-breaker alert check failed")

    # Notify every fired alert: WARNING log always; 飞书 when opted in.
    for alert in fired:
        logger.warning("ALERT %s: %s", alert["key"], alert["detail"])
        if config.alert_feishu_enabled and config.feishu_webhook_url:
            await _notify_feishu(alert)

    return fired


async def alerts_loop() -> None:
    """Background task: periodically run :func:`check_alerts`."""
    while True:
        await asyncio.sleep(config.alert_check_interval_seconds)
        try:
            await check_alerts()
        except Exception:
            logger.exception("Alert check failed")
