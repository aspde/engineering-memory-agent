"""API rate limiting — per-key token bucket guard for ``/api`` routes.

Why
---
``MAX_AGENT_CONCURRENCY`` bounds how many agent runs overlap, but not how
many calls an unbounded client can fire: one caller can loop
``POST /api/agent/chat`` forever, each round costing real LLM tokens
(measured ~28.6k tokens / round).  This middleware caps request volume per
API key so an abusive / runaway client is limited to a bounded cost.

Design
------
- **Per-key buckets.**  The bucket key is the ``Authorization: Bearer``
  token (the API key itself); requests without a token share an
  ``"anonymous"`` bucket.  EMA's auth is a single shared key, so every
  caller — including someone who lifted the public ``VITE_EMA_API_KEY``
  from the frontend bundle — is throttled by the same bucket.
- **Two tiers.**  LLM-dense endpoints get a stricter bucket than the rest:
  ``chat`` (``/api/agent/chat*``, ``/api/scenarios*``) and ``general``
  (every other ``/api/*`` route).  Costs are dominated by chat / scenario
  runs, so they are the tier that actually needs throttling.
- **Token bucket, process-local.**  Same class of state as the circuit
  breakers and the agent concurrency counter: correct for the documented
  single-instance deployment, and — like them — each replica would count
  independently under multi-instance scaling (see deployment.md "单实例部署
  约束").  Buckets idle past ``_IDLE_TIMEOUT`` are pruned lazily so the
  table never grows unboundedly with distinct keys.
- **Bypassed in ``APP_ENV=test``** — the API test suite drives real route
  handlers through ``ASGITransport`` and must not be throttled.  Same
  convention as ``backend/api/auth.py``.

Implementation note: this is a **pure-ASGI** middleware, not a
``BaseHTTPMiddleware``.  The streaming chat endpoint (`/api/agent/chat/stream`)
serves an SSE response; ``BaseHTTPMiddleware`` buffers/intercepts response
bodies and breaks streaming.  Like ``MetricsMiddleware``, the limiter talks
directly to the ``send`` callable.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time

from backend.shared.config import config

# API paths exempt from limiting — probes, scrape targets and static assets
# must never be throttled (health checks come from load balancers / k8s).
_API_PREFIX = "/api/"

# High-cost tiers — agent chat (SSE + non-streaming) and scenario runs both
# hold LLM slots for their whole run, so they are throttled more tightly.
_CHAT_TIER_PREFIXES = ("/api/agent/chat", "/api/scenarios")


def _tier_for_path(path: str) -> str:
    return "chat" if path.startswith(_CHAT_TIER_PREFIXES) else "general"


class _Bucket:
    __slots__ = ("tokens", "rate", "last_refill", "last_access")

    def __init__(self, capacity: float, rate: float, now: float) -> None:
        self.tokens = capacity
        self.rate = rate  # tokens per second
        self.last_refill = now
        self.last_access = now


class RateLimiter:
    """Per-(tier, key) token-bucket limiter.

    ``allow`` refills the bucket from the elapsed time, consumes one token
    on approval, and reports how long the caller must wait when it is empty.
    All state is process-local and guarded by a lock (defensive — the
    middleware runs on a single event loop, but the limiter is a shared
    singleton).
    """

    _IDLE_TIMEOUT = 600.0  # seconds — prune buckets unused this long
    _MAX_BUCKETS = 4096    # hard cap on distinct (tier, key) pairs

    def __init__(self) -> None:
        self._buckets: dict[tuple[str, str], _Bucket] = {}
        self._lock = threading.Lock()

    def allow(
        self,
        *,
        tier: str,
        key: str,
        requests: int,
        window_seconds: int,
    ) -> tuple[bool, int]:
        """Decide whether one request from *key* in *tier* may proceed.

        Returns ``(allowed, retry_after)`` where ``retry_after`` is the
        number of seconds to wait until a token is available (0 when
        allowed).  A fresh bucket is always allowed; the first request in a
        window never gets a 429.
        """
        now = time.monotonic()
        rate = requests / window_seconds if window_seconds > 0 else float("inf")
        with self._lock:
            bucket = self._buckets.get((tier, key))
            if bucket is None:
                if len(self._buckets) >= self._MAX_BUCKETS:
                    self._prune(now)
                bucket = _Bucket(float(requests), rate, now)
                # A fresh bucket is full; the request that created it consumes
                # a token like any other, so a quota of N admits exactly N
                # back-to-back requests (not N+1).
                bucket.tokens -= 1.0
                self._buckets[(tier, key)] = bucket
                return True, 0
            bucket.last_access = now
            bucket.tokens = min(
                float(requests),
                bucket.tokens + (now - bucket.last_refill) * bucket.rate,
            )
            bucket.last_refill = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, 0
            # Empty — how long until a whole token accrues.
            wait = (1.0 - bucket.tokens) / bucket.rate if bucket.rate > 0 else 1.0
            return False, max(1, math.ceil(wait))

    def _prune(self, now: float) -> None:
        """Drop buckets not touched within ``_IDLE_TIMEOUT``."""
        expired = [
            key
            for key, b in self._buckets.items()
            if now - b.last_access > self._IDLE_TIMEOUT
        ]
        for key in expired:
            self._buckets.pop(key, None)

    def reset(self) -> None:
        """Clear every bucket — tests and config reloads use this."""
        with self._lock:
            self._buckets.clear()


# Process-local singleton the middleware reads at request time (so runtime
# config changes take effect immediately).  ``reset_rate_limits`` lets the
# test suite isolate state between cases.
_limiter = RateLimiter()


def reset_rate_limits() -> None:
    _limiter.reset()


def _extract_key(scope: dict) -> str:
    """The Bearer token from the ``Authorization`` header, or ``anonymous``."""
    for name, value in scope.get("headers") or ():
        if name.lower() == b"authorization":
            text = value.decode("latin-1")
            scheme, _, token = text.partition(" ")
            if scheme.lower() == "bearer" and token.strip():
                return token.strip()
            return "anonymous"
    return "anonymous"


def _read_limit_config(tier: str) -> tuple[int, int]:
    if tier == "chat":
        return config.rate_limit_chat_requests, config.rate_limit_chat_window_seconds
    return config.rate_limit_general_requests, config.rate_limit_general_window_seconds


async def _send_too_many(send, retry_after: int) -> None:
    body = json.dumps({"detail": "请求过于频繁，请稍后重试。"}, ensure_ascii=False).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"retry-after", str(retry_after).encode("ascii")),
    ]
    await send({"type": "http.response.start", "status": 429, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class RateLimitMiddleware:
    """ASGI middleware that rejects /api requests beyond their per-key quota.

    Mounted on the app in ``backend/main.py``; request flow is
    ``MetricsMiddleware → RateLimitMiddleware → routes`` (metrics is added
    *after* the limiter, so it sits outer and still records the 429s the
    limiter produces — rejections stay visible in the HTTP status
    distribution).
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "") or ""
        if not path.startswith(_API_PREFIX):
            await self.app(scope, receive, send)
            return
        # Same bypass convention as require_api_key: the API test suite runs
        # real handlers through ASGITransport and must not be throttled.
        if os.environ.get("APP_ENV") == "test" or not config.rate_limit_enabled:
            await self.app(scope, receive, send)
            return

        tier = _tier_for_path(path)
        requests, window = _read_limit_config(tier)
        allowed, retry_after = _limiter.allow(
            tier=tier,
            key=_extract_key(scope),
            requests=requests,
            window_seconds=window,
        )
        if not allowed:
            await _send_too_many(send, retry_after)
            return
        await self.app(scope, receive, send)
