"""API authentication — shared FastAPI dependency guarding all ``/api`` routes.

Security model
--------------
Every route under ``backend.api.router.api_router`` (memory write/search,
ingest, connectors, patrol, agent chat, webhooks, ...) requires a valid
``Authorization: Bearer <key>`` header where ``<key>`` equals ``EMA_API_KEY``.
Unauthenticated or mismatched requests receive a generic 401 with no detail
that could help an attacker: the supplied key is never echoed back and the
message gives no hint about *why* the check failed.

``APP_ENV=test`` bypasses the guard entirely — the API test suite exercises
real route handlers with mocked providers and must not need a key.  See
``tests/conftest.py``, which sets ``APP_ENV=test`` before any import.
"""

from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException, status

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Invalid or missing API key",
    headers={"WWW-Authenticate": "Bearer"},
)


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    """Reject the request unless it carries a valid ``Bearer`` API key.

    Bypassed when ``APP_ENV=test`` (tests rely on mocks).  In every other
    environment a missing/empty ``EMA_API_KEY``, a missing header, a non
    ``Bearer`` scheme, or a key that does not match the configured one all
    raise a 401.  The comparison is constant-time (``secrets.compare_digest``)
    to avoid timing side-channels on the key.
    """
    if os.environ.get("APP_ENV") == "test":
        return

    expected = os.environ.get("EMA_API_KEY")
    if not expected:
        raise _UNAUTHORIZED

    if authorization is None:
        raise _UNAUTHORIZED

    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise _UNAUTHORIZED

    if not secrets.compare_digest(
        token.strip().encode("utf-8"),
        expected.encode("utf-8"),
    ):
        raise _UNAUTHORIZED
