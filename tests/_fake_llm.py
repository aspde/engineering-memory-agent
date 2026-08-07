"""Shared fake streaming-LLM helpers for agent tests.

The agent nodes now consume ``chat_raw_stream`` / ``chat_stream`` (true
token streaming).  These helpers build async-generator factories that an
``AsyncMock`` ``side_effect`` list can hand out — one fresh generator per
call, in order — so tests exercise the streaming code path without a real
LLM.

Usage::

    mock_provider = AsyncMock()
    mock_provider.chat_raw_stream.side_effect = [
        tool_call_stream([{"id": "c1", "name": "search", "args": {}}]),
        content_stream("Here is the answer."),
    ]
    mock_provider.chat_stream.side_effect = [text_stream("Final answer.")]
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import Mock


def raw_stream(*events: dict[str, Any]):
    """Return an async-generator factory yielding raw dict events verbatim."""

    async def _gen(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        for e in events:
            yield e

    return _gen


def content_stream(text: str):
    """``chat_raw_stream`` factory: one content delta for *text*."""
    return raw_stream({"type": "content", "text": text})


def tool_call_stream(tool_calls: list[dict[str, Any]]):
    """``chat_raw_stream`` factory: a trailing tool_calls event."""
    return raw_stream({"type": "tool_calls", "tool_calls": tool_calls})


def text_stream(*texts: str):
    """``chat_stream`` factory: yield each *text* as one string token."""

    async def _gen(*args: Any, **kwargs: Any) -> AsyncIterator[str]:
        for t in texts:
            yield t

    return _gen


def raise_stream(exc: Exception):
    """Factory that raises immediately (simulates a provider outage)."""

    async def _gen(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        raise exc
        yield  # pragma: no cover

    return _gen


def sequential_stream(*factories):
    """Return a plain ``Mock`` handing out one generator per call.

    Binds a sequence of async-generator factories (``content_stream``,
    ``tool_call_stream``, ...) to a mock method: each call pops the next
    factory and returns the generator it builds, in order.  A plain
    ``Mock`` (not ``AsyncMock``) is used so ``async for`` receives the
    generator directly, while ``call_args`` / ``assert_called_*`` still
    work.  Assign it to the mock attribute:
    ``mock.chat_raw_stream = sequential_stream(...)``.
    """

    remaining = list(factories)

    def _fn(*args: Any, **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        if not remaining:
            raise AssertionError(
                "sequential_stream exhausted — more provider calls than factories"
            )
        return remaining.pop(0)(*args, **kwargs)

    return Mock(side_effect=_fn)
