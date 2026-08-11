"""Unit tests for structured logging (``backend/shared/logging_config.py``).

Covers the JSON formatter's field shape and ``setup_logging()``'s
idempotency / format switching.  The trace/thread contextvars are exercised
to prove the run ids end up in the JSON payload (that is the whole point of
structured logs — a run is followable by one id).
"""

from __future__ import annotations

import json
import logging

from backend.shared.config import current_thread_id, current_trace_id
from backend.shared.logging_config import JsonFormatter, setup_logging


class TestJsonFormatter:
    def test_json_contains_core_fields(self) -> None:
        record = logging.LogRecord(
            name="backend.api.routes.agent_routes",
            level=logging.WARNING,
            pathname="agent_routes.py",
            lineno=42,
            msg="agent_chat refused — concurrency cap reached (max=%d)",
            args=(4,),
            exc_info=None,
        )
        parsed = json.loads(JsonFormatter().format(record))
        assert parsed["level"] == "WARNING"
        assert parsed["logger"] == "backend.api.routes.agent_routes"
        assert parsed["msg"] == "agent_chat refused — concurrency cap reached (max=4)"
        assert parsed["func"] == ""
        assert parsed["line"] == 42
        # ISO-8601 UTC timestamp.
        assert parsed["ts"].endswith("+00:00") or parsed["ts"].endswith("Z")

    def test_json_omits_trace_fields_when_unset(self) -> None:
        record = logging.LogRecord(
            name="backend", level=logging.INFO, pathname="x.py", lineno=1,
            msg="plain", args=(), exc_info=None,
        )
        parsed = json.loads(JsonFormatter().format(record))
        assert "trace_id" not in parsed
        assert "thread_id" not in parsed

    def test_json_includes_trace_and_thread_ids(self) -> None:
        current_trace_id.set("trace-1")
        current_thread_id.set("thread-2")
        try:
            record = logging.LogRecord(
                name="backend", level=logging.INFO, pathname="x.py", lineno=1,
                msg="hello", args=(), exc_info=None,
            )
            parsed = json.loads(JsonFormatter().format(record))
            assert parsed["trace_id"] == "trace-1"
            assert parsed["thread_id"] == "thread-2"
        finally:
            current_trace_id.set("")
            current_thread_id.set("")

    def test_json_serializes_exception_info(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            record = logging.LogRecord(
                name="backend", level=logging.ERROR, pathname="x.py", lineno=1,
                msg="failed", args=(), exc_info=True,
            )
            parsed = json.loads(JsonFormatter().format(record))
        assert "exc" in parsed
        assert "ValueError: boom" in parsed["exc"]


class TestSetupLogging:
    def test_text_format_default(self) -> None:
        setup_logging()
        handler = next(
            h for h in logging.getLogger().handlers
            if type(h) is logging.StreamHandler
        )
        assert type(handler.formatter) is logging.Formatter
        assert not isinstance(handler.formatter, JsonFormatter)

    def test_json_format_switches_formatter(self, monkeypatch) -> None:
        monkeypatch.setenv("LOG_FORMAT", "json")
        setup_logging()
        handler = next(
            h for h in logging.getLogger().handlers
            if type(h) is logging.StreamHandler
        )
        assert isinstance(handler.formatter, JsonFormatter)

    def test_setup_is_idempotent_no_duplicate_handlers(self) -> None:
        # Two calls must not append a second StreamHandler (double-printed lines).
        setup_logging()
        setup_logging()
        stream_handlers = [
            h for h in logging.getLogger().handlers
            if type(h) is logging.StreamHandler
        ]
        assert len(stream_handlers) == 1

    def test_level_applied_to_backend_loggers(self, monkeypatch) -> None:
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        setup_logging()
        assert logging.getLogger("backend").level == logging.DEBUG
        assert logging.getLogger("backend.agent").level == logging.DEBUG
