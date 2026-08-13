"""Structured logging setup — JSON output for log collection pipelines.

EMA logs to stdout (the container's ``docker compose logs``), which is
where any log shipper (Loki / ELK / vector) picks it up.  This module adds
two things on top of the default text format:

- ``LOG_FORMAT=json`` switches stdout lines to a single-line JSON object
  per record — machine-parseable without a grok regex, so a collector can
  index by level / logger / trace_id directly.
- The ``ts`` field is UTC ISO-8601 (``docker compose logs`` timestamps are
  local and not in a parseable shape); ``trace_id`` / ``thread_id`` come
  from the same contextvars the usage tracing uses, so a run is followable
  across log lines without extra plumbing.

Design constraints
------------------
- **Backwards-compatible default.**  ``LOG_FORMAT`` unset / ``text`` keeps
  the current human-readable format exactly.  ``json`` is opt-in so existing
  log consumers are unaffected.
- **Idempotent setup.**  ``setup_logging()`` replaces the root handler in
  place rather than appending, so it can be called again (tests, reload)
  without duplicating output lines.
- **No new dependency.**  Uses ``logging`` + ``json`` stdlib; the trace /
  thread contextvars are already imported from ``backend.shared.config``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime

from backend.shared.config import current_thread_id, current_trace_id


class JsonFormatter(logging.Formatter):
    """Render a log record as one JSON object with stable field names."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # The run context is set via contextvars in the API layer; attach it
        # so a whole agent run / request is greppable by one id.  Missing ids
        # (background tasks, startup) simply omit the field.
        try:
            trace_id = current_trace_id.get()
            if trace_id:
                payload["trace_id"] = trace_id
            thread_id = current_thread_id.get()
            if thread_id:
                payload["thread_id"] = thread_id
        except Exception:
            pass
        # Record provenance — cheap and useful for debugging which call site
        # emitted the line.
        payload["func"] = record.funcName or ""
        payload["line"] = record.lineno
        if record.exc_info:
            exc_info = record.exc_info
            # Logger._log normally converts exc_info=True to the sys.exc_info()
            # tuple, but a LogRecord built directly may still carry the raw
            # True — resolve it here so formatting never crashes the log path.
            if exc_info is True:
                exc_info = sys.exc_info()
            payload["exc"] = self.formatException(exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging() -> None:
    """Configure the root logger once: level from LOG_LEVEL, format from
    LOG_FORMAT (``json`` or ``text``).  Replaces any existing StreamHandler
    so repeated calls (tests) never double-print."""
    level = getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO)
    fmt = os.getenv("LOG_FORMAT", "text").lower()

    root = logging.getLogger()
    # Drop existing handlers so setup is idempotent — but keep any handler
    # pytest installed (caplog) by only removing plain StreamHandlers.
    for handler in list(root.handlers):
        if type(handler) is logging.StreamHandler:
            root.removeHandler(handler)

    handler = logging.StreamHandler()
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    root.addHandler(handler)
    root.setLevel(level)
    for _name in ("backend", "backend.agent"):
        logging.getLogger(_name).setLevel(level)
