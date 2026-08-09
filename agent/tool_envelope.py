"""Agent tool-result envelope — the single ``{"display", "sources"}`` contract.

Retrieval tools return a JSON envelope so the UI can render sources and the
LLM reads a clean display line.  This module is the only place that envelope
is built and parsed — callers never hand-roll ``json.dumps`` / ``json.loads``,
and tool-result truncation is always envelope-aware: the display text is
unwrapped *first*, then capped.  Truncating the raw envelope instead would
turn a long search result into a half-cut JSON blob the ReAct loop re-reads
as assistant history.

Non-envelope tool results (plain text, write/ingest/entity/notify JSON)
round-trip untouched: parsing fails, the raw text is returned verbatim.
"""

from __future__ import annotations

import json
from typing import Any

# Default per-ToolMessage content cap when resent to the LLM.  Search /
# retrieval results can be thousands of characters; the model only needs
# the relevant head.
MAX_TOOL_CONTENT_CHARS = 800


def build_tool_envelope(display: str, sources: list[Any] | None = None) -> str:
    """Wrap *display* (+ optional *sources*) into the tool-result JSON envelope.

    The envelope's ``display`` field is what the LLM reads (and what nodes
    re-send on later turns); ``sources`` carries structured references for
    the UI.  The only constructor — tools never assemble this by hand.
    """
    return json.dumps(
        {"display": display, "sources": sources or []},
        ensure_ascii=False,
    )


def parse_tool_envelope(raw: str) -> dict[str, Any] | None:
    """Parse a tool-result envelope; ``None`` when *raw* is not one.

    An envelope is a JSON object carrying a ``display`` field.  Plain text,
    non-JSON garbage, and JSON objects without ``display`` (e.g. a
    write/ingest/entity result) all return ``None`` so callers fall back to
    the raw text — never raises.
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(parsed, dict) and "display" in parsed:
        return parsed
    return None


def envelope_display(raw: str) -> str:
    """The display text of a tool result, unwrapped from its envelope.

    Returns *raw* verbatim when it is not an envelope.
    """
    parsed = parse_tool_envelope(raw)
    if parsed is not None:
        return str(parsed["display"])
    return raw


def truncate_tool_content(raw: str, limit: int = MAX_TOOL_CONTENT_CHARS) -> str:
    """Cap a ToolMessage's content for the LLM, unwrapping the envelope first.

    Search/retrieval results can be thousands of characters; the model only
    needs the relevant head.  The truncation marker keeps the model aware the
    result was longer than shown.  For an envelope the ``display`` text is
    extracted *before* truncating — ``sources`` alone can exceed the cap, and
    cutting the raw JSON would fall back to sending the model truncated raw
    JSON instead of the clean display line.  Non-envelope text is truncated
    verbatim.
    """
    display = envelope_display(raw)
    if len(display) <= limit:
        return display
    return display[:limit] + "\n…[truncated]"
