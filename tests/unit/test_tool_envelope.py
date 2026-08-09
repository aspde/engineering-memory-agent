"""Unit tests for agent/tool_envelope.py — the shared tool-result envelope.

The envelope is the single contract between tools (which build it) and the
nodes / API (which parse and truncate it).  These tests pin the exact shape
and the envelope-aware truncation behaviour — a long result must never turn
the model's tool message into a half-cut JSON blob.
"""

from __future__ import annotations

import json

from agent.tool_envelope import (
    build_tool_envelope,
    envelope_display,
    parse_tool_envelope,
    truncate_tool_content,
)


class TestBuildToolEnvelope:
    def test_builds_display_and_sources(self) -> None:
        env = build_tool_envelope("摘要文本", [{"id": "m1"}])
        assert json.loads(env) == {"display": "摘要文本", "sources": [{"id": "m1"}]}

    def test_sources_default_to_empty_list(self) -> None:
        env = build_tool_envelope("摘要文本")
        assert json.loads(env)["sources"] == []


class TestParseToolEnvelope:
    def test_parses_envelope(self) -> None:
        parsed = parse_tool_envelope('{"display": "d", "sources": [1]}')
        assert parsed == {"display": "d", "sources": [1]}

    def test_plain_text_returns_none(self) -> None:
        assert parse_tool_envelope("plain text") is None

    def test_non_json_returns_none(self) -> None:
        assert parse_tool_envelope("{not json") is None

    def test_dict_without_display_returns_none(self) -> None:
        # A write/ingest/entity tool result is not an envelope.
        assert parse_tool_envelope('{"action": "inserted", "id": "m1"}') is None

    def test_empty_string_returns_none(self) -> None:
        assert parse_tool_envelope("") is None


class TestEnvelopeDisplay:
    def test_unwraps_display(self) -> None:
        assert envelope_display('{"display": "clean", "sources": []}') == "clean"

    def test_plain_text_passthrough(self) -> None:
        assert envelope_display("raw text") == "raw text"

    def test_non_envelope_json_passthrough(self) -> None:
        assert envelope_display('{"action": "inserted"}') == '{"action": "inserted"}'


class TestTruncateToolContent:
    def test_short_content_untouched(self) -> None:
        assert truncate_tool_content("short") == "short"

    def test_long_plain_text_truncated(self) -> None:
        out = truncate_tool_content("x" * 2000)
        assert out.endswith("…[truncated]")
        assert len(out) == 800 + 1 + len("…[truncated]")

    def test_envelope_short_display_long_sources_kept_whole(self) -> None:
        """The sources array alone can exceed the cap — the model still gets
        the complete display text, not a half-cut envelope."""
        env = build_tool_envelope(
            "简短摘要", [{"summary": "x" * 300} for _ in range(50)]
        )
        assert len(env) > 800
        assert truncate_tool_content(env) == "简短摘要"

    def test_envelope_oversized_display_truncated_unwrapped(self) -> None:
        env = build_tool_envelope("摘" * 2000, [{"id": "s1"}])
        out = truncate_tool_content(env)
        assert out.endswith("…[truncated]")
        assert '"display"' not in out
        assert "sources" not in out
        assert len(out) == 800 + 1 + len("…[truncated]")

    def test_custom_limit(self) -> None:
        assert truncate_tool_content("x" * 100, limit=20) == "x" * 20 + "\n…[truncated]"
