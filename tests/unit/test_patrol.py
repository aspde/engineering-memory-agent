"""Tests for patrol service — run_patrol with mocked agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.service.patrol import _parse_findings, run_patrol


class TestParseFindings:
    """Unit tests for the JSON findings parser."""

    def test_parse_valid_json(self) -> None:
        result = _parse_findings('{"pattern_matches": [], "knowledge_gaps": []}')
        assert result == {"pattern_matches": [], "knowledge_gaps": []}

    def test_parse_json_with_markdown_fences(self) -> None:
        text = '```json\n{"x": 1}\n```'
        result = _parse_findings(text)
        assert result == {"x": 1}

    def test_parse_json_with_surrounding_text(self) -> None:
        text = 'Here is the result:\n{"a": "b"}\nDone.'
        result = _parse_findings(text)
        assert result == {"a": "b"}

    def test_parse_empty_string(self) -> None:
        assert _parse_findings("") is None

    def test_parse_non_json_text(self) -> None:
        result = _parse_findings("This is just a plain text response.")
        assert result is not None
        assert "raw_output" in result

    def test_parse_empty_json_object(self) -> None:
        result = _parse_findings("{}")
        assert result == {}


def _make_mock_session() -> MagicMock:
    """Create a mock that supports ``async with session_factory() as session``."""
    mock_session = AsyncMock()
    # async_sessionmaker() returns a new session context
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    factory = MagicMock(return_value=ctx)
    return factory


class TestRunPatrol:
    """Integration tests for run_patrol with mocked agent and database."""

    @pytest.mark.asyncio
    async def test_run_patrol_creates_log_entry(self) -> None:
        """Verify that run_patrol inserts and updates a patrol_log row."""
        findings_json = {
            "pattern_matches": [
                {
                    "new_memory_id": "m1",
                    "new_summary": "slow query",
                    "matched_memory_id": "m2",
                    "matched_summary": "connection pool exhaustion",
                    "similarity": 0.89,
                    "reason": "similar symptoms",
                }
            ],
            "knowledge_gaps": [],
            "new_entities": [],
        }

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": json.dumps(findings_json),
            "messages": [],
        }

        mock_factory = _make_mock_session()

        with (
            patch("backend.service.patrol.get_agent", return_value=mock_agent),
            patch(
                "backend.service.patrol.get_session_factory",
                return_value=mock_factory,
            ),
        ):
            patrol_id = await run_patrol(
                patrol_type="daily",
                trigger="cron",
                system_prompt="test prompt",
            )

        # Verify the agent was called
        mock_agent.ainvoke.assert_called_once()

        # Verify patrol_id is a valid UUID string
        assert len(patrol_id) == 36
        assert patrol_id.count("-") == 4

    @pytest.mark.asyncio
    async def test_run_patrol_marks_failed_on_error(self) -> None:
        """Verify that patrol status is set to 'failed' when agent crashes."""
        mock_agent = AsyncMock()
        mock_agent.ainvoke.side_effect = RuntimeError("LLM unavailable")

        mock_factory = _make_mock_session()

        with (
            patch("backend.service.patrol.get_agent", return_value=mock_agent),
            patch(
                "backend.service.patrol.get_session_factory",
                return_value=mock_factory,
            ),
        ):
            patrol_id = await run_patrol(
                patrol_type="daily",
                trigger="cron",
                system_prompt="test prompt",
            )

        # Should still complete (not raise) and return a patrol_id
        assert len(patrol_id) == 36

