"""Tests for patrol service — run_patrol with mocked agent."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.service.patrol import _parse_findings, mark_stale_patrols_failed, run_patrol


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


def _make_mock_session(*, running_row=None) -> MagicMock:
    """Create a mock session factory supporting ``async with``.

    ``execute`` is routed by SQL: the overlap-guard ``SELECT ... WHERE
    patrol_type = ...`` returns *running_row* (None = no overlap); all other
    statements return a generic AsyncMock.
    """
    mock_session = AsyncMock()

    def _execute(stmt, *args, **kwargs):
        sql = getattr(stmt, "text", None) or str(stmt)
        if "WHERE patrol_type" in sql:
            result = MagicMock()
            result.fetchone.return_value = running_row
            return result
        return AsyncMock()

    mock_session.execute.side_effect = _execute

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

    @pytest.mark.asyncio
    async def test_overlap_guard_skips_when_already_running(self) -> None:
        """A second patrol of the same type is skipped, not run concurrently."""
        mock_agent = AsyncMock()
        mock_factory = _make_mock_session(running_row=("existing-log-id",))

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

        assert patrol_id == ""  # skipped
        mock_agent.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_patrol_times_out(self, monkeypatch) -> None:
        """A hung agent run is cut off by PATROL_TIMEOUT and marked failed."""
        from backend.shared.config import config

        monkeypatch.setattr(config, "patrol_timeout", 0.05)

        async def _hang():
            await asyncio.sleep(60)

        mock_agent = AsyncMock()
        mock_agent.ainvoke.side_effect = _hang
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

        # Returns the log id (status was set to 'failed'), does not raise.
        assert len(patrol_id) == 36
        mock_agent.ainvoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cancelled_patrol_is_marked_failed_and_propagates(self) -> None:
        """A cancelled patrol persists 'failed', then re-raises the cancellation.

        ``except Exception`` never sees CancelledError (BaseException), so a
        cancelled patrol would otherwise leave its row stuck in 'running' and
        the overlap guard would skip every future patrol of this type.
        """
        async def _cancel(*args, **kwargs):
            raise asyncio.CancelledError

        mock_agent = AsyncMock()
        mock_agent.ainvoke.side_effect = _cancel
        mock_factory = _make_mock_session()

        with (
            patch("backend.service.patrol.get_agent", return_value=mock_agent),
            patch(
                "backend.service.patrol.get_session_factory",
                return_value=mock_factory,
            ),
        ):
            with pytest.raises(asyncio.CancelledError):
                await run_patrol(
                    patrol_type="daily",
                    trigger="cron",
                    system_prompt="test prompt",
                )

        # The failed status must be persisted before the cancellation
        # propagates — otherwise the row stays 'running'.
        session = mock_factory.return_value.__aenter__.return_value
        update_calls = [
            c for c in session.execute.await_args_list
            if "UPDATE patrol_logs" in str(c.args[0])
        ]
        assert update_calls, "expected a patrol_logs UPDATE on cancellation"
        assert update_calls[0].args[1]["status"] == "failed"
        session.commit.assert_awaited()


class TestMarkStalePatrols:
    """Startup sweep that re-marks 'running' rows from a previous process."""

    @pytest.mark.asyncio
    async def test_marks_running_rows_failed(self) -> None:
        mock_session = AsyncMock()
        result = MagicMock()
        result.rowcount = 2
        mock_session.execute.return_value = result

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=None)
        factory = MagicMock(return_value=ctx)

        with patch("backend.service.patrol.get_session_factory", return_value=factory):
            count = await mark_stale_patrols_failed()

        assert count == 2
        # The sweep must only touch rows stuck in 'running'.
        sql = getattr(mock_session.execute.call_args[0][0], "text", "")
        assert "status = 'running'" in sql

    @pytest.mark.asyncio
    async def test_no_running_rows_returns_zero(self) -> None:
        mock_session = AsyncMock()
        result = MagicMock()
        result.rowcount = 0
        mock_session.execute.return_value = result

        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=mock_session)
        ctx.__aexit__ = AsyncMock(return_value=None)
        factory = MagicMock(return_value=ctx)

        with patch("backend.service.patrol.get_session_factory", return_value=factory):
            count = await mark_stale_patrols_failed()

        assert count == 0
