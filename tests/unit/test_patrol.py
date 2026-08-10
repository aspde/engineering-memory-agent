"""Tests for patrol service — run_patrol with mocked agent."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.service.patrol import (
    _interrupt_payload,
    _is_conflict_interrupt,
    _parse_findings,
    _PATROL_REPAIR_MIN_CHARS,
    _repair_findings,
    _validate_findings,
    mark_stale_patrols_failed,
    run_patrol,
)


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


class TestValidateFindings:
    """Unit tests for the per-patrol-type structural validation."""

    def test_daily_requires_all_three_keys(self) -> None:
        err = _validate_findings(
            {"pattern_matches": [], "knowledge_gaps": []}, "daily"
        )
        assert err is not None and "new_entities" in err
        assert (
            _validate_findings(
                {"pattern_matches": [], "knowledge_gaps": [], "new_entities": []},
                "daily",
            )
            is None
        )

    def test_weekly_requires_contradiction_keys(self) -> None:
        err = _validate_findings({}, "weekly")
        assert err is not None and "contradictions" in err
        assert "decay_alerts" in err and "entity_coverage" in err

    def test_contradiction_scan_uses_weekly_contract(self) -> None:
        assert (
            _validate_findings(
                {"contradictions": [], "decay_alerts": [], "entity_coverage": []},
                "contradiction_scan",
            )
            is None
        )

    def test_event_driven_requires_matches(self) -> None:
        assert _validate_findings({}, "event_driven") is not None
        assert _validate_findings({"matches": []}, "event_driven") is None

    def test_non_dict_findings_rejected(self) -> None:
        err = _validate_findings([], "weekly")
        assert err is not None and "JSON object" in err
        assert _validate_findings(None, "weekly") is not None

    def test_unknown_type_has_no_contract(self) -> None:
        assert _validate_findings({"anything": 1}, "future_type") is None


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


class TestConflictInterruptHelpers:
    """Conflict-pause detection used by run_patrol's auto-resolution."""

    def test_conflict_payload_detected(self) -> None:
        interrupt = MagicMock()
        interrupt.value = {"type": "conflict", "new_summary": "x"}
        assert _is_conflict_interrupt({"__interrupt__": [interrupt]})

    def test_non_conflict_payload_not_detected(self) -> None:
        interrupt = MagicMock()
        interrupt.value = {"tool_name": "write_memory_tool"}
        assert not _is_conflict_interrupt({"__interrupt__": [interrupt]})

    def test_no_interrupt_not_detected(self) -> None:
        assert not _is_conflict_interrupt({"messages": []})

    def test_interrupt_payload_unwraps_value(self) -> None:
        interrupt = MagicMock()
        interrupt.value = {"type": "conflict"}
        assert _interrupt_payload(interrupt) == {"type": "conflict"}


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
    async def test_run_patrol_bypasses_approval_gate(self) -> None:
        """Automated patrols pass an empty approval set to get_agent().

        A patrol is unattended — no human can approve a paused write/ingest
        call, so gating on approval would stall the scan indefinitely.
        """
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": json.dumps(
                {"pattern_matches": [], "knowledge_gaps": [], "new_entities": []}
            ),
            "messages": [],
        }
        mock_factory = _make_mock_session()

        with (
            patch(
                "backend.service.patrol.get_agent", return_value=mock_agent
            ) as mock_get_agent,
            patch(
                "backend.service.patrol.get_session_factory",
                return_value=mock_factory,
            ),
        ):
            await run_patrol(
                patrol_type="daily",
                trigger="cron",
                system_prompt="test prompt",
            )

        mock_get_agent.assert_called_once_with(
            approval_required_tools=frozenset(), max_steps=15
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("patrol_type", "expected"),
        [
            ("daily", 15),
            ("weekly", 20),
            ("contradiction_scan", 20),
        ],
    )
    async def test_run_patrol_passes_per_type_max_steps(
        self, patrol_type: str, expected: int
    ) -> None:
        """Each patrol type gets its own ReAct budget, not the interactive
        MAX_AGENT_STEPS — a full-memory scan needs more search steps than chat."""
        contract = (
            {"pattern_matches": [], "knowledge_gaps": [], "new_entities": []}
            if patrol_type == "daily"
            else {"contradictions": [], "decay_alerts": [], "entity_coverage": []}
        )
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": json.dumps(contract),
            "messages": [],
        }
        mock_factory = _make_mock_session()

        with (
            patch(
                "backend.service.patrol.get_agent", return_value=mock_agent
            ) as mock_get_agent,
            patch(
                "backend.service.patrol.get_session_factory",
                return_value=mock_factory,
            ),
        ):
            await run_patrol(
                patrol_type=patrol_type,
                trigger="cron",
                system_prompt="test prompt",
            )

        assert mock_get_agent.call_args.kwargs["max_steps"] == expected

    @pytest.mark.asyncio
    async def test_run_patrol_unknown_type_keeps_default_steps(self) -> None:
        """A patrol type without an explicit budget leaves max_steps unset —
        get_agent then falls back to the interactive MAX_AGENT_STEPS."""
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": json.dumps({"matches": []}),
            "messages": [],
        }
        mock_factory = _make_mock_session()

        with (
            patch(
                "backend.service.patrol.get_agent", return_value=mock_agent
            ) as mock_get_agent,
            patch(
                "backend.service.patrol.get_session_factory",
                return_value=mock_factory,
            ),
        ):
            await run_patrol(
                patrol_type="event_driven",
                trigger="webhook",
                system_prompt="test prompt",
            )

        assert mock_get_agent.call_args.kwargs.get("max_steps") is None

    @pytest.mark.asyncio
    async def test_run_patrol_marks_interrupted_on_interrupt(self) -> None:
        """A HITL interrupt (approval / conflict pause) persists 'interrupted',
        never a silent 'completed' with fabricated findings."""
        interrupt = MagicMock()
        interrupt.value = {
            "tool_name": "write_memory_tool",
            "tool_args": {"content": "x"},
        }

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "__interrupt__": [interrupt],
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

        assert len(patrol_id) == 36
        session = mock_factory.return_value.__aenter__.return_value
        update_calls = [
            c for c in session.execute.await_args_list
            if "UPDATE patrol_logs" in str(c.args[0])
        ]
        assert update_calls, "expected a patrol_logs UPDATE"
        assert update_calls[0].args[1]["status"] == "interrupted"
        findings = json.loads(update_calls[0].args[1]["findings"])
        assert findings["interrupt"]["tool_name"] == "write_memory_tool"

    @pytest.mark.asyncio
    async def test_run_patrol_auto_resolves_conflict_interrupt(self) -> None:
        """A write-conflict pause is auto-resolved keep_both so an unattended
        patrol completes instead of sitting 'interrupted' forever."""
        conflict = MagicMock()
        conflict.value = {
            "type": "conflict",
            "new_summary": "new",
            "existing_id": "existing-1",
            "existing_summary": "old",
            "options": ["keep_existing", "overwrite", "merge", "keep_both"],
        }
        completed = {
            "final_response": json.dumps(
                {"pattern_matches": [], "knowledge_gaps": [], "new_entities": []}
            ),
            "messages": [],
        }

        mock_agent = AsyncMock()
        # First ainvoke pauses on conflict; the keep_both resume completes.
        mock_agent.ainvoke.side_effect = [
            {"__interrupt__": [conflict], "messages": []},
            completed,
        ]
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

        assert len(patrol_id) == 36
        # Initial run + one keep_both resume.
        assert mock_agent.ainvoke.await_count == 2
        resume_command = mock_agent.ainvoke.await_args_list[1].args[0]
        assert resume_command.resume == {"resolution": "keep_both"}

        session = mock_factory.return_value.__aenter__.return_value
        update_calls = [
            c for c in session.execute.await_args_list
            if "UPDATE patrol_logs" in str(c.args[0])
        ]
        assert update_calls
        assert update_calls[0].args[1]["status"] == "completed"

    @pytest.mark.asyncio
    async def test_run_patrol_conflict_loop_bounded(self) -> None:
        """A patrol that keeps re-conflicting stops after the auto-resolve
        bound and persists 'interrupted' rather than spinning forever."""
        conflict = MagicMock()
        conflict.value = {
            "type": "conflict",
            "new_summary": "new",
            "existing_id": "existing-1",
            "existing_summary": "old",
            "options": ["keep_existing", "overwrite", "merge", "keep_both"],
        }

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"__interrupt__": [conflict], "messages": []}
        mock_factory = _make_mock_session()

        from backend.service.patrol import _MAX_AUTO_CONFLICT_RESOLUTIONS

        with (
            patch("backend.service.patrol.get_agent", return_value=mock_agent),
            patch(
                "backend.service.patrol.get_session_factory",
                return_value=mock_factory,
            ),
        ):
            await run_patrol(
                patrol_type="daily",
                trigger="cron",
                system_prompt="test prompt",
            )

        # Bound + 1: the initial call plus one resume per bound, then it stops.
        assert (
            mock_agent.ainvoke.await_count
            == _MAX_AUTO_CONFLICT_RESOLUTIONS + 1
        )
        session = mock_factory.return_value.__aenter__.return_value
        update_calls = [
            c for c in session.execute.await_args_list
            if "UPDATE patrol_logs" in str(c.args[0])
        ]
        assert update_calls[0].args[1]["status"] == "interrupted"

    @pytest.mark.asyncio
    async def test_run_patrol_marks_failed_on_unserialisable_findings(self) -> None:
        """A findings payload that JSON can't serialise persists 'failed' —
        it must never leave the patrol stuck in 'running'."""
        interrupt = MagicMock()
        interrupt.value = {"tool_name": "write_memory_tool", "blob": object()}

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "__interrupt__": [interrupt],
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

        assert len(patrol_id) == 36
        session = mock_factory.return_value.__aenter__.return_value
        update_calls = [
            c for c in session.execute.await_args_list
            if "UPDATE patrol_logs" in str(c.args[0])
        ]
        assert update_calls, "expected a patrol_logs UPDATE"
        assert update_calls[0].args[1]["status"] == "failed"
        assert update_calls[0].args[1]["findings"] is None

    @pytest.mark.asyncio
    async def test_run_patrol_marks_failed_on_malformed_findings(self) -> None:
        """Findings that fail the structural contract persist as 'failed' with
        a parse_error — a malformed scan must never be recorded 'completed'."""
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "This is not the JSON contract at all.",
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
                patrol_type="weekly",
                trigger="cron",
                system_prompt="test prompt",
            )

        assert len(patrol_id) == 36
        session = mock_factory.return_value.__aenter__.return_value
        update_calls = [
            c for c in session.execute.await_args_list
            if "UPDATE patrol_logs" in str(c.args[0])
        ]
        assert update_calls[0].args[1]["status"] == "failed"
        findings = json.loads(update_calls[0].args[1]["findings"])
        assert "parse_error" in findings
        assert "raw_output" in findings
        assert "contradictions" in findings["parse_error"]

    @pytest.mark.asyncio
    async def test_run_patrol_valid_findings_stay_completed(self) -> None:
        """A contract-shaped findings JSON still persists as 'completed'."""
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": json.dumps(
                {"pattern_matches": [], "knowledge_gaps": [], "new_entities": []}
            ),
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
            await run_patrol(
                patrol_type="daily",
                trigger="cron",
                system_prompt="test prompt",
            )

        session = mock_factory.return_value.__aenter__.return_value
        update_calls = [
            c for c in session.execute.await_args_list
            if "UPDATE patrol_logs" in str(c.args[0])
        ]
        assert update_calls[0].args[1]["status"] == "completed"
        assert "parse_error" not in json.loads(update_calls[0].args[1]["findings"])

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


class TestRepairFindings:
    """One bounded repair attempt after contract-validation failure."""

    # A weekly-shaped report in prose — long enough to repair (over the
    # fragment threshold) and structurally equivalent to the JSON contract.
    _PROSE_REPORT = (
        "# Weekly report\n\n"
        "## Contradictions\n"
        "- memory-a vs memory-b: opposite recommendations\n"
        "## Decay\n"
        "- memory-1: decay 0.005, recommend archive\n"
        "## Entity coverage\n"
        "- PostgreSQL: missing backup/recovery domain\n"
    ) * 4  # ~900 chars, far above the fragment threshold

    @pytest.mark.asyncio
    async def test_run_patrol_repairs_malformed_prose(self) -> None:
        """A prose report that fails validation is repaired to JSON and the
        patrol persists 'completed' instead of 'failed'."""
        valid = {"contradictions": [], "decay_alerts": [], "entity_coverage": []}
        mock_provider = AsyncMock()
        mock_provider.chat = AsyncMock(return_value=json.dumps(valid))
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": self._PROSE_REPORT,
            "messages": [],
        }
        mock_factory = _make_mock_session()

        with (
            patch("backend.service.patrol.get_agent", return_value=mock_agent),
            patch(
                "backend.service.patrol.get_session_factory",
                return_value=mock_factory,
            ),
            patch(
                "backend.service.patrol.get_llm_provider",
                return_value=mock_provider,
            ),
        ):
            await run_patrol(
                patrol_type="weekly",
                trigger="cron",
                system_prompt="test prompt",
            )

        session = mock_factory.return_value.__aenter__.return_value
        update_calls = [
            c for c in session.execute.await_args_list
            if "UPDATE patrol_logs" in str(c.args[0])
        ]
        assert update_calls[0].args[1]["status"] == "completed"
        findings = json.loads(update_calls[0].args[1]["findings"])
        assert "parse_error" not in findings

        # The repair call is made exactly once, with the raw report as input.
        mock_provider.chat.assert_awaited_once()
        repair_content = mock_provider.chat.await_args.args[0][0]["content"]
        assert self._PROSE_REPORT[:100] in repair_content
        assert "contradictions" in repair_content

    @pytest.mark.asyncio
    async def test_run_patrol_skips_repair_for_fragment_output(self) -> None:
        """A short mid-thought fragment is not a report — no repair call is
        made and the patrol is recorded failed (a re-emitted empty report
        would be a misleading 'completed')."""
        mock_provider = AsyncMock()
        mock_provider.chat = AsyncMock()
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "Let me scan for more memories to get a fuller picture.",
            "messages": [],
        }
        mock_factory = _make_mock_session()

        with (
            patch("backend.service.patrol.get_agent", return_value=mock_agent),
            patch(
                "backend.service.patrol.get_session_factory",
                return_value=mock_factory,
            ),
            patch(
                "backend.service.patrol.get_llm_provider",
                return_value=mock_provider,
            ),
        ):
            await run_patrol(
                patrol_type="daily",
                trigger="cron",
                system_prompt="test prompt",
            )

        mock_provider.chat.assert_not_called()
        session = mock_factory.return_value.__aenter__.return_value
        update_calls = [
            c for c in session.execute.await_args_list
            if "UPDATE patrol_logs" in str(c.args[0])
        ]
        assert update_calls[0].args[1]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_run_patrol_keeps_failed_when_repair_invalid(self) -> None:
        """A repair that still fails validation keeps the patrol 'failed' —
        the retry never papers over a genuinely unusable report."""
        mock_provider = AsyncMock()
        mock_provider.chat = AsyncMock(
            return_value="Still not the JSON contract at all."
        )
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": self._PROSE_REPORT,
            "messages": [],
        }
        mock_factory = _make_mock_session()

        with (
            patch("backend.service.patrol.get_agent", return_value=mock_agent),
            patch(
                "backend.service.patrol.get_session_factory",
                return_value=mock_factory,
            ),
            patch(
                "backend.service.patrol.get_llm_provider",
                return_value=mock_provider,
            ),
        ):
            await run_patrol(
                patrol_type="weekly",
                trigger="cron",
                system_prompt="test prompt",
            )

        session = mock_factory.return_value.__aenter__.return_value
        update_calls = [
            c for c in session.execute.await_args_list
            if "UPDATE patrol_logs" in str(c.args[0])
        ]
        assert update_calls[0].args[1]["status"] == "failed"
        findings = json.loads(update_calls[0].args[1]["findings"])
        assert "parse_error" in findings

    @pytest.mark.asyncio
    async def test_repair_provider_error_keeps_patrol_failed(self) -> None:
        """A repair call that raises must not crash the patrol — it records
        'failed' exactly as if no repair had been attempted."""
        mock_provider = AsyncMock()
        mock_provider.chat = AsyncMock(side_effect=RuntimeError("provider down"))
        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": self._PROSE_REPORT,
            "messages": [],
        }
        mock_factory = _make_mock_session()

        with (
            patch("backend.service.patrol.get_agent", return_value=mock_agent),
            patch(
                "backend.service.patrol.get_session_factory",
                return_value=mock_factory,
            ),
            patch(
                "backend.service.patrol.get_llm_provider",
                return_value=mock_provider,
            ),
        ):
            patrol_id = await run_patrol(
                patrol_type="weekly",
                trigger="cron",
                system_prompt="test prompt",
            )

        assert len(patrol_id) == 36
        session = mock_factory.return_value.__aenter__.return_value
        update_calls = [
            c for c in session.execute.await_args_list
            if "UPDATE patrol_logs" in str(c.args[0])
        ]
        assert update_calls[0].args[1]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_repair_skips_unknown_type(self) -> None:
        """A patrol type without a contract has nothing to repair against."""
        result = await _repair_findings("some long output " * 50, "err", "future_type")
        assert result is None

    @pytest.mark.asyncio
    async def test_repair_min_chars_threshold(self) -> None:
        """Outputs below the fragment threshold are never sent to the LLM."""
        assert (
            await _repair_findings("short", "err", "weekly") is None
        )
        assert len(self._PROSE_REPORT) >= _PATROL_REPAIR_MIN_CHARS


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
