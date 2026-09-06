"""Tests for the postmortem auto-trigger gate (event_analysis tail).

Pure-logic gate tests plus a spawn test with execute_scenario mocked —
no real LLM, no real DB write from the trigger itself.
"""

from __future__ import annotations

from typing import ClassVar
from unittest.mock import AsyncMock, patch

import pytest

import backend.runner.event_analysis as ea
from backend.runner.event_analysis import (
    EventContext,
    maybe_trigger_postmortem,
    reset_postmortem_triggers_for_tests,
)
from backend.shared.config import config


def _set_ea(**overrides) -> None:
    for name, value in overrides.items():
        setattr(config.event_analysis, name, value)


@pytest.fixture(autouse=True)
def _clean_trigger_state():
    reset_postmortem_triggers_for_tests()
    yield
    reset_postmortem_triggers_for_tests()
    # Restore defaults the override fixture may have flipped.
    _set_ea(
        postmortem_enabled=False,
        postmortem_min_severity="critical",
    )


class _FakeConnector:
    event_analysis_cooldown_key: str | None = "job_name"
    event_analysis_display: ClassVar[dict] = {"title_entity": "job_name"}


def _event(job: str = "build-api") -> EventContext:
    return EventContext(
        source="ci_build",
        delivery_id="d1",
        content="Build failed",
        metadata={"job_name": job},
        connector=_FakeConnector(),
        memory_id="11111111-1111-1111-1111-111111111111",
    )


def _analysis(severity: str = "critical", known: bool = True) -> dict:
    return {
        "is_known_issue": known,
        "severity": severity,
        "similar_incidents": [],
        "root_cause_hypothesis": "",
        "recommendation": "",
    }


class TestPostmortemGate:
    def test_gate_passes_first_and_blocks_second_within_window(self):
        assert ea._postmortem_gate("ci_build:job_name:build-api") is True
        assert ea._postmortem_gate("ci_build:job_name:build-api") is False

    def test_gate_none_key_always_passes(self):
        assert ea._postmortem_gate(None) is True
        assert ea._postmortem_gate(None) is True


class TestTriggerConditions:
    async def test_disabled_by_default_no_spawn(self):
        assert config.event_analysis.postmortem_enabled is False
        with patch(
            "backend.runner.scenarios.execute_scenario", new_callable=AsyncMock
        ) as mock_exec:
            maybe_trigger_postmortem(_analysis(), _event())
            await asyncio_drain()
            mock_exec.assert_not_awaited()

    async def test_scenarios_module_disabled_no_spawn(self, monkeypatch):
        """SCENARIOS_ENABLED=false gates the event path too (ADR-011) — the
        postmortem switch being on is not enough to burn scenario runs."""
        _set_ea(postmortem_enabled=True)
        # ``scenarios_active`` is a read-only property folding in the
        # APP_ENV=test exemption — flip its two inputs instead.
        monkeypatch.setattr(config, "scenarios_enabled", False)
        monkeypatch.setattr(config, "app_env", "development")
        with patch(
            "backend.runner.scenarios.execute_scenario", new_callable=AsyncMock
        ) as mock_exec:
            maybe_trigger_postmortem(_analysis(), _event())
            await asyncio_drain()
            mock_exec.assert_not_awaited()

    async def test_not_known_issue_no_spawn(self):
        _set_ea(postmortem_enabled=True)
        with patch(
            "backend.runner.scenarios.execute_scenario", new_callable=AsyncMock
        ) as mock_exec:
            maybe_trigger_postmortem(_analysis(known=False), _event())
            await asyncio_drain()
            mock_exec.assert_not_awaited()

    async def test_below_severity_threshold_no_spawn(self):
        _set_ea(postmortem_enabled=True, postmortem_min_severity="critical")
        with patch(
            "backend.runner.scenarios.execute_scenario", new_callable=AsyncMock
        ) as mock_exec:
            maybe_trigger_postmortem(_analysis(severity="warning"), _event())
            await asyncio_drain()
            mock_exec.assert_not_awaited()

    async def test_qualifying_event_spawns_once(self):
        _set_ea(postmortem_enabled=True, postmortem_min_severity="warning")
        with patch(
            "backend.runner.scenarios.execute_scenario", new_callable=AsyncMock
        ) as mock_exec:
            mock_exec.return_value = {
                "run_id": "r1",
                "status": "completed",
                "findings": {"overview": "x", "root_cause": "y"},
                "contract_ok": True,
                "error": None,
            }
            maybe_trigger_postmortem(_analysis(severity="warning"), _event())
            await asyncio_drain()
            mock_exec.assert_awaited_once()
            args = mock_exec.await_args
            assert args.args[0] == "postmortem"
            assert args.kwargs["trigger"] == "event"
            trigger_event = args.kwargs["params"]["trigger_event"]
            assert trigger_event["memory_id"] == "11111111-1111-1111-1111-111111111111"

            # Same cooldown key within the window → suppressed.
            maybe_trigger_postmortem(_analysis(severity="warning"), _event())
            await asyncio_drain()
            mock_exec.assert_awaited_once()


class TestNotifyPostmortem:
    """Feishu card copy distinguishes failure / success / contract miss."""

    @pytest.fixture(autouse=True)
    def _feishu_enabled(self, monkeypatch):
        monkeypatch.setattr(config, "feishu_webhook_url", "https://hook.example")
        # Title lookup reads the connector's display config via the event.

    def _notify_outcome(self, status: str, findings: dict | None, **extra) -> dict:
        return {
            "run_id": "r1",
            "status": status,
            "findings": findings,
            "contract_ok": True,
            "error": None,
            **extra,
        }

    async def _capture_card(self, outcome: dict) -> str:
        from unittest.mock import patch

        sent: dict = {}

        async def fake_send(message, *, msg_type="", title=""):
            sent["message"] = message
            sent["title"] = title
            return True, "ok"

        with patch(
            "backend.service.notification.send_feishu_message",
            side_effect=fake_send,
        ):
            await ea._notify_postmortem(_event(), outcome)
        return sent["message"]

    async def test_failed_run_reports_failure(self):
        msg = await self._capture_card(
            self._notify_outcome("failed", None, error="timed out after 600s")
        )
        assert "生成失败" in msg
        assert "timed out" in msg

    async def test_completed_with_findings_reports_overview(self):
        msg = await self._capture_card(
            self._notify_outcome(
                "completed", {"overview": "网关 5xx", "root_cause": "连接池"}
            )
        )
        assert "网关 5xx" in msg
        assert "生成失败" not in msg

    async def test_contract_miss_is_not_reported_as_failure(self):
        """完成但契约未过（findings 为空、Markdown 正常）→ 引导看草稿，不报失败。"""
        msg = await self._capture_card(
            self._notify_outcome("completed", None, contract_ok=False)
        )
        assert "生成失败" not in msg
        assert "Markdown" in msg
        assert "r1" in msg


async def asyncio_drain() -> None:
    """Let fire-and-forget tasks created by the trigger run to completion."""

    pending = set(ea._postmortem_tasks)
    for task in pending:
        await task
