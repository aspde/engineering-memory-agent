"""Tests for execute_scenario — the shared scenario execution path.

Covers: run-row lifecycle (running → completed/failed), contract parsing
per outcome, slot release on every exit path, and timeout recording.
The DB is the real ``ema_test`` database (conftest resets it per session);
the agent is mocked so no LLM is called.
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import text

from backend.db import get_session_factory

MOCK_AGENT_PATH = "backend.runner.agent_service.get_agent"

_CONTRACT_JSON = json.dumps(
    {
        "overview": "API 网关 5xx 飙升",
        "timeline": [{"time": "2026-08-20", "event": "告警触发", "source": "CI"}],
        "similar_incidents": [],
        "root_cause": "连接池耗尽",
        "recommendations": [{"text": "加监控", "priority": "高"}],
        "related_entities": [{"name": "PostgreSQL", "incident_count": 2}],
    },
    ensure_ascii=False,
)


def _md_with_contract() -> str:
    return f"# 复盘报告\n\n正文……\n\n```json\n{_CONTRACT_JSON}\n```"


async def _fetch_run(run_id: str) -> dict:
    session_factory = get_session_factory()
    async with session_factory() as session:
        result = await session.execute(
            text(
                "SELECT id::text, scenario_key, trigger, status, result_md, "
                "findings, contract_ok, error "
                "FROM scenario_runs WHERE id = CAST(:id AS UUID)"
            ),
            {"id": run_id},
        )
        row = result.mappings().first()
    assert row is not None, "run row must exist"
    return dict(row)


@pytest.fixture(autouse=True)
def _reset_slots():
    from backend.runner.scenarios import _scenario_slots

    while _scenario_slots.active > 0:
        _scenario_slots.release()
    yield
    while _scenario_slots.active > 0:
        _scenario_slots.release()


class TestExecuteScenarioPersistence:
    """execute_scenario writes a terminal-state row for every outcome."""

    async def test_completed_run_persists_findings_and_contract(self):
        from backend.runner.scenarios import execute_scenario

        agent = AsyncMock()
        agent.ainvoke.return_value = {
            "final_response": _md_with_contract(),
            "messages": [],
        }
        with patch(MOCK_AGENT_PATH, return_value=agent):
            outcome = await execute_scenario("postmortem", params={}, trigger="manual")

        assert outcome["status"] == "completed"
        assert outcome["run_id"]
        assert outcome["contract_ok"] is True
        assert outcome["findings"]["root_cause"] == "连接池耗尽"

        row = await _fetch_run(outcome["run_id"])
        assert row["status"] == "completed"
        assert row["trigger"] == "manual"
        assert row["contract_ok"] is True
        assert row["findings"]["overview"] == "API 网关 5xx 飙升"
        assert "复盘报告" in row["result_md"]

    async def test_unparseable_result_marks_contract_miss(self):
        from backend.runner.scenarios import execute_scenario

        agent = AsyncMock()
        agent.ainvoke.return_value = {
            "final_response": "# 复盘\n\n没有 JSON 块的回答。",
            "messages": [],
        }
        with patch(MOCK_AGENT_PATH, return_value=agent):
            outcome = await execute_scenario("postmortem", params={}, trigger="event")

        # The user still gets the markdown; only the structured contract missed.
        assert outcome["status"] == "completed"
        assert outcome["contract_ok"] is False
        assert outcome["findings"] is None

        row = await _fetch_run(outcome["run_id"])
        assert row["contract_ok"] is False
        assert row["findings"] is None

    async def test_compose_failure_persists_failed_row(self):
        from backend.runner.scenarios import execute_scenario
        from backend.runner.scenarios import postmortem as pm_mod

        async def boom(**kwargs: object) -> str:
            raise RuntimeError("provider exploded")

        original = pm_mod.compose_postmortem
        try:
            pm_mod.compose_postmortem = boom  # type: ignore[assignment]
            # The executor deliberately does not raise: the failure lands on
            # the row (status=failed + error) and in the returned outcome.
            outcome = await execute_scenario("postmortem", params={}, trigger="manual")
        finally:
            pm_mod.compose_postmortem = original  # type: ignore[assignment]

        assert outcome["status"] == "failed"
        assert "RuntimeError" in (outcome["error"] or "")
        assert outcome["error_kind"] == "execution"
        row = await _fetch_run(outcome["run_id"])
        assert row["status"] == "failed"
        assert "provider exploded" in (row["error"] or "")

    async def test_unknown_scenario_raises_keyerror(self):
        from backend.runner.scenarios import execute_scenario

        with pytest.raises(KeyError):
            await execute_scenario("nonexistent")

    async def test_inactive_scenario_raises_keyerror(self):
        from backend.runner.scenarios import SCENARIOS, execute_scenario

        original = SCENARIOS["postmortem"]["status"]
        SCENARIOS["postmortem"]["status"] = "inactive"
        try:
            with pytest.raises(KeyError):
                await execute_scenario("postmortem")
        finally:
            SCENARIOS["postmortem"]["status"] = original


class TestExecuteScenarioSlotAccounting:
    """The concurrency slot must be released on success and failure."""

    async def test_slot_released_after_success(self):
        from backend.runner.scenarios import _scenario_slots, execute_scenario

        agent = AsyncMock()
        agent.ainvoke.return_value = {"final_response": "ok", "messages": []}
        with patch(MOCK_AGENT_PATH, return_value=agent):
            await execute_scenario("postmortem")
        assert _scenario_slots.active == 0

    async def test_slot_released_after_failure(self):
        from backend.runner.scenarios import _scenario_slots, execute_scenario
        from backend.runner.scenarios import postmortem as pm_mod

        async def boom(**kwargs: object) -> str:
            raise RuntimeError("boom")

        original = pm_mod.compose_postmortem
        try:
            pm_mod.compose_postmortem = boom  # type: ignore[assignment]
            outcome = await execute_scenario("postmortem")
            assert outcome["status"] == "failed"
        finally:
            pm_mod.compose_postmortem = original  # type: ignore[assignment]
        assert _scenario_slots.active == 0

    async def test_busy_error_when_cap_reached(self, monkeypatch):
        from backend.runner.scenarios import (
            ScenarioBusyError,
            _scenario_slots,
            execute_scenario,
        )
        from backend.shared.config import config

        monkeypatch.setattr(config, "max_scenario_concurrency", 0)
        with pytest.raises(ScenarioBusyError):
            await execute_scenario("postmortem")
        assert _scenario_slots.active == 0


class TestExecuteScenarioTimeout:
    """A compose that exceeds the deadline records a failed timed-out row."""

    async def test_timeout_recorded_on_row(self, monkeypatch):
        from backend.runner.scenarios import execute_scenario
        from backend.runner.scenarios import postmortem as pm_mod
        from backend.shared.config import config

        async def slow(**kwargs: object) -> str:
            await asyncio.sleep(30)
            return ""

        monkeypatch.setattr(pm_mod, "compose_postmortem", slow)
        monkeypatch.setattr(config, "scenario_timeout", 0.1)

        outcome = await execute_scenario("postmortem", params={}, trigger="manual")
        assert outcome["status"] == "failed"
        assert "timed out" in (outcome["error"] or "")
        # Typed error classification — the route maps this kind onto 504
        # instead of substring-matching the human-readable message.
        assert outcome["error_kind"] == "timeout"

        row = await _fetch_run(outcome["run_id"])
        assert row["status"] == "failed"
        assert "timed out" in (row["error"] or "")


class TestContractParsersAllScenarios:
    """Every registered scenario has a contract parser with its own key set."""

    def test_all_four_scenarios_have_parsers(self):
        from backend.runner.scenarios import _CONTRACT_PARSERS

        assert set(_CONTRACT_PARSERS.keys()) == {
            "postmortem", "code_review", "onboarding", "tech_debt",
        }

    def test_code_review_contract_round_trip(self):
        import json as _json

        from backend.runner.scenarios import _CONTRACT_PARSERS

        findings = {
            "pr_overview": "重构连接池配置",
            "risk_files": [{"path": "db.py", "risk": "高", "history": "2 次故障", "entities": ["PostgreSQL"]}],
            "decision_conflicts": [],
            "review_points": [{"text": "确认池水位告警", "priority": "高"}],
        }
        md = f"## 审查\n\n```json\n{_json.dumps(findings, ensure_ascii=False)}\n```"
        parsed, ok = _CONTRACT_PARSERS["code_review"](md)
        assert ok is True
        assert parsed == findings

    def test_onboarding_contract_missing_key_is_miss(self):
        import json as _json

        from backend.runner.scenarios import _CONTRACT_PARSERS

        findings = {"project_overview": "x", "core_modules": [], "reading_order": []}
        md = f"指南\n\n```json\n{_json.dumps(findings)}\n```"
        parsed, ok = _CONTRACT_PARSERS["onboarding"](md)
        assert ok is False
        assert parsed is None

    def test_tech_debt_contract_round_trip(self):
        import json as _json

        from backend.runner.scenarios import _CONTRACT_PARSERS

        findings = {
            "summary": {"unresolved_workarounds": 1, "doc_gaps": 0, "auto_resolved": 0},
            "unresolved_workarounds": [{"summary": "临时开关", "created": "2026-06-01", "entities": ["auth"], "months_old": 3}],
            "doc_gaps": [],
            "auto_resolved": [],
            "priorities": [{"text": "清理临时开关", "reason": "超过 3 个月"}],
        }
        md = f"报告\n\n```json\n{_json.dumps(findings, ensure_ascii=False)}\n```"
        parsed, ok = _CONTRACT_PARSERS["tech_debt"](md)
        assert ok is True
        assert parsed["summary"]["unresolved_workarounds"] == 1
