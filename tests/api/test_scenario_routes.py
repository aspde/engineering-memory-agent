"""Tests for scenario API routes — list and run endpoints.

All agent invocations are mocked so tests never call a real LLM.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    from backend.main import app

    return TestClient(app)


@pytest.fixture
def mock_agent():
    """Return an AsyncMock that stands in for the compiled agent graph."""
    agent = AsyncMock()
    agent.ainvoke.return_value = {
        "final_response": "场景执行结果",
        "messages": [],
    }
    return agent


MOCK_AGENT_PATH = "backend.service.agent_service.get_agent"


@pytest.fixture(autouse=True)
def _reset_scenario_slots():
    """Drain any leftover scenario-run slots so tests are independent.

    Mirrors the conftest reset fixtures (throttle / compaction cache): a
    scenario test that fails mid-run could otherwise leave its slot held and
    skew a later test's concurrency accounting.
    """
    import backend.service.scenarios as scenarios_mod

    while scenarios_mod._scenario_active > 0:
        scenarios_mod._release_scenario_slot()
    yield
    while scenarios_mod._scenario_active > 0:
        scenarios_mod._release_scenario_slot()


class TestListScenarios:
    """GET /api/scenarios"""

    def test_list_returns_scenarios(self, client):
        """List endpoint returns scenario entries with expected fields."""
        resp = client.get("/api/scenarios")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

        keys = {s["key"] for s in data}
        assert "postmortem" in keys
        assert "code_review" in keys
        assert "onboarding" in keys
        assert "tech_debt" in keys

        for s in data:
            assert "key" in s
            assert "name" in s
            assert "description" in s
            assert "triggers" in s
            assert s["status"] in ("active", "beta", "inactive")


class TestRunScenario:
    """POST /api/scenarios/{name}/run"""

    def test_run_unknown_scenario_returns_404(self, client, mock_agent):
        with patch(MOCK_AGENT_PATH, return_value=mock_agent):
            resp = client.post("/api/scenarios/nonexistent/run", json={"params": {}})
        assert resp.status_code == 404

    def test_run_without_params_uses_defaults(self, client, mock_agent):
        with patch(MOCK_AGENT_PATH, return_value=mock_agent):
            resp = client.post("/api/scenarios/postmortem/run", json={"params": {}})
        assert resp.status_code == 200
        data = resp.json()
        assert data["scenario"] == "postmortem"
        assert data["status"] == "completed"
        assert "result" in data

    def test_run_with_params_passes_through(self, client, mock_agent):
        with patch(MOCK_AGENT_PATH, return_value=mock_agent):
            resp = client.post(
                "/api/scenarios/postmortem/run",
                json={"params": {"incident_memory_id": "test-uuid"}},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["scenario"] == "postmortem"
        assert data["status"] == "completed"

    def test_run_code_review_with_params(self, client, mock_agent):
        with patch(MOCK_AGENT_PATH, return_value=mock_agent):
            resp = client.post(
                "/api/scenarios/code_review/run",
                json={"params": {"pr_diff": "diff", "pr_description": "test"}},
            )
        assert resp.status_code == 200

    def test_run_onboarding_with_params(self, client, mock_agent):
        with patch(MOCK_AGENT_PATH, return_value=mock_agent):
            resp = client.post(
                "/api/scenarios/onboarding/run",
                json={"params": {"scope": "full"}},
            )
        assert resp.status_code == 200

    def test_run_tech_debt_succeeds(self, client, mock_agent):
        with patch(MOCK_AGENT_PATH, return_value=mock_agent):
            resp = client.post("/api/scenarios/tech_debt/run", json={"params": {}})
        assert resp.status_code == 200


class TestScenarioVisibility:
    """Scenario status filtering."""

    def test_inactive_scenario_not_in_list(self, client):
        from backend.service.scenarios import SCENARIOS

        original = SCENARIOS["postmortem"]["status"]
        try:
            SCENARIOS["postmortem"]["status"] = "inactive"
            resp = client.get("/api/scenarios")
            assert resp.status_code == 200
            keys = {s["key"] for s in resp.json()}
            assert "postmortem" not in keys
        finally:
            SCENARIOS["postmortem"]["status"] = original

    def test_beta_scenario_hidden_by_default(self, client):
        """Beta scenarios are hidden unless explicitly requested."""
        from backend.service.scenarios import SCENARIOS

        original = SCENARIOS["postmortem"]["status"]
        try:
            SCENARIOS["postmortem"]["status"] = "beta"
            resp = client.get("/api/scenarios")
            assert resp.status_code == 200
            keys = {s["key"] for s in resp.json()}
            # beta is hidden by default (include_beta=False)
            assert "postmortem" not in keys
        finally:
            SCENARIOS["postmortem"]["status"] = original


class TestScenarioTimeout:
    """Scenario run deadline enforcement (SCENARIO_TIMEOUT_SECONDS)."""

    async def test_slow_compose_is_timed_out(self, async_client, monkeypatch):
        """A compose function that exceeds the deadline returns 504."""
        from backend.service.scenarios import postmortem
        from backend.shared.config import config

        async def slow_compose(**kwargs: object) -> str:
            await asyncio.sleep(30)
            return "never reached"

        monkeypatch.setattr(postmortem, "compose_postmortem", slow_compose)
        monkeypatch.setattr(config, "scenario_timeout", 0.2)

        resp = await async_client.post(
            "/api/scenarios/postmortem/run", json={"params": {}}
        )
        assert resp.status_code == 504
        # The deadline is surfaced, not swallowed as a generic 500.
        assert "超时" in resp.json().get("detail", "")


class TestScenarioConcurrency:
    """Scenario run concurrency cap (MAX_SCENARIO_CONCURRENCY)."""

    async def test_concurrency_cap_rejects_with_503(self, async_client, monkeypatch):
        """A run beyond the cap is refused with 503, and the slot frees after."""
        from backend.service.scenarios import postmortem
        from backend.shared.config import config

        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_compose(**kwargs: object) -> str:
            started.set()
            await release.wait()
            return "composed"

        monkeypatch.setattr(postmortem, "compose_postmortem", slow_compose)
        monkeypatch.setattr(config, "max_scenario_concurrency", 1)

        # First run acquires the single slot and blocks inside compose.
        first = asyncio.create_task(
            async_client.post("/api/scenarios/postmortem/run", json={"params": {}})
        )
        await asyncio.wait_for(started.wait(), timeout=5)

        # Second run is refused because the cap is reached.
        second = await async_client.post(
            "/api/scenarios/postmortem/run", json={"params": {}}
        )
        assert second.status_code == 503

        # Releasing the first run frees the slot for a fresh run.
        release.set()
        first_resp = await asyncio.wait_for(first, timeout=5)
        assert first_resp.status_code == 200
        assert first_resp.json()["result"] == "composed"

        third = await async_client.post(
            "/api/scenarios/postmortem/run", json={"params": {}}
        )
        assert third.status_code == 200
