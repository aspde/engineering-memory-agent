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


MOCK_AGENT_PATH = "backend.runner.agent_service.get_agent"


@pytest.fixture(autouse=True)
def _reset_scenario_slots():
    """Drain any leftover scenario-run slots so tests are independent.

    Mirrors the conftest reset fixtures (throttle / compaction cache): a
    scenario test that fails mid-run could otherwise leave its slot held and
    skew a later test's concurrency accounting.
    """
    import backend.runner.scenarios as scenarios_mod

    while scenarios_mod._scenario_slots.active > 0:
        scenarios_mod._scenario_slots.release()
    yield
    while scenarios_mod._scenario_slots.active > 0:
        scenarios_mod._scenario_slots.release()


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
        # Every run is persisted — the response carries its row id.
        assert data["run_id"]


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
        from backend.runner.scenarios import SCENARIOS

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
        from backend.runner.scenarios import SCENARIOS

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
        from backend.runner.scenarios import postmortem
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


class TestScenarioErrorMapping:
    """Typed error kinds map onto HTTP statuses (no substring matching)."""

    async def test_invalid_params_returns_422(self, async_client):
        """A compose TypeError lands as error_kind=invalid_params → 422."""
        resp = await async_client.post(
            "/api/scenarios/postmortem/run",
            json={"params": {"definitely_not_a_param": True}},
        )
        assert resp.status_code == 422
        assert "definitely_not_a_param" in resp.json().get("detail", "")


class TestScenarioConversationSidebar:
    """The scenario thread gets a conversations row so it shows in history."""

    def test_run_upserts_conversation(self, client, mock_agent):
        from unittest.mock import patch as _patch

        with _patch(
            "backend.api.conversations.upsert_conversation"
        ) as mock_upsert, patch(MOCK_AGENT_PATH, return_value=mock_agent):
            resp = client.post(
                "/api/scenarios/postmortem/run",
                json={"params": {}, "thread_id": "scenario-test-t1"},
            )
        assert resp.status_code == 200
        mock_upsert.assert_awaited_once()
        assert mock_upsert.await_args.args[0] == "scenario-test-t1"
        # Title is the scenario display name so the sidebar reads 中文.
        assert "复盘" in str(mock_upsert.await_args.args[1])

    def test_run_without_thread_id_skips_upsert(self, client, mock_agent):
        from unittest.mock import patch as _patch

        with _patch(
            "backend.api.conversations.upsert_conversation"
        ) as mock_upsert, patch(MOCK_AGENT_PATH, return_value=mock_agent):
            resp = client.post("/api/scenarios/postmortem/run", json={"params": {}})
        assert resp.status_code == 200
        mock_upsert.assert_not_awaited()


class TestScenarioConcurrency:
    """Scenario run concurrency cap (MAX_SCENARIO_CONCURRENCY)."""

    async def test_concurrency_cap_rejects_with_503(self, async_client, monkeypatch):
        """A run beyond the cap is refused with 503, and the slot frees after."""
        from backend.runner.scenarios import postmortem
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


# ── Run persistence & save-as-memory ───────────────────────────────────


def _contract_markdown() -> str:
    import json

    findings = {
        "overview": "API 网关 5xx 飙升",
        "timeline": [{"time": "2026-08-20", "event": "告警", "source": "CI"}],
        "similar_incidents": [],
        "root_cause": "连接池耗尽",
        "recommendations": [{"text": "加监控", "priority": "高"}],
        "related_entities": [{"name": "PostgreSQL", "incident_count": 2}],
    }
    return (
        "# 复盘\n\n正文\n\n```json\n"
        + json.dumps(findings, ensure_ascii=False)
        + "\n```"
    )


class TestRunPersistence:
    """Runs land in scenario_runs; the detail endpoint exposes them."""

    def test_run_detail_endpoint(self, client, mock_agent):
        with patch(MOCK_AGENT_PATH, return_value=mock_agent):
            resp = client.post("/api/scenarios/postmortem/run", json={"params": {}})
        run_id = resp.json()["run_id"]

        detail = client.get(f"/api/scenarios/runs/{run_id}")
        assert detail.status_code == 200
        data = detail.json()
        assert data["id"] == run_id
        assert data["scenario_key"] == "postmortem"
        assert data["status"] == "completed"

    def test_run_detail_unknown_returns_404(self, client):
        import uuid

        resp = client.get(f"/api/scenarios/runs/{uuid.uuid4()}")
        assert resp.status_code == 404


class TestSaveAsMemory:
    """POST /api/scenarios/runs/{id}/save-as-memory"""

    @pytest.fixture
    def completed_postmortem_run(self, client):
        """Run a postmortem with a contract-conformant mocked agent."""
        agent = AsyncMock()
        agent.ainvoke.return_value = {
            "final_response": _contract_markdown(),
            "messages": [],
        }
        with patch(MOCK_AGENT_PATH, return_value=agent):
            resp = client.post("/api/scenarios/postmortem/run", json={"params": {}})
        assert resp.status_code == 200
        return resp.json()["run_id"], _contract_markdown()

    def test_save_inserts_memory(self, client, completed_postmortem_run):
        from unittest.mock import patch as _patch

        run_id, _md = completed_postmortem_run
        import uuid
        mem_id = str(uuid.uuid4())
        write_result = {"action": "inserted", "id": mem_id, "summary": "复盘摘要"}
        # The route imports lazily inside the handler — patch at the source
        # modules (write_memory / persist_pending_conflict), not the route.
        with _patch(
            "backend.service.memory.write_memory", new_callable=AsyncMock
        ) as mock_write, _patch(
            "backend.service.conflicts.persist_pending_conflict",
            new_callable=AsyncMock,
        ):
            mock_write.return_value = write_result
            resp = client.post(f"/api/scenarios/runs/{run_id}/save-as-memory")
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "inserted"
        assert data["memory_id"] == mem_id

    def test_save_defers_conflict_to_queue(self, client, completed_postmortem_run):
        from unittest.mock import patch as _patch

        run_id, _md = completed_postmortem_run
        conflict_result = {
            "action": "conflict",
            "summary": "新复盘",
            "existing_id": "existing-1",
            "existing_summary": "已有记忆",
            "_deferred": {},
        }
        with _patch(
            "backend.service.memory.write_memory", new_callable=AsyncMock
        ) as mock_write, _patch(
            "backend.service.conflicts.persist_pending_conflict",
            new_callable=AsyncMock,
        ) as mock_persist:
            mock_write.return_value = conflict_result
            mock_persist.return_value = {"id": "pc-1"}
            resp = client.post(f"/api/scenarios/runs/{run_id}/save-as-memory")
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "conflict"
        assert data["conflict_id"] == "pc-1"
        mock_persist.assert_awaited_once()

    def test_save_is_idempotent(self, client, completed_postmortem_run):
        from unittest.mock import patch as _patch

        from sqlalchemy import text as sql_text

        from backend.db import get_session_factory

        run_id, _md = completed_postmortem_run
        import uuid

        mem_id = str(uuid.uuid4())
        write_result = {"action": "inserted", "id": mem_id, "summary": "s"}

        # The mocked write must leave the row the duplicate probe queries —
        # otherwise idempotency is untestable through the real endpoint.
        async def fake_write(content, **kwargs):
            meta = kwargs.get("metadata") or {}
            factory = get_session_factory()
            async with factory() as session:
                await session.execute(
                    sql_text(
                        "INSERT INTO memories (id, source_type, summary, meta) "
                        "VALUES (CAST(:id AS UUID), 'postmortem', :summary, "
                        "CAST(:meta AS JSONB))"
                    ),
                    {
                        "id": mem_id,
                        "summary": str(meta.get("scenario_run_id", "")),
                        "meta": f'{{"scenario_run_id": "{meta.get("scenario_run_id", "")}"}}',
                    },
                )
                await session.commit()
            return write_result

        with _patch(
            "backend.service.memory.write_memory", side_effect=fake_write
        ) as mock_write:
            first = client.post(f"/api/scenarios/runs/{run_id}/save-as-memory")
            assert first.status_code == 200
            assert first.json()["action"] == "inserted"

            # Second click: the duplicate probe hits the saved memory — no
            # second write_memory call.
            second = client.post(f"/api/scenarios/runs/{run_id}/save-as-memory")
        assert second.status_code == 200
        assert second.json()["duplicate"] is True
        assert second.json()["memory_id"] == mem_id
        assert mock_write.await_count == 1

    def test_save_links_contract_entities(self, client, completed_postmortem_run):
        """Story 5: the saved memory links to the contract's related_entities."""
        import uuid
        from unittest.mock import patch as _patch

        run_id, _md = completed_postmortem_run
        write_result = {"action": "inserted", "id": str(uuid.uuid4()), "summary": "s"}
        with _patch(
            "backend.service.memory.write_memory", new_callable=AsyncMock
        ) as mock_write, _patch(
            "backend.service.conflicts.persist_pending_conflict",
            new_callable=AsyncMock,
        ), _patch(
            "backend.service.ingestion.entity.link_memory_to_entities",
            new_callable=AsyncMock,
        ) as mock_link:
            mock_write.return_value = write_result
            resp = client.post(f"/api/scenarios/runs/{run_id}/save-as-memory")
        assert resp.status_code == 200
        mock_link.assert_awaited_once()
        names = mock_link.await_args.args[1]
        # The contract markdown fixture declares PostgreSQL as related.
        assert "PostgreSQL" in names

    def test_save_conflict_skips_entity_linking(self, client, completed_postmortem_run):
        """A deferred conflict has no memory id — nothing to link yet."""
        from unittest.mock import patch as _patch

        run_id, _md = completed_postmortem_run
        conflict_result = {
            "action": "conflict",
            "summary": "新复盘",
            "existing_id": "existing-1",
            "existing_summary": "已有记忆",
            "_deferred": {},
        }
        with _patch(
            "backend.service.memory.write_memory", new_callable=AsyncMock
        ) as mock_write, _patch(
            "backend.service.conflicts.persist_pending_conflict",
            new_callable=AsyncMock,
        ) as mock_persist, _patch(
            "backend.service.ingestion.entity.link_memory_to_entities",
            new_callable=AsyncMock,
        ) as mock_link:
            mock_write.return_value = conflict_result
            mock_persist.return_value = {"id": "pc-1"}
            resp = client.post(f"/api/scenarios/runs/{run_id}/save-as-memory")
        assert resp.status_code == 200
        assert resp.json()["action"] == "conflict"
        mock_link.assert_not_awaited()

    def test_save_rejects_non_postmortem(self, client, mock_agent):
        with patch(MOCK_AGENT_PATH, return_value=mock_agent):
            resp = client.post("/api/scenarios/onboarding/run", json={"params": {}})
        run_id = resp.json()["run_id"]
        save = client.post(f"/api/scenarios/runs/{run_id}/save-as-memory")
        assert save.status_code == 400

    def test_save_unknown_run_returns_404(self, client):
        import uuid

        resp = client.post(f"/api/scenarios/runs/{uuid.uuid4()}/save-as-memory")
        assert resp.status_code == 404
