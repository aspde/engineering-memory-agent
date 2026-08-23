"""Tests for the event-driven analysis runner (backend/service/event_analysis.py).

Pure-logic tests for the cooldown gate, output-contract validation,
Feishu card formatting, and runner configuration — plus runner tests with
the agent mocked.  No real LLM, no real Feishu.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import backend.service.event_analysis as ea
from backend.service.event_analysis import (
    EventContext,
    build_event_user_message,
    format_feishu_card,
    maybe_analyze_event,
    reset_cooldowns_for_tests,
    run_event_analysis,
    should_notify,
    validate_analysis,
)
from backend.shared.config import config


@pytest.fixture(autouse=True)
def _clean_cooldowns():
    reset_cooldowns_for_tests()
    yield
    reset_cooldowns_for_tests()


def _set_event_analysis(**overrides) -> None:
    """Apply attribute overrides to the shared ``config.event_analysis``."""
    for name, value in overrides.items():
        setattr(config.event_analysis, name, value)


def _restore_event_analysis(saved: dict) -> None:
    """Put previously captured ``config.event_analysis`` values back."""
    _set_event_analysis(**saved)


# ── Cooldown gate ─────────────────────────────────────────────────────


class TestCooldownGate:
    def test_first_event_passes_and_records(self):
        key = "ci_build:job_name:build-api"
        assert ea._cooldown_gate(key) == "pass"
        assert key in ea._cooldowns

    def test_second_event_within_window_is_skipped(self):
        key = "ci_build:job_name:build-api"
        assert ea._cooldown_gate(key) == "pass"
        assert ea._cooldown_gate(key) == "skipped_cooldown"

    def test_expired_window_replays(self):
        key = "ci_build:job_name:build-api"
        ea._cooldown_gate(key)
        # Simulate an old timestamp beyond any sane cooldown window.
        ea._cooldowns[key] -= 10_000.0
        assert ea._cooldown_gate(key) == "pass"

    def test_distinct_keys_are_independent(self):
        assert ea._cooldown_gate("ci_build:job_name:a") == "pass"
        assert ea._cooldown_gate("ci_build:job_name:b") == "pass"
        assert ea._cooldown_gate("ci_build:job_name:a") == "skipped_cooldown"
        assert ea._cooldown_gate("ci_build:job_name:b") == "skipped_cooldown"

    def test_none_key_passes_ungated(self):
        # A missing/None cooldown key must not silently drop events.
        assert ea._cooldown_gate(None) == "pass"

    def test_key_building_requires_string_value(self):
        meta = {"job_name": "build-api"}
        assert (
            ea._cooldown_key("ci_build", meta, "job_name")
            == "ci_build:job_name:build-api"
        )
        assert ea._cooldown_key("ci_build", {}, "job_name") is None
        assert ea._cooldown_key("ci_build", {"job_name": 42}, "job_name") is None
        assert ea._cooldown_key("ci_build", {"job_name": ""}, "job_name") is None
        assert ea._cooldown_key("ci_build", meta, None) is None


# ── Output contract ───────────────────────────────────────────────────


class TestValidateAnalysis:
    def _valid(self) -> dict:
        return {
            "is_known_issue": True,
            "similar_incidents": [
                {"memory_id": "a1b2c3d4", "summary": "pool OOM", "similarity": 0.9}
            ],
            "root_cause_hypothesis": "connection pool exhaustion",
            "recommendation": "raise pool size (memory: a1b2c3d4)",
            "severity": "warning",
        }

    def test_valid_analysis_passes(self):
        analysis, err = validate_analysis(self._valid())
        assert err is None
        assert analysis is not None

    def test_none_response_is_error_not_empty_verdict(self):
        analysis, err = validate_analysis(None)
        assert analysis is None
        assert err == "empty response"

    def test_missing_keys_are_reported(self):
        broken = self._valid()
        del broken["severity"]
        del broken["recommendation"]
        analysis, err = validate_analysis(broken)
        assert analysis is None
        assert "severity" in err and "recommendation" in err

    def test_invalid_severity_is_rejected(self):
        broken = self._valid()
        broken["severity"] = "catastrophic"
        analysis, err = validate_analysis(broken)
        assert analysis is None
        assert "invalid severity" in err

    def test_non_dict_input_rejected(self):
        analysis, err = validate_analysis(["not", "a", "dict"])
        assert analysis is None
        assert "expected JSON object" in err


# ── Notification gating & card rendering ──────────────────────────────


class TestNotification:
    # Saved in setup, restored unconditionally in teardown — no per-test
    # try/finally needed (teardown runs even on assertion failure).
    _saved: dict

    def setup_method(self):
        self._saved = {
            "notify_enabled": config.event_analysis.notify_enabled,
            "notify_severity": config.event_analysis.notify_severity,
        }

    def teardown_method(self):
        _restore_event_analysis(self._saved)

    def test_threshold_ordering(self):
        _set_event_analysis(notify_enabled=True, notify_severity="warning")
        assert should_notify({"severity": "critical"}) is True
        assert should_notify({"severity": "warning"}) is True
        assert should_notify({"severity": "info"}) is False
        _set_event_analysis(notify_enabled=True, notify_severity="info")
        assert should_notify({"severity": "info"}) is True

    def test_disabled_switch_never_notifies(self):
        _set_event_analysis(notify_enabled=False, notify_severity="info")
        assert should_notify({"severity": "critical"}) is False

    def test_card_contains_verdict_fields(self):
        title, markdown = format_feishu_card(
            "ci_build",
            "job_name",
            {"job_name": "build-api"},
            {
                "is_known_issue": True,
                "similar_incidents": [
                    {
                        "memory_id": "a1b2c3d4-0000",
                        "summary": "pool OOM",
                        "similarity": 0.91,
                    }
                ],
                "root_cause_hypothesis": "pool exhaustion",
                "recommendation": "raise pool size",
                "severity": "warning",
            },
        )
        assert "build-api" in title
        assert "a1b2c3d4" in markdown
        assert "pool exhaustion" in markdown
        assert "raise pool size" in markdown
        assert "warning" in markdown

    def test_card_without_title_entity_is_source_only(self):
        """A connector that declares no title_entity gets a clean title."""
        title, _ = format_feishu_card(
            "feishu",
            None,
            {},
            {
                "is_known_issue": False,
                "similar_incidents": [],
                "root_cause_hypothesis": "",
                "recommendation": "r",
                "severity": "info",
            },
        )
        assert title == "EMA 事件分析 · feishu"

    def test_card_handles_no_incidents(self):
        title, markdown = format_feishu_card(
            "ci_build",
            "job_name",
            {"job_name": "j"},
            {
                "is_known_issue": False,
                "similar_incidents": [],
                "root_cause_hypothesis": "",
                "recommendation": "investigate",
                "severity": "info",
            },
        )
        assert "首次出现" in markdown
        assert "investigate" in markdown


# ── Runner behaviour (agent mocked) ──────────────────────────────────


def _agent_result(final_text: str, **extra) -> dict:
    result = {"messages": [], "final_response": final_text}
    result.update(extra)
    return result


_VALID_JSON = (
    '{"is_known_issue": true, "similar_incidents": [{"memory_id": "a1b2", '
    '"summary": "s", "similarity": 0.9}], "root_cause_hypothesis": "h", '
    '"recommendation": "r (memory: a1b2)", "severity": "warning"}'
)


class TestRunEventAnalysis:
    @pytest.fixture(autouse=True)
    def _connector(self):
        """Runner tests receive the opted-in CI connector (as the gate does)."""
        self.connector = _OptedInConnector()

    async def _run(
        self,
        delivery_id: str,
        content: str = "CI Build: j — FAILED",
        metadata: dict | None = None,
    ):
        await run_event_analysis(
            EventContext(
                source="ci_build",
                delivery_id=delivery_id,
                content=content,
                metadata=metadata or {},
                connector=self.connector,
            )
        )

    async def test_persists_completed_verdict(self):
        agent = MagicMock()
        agent.ainvoke = AsyncMock(return_value=_agent_result(_VALID_JSON))
        persist = AsyncMock()
        with (
            patch("backend.service.event_analysis.get_agent", return_value=agent),
            patch.object(ea, "persist_analysis", persist),
            patch.object(ea, "send_feishu_message", AsyncMock(return_value=(True, "0"))),
        ):
            await self._run("d1")
        persist.assert_awaited_once()
        record = persist.await_args.args[1]
        assert record["status"] == "completed"
        assert record["is_known_issue"] is True

    async def test_agent_configuration_is_unattended_read_only(self):
        """The analysis agent runs unattended with a retrieval-only tool surface."""
        agent = MagicMock()
        agent.ainvoke = AsyncMock(return_value=_agent_result(_VALID_JSON))
        with (
            patch(
                "backend.service.event_analysis.get_agent", return_value=agent
            ) as mock_get,
            patch.object(ea, "persist_analysis", AsyncMock()),
        ):
            await self._run("d2")

        kwargs = mock_get.call_args.kwargs
        assert kwargs["approval_required_tools"] == frozenset()
        tool_names = {t.name for t in kwargs["llm_tools"]}
        assert tool_names == {
            "search_memories_tool",
            "query_entity_tool",
            "retrieve_chunks_tool",
        }
        assert kwargs["max_steps"] == ea._EVENT_MAX_STEPS

        input_messages = agent.ainvoke.await_args.kwargs["input"]["messages"]
        assert input_messages[0]["role"] == "system"
        from backend.service.prompts import get_prompt

        version, text = get_prompt("event.ci_failure")
        assert text in input_messages[0]["content"]

    async def test_malformed_output_recorded_as_failed(self):
        agent = MagicMock()
        agent.ainvoke = AsyncMock(return_value=_agent_result("not json at all"))
        persist = AsyncMock()
        with (
            patch("backend.service.event_analysis.get_agent", return_value=agent),
            patch.object(ea, "persist_analysis", persist),
        ):
            await self._run("d3")
        record = persist.await_args.args[1]
        assert record["status"] == "failed"
        assert record.get("raw_output")

    async def test_timeout_recorded_as_failed(self):
        async def _hang(*args, **kwargs):
            import asyncio

            await asyncio.sleep(3600)

        agent = MagicMock()
        agent.ainvoke = AsyncMock(side_effect=_hang)
        persist = AsyncMock()
        saved = {"timeout_seconds": config.event_analysis.timeout_seconds}
        with (
            patch("backend.service.event_analysis.get_agent", return_value=agent),
            patch.object(ea, "persist_analysis", persist),
        ):
            _set_event_analysis(timeout_seconds=1)
            try:
                await self._run("d4")
            finally:
                _restore_event_analysis(saved)
        record = persist.await_args.args[1]
        assert record["status"] == "failed"
        assert "timeout" in record["error"]

    async def test_interrupt_surfaced_not_fabricated(self):
        interrupt_obj = MagicMock()
        interrupt_obj.value = {"type": "approval"}
        agent = MagicMock()
        agent.ainvoke = AsyncMock(
            return_value={"__interrupt__": [interrupt_obj], "messages": []}
        )
        persist = AsyncMock()
        with (
            patch("backend.service.event_analysis.get_agent", return_value=agent),
            patch.object(ea, "persist_analysis", persist),
        ):
            await self._run("d5")
        record = persist.await_args.args[1]
        assert record["status"] == "interrupted"

    async def test_low_severity_completes_without_feishu(self):
        low_json = _VALID_JSON.replace('"warning"', '"info"')
        agent = MagicMock()
        agent.ainvoke = AsyncMock(return_value=_agent_result(low_json))
        send = AsyncMock(return_value=(True, "0"))
        with (
            patch("backend.service.event_analysis.get_agent", return_value=agent),
            patch.object(ea, "persist_analysis", AsyncMock()),
            patch.object(ea, "send_feishu_message", send),
        ):
            await self._run("d6")
        send.assert_not_awaited()

    async def test_exception_recorded_as_failed_not_raised(self):
        agent = MagicMock()
        agent.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        persist = AsyncMock()
        with (
            patch("backend.service.event_analysis.get_agent", return_value=agent),
            patch.object(ea, "persist_analysis", persist),
        ):
            await self._run("d7")
        record = persist.await_args.args[1]
        assert record["status"] == "failed"
        assert "boom" in record["error"]

    async def test_trace_id_follows_delivery(self):
        """The trace id matches the intake path so usage replay covers both."""
        from backend.shared.config import current_trace_id

        captured: dict = {}
        agent = MagicMock()

        async def _invoke(*args, **kwargs):
            captured["trace"] = current_trace_id.get("")
            return _agent_result(_VALID_JSON)

        agent.ainvoke = _invoke
        with (
            patch("backend.service.event_analysis.get_agent", return_value=agent),
            patch.object(ea, "persist_analysis", AsyncMock()),
        ):
            await self._run("delivery-x")
        assert captured["trace"] == "webhook:delivery-x"


# ── maybe_analyze_event gating ────────────────────────────────────────


class _OptedInConnector:
    triggers_event_analysis = True
    event_analysis_cooldown_key = "job_name"
    event_analysis_prompt_key = "event.ci_failure"
    event_analysis_display = {
        "title_entity": "job_name",
        "context_fields": ["branch", "source_url"],
    }


class TestMaybeAnalyzeEvent:
    async def _dispatch(
        self, source: str, delivery_id: str, connector, metadata: dict | None = None
    ):
        await maybe_analyze_event(
            EventContext(
                source=source,
                delivery_id=delivery_id,
                content="content",
                metadata=metadata or {},
                connector=connector,
            )
        )

    async def test_disabled_flag_returns_silently(self, caplog):
        runner = AsyncMock()
        with (
            patch.object(ea.config.event_analysis, "enabled", False),
            patch.object(ea, "run_event_analysis", runner),
        ):
            await self._dispatch("ci_build", "d8", _OptedInConnector())
        runner.assert_not_awaited()

    async def test_connector_without_capability_skipped(self):
        class _PlainConnector:
            triggers_event_analysis = False

        runner = AsyncMock()
        with (
            patch.object(ea.config.event_analysis, "enabled", True),
            patch.object(ea, "run_event_analysis", runner),
        ):
            await self._dispatch("feishu", "d9", _PlainConnector())
        runner.assert_not_awaited()

    async def test_opted_in_without_prompt_key_skips_with_error(self, caplog):
        """A misconfigured connector must not crash the delivery path."""
        import logging

        class _BrokenConnector:
            triggers_event_analysis = True
            event_analysis_cooldown_key = None
            event_analysis_prompt_key = None

        runner = AsyncMock()
        with (
            patch.object(ea.config.event_analysis, "enabled", True),
            patch.object(ea, "run_event_analysis", runner),
            caplog.at_level(logging.ERROR, logger="backend.service.event_analysis"),
        ):
            await self._dispatch("ci_build", "d9b", _BrokenConnector())
        runner.assert_not_awaited()
        assert any("event_analysis_prompt_key" in r.message for r in caplog.records)

    async def test_cooldown_hit_records_skip_and_does_not_run(self):
        connector = _OptedInConnector()
        metadata = {"job_name": "build-api"}
        runner = AsyncMock()
        persist = AsyncMock()
        with (
            patch.object(ea.config.event_analysis, "enabled", True),
            patch.object(ea, "run_event_analysis", runner),
            patch.object(ea, "persist_analysis", persist),
        ):
            await self._dispatch("ci_build", "d10", connector, metadata)
            runner.assert_awaited_once()  # first event analysed
            await self._dispatch("ci_build", "d11", connector, metadata)
            runner.assert_awaited_once()  # second skipped

        # First delivery got no skip row; second did.
        assert persist.await_count == 1
        assert persist.await_args.args[1]["status"] == "skipped_cooldown"

    async def test_slot_exhaustion_does_not_burn_cooldown(self):
        connector = _OptedInConnector()
        metadata = {"job_name": "build-api"}
        runner = AsyncMock()
        persist = AsyncMock()
        with (
            patch.object(ea.config.event_analysis, "enabled", True),
            patch.object(ea, "run_event_analysis", runner),
            patch.object(ea, "persist_analysis", persist),
            patch.object(ea, "_try_acquire_event_slot", return_value=False),
        ):
            await self._dispatch("ci_build", "d12", connector, metadata)
        runner.assert_not_awaited()
        assert persist.await_args.args[1]["status"] == "skipped_concurrency"
        # The cooldown slot was released so a later retry can analyse.
        key = "ci_build:job_name:build-api"
        assert key not in ea._cooldowns


# ── User-message assembly ─────────────────────────────────────────────


class TestBuildEventUserMessage:
    def test_includes_content_and_declared_context_fields(self):
        msg = build_event_user_message(
            "CI Build: build-api — FAILED\nError:\nboom",
            {"branch": "main", "source_url": "https://ci.example.com/1"},
            context_fields=["branch", "source_url"],
        )
        assert "CI Build: build-api — FAILED" in msg
        assert "branch: main" in msg
        assert "source_url: https://ci.example.com/1" in msg

    def test_omits_absent_metadata(self):
        msg = build_event_user_message(
            "content", {}, context_fields=["branch", "source_url"]
        )
        assert msg.strip().endswith("content")

    def test_no_context_fields_no_context_line(self):
        msg = build_event_user_message("content", {"branch": "main"}, context_fields=[])
        assert msg.strip().endswith("content")
        assert "main" not in msg

    def test_non_string_metadata_values_skipped(self):
        msg = build_event_user_message(
            "content", {"branch": 42}, context_fields=["branch"]
        )
        assert msg.strip().endswith("content")
