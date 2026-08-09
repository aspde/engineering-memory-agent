"""Tests for the shared scenario agent-invocation helper.

``invoke_scenario_agent`` is the single place every vertical scenario calls
the agent, so the approval-gate bypass and the HITL interrupt surfacing for
automated scenario runs live here.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestInvokeScenarioAgent:
    """The shared agent-invocation helper (backend.service.scenarios)."""

    @pytest.mark.asyncio
    async def test_bypasses_approval_gate_for_automated_runs(self) -> None:
        """Automated scenarios pass an empty approval set — no human is
        attached to a scheduled / manual-trigger run, so write tools must not
        pause on approval."""
        from backend.service.scenarios import invoke_scenario_agent

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {"final_response": "ok", "messages": []}

        with patch(
            "backend.service.agent_service.get_agent", return_value=mock_agent
        ) as mock_get_agent:
            result = await invoke_scenario_agent("system prompt", "user message")

        assert result == "ok"
        mock_get_agent.assert_called_once_with(approval_required_tools=frozenset())

    @pytest.mark.asyncio
    async def test_returns_interrupt_message_not_fabricated_result(self) -> None:
        """A HITL interrupt (conflict pause) surfaces as an explicit message —
        never as a silent fallback to the last AIMessage."""
        from backend.service.scenarios import invoke_scenario_agent

        interrupt = MagicMock()
        interrupt.value = {
            "type": "conflict",
            "existing_id": "m1",
            "new_summary": "new finding",
        }

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "__interrupt__": [interrupt],
            "messages": [
                type(
                    "FakeMsg",
                    (),
                    {"content": "fabricated fallback", "tool_calls": None},
                )(),
            ],
        }

        with patch(
            "backend.service.agent_service.get_agent", return_value=mock_agent
        ):
            result = await invoke_scenario_agent("system prompt", "user message")

        assert "被中断" in result
        assert "conflict" in result
        assert "fabricated fallback" not in result

    @pytest.mark.asyncio
    async def test_returns_error_message_on_exception(self) -> None:
        """Agent failures return an error message, not an exception."""
        from backend.service.scenarios import invoke_scenario_agent

        mock_agent = AsyncMock()
        mock_agent.ainvoke.side_effect = RuntimeError("boom")

        with patch(
            "backend.service.agent_service.get_agent", return_value=mock_agent
        ):
            result = await invoke_scenario_agent("system prompt", "user message")

        assert "失败" in result
        assert "boom" in result

    @pytest.mark.asyncio
    async def test_falls_back_to_last_message_when_no_final_response(self) -> None:
        """The last-message fallback is preserved for the normal path."""
        from backend.service.scenarios import invoke_scenario_agent

        mock_agent = AsyncMock()
        mock_agent.ainvoke.return_value = {
            "final_response": "",
            "messages": [
                type(
                    "FakeMsg",
                    (),
                    {"content": "tool-based answer", "tool_calls": None},
                )(),
            ],
        }

        with patch(
            "backend.service.agent_service.get_agent", return_value=mock_agent
        ):
            result = await invoke_scenario_agent("system prompt", "user message")

        assert "tool-based answer" in result
