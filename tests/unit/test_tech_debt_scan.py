"""Tests for the weekly tech-debt scan entry point.

``run_tech_debt_scan`` is the scheduler-facing wrapper around
``execute_scenario``: the weekly cron path must persist its run through the
shared executor (scenario_runs + contract parsing), same as the manual
route and the event trigger.  The agent is mocked — no LLM, no compose.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import backend.main  # noqa: F401  — imports must not break the wiring
from backend.runner.scenarios.tech_debt import run_tech_debt_scan


class TestRunTechDebtScan:
    async def test_routes_through_execute_scenario(self):
        """The scan goes through execute_scenario with the weekly trigger.

        Patch the tech_debt module's own binding (``from ... import`` at
        module top) — patching the source module would leave the real
        executor running here.
        """
        with patch(
            "backend.runner.scenarios.tech_debt.execute_scenario",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_exec.return_value = {
                "run_id": "r1",
                "status": "completed",
                "findings": {},
                "contract_ok": True,
                "error": None,
                "error_kind": None,
            }
            await run_tech_debt_scan()

        mock_exec.assert_awaited_once()
        args = mock_exec.await_args
        assert args.args[0] == "tech_debt"
        assert args.kwargs["trigger"] == "weekly_patrol"

    async def test_swallows_failure_never_raises(self):
        """A failed scan is terminal on its run row — the wrapper must not
        raise into the scheduler loop."""
        with patch(
            "backend.runner.scenarios.tech_debt.execute_scenario",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_exec.side_effect = RuntimeError("db gone")
            await run_tech_debt_scan()  # must not raise

    async def test_busy_cap_is_swallowed_too(self):
        """ScenarioBusyError (concurrency cap) is also swallowed — a busy
        slot means skip this week, not crash the scheduler callback."""
        from backend.runner.scenarios import ScenarioBusyError

        with patch(
            "backend.runner.scenarios.tech_debt.execute_scenario",
            new_callable=AsyncMock,
        ) as mock_exec:
            mock_exec.side_effect = ScenarioBusyError("tech_debt")
            await run_tech_debt_scan()  # must not raise


class TestMainWiring:
    """The lifespan closure delegates to the importable entry point."""

    def test_main_imports_run_tech_debt_scan(self):
        import inspect

        from backend.main import app  # noqa: F401

        source = inspect.getsource(__import__("backend.main", fromlist=["app"]))
        assert "run_tech_debt_scan" in source
        # The direct compose call must be gone from the scheduler wiring.
        scan_section = source.split("_run_tech_debt_scan")[1].split("await run_tech_debt_scan")[0]
        assert "compose_tech_debt_report" not in scan_section
