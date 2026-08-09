"""Unit tests for tests/eval/run_task_eval.py — CLI orchestration.

The task runner is patched to return fabricated results (no LLM / no DB / no
graph).  Covers the glue: ``--sample`` slicing, exit codes for execution
errors (1) and regression-gate failures (2), ``--validate-only`` never
running a task, and the judge-provider self-judging guard.
"""

from __future__ import annotations

import pytest

from tests.eval.run_task_eval import _build_parser, _run
from tests.eval.task_runner import TaskEvalResult


def _fake_result(overall: dict, **kw) -> TaskEvalResult:
    keys = tuple(overall)
    return TaskEvalResult(
        suite="task",
        judge=kw.get("judge", "deterministic"),
        per_query=[],
        overall=overall,
        by_category={},
        metric_keys=keys,
        n_items=1,
        errors=[],
        judge_errors=[],
    )


@pytest.fixture
def patch_runner(monkeypatch):
    """Route run_tasks to a fake that records items + forwards results."""
    import tests.eval.run_task_eval as run_mod

    # The self-judging guard makes --judge llm (default) fail without a
    # complete LLM_JUDGE_* block — simulate a fully-configured judge provider.
    monkeypatch.setattr(run_mod.config.llm, "judge_provider", "zhipu")
    monkeypatch.setattr(run_mod.config.llm, "judge_model", "glm-4.7-flash")
    monkeypatch.setattr(run_mod.config.llm, "judge_api_key", "test-judge-key")
    monkeypatch.setattr(
        run_mod.config.llm, "judge_base_url", "https://open.bigmodel.cn/api/paas/v4"
    )

    state = {"calls": [], "result": _fake_result(
        {
            "completed": 1.0,
            "tool_recall": 1.0,
            "unexpected_rate": 0.0,
            "within_budget": 1.0,
            "n_steps": 3.0,
            "fact_coverage": 1.0,
            "groundedness": 1.0,
            "hallucination_rate": 0.0,
            "citation_rate": 1.0,
        }
    )}

    async def _fake(**kw):
        state["calls"].append(kw)
        return state["result"]

    monkeypatch.setattr("tests.eval.task_runner.run_tasks", _fake)
    return state


class TestRun:
    @pytest.mark.asyncio
    async def test_validate_only_runs_no_task(self, patch_runner) -> None:
        args = _build_parser().parse_args(["--validate-only"])
        assert await _run(args) == 0
        assert patch_runner["calls"] == []

    @pytest.mark.asyncio
    async def test_default_run_executes_tasks(self, patch_runner) -> None:
        args = _build_parser().parse_args([])
        assert await _run(args) == 0
        assert len(patch_runner["calls"]) == 1
        assert patch_runner["calls"][0]["judge"] == "llm"

    @pytest.mark.asyncio
    async def test_judge_flag_forwarded(self, patch_runner) -> None:
        args = _build_parser().parse_args(["--judge", "deterministic"])
        assert await _run(args) == 0
        assert patch_runner["calls"][0]["judge"] == "deterministic"

    @pytest.mark.asyncio
    async def test_max_steps_forwarded(self, patch_runner) -> None:
        args = _build_parser().parse_args(["--max-steps", "3"])
        assert await _run(args) == 0
        assert patch_runner["calls"][0]["max_steps"] == 3

    @pytest.mark.asyncio
    async def test_sample_slices_items(self, patch_runner) -> None:
        args = _build_parser().parse_args(["--sample", "2"])
        assert await _run(args) == 0
        # Full task set is 8; --sample 2 passes 2 items.
        assert len(patch_runner["calls"][0]["items"]) == 2

    @pytest.mark.asyncio
    async def test_threshold_gate_fails_with_exit_2(self, patch_runner, capsys) -> None:
        patch_runner["result"].overall = {
            "completed": 0.4,
            "tool_recall": 0.4,
            "within_budget": 0.9,
            "fact_coverage": 0.8,
            "groundedness": 0.8,
            "citation_rate": 0.8,
        }
        args = _build_parser().parse_args(["--min-completed", "0.75"])
        assert await _run(args) == 2
        assert "completed: 0.400 < required 0.750" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_threshold_for_absent_metric_warns_not_gates(
        self, patch_runner, capsys
    ) -> None:
        args = _build_parser().parse_args(["--min-tool-recall", "0.9"])
        # Default result reports tool_recall → the gate passes.
        assert await _run(args) == 0

    @pytest.mark.asyncio
    async def test_execution_errors_exit_1(self, patch_runner) -> None:
        patch_runner["result"].errors = [{"id": "task-001", "error": "provider down"}]
        args = _build_parser().parse_args([])
        assert await _run(args) == 1


class TestJudgeProviderGuard:
    """--judge llm without a dedicated judge provider must fail, not self-judge."""

    @pytest.mark.asyncio
    async def test_llm_judge_without_provider_fails(self, patch_runner, monkeypatch) -> None:
        import tests.eval.run_task_eval as run_mod

        monkeypatch.setattr(run_mod.config.llm, "judge_provider", "")
        args = _build_parser().parse_args([])  # default --judge llm
        with pytest.raises(SystemExit, match="judge provider"):
            await _run(args)

    @pytest.mark.asyncio
    async def test_deterministic_judge_needs_no_provider(
        self, patch_runner, monkeypatch
    ) -> None:
        import tests.eval.run_task_eval as run_mod

        monkeypatch.setattr(run_mod.config.llm, "judge_provider", "")
        args = _build_parser().parse_args(["--judge", "deterministic"])
        assert await _run(args) == 0

    @pytest.mark.asyncio
    async def test_configured_provider_allows_llm_judge(self, patch_runner) -> None:
        args = _build_parser().parse_args([])
        assert await _run(args) == 0

    @pytest.mark.asyncio
    async def test_validate_only_skips_guard(self, patch_runner, monkeypatch) -> None:
        import tests.eval.run_task_eval as run_mod

        monkeypatch.setattr(run_mod.config.llm, "judge_provider", "")
        args = _build_parser().parse_args(["--validate-only"])
        assert await _run(args) == 0


class TestJudgeChannelExitCode:
    """≥50% judge failures on the task suite reads as a broken run (exit 1)."""

    @pytest.mark.asyncio
    async def test_majority_judge_failures_exit_1(self, patch_runner, capsys) -> None:
        r = patch_runner["result"]
        r.judge = "llm"
        r.n_items = 2
        r.judge_errors = [{"id": "task-001", "error": "down"}, {"id": "task-002", "error": "down"}]
        args = _build_parser().parse_args([])
        assert await _run(args) == 1
        assert "judge channel failed" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_minority_judge_failures_do_not_fail_run(
        self, patch_runner, capsys
    ) -> None:
        r = patch_runner["result"]
        r.judge = "llm"
        r.n_items = 4
        r.judge_errors = [{"id": "task-001", "error": "down"}]  # 25% < 50%
        args = _build_parser().parse_args([])
        assert await _run(args) == 0
        assert "judge degraded on 1/4 rows" in capsys.readouterr().err
