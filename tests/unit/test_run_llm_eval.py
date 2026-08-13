"""Unit tests for tests/eval/run_llm_eval.py — CLI orchestration.

The runner functions are patched to return fabricated results (no LLM / no
DB).  Covers the glue `_run_one` cannot be trusted to be right by itself:
suite selection, `--sample` slicing, exit codes for execution errors (1) and
regression-gate failures (2), and that `--validate-only` never invokes a
suite.
"""

from __future__ import annotations

import pytest

from tests.eval.llm_runner import LlmEvalResult
from tests.eval.run_llm_eval import _build_parser, _run


def _fake_result(suite: str, overall: dict) -> LlmEvalResult:
    keys = tuple(overall)
    return LlmEvalResult(
        suite=suite,
        judge="deterministic",
        per_query=[],
        overall=overall,
        by_category={},
        metric_keys=keys,
        n_items=1,
        errors=[],
        judge_errors=[],
    )


@pytest.fixture
def patch_runners(monkeypatch):
    """Route all four suite runners to fakes that record their items arg."""
    import tests.eval.run_llm_eval as run_mod

    # The self-judging guard makes --judge llm (the default) fail when the
    # LLM_JUDGE_* block is incomplete.  Simulate a fully-configured judge
    # provider so the suites run; the guard itself is tested separately.
    monkeypatch.setattr(run_mod.config.llm, "judge_provider", "zhipu")
    monkeypatch.setattr(run_mod.config.llm, "judge_model", "glm-4.7-flash")
    monkeypatch.setattr(run_mod.config.llm, "judge_api_key", "test-judge-key")
    monkeypatch.setattr(
        run_mod.config.llm, "judge_base_url", "https://open.bigmodel.cn/api/paas/v4"
    )

    calls: dict[str, list] = {
        "tool_selection": [],
        "extraction": [],
        "answer": [],
        "e2e": [],
    }
    results: dict[str, LlmEvalResult] = {}

    def _make(suite: str):
        async def _fake(**kw):
            calls[suite].append(kw.get("items"))
            return results[suite]

        return _fake

    monkeypatch.setattr(
        "tests.eval.llm_runner.run_tool_selection", _make("tool_selection")
    )
    monkeypatch.setattr(
        "tests.eval.llm_runner.run_extraction", _make("extraction")
    )
    monkeypatch.setattr(
        "tests.eval.llm_runner.run_answer", _make("answer")
    )
    monkeypatch.setattr(
        "tests.eval.llm_runner.run_e2e", _make("e2e")
    )
    results["tool_selection"] = _fake_result(
        "tool_selection", {"tool_accuracy": 1.0}
    )
    results["extraction"] = _fake_result(
        "extraction", {"entity_f1": 1.0, "relation_f1": 1.0}
    )
    results["answer"] = _fake_result(
        "answer", {"fact_coverage": 1.0, "groundedness": 1.0}
    )
    results["e2e"] = _fake_result(
        "e2e",
        {
            "context_recall": 1.0,
            "fact_coverage": 1.0,
            "groundedness": 1.0,
            "citation_rate": 1.0,
        },
    )
    return calls, results


class TestRun:
    @pytest.mark.asyncio
    async def test_validate_only_runs_no_suite(self, patch_runners) -> None:
        calls, _ = patch_runners
        args = _build_parser().parse_args(["--validate-only"])
        assert await _run(args) == 0
        assert calls["tool_selection"] == []
        assert calls["extraction"] == []
        assert calls["answer"] == []
        assert calls["e2e"] == []

    @pytest.mark.asyncio
    async def test_all_runs_every_suite(self, patch_runners) -> None:
        calls, _ = patch_runners
        args = _build_parser().parse_args([])
        assert await _run(args) == 0
        assert len(calls["tool_selection"]) == 1
        assert len(calls["extraction"]) == 1
        assert len(calls["answer"]) == 1
        assert len(calls["e2e"]) == 1

    @pytest.mark.asyncio
    async def test_single_suite_selection(self, patch_runners) -> None:
        calls, _ = patch_runners
        args = _build_parser().parse_args(["--suite", "answer"])
        assert await _run(args) == 0
        assert calls["tool_selection"] == []
        assert len(calls["answer"]) == 1
        assert calls["e2e"] == []

    @pytest.mark.asyncio
    async def test_comma_separated_suite_selection(self, patch_runners) -> None:
        calls, _ = patch_runners
        args = _build_parser().parse_args(["--suite", "answer,e2e"])
        assert await _run(args) == 0
        assert calls["tool_selection"] == []
        assert calls["extraction"] == []
        assert len(calls["answer"]) == 1
        assert len(calls["e2e"]) == 1

    @pytest.mark.asyncio
    async def test_unknown_suite_is_a_hard_error(self, patch_runners) -> None:
        calls, _ = patch_runners
        args = _build_parser().parse_args(["--suite", "bogus"])
        with pytest.raises(SystemExit, match="bogus"):
            await _run(args)
        assert calls["tool_selection"] == []

    @pytest.mark.asyncio
    async def test_sample_slices_items(self, patch_runners) -> None:
        calls, _ = patch_runners
        args = _build_parser().parse_args(["--sample", "1"])
        await _run(args)
        # Full tool-selection set is 15 items; --sample 1 passes 1 item.
        assert len(calls["tool_selection"][0]) == 1

    @pytest.mark.asyncio
    async def test_judge_flag_forwarded(self, patch_runners) -> None:
        _, results = patch_runners
        seen: dict = {}

        async def _fake_extraction(**kw):
            seen.update(kw)
            return results["extraction"]

        from unittest.mock import patch

        import tests.eval.llm_runner as runner_mod

        with patch.object(runner_mod, "run_extraction", _fake_extraction):
            args = _build_parser().parse_args(
                ["--suite", "extraction", "--judge", "deterministic"]
            )
            assert await _run(args) == 0
        assert seen.get("judge") == "deterministic"

    @pytest.mark.asyncio
    async def test_threshold_gate_fails_with_exit_2(self, monkeypatch) -> None:

        async def _low(**kw):
            return _fake_result("tool_selection", {"tool_accuracy": 0.3})

        monkeypatch.setattr(
            "tests.eval.llm_runner.run_tool_selection", _low
        )
        args = _build_parser().parse_args(
            ["--suite", "tool_selection", "--min-tool-accuracy", "0.9"]
        )
        assert await _run(args) == 2

    @pytest.mark.asyncio
    async def test_threshold_gate_applies_only_to_suites_reporting_metric(
        self, patch_runners, monkeypatch, capsys
    ) -> None:
        """Cross-suite flags must not fail a suite that doesn't produce them.

        Regression: with --suite all and the eval.yml flag set, every suite
        was gated on every metric; check_thresholds reads an absent metric as
        0.0, so tool_selection failed on entity_f1/relation_f1/fact_coverage/
        groundedness and the gate could never pass — even at perfect scores.
        """
        _, results = patch_runners
        low_extraction = _fake_result(
            "extraction", {"entity_f1": 0.2, "relation_f1": 0.2}
        )

        async def _low_extraction(**kw):
            return low_extraction

        monkeypatch.setattr(
            "tests.eval.llm_runner.run_extraction", _low_extraction
        )
        args = _build_parser().parse_args(
            [
                "--suite", "all",
                "--min-tool-accuracy", "0.70",
                "--min-entity-f1", "0.60",
                "--min-relation-f1", "0.50",
                "--min-fact-coverage", "0.60",
                "--min-groundedness", "0.80",
            ]
        )
        # Extraction fails its own entity_f1 floor → exit 2, and only
        # extraction is flagged — not the suites that don't report entity_f1.
        assert await _run(args) == 2
        err = capsys.readouterr().err
        assert "failed for extraction" in err
        assert "failed for tool_selection" not in err
        assert "failed for answer" not in err

        # With all suites at perfect scores the same flag set passes.
        low_extraction.overall = {"entity_f1": 1.0, "relation_f1": 1.0}
        assert await _run(args) == 0

    @pytest.mark.asyncio
    async def test_threshold_for_absent_metric_warns_not_gates(
        self, patch_runners, capsys
    ) -> None:
        """A --min-* flag for a metric no run suite produces warns and exits 0."""
        args = _build_parser().parse_args(
            ["--suite", "tool_selection", "--min-entity-f1", "0.9"]
        )
        assert await _run(args) == 0
        assert "no suite in this run reports that metric" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_execution_errors_exit_1(self, monkeypatch) -> None:
        def _err_result(suite: str) -> LlmEvalResult:
            r = _fake_result(suite, {"tool_accuracy": 0.0})
            r.errors = [{"id": "x", "error": "provider down"}]
            return r

        async def _broken(**kw):
            return _err_result("tool_selection")

        monkeypatch.setattr(
            "tests.eval.llm_runner.run_tool_selection", _broken
        )
        args = _build_parser().parse_args(["--suite", "tool_selection"])
        assert await _run(args) == 1


class TestJudgeProviderGuard:
    """--judge llm without a dedicated judge provider must fail, not self-judge."""

    @pytest.mark.asyncio
    async def test_llm_judge_without_provider_fails(
        self, patch_runners, monkeypatch
    ) -> None:
        import tests.eval.run_llm_eval as run_mod

        monkeypatch.setattr(run_mod.config.llm, "judge_provider", "")
        args = _build_parser().parse_args(["--suite", "answer"])
        with pytest.raises(SystemExit, match="judge provider"):
            await _run(args)

    @pytest.mark.asyncio
    async def test_configured_provider_allows_llm_judge(
        self, patch_runners
    ) -> None:
        # patch_runners sets the full LLM_JUDGE_* block → the run proceeds.
        args = _build_parser().parse_args(["--suite", "answer"])
        assert await _run(args) == 0

    @pytest.mark.asyncio
    async def test_llm_judge_with_provider_but_missing_model_fails(
        self, patch_runners, monkeypatch
    ) -> None:
        import tests.eval.run_llm_eval as run_mod

        monkeypatch.setattr(run_mod.config.llm, "judge_provider", "zhipu")
        monkeypatch.setattr(run_mod.config.llm, "judge_model", "")
        monkeypatch.setattr(run_mod.config.llm, "judge_api_key", "test-key")
        monkeypatch.setattr(
            run_mod.config.llm, "judge_base_url", "https://open.bigmodel.cn/api/paas/v4"
        )
        args = _build_parser().parse_args(["--suite", "answer"])
        with pytest.raises(SystemExit, match="LLM_JUDGE_MODEL"):
            await _run(args)

    @pytest.mark.asyncio
    async def test_llm_judge_with_provider_but_missing_api_key_fails(
        self, patch_runners, monkeypatch
    ) -> None:
        import tests.eval.run_llm_eval as run_mod

        monkeypatch.setattr(run_mod.config.llm, "judge_provider", "zhipu")
        monkeypatch.setattr(run_mod.config.llm, "judge_model", "glm-4.7-flash")
        monkeypatch.setattr(run_mod.config.llm, "judge_api_key", "")
        monkeypatch.setattr(
            run_mod.config.llm, "judge_base_url", "https://open.bigmodel.cn/api/paas/v4"
        )
        args = _build_parser().parse_args(["--suite", "answer"])
        with pytest.raises(SystemExit, match="LLM_JUDGE_API_KEY"):
            await _run(args)

    @pytest.mark.asyncio
    async def test_llm_judge_openai_compat_requires_base_url(
        self, patch_runners, monkeypatch
    ) -> None:
        import tests.eval.run_llm_eval as run_mod

        monkeypatch.setattr(run_mod.config.llm, "judge_provider", "openai")
        monkeypatch.setattr(run_mod.config.llm, "judge_model", "gpt-4o-mini")
        monkeypatch.setattr(run_mod.config.llm, "judge_api_key", "test-key")
        monkeypatch.setattr(run_mod.config.llm, "judge_base_url", "")
        args = _build_parser().parse_args(["--suite", "answer"])
        with pytest.raises(SystemExit, match="LLM_JUDGE_BASE_URL"):
            await _run(args)

    @pytest.mark.asyncio
    async def test_llm_judge_anthropic_needs_no_base_url(
        self, patch_runners, monkeypatch
    ) -> None:
        import tests.eval.run_llm_eval as run_mod

        monkeypatch.setattr(run_mod.config.llm, "judge_provider", "anthropic")
        monkeypatch.setattr(run_mod.config.llm, "judge_model", "claude-sonnet-5")
        monkeypatch.setattr(run_mod.config.llm, "judge_api_key", "test-key")
        monkeypatch.setattr(run_mod.config.llm, "judge_base_url", "")
        args = _build_parser().parse_args(["--suite", "answer"])
        assert await _run(args) == 0

    @pytest.mark.asyncio
    async def test_deterministic_judge_needs_no_provider(
        self, patch_runners, monkeypatch
    ) -> None:
        import tests.eval.run_llm_eval as run_mod

        monkeypatch.setattr(run_mod.config.llm, "judge_provider", "")
        args = _build_parser().parse_args(
            ["--suite", "answer", "--judge", "deterministic"]
        )
        assert await _run(args) == 0

    @pytest.mark.asyncio
    async def test_validate_only_skips_guard(
        self, patch_runners, monkeypatch
    ) -> None:
        import tests.eval.run_llm_eval as run_mod

        monkeypatch.setattr(run_mod.config.llm, "judge_provider", "")
        args = _build_parser().parse_args(["--validate-only"])
        assert await _run(args) == 0

    @pytest.mark.asyncio
    async def test_tool_selection_only_skips_guard(
        self, patch_runners, monkeypatch
    ) -> None:
        # tool_selection never consults the judge → no provider required.
        import tests.eval.run_llm_eval as run_mod

        monkeypatch.setattr(run_mod.config.llm, "judge_provider", "")
        args = _build_parser().parse_args(["--suite", "tool_selection"])
        assert await _run(args) == 0


class TestJudgeChannelExitCode:
    """≥50% judge failures on a suite reads as a broken run (exit 1)."""

    @pytest.mark.asyncio
    async def test_majority_judge_failures_exit_1(
        self, patch_runners, monkeypatch, capsys
    ) -> None:
        r = _fake_result("answer", {"fact_coverage": 0.0, "groundedness": 0.0})
        r.judge = "llm"
        r.n_items = 2
        r.judge_errors = [
            {"id": "a1", "error": "down"},
            {"id": "a2", "error": "down"},
        ]

        async def _fake(**kw):
            return r

        monkeypatch.setattr("tests.eval.llm_runner.run_answer", _fake)
        args = _build_parser().parse_args(["--suite", "answer"])
        assert await _run(args) == 1
        assert "judge channel failed" in capsys.readouterr().err

    @pytest.mark.asyncio
    async def test_minority_judge_failures_do_not_fail_run(
        self, patch_runners, monkeypatch, capsys
    ) -> None:
        r = _fake_result("answer", {"fact_coverage": 1.0, "groundedness": 1.0})
        r.judge = "llm"
        r.n_items = 4
        r.judge_errors = [{"id": "a1", "error": "down"}]  # 25% < 50%

        async def _fake(**kw):
            return r

        monkeypatch.setattr("tests.eval.llm_runner.run_answer", _fake)
        args = _build_parser().parse_args(["--suite", "answer"])
        assert await _run(args) == 0
        err = capsys.readouterr().err
        assert "judge degraded on 1/4 rows" in err
        assert "judge channel failed" not in err
