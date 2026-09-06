"""Unit tests for evals/task_metrics.py — task completion scoring.

Metrics are pure functions — no LLM / no DB.  (The task ground-truth
validator tests live in ``test_dataset_validation.py``.)
"""

from __future__ import annotations

import pytest

from evals.task_metrics import (
    clean_completed_mean,
    is_apology_stub,
    outcome_class,
    task_completion_metrics,
)

SUBSTANTIVE = "向量检索后端选用了 pgvector 而不是 Elasticsearch，因为同库事务一致。"


class TestTaskCompletionMetrics:
    def test_all_expected_called_and_substantive(self) -> None:
        m = task_completion_metrics(
            called=["search_memories_tool", "retrieve_chunks_tool"],
            expected=["search_memories_tool", "retrieve_chunks_tool"],
            answer=SUBSTANTIVE,
        )
        assert m["completed"] == 1.0
        assert m["tool_recall"] == 1.0
        assert m["unexpected_rate"] == 0.0

    def test_missing_expected_tool_partial_recall(self) -> None:
        m = task_completion_metrics(
            called=["search_memories_tool"],
            expected=["search_memories_tool", "retrieve_chunks_tool"],
            answer=SUBSTANTIVE,
        )
        assert m["completed"] == 0.0  # task not done the intended way
        assert m["tool_recall"] == pytest.approx(0.5)

    def test_unexpected_tool_fails_completion_and_flags_unexpected(self) -> None:
        m = task_completion_metrics(
            called=["search_memories_tool", "ingest_git_repo_tool"],
            expected=["search_memories_tool"],
            answer=SUBSTANTIVE,
        )
        assert m["completed"] == 0.0
        assert m["unexpected_rate"] == 1.0

    def test_allowed_tool_not_unexpected_but_not_completed(self) -> None:
        m = task_completion_metrics(
            called=["retrieve_chunks_tool"],
            expected=["search_memories_tool"],
            allowed=["retrieve_chunks_tool"],
            answer=SUBSTANTIVE,
        )
        assert m["unexpected_rate"] == 0.0  # allowed substitute
        assert m["completed"] == 0.0        # still not the expected tool
        assert m["tool_recall"] == 0.0

    def test_forbidden_tool_fails_completion(self) -> None:
        m = task_completion_metrics(
            called=["search_memories_tool", "notify_feishu_tool"],
            expected=["search_memories_tool"],
            forbidden=["notify_feishu_tool"],
            answer=SUBSTANTIVE,
        )
        assert m["completed"] == 0.0

    def test_forbidden_wins_over_allowed(self) -> None:
        """A tool listed in BOTH allowed and forbidden is still unexpected.

        The dataset contradiction is the author's bug; the metric must not
        silently report the forbidden call as acceptable.  forbidden wins,
        matching the completion rule.
        """
        m = task_completion_metrics(
            called=["search_memories_tool", "notify_feishu_tool"],
            expected=["search_memories_tool"],
            allowed=["notify_feishu_tool"],  # contradictory with forbidden
            forbidden=["notify_feishu_tool"],
            answer=SUBSTANTIVE,
        )
        assert m["unexpected_rate"] == 1.0
        assert m["completed"] == 0.0

    def test_refrain_task_calls_nothing(self) -> None:
        m = task_completion_metrics(called=[], expected=[], answer="好的，明白了，谢谢！")
        assert m["completed"] == 1.0
        assert m["tool_recall"] == 1.0

    def test_refrain_task_calling_a_tool_fails(self) -> None:
        m = task_completion_metrics(
            called=["search_memories_tool"], expected=[], answer="好的，明白了，谢谢！"
        )
        assert m["completed"] == 0.0
        assert m["tool_recall"] == 1.0  # empty expected → vacuous recall

    def test_empty_answer_fails_completion(self) -> None:
        m = task_completion_metrics(
            called=["search_memories_tool"],
            expected=["search_memories_tool"],
            answer="",
        )
        assert m["completed"] == 0.0

    def test_apology_stub_fails_completion(self) -> None:
        assert is_apology_stub("抱歉，当前回答生成失败，请稍后重试。")
        m = task_completion_metrics(
            called=["search_memories_tool"],
            expected=["search_memories_tool"],
            answer="抱歉，当前回答生成失败，请稍后重试。",
        )
        assert m["completed"] == 0.0

    def test_had_error_fails_completion(self) -> None:
        m = task_completion_metrics(
            called=["search_memories_tool"],
            expected=["search_memories_tool"],
            answer=SUBSTANTIVE,
            had_error=True,
        )
        assert m["completed"] == 0.0

    def test_within_budget_passthrough(self) -> None:
        m = task_completion_metrics(
            called=["search_memories_tool"],
            expected=["search_memories_tool"],
            answer=SUBSTANTIVE,
            within_budget=False,
        )
        # Completed despite being wasteful — the two signals are distinct.
        assert m["completed"] == 1.0
        assert m["within_budget"] == 0.0


class TestOutcomeClass:
    """Environment-noise classification — the 2026-08-24 post-discipline
    runs lost 14/19 failure cells to provider outages and wall-clock
    timeouts; these tests pin the separation."""

    def test_clean_substantive_run_is_ok(self) -> None:
        assert outcome_class(SUBSTANTIVE, had_error=False) == "ok"

    def test_apology_stub_is_provider_error(self) -> None:
        assert outcome_class("抱歉，生成回复时出现错误，请稍后重试。", had_error=False) == (
            "provider_error"
        )

    def test_graph_error_is_provider_error_even_with_substantive_answer(self) -> None:
        assert outcome_class(SUBSTANTIVE, had_error=True) == "provider_error"

    def test_timeout_wins_over_had_error(self) -> None:
        assert outcome_class("", had_error=True, error="timeout") == "timeout"

    def test_within_budget_does_not_affect_the_class(self) -> None:
        # Running out of ReAct steps is agent behaviour, not environment noise.
        assert (
            outcome_class(SUBSTANTIVE, had_error=False, within_budget=False) == "ok"
        )


class TestCleanCompletedMean:
    def test_polluted_rows_drop_out_of_denominator(self) -> None:
        rows = [
            {"outcome_class": "ok", "completed": 1.0},
            {"outcome_class": "provider_error", "completed": 0.0},
            {"outcome_class": "ok", "completed": 1.0},
            {"outcome_class": "timeout", "completed": 0.0},
        ]
        assert clean_completed_mean(rows) == pytest.approx(1.0)

    def test_partial_clean_mean(self) -> None:
        rows = [
            {"outcome_class": "ok", "completed": 1.0},
            {"outcome_class": "ok", "completed": 0.0},
            {"outcome_class": "provider_error", "completed": 0.0},
        ]
        assert clean_completed_mean(rows) == pytest.approx(0.5)

    def test_no_clean_rows_yields_zero(self) -> None:
        rows = [{"outcome_class": "provider_error", "completed": 0.0}]
        assert clean_completed_mean(rows) == 0.0

    def test_empty_rows_yield_zero(self) -> None:
        assert clean_completed_mean([]) == 0.0

    def test_row_without_outcome_class_counts_as_clean(self) -> None:
        # A row missing the key (e.g. a crash row that never got classified)
        # defaults to "ok" — consistent with the report's per-cell default.
        rows = [{"completed": 1.0}]
        assert clean_completed_mean(rows) == pytest.approx(1.0)
