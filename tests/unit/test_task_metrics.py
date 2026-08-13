"""Unit tests for tests/eval/task_metrics.py + task_ground_truth.py.

Metrics are pure functions — no LLM / no DB.  The ground-truth tests validate
the labeled task set's internal consistency (ids, categories, tool names) and
its reachability contract (every required fact must exist in the e2e seed
corpus, or the fact_coverage metric is structurally capped below 1.0).
"""

from __future__ import annotations

import pytest

from tests.eval.task_ground_truth import (
    TASK_ITEMS,
    validate_task_dataset,
)
from tests.eval.task_metrics import (
    is_apology_stub,
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

    def test_answer_metrics_delegate(self) -> None:
        from tests.eval.task_metrics import answer_metrics

        m = answer_metrics(SUBSTANTIVE, ["pgvector", "Elasticsearch"], ["选了 Qdrant"])
        assert m["fact_coverage"] == 1.0
        assert m["groundedness"] == 1.0
        m2 = answer_metrics("我们选了 Qdrant", ["pgvector"], ["选了 Qdrant"])
        assert m2["groundedness"] == 0.0


class TestValidateTaskDataset:
    def test_default_set_is_clean(self) -> None:
        assert validate_task_dataset() == []

    def test_duplicate_id_raises(self) -> None:

        items = list(TASK_ITEMS) + [TASK_ITEMS[0]]
        with pytest.raises(ValueError, match="duplicate task item id"):
            _validate(items)

    def test_unknown_category_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown category"):
            _validate([_item(category="bogus")])

    def test_unknown_tool_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown tool"):
            _validate([_item(expected_tools=["no_such_tool"])])

    def test_tool_task_without_facts_raises(self) -> None:
        with pytest.raises(ValueError, match="required_facts is empty"):
            _validate([_item(required_facts=[])])

    def test_fact_not_in_seed_corpus_warns(self) -> None:
        warnings = _validate([_item(required_facts=["zzz-no-such-fact-in-seed"])])
        assert any("not in any e2e seed" in w for w in warnings)

    def test_empty_expected_with_forbidden_warns(self) -> None:
        warnings = _validate(
            [_item(expected_tools=[], forbidden_tools=["notify_feishu_tool"])]
        )
        assert any("vacuous" in w for w in warnings)


# ── Helpers ──────────────────────────────────────────────────────────


def _item(**overrides) -> object:
    from dataclasses import replace


    base = TASK_ITEMS[0]
    return replace(base, **overrides)


def _validate(items: list) -> list[str]:
    """Validate a custom item list without touching the module global."""
    from tests.eval.task_ground_truth import validate_task_dataset

    return validate_task_dataset(items=list(items))
