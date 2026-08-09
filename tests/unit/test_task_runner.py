"""Unit tests for tests/eval/task_runner.py — orchestration + aggregation.

Uses fake executors (no LLM / no DB / no graph) so the tests are fast and
deterministic.  The contract being tested mirrors ``test_llm_runner``:
    - run_tasks executes every item and aggregates per-task metrics
    - by_category roll-ups cover every registered category
    - executor failures are recorded in errors AND count as zero rows
    - judge degradation zeroes the judge-owned metric keys (never re-feeding
      the deterministic channel into the same gate keys)
    - deterministic mode drops the judge-owned keys
"""

from __future__ import annotations

import pytest

from tests.eval.task_executors import TaskOutcome
from tests.eval.task_ground_truth import TaskItem
from tests.eval.task_runner import run_tasks

MEMORY_ANSWER = (
    "向量检索后端选用了 pgvector，文档用 AST 按函数边界切分（记忆 c4a11b2e）。"
)


def _task(**overrides) -> TaskItem:
    base = TaskItem(
        id="t1",
        query="选型是什么",
        expected_tools=["search_memories_tool", "retrieve_chunks_tool"],
        category="multi_retrieve",
        required_facts=["pgvector", "AST"],
        prohibited_claims=["选了 Milvus"],
    )
    for k, v in overrides.items():
        object.__setattr__(base, k, v)
    return base


def _outcome(**overrides) -> TaskOutcome:
    base = TaskOutcome(
        answer=MEMORY_ANSWER,
        tool_calls=[
            {"name": "search_memories_tool", "args": {"query": "选型"}},
            {"name": "retrieve_chunks_tool", "args": {"query": "分块"}},
        ],
        n_steps=3,
        within_budget=True,
        had_error=False,
        context_text="<memory>pgvector 选型</memory>\n<doc>AST 分块</doc>",
        source_ids=["c4a11b2e", "doc-1"],
    )
    for k, v in overrides.items():
        object.__setattr__(base, k, v)
    return base


class TestRunTasks:
    @pytest.mark.asyncio
    async def test_scores_and_aggregates(self) -> None:
        async def fake_executor(query: str) -> TaskOutcome:
            return _outcome()

        result = await run_tasks(
            items=[_task()], executor=fake_executor, judge="deterministic"
        )
        assert result.suite == "task"
        assert result.judge == "deterministic"
        assert result.n_items == 1
        assert result.errors == []
        row = result.per_query[0]
        assert row["completed"] == 1.0
        assert row["tool_recall"] == 1.0
        assert row["within_budget"] == 1.0
        assert row["fact_coverage"] == 1.0  # pgvector + AST in answer
        assert row["groundedness"] == 1.0  # deterministic substring channel
        assert row["citation_rate"] == 1.0  # answer cites c4a11b2e
        assert row["n_steps"] == 3
        assert "fact_coverage" in result.metric_keys
        # Metric keys are stable across judge modes (answer/e2e convention) —
        # a deterministic run is a cheaper version of the same keys.
        assert "groundedness" in result.metric_keys
        assert result.overall["completed"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_partial_trajectory_scores_partial_recall(self) -> None:
        async def fake_executor(query: str) -> TaskOutcome:
            o = _outcome()
            o.tool_calls = [{"name": "search_memories_tool", "args": {}}]
            return o

        result = await run_tasks(
            items=[_task()], executor=fake_executor, judge="deterministic"
        )
        row = result.per_query[0]
        assert row["completed"] == 0.0  # missing retrieve_chunks_tool
        assert row["tool_recall"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_timeout_reads_as_failed_task(self) -> None:
        async def fake_executor(query: str) -> TaskOutcome:
            return _outcome(
                answer="",
                tool_calls=[],
                had_error=True,
                error="timeout",
                n_steps=0,
            )

        result = await run_tasks(
            items=[_task()], executor=fake_executor, judge="deterministic"
        )
        row = result.per_query[0]
        assert row["completed"] == 0.0
        assert row["tool_recall"] == 0.0

    @pytest.mark.asyncio
    async def test_exception_recorded_and_counts_as_zero(self) -> None:
        async def flaky(query: str) -> TaskOutcome:
            raise RuntimeError("graph exploded")

        result = await run_tasks(
            items=[_task()], executor=flaky, judge="deterministic"
        )
        assert len(result.errors) == 1
        assert result.errors[0]["id"] == "t1"
        row = result.per_query[0]
        assert row["error"]
        assert row["completed"] == 0.0
        assert row["fact_coverage"] == 0.0

    @pytest.mark.asyncio
    async def test_judge_channel_overrides(self, monkeypatch) -> None:
        import tests.eval.task_runner as runner_mod

        async def fake_executor(query: str) -> TaskOutcome:
            return _outcome()

        async def fake_judge(query, context, answer, required_facts):
            return {
                "covered_facts": ["pgvector"],  # reports only one of two facts
                "grounded": True,
                "ungrounded_claims": [],
            }

        monkeypatch.setattr(runner_mod, "judge_answer", fake_judge)
        result = await run_tasks(
            items=[_task()], executor=fake_executor, judge="llm"
        )
        row = result.per_query[0]
        assert result.judge == "llm"
        assert "groundedness" in result.metric_keys
        assert row["fact_coverage"] == pytest.approx(0.5)  # only pgvector covered
        assert row["groundedness"] == 1.0

    @pytest.mark.asyncio
    async def test_judge_failure_zeroes_judge_metrics_keeps_det_channel(
        self, monkeypatch
    ) -> None:
        import tests.eval.task_runner as runner_mod

        async def fake_executor(query: str) -> TaskOutcome:
            return _outcome()

        async def failing_judge(query, context, answer, required_facts):
            raise RuntimeError("judge down")

        monkeypatch.setattr(runner_mod, "judge_answer", failing_judge)
        result = await run_tasks(
            items=[_task()], executor=fake_executor, judge="llm"
        )
        assert len(result.judge_errors) == 1
        assert result.errors == []
        row = result.per_query[0]
        assert row["judge_error"]
        # Judge-owned keys zeroed — NOT fed the deterministic channel.
        assert row["fact_coverage"] == 0.0
        assert row["groundedness"] == 0.0
        assert row["hallucination_rate"] == 0.0
        # Deterministic channel survives under det_* and citation is untouched.
        assert row["det_fact_coverage"] == 1.0
        assert row["citation_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_by_category_covers_all_registered(self) -> None:
        async def fake_executor(query: str) -> TaskOutcome:
            if query == "refrain":
                return _outcome(tool_calls=[], answer="好的，明白了，谢谢！", source_ids=[])
            return _outcome()

        items = [
            _task(id="a", category="factual", expected_tools=["search_memories_tool"]),
            _task(id="b", category="multi_retrieve"),
            _task(id="c", category="write", expected_tools=["write_memory_tool"]),
            _task(id="d", category="conceptual"),
            _task(id="e", category="notify"),
            _task(id="f", category="no_tool", expected_tools=[], query="refrain"),
        ]
        result = await run_tasks(items=items, executor=fake_executor, judge="deterministic")
        for cat in ("factual", "multi_retrieve", "write", "conceptual", "notify", "no_tool"):
            assert cat in result.by_category
        # A refrain task that calls nothing is completed.
        assert result.by_category["no_tool"]["completed"] == 1.0


class TestMakeDefaultTaskExecutor:
    @pytest.mark.asyncio
    async def test_default_executor_is_used_when_none_injected(self, monkeypatch) -> None:
        """run_tasks builds the real executor via make_task_runner."""
        import tests.eval.task_runner as runner_mod

        built = []

        def _fake_factory(**kwargs):
            built.append(kwargs)

            async def fake_executor(query: str) -> TaskOutcome:
                return _outcome()

            return fake_executor

        monkeypatch.setattr(runner_mod, "make_task_runner", _fake_factory)
        await run_tasks(items=[_task()], judge="deterministic")
        assert built, "the default executor factory must be invoked"
        # max_steps must be forwarded to the factory when provided.
        await run_tasks(items=[_task()], judge="deterministic", max_steps=3)
        assert built[-1].get("max_steps") == 3
