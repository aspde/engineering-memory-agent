"""Unit tests for tests/eval/llm_runner.py — orchestration + aggregation.

Uses fake executors (no LLM / no DB) so the tests are fast and
deterministic.  The contract being tested:
    - each suite executes every item and aggregates per-item metrics
    - by_category roll-ups cover every registered category
    - executor failures are recorded in errors AND count as zero rows
    - judge degradation (extraction / answer) is recorded separately and
      falls back to the deterministic channel
    - judge mode selection controls which metric keys appear
"""

from __future__ import annotations

import pytest

import tests.eval.llm_runner as runner_mod
from tests.eval.llm_ground_truth import (
    AnswerItem,
    ExtractionItem,
    ToolSelectionItem,
)
from tests.eval.llm_runner import (
    run_answer,
    run_extraction,
    run_tool_selection,
)


# ── Fixtures: minimal labeled items ──────────────────────────────


@pytest.fixture
def tool_items() -> list[ToolSelectionItem]:
    return [
        ToolSelectionItem(
            id="t1", query="怎么解决的", expected_tools=["search_memories_tool"],
            category="memory_search",
        ),
        ToolSelectionItem(
            id="t2", query="记住这个", expected_tools=["write_memory_tool"],
            category="write", expected_args={"write_memory_tool": ["CI"]},
        ),
        ToolSelectionItem(
            id="t3", query="你好", expected_tools=[], category="no_tool",
        ),
    ]


@pytest.fixture
def extraction_items() -> list[ExtractionItem]:
    return [
        ExtractionItem(
            id="e1",
            content="用 pgvector 而非 Elasticsearch 做检索",
            expected_entities=[
                {"name": "pgvector", "type": "technology"},
                {"name": "Elasticsearch", "type": "technology"},
            ],
            expected_relations=[
                {"from": "pgvector", "to": "Elasticsearch", "type": "supersedes"}
            ],
            category="code_decision",
            summary_keywords=["pgvector"],
        ),
        ExtractionItem(
            id="e2",
            content="连接池占满导致 502",
            expected_entities=[{"name": "连接池", "type": "concept"}],
            expected_relations=[
                {"from": "连接池", "to": "502", "type": "causes"}
            ],
            category="incident",
            summary_keywords=["连接池"],
        ),
    ]


@pytest.fixture
def answer_items() -> list[AnswerItem]:
    return [
        AnswerItem(
            id="a1",
            query="选型是什么",
            context="用 pgvector 而非 Elasticsearch",
            required_facts=["pgvector"],
            category="factual",
            prohibited_claims=["用了 Qdrant"],
        ),
        AnswerItem(
            id="a2",
            query="为什么 502",
            context="连接池被占满",
            required_facts=["连接池"],
            category="causal",
            prohibited_claims=[],
        ),
    ]


# ── Tool selection ───────────────────────────────────────────────


class TestRunToolSelection:
    @pytest.mark.asyncio
    async def test_scores_and_aggregates(self, tool_items) -> None:
        async def fake_selector(query: str):
            if "记住" in query:
                return [{"name": "write_memory_tool", "args": {"content": "CI 迁移"}}]
            if "你好" in query:
                return []
            return [{"name": "search_memories_tool", "args": {"query": query}}]

        result = await run_tool_selection(items=tool_items, selector=fake_selector)
        assert result.suite == "tool_selection"
        assert result.n_items == 3
        assert result.errors == []
        by_id = {q["id"]: q for q in result.per_query}
        assert by_id["t1"]["tool_accuracy"] == 1.0
        assert by_id["t2"]["tool_accuracy"] == 1.0
        assert by_id["t2"]["arg_match_rate"] == 1.0  # "CI" in args
        assert by_id["t3"]["tool_accuracy"] == 1.0  # refrained
        assert result.overall["tool_accuracy"] == 1.0
        # Every registered category must appear in by_category.
        assert "memory_search" in result.by_category
        assert "no_tool" in result.by_category

    @pytest.mark.asyncio
    async def test_wrong_choice_scores_zero(self, tool_items) -> None:
        async def always_write(query: str):
            return [{"name": "write_memory_tool", "args": {"content": query}}]

        result = await run_tool_selection(items=tool_items, selector=always_write)
        by_id = {q["id"]: q for q in result.per_query}
        assert by_id["t1"]["tool_accuracy"] == 0.0
        assert by_id["t3"]["tool_accuracy"] == 0.0  # called a tool on a refrain item
        assert result.overall["tool_accuracy"] == pytest.approx(1 / 3)

    @pytest.mark.asyncio
    async def test_failure_is_recorded_and_counts_as_zero(self, tool_items) -> None:
        async def flaky(query: str):
            if "记住" in query:
                raise RuntimeError("provider down")
            return [{"name": "search_memories_tool", "args": {}}]

        result = await run_tool_selection(items=tool_items, selector=flaky)
        assert len(result.errors) == 1
        assert result.errors[0]["id"] == "t2"
        zero_row = next(q for q in result.per_query if q["id"] == "t2")
        assert zero_row["error"]
        assert zero_row["tool_accuracy"] == 0.0


# ── Extraction ───────────────────────────────────────────────────


class TestRunExtraction:
    @pytest.mark.asyncio
    async def test_deterministic_metrics_no_judge_keys(self, extraction_items) -> None:
        async def fake_extractor(content: str):
            if "连接池" in content:
                return {
                    "summary": "连接池占满导致 502",
                    "entities": [{"name": "连接池", "type": "concept"}],
                    "relations": [
                        {"from": "连接池", "to": "502", "type": "causes"}
                    ],
                }
            return {
                "summary": "用 pgvector 做向量检索",
                "entities": [
                    {"name": "pgvector", "type": "technology"},
                    {"name": "Elasticsearch", "type": "technology"},
                ],
                "relations": [
                    {"from": "pgvector", "to": "Elasticsearch", "type": "supersedes"}
                ],
            }

        result = await run_extraction(
            items=extraction_items, extractor=fake_extractor, judge="deterministic"
        )
        assert result.judge == "deterministic"
        assert "summary_faithfulness" not in result.metric_keys
        assert "entity_f1" in result.metric_keys
        assert result.overall["entity_f1"] == pytest.approx(1.0)
        assert result.overall["relation_f1"] == pytest.approx(1.0)
        assert result.overall["summary_coverage"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_judge_mode_adds_judge_keys(self, extraction_items, monkeypatch) -> None:
        async def fake_extractor(content: str):
            return {"summary": "s", "entities": [], "relations": []}

        async def fake_judge(source: str, summary: str):
            return {"faithfulness": 0.8, "completeness": 0.6}

        monkeypatch.setattr(runner_mod, "judge_summary", fake_judge)
        result = await run_extraction(
            items=extraction_items, extractor=fake_extractor, judge="llm"
        )
        assert "summary_faithfulness" in result.metric_keys
        assert result.overall["summary_faithfulness"] == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_judge_failure_degrades_not_fatal(self, extraction_items, monkeypatch) -> None:
        async def fake_extractor(content: str):
            return {"summary": "s", "entities": [], "relations": []}

        async def failing_judge(source: str, summary: str):
            raise RuntimeError("judge down")

        monkeypatch.setattr(runner_mod, "judge_summary", failing_judge)
        result = await run_extraction(
            items=extraction_items, extractor=fake_extractor, judge="llm"
        )
        assert len(result.judge_errors) == 2
        assert result.errors == []  # execution succeeded; only judgment degraded
        assert result.overall["summary_faithfulness"] == 0.0


# ── Final answer ─────────────────────────────────────────────────


class TestRunAnswer:
    @pytest.mark.asyncio
    async def test_deterministic_channel(self, answer_items) -> None:
        async def fake_generator(query: str, context: str):
            if "502" in query:
                return "连接池被占满导致 502"
            return "我们选用了 pgvector"

        result = await run_answer(
            items=answer_items, generator=fake_generator, judge="deterministic"
        )
        assert result.judge == "deterministic"
        by_id = {q["id"]: q for q in result.per_query}
        assert by_id["a1"]["fact_coverage"] == 1.0
        assert by_id["a1"]["groundedness"] == 1.0
        assert by_id["a2"]["groundedness"] == 1.0
        assert result.overall["fact_coverage"] == 1.0
        # det_* sub-keys always present for cross-checking.
        assert "det_fact_coverage" in by_id["a1"]

    @pytest.mark.asyncio
    async def test_deterministic_catches_prohibited_claim(self, answer_items) -> None:
        async def hallucinating(query: str, context: str):
            return "我们用了 Qdrant 而不是 pgvector"

        result = await run_answer(
            items=answer_items, generator=hallucinating, judge="deterministic"
        )
        by_id = {q["id"]: q for q in result.per_query}
        assert by_id["a1"]["groundedness"] == 0.0
        assert by_id["a1"]["hallucination_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_judge_channel_overrides(self, answer_items, monkeypatch) -> None:
        async def fake_generator(query: str, context: str):
            return "答案内容"

        async def fake_judge(query, context, answer, required_facts):
            return {
                "covered_facts": ["pgvector"],
                "grounded": True,
                "ungrounded_claims": [],
            }

        monkeypatch.setattr(runner_mod, "judge_answer", fake_judge)
        result = await run_answer(
            items=answer_items, generator=fake_generator, judge="llm"
        )
        by_id = {q["id"]: q for q in result.per_query}
        # a1 requires ["pgvector"] → judge reports covered → 1.0;
        # a2 requires ["连接池"] → judge reports only pgvector → 0.0.
        assert by_id["a1"]["fact_coverage"] == 1.0
        assert by_id["a2"]["fact_coverage"] == 0.0
        assert result.overall["groundedness"] == 1.0

    @pytest.mark.asyncio
    async def test_judge_failure_falls_back_to_deterministic(
        self, answer_items, monkeypatch
    ) -> None:
        async def fake_generator(query: str, context: str):
            return "我们选了 pgvector 而非 Elasticsearch"

        async def failing_judge(query, context, answer, required_facts):
            raise RuntimeError("judge down")

        monkeypatch.setattr(runner_mod, "judge_answer", failing_judge)
        result = await run_answer(
            items=answer_items, generator=fake_generator, judge="llm"
        )
        assert len(result.judge_errors) == 2
        assert result.errors == []
        # Fallback: a1's answer covers "pgvector" → 1.0; a2 needs "连接池" → 0.0.
        by_id = {q["id"]: q for q in result.per_query}
        assert by_id["a1"]["fact_coverage"] == 1.0
        assert by_id["a2"]["fact_coverage"] == 0.0
