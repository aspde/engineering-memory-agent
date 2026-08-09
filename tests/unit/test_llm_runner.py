"""Unit tests for tests/eval/llm_runner.py — orchestration + aggregation.

Uses fake executors (no LLM / no DB) so the tests are fast and
deterministic.  The contract being tested:
    - each suite executes every item and aggregates per-item metrics
    - by_category roll-ups cover every registered category
    - executor failures are recorded in errors AND count as zero rows
    - judge degradation is recorded separately in judge_errors: the extraction
      suite marks the row judge_error without writing a fake 0.0 verdict, and
      the answer / e2e suites zero the judge-owned metric keys (rather than
      re-feeding the deterministic channel into the same gate keys)
    - judge mode selection controls which metric keys appear
"""

from __future__ import annotations

import pytest

import tests.eval.llm_runner as runner_mod
from tests.eval.llm_executors import E2EOutcome
from tests.eval.llm_ground_truth import (
    AnswerItem,
    E2EItem,
    ExtractionItem,
    ToolSelectionItem,
)
from tests.eval.llm_runner import (
    run_answer,
    run_e2e,
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
            source_ids=["a1b2c3d4"],
        ),
        AnswerItem(
            id="a2",
            query="为什么 502",
            context="连接池被占满",
            required_facts=["连接池"],
            category="causal",
            prohibited_claims=[],
            source_ids=["e5f6a7b8"],
        ),
    ]


@pytest.fixture
def e2e_items() -> list[E2EItem]:
    return [
        E2EItem(
            id="x1",
            query="选型是什么",
            source_content="用 pgvector 而非 Elasticsearch 做向量检索，支持 cosine",
            required_facts=["pgvector", "Elasticsearch"],
            category="factual",
            prohibited_claims=["选了 Milvus"],
        ),
        E2EItem(
            id="x2",
            query="为什么 502",
            source_content="连接池被占满导致 502",
            required_facts=["连接池", "502"],
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
    async def test_judge_failure_is_marked_not_faked(self, extraction_items, monkeypatch) -> None:
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
        # A failed judgment must NOT be recorded as a real 0.0 verdict: the row
        # is marked judge_error and the summary keys are left unset rather than
        # written as a fake score.
        for q in result.per_query:
            assert q["judge_error"]
            assert "summary_faithfulness" not in q
            assert "summary_completeness" not in q
        # All rows degraded → the aggregate still reads as 0.0 (a missing key
        # counts against the denominator), so a judge outage is never mistaken
        # for perfect summaries.
        assert result.overall["summary_faithfulness"] == 0.0


# ── Final answer ─────────────────────────────────────────────────


class TestRunAnswer:
    @pytest.mark.asyncio
    async def test_deterministic_channel(self, answer_items) -> None:
        async def fake_generator(query: str, context: str, source_ids=None):
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
        # Traceability: neither fake answer cites a golden source id.
        assert by_id["a1"]["citation_rate"] == 0.0
        assert result.overall["citation_rate"] == 0.0
        # det_* sub-keys always present for cross-checking.
        assert "det_fact_coverage" in by_id["a1"]

    @pytest.mark.asyncio
    async def test_deterministic_catches_prohibited_claim(self, answer_items) -> None:
        async def hallucinating(query: str, context: str, source_ids=None):
            return "我们用了 Qdrant 而不是 pgvector"

        result = await run_answer(
            items=answer_items, generator=hallucinating, judge="deterministic"
        )
        by_id = {q["id"]: q for q in result.per_query}
        assert by_id["a1"]["groundedness"] == 0.0
        assert by_id["a1"]["hallucination_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_citation_credited_when_answer_cites_source(self, answer_items) -> None:
        async def citing(query: str, context: str, source_ids=None):
            # Cite the id exactly as the search display exposes it.
            return "我们选用了 pgvector（记忆 a1b2c3d4）"

        result = await run_answer(
            items=answer_items, generator=citing, judge="deterministic"
        )
        by_id = {q["id"]: q for q in result.per_query}
        # a1's answer cites its golden id → 1.0; a2's cites a *different* id
        # (not among its own source_ids) → 0.0.
        assert by_id["a1"]["citation_rate"] == 1.0
        assert by_id["a2"]["citation_rate"] == 0.0
        assert result.overall["citation_rate"] == 0.5

    @pytest.mark.asyncio
    async def test_judge_channel_overrides(self, answer_items, monkeypatch) -> None:
        async def fake_generator(query: str, context: str, source_ids=None):
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
    async def test_judge_failure_zeroes_judge_metrics_keeps_det_channel(
        self, answer_items, monkeypatch
    ) -> None:
        async def fake_generator(query: str, context: str, source_ids=None):
            return "我们选了 pgvector 而非 Elasticsearch"

        async def failing_judge(query, context, answer, required_facts):
            raise RuntimeError("judge down")

        monkeypatch.setattr(runner_mod, "judge_answer", failing_judge)
        result = await run_answer(
            items=answer_items, generator=fake_generator, judge="llm"
        )
        assert len(result.judge_errors) == 2
        assert result.errors == []
        by_id = {q["id"]: q for q in result.per_query}
        # Judge-owned metric keys are zeroed — NOT overwritten with the
        # deterministic channel — so the CI gate on --min-groundedness fails
        # loudly instead of silently grading substring matches.
        assert by_id["a1"]["fact_coverage"] == 0.0
        assert by_id["a1"]["groundedness"] == 0.0
        assert by_id["a1"]["hallucination_rate"] == 0.0
        assert by_id["a2"]["fact_coverage"] == 0.0
        assert by_id["a1"]["judge_error"]
        # The deterministic channel survives under det_* for cross-checking,
        # and deterministic citation measurement is untouched.
        assert by_id["a1"]["det_fact_coverage"] == 1.0  # covers "pgvector"
        assert by_id["a2"]["det_fact_coverage"] == 0.0  # needs "连接池"
        assert by_id["a1"]["citation_rate"] == 0.0  # no golden id cited


# ── End-to-end (E2E) ─────────────────────────────────────────────


def _e2e_outcome(
    answer: str,
    context_text: str,
    source_ids: list[str],
    n_retrieved: int = 1,
) -> E2EOutcome:
    return E2EOutcome(
        answer=answer,
        context_text=context_text,
        retrieved_source_ids=source_ids,
        n_retrieved=n_retrieved,
    )


def _memory_context(text: str, sid: str) -> str:
    # Mirrors the production search_memories_tool display format.
    return (
        f"<memory source=\"search_memories_tool\">\n"
        f"[1] (memory: {sid[:8]}, relevance: 0.90, decay: 1.00) {text}\n"
        f"</memory>"
    )


class TestRunE2E:
    @pytest.mark.asyncio
    async def test_grounded_full_context(self, e2e_items) -> None:
        async def fake_runner(query: str):
            # Both items' retrieval surfaces the full context → both score 1.0,
            # so the aggregate is meaningful too.
            if "502" in query:
                text = "连接池被占满导致 502"
                sid = "e5f6a7b8"
                answer = "根因是连接池被占满导致 502（记忆 e5f6a7b8）"
            else:
                text = "用 pgvector 而非 Elasticsearch 做向量检索，支持 cosine"
                sid = "a1b2c3d4"
                answer = "我们选用 pgvector 而非 Elasticsearch（记忆 a1b2c3d4）"
            return _e2e_outcome(
                answer=answer,
                context_text=_memory_context(text, sid),
                source_ids=[sid],
            )

        result = await run_e2e(
            items=e2e_items, runner=fake_runner, judge="deterministic"
        )
        by_id = {q["id"]: q for q in result.per_query}
        # x1: both facts in retrieved context AND in the answer → 1.0 each.
        assert by_id["x1"]["context_recall"] == 1.0
        assert by_id["x1"]["fact_coverage"] == 1.0
        assert by_id["x1"]["groundedness"] == 1.0
        assert by_id["x1"]["citation_rate"] == 1.0  # cited a retrieved id
        assert by_id["x1"]["n_retrieved"] == 1
        assert result.overall["context_recall"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_partial_context_bounds_coverage(self, e2e_items) -> None:
        # Retrieval returned context missing "Elasticsearch" → context_recall
        # is partial, and the answer (built from that context) can't cover it.
        async def fake_runner(query: str):
            return _e2e_outcome(
                answer="我们选用 pgvector",
                context_text=_memory_context("用 pgvector 做向量检索", "a1b2c3d4"),
                source_ids=["a1b2c3d4"],
            )

        result = await run_e2e(
            items=e2e_items, runner=fake_runner, judge="deterministic"
        )
        by_id = {q["id"]: q for q in result.per_query}
        assert by_id["x1"]["context_recall"] == pytest.approx(0.5)
        assert by_id["x1"]["fact_coverage"] == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_hallucinating_answer_despite_good_context(self, e2e_items) -> None:
        # Retrieval was fine (context_recall 1.0) but the answer invented a
        # prohibited claim — groundedness catches the generation failure.
        async def fake_runner(query: str):
            return _e2e_outcome(
                answer="我们选了 Milvus 而不是 pgvector",
                context_text=_memory_context(
                    "用 pgvector 而非 Elasticsearch 做向量检索，支持 cosine", "a1b2c3d4"
                ),
                source_ids=["a1b2c3d4"],
            )

        result = await run_e2e(
            items=e2e_items, runner=fake_runner, judge="deterministic"
        )
        by_id = {q["id"]: q for q in result.per_query}
        assert by_id["x1"]["context_recall"] == 1.0
        assert by_id["x1"]["groundedness"] == 0.0
        assert by_id["x1"]["hallucination_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_citation_only_against_retrieved_ids(self, e2e_items) -> None:
        # The answer cites an id that was NOT retrieved → no citation credit.
        async def fake_runner(query: str):
            return _e2e_outcome(
                answer="我们选用 pgvector（记忆 e5f6a7b8）",
                context_text=_memory_context(
                    "用 pgvector 而非 Elasticsearch 做向量检索，支持 cosine", "a1b2c3d4"
                ),
                source_ids=["a1b2c3d4"],
            )

        result = await run_e2e(
            items=e2e_items, runner=fake_runner, judge="deterministic"
        )
        by_id = {q["id"]: q for q in result.per_query}
        assert by_id["x1"]["citation_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_judge_channel_overrides(self, e2e_items, monkeypatch) -> None:
        async def fake_runner(query: str):
            return _e2e_outcome(
                answer="答案内容",
                context_text=_memory_context(
                    "用 pgvector 而非 Elasticsearch 做向量检索，支持 cosine", "a1b2c3d4"
                ),
                source_ids=["a1b2c3d4"],
            )

        async def fake_judge(query, context, answer, required_facts):
            return {"covered_facts": ["pgvector"], "grounded": True, "ungrounded_claims": []}

        monkeypatch.setattr(runner_mod, "judge_answer", fake_judge)
        result = await run_e2e(
            items=e2e_items, runner=fake_runner, judge="llm"
        )
        by_id = {q["id"]: q for q in result.per_query}
        # x1 requires ["pgvector","Elasticsearch"]; judge reports only pgvector.
        assert by_id["x1"]["fact_coverage"] == pytest.approx(0.5)
        assert by_id["x1"]["groundedness"] == 1.0

    @pytest.mark.asyncio
    async def test_judge_failure_zeroes_judge_metrics(self, e2e_items, monkeypatch) -> None:
        async def fake_runner(query: str):
            return _e2e_outcome(
                answer="我们选用 pgvector",
                context_text=_memory_context(
                    "用 pgvector 而非 Elasticsearch 做向量检索，支持 cosine", "a1b2c3d4"
                ),
                source_ids=["a1b2c3d4"],
            )

        async def failing_judge(query, context, answer, required_facts):
            raise RuntimeError("judge down")

        monkeypatch.setattr(runner_mod, "judge_answer", failing_judge)
        result = await run_e2e(
            items=e2e_items, runner=fake_runner, judge="llm"
        )
        assert len(result.judge_errors) == 2
        assert result.errors == []
        by_id = {q["id"]: q for q in result.per_query}
        # Judge metric keys are zeroed rather than fed the deterministic
        # channel, so the groundedness gate fails loudly on a judge outage.
        assert by_id["x1"]["fact_coverage"] == 0.0
        assert by_id["x1"]["groundedness"] == 0.0
        assert by_id["x1"]["judge_error"]
        assert by_id["x1"]["det_fact_coverage"] == pytest.approx(0.5)
        # context_recall is deterministic and survives judge degradation.
        assert by_id["x1"]["context_recall"] == 1.0

    @pytest.mark.asyncio
    async def test_failure_recorded_and_counts_as_zero(self, e2e_items) -> None:
        async def flaky(query: str):
            if "502" in query:
                raise RuntimeError("provider down")
            return _e2e_outcome(
                answer="我们选用 pgvector",
                context_text=_memory_context("用 pgvector 做向量检索", "a1b2c3d4"),
                source_ids=["a1b2c3d4"],
            )

        result = await run_e2e(
            items=e2e_items, runner=flaky, judge="deterministic"
        )
        assert len(result.errors) == 1
        assert result.errors[0]["id"] == "x2"
        zero_row = next(q for q in result.per_query if q["id"] == "x2")
        assert zero_row["error"]
        assert zero_row["context_recall"] == 0.0
        assert zero_row["fact_coverage"] == 0.0

    @pytest.mark.asyncio
    async def test_mixed_retrieval_modes_dispatch_per_item(self, monkeypatch) -> None:
        """Each item runs through the runner matching its retrieval_mode.

        Regression (Critical-C1): run_e2e built ONE global runner (from the
        module-level ``retrieval_mode``, default "memory") and reused it for
        every item, so a chunk-mode item ran through the memory retrieval path
        — context_recall was structurally 0 for it and the CI e2e gate
        (--min-context-recall 0.90 over the 7 memory + 1 chunk labeled items)
        could never pass.  Per-item dispatch must send memory items through a
        memory runner and chunk items through a chunk runner.
        """
        built: list[str] = []  # retrieval_mode of every default runner created
        calls: list[tuple[str, str]] = []  # (query, runner mode) actually used

        def _fake_make_default_e2e_runner(*, top_k=5, retrieval_mode="memory"):
            built.append(retrieval_mode)

            async def fake_runner(query: str):
                calls.append((query, retrieval_mode))
                if "分块" in query:
                    text = "文档用 AST 按函数边界切分，递归分隔符切普通文档"
                    sid = "c1c1c1c1"
                else:
                    text = "用 pgvector 做向量检索，支持 cosine"
                    sid = "a1b2c3d4"
                return _e2e_outcome(
                    answer=text,
                    context_text=_memory_context(text, sid),
                    source_ids=[sid],
                )

            return fake_runner

        monkeypatch.setattr(
            runner_mod, "make_default_e2e_runner", _fake_make_default_e2e_runner
        )
        items = [
            E2EItem(
                id="m1", query="选型是什么", source_content="用 pgvector",
                required_facts=["pgvector"], category="factual",
                retrieval_mode="memory",
            ),
            E2EItem(
                id="c1", query="分块怎么切", source_content="AST 切块",
                required_facts=["AST"], category="instruction",
                retrieval_mode="chunk",
            ),
            E2EItem(
                id="m2", query="为什么 502", source_content="连接池占满",
                required_facts=["连接池"], category="causal",
                retrieval_mode="memory",
            ),
        ]
        result = await run_e2e(items=items, judge="deterministic")
        # Exactly one runner per mode is built (cached), not one per item.
        assert built == ["memory", "chunk"]
        by_query = dict(calls)
        assert by_query["选型是什么"] == "memory"
        assert by_query["分块怎么切"] == "chunk"
        assert by_query["为什么 502"] == "memory"
        # The chunk item must not be structurally zeroed by a memory-path run.
        by_id = {q["id"]: q for q in result.per_query}
        assert by_id["c1"]["context_recall"] == 1.0
