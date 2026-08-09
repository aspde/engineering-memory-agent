"""Unit tests for tests/eval/llm_metrics.py — pure-function correctness.

Covers the three suites' scoring: tool-selection accuracy (incl. the empty-
expected "refrain" case), arg-match, entity/relation P/R/F1 with tolerant
name matching, summary keyword coverage, and both answer channels
(deterministic substring + judge verdict).  Hand-computed cases so a
regression in the formulas surfaces immediately.
"""

from __future__ import annotations

import pytest

from tests.eval.llm_metrics import (
    answer_deterministic_metrics,
    answer_judge_metrics,
    citation_presence,
    context_recall,
    entity_metrics,
    names_match,
    normalize_name,
    relation_metrics,
    summary_keyword_coverage,
    tool_arg_match_rate,
    tool_selection_metrics,
)


class TestNamesMatch:
    def test_equal_normalized(self) -> None:
        assert names_match("PostgreSQL", "postgresql")
        assert names_match(" BGE-M3 ", "bge-m3")

    def test_containment_fires_above_min_length(self) -> None:
        assert names_match("BGE-M3", "BGE-M3 模型")
        assert names_match("connection leak", "database connection leak")

    def test_short_name_containment_is_guarded(self) -> None:
        # "CI" must not match "CI 构建" (len < min_contain)
        assert not names_match("CI", "CI 构建")

    def test_empty_names_never_match(self) -> None:
        assert not names_match("", "pgvector")
        assert not names_match("", "")

    def test_normalize_strips_whitespace(self) -> None:
        assert normalize_name("PostgreSQL 连接池") == "postgresql连接池"


class TestToolSelectionMetrics:
    def test_exact_match(self) -> None:
        m = tool_selection_metrics(
            ["search_memories_tool"],
            ["search_memories_tool"],
        )
        assert m["tool_accuracy"] == 1.0
        assert m["expected_recall"] == 1.0
        assert m["unexpected_rate"] == 0.0
        assert m["no_call"] == 0.0

    def test_partial_recall(self) -> None:
        m = tool_selection_metrics(
            ["search_memories_tool", "write_memory_tool"],
            ["search_memories_tool", "query_entity_tool"],
        )
        assert m["expected_recall"] == pytest.approx(0.5)
        assert m["tool_accuracy"] == 0.0

    def test_unexpected_call_is_wrong(self) -> None:
        m = tool_selection_metrics(
            ["search_memories_tool", "notify_feishu_tool"],
            ["search_memories_tool"],
        )
        assert m["unexpected_rate"] == 1.0
        assert m["tool_accuracy"] == 0.0

    def test_allowed_substitute_not_wrong_but_unsatisfied(self) -> None:
        m = tool_selection_metrics(
            ["retrieve_chunks_tool"],
            ["query_rewrite_and_search_tool"],
            allowed=["retrieve_chunks_tool"],
        )
        assert m["unexpected_rate"] == 0.0  # allowed, not wrong
        assert m["tool_accuracy"] == 0.0  # expected not called
        assert m["expected_recall"] == 0.0

    def test_forbidden_call_is_wrong(self) -> None:
        m = tool_selection_metrics(
            ["search_memories_tool"],
            ["search_memories_tool"],
            forbidden=["notify_feishu_tool"],
        )
        assert m["tool_accuracy"] == 1.0  # forbidden tool not called
        m2 = tool_selection_metrics(
            ["search_memories_tool", "notify_feishu_tool"],
            ["search_memories_tool"],
            forbidden=["notify_feishu_tool"],
        )
        assert m2["tool_accuracy"] == 0.0

    def test_empty_expected_requires_refraining(self) -> None:
        assert tool_selection_metrics([], [])["tool_accuracy"] == 1.0
        assert tool_selection_metrics(["write_memory_tool"], [])["tool_accuracy"] == 0.0

    def test_no_call_flag(self) -> None:
        m = tool_selection_metrics([], ["search_memories_tool"])
        assert m["no_call"] == 1.0
        assert m["tool_accuracy"] == 0.0


class TestToolArgMatchRate:
    def test_no_constraints(self) -> None:
        assert tool_arg_match_rate([], {}) == 1.0

    def test_arg_substring_present(self) -> None:
        calls = [
            {"name": "write_memory_tool", "args": {"content": "CI 迁移到自建 runner"}}
        ]
        rate = tool_arg_match_rate(calls, {"write_memory_tool": ["CI"]})
        assert rate == 1.0

    def test_missing_call_fails_constraint(self) -> None:
        rate = tool_arg_match_rate([], {"write_memory_tool": ["CI"]})
        assert rate == 0.0

    def test_partial(self) -> None:
        calls = [{"name": "write_memory_tool", "args": {"content": "CI 迁移"}}]
        rate = tool_arg_match_rate(
            calls, {"write_memory_tool": ["CI", "runner"]}
        )
        assert rate == pytest.approx(0.5)


class TestEntityMetrics:
    def test_perfect(self) -> None:
        predicted = [
            {"name": "pgvector", "type": "technology"},
            {"name": "PostgreSQL", "type": "technology"},
        ]
        expected = [
            {"name": "pgvector", "type": "technology"},
            {"name": "PostgreSQL", "type": "technology"},
        ]
        m = entity_metrics(predicted, expected)
        assert m["entity_precision"] == 1.0
        assert m["entity_recall"] == 1.0
        assert m["entity_f1"] == 1.0
        assert m["entity_type_accuracy"] == 1.0

    def test_containment_and_type_penalty(self) -> None:
        predicted = [
            {"name": "BGE-M3 模型", "type": "technology"},  # matches BGE-M3
            {"name": "pgvector", "type": "concept"},  # wrong type
            {"name": "虚构实体", "type": "technology"},  # fp
        ]
        expected = [
            {"name": "BGE-M3", "type": "technology"},
            {"name": "pgvector", "type": "technology"},
        ]
        m = entity_metrics(predicted, expected)
        assert m["entity_precision"] == pytest.approx(2 / 3)
        assert m["entity_recall"] == pytest.approx(1.0)
        assert m["entity_f1"] == pytest.approx(0.8)
        assert m["entity_type_accuracy"] == pytest.approx(0.5)

    def test_no_predicted(self) -> None:
        m = entity_metrics([], [{"name": "x", "type": "concept"}])
        assert m["entity_precision"] == 0.0
        assert m["entity_recall"] == 0.0
        assert m["entity_f1"] == 0.0

    def test_empty_expected_is_zero(self) -> None:
        m = entity_metrics([{"name": "x", "type": "concept"}], [])
        assert m["entity_f1"] == 0.0


class TestRelationMetrics:
    def test_perfect(self) -> None:
        predicted = [
            {"from": "连接池", "to": "502", "type": "causes"},
        ]
        expected = [
            {"from": "连接池", "to": "502", "type": "causes"},
        ]
        m = relation_metrics(predicted, expected)
        assert m["relation_f1"] == 1.0

    def test_wrong_type_does_not_match(self) -> None:
        predicted = [{"from": "连接池", "to": "502", "type": "relates_to"}]
        expected = [{"from": "连接池", "to": "502", "type": "causes"}]
        m = relation_metrics(predicted, expected)
        assert m["relation_precision"] == 0.0
        assert m["relation_recall"] == 0.0

    def test_partial(self) -> None:
        predicted = [
            {"from": "测试", "to": "发布流程", "type": "part_of"},
            {"from": "不存在", "to": "发布流程", "type": "part_of"},
        ]
        expected = [
            {"from": "测试", "to": "发布流程", "type": "part_of"},
            {"from": "构建", "to": "发布流程", "type": "part_of"},
        ]
        m = relation_metrics(predicted, expected)
        assert m["relation_precision"] == pytest.approx(0.5)
        assert m["relation_recall"] == pytest.approx(0.5)
        assert m["relation_f1"] == pytest.approx(0.5)


class TestSummaryKeywordCoverage:
    def test_all_present(self) -> None:
        assert summary_keyword_coverage("用 pgvector 做检索", ["pgvector"]) == 1.0

    def test_partial(self) -> None:
        assert summary_keyword_coverage("用 pgvector 做检索", ["pgvector", "cosine"]) == 0.5

    def test_empty_keywords(self) -> None:
        assert summary_keyword_coverage("anything", []) == 1.0


class TestAnswerMetrics:
    def test_deterministic_perfect(self) -> None:
        m = answer_deterministic_metrics(
            "我们选了 pgvector 而非 Elasticsearch", ["pgvector", "Elasticsearch"], []
        )
        assert m["fact_coverage"] == 1.0
        assert m["groundedness"] == 1.0

    def test_deterministic_prohibited_claim(self) -> None:
        m = answer_deterministic_metrics(
            "我们最终选用了 Elasticsearch", ["pgvector"], ["最终选用了 Elasticsearch"]
        )
        assert m["fact_coverage"] == 0.0
        assert m["groundedness"] == 0.0
        assert m["hallucination_rate"] == 1.0

    def test_judge_verdict(self) -> None:
        verdict = {
            "covered_facts": ["pgvector"],
            "grounded": True,
            "ungrounded_claims": [],
        }
        m = answer_judge_metrics(verdict, ["pgvector", "Elasticsearch"])
        assert m["fact_coverage"] == pytest.approx(0.5)
        assert m["groundedness"] == 1.0
        assert m["hallucination_rate"] == 0.0

    def test_judge_ungrounded(self) -> None:
        verdict = {
            "covered_facts": ["pgvector"],
            "grounded": False,
            "ungrounded_claims": ["我们用了 Qdrant"],
        }
        m = answer_judge_metrics(verdict, ["pgvector"])
        assert m["groundedness"] == 0.0
        assert m["hallucination_rate"] == 1.0
        assert m["ungrounded_claims"] == 1.0


class TestCitationPresence:
    def test_full_id_cited(self) -> None:
        assert citation_presence("选型是 pgvector（记忆 a1b2c3d4）", ["a1b2c3d4"]) == 1.0

    def test_short_tail_cited(self) -> None:
        # The model may cite only the 8-char tail of a longer id.
        assert citation_presence("（文档 a1b2c3d4）", ["mem-a1b2c3d4"]) == 1.0

    def test_short_head_cited(self) -> None:
        # The memory display exposes the 8-char HEAD of a long id (mid[:8] in
        # _format_memory_display); citing that short id is a real citation.
        long_id = "550e8400-e29b-41d4-a716-446655440000"
        assert citation_presence("（记忆 550e8400）", [long_id]) == 1.0

    def test_prefix_stripped(self) -> None:
        assert citation_presence("pgvector（a1b2c3d4）", ["mem-a1b2c3d4"]) == 1.0

    def test_no_citation(self) -> None:
        assert citation_presence("选型是 pgvector", ["a1b2c3d4"]) == 0.0

    def test_wrong_id_does_not_count(self) -> None:
        assert citation_presence("（记忆 e5f6a7b8）", ["a1b2c3d4"]) == 0.0

    def test_empty_source_ids_is_vacuously_passing(self) -> None:
        assert citation_presence("没有任何引用", []) == 1.0

    def test_empty_answer(self) -> None:
        assert citation_presence("", ["a1b2c3d4"]) == 0.0


class TestContextRecall:
    def test_all_facts_in_context(self) -> None:
        assert context_recall(
            ["pgvector", "Elasticsearch"], "选用 pgvector 而非 Elasticsearch"
        ) == 1.0

    def test_missing_fact_is_partial(self) -> None:
        assert context_recall(
            ["pgvector", "Elasticsearch"], "选用 pgvector"
        ) == pytest.approx(0.5)

    def test_no_facts_is_vacuously_passing(self) -> None:
        assert context_recall([], "任意上下文") == 1.0

    def test_empty_context(self) -> None:
        assert context_recall(["pgvector"], "") == 0.0

    def test_none_context_never_raises(self) -> None:
        assert context_recall(["pgvector"], None) == 0.0
