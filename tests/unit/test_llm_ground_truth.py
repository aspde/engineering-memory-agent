"""Unit tests for tests/eval/llm_ground_truth.py — dataset consistency.

The full labeled sets must pass validation (that is the CI zero-cost gate),
and the validators must catch the corruptions they exist to catch — a
wrong tool name, a relation endpoint that is not a golden entity, a bad
entity type — so a future edit that breaks a golden label fails loudly.
"""

from __future__ import annotations

import pytest

import tests.eval.llm_ground_truth as gt
from tests.eval.llm_ground_truth import (
    TOOL_SELECTION_CATEGORIES,
    TOOL_SELECTION_ITEMS,
    AnswerItem,
    E2EItem,
    ExtractionItem,
    ToolSelectionItem,
    validate_llm_dataset,
)


class TestBuiltinDatasets:
    def test_validate_passes_on_builtin_sets(self) -> None:
        # The built-in labeled sets must always be clean — CI's --validate-only
        # gate fails otherwise.
        assert validate_llm_dataset() == []

    def test_expected_tools_cover_all_suite_categories(self) -> None:
        categories = {it.category for it in TOOL_SELECTION_ITEMS}
        # Every registered category must have at least one item, or per-category
        # aggregates silently read all-zeros.
        assert set(TOOL_SELECTION_CATEGORIES).issubset(categories)


class TestToolSelectionValidation:
    def test_unknown_tool_raises(self, monkeypatch) -> None:
        bad = ToolSelectionItem(
            id="x1", query="q", expected_tools=["no_such_tool"],
            category="memory_search",
        )
        monkeypatch.setattr(gt, "TOOL_SELECTION_ITEMS", [bad])
        with pytest.raises(ValueError, match="no_such_tool"):
            validate_llm_dataset()

    def test_duplicate_id_raises(self, monkeypatch) -> None:
        a = ToolSelectionItem(
            id="x1", query="q", expected_tools=["search_memories_tool"],
            category="memory_search",
        )
        b = ToolSelectionItem(
            id="x1", query="q2", expected_tools=["search_memories_tool"],
            category="memory_search",
        )
        monkeypatch.setattr(gt, "TOOL_SELECTION_ITEMS", [a, b])
        with pytest.raises(ValueError, match="duplicate"):
            validate_llm_dataset()

    def test_empty_query_raises(self, monkeypatch) -> None:
        bad = ToolSelectionItem(
            id="x1", query="  ", expected_tools=["search_memories_tool"],
            category="memory_search",
        )
        monkeypatch.setattr(gt, "TOOL_SELECTION_ITEMS", [bad])
        with pytest.raises(ValueError, match="empty query"):
            validate_llm_dataset()


class TestExtractionValidation:
    def test_relation_endpoint_must_be_golden_entity(self, monkeypatch) -> None:
        """A relation whose endpoint is not in the golden entities can never
        match (extraction filters such relations out) — validation must catch it."""
        bad = ExtractionItem(
            id="e1",
            content="text",
            expected_entities=[{"name": "pgvector", "type": "technology"}],
            expected_relations=[
                {"from": "pgvector", "to": "Elasticsearch", "type": "relates_to"}
            ],
            category="code_decision",
            summary_keywords=["pgvector"],
        )
        monkeypatch.setattr(gt, "EXTRACTION_ITEMS", [bad])
        with pytest.raises(ValueError, match="endpoint"):
            validate_llm_dataset()

    def test_invalid_entity_type_raises(self, monkeypatch) -> None:
        bad = ExtractionItem(
            id="e1",
            content="text",
            expected_entities=[{"name": "x", "type": "not_a_type"}],
            expected_relations=[],
            category="code_decision",
            summary_keywords=["x"],
        )
        monkeypatch.setattr(gt, "EXTRACTION_ITEMS", [bad])
        with pytest.raises(ValueError, match="invalid type"):
            validate_llm_dataset()

    def test_empty_entities_raises(self, monkeypatch) -> None:
        bad = ExtractionItem(
            id="e1", content="text", expected_entities=[], expected_relations=[],
            category="code_decision", summary_keywords=["x"],
        )
        monkeypatch.setattr(gt, "EXTRACTION_ITEMS", [bad])
        with pytest.raises(ValueError, match="empty"):
            validate_llm_dataset()


class TestAnswerValidation:
    def test_empty_required_facts_raises(self, monkeypatch) -> None:
        bad = AnswerItem(
            id="a1", query="q", context="c", required_facts=[], category="factual"
        )
        monkeypatch.setattr(gt, "ANSWER_ITEMS", [bad])
        with pytest.raises(ValueError, match="required_facts"):
            validate_llm_dataset()


class TestE2EValidation:
    def test_required_fact_not_in_source_content_raises(self, monkeypatch) -> None:
        # A fact absent from source_content can never be retrieved, so
        # context_recall can never reach 1.0 — a hard label error.
        bad = E2EItem(
            id="x1",
            query="选型是什么",
            source_content="用 pgvector 做向量检索",
            required_facts=["pgvector", "Elasticsearch"],
            category="factual",
        )
        monkeypatch.setattr(gt, "E2E_ITEMS", [bad])
        with pytest.raises(ValueError, match="not a substring"):
            validate_llm_dataset()

    def test_duplicate_id_raises(self, monkeypatch) -> None:
        item = E2EItem(
            id="x1",
            query="q",
            source_content="用 pgvector 做向量检索",
            required_facts=["pgvector"],
            category="factual",
        )
        monkeypatch.setattr(gt, "E2E_ITEMS", [item, item])
        with pytest.raises(ValueError, match="duplicate e2e item id"):
            validate_llm_dataset()

    def test_unknown_retrieval_mode_raises(self, monkeypatch) -> None:
        bad = E2EItem(
            id="x1",
            query="q",
            source_content="用 pgvector 做向量检索",
            required_facts=["pgvector"],
            category="factual",
            retrieval_mode="hybrid",
        )
        monkeypatch.setattr(gt, "E2E_ITEMS", [bad])
        with pytest.raises(ValueError, match="retrieval_mode"):
            validate_llm_dataset()

    def test_invalid_category_raises(self, monkeypatch) -> None:
        bad = E2EItem(
            id="x1",
            query="q",
            source_content="用 pgvector 做向量检索",
            required_facts=["pgvector"],
            category="bogus",
        )
        monkeypatch.setattr(gt, "E2E_ITEMS", [bad])
        with pytest.raises(ValueError, match="unknown category"):
            validate_llm_dataset()
