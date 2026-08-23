"""Tests for evals dataset consistency — llm_ground_truth + task_ground_truth.

The full labeled sets must pass validation (that is the CI zero-cost gate),
and the validators must catch the corruptions they exist to catch — a
wrong tool name, a relation endpoint that is not a golden entity, a bad
entity type — so a future edit that breaks a golden label fails loudly.
(The retrieval ground truth has its own validator suite in
``test_eval_dataset.py``.)
"""

from __future__ import annotations

import pytest

import evals.llm_ground_truth as gt
from evals.llm_ground_truth import (
    TOOL_SELECTION_CATEGORIES,
    TOOL_SELECTION_ITEMS,
    AnswerItem,
    E2EItem,
    ExtractionItem,
    ToolSelectionItem,
    validate_llm_dataset,
)
from evals.task_ground_truth import (
    TASK_ITEMS,
    validate_task_dataset,
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


# ── Task dataset validation ──────────────────────────────────────


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


def _item(**overrides) -> object:
    from dataclasses import replace

    base = TASK_ITEMS[0]
    return replace(base, **overrides)


def _validate(items: list) -> list[str]:
    """Validate a custom item list without touching the module global."""
    return validate_task_dataset(items=list(items))
