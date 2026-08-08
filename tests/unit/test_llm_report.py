"""Unit tests for tests/eval/llm_report.py — report rendering.

Builds a fabricated ``LlmEvalResult`` and asserts the Markdown / JSON
reports contain the expected sections (overall table, category breakdown,
per-query detail, error sections) and that ``summarize`` prints a stable
one-liner per suite.
"""

from __future__ import annotations

import json

from tests.eval.llm_ground_truth import (
    ANSWER_CATEGORIES,
    EXTRACTION_CATEGORIES,
    TOOL_SELECTION_CATEGORIES,
)
from tests.eval.llm_report import (
    summarize,
    to_json,
    to_markdown,
)
from tests.eval.llm_runner import LlmEvalResult


def _fake_result(
    suite: str,
    keys: tuple[str, ...],
    categories: tuple[str, ...],
    *,
    judge: str = "llm",
    errors: list | None = None,
) -> LlmEvalResult:
    rows = [
        {"id": f"{suite}-1", "category": categories[0], **{k: 1.0 for k in keys}},
        {"id": f"{suite}-2", "category": categories[1], **{k: 0.0 for k in keys}},
    ]
    overall = {k: 0.5 for k in keys}
    by_category = {
        cat: ({k: 1.0 for k in keys} if cat == categories[0] else {k: 0.0 for k in keys})
        for cat in categories
    }
    return LlmEvalResult(
        suite=suite,
        judge=judge,
        per_query=rows,
        overall=overall,
        by_category=by_category,
        metric_keys=keys,
        n_items=len(rows),
        errors=errors or [],
        judge_errors=[{"id": f"{suite}-1", "error": "judge degraded"}] if judge == "llm" else [],
    )


class TestToMarkdown:
    def test_tool_selection_sections(self) -> None:
        r = _fake_result(
            "tool_selection",
            ("tool_accuracy", "expected_recall"),
            TOOL_SELECTION_CATEGORIES,
            judge="deterministic",
        )
        md = to_markdown([r])
        assert "# EMA LLM Behavior Evaluation Report" in md
        assert "工具选择" in md
        assert "tool_accuracy" in md
        assert "| memory_search |" in md  # category table row
        assert "Per-query detail" in md

    def test_extraction_judge_section(self) -> None:
        r = _fake_result(
            "extraction",
            ("entity_f1", "summary_faithfulness"),
            EXTRACTION_CATEGORIES,
            judge="llm",
        )
        md = to_markdown([r])
        assert "知识抽取" in md
        assert "judge degradations" in md
        assert "judge degraded" in md

    def test_answer_error_section(self) -> None:
        r = _fake_result(
            "answer",
            ("fact_coverage", "groundedness"),
            ANSWER_CATEGORIES,
            judge="deterministic",
            errors=[{"id": "answer-1", "error": "provider down"}],
        )
        md = to_markdown([r])
        assert "最终答案" in md
        assert "execution errors" in md
        assert "provider down" in md

    def test_answer_uncited_answers_are_flagged(self) -> None:
        """An answer that cites no source id is flagged for prompt iteration."""
        rows = [
            {
                "id": "ans-x",
                "category": ANSWER_CATEGORIES[0],
                "fact_coverage": 1.0,
                "groundedness": 1.0,
                "citation_rate": 0.0,
                "answer_len": 10,
                "answer_preview": "答案是 pgvector",
            }
        ]
        r = LlmEvalResult(
            suite="answer",
            judge="deterministic",
            per_query=rows,
            overall={"fact_coverage": 1.0, "groundedness": 1.0, "citation_rate": 0.0},
            by_category={ANSWER_CATEGORIES[0]: {"citation_rate": 0.0}},
            metric_keys=("fact_coverage", "groundedness", "citation_rate"),
            n_items=1,
        )
        md = to_markdown([r])
        assert "citation=0.000" in md
        assert "no source cited — candidate for prompt/eval iteration" in md

    def test_summarize_includes_citation_rate(self) -> None:
        r = _fake_result(
            "answer",
            ("fact_coverage", "groundedness", "citation_rate"),
            ANSWER_CATEGORIES,
            judge="deterministic",
        )
        line = summarize(r)
        assert line.startswith("[answer]")
        assert "citation=0.500" in line


class TestToJson:
    def test_round_trips_overall_and_rows(self) -> None:
        r = _fake_result(
            "answer",
            ("fact_coverage", "groundedness"),
            ANSWER_CATEGORIES,
            judge="deterministic",
        )
        payload = json.loads(to_json([r]))
        assert payload["results"][0]["suite"] == "answer"
        assert payload["results"][0]["overall"]["fact_coverage"] == 0.5
        assert len(payload["results"][0]["per_query"]) == 2
        assert "n_judge_errors" in payload["results"][0]


class TestSummarize:
    def test_one_line_per_suite(self) -> None:
        for suite, keys, cats in (
            ("tool_selection", ("tool_accuracy",), TOOL_SELECTION_CATEGORIES),
            ("extraction", ("entity_f1", "relation_f1"), EXTRACTION_CATEGORIES),
            ("answer", ("fact_coverage", "groundedness"), ANSWER_CATEGORIES),
        ):
            r = _fake_result(suite, keys, cats, judge="deterministic")
            line = summarize(r)
            assert line.startswith(f"[{suite}]")
            assert "errors=0" in line
