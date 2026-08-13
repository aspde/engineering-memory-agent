"""Unit tests for tests/eval/report.py — Markdown + JSON rendering.

Verifies the report contains the expected sections and that A/B delta tables
render when ≥2 results are passed. Uses synthetic EvalResult objects built
without running the runner.
"""

from __future__ import annotations

import json

from tests.eval.report import (
    summarize,
    to_json,
    to_markdown,
    write_json,
    write_markdown,
)
from tests.eval.runner import METRIC_KEYS, EvalConfig, EvalResult


def _make_result(name: str, recall: float = 0.7, mrr: float = 0.6) -> EvalResult:
    """Build a minimal EvalResult with synthetic aggregates."""
    cfg = EvalConfig(name=name, retriever="memory", top_k=5)
    r = EvalResult(config=cfg, n_queries=10)
    r.overall = {k: recall if k == "recall@5" else mrr if k == "mrr" else 0.5
                 for k in METRIC_KEYS}
    r.overall["latency_ms"] = 123.0
    r.overall["n_retrieved"] = 5.0
    r.overall["n_relevant"] = 1.0
    r.by_category = {"技术决策": dict(r.overall), "故障复盘": dict(r.overall)}
    r.by_difficulty = {"easy": dict(r.overall), "medium": dict(r.overall), "hard": dict(r.overall)}
    r.per_query = [{
        "id": "q1", "query": "test", "category": "技术决策", "difficulty": "easy",
        "n_retrieved": 5, "n_relevant": 1, "latency_ms": 100.0,
        "semantic_relevance": True,
        "substring_hits": 1, "semantic_only_hits": 0,
        **{k: 0.5 for k in METRIC_KEYS},
    }]
    return r


class TestMarkdownReport:
    def test_single_result_has_required_sections(self) -> None:
        md = to_markdown([_make_result("solo")])
        assert "# EMA Retrieval Evaluation Report" in md
        assert "## Overall" in md
        assert "## Recall@5 by category" in md
        assert "## MRR by category" in md
        assert "## Recall@5 by difficulty" in md
        assert "solo" in md

    def test_ab_comparison_section_appears_with_two_results(self) -> None:
        md = to_markdown([_make_result("A", recall=0.6), _make_result("B", recall=0.8)])
        assert "## A/B comparison" in md
        assert "Δ B − A" in md
        # delta should show +0.200 for recall@5
        assert "+0.200" in md

    def test_no_ab_section_with_single_result(self) -> None:
        md = to_markdown([_make_result("only")])
        assert "## A/B comparison" not in md

    def test_per_query_detail_collapsible(self) -> None:
        md = to_markdown([_make_result("detail")])
        assert "<details>" in md
        assert "Per-query detail" in md

    def test_empty_results_does_not_crash(self) -> None:
        md = to_markdown([])
        assert "EMA Retrieval Evaluation Report" in md

    def test_report_semantic_channel_state_in_header(self) -> None:
        md = to_markdown([_make_result("sem")])
        assert "Semantic relevance: enabled" in md

    def test_report_semantic_rescued_column(self) -> None:
        r = _make_result("rescued")
        r.per_query = [
            # lexical miss, semantic hit → rescued (counts toward the column)
            {"id": "q1", "query": "t1", "category": "技术决策", "difficulty": "hard",
             "n_retrieved": 3, "n_relevant": 1, "latency_ms": 10.0,
             "semantic_relevance": True,
             "substring_hits": 0, "semantic_only_hits": 1,
             **{k: 0.0 for k in METRIC_KEYS}},
            # lexical hit → not rescued
            {"id": "q2", "query": "t2", "category": "技术决策", "difficulty": "easy",
             "n_retrieved": 3, "n_relevant": 1, "latency_ms": 10.0,
             "semantic_relevance": True,
             "substring_hits": 1, "semantic_only_hits": 0,
             **{k: 1.0 for k in METRIC_KEYS}},
        ]
        md = to_markdown([r])
        # overall table exposes the rescued-query count
        assert "semantic_rescued" in md
        row = next(line for line in md.splitlines() if line.startswith("| rescued |"))
        assert "| 1 |" in row
        # per-query markers: ✓ (rescued), — (lexical hit)
        assert "| ✓ |" in md
        assert "| — |" in md


class TestJsonReport:
    def test_valid_json(self) -> None:
        s = to_json([_make_result("j1")])
        data = json.loads(s)
        assert "generated_at" in data
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["config"]["name"] == "j1"

    def test_results_array_order_preserved(self) -> None:
        s = to_json([_make_result("first"), _make_result("second")])
        data = json.loads(s)
        assert data["results"][0]["config"]["name"] == "first"
        assert data["results"][1]["config"]["name"] == "second"


class TestWriteFiles:
    def test_write_markdown(self, tmp_path) -> None:
        path = tmp_path / "report.md"
        returned = write_markdown([_make_result("w")], str(path))
        assert returned == str(path)
        assert path.exists()
        assert "EMA Retrieval Evaluation Report" in path.read_text(encoding="utf-8")

    def test_write_json(self, tmp_path) -> None:
        path = tmp_path / "report.json"
        returned = write_json([_make_result("w")], str(path))
        assert returned == str(path)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["results"][0]["config"]["name"] == "w"


class TestSummarize:
    def test_one_line_summary(self) -> None:
        r = _make_result("sum", recall=0.75, mrr=0.65)
        line = summarize(r)
        assert "[sum]" in line
        assert "recall@5=0.750" in line
        assert "mrr=0.650" in line
        assert "queries=10" in line
