"""Tests for the memory write-path eval — metrics, runners, executors.

Fake-executor contract tests mirror ``evals/tests/test_runners.py``:

    - each suite executes every item and aggregates per-item metrics
    - by_category roll-ups cover every registered category
    - executor failures are recorded in errors AND count as zero rows
    - the merge suite's judge failure marks the row judge_error and leaves
      the judge keys unset (extraction-suite policy), while deterministic
      coverage still lands

The executors' injection points are also exercised directly: the gate
executor must use the *raising* core (``_llm_gate_verdict``), so a provider
outage surfaces as an execution error rather than a unanimous-allow score.
"""

from __future__ import annotations

import pytest

from evals.llm_ground_truth import (
    AutoGateItem,
    WriteConflictItem,
    WriteMergeItem,
)
from evals.write_eval_metrics import (
    conflict_cell_metrics,
    derive_binary_prf,
    gate_cell_metrics,
    merge_fact_coverage,
)
from evals.write_eval_runner import (
    AUTO_GATE_METRIC_KEYS,
    run_auto_gate,
    run_write_conflict,
    run_write_merge,
)

# ── Pure metric functions ────────────────────────────────────────────


class TestBinaryMetrics:
    def test_cell_indicators_true_positive(self):
        m = conflict_cell_metrics(True, True)
        assert m == {
            "conflict_tp": 1.0, "conflict_fp": 0.0,
            "conflict_fn": 0.0, "conflict_tn": 0.0,
        }

    def test_cell_indicators_false_positive(self):
        m = gate_cell_metrics(True, False)
        assert m["worthy_fp"] == 1.0
        assert m["worthy_tp"] == 0.0

    def test_derive_perfect_classification_with_negatives(self):
        """The regression case: mixed positives and negatives must derive to
        perfect micro-P/R — averaging per-item precision would not."""
        cells = {"conflict_tp": 0.5, "conflict_tn": 0.5,
                 "conflict_fp": 0.0, "conflict_fn": 0.0}
        d = derive_binary_prf(cells, "conflict")
        assert d["conflict_precision"] == 1.0
        assert d["conflict_recall"] == 1.0
        assert d["conflict_f1"] == 1.0
        assert d["conflict_accuracy"] == 1.0
        assert d["conflict_false_positive_rate"] == 0.0

    def test_derive_all_false_positive(self):
        cells = {"conflict_tp": 0.0, "conflict_tn": 0.0,
                 "conflict_fp": 1.0, "conflict_fn": 0.0}
        d = derive_binary_prf(cells, "conflict")
        assert d["conflict_precision"] == 0.0
        assert d["conflict_recall"] == 0.0  # degenerate: no positive in truth
        assert d["conflict_false_positive_rate"] == 1.0

    def test_derive_empty_bucket_all_zero(self):
        d = derive_binary_prf({}, "worthy")
        assert d["worthy_f1"] == 0.0

    def test_merge_fact_coverage(self):
        assert merge_fact_coverage("用了 A 和 B，成本是 C", ["A", "B", "C"]) == 1.0
        assert merge_fact_coverage("只提到 A", ["A", "B"]) == 0.5
        assert merge_fact_coverage("", ["A"]) == 0.0
        assert merge_fact_coverage("anything", []) == 1.0


# ── Runner contract tests (fake executors) ──────────────────────────


@pytest.fixture
def conflict_items() -> list[WriteConflictItem]:
    return [
        WriteConflictItem(
            id="wc-t1", existing_summary="旧摘要", new_summary="矛盾的新摘要",
            expected_conflict=True, category="direct_contradiction",
        ),
        WriteConflictItem(
            id="wc-t2", existing_summary="旧摘要", new_summary="补充性新摘要",
            expected_conflict=False, category="supplement",
        ),
    ]


@pytest.fixture
def merge_items() -> list[WriteMergeItem]:
    return [
        WriteMergeItem(
            id="wm-t1", existing_summary="方案用 pgvector",
            new_summary="理由是省运维成本",
            required_facts=["pgvector", "运维成本"],
            category="paraphrase",
        ),
    ]


@pytest.fixture
def gate_items() -> list[AutoGateItem]:
    return [
        AutoGateItem(
            id="ag-t1", content="决定把超时改成 90 秒",
            expected_worthy=True, category="technical_decision",
        ),
        AutoGateItem(
            id="ag-t2", content="今天天气不错",
            expected_worthy=False, category="chitchat",
        ),
    ]


class TestRunWriteConflict:
    @pytest.mark.asyncio
    async def test_scores_and_aggregates(self, conflict_items):
        async def detect(existing: str, new: str) -> bool:
            return True  # over-eager detector: flags everything

        result = await run_write_conflict(conflict_items, detect)
        assert result.n_items == 2
        assert len(result.errors) == 0
        # t1 correct, t2 false positive → recall perfect, precision halved.
        assert result.metric("conflict_recall") == 1.0
        assert result.metric("conflict_precision") == 0.5
        assert result.metric("conflict_false_positive_rate") == 0.5
        assert set(result.by_category) >= {"direct_contradiction", "supplement"}

    @pytest.mark.asyncio
    async def test_executor_failure_counts_as_zero_row(self, conflict_items):
        async def broken(existing: str, new: str) -> bool:
            raise RuntimeError("LLM down")

        result = await run_write_conflict(conflict_items, broken)
        assert len(result.errors) == 2
        assert result.metric("conflict_f1") == 0.0


class TestRunAutoGate:
    @pytest.mark.asyncio
    async def test_scores_binary_classification(self, gate_items):
        async def check(content: str) -> bool:
            return "决定" in content

        result = await run_auto_gate(gate_items, check)
        assert result.n_items == 2
        # One TP + one TN → perfect micro-P/R despite mixed labels.
        assert result.metric("worthy_recall") == 1.0
        assert result.metric("worthy_precision") == 1.0
        assert result.metric("worthy_accuracy") == 1.0
        assert result.overall.keys() >= {k for k in AUTO_GATE_METRIC_KEYS}

    @pytest.mark.asyncio
    async def test_outage_is_error_not_unanimous_allow(self, gate_items):
        """The raising-core contract: a provider outage must surface as
        execution errors, never as every item silently scoring 'worthy'."""
        async def broken(content: str) -> bool:
            raise RuntimeError("provider unreachable")

        result = await run_auto_gate(gate_items, broken)
        assert len(result.errors) == 2
        # All-fn cells (expected worthy, nothing predicted) → recall 0.
        assert result.metric("worthy_recall") == 0.0


class TestRunWriteMerge:
    @pytest.mark.asyncio
    async def test_deterministic_coverage_only(self, merge_items):
        async def merge(existing: str, new: str) -> str:
            return "合并后同时保留 pgvector 和运维成本两个要点"

        result = await run_write_merge(merge_items, merge, judge="deterministic")
        assert result.metric("merge_fact_coverage") == 1.0
        assert "merge_faithfulness" not in result.overall
        assert len(result.judge_errors) == 0

    @pytest.mark.asyncio
    async def test_judge_mode_records_verdict(self, merge_items, monkeypatch):
        async def merge(existing: str, new: str) -> str:
            return "pgvector 与运维成本"

        async def fake_judge(*args, **kwargs):
            return {"faithfulness": 0.9, "completeness": 0.8}

        import evals.llm_judge as judge_mod

        monkeypatch.setattr(judge_mod, "judge_merge", fake_judge)

        result = await run_write_merge(merge_items, merge, judge="llm")
        assert result.metric("merge_faithfulness") == 0.9
        assert result.metric("merge_completeness") == 0.8

    @pytest.mark.asyncio
    async def test_judge_failure_leaves_keys_unset(self, merge_items, monkeypatch):
        async def merge(existing: str, new: str) -> str:
            return "pgvector 与运维成本"

        async def broken_judge(*args, **kwargs):
            raise RuntimeError("judge down")

        import evals.llm_judge as judge_mod

        monkeypatch.setattr(judge_mod, "judge_merge", broken_judge)

        result = await run_write_merge(merge_items, merge, judge="llm")
        assert len(result.judge_errors) == 1
        assert len(result.errors) == 0  # execution succeeded
        # Judge keys unset — aggregated as 0.0 by the denominator policy,
        # but never written as a fabricated verdict.
        assert "merge_faithfulness" not in result.per_query[0]
        # Deterministic coverage unaffected.
        assert result.per_query[0]["merge_fact_coverage"] == 1.0


# ── Real executor wiring ─────────────────────────────────────────────


class TestExecutorsUseProductionPaths:
    @pytest.mark.asyncio
    async def test_conflict_executor_calls_detect_conflict(self, monkeypatch):
        from evals.write_eval_executors import make_conflict_detector

        captured = {}

        async def fake_detect(existing, new):
            captured["pair"] = (existing["summary"], new["summary"])
            return True

        import backend.service.memory as memory_mod

        monkeypatch.setattr(memory_mod, "_detect_conflict", fake_detect)
        detector = make_conflict_detector()
        assert await detector("旧", "新") is True
        assert captured["pair"] == ("旧", "新")

    @pytest.mark.asyncio
    async def test_gate_executor_uses_raising_core(self, monkeypatch):
        """The gate executor must NOT go through the fail-open wrapper."""
        import backend.agent.nodes as nodes_mod
        from evals.write_eval_executors import make_gate_checker

        async def raising_core(content: str) -> bool:
            raise RuntimeError("structured output exhausted")

        monkeypatch.setattr(nodes_mod, "_llm_gate_verdict", raising_core)
        checker = make_gate_checker()
        with pytest.raises(RuntimeError):
            await checker("some content")

    @pytest.mark.asyncio
    async def test_merge_executor_calls_shared_merge_summaries(self, monkeypatch):
        import backend.service.memory as memory_mod
        from evals.write_eval_executors import make_merge_summarizer

        async def fake_merge(existing: str, new: str) -> str:
            return f"{existing}+{new}"

        monkeypatch.setattr(memory_mod, "merge_summaries", fake_merge)
        summarizer = make_merge_summarizer()
        assert await summarizer("A", "B") == "A+B"
