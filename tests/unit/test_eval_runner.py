"""Unit tests for tests/eval/runner.py — orchestration and aggregation.

Uses synthetic retriever adapters (no DB / no LLM) so the tests are fast and
deterministic. The contract being tested:
    - run_eval executes every query and aggregates per-query metrics
    - by_category / by_difficulty roll-ups are correct
    - retrieval errors are recorded, not fatal
    - category filter is honored
    - compare_eval preserves config order
    - result_to_dict round-trips all fields
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from tests.eval.dataset import RetrieverAdapter
from tests.eval.ground_truth import CATEGORIES, DIFFICULTIES, GroundTruthItem
from tests.eval.runner import (
    METRIC_KEYS,
    EvalConfig,
    compare_eval,
    config_from_dict,
    result_to_dict,
    run_eval,
)


# ── Fixtures ──────────────────────────────────────────────────


@pytest.fixture
def fake_items() -> list[GroundTruthItem]:
    return [
        GroundTruthItem(
            id="q1", query="alpha query", seed_ids=["s1"],
            relevant_fingerprints=["alpha"], category="技术决策",
            difficulty="easy",
        ),
        GroundTruthItem(
            id="q2", query="beta query", seed_ids=["s2"],
            relevant_fingerprints=["beta"], category="技术决策",
            difficulty="hard",
        ),
        GroundTruthItem(
            id="q3", query="gamma query", seed_ids=["s3"],
            relevant_fingerprints=["gamma"], category="故障复盘",
            difficulty="medium",
        ),
    ]


def _make_fake_adapter(
    match_field: str,
    results_per_query: dict[str, list[dict[str, Any]]],
    *,
    fail_on: set[str] | None = None,
) -> RetrieverAdapter:
    """Build a synthetic adapter returning canned results per query.

    ``fail_on`` is a set of query strings that should raise (to test error
    handling). Results are pre-ranked lists of dicts with ``match_field``.
    """

    async def fn(query: str, top_k: int) -> list[dict[str, Any]]:
        if fail_on and query in fail_on:
            raise RuntimeError(f"synthetic failure for {query!r}")
        return list(results_per_query.get(query, []))[:top_k]

    return RetrieverAdapter(name="fake", fn=fn, match_field=match_field)


def _patch_adapter(monkeypatch, adapter: RetrieverAdapter) -> None:
    """Make EvalConfig.adapter() return our synthetic adapter."""
    monkeypatch.setattr(
        "tests.eval.runner.build_adapter",
        lambda *a, **kw: adapter,
    )


# ── run_eval ──────────────────────────────────────────────────


class TestRunEval:
    @pytest.mark.asyncio
    async def test_perfect_retrieval(
        self, monkeypatch, fake_items
    ) -> None:
        # Every query gets its relevant item at position 1
        results_per_query = {
            "alpha query": [{"content": "alpha here"}, {"content": "noise"}],
            "beta query": [{"content": "beta here"}, {"content": "noise"}],
            "gamma query": [{"content": "gamma here"}, {"content": "noise"}],
        }
        adapter = _make_fake_adapter("content", results_per_query)
        _patch_adapter(monkeypatch, adapter)

        cfg = EvalConfig(name="perfect", retriever="chunk", top_k=5)
        result = await run_eval(cfg, fake_items)

        assert result.n_queries == 3
        assert len(result.errors) == 0
        assert result.metric("recall@5") == 1.0
        assert result.metric("mrr") == 1.0
        assert result.metric("ndcg@5") == 1.0
        assert result.metric("hit_rate@5") == 1.0

    @pytest.mark.asyncio
    async def test_partial_retrieval(
        self, monkeypatch, fake_items
    ) -> None:
        # q1 hits, q2 misses, q3 hits at position 2
        results_per_query = {
            "alpha query": [{"content": "alpha"}, {"content": "x"}],
            "beta query": [{"content": "x"}, {"content": "y"}],  # no beta
            "gamma query": [{"content": "x"}, {"content": "gamma"}],
        }
        adapter = _make_fake_adapter("content", results_per_query)
        _patch_adapter(monkeypatch, adapter)

        cfg = EvalConfig(name="partial", retriever="chunk", top_k=5)
        result = await run_eval(cfg, fake_items)

        # recall@5: q1=1.0, q2=0.0, q3=1.0 → mean 2/3
        assert result.metric("recall@5") == pytest.approx(2 / 3)
        # mrr: q1=1.0, q2=0.0, q3=0.5 → mean 0.5
        assert result.metric("mrr") == pytest.approx(0.5)
        # hit_rate@5: q1=1, q2=0, q3=1 → mean 2/3
        assert result.metric("hit_rate@5") == pytest.approx(2 / 3)
        assert len(result.per_query) == 3

    @pytest.mark.asyncio
    async def test_error_handling_records_and_continues(
        self, monkeypatch, fake_items
    ) -> None:
        # q2 raises; q1 and q3 succeed
        results_per_query = {
            "alpha query": [{"content": "alpha"}],
            "gamma query": [{"content": "gamma"}],
        }
        adapter = _make_fake_adapter(
            "content", results_per_query, fail_on={"beta query"}
        )
        _patch_adapter(monkeypatch, adapter)

        cfg = EvalConfig(name="errors", retriever="chunk", top_k=5)
        result = await run_eval(cfg, fake_items)

        assert len(result.errors) == 1
        assert result.errors[0]["id"] == "q2"
        assert "synthetic failure" in result.errors[0]["error"]
        # Failed queries still produce per_query rows (zero-recall), so they
        # count toward the aggregate denominator instead of inflating it.
        assert len(result.per_query) == 3
        q2_row = next(q for q in result.per_query if q["id"] == "q2")
        assert q2_row["error"] == "synthetic failure for 'beta query'"
        assert q2_row["recall@5"] == 0.0
        # Aggregates include the failed query: q1=1.0, q2=0.0, q3=1.0 → 2/3.
        # (Previously the failure was dropped, so recall read 1.0 while a
        # rising error rate made the numbers look *better*, not worse.)
        assert result.n_queries == 3
        assert result.metric("recall@5") == pytest.approx(2 / 3)
        assert result.metric("mrr") == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_reraise_propagates(
        self, monkeypatch, fake_items
    ) -> None:
        adapter = _make_fake_adapter(
            "content", {}, fail_on={"alpha query"}
        )
        _patch_adapter(monkeypatch, adapter)

        cfg = EvalConfig(name="reraise", retriever="chunk", top_k=5)
        with pytest.raises(RuntimeError, match="synthetic failure"):
            await run_eval(cfg, fake_items, reraise=True)

    @pytest.mark.asyncio
    async def test_category_filter(self, monkeypatch, fake_items) -> None:
        results_per_query = {
            "alpha query": [{"content": "alpha"}],
            "beta query": [{"content": "beta"}],
            # gamma query never called because it's filtered out
        }
        adapter = _make_fake_adapter("content", results_per_query)
        _patch_adapter(monkeypatch, adapter)

        cfg = EvalConfig(
            name="filtered", retriever="chunk", top_k=5,
            categories=["技术决策"],
        )
        result = await run_eval(cfg, fake_items)

        # Only q1 and q2 should be in per_query (both are 技术决策)
        ids = [r["id"] for r in result.per_query]
        assert set(ids) == {"q1", "q2"}
        assert result.n_queries == 2

    @pytest.mark.asyncio
    async def test_semantic_relevance_supplements_substring(
        self, monkeypatch, fake_items
    ) -> None:
        """With semantic_relevance=True, a result that misses the fingerprint
        but is semantically close still counts as relevant (OR combination)."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        import tests.eval.runner as runner_mod

        # q1's retrieved item paraphrases the target — no substring match.
        results_per_query = {
            "alpha query": [{"content": "a paraphrase that omits the keyword"}],
            "beta query": [{"content": "beta"}],
            "gamma query": [{"content": "gamma"}],
        }
        adapter = _make_fake_adapter("content", results_per_query)
        _patch_adapter(monkeypatch, adapter)

        # The fake labeled set's seed_ids (s1) are not in the real seed file,
        # so supply a matching seed summary for the semantic pass to target.
        monkeypatch.setattr(
            runner_mod,
            "load_seed_memories",
            lambda: [SimpleNamespace(id="s1", summary="alpha summary")],
        )
        # The semantic channel marks q1's single result relevant.
        monkeypatch.setattr(
            runner_mod,
            "semantic_relevance_mask",
            AsyncMock(return_value=[True]),
        )

        cfg = EvalConfig(name="semantic", retriever="chunk", top_k=5)
        result = await run_eval(cfg, fake_items, semantic_relevance=True)

        # q1 now counts as a hit even though the fingerprint missed.
        assert result.metric("recall@5") == 1.0
        assert result.metric("mrr") == 1.0
        q1 = next(q for q in result.per_query if q["id"] == "q1")
        assert q1["semantic_relevance"] is True

    @pytest.mark.asyncio
    async def test_semantic_relevance_off_by_default(
        self, monkeypatch, fake_items
    ) -> None:
        """The semantic channel is OFF by default: without an explicit flag,
        relevance is pure substring matching and the embedding channel is
        never consulted (the self-scored channel is opt-in)."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        import tests.eval.runner as runner_mod

        # q1's retrieved item paraphrases the target — no substring match.
        results_per_query = {
            "alpha query": [{"content": "a paraphrase that omits the keyword"}],
            "beta query": [{"content": "beta"}],
            "gamma query": [{"content": "gamma"}],
        }
        adapter = _make_fake_adapter("content", results_per_query)
        _patch_adapter(monkeypatch, adapter)

        monkeypatch.setattr(
            runner_mod,
            "load_seed_memories",
            lambda: [SimpleNamespace(id="s1", summary="alpha summary")],
        )
        monkeypatch.setattr(
            runner_mod,
            "semantic_relevance_mask",
            AsyncMock(return_value=[True]),
        )

        # No explicit flag — default behaviour must NOT invoke the channel.
        cfg = EvalConfig(name="default-off", retriever="chunk", top_k=5)
        result = await run_eval(cfg, fake_items)

        # q1 missed (fingerprint absent) and the channel was not consulted:
        # recall is 2/3, matching the deterministic substring baseline.
        assert result.metric("recall@5") == pytest.approx(2 / 3)
        q1 = next(q for q in result.per_query if q["id"] == "q1")
        assert q1["semantic_relevance"] is False
        runner_mod.semantic_relevance_mask.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_semantic_relevance_explicitly_enabled(
        self, monkeypatch, fake_items
    ) -> None:
        """Opting in via semantic_relevance=True invokes the embedding channel
        and adds the rescues it finds."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        import tests.eval.runner as runner_mod

        # q1's retrieved item paraphrases the target — no substring match.
        results_per_query = {
            "alpha query": [{"content": "a paraphrase that omits the keyword"}],
            "beta query": [{"content": "beta"}],
            "gamma query": [{"content": "gamma"}],
        }
        adapter = _make_fake_adapter("content", results_per_query)
        _patch_adapter(monkeypatch, adapter)

        monkeypatch.setattr(
            runner_mod,
            "load_seed_memories",
            lambda: [SimpleNamespace(id="s1", summary="alpha summary")],
        )
        monkeypatch.setattr(
            runner_mod,
            "semantic_relevance_mask",
            AsyncMock(return_value=[True]),
        )

        cfg = EvalConfig(name="semantic", retriever="chunk", top_k=5)
        result = await run_eval(cfg, fake_items, semantic_relevance=True)

        assert result.metric("recall@5") == 1.0
        q1 = next(q for q in result.per_query if q["id"] == "q1")
        assert q1["semantic_relevance"] is True
        runner_mod.semantic_relevance_mask.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_semantic_relevance_can_be_disabled(
        self, monkeypatch, fake_items
    ) -> None:
        """Explicitly disabling the channel keeps the lexical baseline: a
        fingerprint miss stays a miss even when the embedding channel would
        have rescued it."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        import tests.eval.runner as runner_mod

        results_per_query = {
            "alpha query": [{"content": "a paraphrase that omits the keyword"}],
            "beta query": [{"content": "beta"}],
            "gamma query": [{"content": "gamma"}],
        }
        adapter = _make_fake_adapter("content", results_per_query)
        _patch_adapter(monkeypatch, adapter)

        monkeypatch.setattr(
            runner_mod,
            "load_seed_memories",
            lambda: [SimpleNamespace(id="s1", summary="alpha summary")],
        )
        monkeypatch.setattr(
            runner_mod,
            "semantic_relevance_mask",
            AsyncMock(return_value=[True]),
        )

        cfg = EvalConfig(name="baseline", retriever="chunk", top_k=5)
        result = await run_eval(cfg, fake_items, semantic_relevance=False)

        # q1 missed (fingerprint absent) and the semantic channel was not
        # consulted — recall is 2/3, matching the substring-only baseline.
        assert result.metric("recall@5") == pytest.approx(2 / 3)
        runner_mod.semantic_relevance_mask.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_substring_vs_semantic_hits_split(
        self, monkeypatch, fake_items
    ) -> None:
        """per-query rows distinguish lexical hits from semantic-only hits."""
        from types import SimpleNamespace
        from unittest.mock import AsyncMock

        import tests.eval.runner as runner_mod

        # q1: paraphrase — substring miss, semantic channel rescues it.
        # q2: exact term — substring hit, no semantic contribution needed.
        # q3: unrelated — substring miss and nothing semantically close.
        results_per_query = {
            "alpha query": [{"content": "paraphrase without the keyword"}],
            "beta query": [{"content": "beta"}],
            "gamma query": [{"content": "unrelated noise"}],
        }
        adapter = _make_fake_adapter("content", results_per_query)
        _patch_adapter(monkeypatch, adapter)

        monkeypatch.setattr(
            runner_mod,
            "load_seed_memories",
            lambda: [SimpleNamespace(id="s1", summary="alpha summary")],
        )
        # Semantic mask marks only the q1 result relevant.
        monkeypatch.setattr(
            runner_mod,
            "semantic_relevance_mask",
            AsyncMock(return_value=[True]),
        )

        cfg = EvalConfig(name="split", retriever="chunk", top_k=5)
        result = await run_eval(cfg, fake_items, semantic_relevance=True)

        q1 = next(q for q in result.per_query if q["id"] == "q1")
        assert q1["substring_hits"] == 0
        assert q1["semantic_only_hits"] == 1
        q2 = next(q for q in result.per_query if q["id"] == "q2")
        assert q2["substring_hits"] == 1
        assert q2["semantic_only_hits"] == 0
        q3 = next(q for q in result.per_query if q["id"] == "q3")
        assert q3["substring_hits"] == 0
        assert q3["semantic_only_hits"] == 0


# ── Aggregation ───────────────────────────────────────────────


class TestAggregation:
    @pytest.mark.asyncio
    async def test_by_category(
        self, monkeypatch, fake_items
    ) -> None:
        # q1 (技术决策) hits; q2 (技术决策) misses; q3 (故障复盘) hits
        results_per_query = {
            "alpha query": [{"content": "alpha"}],
            "beta query": [{"content": "noise"}],
            "gamma query": [{"content": "gamma"}],
        }
        adapter = _make_fake_adapter("content", results_per_query)
        _patch_adapter(monkeypatch, adapter)

        cfg = EvalConfig(name="agg", retriever="chunk", top_k=5)
        result = await run_eval(cfg, fake_items)

        assert "技术决策" in result.by_category
        assert "故障复盘" in result.by_category
        # 技术决策: q1=1.0, q2=0.0 → mean 0.5
        assert result.by_category["技术决策"]["recall@5"] == pytest.approx(0.5)
        # 故障复盘: q3=1.0
        assert result.by_category["故障复盘"]["recall@5"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_by_difficulty(
        self, monkeypatch, fake_items
    ) -> None:
        # q1 easy hits, q2 hard misses, q3 medium hits
        results_per_query = {
            "alpha query": [{"content": "alpha"}],
            "beta query": [{"content": "noise"}],
            "gamma query": [{"content": "gamma"}],
        }
        adapter = _make_fake_adapter("content", results_per_query)
        _patch_adapter(monkeypatch, adapter)

        cfg = EvalConfig(name="diff", retriever="chunk", top_k=5)
        result = await run_eval(cfg, fake_items)

        assert result.by_difficulty["easy"]["recall@5"] == 1.0
        assert result.by_difficulty["hard"]["recall@5"] == 0.0
        assert result.by_difficulty["medium"]["recall@5"] == 1.0

    @pytest.mark.asyncio
    async def test_overall_has_all_metric_keys(
        self, monkeypatch, fake_items
    ) -> None:
        adapter = _make_fake_adapter("content", {})
        _patch_adapter(monkeypatch, adapter)

        cfg = EvalConfig(name="keys", retriever="chunk", top_k=5)
        result = await run_eval(cfg, fake_items)

        for k in METRIC_KEYS:
            assert k in result.overall, f"missing overall metric: {k}"
        # Auxiliary keys
        assert "latency_ms" in result.overall
        assert "n_retrieved" in result.overall
        assert "n_relevant" in result.overall

    @pytest.mark.asyncio
    async def test_per_query_row_shape(
        self, monkeypatch, fake_items
    ) -> None:
        results_per_query = {"alpha query": [{"content": "alpha"}]}
        adapter = _make_fake_adapter("content", results_per_query)
        _patch_adapter(monkeypatch, adapter)

        cfg = EvalConfig(name="shape", retriever="chunk", top_k=5)
        result = await run_eval(cfg, fake_items)

        row = result.per_query[0]
        for key in ("id", "query", "category", "difficulty", "n_retrieved",
                    "n_relevant", "latency_ms"):
            assert key in row, f"per_query missing: {key}"
        for k in METRIC_KEYS:
            assert k in row

    @pytest.mark.asyncio
    async def test_empty_buckets_present_as_zeros(
        self, monkeypatch, fake_items
    ) -> None:
        # fake_items only covers 技术决策 + 故障复盘; the other three
        # categories must still appear (stable display contract) and every
        # difficulty bucket must be present, with all-zero metrics for
        # buckets that have no queries.
        adapter = _make_fake_adapter("content", {})
        _patch_adapter(monkeypatch, adapter)

        cfg = EvalConfig(name="buckets", retriever="chunk", top_k=5)
        result = await run_eval(cfg, fake_items)

        assert set(result.by_category.keys()) == set(CATEGORIES)
        for empty_cat in ("架构设计", "代码实现", "历史背景"):
            assert result.by_category[empty_cat]["recall@5"] == 0.0
            assert result.by_category[empty_cat]["mrr"] == 0.0

        assert set(result.by_difficulty.keys()) == set(DIFFICULTIES)


# ── compare_eval ──────────────────────────────────────────────


class TestCompareEval:
    @pytest.mark.asyncio
    async def test_preserves_order(
        self, monkeypatch, fake_items
    ) -> None:
        # Two adapters with different results — use a single monkeypatch
        # that returns different adapters per call count.
        call_count = {"n": 0}
        adapters = [
            _make_fake_adapter("content", {
                "alpha query": [{"content": "alpha"}],
            }),
            _make_fake_adapter("content", {
                "alpha query": [{"content": "noise"}],  # misses
            }),
        ]

        def fake_build(*a, **kw):
            a = adapters[call_count["n"]]
            call_count["n"] += 1
            return a

        monkeypatch.setattr("tests.eval.runner.build_adapter", fake_build)

        cfg_a = EvalConfig(name="A", retriever="chunk", top_k=5)
        cfg_b = EvalConfig(name="B", retriever="chunk", top_k=5)
        results = await compare_eval([cfg_a, cfg_b], fake_items)

        assert len(results) == 2
        assert results[0].config.name == "A"
        assert results[1].config.name == "B"
        # A hits alpha → recall mean over 3 queries = 1/3
        assert results[0].metric("recall@5") == pytest.approx(1 / 3)
        # B misses everything
        assert results[1].metric("recall@5") == 0.0

    @pytest.mark.asyncio
    async def test_reraise_propagates(
        self, monkeypatch, fake_items
    ) -> None:
        # reraise=True should propagate the first retrieval error from any
        # config in the comparison, instead of recording it.
        adapter = _make_fake_adapter(
            "content", {}, fail_on={"alpha query"}
        )
        monkeypatch.setattr(
            "tests.eval.runner.build_adapter", lambda *a, **kw: adapter
        )

        cfg = EvalConfig(name="boom", retriever="chunk", top_k=5)
        with pytest.raises(RuntimeError, match="synthetic failure"):
            await compare_eval([cfg], fake_items, reraise=True)


# ── Serialization ─────────────────────────────────────────────


class TestSerialization:
    @pytest.mark.asyncio
    async def test_result_to_dict_roundtrip(
        self, monkeypatch, fake_items
    ) -> None:
        adapter = _make_fake_adapter("content", {
            "alpha query": [{"content": "alpha"}],
        })
        _patch_adapter(monkeypatch, adapter)

        cfg = EvalConfig(name="rt", retriever="chunk", top_k=5)
        result = await run_eval(cfg, fake_items)
        d = result_to_dict(result)

        for key in ("config", "n_queries", "total_latency_ms", "overall",
                    "by_category", "by_difficulty", "per_query", "errors"):
            assert key in d
        assert d["n_queries"] == 3
        assert d["config"]["name"] == "rt"
        assert isinstance(d["per_query"], list)

    def test_config_from_dict(self) -> None:
        d = {
            "name": "from_json",
            "retriever": "memory",
            "top_k": 10,
            "use_llm_rerank": True,
            "threshold": 0.4,
            "categories": ["技术决策"],
        }
        cfg = config_from_dict(d)
        assert cfg.name == "from_json"
        assert cfg.retriever == "memory"
        assert cfg.top_k == 10
        assert cfg.use_llm_rerank is True
        assert cfg.threshold == 0.4
        assert cfg.categories == ["技术决策"]

    def test_config_from_dict_defaults(self) -> None:
        cfg = config_from_dict({"name": "minimal"})
        assert cfg.retriever == "memory"
        assert cfg.top_k == 5
        assert cfg.use_llm_rerank is False
        assert cfg.threshold is None
        assert cfg.categories is None

    def test_label_property(self) -> None:
        cfg = EvalConfig(name="x", retriever="memory", top_k=10, use_llm_rerank=True)
        assert cfg.label == "memory:llm@k10"
        cfg2 = EvalConfig(name="y", retriever="chunk", top_k=3)
        assert cfg2.label == "chunk:norank@k3"
        cfg3 = EvalConfig(name="z", retriever="memory", top_k=5, use_cross_encoder=True)
        assert cfg3.label == "memory:ce@k5"
