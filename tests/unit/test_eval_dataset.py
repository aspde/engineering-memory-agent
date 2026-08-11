"""Unit tests for tests/eval/dataset.py — loading, validation, matching.

Covers:
    - Seed corpus loads from JSONL and fields parse correctly
    - Ground truth ↔ seed corpus consistency (the real ``seed_memories.jsonl``)
    - Fingerprint uniqueness and existence checks (with a synthetic bad corpus)
    - ``is_relevant`` / ``relevance_mask`` matching behavior
    - ``build_adapter`` wiring (without hitting the DB)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.eval.dataset import (
    SEED_FILE,
    RetrieverAdapter,
    SeedMemory,
    build_adapter,
    is_relevant,
    load_ground_truth,
    load_seed_memories,
    relevance_mask,
    validate_dataset,
)
from tests.eval.ground_truth import CATEGORIES, GROUND_TRUTH, GroundTruthItem, assert_complete


# ── Ground truth structural invariants ────────────────────────


class TestGroundTruthStructure:
    def test_assert_complete_passes(self) -> None:
        # Should not raise — catches duplicate IDs, empty categories, etc.
        assert_complete()

    def test_has_70_queries(self) -> None:
        assert len(GROUND_TRUTH) == 70

    def test_five_categories_fourteen_each(self) -> None:
        from collections import Counter

        counts = Counter(it.category for it in GROUND_TRUTH)
        assert set(counts) == set(CATEGORIES)
        for cat in CATEGORIES:
            assert counts[cat] == 14, f"{cat}: expected 14, got {counts[cat]}"

    def test_every_item_has_fingerprint_and_seed(self) -> None:
        for it in GROUND_TRUTH:
            assert it.relevant_fingerprints, f"{it.id}: no fingerprints"
            assert it.seed_ids, f"{it.id}: no seed_ids"

    def test_difficulty_distribution(self) -> None:
        from tests.eval.ground_truth import difficulty_distribution

        dist = difficulty_distribution(GROUND_TRUTH)
        assert dist["easy"] + dist["medium"] + dist["hard"] == 70
        # At least one of each — otherwise per-difficulty breakdown is useless
        for d in ("easy", "medium", "hard"):
            assert dist[d] >= 1, f"no {d} queries"

    def test_ids_unique(self) -> None:
        ids = [it.id for it in GROUND_TRUTH]
        assert len(set(ids)) == len(ids)


# ── Seed corpus loading ───────────────────────────────────────


class TestSeedLoading:
    def test_seed_file_exists(self) -> None:
        assert SEED_FILE.exists(), f"missing seed file: {SEED_FILE}"

    def test_loads_70_seeds(self) -> None:
        seeds = load_seed_memories()
        assert len(seeds) == 70

    def test_seed_ids_match_ground_truth(self) -> None:
        seeds = load_seed_memories()
        seed_ids = {s.id for s in seeds}
        for it in GROUND_TRUTH:
            for sid in it.seed_ids:
                assert sid in seed_ids, f"{it.id}: seed_id {sid} not in seed file"

    def test_every_seed_has_summary_and_content(self) -> None:
        for s in load_seed_memories():
            assert s.summary, f"{s.id}: empty summary"
            assert s.content, f"{s.id}: empty content"

    def test_seed_categories_match_ground_truth(self) -> None:
        seeds = load_seed_memories()
        seed_by_id = {s.id: s for s in seeds}
        for it in GROUND_TRUTH:
            for sid in it.seed_ids:
                seed = seed_by_id[sid]
                assert seed.category == it.category, (
                    f"{it.id} category={it.category} but seed {sid} category={seed.category}"
                )

    def test_malformed_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text('{"id": "x", "summary": "ok"}\n{not json}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="invalid seed entry"):
            load_seed_memories(bad)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_seed_memories(tmp_path / "nope.jsonl")

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.jsonl"
        empty.write_text("\n\n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_seed_memories(empty)


# ── Dataset validation (real corpus) ──────────────────────────


class TestValidateDatasetReal:
    def test_real_corpus_is_clean(self) -> None:
        """The shipped seed_memories.jsonl must pass validation cleanly.

        This is the single most important invariant: if this test fails, the
        eval pipeline cannot produce trustworthy numbers.
        """
        warnings = validate_dataset()
        assert warnings == [], "dataset validation warnings:\n" + "\n".join(warnings)

    def test_every_fingerprint_in_claimed_seed(self) -> None:
        """Each fingerprint must appear in the summary OR content of every
        seed it claims — summary for the memory retriever path, content
        for the chunk/hybrid/rewrite paths."""
        seeds = load_seed_memories()
        seed_by_id = {s.id: s for s in seeds}
        for it in GROUND_TRUTH:
            for fp in it.relevant_fingerprints:
                for sid in it.seed_ids:
                    seed = seed_by_id[sid]
                    assert fp in seed.summary or fp in seed.content, (
                        f"{it.id}: fingerprint {fp!r} not in seed {sid} "
                        f"summary or content"
                    )

    def test_every_fingerprint_unique_across_seeds(self) -> None:
        """Each fingerprint should appear in exactly one seed (summary or content)."""
        seeds = load_seed_memories()
        all_fingerprints = {fp for it in GROUND_TRUTH for fp in it.relevant_fingerprints}
        for fp in all_fingerprints:
            owners = [
                s.id for s in seeds if fp in s.summary or fp in s.content
            ]
            assert len(owners) == 1, (
                f"fingerprint {fp!r} owned by {owners}; expected exactly 1"
            )


# ── Dataset validation (synthetic bad corpus) ─────────────────


def _write_seeds(path: Path, seeds: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


class TestValidateDatasetSynthetic:
    """Validates the validator itself using a controlled corpus."""

    @pytest.fixture
    def base_items(self) -> list:
        # Two items, each pointing at one seed. Wrapped in GroundTruthItem
        # so validate_dataset can use attribute access (seed_ids, etc.).
        return [
            GroundTruthItem({"id": "q1", "query": "q1", "seed_ids": ["s1"],
             "relevant_fingerprints": ["alpha"], "category": "技术决策",
             "difficulty": "easy"}),
            GroundTruthItem({"id": "q2", "query": "q2", "seed_ids": ["s2"],
             "relevant_fingerprints": ["beta"], "category": "技术决策",
             "difficulty": "easy"}),
        ]

    @pytest.fixture
    def base_seeds(self) -> list[dict]:
        return [
            {"id": "s1", "summary": "alpha is the first", "content": "alpha",
             "category": "技术决策"},
            {"id": "s2", "summary": "beta is the second", "content": "beta",
             "category": "技术决策"},
        ]

    def test_clean(self, tmp_path: Path, base_items, base_seeds) -> None:
        seed_file = tmp_path / "seeds.jsonl"
        _write_seeds(seed_file, base_seeds)
        seeds = [SeedMemory.from_dict(s) for s in base_seeds]
        items = base_items
        warnings = validate_dataset(items, seeds)
        assert warnings == []

    def test_missing_seed_id_raises(self, base_items, base_seeds) -> None:
        # Reference a seed_id that doesn't exist
        items = list(base_items)
        items[0]["seed_ids"] = ["nonexistent"]
        seeds = [SeedMemory.from_dict(s) for s in base_seeds]
        with pytest.raises(ValueError, match="missing seed_id"):
            validate_dataset(items, seeds)

    def test_fingerprint_not_in_any_seed_raises(
        self, tmp_path: Path, base_items, base_seeds
    ) -> None:
        # fingerprint that no seed contains
        items = list(base_items)
        items[0]["relevant_fingerprints"] = ["orphan"]
        seeds = [SeedMemory.from_dict(s) for s in base_seeds]
        with pytest.raises(ValueError, match="not found in any seed"):
            validate_dataset(items, seeds)

    def test_fingerprint_in_multiple_seeds_warns(
        self, tmp_path: Path, base_items, base_seeds
    ) -> None:
        # Both seeds contain "shared" → ambiguous ownership
        base_seeds[0]["summary"] = "shared alpha"
        base_seeds[1]["summary"] = "shared beta"
        items = list(base_items)
        items[0]["relevant_fingerprints"] = ["shared"]
        seeds = [SeedMemory.from_dict(s) for s in base_seeds]
        warnings = validate_dataset(items, seeds)
        assert any("matches multiple seeds" in w for w in warnings)

    def test_cross_reference_mismatch_warns(
        self, tmp_path: Path, base_items, base_seeds
    ) -> None:
        # fingerprint is in s2 but item claims s1
        items = list(base_items)
        items[0] = GroundTruthItem(
            {"id": "q1", "query": "q1", "seed_ids": ["s1"],
             "relevant_fingerprints": ["beta"], "category": "技术决策",
             "difficulty": "easy"}
        )
        seeds = [SeedMemory.from_dict(s) for s in base_seeds]
        warnings = validate_dataset(items, seeds)
        assert any("is owned by" in w for w in warnings)


# ── Fingerprint matching ──────────────────────────────────────


class TestFingerprintMatching:
    def test_is_relevant_hit(self) -> None:
        result = {"content": "we chose pgvector 而非 Elasticsearch for vectors"}
        assert is_relevant(result, ["pgvector 而非 Elasticsearch"], "content")

    def test_is_relevant_miss(self) -> None:
        result = {"content": "we chose MySQL"}
        assert not is_relevant(result, ["pgvector"], "content")

    def test_is_relevant_missing_field(self) -> None:
        assert not is_relevant({}, ["pgvector"], "content")

    def test_is_relevant_empty_field(self) -> None:
        assert not is_relevant({"content": ""}, ["pgvector"], "content")

    def test_is_relevant_any_fingerprint(self) -> None:
        result = {"summary": "BGE-M3 是嵌入模型"}
        assert is_relevant(result, ["pgvector", "BGE-M3"], "summary")

    def test_case_sensitive(self) -> None:
        # Fingerprints are deliberately cased; lowercase must not match
        assert is_relevant({"content": "BGE-M3"}, ["BGE-M3"], "content")
        assert not is_relevant({"content": "BGE-M3"}, ["bge-m3"], "content")

    def test_relevance_mask(self) -> None:
        results = [
            {"content": "alpha"},
            {"content": "beta"},
            {"content": "alpha beta"},
            {"content": "gamma"},
        ]
        mask = relevance_mask(results, ["alpha"], "content")
        assert mask == [True, False, True, False]


# ── Semantic relevance (supplementary) ─────────────────────────


class TestSemanticMatching:
    """semantic_relevance_mask — embedding-similarity relevance channel.

    The provider is mocked so no model is loaded; vectors are hand-built so
    cosine behaviour is exact.
    """

    @pytest.mark.asyncio
    async def test_semantic_hit_and_miss(self, monkeypatch) -> None:
        from unittest.mock import AsyncMock

        import backend.service.embedding_service as emb_mod

        from tests.eval.dataset import semantic_relevance_mask

        provider = AsyncMock()
        # embed() is called twice: once for target summaries, once for results.
        provider.embed.side_effect = [
            [[1.0, 0.0]],  # the target summary vector
            [[0.99, 0.1], [0.0, 1.0]],  # result 1 close, result 2 orthogonal
        ]
        monkeypatch.setattr(emb_mod, "get_embedding_provider", lambda: provider)

        results = [
            {"content": "paraphrase of the target meaning"},
            {"content": "unrelated text"},
        ]
        mask = await semantic_relevance_mask(
            results, ["the target summary"], "content"
        )
        assert mask == [True, False]
        # Two embed calls: targets, then the result texts.
        assert provider.embed.await_count == 2

    @pytest.mark.asyncio
    async def test_threshold_respected(self, monkeypatch) -> None:
        from unittest.mock import AsyncMock

        import backend.service.embedding_service as emb_mod

        from tests.eval.dataset import semantic_relevance_mask

        provider = AsyncMock()
        # cosine(target, result) = 0.75 — below the 0.80 floor.
        provider.embed.side_effect = [
            [[1.0, 0.0]],
            [[0.6, 0.8]],  # dot = 0.6 < 0.80
        ]
        monkeypatch.setattr(emb_mod, "get_embedding_provider", lambda: provider)

        mask = await semantic_relevance_mask(
            [{"content": "marginal text"}], ["target"], "content"
        )
        assert mask == [False]

    @pytest.mark.asyncio
    async def test_empty_inputs_all_false(self) -> None:
        from tests.eval.dataset import semantic_relevance_mask

        assert await semantic_relevance_mask([], ["t"], "content") == []
        assert await semantic_relevance_mask(
            [{"content": "x"}], [], "content"
        ) == [False]


# ── Retriever adapter wiring ──────────────────────────────────


class TestBuildAdapter:
    def test_chunk_adapter_name_and_field(self) -> None:
        adapter = build_adapter("chunk")
        assert adapter.name == "chunk:norank"
        assert adapter.match_field == "content"

    def test_chunk_adapter_llm_variant(self) -> None:
        adapter = build_adapter("chunk", use_llm_rerank=True)
        assert adapter.name == "chunk:llm"

    def test_chunk_adapter_cross_encoder_variant(self) -> None:
        adapter = build_adapter("chunk", use_cross_encoder=True)
        assert adapter.name == "chunk:ce"

    def test_memory_adapter_name_and_field(self) -> None:
        adapter = build_adapter("memory")
        assert adapter.name == "memory:norank"
        assert adapter.match_field == "summary"

    def test_memory_adapter_cross_encoder_variant(self) -> None:
        adapter = build_adapter("memory", use_cross_encoder=True)
        assert adapter.name == "memory:ce"

    def test_memory_adapter_threshold_default(self) -> None:
        # Default threshold for memory path is 0.3 — just ensure it builds.
        adapter = build_adapter("memory")
        assert isinstance(adapter, RetrieverAdapter)

    def test_vector_adapter_name_and_field(self) -> None:
        adapter = build_adapter("vector")
        assert adapter.name == "vector:raw"
        assert adapter.match_field == "content"

    def test_hybrid_adapter_name_and_field(self) -> None:
        adapter = build_adapter("hybrid")
        assert adapter.name == "hybrid:norank"
        assert adapter.match_field == "content"

    def test_hybrid_adapter_cross_encoder_variant(self) -> None:
        adapter = build_adapter("hybrid", use_cross_encoder=True)
        assert adapter.name == "hybrid:ce"

    def test_hybrid_adapter_llm_variant(self) -> None:
        """Regression: --llm-rerank on hybrid must actually select the LLM
        reranker, not fall through to the no-rerank path."""
        adapter = build_adapter("hybrid", use_llm_rerank=True)
        assert adapter.name == "hybrid:llm"

    def test_rewrite_adapter_name_and_field(self) -> None:
        adapter = build_adapter("rewrite")
        assert adapter.name == "rewrite:norank"
        assert adapter.match_field == "content"

    def test_rewrite_adapter_cross_encoder_variant(self) -> None:
        adapter = build_adapter("rewrite", use_cross_encoder=True)
        assert adapter.name == "rewrite:ce"

    def test_unknown_retriever_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown retriever"):
            build_adapter("foo")


class TestRetrieverAdapterContract:
    """The adapter contract: fn is awaitable, returns list[dict].

    We don't hit the DB here — we construct a synthetic adapter and verify
    the runner-facing contract. The runner test exercises the full path.
    """

    @pytest.mark.asyncio
    async def test_synthetic_adapter(self) -> None:
        async def fake_fn(query: str, top_k: int) -> list[dict]:
            return [{"content": f"result for {query}", "score": 0.9}] * top_k

        adapter = RetrieverAdapter(name="fake", fn=fake_fn, match_field="content")
        results = await adapter.fn("test", 3)
        assert len(results) == 3
        assert results[0]["content"] == "result for test"
        # match_field works against is_relevant
        assert is_relevant(results[0], ["result for test"], adapter.match_field)
