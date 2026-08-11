"""Tests for the eval query-generation script (tests/eval/experiments/generate_queries.py).

The LLM is never hit: the generation pipeline is driven with an injected
fake generator, so the tests cover the parsing, validation, and output-file
merge logic — the parts that would corrupt the candidates file if wrong.
"""

from __future__ import annotations

import json

import pytest

from tests.eval.experiments.generate_queries import (
    _build_prompt,
    _merge_candidates,
    _processed_seed_ids,
    generate_candidates,
    parse_generation,
)
from tests.eval.dataset import SeedMemory


def _seed(seed_id: str, category: str = "技术决策", summary: str = "默认摘要内容") -> SeedMemory:
    return SeedMemory(
        id=seed_id,
        category=category,
        source_type="eval_seed",
        summary=summary,
        content=summary,
        entities=[],
        relations=[],
    )


def _good_response() -> dict:
    return {
        "positive_queries": [
            {"query": "为什么用 pgvector 不用 ES", "difficulty": "easy"},
            {"query": "向量检索选型理由", "difficulty": "medium"},
            {"query": "同库方案是怎么考虑的", "difficulty": "hard"},
        ],
        "hard_negative": {
            "query": "embedding 模型当初怎么选的",
            "target_memory_id": "seed-002",
            "reason": "都涉及向量选型，用户其实在问 embedding 模型",
        },
    }


def _fake_generator():
    """A generator callable that picks the first peer as the hard-negative target.

    Mirrors what a real LLM should produce: the target is always a *different*
    memory from the one being processed, so every seed yields 4 candidates
    whenever it has at least one peer.
    """

    async def _gen(seed, peers):
        target = peers[0].id if peers else None
        hard_negative = (
            {
                "query": f"易混-{seed.id}-针对{target}",
                "target_memory_id": target,
                "reason": f"选 {target} 作为混淆目标",
            }
            if target
            else {"query": "", "target_memory_id": "", "reason": ""}
        )
        return {
            "positive_queries": [
                {"query": f"正例-easy-{seed.id}", "difficulty": "easy"},
                {"query": f"正例-medium-{seed.id}", "difficulty": "medium"},
                {"query": f"正例-hard-{seed.id}", "difficulty": "hard"},
            ],
            "hard_negative": hard_negative,
        }

    return _gen


def _counting_generator(called: list[str]):
    base = _fake_generator()

    async def _gen(seed, peers):
        called.append(seed.id)
        return await base(seed, peers)

    return _gen


class TestBuildPrompt:
    def test_includes_seed_and_peers(self) -> None:
        seed = _seed("seed-001", summary="决定用 pgvector 而非 ES")
        peers = [_seed("seed-002", summary="决定用 BGE-M3 做嵌入"), _seed("seed-003")]
        prompt = _build_prompt(seed, peers)

        assert "seed-001" in prompt
        assert "决定用 pgvector 而非 ES" in prompt
        assert "[seed-002]" in prompt and "[seed-003]" in prompt
        # Task requirements must be spelled out, not implied.
        assert "positive_queries" in prompt
        assert "hard_negative" in prompt
        assert "target_memory_id" in prompt

    def test_peers_summary_is_truncated(self) -> None:
        long_summary = "长" * 500
        prompt = _build_prompt(
            _seed("seed-001"), [_seed("seed-002", summary=long_summary)]
        )
        # The capped peer summary must not be dumped verbatim into the prompt.
        assert "长" * 500 not in prompt
        assert "…" in prompt


class TestParseGeneration:
    def test_produces_three_positives_and_one_hard_negative(self) -> None:
        seed = _seed("seed-001", category="技术决策", summary="pgvector 选型")
        peers = [_seed("seed-002", category="架构设计", summary="BGE-M3 嵌入模型选型")]
        candidates, warnings = parse_generation(seed, peers, _good_response())

        assert warnings == []
        positives = [c for c in candidates if c["kind"] == "positive"]
        hardneg = [c for c in candidates if c["kind"] == "hard_negative"]
        assert len(positives) == 3
        assert len(hardneg) == 1

        by_diff = {c["difficulty"]: c for c in positives}
        assert set(by_diff) == {"easy", "medium", "hard"}
        for diff, c in by_diff.items():
            assert c["id"] == f"qg-seed-001-{diff}"
            assert c["source_seed_id"] == "seed-001"
            assert c["seed_ids"] == ["seed-001"]
            assert c["category"] == "技术决策"
            assert c["status"] == "candidate"
            assert c["review"] is None

        hn = hardneg[0]
        assert hn["id"] == "qg-seed-001-hardneg"
        assert hn["seed_ids"] == ["seed-002"]  # true intent
        assert hn["distractor_seed_ids"] == ["seed-001"]  # the memory it risks outranking
        assert hn["category"] == "架构设计"  # inherits the TARGET's category
        assert hn["difficulty"] == "hard"
        assert "embedding 模型" in hn["query"]

    def test_unknown_difficulty_defaults_to_medium(self) -> None:
        seed = _seed("seed-001")
        resp = _good_response()
        resp["positive_queries"][1]["difficulty"] = "very_hard"
        candidates, warnings = parse_generation(seed, [_seed("seed-002")], resp)

        medium = [c for c in candidates if c["kind"] == "positive" and c["difficulty"] == "medium"]
        assert len(medium) == 1
        assert any("unknown difficulty" in w for w in warnings)

    def test_drops_hard_negative_targeting_unknown_memory(self) -> None:
        """A hallucinated target id must be dropped, never written."""
        seed = _seed("seed-001")
        resp = _good_response()
        resp["hard_negative"]["target_memory_id"] = "seed-999"
        candidates, warnings = parse_generation(seed, [_seed("seed-002")], resp)

        assert all(c["kind"] != "hard_negative" for c in candidates)
        assert any("unknown memory 'seed-999'" in w for w in warnings)

    def test_drops_empty_positive_query(self) -> None:
        seed = _seed("seed-001")
        resp = _good_response()
        resp["positive_queries"][2] = {"query": "   ", "difficulty": "hard"}
        candidates, warnings = parse_generation(seed, [_seed("seed-002")], resp)

        assert len([c for c in candidates if c["kind"] == "positive"]) == 2
        assert any("empty positive query" in w for w in warnings)

    def test_hard_negative_missing_query_is_skipped(self) -> None:
        seed = _seed("seed-001")
        resp = _good_response()
        resp["hard_negative"]["query"] = ""
        candidates, warnings = parse_generation(seed, [_seed("seed-002")], resp)

        assert all(c["kind"] != "hard_negative" for c in candidates)
        assert any("missing query or target" in w for w in warnings)


class TestGenerateCandidates:
    @pytest.mark.asyncio
    async def test_merges_into_output_file(self, tmp_path) -> None:
        out = tmp_path / "candidates.jsonl"
        seeds = [_seed("seed-001", summary="pgvector 选型"), _seed("seed-002", summary="BGE-M3")]
        n, warnings = await generate_candidates(
            seeds, out_path=out, generator=_fake_generator()
        )
        # 3 positives + 1 hard negative, × 2 seeds (each has the other as peer).
        assert n == 8
        assert warnings == []

        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(rows) == 8
        assert {r["source_seed_id"] for r in rows} == {"seed-001", "seed-002"}
        assert any(r["kind"] == "hard_negative" for r in rows)

    @pytest.mark.asyncio
    async def test_skips_already_processed_seeds(self, tmp_path) -> None:
        """Resume: a seed already in the file is not regenerated."""
        out = tmp_path / "candidates.jsonl"
        seed_a = _seed("seed-001", summary="pgvector 选型")
        seed_b = _seed("seed-002", summary="BGE-M3")
        called: list[str] = []

        await generate_candidates(
            [seed_a], out_path=out, generator=_counting_generator(called)
        )
        assert _processed_seed_ids(out) == {"seed-001"}
        called.clear()

        n, _ = await generate_candidates(
            [seed_a, seed_b], out_path=out, generator=_counting_generator(called)
        )
        # Only seed-002 is new; seed-001 is skipped.
        assert n == 4
        assert called == ["seed-002"]
        assert _processed_seed_ids(out) == {"seed-001", "seed-002"}

    @pytest.mark.asyncio
    async def test_force_regenerates_replacing_old_rows(self, tmp_path) -> None:
        out = tmp_path / "candidates.jsonl"
        seeds = [_seed("seed-001", summary="pgvector 选型"), _seed("seed-002", summary="BGE-M3")]
        await generate_candidates(seeds, out_path=out, generator=_fake_generator())
        assert len(out.read_text(encoding="utf-8").splitlines()) == 8

        await generate_candidates(seeds, out_path=out, force=True, generator=_fake_generator())
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        # Same 8 candidates, no duplicates — the old rows were replaced.
        assert len(rows) == 8
        assert len({r["id"] for r in rows}) == 8

    @pytest.mark.asyncio
    async def test_peers_come_from_corpus_not_subset(self, tmp_path) -> None:
        """--limit/--seed process a subset, but hard-negative targets must be
        pickable from the FULL corpus — a lone seed otherwise gets an empty
        peer list and the LLM invents a target."""
        out = tmp_path / "candidates.jsonl"
        seed_a = _seed("seed-001", summary="pgvector 选型")
        seed_b = _seed("seed-002", summary="BGE-M3")
        seed_c = _seed("seed-003", summary="LLMProvider 抽象")
        seen_peers: list[set[str]] = []

        async def _gen(seed, peers):
            seen_peers.append({p.id for p in peers})
            return await _fake_generator()(seed, peers)

        # Process only seed-001, but the full corpus is seed-001..003.
        n, warnings = await generate_candidates(
            [seed_a], out_path=out, generator=_gen, corpus=[seed_a, seed_b, seed_c]
        )
        assert n == 4  # 3 positives + 1 hard negative (peers non-empty)
        assert warnings == []
        # The peer list must include seeds outside the processed subset.
        assert seen_peers[0] == {"seed-002", "seed-003"}

    @pytest.mark.asyncio
    async def test_per_seed_failure_does_not_abort_batch(self, tmp_path) -> None:
        out = tmp_path / "candidates.jsonl"
        seed_a = _seed("seed-001", summary="pgvector 选型")
        seed_b = _seed("seed-002", summary="BGE-M3")

        async def _gen(seed, peers):
            if seed.id == "seed-001":
                raise RuntimeError("llm down")
            return await _fake_generator()(seed, peers)

        n, warnings = await generate_candidates([seed_a, seed_b], out_path=out, generator=_gen)
        assert n == 4  # only seed-002 succeeded (its peer seed-001 is a valid target)
        assert any("seed-001" in w and "failed" in w for w in warnings)
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert {r["source_seed_id"] for r in rows} == {"seed-002"}


class TestMergeCandidates:
    def test_merge_replaces_only_fresh_seeds(self, tmp_path) -> None:
        out = tmp_path / "c.jsonl"
        out.write_text(
            json.dumps({"source_seed_id": "seed-001", "id": "qg-seed-001-easy", "query": "old"}, ensure_ascii=False)
            + "\n"
            + json.dumps({"source_seed_id": "seed-002", "id": "qg-seed-002-easy", "query": "keep"}, ensure_ascii=False)
            + "\n",
            encoding="utf-8",
        )
        total = _merge_candidates(
            out,
            [
                {"source_seed_id": "seed-001", "id": "qg-seed-001-easy", "query": "new"},
                {"source_seed_id": "seed-003", "id": "qg-seed-003-easy", "query": "fresh"},
            ],
        )
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert total == len(rows) == 3
        by_id = {r["id"]: r["query"] for r in rows}
        assert by_id["qg-seed-001-easy"] == "new"  # replaced
        assert by_id["qg-seed-002-easy"] == "keep"  # preserved
        assert by_id["qg-seed-003-easy"] == "fresh"  # appended
