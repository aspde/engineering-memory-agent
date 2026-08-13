"""Tests for the fingerprint review helper (tests/eval/experiments/fingerprint_review.py).

Pure CPU — no LLM, no DB, no models.  Covers tokenisation, phrase
extraction (every candidate must be a real substring), uniqueness / summary
annotation, suggested selection, and the apply step that refuses fingerprints
that could never be matched.
"""

from __future__ import annotations

import json

from tests.eval.dataset import SeedMemory
from tests.eval.experiments.fingerprint_review import (
    _phrase_candidates,
    _tokens,
    build_review_rows,
    extract_fingerprint_candidates,
    finalize_review,
    render_jsonl_lines,
    select_suggested,
)


def _seed(
    seed_id: str,
    summary: str,
    content: str | None = None,
    entities: list[dict] | None = None,
    category: str = "技术决策",
) -> SeedMemory:
    return SeedMemory(
        id=seed_id,
        category=category,
        source_type="eval_seed",
        summary=summary,
        content=content if content is not None else summary,
        entities=entities or [],
        relations=[],
    )


# ── Fixtures: three seeds, one shared term to exercise uniqueness ──

_S1_SUMMARY = "决定使用 PostgreSQL 的 pgvector 扩展而非 Elasticsearch 做向量检索，pgvector 与业务库同库"
_S2_SUMMARY = "选择 BGE-M3 嵌入模型，1024 维支持中英双语，本地推理无 API 成本"
# S3 mentions PostgreSQL too — the shared term must be flagged non-unique.
_S3_SUMMARY = "PostgreSQL 连接池使用 NullPool 规避 Windows 事件循环问题"

SEEDS = [
    _seed(
        "seed-001",
        _S1_SUMMARY,
        entities=[
            {"name": "pgvector", "type": "technology"},
            {"name": "PostgreSQL", "type": "technology"},
        ],
    ),
    _seed(
        "seed-002",
        _S2_SUMMARY,
        entities=[{"name": "BGE-M3", "type": "technology"}],
    ),
    _seed("seed-003", _S3_SUMMARY, entities=[{"name": "PostgreSQL", "type": "technology"}]),
]


def _candidate(
    cid: str = "qg-seed-001-easy",
    query: str = "为什么用 pgvector 不用 ES",
    seed_ids: list[str] | None = None,
    kind: str = "positive",
    category: str = "技术决策",
    difficulty: str = "easy",
    status: str = "approved",
) -> dict:
    return {
        "id": cid,
        "source_seed_id": seed_ids[0] if seed_ids else "seed-001",
        "seed_ids": seed_ids or ["seed-001"],
        "query": query,
        "kind": kind,
        "category": category,
        "difficulty": difficulty,
        "status": status,
        "review": None,
    }


class TestTokenise:
    def test_splits_identifiers_numbers_and_cjk(self) -> None:
        toks = _tokens("pgvector 扩展而非 Elasticsearch 0.92/0.75")
        assert "pgvector" in toks
        assert "Elasticsearch" in toks
        assert "扩展而非" in toks
        assert "0.92/0.75" in toks

    def test_phrase_candidates_are_real_substrings(self) -> None:
        text = "决定使用 PostgreSQL 的 pgvector 扩展而非 Elasticsearch"
        for phrase in _phrase_candidates(text):
            assert phrase in text


class TestExtractFingerprintCandidates:
    def test_entity_and_phrase_candidates_annotated(self) -> None:
        target = SEEDS[0]
        cands = extract_fingerprint_candidates(target, SEEDS)
        by_fp = {c["fingerprint"]: c for c in cands}

        # "pgvector" appears only in seed-001 → unique.
        pgv = by_fp.get("pgvector")
        assert pgv is not None and pgv["unique"] is True and pgv["in_summary"] is True
        assert pgv["source"] == "entity"

        # "PostgreSQL" appears in seed-001 AND seed-003 → not unique.
        pg = by_fp.get("PostgreSQL")
        assert pg is not None and pg["unique"] is False

        # A distinctive phrase is unique and hits the summary.
        assert any(
            c["fingerprint"] == "pgvector 扩展而非 Elasticsearch"
            and c["unique"] is True
            and c["in_summary"] is True
            for c in cands
        )

    def test_content_only_phrase_flagged_not_in_summary(self) -> None:
        content = "补充细节：pgvector 支持余弦距离，索引用 ivfflat lists=100"
        target = _seed("seed-004", "向量检索使用 pgvector", content=content)
        others = [s for s in SEEDS if s.id != target.id]
        cands = extract_fingerprint_candidates(target, [target] + others)
        # "ivfflat lists=100" lives only in content → in_summary False.
        hit = next((c for c in cands if "ivfflat lists=100" in c["fingerprint"]), None)
        assert hit is not None and hit["in_summary"] is False


class TestSelectSuggested:
    def test_prefers_unique_summary_hitting_phrase(self) -> None:
        target = SEEDS[0]
        cands = extract_fingerprint_candidates(target, SEEDS)
        suggested = select_suggested(cands, max_suggested=2)
        assert suggested, "should suggest at least one fingerprint"
        for fp in suggested:
            entry = next(c for c in cands if c["fingerprint"] == fp)
            assert entry["unique"] is True and entry["in_summary"] is True

    def test_no_suggestions_when_nothing_unique(self) -> None:
        # A memory whose every candidate appears elsewhere gets no suggestion.
        s_dup = _seed("seed-x", _S2_SUMMARY, entities=[{"name": "BGE-M3", "type": "technology"}])
        cands = extract_fingerprint_candidates(s_dup, [s_dup, SEEDS[1]])
        # BGE-M3 + every phrase of S2's summary lives in seed-002 → all non-unique.
        assert all(c["unique"] is False for c in cands)
        assert select_suggested(cands) == []


class TestBuildReviewRows:
    def test_hard_negative_targets_true_intent(self) -> None:
        """A hard negative's fingerprints come from seed_ids[0] (true intent),
        not the distractor."""
        item = _candidate(
            cid="qg-seed-001-hardneg",
            query="embedding 模型当初怎么选的",
            seed_ids=["seed-002"],  # true intent
            kind="hard_negative",
            category="技术决策",
        )
        row = build_review_rows([item], SEEDS)[0]
        assert row["target_missing"] is False
        fps = [c["fingerprint"] for c in row["candidate_fingerprints"]]
        # BGE-M3 is seed-002's entity — the target's text, not seed-001's.
        assert "BGE-M3" in fps
        bg = next(c for c in row["candidate_fingerprints"] if c["fingerprint"] == "BGE-M3")
        assert bg["in_summary"] is True  # BGE-M3 appears in seed-002's summary

    def test_missing_target_seed_is_flagged(self) -> None:
        item = _candidate(seed_ids=["seed-999"])
        row = build_review_rows([item], SEEDS)[0]
        assert row["target_missing"] is True
        assert row["candidate_fingerprints"] == []

    def test_review_row_carries_suggestions(self) -> None:
        item = _candidate(seed_ids=["seed-001"])
        row = build_review_rows([item], SEEDS)[0]
        assert row["candidate_id"] == "qg-seed-001-easy"
        assert row["status"] == "pending"
        assert row["chosen_fingerprints"] is None
        assert row["suggested_fingerprints"]


class TestFinalizeReview:
    def test_approved_with_chosen_emits_addition(self) -> None:
        row = build_review_rows([_candidate(seed_ids=["seed-001"])], SEEDS)[0]
        row["status"] = "approved"
        row["chosen_fingerprints"] = ["pgvector 扩展而非 Elasticsearch"]
        additions, warnings = finalize_review([row], SEEDS)

        assert warnings == []
        assert len(additions) == 1
        add = additions[0]
        assert add["id"] == "qg-seed-001-easy"
        assert add["seed_ids"] == ["seed-001"]
        assert add["relevant_fingerprints"] == ["pgvector 扩展而非 Elasticsearch"]
        assert add["difficulty"] == "easy"

    def test_refuses_non_substring_fingerprint(self) -> None:
        row = build_review_rows([_candidate(seed_ids=["seed-001"])], SEEDS)[0]
        row["status"] = "approved"
        row["chosen_fingerprints"] = ["根本不存在的指纹文本"]
        additions, warnings = finalize_review([row], SEEDS)

        assert additions == []
        assert any("not a substring" in w for w in warnings)

    def test_refuses_ambiguous_shared_fingerprint(self) -> None:
        """'PostgreSQL' lives in seed-001 and seed-003 — ambiguous, refused."""
        row = build_review_rows([_candidate(seed_ids=["seed-001"])], SEEDS)[0]
        row["status"] = "approved"
        row["chosen_fingerprints"] = ["PostgreSQL"]
        additions, warnings = finalize_review([row], SEEDS)

        assert additions == []
        assert any("another seed" in w for w in warnings)

    def test_approved_without_chosen_is_skipped_with_warning(self) -> None:
        row = build_review_rows([_candidate(seed_ids=["seed-001"])], SEEDS)[0]
        row["status"] = "approved"  # but chosen_fingerprints left None
        additions, warnings = finalize_review([row], SEEDS)

        assert additions == []
        assert any("no chosen fingerprints" in w for w in warnings)

    def test_pending_rows_are_ignored(self) -> None:
        row = build_review_rows([_candidate(seed_ids=["seed-001"])], SEEDS)[0]
        # status stays "pending"
        additions, _ = finalize_review([row], SEEDS)
        assert additions == []

    def test_duplicate_approved_ids_are_dropped(self) -> None:
        """Re-applying a review must not emit the same query twice."""
        row = build_review_rows([_candidate(seed_ids=["seed-001"])], SEEDS)[0]
        row["status"] = "approved"
        row["chosen_fingerprints"] = ["pgvector 扩展而非 Elasticsearch"]
        dup = json.loads(json.dumps(row))  # identical second row
        additions, warnings = finalize_review([row, dup], SEEDS)

        assert len(additions) == 1
        assert any("duplicate approved row" in w for w in warnings)


class TestRenderJsonlLines:
    def _additions(self) -> list[dict]:
        return [
            {
                "id": "qg-seed-001-easy",
                "query": "为什么用 pgvector 不用 Elasticsearch",
                "seed_ids": ["seed-001"],
                "relevant_fingerprints": ["pgvector 扩展而非 Elasticsearch"],
                "category": "技术决策",
                "difficulty": "easy",
                "notes": "generated from seed-001 (reviewed)",
            },
            {
                "id": "qg-seed-007-hard",
                "query": "koa-connect 之前出过什么问题",
                "seed_ids": ["seed-007"],
                "relevant_fingerprints": ["ctx 泄漏"],
                "category": "故障复盘",
                "difficulty": "hard",
            },
        ]

    def test_renders_one_json_object_per_line_and_preserves_values(self) -> None:
        additions = self._additions()
        lines = [
            json.loads(ln)
            for ln in render_jsonl_lines(additions).splitlines()
            if ln.strip()
        ]

        assert len(lines) == 2
        by_id = {it["id"]: it for it in lines}
        assert by_id["qg-seed-001-easy"]["query"] == "为什么用 pgvector 不用 Elasticsearch"
        assert by_id["qg-seed-001-easy"]["relevant_fingerprints"] == [
            "pgvector 扩展而非 Elasticsearch"
        ]
        assert by_id["qg-seed-001-easy"]["seed_ids"] == ["seed-001"]
        assert by_id["qg-seed-001-easy"]["category"] == "技术决策"
        assert by_id["qg-seed-007-hard"]["difficulty"] == "hard"
        # Second addition has no notes → the key is simply absent.
        assert "notes" not in by_id["qg-seed-007-hard"]

    def test_special_characters_roundtrip_through_json(self) -> None:
        weird = '她说"pgvector 扩展"最稳定，路径是 C:\\proj\\ema，换行\n第二行'
        additions = [
            {
                "id": "qg-weird-hard",
                "query": weird,
                "seed_ids": ["seed-001"],
                "relevant_fingerprints": ["pgvector 扩展"],
                "category": "技术决策",
                "difficulty": "hard",
                "notes": '含引号 " 和反斜杠 \\ 与制表符\t',
            }
        ]
        lines = render_jsonl_lines(additions).splitlines()
        assert len(lines) == 1
        it = json.loads(lines[0])
        assert it["query"] == weird
        assert it["notes"] == '含引号 " 和反斜杠 \\ 与制表符\t'

    def test_rows_are_self_contained_ground_truth_items(self) -> None:
        """Each line is a complete ground-truth row (loader shape) — no
        surrounding syntax, so the block appends to ground_truth.jsonl."""
        adds = self._additions()
        rows = [json.loads(ln) for ln in render_jsonl_lines(adds).splitlines() if ln.strip()]
        assert len(rows) == 2
        for r in rows:
            for key in ("id", "query", "seed_ids", "relevant_fingerprints",
                        "category", "difficulty"):
                assert key in r, f"row missing {key}"

    def test_notes_key_omitted_when_absent(self) -> None:
        lines = render_jsonl_lines(self._additions()).splitlines()
        assert "notes" in json.loads(lines[0])
        # The second addition has no notes → the key is simply absent.
        assert "notes" not in json.loads(lines[1])


class TestEndToEnd:
    def test_review_then_apply_over_temp_files(self, tmp_path) -> None:
        from tests.eval.experiments.fingerprint_review import _read_jsonl, _write_jsonl

        candidates = [_candidate(seed_ids=["seed-001"])]
        cand_path = tmp_path / "candidates.jsonl"
        _write_jsonl(cand_path, candidates)

        # Review step.
        review_path = tmp_path / "review.jsonl"
        rows = build_review_rows(candidates, SEEDS)
        _write_jsonl(review_path, rows)
        loaded = _read_jsonl(review_path)
        assert len(loaded) == 1

        # Reviewer approves and accepts a suggested fingerprint.
        loaded[0]["status"] = "approved"
        loaded[0]["chosen_fingerprints"] = loaded[0]["suggested_fingerprints"][:1]
        _write_jsonl(review_path, loaded)

        additions, warnings = finalize_review(_read_jsonl(review_path), SEEDS)
        assert len(additions) == 1 and warnings == []
        # The applied fingerprint is a real, unique substring of seed-001.
        fp = additions[0]["relevant_fingerprints"][0]
        assert fp in _S1_SUMMARY
        assert json.dumps(additions[0], ensure_ascii=False)
