"""Tests for retrieval pipeline — sparse_search (jieba + Jaccard) and helpers.

DB access is mocked at the session-factory boundary so tests stay fast and
do not require a running Postgres.  The Jaccard scoring logic is exercised
end-to-end with realistic Chinese/English chunks.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _row(id: str, content: str, meta: dict | None = None, chunk_index: int = 0):
    """Build a mock SQLAlchemy Row with a ``_mapping`` attribute.

    Includes a ``tokens`` field (jieba segmentation) to mirror what the
    DB-side ``sparse_search`` query returns after the GIN-index migration.
    """
    from backend.service.retrieval import _tokenize

    r = MagicMock()
    r._mapping = {
        "id": id,
        "content": content,
        "meta": meta or {},
        "chunk_index": chunk_index,
        "tokens": list(_tokenize(content)),
    }
    return r


def _mock_session_factory(rows: list):
    """Return a session factory whose execute() yields ``rows``."""
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.__iter__ = lambda self: iter(rows)
    mock_session.execute.return_value = mock_result

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    factory = MagicMock(return_value=ctx)
    return factory


class TestTokenize:
    """The jieba-based tokenizer is the core of sparse_search."""

    def test_chinese_segmented(self) -> None:
        from backend.service.retrieval import _tokenize

        tokens = _tokenize("为什么用 pgvector 做向量检索")
        # Chinese is segmented into multi-char tokens; stopwords dropped
        assert "pgvector" in tokens
        assert "向量" in tokens
        assert "检索" in tokens
        # "的" is a stopword and must be dropped
        assert "的" not in _tokenize("的 pgvector")
        assert len(tokens) > 0

    def test_stopwords_dropped(self) -> None:
        from backend.service.retrieval import _tokenize

        tokens = _tokenize("的 了 是 在")
        assert tokens == set()

    def test_ascii_single_char_kept(self) -> None:
        from backend.service.retrieval import _tokenize

        # Single ASCII chars (e.g. variable names) carry signal — kept.
        tokens = _tokenize("variable c")
        assert "c" in tokens
        assert "variable" in tokens

    def test_empty_string(self) -> None:
        from backend.service.retrieval import _tokenize

        assert _tokenize("") == set()


class TestSparseSearch:
    @pytest.mark.asyncio
    async def test_chinese_query_rescues_keyword_overlap(self) -> None:
        """A Chinese query with shared keywords should rank matching chunks."""
        from backend.service import retrieval as mod

        rows = [
            _row("c1", "PostgresSaver Windows 兼容性问题导致 checkpoint 写入失败"),
            _row("c2", "BGE-M3 CPU 推理是检索延迟的主要瓶颈"),
            _row("c3", "LangGraph 状态图采用 5 节点设计"),
        ]
        with patch.object(
            mod, "get_session_factory", return_value=_mock_session_factory(rows)
        ):
            result = await mod.sparse_search("PostgresSaver 在 Windows 上为什么不能用", top_k=5)

        assert len(result) >= 1
        # The chunk sharing Windows/PostgresSaver keywords must rank first
        assert result[0]["id"] == "c1"
        assert result[0]["rank"] > 0.0

    @pytest.mark.asyncio
    async def test_no_overlap_returns_empty(self) -> None:
        """A query sharing no tokens with any chunk returns nothing."""
        from backend.service import retrieval as mod

        rows = [_row("c1", "完全无关的内存内容关于饮食偏好")]
        with patch.object(
            mod, "get_session_factory", return_value=_mock_session_factory(rows)
        ):
            result = await mod.sparse_search("pgvector 向量检索", top_k=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_stopword_only_query_returns_empty(self) -> None:
        """A query that tokenizes to nothing (all stopwords) short-circuits."""
        from backend.service import retrieval as mod

        rows = [_row("c1", "some chunk")]
        with patch.object(
            mod, "get_session_factory", return_value=_mock_session_factory(rows)
        ):
            result = await mod.sparse_search("的 了 是", top_k=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_db_returns_empty(self) -> None:
        """No chunks in DB → empty result."""
        from backend.service import retrieval as mod

        with patch.object(
            mod, "get_session_factory", return_value=_mock_session_factory([])
        ):
            result = await mod.sparse_search("anything", top_k=5)
        assert result == []

    @pytest.mark.asyncio
    async def test_top_k_limit(self) -> None:
        """Result count respects top_k even when more chunks match."""
        from backend.service import retrieval as mod

        rows = [
            _row(f"c{i}", f"pgvector 向量检索 chunk {i}") for i in range(10)
        ]
        with patch.object(
            mod, "get_session_factory", return_value=_mock_session_factory(rows)
        ):
            result = await mod.sparse_search("pgvector 向量检索", top_k=3)
        assert len(result) == 3
        # All results should have a positive Jaccard rank
        assert all(r["rank"] > 0.0 for r in result)

    @pytest.mark.asyncio
    async def test_ranking_descending_by_jaccard(self) -> None:
        """Chunks with higher token overlap rank higher."""
        from backend.service import retrieval as mod

        rows = [
            # c1 shares all query tokens → highest Jaccard
            _row("c1", "pgvector 向量检索 性能 调优"),
            # c2 shares only two tokens → lower Jaccard
            _row("c2", "pgvector 向量"),
            # c3 shares one token → lowest
            _row("c3", "pgvector 无关内容"),
        ]
        with patch.object(
            mod, "get_session_factory", return_value=_mock_session_factory(rows)
        ):
            result = await mod.sparse_search("pgvector 向量检索 性能", top_k=3)

        assert [r["id"] for r in result] == ["c1", "c2", "c3"]
        assert result[0]["rank"] > result[1]["rank"] > result[2]["rank"]

    @pytest.mark.asyncio
    async def test_result_shape(self) -> None:
        """Returned dicts carry id/content/meta/chunk_index/rank."""
        from backend.service import retrieval as mod

        rows = [_row("c1", "pgvector 向量检索", {"doc": "a.py"}, 2)]
        with patch.object(
            mod, "get_session_factory", return_value=_mock_session_factory(rows)
        ):
            result = await mod.sparse_search("pgvector", top_k=5)

        assert len(result) == 1
        r = result[0]
        assert set(r.keys()) >= {"id", "content", "meta", "chunk_index", "rank"}
        assert r["meta"] == {"doc": "a.py"}
        assert r["chunk_index"] == 2


class TestRetrieveHybrid:
    """Union/dedup/fallback orchestration in retrieve_hybrid."""

    def _patch_sources(self, monkeypatch, dense, sparse):
        from backend.service import retrieval as mod

        monkeypatch.setattr(mod, "embed_query", AsyncMock(return_value=[0.1]))
        monkeypatch.setattr(mod, "vector_search", AsyncMock(return_value=dense))
        monkeypatch.setattr(mod, "sparse_search", AsyncMock(return_value=sparse))

    @pytest.mark.asyncio
    async def test_union_dedup_by_chunk_id(self, monkeypatch) -> None:
        from backend.service import retrieval as mod

        dense = [
            {"id": "c1", "content": "c1 dense", "meta": {}, "similarity": 0.8},
            {"id": "c2", "content": "c2 dense", "meta": {}, "similarity": 0.6},
        ]
        sparse = [
            # c2 appears in both — must be deduped, dense copy wins the rerank slot
            {"id": "c2", "content": "c2 sparse", "meta": {}, "rank": 0.5},
            {"id": "c3", "content": "c3 sparse", "meta": {}, "rank": 0.4},
        ]
        self._patch_sources(monkeypatch, dense, sparse)
        monkeypatch.setattr(
            "backend.service.rerank.rerank_cross_encoder",
            AsyncMock(return_value=[(0, 0.9), (1, 0.8), (2, 0.2)]),
        )

        results = await mod.retrieve_hybrid("query", top_k=5)

        assert [r.content for r in results] == ["c1 dense", "c2 dense", "c3 sparse"]
        assert [r.score for r in results] == [0.9, 0.8, 0.2]

    @pytest.mark.asyncio
    async def test_skip_rerank_keeps_both_scores_for_overlap(self, monkeypatch) -> None:
        """Regression: a chunk in BOTH sets must rank by max(dense, sparse),
        not by the dense score alone — the old union dropped the sparse rank."""
        from backend.service import retrieval as mod

        dense = [{"id": "c1", "content": "c1", "meta": {}, "similarity": 0.5}]
        sparse = [{"id": "c1", "content": "c1", "meta": {}, "rank": 0.9}]
        self._patch_sources(monkeypatch, dense, sparse)

        results = await mod.retrieve_hybrid("query", top_k=5, skip_rerank=True)

        assert len(results) == 1
        assert results[0].score == pytest.approx(0.9)  # max(0.5, 0.9), not 0.5

    @pytest.mark.asyncio
    async def test_skip_rerank_sorts_by_max_score(self, monkeypatch) -> None:
        from backend.service import retrieval as mod

        dense = [
            {"id": "c1", "content": "c1", "meta": {}, "similarity": 0.7},
            {"id": "c2", "content": "c2", "meta": {}, "similarity": 0.2},
        ]
        sparse = [
            {"id": "c1", "content": "c1", "meta": {}, "rank": 0.1},
            {"id": "c2", "content": "c2", "meta": {}, "rank": 0.6},
        ]
        self._patch_sources(monkeypatch, dense, sparse)

        results = await mod.retrieve_hybrid("query", top_k=5, skip_rerank=True)

        # c1: max(0.7, 0.1)=0.7 ; c2: max(0.2, 0.6)=0.6 → c1 first
        assert [r.content for r in results] == ["c1", "c2"]
        assert results[0].score == pytest.approx(0.7)
        assert results[1].score == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_floor_filters_below_threshold(self, monkeypatch) -> None:
        from backend.service import retrieval as mod

        dense = [
            {"id": "c1", "content": "c1", "meta": {}, "similarity": 0.8},
            {"id": "c2", "content": "c2", "meta": {}, "similarity": 0.5},
        ]
        self._patch_sources(monkeypatch, dense, [])
        monkeypatch.setattr(
            "backend.service.rerank.rerank_cross_encoder",
            AsyncMock(return_value=[(0, 0.9), (1, 0.1)]),
        )

        results = await mod.retrieve_hybrid("query", top_k=5)

        # index 1 scored 0.1 < _RERANK_FLOOR 0.15 → dropped
        assert len(results) == 1
        assert results[0].content == "c1"

    @pytest.mark.asyncio
    async def test_no_candidates_returns_empty(self, monkeypatch) -> None:
        from backend.service import retrieval as mod

        self._patch_sources(monkeypatch, [], [])

        results = await mod.retrieve_hybrid("query", top_k=5)
        assert results == []


class TestRetrieveMultiQuery:
    """Multi-query rewrite → batch embed → union → dedup → rerank."""

    def _patch_llm_and_embed(self, monkeypatch, queries, vector_search):
        from backend.service import retrieval as mod

        monkeypatch.setattr(
            "backend.service.query_rewrite.rewrite_query",
            AsyncMock(return_value=queries),
        )
        provider = MagicMock()
        provider.embed = AsyncMock(return_value=[[0.1 + i] for i in range(len(queries))])
        monkeypatch.setattr(mod, "get_embedding_provider", lambda: provider)
        monkeypatch.setattr(mod, "vector_search", vector_search)
        return provider

    @pytest.mark.asyncio
    async def test_unions_and_dedups_across_variations(self, monkeypatch) -> None:
        from backend.service import retrieval as mod

        queries = ["原查询", "变体一", "变体二"]
        vector_search = AsyncMock(side_effect=[
            [{"id": "c1", "content": "c1", "meta": {}, "similarity": 0.7}],
            [{"id": "c2", "content": "c2", "meta": {}, "similarity": 0.6}],
            [{"id": "c2", "content": "c2 dup", "meta": {}, "similarity": 0.5},
             {"id": "c3", "content": "c3", "meta": {}, "similarity": 0.4}],
        ])
        provider = self._patch_llm_and_embed(monkeypatch, queries, vector_search)
        monkeypatch.setattr(
            "backend.service.rerank.rerank_cross_encoder",
            AsyncMock(return_value=[(0, 0.9), (1, 0.8), (2, 0.7)]),
        )

        results = await mod.retrieve_multi_query("原查询", top_k=5)

        # c2 appears in two variations but is deduped to one result
        assert [r.content for r in results] == ["c1", "c2", "c3"]
        # All variations embedded in ONE provider call
        provider.embed.assert_awaited_once_with(queries)

    @pytest.mark.asyncio
    async def test_empty_results_returns_empty(self, monkeypatch) -> None:
        from backend.service import retrieval as mod

        self._patch_llm_and_embed(
            monkeypatch, ["query"], AsyncMock(return_value=[])
        )

        results = await mod.retrieve_multi_query("query", top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_floor_filters_low_scores(self, monkeypatch) -> None:
        from backend.service import retrieval as mod

        vector_search = AsyncMock(return_value=[
            {"id": "c1", "content": "c1", "meta": {}, "similarity": 0.9},
        ])
        self._patch_llm_and_embed(monkeypatch, ["query"], vector_search)
        monkeypatch.setattr(
            "backend.service.rerank.rerank_cross_encoder",
            AsyncMock(return_value=[(0, 0.1)]),
        )

        results = await mod.retrieve_multi_query("query", top_k=5)
        assert results == []
