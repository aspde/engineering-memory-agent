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


def _mock_session_factory_with_session(mock_session: AsyncMock, results: list):
    """Wrap a caller-provided ``mock_session`` whose execute() returns
    successive ``results`` items (for multi-call flows like write_chunks).
    """
    mock_session.execute.side_effect = results
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_session)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return MagicMock(return_value=ctx)


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

    def test_tokenize_chunks_batch(self) -> None:
        """_tokenize_chunks tokenizes each chunk independently, in order."""
        from backend.service.retrieval import _tokenize, _tokenize_chunks

        batches = _tokenize_chunks(["pgvector 向量检索", "的 了 是", "variable c"])
        assert len(batches) == 3
        assert batches[0] == _tokenize("pgvector 向量检索")
        assert batches[1] == set()  # all stopwords
        assert batches[2] == _tokenize("variable c")

    def test_tokenize_concurrent_threads(self) -> None:
        """Tokenization is safe under concurrent thread-pool offloads.

        The async paths run jieba in the thread pool; a module lock must keep
        concurrent calls from racing on jieba's shared tokenizer singleton.
        Smoke test: N threads each tokenize, all results come back correct.
        """
        import threading

        from backend.service.retrieval import _tokenize

        sample = "pgvector 向量检索 和 PostgreSQL 的 embedding 支持"
        results: list[set[str]] = []
        errors: list[BaseException] = []

        def _run() -> None:
            try:
                results.append(_tokenize(sample))
            except BaseException as exc:  # pragma: no cover
                errors.append(exc)

        threads = [threading.Thread(target=_run) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(results) == 8
        assert all(r == results[0] for r in results)


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

        results = await mod.retrieve_hybrid("query", top_k=5, skip_rerank=False)

        assert [r.content for r in results] == ["c1 dense", "c2 dense", "c3 sparse"]
        assert [r.score for r in results] == [0.9, 0.8, 0.2]

    @pytest.mark.asyncio
    async def test_skip_rerank_fuses_ranks_for_overlap(self, monkeypatch) -> None:
        """A chunk in BOTH sets contributes BOTH ranks to its RRF score —
        not the old max(dense, sparse), where the dense-only score would
        have ignored the sparse rank entirely."""
        from backend.service import retrieval as mod

        dense = [{"id": "c1", "content": "c1", "meta": {}, "similarity": 0.5}]
        sparse = [{"id": "c1", "content": "c1", "meta": {}, "rank": 0.9}]
        self._patch_sources(monkeypatch, dense, sparse)

        results = await mod.retrieve_hybrid("query", top_k=5, skip_rerank=True)

        assert len(results) == 1
        # dense rank 0 + sparse rank 0 → (1/60 + 1/60), normalised by the
        # max achievable fusion (2/60) back to the 0-1 similarity scale.
        assert results[0].score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_default_skips_rerank(self, monkeypatch) -> None:
        """Default behavior is the fast path: no cross-encoder call.

        The eval report shows rerank costs ~90x latency while lowering
        recall@5, so ``skip_rerank`` defaults to True.  Regression guard:
        if the default flips back, ``rerank_cross_encoder`` (mocked to
        raise) would be invoked and this test would error.
        """
        from backend.service import retrieval as mod

        dense = [
            {"id": "c1", "content": "c1", "meta": {}, "similarity": 0.7},
            {"id": "c2", "content": "c2", "meta": {}, "similarity": 0.2},
        ]
        sparse = [{"id": "c2", "content": "c2", "meta": {}, "rank": 0.6}]
        self._patch_sources(monkeypatch, dense, sparse)
        monkeypatch.setattr(
            "backend.service.rerank.rerank_cross_encoder",
            AsyncMock(side_effect=AssertionError("rerank must not run by default")),
        )

        results = await mod.retrieve_hybrid("query", top_k=5)

        # c1 is dense-only (rank 0) → 1/60, normalised to 0.5; c2 is ranked
        # by BOTH retrievers (dense rank 1, sparse rank 0) → (1/61 + 1/60),
        # normalised to 30/61 + 0.5 ≈ 0.99, so RRF promotes it.
        assert [r.content for r in results] == ["c2", "c1"]
        assert results[0].score == pytest.approx(30 / 61 + 0.5)
        assert results[1].score == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_skip_rerank_sorts_by_rrf(self, monkeypatch) -> None:
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

        # Both in both lists: c1 at (dense 0, sparse 0) → 2/60 → 1.0 beats
        # c2 at (dense 1, sparse 1) → 2/61 → 60/61 (both normalised by 2/60).
        assert [r.content for r in results] == ["c1", "c2"]
        assert results[0].score == pytest.approx(1.0)
        assert results[1].score == pytest.approx(60 / 61)

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

        results = await mod.retrieve_hybrid("query", top_k=5, skip_rerank=False)

        # index 1 scored 0.1 < _RERANK_FLOOR 0.15 → dropped
        assert len(results) == 1
        assert results[0].content == "c1"

    @pytest.mark.asyncio
    async def test_no_candidates_returns_empty(self, monkeypatch) -> None:
        from backend.service import retrieval as mod

        self._patch_sources(monkeypatch, [], [])

        results = await mod.retrieve_hybrid("query", top_k=5)
        assert results == []


class TestQueryMemoriesDecayBatch:
    """query_memories updates decay in ONE batch; a write failure can't sink the search."""

    @staticmethod
    def _candidate(mid: str, decay: float = 0.8):
        return {
            "id": mid,
            "source_type": "conversation",
            "summary": f"summary {mid}",
            "entities": [],
            "relations": [],
            "decay_factor": decay,
            "recall_count": 1,
            "meta": {},
            "created_at": None,
        }

    def _patch(self, monkeypatch, candidates, ranked, decay_map=None, decay_raises=False):
        from backend.service import retrieval as mod
        from backend.service import decay as decay_mod

        provider = MagicMock()
        provider.embed = AsyncMock(return_value=[[0.1]])
        monkeypatch.setattr(mod, "get_embedding_provider", lambda: provider)
        monkeypatch.setattr(mod, "search_memories", AsyncMock(return_value=candidates))
        monkeypatch.setattr(
            "backend.service.rerank.rerank_cross_encoder",
            AsyncMock(return_value=ranked),
        )
        if decay_raises:
            monkeypatch.setattr(
                decay_mod,
                "update_decay_batch",
                AsyncMock(side_effect=RuntimeError("db down")),
            )
        else:
            monkeypatch.setattr(
                decay_mod,
                "update_decay_batch",
                AsyncMock(return_value=decay_map or {}),
            )
        return mod

    @pytest.mark.asyncio
    async def test_batch_update_in_single_call(self, monkeypatch) -> None:
        """All surviving memory ids go through ONE update_decay_batch call,
        not N sequential update_decay commits (the N+1 fix)."""
        from backend.service import retrieval as mod
        from backend.service import decay as decay_mod

        cands = [self._candidate("m1"), self._candidate("m2", 0.5)]
        self._patch(
            monkeypatch,
            cands,
            [(0, 0.9), (1, 0.8)],
            decay_map={"m1": 0.99, "m2": 0.77},
        )

        results = await mod.query_memories("q", top_k=5)

        assert [r["id"] for r in results] == ["m1", "m2"]
        decay_mod.update_decay_batch.assert_awaited_once_with(["m1", "m2"])
        # New factors from the batch overwrite the candidates' stale ones.
        assert results[0]["decay_factor"] == pytest.approx(0.99)
        assert results[1]["decay_factor"] == pytest.approx(0.77)

    @pytest.mark.asyncio
    async def test_decay_failure_returns_stale_factors(self, monkeypatch) -> None:
        """A decay-write failure must not fail the search — stale factors
        are returned and the error is logged, not propagated."""
        from backend.service import retrieval as mod

        cands = [self._candidate("m1", 0.8)]
        self._patch(monkeypatch, cands, [(0, 0.9)], decay_raises=True)

        results = await mod.query_memories("q", top_k=5)

        assert len(results) == 1
        assert results[0]["id"] == "m1"
        assert results[0]["decay_factor"] == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_floor_filters_before_decay(self, monkeypatch) -> None:
        """Scores below the rerank floor are dropped before the decay batch,
        so they never consume an update."""
        from backend.service import retrieval as mod
        from backend.service import decay as decay_mod

        cands = [self._candidate("m1"), self._candidate("m2")]
        self._patch(monkeypatch, cands, [(0, 0.9), (1, 0.1)])  # m2 below floor

        results = await mod.query_memories("q", top_k=5)

        assert [r["id"] for r in results] == ["m1"]
        decay_mod.update_decay_batch.assert_awaited_once_with(["m1"])


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


class TestWriteChunksIdempotency:
    """Content-hash idempotency in write_chunks."""

    @pytest.mark.asyncio
    async def test_reingest_skips_all_existing_chunks(self) -> None:
        """Re-ingesting an unchanged document embeds/writes nothing."""
        from backend.service import retrieval as mod

        chunks = ["chunk one", "chunk two"]
        existing_hashes = [mod._content_hash(c) for c in chunks]

        select_result = MagicMock()
        select_result.fetchall.return_value = [(h,) for h in existing_hashes]
        insert_result = MagicMock()
        mock_session = AsyncMock()

        mock_emb = AsyncMock()
        with (
            patch.object(
                mod,
                "get_session_factory",
                return_value=_mock_session_factory_with_session(
                    mock_session, [select_result, insert_result]
                ),
            ),
            patch.object(mod, "get_embedding_provider", return_value=mock_emb),
        ):
            count = await mod.write_chunks("doc-1", chunks)

        assert count == 0
        mock_emb.embed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_partial_reingest_writes_only_new_chunks(self) -> None:
        """Only chunks whose hash isn't stored yet are embedded + inserted."""
        from backend.service import retrieval as mod

        chunks = ["brand new chunk", "unchanged chunk"]
        existing_hashes = [mod._content_hash(chunks[1])]  # second already stored

        select_result = MagicMock()
        select_result.fetchall.return_value = [(h,) for h in existing_hashes]
        insert_result = MagicMock()
        mock_session = AsyncMock()

        mock_emb = AsyncMock()
        mock_emb.embed.return_value = [[0.1] * 1024]  # only the new chunk
        with (
            patch.object(
                mod,
                "get_session_factory",
                return_value=_mock_session_factory_with_session(
                    mock_session, [select_result, insert_result]
                ),
            ),
            patch.object(mod, "get_embedding_provider", return_value=mock_emb),
        ):
            count = await mod.write_chunks("doc-1", chunks)

        assert count == 1
        # Embed called with only the brand-new chunk
        assert mock_emb.embed.call_args[0][0] == ["brand new chunk"]

        # INSERT (2nd execute) carries the new chunk's hash + original index,
        # and an ON CONFLICT clause for race-safe idempotency.
        insert_sql, insert_params = mock_session.execute.call_args_list[1][0]
        assert insert_params["content_0"] == "brand new chunk"
        assert insert_params["idx_0"] == 0
        assert insert_params["hash_0"] == mod._content_hash(chunks[0])
        assert "ON CONFLICT (document_id, content_hash) DO NOTHING" in str(insert_sql)


class TestDocumentIdTraceability:
    """Chunk retrieval carries ``document_id`` through to result metadata.

    Fix for the audit finding: sources' ``document_id`` was always "" because
    vector/sparse search never selected the column and the rerank tail dropped
    it.  Both the rerank path and the skip_rerank fallback must attach it, and
    rows without the key (legacy mocks / pre-migration data) must not gain a
    ``document_id: None`` entry that serializes to the string "None".
    """

    def _patch_sources(self, monkeypatch, dense, sparse):
        from backend.service import retrieval as mod

        monkeypatch.setattr(mod, "embed_query", AsyncMock(return_value=[0.1]))
        monkeypatch.setattr(mod, "vector_search", AsyncMock(return_value=dense))
        monkeypatch.setattr(mod, "sparse_search", AsyncMock(return_value=sparse))

    @pytest.mark.asyncio
    async def test_rerank_path_attaches_document_id(self, monkeypatch) -> None:
        from backend.service import retrieval as mod

        dense = [
            {"id": "c1", "document_id": "a.py", "content": "c1",
             "meta": {"lang": "py"}, "similarity": 0.8},
        ]
        self._patch_sources(monkeypatch, dense, [])
        monkeypatch.setattr(
            "backend.service.rerank.rerank_cross_encoder",
            AsyncMock(return_value=[(0, 0.9)]),
        )

        results = await mod.retrieve_hybrid("query", top_k=5, skip_rerank=False)

        assert len(results) == 1
        assert results[0].metadata["document_id"] == "a.py"
        # Original meta keys are preserved alongside the column value.
        assert results[0].metadata["lang"] == "py"

    @pytest.mark.asyncio
    async def test_skip_rerank_path_attaches_document_id(self, monkeypatch) -> None:
        from backend.service import retrieval as mod

        dense = [
            {"id": "c1", "document_id": "a.py", "content": "c1",
             "meta": {}, "similarity": 0.8},
        ]
        self._patch_sources(monkeypatch, dense, [])

        results = await mod.retrieve_hybrid("query", top_k=5, skip_rerank=True)

        assert len(results) == 1
        assert results[0].metadata["document_id"] == "a.py"

    @pytest.mark.asyncio
    async def test_sparse_only_source_attaches_document_id(self, monkeypatch) -> None:
        """A chunk found only via sparse search still carries its document."""
        from backend.service import retrieval as mod

        sparse = [
            {"id": "c1", "document_id": "b.md", "content": "c1",
             "meta": {}, "rank": 0.9},
        ]
        self._patch_sources(monkeypatch, [], sparse)

        results = await mod.retrieve_hybrid("query", top_k=5, skip_rerank=True)

        assert len(results) == 1
        assert results[0].metadata["document_id"] == "b.md"

    @pytest.mark.asyncio
    async def test_missing_document_id_keeps_metadata_clean(self, monkeypatch) -> None:
        """Rows without a ``document_id`` key don't gain a None entry."""
        from backend.service import retrieval as mod

        dense = [{"id": "c1", "content": "c1", "meta": {}, "similarity": 0.8}]
        self._patch_sources(monkeypatch, dense, [])
        monkeypatch.setattr(
            "backend.service.rerank.rerank_cross_encoder",
            AsyncMock(return_value=[(0, 0.9)]),
        )

        results = await mod.retrieve_hybrid("query", top_k=5, skip_rerank=False)

        assert len(results) == 1
        assert "document_id" not in results[0].metadata


class TestEmbedQueryCache:
    """LRU caching on embed_query — repeated queries skip the provider."""

    @pytest.fixture(autouse=True)
    def _clean_cache(self) -> None:
        from backend.service import retrieval as mod

        mod.clear_embed_query_cache()
        yield
        mod.clear_embed_query_cache()

    def _mock_provider(self, monkeypatch):
        from backend.service import retrieval as mod

        provider = MagicMock()
        provider.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        monkeypatch.setattr(mod, "get_embedding_provider", lambda: provider)
        return provider

    @pytest.mark.asyncio
    async def test_identical_query_hits_cache(self, monkeypatch) -> None:
        """Second call with the same query must not re-embed."""
        from backend.service import retrieval as mod

        provider = self._mock_provider(monkeypatch)
        first = await mod.embed_query("pgvector 检索")
        second = await mod.embed_query("pgvector 检索")
        provider.embed.assert_awaited_once()
        assert first == [0.1, 0.2, 0.3]
        assert second == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_different_queries_embed_separately(self, monkeypatch) -> None:
        from backend.service import retrieval as mod

        provider = self._mock_provider(monkeypatch)
        await mod.embed_query("查询 A")
        await mod.embed_query("查询 B")
        assert provider.embed.await_count == 2

    @pytest.mark.asyncio
    async def test_returns_list_for_pgvector_syntax(self, monkeypatch) -> None:
        """str(vec) must yield [..] (list), not (..) (tuple) — pgvector input."""
        from backend.service import retrieval as mod

        self._mock_provider(monkeypatch)
        vec = await mod.embed_query("x")
        assert isinstance(vec, list)
        assert str(vec) == "[0.1, 0.2, 0.3]"

    @pytest.mark.asyncio
    async def test_lru_evicts_least_recently_used(self, monkeypatch) -> None:
        from backend.service import retrieval as mod

        provider = self._mock_provider(monkeypatch)
        monkeypatch.setattr(mod, "_QUERY_EMBED_CACHE_MAX", 2)

        await mod.embed_query("q1")
        await mod.embed_query("q2")
        await mod.embed_query("q3")  # evicts q1 (oldest)
        assert provider.embed.await_count == 3

        # q1's re-insertion evicts q2 (now oldest); q3 survives.
        await mod.embed_query("q1")
        assert provider.embed.await_count == 4
        await mod.embed_query("q3")  # still cached
        assert provider.embed.await_count == 4

    @pytest.mark.asyncio
    async def test_hit_reorders_lru_slot(self, monkeypatch) -> None:
        """A hit makes the key newest, so it survives a later eviction."""
        from backend.service import retrieval as mod

        provider = self._mock_provider(monkeypatch)
        monkeypatch.setattr(mod, "_QUERY_EMBED_CACHE_MAX", 2)

        await mod.embed_query("q1")
        await mod.embed_query("q2")
        await mod.embed_query("q1")  # hit → q1 newest, q2 oldest
        await mod.embed_query("q3")  # evicts q2
        assert provider.embed.await_count == 3  # q1 hit was free

        # q1 survived the eviction (the reorder paid off); q2 was evicted.
        await mod.embed_query("q1")
        assert provider.embed.await_count == 3
        await mod.embed_query("q2")  # evicted → re-embedded
        assert provider.embed.await_count == 4

    @pytest.mark.asyncio
    async def test_clear_embed_query_cache(self, monkeypatch) -> None:
        from backend.service import retrieval as mod

        provider = self._mock_provider(monkeypatch)
        await mod.embed_query("q")
        mod.clear_embed_query_cache()
        await mod.embed_query("q")
        assert provider.embed.await_count == 2
