"""Tests for agent tool definitions — mock underlying services."""

import json
from unittest.mock import AsyncMock

import pytest

from backend.agent.tools import (
    extract_memory_tool,
    ingest_document_tool,
    ingest_git_repo_tool,
    query_entity_tool,
    query_rewrite_and_search_tool,
    retrieve_chunks_tool,
    search_memories_tool,
    write_memory_tool,
)


class TestSearchMemoriesTool:
    @pytest.mark.asyncio
    async def test_returns_formatted_results(self, monkeypatch) -> None:
        from backend.agent import tools as mod

        async def mock_query(*args, **kwargs):
            return [
                {
                    "id": "mem-001",
                    "summary": "PostgreSQL is the primary database",
                    "rerank_score": 0.95,
                }
            ]

        monkeypatch.setattr(mod, "query_memories", mock_query)

        result = await search_memories_tool.ainvoke({"query": "database"})
        data = json.loads(result)
        assert "display" in data
        assert "sources" in data
        assert "PostgreSQL" in data["display"]
        assert "0.95" in data["display"]
        # The display exposes the memory short ID so the LLM can cite it inline.
        assert "memory: mem-001" in data["display"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["id"] == "mem-001"

    @pytest.mark.asyncio
    async def test_empty_results(self, monkeypatch) -> None:
        from backend.agent import tools as mod

        monkeypatch.setattr(mod, "query_memories", AsyncMock(return_value=[]))
        result = await search_memories_tool.ainvoke({"query": "nothing"})
        assert "No relevant memories" in result


class TestRetrieveChunksTool:
    @pytest.mark.asyncio
    async def test_returns_formatted_results(self, monkeypatch) -> None:
        from backend.agent import tools as mod
        from backend.service.retrieval import RetrievalResult

        async def mock_retrieve(*args, **kwargs):
            return [RetrievalResult(content="def foo(): pass", score=0.88, metadata={"document_id": "test.py"})]

        monkeypatch.setattr(mod, "retrieve_hybrid", mock_retrieve)

        result = await retrieve_chunks_tool.ainvoke({"query": "foo function"})
        data = json.loads(result)
        assert "display" in data
        assert "sources" in data
        assert "def foo" in data["display"]
        assert "0.88" in data["display"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["document_id"] == "test.py"
        assert data["sources"][0]["type"] == "chunk"

    @pytest.mark.asyncio
    async def test_empty_results(self, monkeypatch) -> None:
        from backend.agent import tools as mod

        monkeypatch.setattr(mod, "retrieve_hybrid", AsyncMock(return_value=[]))
        result = await retrieve_chunks_tool.ainvoke({"query": "nothing"})
        assert "No relevant document chunks" in result


class TestQueryRewriteAndSearchTool:
    @pytest.mark.asyncio
    async def test_returns_formatted_results(self, monkeypatch) -> None:
        from backend.service.retrieval import RetrievalResult

        async def mock_multi_query(*args, **kwargs):
            return [
                RetrievalResult(
                    content="koa-connect ctx 泄漏",
                    score=0.87,
                    metadata={"document_id": "postmortem.md"},
                )
            ]

        monkeypatch.setattr(
            "backend.service.retrieval.retrieve_multi_query", mock_multi_query
        )

        result = await query_rewrite_and_search_tool.ainvoke(
            {"query": "之前出过什么问题"}
        )
        data = json.loads(result)
        assert "Found 1 relevant chunks" in data["display"]
        assert "0.87" in data["display"]
        assert len(data["sources"]) == 1
        assert data["sources"][0]["document_id"] == "postmortem.md"
        assert data["sources"][0]["type"] == "chunk"

    @pytest.mark.asyncio
    async def test_empty_results(self, monkeypatch) -> None:
        monkeypatch.setattr(
            "backend.service.retrieval.retrieve_multi_query",
            AsyncMock(return_value=[]),
        )
        result = await query_rewrite_and_search_tool.ainvoke({"query": "nothing"})
        assert "No relevant document chunks" in result


class TestWriteMemoryTool:
    @pytest.mark.asyncio
    async def test_inserts_and_returns_json(self, monkeypatch) -> None:
        from backend.agent import tools as mod

        async def mock_write(content, source_type, metadata):
            return {"id": "abc-123", "action": "inserted", "summary": "A new memory."}

        monkeypatch.setattr(mod, "write_memory", mock_write)

        result = await write_memory_tool.ainvoke({"content": "some text"})
        data = json.loads(result)
        assert data["action"] == "inserted"
        assert data["id"] == "abc-123"

    @pytest.mark.asyncio
    async def test_conflict_returns_structured_data(self, monkeypatch) -> None:
        """When write_memory detects a conflict, the JSON result includes
        existing_id, existing_summary, and _deferred for HITL resolution."""
        from backend.agent import tools as mod

        async def mock_write(content, source_type, metadata):
            return {
                "action": "conflict",
                "summary": "EMA uses MySQL",
                "existing_id": "mem-456",
                "existing_summary": "EMA uses PostgreSQL",
                "entities": [{"name": "EMA", "type": "project"}],
                "relations": [],
                "_deferred": {
                    "extracted": {"summary": "EMA uses MySQL"},
                    "embedding": "[1.0, 2.0]",
                    "source_type": "conversation",
                    "metadata": {"conflicts_with": "mem-456"},
                },
            }

        monkeypatch.setattr(mod, "write_memory", mock_write)

        result = await write_memory_tool.ainvoke({"content": "EMA uses MySQL"})
        data = json.loads(result)
        assert data["action"] == "conflict"
        assert data["existing_id"] == "mem-456"
        assert data["existing_summary"] == "EMA uses PostgreSQL"
        assert "_deferred" in data

    @pytest.mark.asyncio
    async def test_success_records_auto_memory_throttle(self, monkeypatch) -> None:
        """A successful write records the thread in the auto-memory throttle
        table, so an auto-capture within the interval skips — the user just
        explicitly wrote."""
        from backend.agent import nodes as nodes_mod
        from backend.agent import tools as mod
        from backend.shared.config import current_thread_id

        async def mock_write(content, source_type, metadata):
            return {"id": "abc-123", "action": "inserted", "summary": "A new memory."}

        monkeypatch.setattr(mod, "write_memory", mock_write)
        monkeypatch.setattr(nodes_mod, "_auto_memory_last_write", {})

        token = current_thread_id.set("thread-tool")
        try:
            await write_memory_tool.ainvoke({"content": "some text"})
        finally:
            current_thread_id.reset(token)

        assert "thread-tool" in nodes_mod._auto_memory_last_write

    @pytest.mark.asyncio
    async def test_conflict_does_not_record_throttle(self, monkeypatch) -> None:
        """A conflict persists nothing until a human resolves it, so it must
        not throttle auto-memory — the knowledge is still up for capture."""
        from backend.agent import nodes as nodes_mod
        from backend.agent import tools as mod
        from backend.shared.config import current_thread_id

        async def mock_write(content, source_type, metadata):
            return {
                "action": "conflict",
                "summary": "EMA uses MySQL",
                "existing_id": "mem-456",
                "existing_summary": "EMA uses PostgreSQL",
                "entities": [],
                "relations": [],
                "_deferred": {},
            }

        monkeypatch.setattr(mod, "write_memory", mock_write)
        monkeypatch.setattr(nodes_mod, "_auto_memory_last_write", {})

        token = current_thread_id.set("thread-conflict")
        try:
            await write_memory_tool.ainvoke({"content": "EMA uses MySQL"})
        finally:
            current_thread_id.reset(token)

        assert "thread-conflict" not in nodes_mod._auto_memory_last_write


class TestExtractMemoryTool:
    @pytest.mark.asyncio
    async def test_extracts_and_returns_json(self, monkeypatch) -> None:
        from backend.agent import tools as mod

        async def mock_extract(content):
            return {
                "summary": "A test summary.",
                "entities": [{"name": "Python", "type": "technology"}],
                "relations": [],
            }

        monkeypatch.setattr(mod, "extract_memory", mock_extract)

        result = await extract_memory_tool.ainvoke({"content": "We use Python."})
        data = json.loads(result)
        assert data["summary"] == "A test summary."
        assert len(data["entities"]) == 1


class TestIngestGitRepo:
    @pytest.mark.asyncio
    async def test_returns_formatted_results(self, monkeypatch) -> None:
        from backend.agent import tools as mod

        async def mock_ingest(repo_path, max_commits, branch):
            return [
                {"id": "1", "action": "inserted", "summary": "feat: add login"},
                {"id": "2", "action": "inserted", "summary": "fix: null pointer"},
            ]

        monkeypatch.setattr(mod, "ingest_repo", mock_ingest)

        result = await ingest_git_repo_tool.ainvoke({"repo_path": "/tmp/repo"})
        assert "2 commits" in result
        assert "feat: add login" in result

    @pytest.mark.asyncio
    async def test_empty_repo(self, monkeypatch) -> None:
        from backend.agent import tools as mod

        monkeypatch.setattr(mod, "ingest_repo", AsyncMock(return_value=[]))
        result = await ingest_git_repo_tool.ainvoke({"repo_path": "/tmp/empty"})
        assert "No commits" in result


class TestIngestDocument:
    @pytest.mark.asyncio
    async def test_returns_count(self, monkeypatch) -> None:
        from backend.agent import tools as mod

        monkeypatch.setattr(mod, "chunk_text", lambda content, **kw: ["chunk1", "chunk2"])
        monkeypatch.setattr(mod, "write_chunks", AsyncMock(return_value=2))

        result = await ingest_document_tool.ainvoke(
            {"document_id": "test.py", "content": "print('hello')\nprint('world')"}
        )
        assert "2 chunks" in result
        assert "test.py" in result

    @pytest.mark.asyncio
    async def test_python_language_uses_chunk_code(self, monkeypatch) -> None:
        from backend.agent import tools as mod

        monkeypatch.setattr(mod, "chunk_code", lambda code, **kw: ["def foo(): pass", "class Bar: pass"])
        monkeypatch.setattr(mod, "write_chunks", AsyncMock(return_value=2))

        result = await ingest_document_tool.ainvoke(
            {"document_id": "app.py", "content": "def foo(): pass\nclass Bar: pass", "language": "python"}
        )
        assert "2 chunks" in result
        assert "app.py" in result

    @pytest.mark.asyncio
    async def test_python_language_includes_language_in_metadata(self, monkeypatch) -> None:
        from backend.agent import tools as mod

        monkeypatch.setattr(mod, "chunk_code", lambda code, **kw: ["chunk1"])
        mock_write = AsyncMock(return_value=1)
        monkeypatch.setattr(mod, "write_chunks", mock_write)

        await ingest_document_tool.ainvoke(
            {"document_id": "app.py", "content": "x = 1", "language": "python"}
        )
        # Verify the metadata passed to write_chunks includes language
        call_args = mock_write.call_args
        meta = call_args[1]["meta"] if "meta" in call_args[1] else call_args[0][2]
        assert meta["language"] == "python"

    @pytest.mark.asyncio
    async def test_default_language_uses_chunk_text(self, monkeypatch) -> None:
        from backend.agent import tools as mod

        monkeypatch.setattr(mod, "chunk_text", lambda content, **kw: ["paragraph 1"])
        monkeypatch.setattr(mod, "write_chunks", AsyncMock(return_value=1))

        result = await ingest_document_tool.ainvoke(
            {"document_id": "readme.md", "content": "# Hello\n\nWorld"}
        )
        assert "1 chunks" in result


class TestQueryEntityTool:
    @pytest.mark.asyncio
    async def test_returns_entity_profile(self, monkeypatch) -> None:
        from backend.agent import tools as mod
        from backend.agent.tools import query_entity_tool

        async def mock_get_entity(name: str):
            return {
                "id": "e-001",
                "name": "pg",
                "canonical_name": "PostgreSQL",
                "type": "technology",
                "memory_count": 5,
            }

        async def mock_get_relations(eid: str):
            return {
                "related_entities": [
                    {"id": "e-002", "name": "pgvector", "type": "technology", "memory_count": 3}
                ],
                "recent_memories": [
                    {"id": "m-001", "summary": "Using pgvector", "source_type": "conversation", "created_at": "2026-01-01T00:00:00"}
                ],
            }

        monkeypatch.setattr(mod, "get_entity_by_name", mock_get_entity)
        monkeypatch.setattr(mod, "get_entity_relations_for_tool", mock_get_relations)

        result = await query_entity_tool.ainvoke({"entity_name": "PostgreSQL"})
        data = json.loads(result)
        assert data["found"] is True
        assert data["entity"]["canonical_name"] == "PostgreSQL"
        assert len(data["related_entities"]) == 1
        assert data["related_entities"][0]["name"] == "pgvector"
        assert len(data["recent_memories"]) == 1

    @pytest.mark.asyncio
    async def test_entity_not_found(self, monkeypatch) -> None:
        from backend.agent import tools as mod
        from backend.agent.tools import query_entity_tool

        monkeypatch.setattr(mod, "get_entity_by_name", AsyncMock(return_value=None))

        result = await query_entity_tool.ainvoke({"entity_name": "NonexistentTech"})
        data = json.loads(result)
        assert data["found"] is False
        assert "no entity found" in data["message"].lower()


class TestSearchMemoriesToolWithEntities:
    @pytest.mark.asyncio
    async def test_includes_entities_in_sources(self, monkeypatch) -> None:
        from backend.agent import tools as mod
        from backend.agent.tools import search_memories_tool

        async def mock_query(*args, **kwargs):
            return [
                {
                    "id": "mem-001",
                    "summary": "PostgreSQL with pgvector",
                    "rerank_score": 0.95,
                }
            ]

        async def mock_entity_batch(ids):
            return {
                "mem-001": [
                    {"entity_id": "e-001", "canonical_name": "PostgreSQL", "type": "technology"},
                    {"entity_id": "e-002", "canonical_name": "pgvector", "type": "technology"},
                ]
            }

        monkeypatch.setattr(mod, "query_memories", mock_query)
        monkeypatch.setattr(mod, "get_memory_entities_batch", mock_entity_batch)

        result = await search_memories_tool.ainvoke({"query": "database"})
        data = json.loads(result)
        assert "entities" in data["sources"][0]
        assert len(data["sources"][0]["entities"]) == 2
        assert data["sources"][0]["entities"][0]["canonical_name"] == "PostgreSQL"
        assert "PostgreSQL" in data["display"]
        assert "pgvector" in data["display"]


class TestNotifyFeishuTool:
    """Tests for the notify_feishu notification tool."""

    @pytest.mark.asyncio
    async def test_returns_error_when_not_configured(self, monkeypatch) -> None:
        """When FEISHU_WEBHOOK_URL is empty, the tool returns ok=false."""
        from backend.agent.tools import notify_feishu_tool

        monkeypatch.setattr(
            "backend.shared.config.config.feishu_webhook_url", ""
        )

        result = await notify_feishu_tool.ainvoke({
            "message": "Hello",
        })
        data = json.loads(result)
        assert data["ok"] is False
        assert "FEISHU_WEBHOOK_URL" in data["error"]

    @pytest.mark.asyncio
    async def test_tool_schema_has_required_params(self) -> None:
        """Verify the tool has the expected parameter schema."""
        from backend.agent.tools import notify_feishu_tool

        schema = notify_feishu_tool.args_schema.model_json_schema()
        props = schema.get("properties", {})
        assert "message" in props
        assert "msg_type" in props
        assert "title" in props
        required = schema.get("required", [])
        assert "message" in required


class TestConnectorAwareness:
    """Tool descriptions and prompts should mention connector data sources."""

    def test_search_memories_tool_mentions_connectors(self):
        """The tool description should mention connector source types."""
        desc = search_memories_tool.description.lower()
        assert "pingcode" in desc
        assert "ci" in desc or "ci/cd" in desc
        assert "feishu" in desc or "飞书" in desc

    def test_query_entity_tool_mentions_connectors(self):
        """The entity tool description should mention connector-derived entities."""
        desc = query_entity_tool.description.lower()
        assert any(
            word in desc for word in ("pingcode", "ci", "feishu", "飞书", "connectors", "sources")
        )

    def test_tool_count(self):
        """Sanity check: the ALL_TOOLS roster should have 9 tools (7 core + query_rewrite + notify)."""
        from backend.agent.tools import ALL_TOOLS

        assert len(ALL_TOOLS) == 9


class TestToolParamBounds:
    """Retrieval/ingestion parameters must have schema-level bounds so a
    zero or negative top_k / max_commits never reaches the backend."""

    @pytest.mark.parametrize(
        "tool",
        [
            search_memories_tool,
            retrieve_chunks_tool,
            query_rewrite_and_search_tool,
        ],
    )
    def test_retrieval_top_k_bounds_in_schema(self, tool) -> None:
        props = tool.args_schema.model_json_schema()["properties"]
        assert props["top_k"]["minimum"] == 1, tool.name
        assert props["top_k"]["maximum"] == 20, tool.name

    def test_ingest_max_commits_bounds_in_schema(self) -> None:
        props = ingest_git_repo_tool.args_schema.model_json_schema()["properties"]
        assert props["max_commits"]["minimum"] == 1
        assert props["max_commits"]["maximum"] == 200

    @pytest.mark.asyncio
    @pytest.mark.parametrize("top_k", [0, -5])
    async def test_retrieval_rejects_out_of_range_top_k(self, monkeypatch, top_k) -> None:
        """A below-bounds top_k is rejected before any backend call."""
        from backend.agent import tools as mod

        monkeypatch.setattr(mod, "query_memories", AsyncMock())
        with pytest.raises(Exception, match="greater than or equal to 1"):
            await search_memories_tool.ainvoke({"query": "x", "top_k": top_k})
        mod.query_memories.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ingest_rejects_out_of_range_max_commits(self, monkeypatch) -> None:
        from backend.agent import tools as mod

        monkeypatch.setattr(mod, "ingest_repo", AsyncMock())
        with pytest.raises(Exception, match="greater than or equal to 1"):
            await ingest_git_repo_tool.ainvoke({"repo_path": "/x", "max_commits": 0})
        mod.ingest_repo.assert_not_awaited()


class TestLLMRerankExposureBoundary:
    """Agent tool schemas must NEVER surface ``use_llm_rerank`` to the model.

    LLM rerank costs ~2.5s per candidate; DeepSeek once enabled it on nearly
    every tool call, adding ~40s and ~46% of cost per chat round before it
    was locked out of the tool schemas (see
    ``docs/engineering/gap-remediation.md`` §3.1.1).  The retrieval functions
    keep the parameter for explicit callers (the API client and the eval
    harness), but exposing it to the model lets the model re-enable the slow
    path on its own — this test makes that regression fail CI.
    """

    def test_no_tool_exposes_use_llm_rerank(self) -> None:
        from backend.agent.tools import ALL_TOOLS

        for tool in ALL_TOOLS:
            props = tool.args_schema.model_json_schema().get("properties", {})
            assert "use_llm_rerank" not in props, (
                f"{tool.name} must not expose use_llm_rerank to the model"
            )
