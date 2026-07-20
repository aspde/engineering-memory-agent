"""Tests for agent tool definitions — mock underlying services."""

import json
from unittest.mock import AsyncMock

import pytest

from agent.tools import (
    extract_memory_tool,
    ingest_document_tool,
    ingest_git_repo_tool,
    retrieve_chunks_tool,
    search_memories_tool,
    write_memory_tool,
)


class TestSearchMemoriesTool:
    @pytest.mark.asyncio
    async def test_returns_formatted_results(self, monkeypatch) -> None:
        from agent import tools as mod

        async def mock_query(*args, **kwargs):
            return [
                {
                    "summary": "PostgreSQL is the primary database",
                    "rerank_score": 0.95,
                    "decay_factor": 0.9,
                }
            ]

        monkeypatch.setattr(mod, "query_memories", mock_query)

        result = await search_memories_tool.ainvoke({"query": "database"})
        assert "PostgreSQL" in result
        assert "0.95" in result

    @pytest.mark.asyncio
    async def test_empty_results(self, monkeypatch) -> None:
        from agent import tools as mod

        monkeypatch.setattr(mod, "query_memories", AsyncMock(return_value=[]))
        result = await search_memories_tool.ainvoke({"query": "nothing"})
        assert "No relevant memories" in result


class TestRetrieveChunksTool:
    @pytest.mark.asyncio
    async def test_returns_formatted_results(self, monkeypatch) -> None:
        from agent import tools as mod

        from backend.service.retrieval import RetrievalResult

        async def mock_retrieve(*args, **kwargs):
            return [RetrievalResult(content="def foo(): pass", score=0.88, metadata={})]

        monkeypatch.setattr(mod, "retrieve", mock_retrieve)

        result = await retrieve_chunks_tool.ainvoke({"query": "foo function"})
        assert "def foo" in result
        assert "0.88" in result

    @pytest.mark.asyncio
    async def test_empty_results(self, monkeypatch) -> None:
        from agent import tools as mod

        monkeypatch.setattr(mod, "retrieve", AsyncMock(return_value=[]))
        result = await retrieve_chunks_tool.ainvoke({"query": "nothing"})
        assert "No relevant document chunks" in result


class TestWriteMemoryTool:
    @pytest.mark.asyncio
    async def test_inserts_and_returns_json(self, monkeypatch) -> None:
        from agent import tools as mod

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
        from agent import tools as mod

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


class TestExtractMemoryTool:
    @pytest.mark.asyncio
    async def test_extracts_and_returns_json(self, monkeypatch) -> None:
        from agent import tools as mod

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
        from agent import tools as mod

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
        from agent import tools as mod

        monkeypatch.setattr(mod, "ingest_repo", AsyncMock(return_value=[]))
        result = await ingest_git_repo_tool.ainvoke({"repo_path": "/tmp/empty"})
        assert "No commits" in result


class TestIngestDocument:
    @pytest.mark.asyncio
    async def test_returns_count(self, monkeypatch) -> None:
        from agent import tools as mod

        monkeypatch.setattr(mod, "chunk_text", lambda content, **kw: ["chunk1", "chunk2"])
        monkeypatch.setattr(mod, "write_chunks", AsyncMock(return_value=2))

        result = await ingest_document_tool.ainvoke(
            {"document_id": "test.py", "content": "print('hello')\nprint('world')"}
        )
        assert "2 chunks" in result
        assert "test.py" in result
