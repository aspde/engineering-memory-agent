"""Tests for extraction functions — mock LLM to avoid real API calls."""

import json
from unittest.mock import AsyncMock

import pytest

from backend.service.extraction import (
    extract_entities,
    extract_memory,
    extract_relations,
    extract_summary,
)


class TestExtractSummary:
    @pytest.mark.asyncio
    async def test_returns_string(self, monkeypatch) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = "A concise summary."

        import backend.service.extraction as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_llm)

        result = await extract_summary("Some content about PostgreSQL migrations.")
        assert result == "A concise summary."
        mock_llm.chat.assert_called_once()


class TestExtractEntities:
    @pytest.mark.asyncio
    async def test_parses_json_list(self, monkeypatch) -> None:
        mock_llm = AsyncMock()
        entities = [{"name": "PostgreSQL", "type": "technology"}, {"name": "Alice", "type": "person"}]
        mock_llm.chat.return_value = json.dumps(entities)

        import backend.service.extraction as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_llm)

        result = await extract_entities("We use PostgreSQL. Alice wrote the migration.")
        assert len(result) == 2
        assert result[0]["name"] == "PostgreSQL"

    @pytest.mark.asyncio
    async def test_invalid_json_returns_empty(self, monkeypatch) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = "not json"

        import backend.service.extraction as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_llm)

        result = await extract_entities("some summary")
        assert result == []


class TestExtractRelations:
    @pytest.mark.asyncio
    async def test_parses_relations(self, monkeypatch) -> None:
        mock_llm = AsyncMock()
        relations = [{"from": "PostgreSQL", "to": "pgvector", "type": "depends_on"}]
        mock_llm.chat.return_value = json.dumps(relations)

        import backend.service.extraction as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_llm)

        entities = [{"name": "PostgreSQL", "type": "technology"}, {"name": "pgvector", "type": "technology"}]
        result = await extract_relations("We use pgvector on PostgreSQL.", entities)
        assert len(result) == 1
        assert result[0]["type"] == "depends_on"

    @pytest.mark.asyncio
    async def test_less_than_two_entities_returns_empty(self, monkeypatch) -> None:
        result = await extract_relations("summary", [{"name": "only one"}])
        assert result == []

    @pytest.mark.asyncio
    async def test_invalid_json_returns_empty(self, monkeypatch) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = "garbage"

        import backend.service.extraction as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_llm)

        entities = [{"name": "A", "type": "concept"}, {"name": "B", "type": "concept"}]
        result = await extract_relations("summary", entities)
        assert result == []


class TestExtractMemory:
    @pytest.mark.asyncio
    async def test_orchestration_returns_dict(self, monkeypatch) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = [
            "A summary.",
            json.dumps([{"name": "PG", "type": "technology"}, {"name": "vector", "type": "technology"}]),
            json.dumps([{"from": "PG", "to": "vector", "type": "depends_on"}]),
        ]

        import backend.service.extraction as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_llm)

        result = await extract_memory("Some content")
        assert result["summary"] == "A summary."
        assert len(result["entities"]) == 2
        assert len(result["relations"]) == 1
