"""Tests for extraction functions — mock LLM to avoid real API calls."""

import json
from unittest.mock import AsyncMock

import pytest

from backend.model.llm import LLMStructuredError
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


def _patch_provider(monkeypatch: pytest.MonkeyPatch, mock_llm: AsyncMock) -> None:
    """chat_structured resolves get_llm_provider inside structured.py,
    not in extraction.py — so the patch must target that module."""
    monkeypatch.setattr("backend.service.structured.get_llm_provider", lambda: mock_llm)


class TestExtractEntities:
    @pytest.mark.asyncio
    async def test_parses_json_list(self, monkeypatch) -> None:
        mock_llm = AsyncMock()
        entities = [{"name": "PostgreSQL", "type": "technology"}, {"name": "Alice", "type": "person"}]
        mock_llm.chat_json.return_value = json.dumps(entities)
        _patch_provider(monkeypatch, mock_llm)

        result = await extract_entities("We use PostgreSQL. Alice wrote the migration.")
        assert len(result) == 2
        assert result[0]["name"] == "PostgreSQL"

    @pytest.mark.asyncio
    async def test_persistent_failure_degrades_to_empty(self, monkeypatch) -> None:
        """Structured output that never validates → loud degradation to []."""
        import backend.service.extraction as mod
        failures = []

        async def _raise(*args, **kwargs):
            raise LLMStructuredError("no schema-valid JSON after retries")

        monkeypatch.setattr(mod, "chat_structured", _raise)
        monkeypatch.setattr(mod, "record_structured_failure", lambda s: failures.append(s))

        result = await extract_entities("some summary")
        assert result == []
        assert failures == ["extraction_entities"]


class TestExtractRelations:
    @pytest.mark.asyncio
    async def test_parses_relations(self, monkeypatch) -> None:
        mock_llm = AsyncMock()
        relations = [{"from": "PostgreSQL", "to": "pgvector", "type": "depends_on"}]
        mock_llm.chat_json.return_value = json.dumps(relations)
        _patch_provider(monkeypatch, mock_llm)

        entities = [{"name": "PostgreSQL", "type": "technology"}, {"name": "pgvector", "type": "technology"}]
        result = await extract_relations("We use pgvector on PostgreSQL.", entities)
        assert len(result) == 1
        assert result[0]["type"] == "depends_on"

    @pytest.mark.asyncio
    async def test_drops_relations_referencing_unknown_entities(self, monkeypatch) -> None:
        """Relations whose endpoints are not in the entity list are hallucinated."""
        mock_llm = AsyncMock()
        relations = [
            {"from": "PostgreSQL", "to": "pgvector", "type": "depends_on"},
            {"from": "NotAnEntity", "to": "pgvector", "type": "relates_to"},
        ]
        mock_llm.chat_json.return_value = json.dumps(relations)
        _patch_provider(monkeypatch, mock_llm)

        entities = [{"name": "PostgreSQL", "type": "technology"}, {"name": "pgvector", "type": "technology"}]
        result = await extract_relations("summary", entities)
        assert len(result) == 1
        assert result[0]["from"] == "PostgreSQL"

    @pytest.mark.asyncio
    async def test_less_than_two_entities_returns_empty(self, monkeypatch) -> None:
        result = await extract_relations("summary", [{"name": "only one"}])
        assert result == []

    @pytest.mark.asyncio
    async def test_persistent_failure_degrades_to_empty(self, monkeypatch) -> None:
        import backend.service.extraction as mod
        failures = []

        async def _raise(*args, **kwargs):
            raise LLMStructuredError("no schema-valid JSON after retries")

        monkeypatch.setattr(mod, "chat_structured", _raise)
        monkeypatch.setattr(mod, "record_structured_failure", lambda s: failures.append(s))

        entities = [{"name": "A", "type": "concept"}, {"name": "B", "type": "concept"}]
        result = await extract_relations("summary", entities)
        assert result == []
        assert failures == ["extraction_relations"]


class TestExtractMemory:
    @pytest.mark.asyncio
    async def test_orchestration_returns_dict(self, monkeypatch) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = ["A summary."]  # extract_summary
        mock_llm.chat_json.side_effect = [  # extract_entities → extract_relations
            json.dumps([
                {"name": "PG", "type": "technology"},
                {"name": "vector", "type": "technology"},
            ]),
            json.dumps([{"from": "PG", "to": "vector", "type": "depends_on"}]),
        ]

        import backend.service.extraction as mod
        monkeypatch.setattr(mod, "get_llm_provider", lambda: mock_llm)  # summary
        _patch_provider(monkeypatch, mock_llm)  # entities / relations

        result = await extract_memory("Some content")
        assert result["summary"] == "A summary."
        assert len(result["entities"]) == 2
        assert len(result["relations"]) == 1
