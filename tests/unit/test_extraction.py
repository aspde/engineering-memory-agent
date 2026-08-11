"""Tests for extraction functions — mock LLM to avoid real API calls."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.model.llm import LLMStructuredError
from backend.service.extraction import (
    extract_entities,
    extract_memory,
    extract_relations,
    extract_summary,
)
from backend.shared.resilience import CircuitOpenError


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
    """Patch the LLM at BOTH entry points the new extraction flow touches:
    ``extraction.get_llm_provider`` (the function-calling channel resolves
    the provider there) and ``structured.get_llm_provider`` (which
    ``chat_structured`` resolves internally)."""
    monkeypatch.setattr("backend.service.extraction.get_llm_provider", lambda: mock_llm)
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

        async def _raise(*args, **kwargs):
            raise LLMStructuredError("no schema-valid JSON after retries")

        # A provider without PROVIDER_NAME skips the function-calling channel
        # and lands straight on chat_structured (patched to raise below).
        mock_inc = MagicMock()
        monkeypatch.setattr(mod, "inc_structured_failures", mock_inc)
        monkeypatch.setattr(mod, "get_llm_provider", lambda: MagicMock())
        monkeypatch.setattr(mod, "chat_structured", _raise)

        result = await extract_entities("some summary")
        assert result == []
        mock_inc.assert_called_once_with("extraction_entities")

    @pytest.mark.asyncio
    async def test_circuit_open_degrades_to_empty(self, monkeypatch) -> None:
        """Circuit breaker open → chat_structured fails fast with
        CircuitOpenError; enrichment must degrade to [] so the memory write
        proceeds instead of crashing write_memory."""
        import backend.service.extraction as mod

        async def _raise(*args, **kwargs):
            raise CircuitOpenError("Circuit breaker 'x' is open")

        mock_inc = MagicMock()
        monkeypatch.setattr(mod, "inc_structured_failures", mock_inc)
        monkeypatch.setattr(mod, "get_llm_provider", lambda: MagicMock())
        monkeypatch.setattr(mod, "chat_structured", _raise)

        result = await extract_entities("some summary")
        assert result == []
        mock_inc.assert_called_once_with("extraction_entities")


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

        async def _raise(*args, **kwargs):
            raise LLMStructuredError("no schema-valid JSON after retries")

        mock_inc = MagicMock()
        monkeypatch.setattr(mod, "inc_structured_failures", mock_inc)
        monkeypatch.setattr(mod, "get_llm_provider", lambda: MagicMock())
        monkeypatch.setattr(mod, "chat_structured", _raise)

        entities = [{"name": "A", "type": "concept"}, {"name": "B", "type": "concept"}]
        result = await extract_relations("summary", entities)
        assert result == []
        mock_inc.assert_called_once_with("extraction_relations")

    @pytest.mark.asyncio
    async def test_circuit_open_degrades_to_empty(self, monkeypatch) -> None:
        """Circuit breaker open → chat_structured fails fast with
        CircuitOpenError; relation extraction must degrade to [] so the memory
        write proceeds instead of crashing write_memory."""
        import backend.service.extraction as mod

        async def _raise(*args, **kwargs):
            raise CircuitOpenError("Circuit breaker 'x' is open")

        mock_inc = MagicMock()
        monkeypatch.setattr(mod, "inc_structured_failures", mock_inc)
        monkeypatch.setattr(mod, "get_llm_provider", lambda: MagicMock())
        monkeypatch.setattr(mod, "chat_structured", _raise)

        entities = [{"name": "A", "type": "concept"}, {"name": "B", "type": "concept"}]
        result = await extract_relations("summary", entities)
        assert result == []
        mock_inc.assert_called_once_with("extraction_relations")


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


class TestFunctionCallingChannel:
    """The dedicated extraction function tool is the preferred path on
    OpenAI-compatible providers (enum constraints at generation time);
    ``chat_structured`` is the graceful fallback.  Anthropic skips the tool
    channel — its ``chat_json`` already forces a schema-constrained
    ``tool_use`` block."""

    @staticmethod
    def _openai_provider(chat_raw_result: dict, chat_json_result: str = "[]"):
        provider = MagicMock()
        provider.PROVIDER_NAME = "openai-compatible"
        provider.chat_raw = AsyncMock(return_value=chat_raw_result)
        provider.chat_json = AsyncMock(return_value=chat_json_result)
        return provider

    @staticmethod
    def _patch_provider(monkeypatch: pytest.MonkeyPatch, provider: MagicMock) -> None:
        monkeypatch.setattr("backend.service.extraction.get_llm_provider", lambda: provider)
        monkeypatch.setattr("backend.service.structured.get_llm_provider", lambda: provider)

    @pytest.mark.asyncio
    async def test_entities_prefer_function_call(self, monkeypatch) -> None:
        """The tool's enum-constrained args are returned directly — no
        chat_json / schema-validation round-trip."""
        provider = self._openai_provider(
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "extract_entities",
                        "args": {
                            "entities": [
                                {"name": "PostgreSQL", "type": "technology"},
                                {"name": "Alice", "type": "person"},
                            ]
                        },
                    }
                ],
            }
        )
        self._patch_provider(monkeypatch, provider)

        result = await extract_entities("We use PostgreSQL. Alice wrote the migration.")

        assert len(result) == 2
        assert result[0] == {"name": "PostgreSQL", "type": "technology"}
        provider.chat_raw.assert_awaited_once()
        provider.chat_json.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_entities_fallback_when_no_tool_call(self, monkeypatch) -> None:
        """Model returns content instead of a tool call → chat_structured."""
        provider = self._openai_provider(
            {"content": "[]", "tool_calls": []},
            chat_json_result=json.dumps(
                [{"name": "PostgreSQL", "type": "technology"}]
            ),
        )
        self._patch_provider(monkeypatch, provider)

        result = await extract_entities("We use PostgreSQL.")

        assert result == [{"name": "PostgreSQL", "type": "technology"}]
        provider.chat_json.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_anthropic_skips_function_calling(self, monkeypatch) -> None:
        """Anthropic's chat_json already forces a schema-constrained tool_use
        block — the second tool would be a wasted call, so chat_raw must
        never run."""
        provider = self._openai_provider({"content": "", "tool_calls": []})
        provider.PROVIDER_NAME = "anthropic"
        provider.chat_json = AsyncMock(
            return_value=json.dumps([{"name": "PostgreSQL", "type": "technology"}])
        )
        self._patch_provider(monkeypatch, provider)

        result = await extract_entities("We use PostgreSQL.")

        assert result == [{"name": "PostgreSQL", "type": "technology"}]
        provider.chat_raw.assert_not_awaited()
        provider.chat_json.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_relations_function_call_endpoint_filtered(self, monkeypatch) -> None:
        """Relation tool args pass through the endpoint filter — a hallucinated
        endpoint is dropped exactly like the chat_structured path."""
        provider = self._openai_provider(
            {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "name": "extract_relations",
                        "args": {
                            "relations": [
                                {"from": "PostgreSQL", "to": "pgvector", "type": "depends_on"},
                                {"from": "Ghost", "to": "pgvector", "type": "relates_to"},
                            ]
                        },
                    }
                ],
            }
        )
        self._patch_provider(monkeypatch, provider)
        entities = [
            {"name": "PostgreSQL", "type": "technology"},
            {"name": "pgvector", "type": "technology"},
        ]

        result = await extract_relations("We use pgvector on PostgreSQL.", entities)

        assert result == [{"from": "PostgreSQL", "to": "pgvector", "type": "depends_on"}]
        provider.chat_raw.assert_awaited_once()
        provider.chat_json.assert_not_awaited()
