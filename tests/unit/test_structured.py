"""Tests for chat_structured — JSON-schema enforcement + bounded retry.

The provider (``llm.chat_json``) is mocked; these tests exercise parsing,
``jsonschema`` validation, retry-on-failure, and the LLMStructuredError
contract that correctness-critical call sites rely on.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from backend.model.llm import LLMStructuredError
from backend.service.structured import chat_structured
from backend.shared.resilience import CircuitOpenError

SCHEMA = {
    "type": "object",
    "required": ["ok"],
    "properties": {"ok": {"type": "boolean"}},
}


@pytest.fixture
def mock_llm(monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
    llm = AsyncMock()
    monkeypatch.setattr("backend.service.structured.get_llm_provider", lambda: llm)
    return llm


def _messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "reply as json"}]


class TestChatStructured:
    @pytest.mark.asyncio
    async def test_valid_json_returned(self, mock_llm) -> None:
        mock_llm.chat_json.return_value = json.dumps({"ok": True})
        result = await chat_structured(_messages(), json_schema=SCHEMA, scenario="test")
        assert result == {"ok": True}
        mock_llm.chat_json.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retries_then_succeeds_on_invalid_json(self, mock_llm) -> None:
        """Malformed text is retried; a later valid response is returned."""
        mock_llm.chat_json.side_effect = [
            "not json at all",
            json.dumps({"ok": True}),
        ]
        result = await chat_structured(
            _messages(), json_schema=SCHEMA, scenario="test", max_attempts=3, backoff=0
        )
        assert result == {"ok": True}
        assert mock_llm.chat_json.await_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_schema_mismatch(self, mock_llm) -> None:
        """Valid JSON that violates the schema (wrong type) is retried."""
        mock_llm.chat_json.side_effect = [
            json.dumps({"ok": "yes"}),  # string, not boolean → ValidationError
            json.dumps({"ok": True}),
        ]
        result = await chat_structured(
            _messages(), json_schema=SCHEMA, scenario="test", max_attempts=3, backoff=0
        )
        assert result == {"ok": True}
        assert mock_llm.chat_json.await_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_transient_provider_error(self, mock_llm) -> None:
        """Transient API errors (429/5xx) are retried, not swallowed."""
        mock_llm.chat_json.side_effect = [
            RuntimeError("429 rate limit"),
            json.dumps({"ok": True}),
        ]
        result = await chat_structured(
            _messages(), json_schema=SCHEMA, scenario="test", max_attempts=3, backoff=0
        )
        assert result == {"ok": True}
        assert mock_llm.chat_json.await_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_exhaustion(self, mock_llm) -> None:
        """Persistent invalid output raises LLMStructuredError, never returns bad data."""
        mock_llm.chat_json.return_value = "still not json"
        with pytest.raises(LLMStructuredError):
            await chat_structured(
                _messages(), json_schema=SCHEMA, scenario="test", max_attempts=2, backoff=0
            )
        assert mock_llm.chat_json.await_count == 2

    @pytest.mark.asyncio
    async def test_propagates_circuit_open_error_without_retry(self, mock_llm) -> None:
        """An open circuit breaker fails fast — CircuitOpenError is re-raised,
        not retried through the semantic budget (provider is already failing fast).
        """
        mock_llm.chat_json.side_effect = CircuitOpenError("breaker open")
        with pytest.raises(CircuitOpenError):
            await chat_structured(
                _messages(), json_schema=SCHEMA, scenario="test", max_attempts=3, backoff=0
            )
        assert mock_llm.chat_json.await_count == 1  # no retries burned

    @pytest.mark.asyncio
    async def test_scenario_and_schema_passed_through(self, mock_llm) -> None:
        mock_llm.chat_json.return_value = json.dumps({"ok": True})
        await chat_structured(_messages(), json_schema=SCHEMA, scenario="my_scenario")
        kwargs = mock_llm.chat_json.await_args.kwargs
        assert kwargs["scenario"] == "my_scenario"
        assert kwargs["json_schema"] == SCHEMA
