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
        """Valid JSON that violates the schema (wrong type) is retried — and
        the retry carries a corrective user message instead of a blind replay
        of the same prompt."""
        mock_llm.chat_json.side_effect = [
            json.dumps({"ok": "yes"}),  # string, not boolean → ValidationError
            json.dumps({"ok": True}),
        ]
        result = await chat_structured(
            _messages(), json_schema=SCHEMA, scenario="test", max_attempts=3, backoff=0
        )
        assert result == {"ok": True}
        assert mock_llm.chat_json.await_count == 2
        retry_msgs = mock_llm.chat_json.await_args.args[0]
        assert len(retry_msgs) == 2  # original prompt + one feedback message
        assert retry_msgs[1]["role"] == "user"
        # Validation message + JSON path of the offending field are fed back
        assert "'yes' is not of type 'boolean'" in retry_msgs[1]["content"]
        assert "$.ok" in retry_msgs[1]["content"]
        assert '{"ok": "yes"}' in retry_msgs[1]["content"]  # offending raw snippet

    @pytest.mark.asyncio
    async def test_retry_feedback_does_not_mutate_caller_messages(
        self, mock_llm
    ) -> None:
        """The feedback message is appended to a copy — the caller's original
        ``messages`` list must stay untouched."""
        msgs = _messages()
        before = list(msgs)
        mock_llm.chat_json.side_effect = [
            "not json at all",
            json.dumps({"ok": True}),
        ]
        await chat_structured(
            msgs, json_schema=SCHEMA, scenario="test", max_attempts=3, backoff=0
        )
        assert msgs == before
        # The provider call saw the original plus one feedback message.
        assert len(mock_llm.chat_json.await_args.args[0]) == 2

    @pytest.mark.asyncio
    async def test_multiple_failures_accumulate_feedback(self, mock_llm) -> None:
        """Each parse/validation failure appends one corrective message, so the
        model sees every prior mistake on the last attempt."""
        mock_llm.chat_json.side_effect = [
            "not json at all",
            json.dumps({"ok": "yes"}),
            json.dumps({"ok": True}),
        ]
        result = await chat_structured(
            _messages(), json_schema=SCHEMA, scenario="test", max_attempts=3, backoff=0
        )
        assert result == {"ok": True}
        msgs = mock_llm.chat_json.await_args.args[0]
        assert len(msgs) == 3  # original + feedback for each failed attempt
        assert all(m["role"] == "user" for m in msgs[1:])

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

    @pytest.mark.asyncio
    async def test_defaults_to_structured_temperature(self, mock_llm) -> None:
        """Structured output defaults to a low, deterministic temperature."""
        from backend.shared.config import config

        mock_llm.chat_json.return_value = json.dumps({"ok": True})
        await chat_structured(_messages(), json_schema=SCHEMA, scenario="test")
        kwargs = mock_llm.chat_json.await_args.kwargs
        assert kwargs["temperature"] == config.llm.structured_temperature

    @pytest.mark.asyncio
    async def test_caller_temperature_overrides_default(self, mock_llm) -> None:
        mock_llm.chat_json.return_value = json.dumps({"ok": True})
        await chat_structured(
            _messages(), json_schema=SCHEMA, scenario="test", temperature=0.7
        )
        assert mock_llm.chat_json.await_args.kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_custom_provider_overrides_default(self, monkeypatch) -> None:
        """A caller-supplied provider is used instead of the get_llm_provider
        singleton — the eval's LLM-as-judge injects get_judge_provider() here.
        """
        injected = AsyncMock()
        injected.chat_json.return_value = json.dumps({"ok": True})
        default = AsyncMock()
        default.chat_json.return_value = json.dumps({"ok": True})
        monkeypatch.setattr(
            "backend.service.structured.get_llm_provider", lambda: default
        )

        result = await chat_structured(
            _messages(), json_schema=SCHEMA, scenario="test", provider=injected
        )

        assert result == {"ok": True}
        injected.chat_json.assert_awaited_once()
        default.chat_json.assert_not_awaited()
