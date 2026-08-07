"""Structured LLM output — JSON-schema enforcement + bounded retry.

Replaces the ad-hoc ``json.loads`` + silent-fallback pattern at structured
call sites.  Providers constrain the raw response to valid JSON (via
``response_format`` on OpenAI-compatible APIs, forced ``tool_use`` on
Anthropic); this module parses it, validates the shape with ``jsonschema``,
and retries transient / parse / validation failures with linear backoff.

After retries are exhausted it raises :class:`LLMStructuredError`.  Callers
then decide:
  - correctness-critical (conflict detection, entity match): propagate —
    the write/operation fails rather than storing unverified data;
  - enrichment (entity/relation extraction): catch, log at ERROR level,
    count via ``record_structured_failure``, and degrade to ``[]``.

Plain ``for`` loop instead of tenacity: smaller, no SDK-exception coupling,
and trivially testable with mocks.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from jsonschema import ValidationError, validate as jsonschema_validate

from backend.model.llm import LLMStructuredError
from backend.service.llm_service import get_llm_provider
from backend.shared.config import config

logger = logging.getLogger(__name__)


async def chat_structured(
    messages: list[dict[str, str]],
    *,
    json_schema: dict[str, Any],
    scenario: str,
    max_attempts: int | None = None,
    backoff: float | None = None,
    **kwargs,
) -> dict[str, Any] | list[Any]:
    """Call the LLM for a schema-valid JSON value, retrying on failure.

    Args:
        messages: Prompt messages (the caller is expected to already
            instruct the model to reply with JSON matching *json_schema*).
        json_schema: JSON Schema the parsed output must validate against.
        scenario: Cost-observability tag (also used for failure counting
            at the call site).
        max_attempts: Retry budget, defaulting to
            ``config.llm.structured_max_attempts``.
        backoff: Base linear-backoff seconds, defaulting to
            ``config.llm.structured_backoff``.

    Returns:
        The parsed JSON value (dict or list) that validates against
        *json_schema*.

    Raises:
        LLMStructuredError: if *max_attempts* consecutive attempts all fail
            to produce schema-valid JSON (whether malformed text, structural
            mismatch, or a transient provider error).
    """
    llm = get_llm_provider()
    attempts = max_attempts if max_attempts is not None else config.llm.structured_max_attempts
    delay = backoff if backoff is not None else config.llm.structured_backoff

    last_error: Exception | None = None
    raw: str | None = None
    for attempt in range(attempts):
        try:
            raw = await llm.chat_json(
                messages, json_schema=json_schema, scenario=scenario, **kwargs
            )
            data = json.loads(raw)
            jsonschema_validate(data, json_schema)
            return data
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            last_error = exc
            logger.warning(
                "Structured output invalid (attempt %d/%d, scenario=%s): %s — raw=%r",
                attempt + 1,
                attempts,
                scenario,
                exc,
                raw,
            )
        except LLMStructuredError:
            raise
        except Exception as exc:  # transient provider/API error → retry
            last_error = exc
            logger.warning(
                "Structured LLM call failed (attempt %d/%d, scenario=%s): %s",
                attempt + 1,
                attempts,
                scenario,
                exc,
            )

        if attempt < attempts - 1:
            await asyncio.sleep(delay * (attempt + 1))

    raise LLMStructuredError(
        f"LLM failed to produce schema-valid JSON after {attempts} attempts "
        f"(scenario={scenario}): {last_error}"
    )
