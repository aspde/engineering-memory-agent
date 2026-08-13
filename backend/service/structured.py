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
  - enrichment (entity/relation extraction): catch, log at ERROR level, and
    degrade to ``[]``.

Plain ``for`` loop instead of tenacity: smaller, no SDK-exception coupling,
and trivially testable with mocks.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from jsonschema import ValidationError
from jsonschema import validate as jsonschema_validate

from backend.model.llm import LLMProvider, LLMStructuredError
from backend.service.llm_service import get_llm_provider
from backend.shared.config import config
from backend.shared.resilience import CircuitOpenError

logger = logging.getLogger(__name__)


async def chat_structured(
    messages: list[dict[str, str]],
    *,
    json_schema: dict[str, Any],
    scenario: str,
    max_attempts: int | None = None,
    backoff: float | None = None,
    provider: LLMProvider | None = None,
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
        provider: Provider to call, overriding the default singleton.  The
            eval's LLM-as-judge passes ``get_judge_provider()`` so verdicts
            run on an independent model.  ``None`` (the default) keeps the
            primary ``get_llm_provider()`` behaviour.

    Returns:
        The parsed JSON value (dict or list) that validates against
        *json_schema*.

    Raises:
        LLMStructuredError: if *max_attempts* consecutive attempts all fail
            to produce schema-valid JSON (whether malformed text, structural
            mismatch, or a transient provider error).
    """
    llm = provider or get_llm_provider()
    attempts = max_attempts if max_attempts is not None else config.llm.structured_max_attempts
    delay = backoff if backoff is not None else config.llm.structured_backoff
    # Structured output should be deterministic — default to the low
    # structured temperature unless the caller overrides per-call.
    kwargs.setdefault("temperature", config.llm.structured_temperature)

    last_error: Exception | None = None
    raw: str | None = None
    prompt = list(messages)
    for attempt in range(attempts):
        try:
            raw = await llm.chat_json(
                prompt, json_schema=json_schema, scenario=scenario, **kwargs
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
            # Blindly replaying the same prompt at the low structured
            # temperature tends to reproduce the same malformed output.  Feed
            # the failure back so the next attempt has a real signal.  Copy
            # before appending — the caller's original ``messages`` list must
            # not be mutated.
            prompt = list(prompt) + [_feedback_message(exc, raw)]
        except LLMStructuredError:
            raise
        except CircuitOpenError:
            # Circuit breaker is open — fail fast, don't burn the semantic
            # retry budget on a provider that is already failing fast.
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


def _feedback_message(exc: BaseException, raw: str | None) -> dict[str, str]:
    """Build a corrective user message describing the last structured failure.

    For a ``ValidationError`` the leaf message plus its JSON path (e.g.
    ``'yes' is not of type 'boolean' (at $.ok)``) tells the model exactly
    which field was wrong; other parse errors carry their message.  The raw
    output is truncated so a long garbage blob doesn't bloat the retry prompt.
    """
    if isinstance(exc, ValidationError):
        path = getattr(exc, "json_path", None)
        detail = exc.message + (f" (at {path})" if path else "")
    else:
        detail = str(exc)
    snippet = (raw or "")[:200]
    return {
        "role": "user",
        "content": (
            "你上一次的输出不符合要求的 JSON Schema，请修正后重新输出。"
            "只返回符合 schema 的 JSON 值，不要包含任何其他文字。\n"
            f"校验错误：{detail}\n"
            f"上次输出（截断）：{snippet!r}"
        ),
    }
