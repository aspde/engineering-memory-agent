"""LLM service — factory for LLM providers."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from backend.model.llm import LLMProvider
from backend.shared.config import config
from backend.shared.metrics import pop_scenario, record_usage
from backend.shared.resilience import (
    call_with_resilience,
    call_with_resilience_sync,
    circuit_breaker_guard,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """Generic provider for any OpenAI-compatible API (DeepSeek, OpenAI, etc.)."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 60,
    ) -> None:
        from openai import AsyncOpenAI, OpenAI

        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

        self._async_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        self._sync_client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )
        logger.info("LLM provider ready: %s @ %s", model, base_url)

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        scenario = pop_scenario(kwargs)
        kwargs.setdefault("temperature", self._temperature)
        kwargs.setdefault("max_tokens", self._max_tokens)

        async def _op() -> Any:
            return await self._async_client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            )

        response = await call_with_resilience("llm:openai", _op)
        record_usage(scenario, getattr(response, "usage", None))
        return response.choices[0].message.content or ""

    async def chat_raw(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, object]] | None = None,
        **kwargs,
    ) -> dict[str, object]:
        scenario = pop_scenario(kwargs)
        kwargs.setdefault("temperature", self._temperature)
        kwargs.setdefault("max_tokens", self._max_tokens)
        create_kwargs: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            **kwargs,
        }
        if tools:
            create_kwargs["tools"] = tools

        async def _op() -> Any:
            return await self._async_client.chat.completions.create(**create_kwargs)  # type: ignore[arg-type]

        response = await call_with_resilience("llm:openai", _op)
        record_usage(scenario, getattr(response, "usage", None))
        msg = response.choices[0].message
        result: dict[str, object] = {"content": msg.content or ""}

        if msg.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                }
                for tc in msg.tool_calls
            ]
        return result

    def chat_sync(self, messages: list[dict[str, str]], **kwargs) -> str:
        scenario = pop_scenario(kwargs)
        kwargs.setdefault("temperature", self._temperature)
        kwargs.setdefault("max_tokens", self._max_tokens)

        def _op() -> Any:
            return self._sync_client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            )

        response = call_with_resilience_sync("llm:openai", _op)
        record_usage(scenario, getattr(response, "usage", None))
        return response.choices[0].message.content or ""

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict | None = None,
        **kwargs,
    ) -> str:
        """Constrained JSON output via ``response_format=json_object``.

        Supported by DeepSeek and OpenAI; guarantees a valid JSON response
        shape so downstream ``jsonschema`` validation only has to fight
        structural drift, not malformed text.

        Wired to the circuit breaker only — transport retries for this path
        live in ``structured.py`` (semantic retry), so tenacity is not
        layered here.
        """
        scenario = pop_scenario(kwargs)
        kwargs.setdefault("temperature", self._temperature)
        kwargs.setdefault("max_tokens", self._max_tokens)

        async def _op() -> Any:
            return await self._async_client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                response_format={"type": "json_object"},
                **kwargs,
            )

        async with circuit_breaker_guard("llm:openai"):
            response = await _op()
        record_usage(scenario, getattr(response, "usage", None))
        return response.choices[0].message.content or ""

    async def chat_raw_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, object]] | None = None,
        **kwargs,
    ) -> AsyncIterator[dict[str, object]]:
        """True token streaming for ``chat_raw``.

        Yields ``{"type": "content", "text": <delta>}`` as the model
        generates, then a single ``{"type": "tool_calls", "tool_calls":
        [...]}`` event when the turn ends with tool calls.  ``tool_calls``
        use the same shape as :meth:`chat_raw` (``{"id", "name", "args"}``).

        The circuit breaker guards connection establishment; token
        iteration is not retried (an established stream is consumed once).
        Token usage is not recorded on this path (streamed ``usage`` is
        provider-inconsistent) — the non-streaming calls still report it.
        """
        scenario = pop_scenario(kwargs)
        kwargs.setdefault("temperature", self._temperature)
        kwargs.setdefault("max_tokens", self._max_tokens)
        create_kwargs: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }
        if tools:
            create_kwargs["tools"] = tools

        async def _op() -> Any:
            return await self._async_client.chat.completions.create(**create_kwargs)  # type: ignore[arg-type]

        async with circuit_breaker_guard("llm:openai"):
            response = await _op()

        tool_calls_map: dict[int, dict[str, str]] = {}
        tool_call_order: list[int] = []
        async for chunk in response:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = choices[0].delta
            if delta is None:
                continue
            if delta.content:
                yield {"type": "content", "text": delta.content}
            for tc in delta.tool_calls or []:
                idx = tc.index
                if idx not in tool_calls_map:
                    tool_calls_map[idx] = {"id": "", "name": "", "arguments": ""}
                    tool_call_order.append(idx)
                if tc.id:
                    tool_calls_map[idx]["id"] = tc.id
                if tc.function:
                    if tc.function.name:
                        tool_calls_map[idx]["name"] = tc.function.name
                    if tc.function.arguments:
                        tool_calls_map[idx]["arguments"] += tc.function.arguments

        if tool_calls_map:
            tool_calls: list[dict[str, object]] = []
            for idx in tool_call_order:
                info = tool_calls_map[idx]
                try:
                    args: object = json.loads(info["arguments"]) if info["arguments"] else {}
                except json.JSONDecodeError:
                    args = {"raw": info["arguments"]}
                tool_calls.append({"id": info["id"], "name": info["name"], "args": args})
            yield {"type": "tool_calls", "tool_calls": tool_calls}

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream response text token-by-token (no tools)."""
        async for event in self.chat_raw_stream(messages, **kwargs):
            if event.get("type") == "content":
                yield str(event.get("text", ""))

    @property
    def model(self) -> str:
        return self._model


class AnthropicProvider(LLMProvider):
    """Provider for Anthropic Claude API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        timeout: int = 60,
    ) -> None:
        from anthropic import Anthropic, AsyncAnthropic

        self._model = model
        self._max_tokens = max_tokens

        self._async_client = AsyncAnthropic(
            api_key=api_key,
            timeout=timeout,
        )
        self._sync_client = Anthropic(
            api_key=api_key,
            timeout=timeout,
        )
        logger.info("Anthropic provider ready: %s", model)

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        scenario = pop_scenario(kwargs)
        system, user_messages = self._split_messages(messages)
        kwargs.setdefault("max_tokens", self._max_tokens)

        async def _op() -> Any:
            return await self._async_client.messages.create(
                model=self._model,
                system=system,
                messages=user_messages,  # type: ignore[arg-type]
                **kwargs,
            )

        response = await call_with_resilience("llm:anthropic", _op)
        record_usage(scenario, getattr(response, "usage", None))
        return self._extract_text(response.content)

    async def chat_raw(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, object]] | None = None,
        **kwargs,
    ) -> dict[str, object]:
        scenario = pop_scenario(kwargs)
        system, user_messages = self._split_messages(messages)
        kwargs.setdefault("max_tokens", self._max_tokens)
        create_kwargs: dict[str, object] = {
            "model": self._model,
            "system": system,
            "messages": self._to_anthropic_messages(user_messages),
            **kwargs,
        }
        if tools:
            create_kwargs["tools"] = self._to_anthropic_tools(tools)

        async def _op() -> Any:
            return await self._async_client.messages.create(**create_kwargs)  # type: ignore[arg-type]

        response = await call_with_resilience("llm:anthropic", _op)
        record_usage(scenario, getattr(response, "usage", None))
        content_blocks: list[object] = response.content  # type: ignore[assignment]
        result: dict[str, object] = {"content": self._extract_text(content_blocks)}

        tool_calls: list[dict[str, object]] = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "args": block.get("input", {}),
                    }
                )
            elif hasattr(block, "type") and getattr(block, "type", "") == "tool_use":
                tool_calls.append(
                    {
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "args": getattr(block, "input", {}),
                    }
                )
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    def chat_sync(self, messages: list[dict[str, str]], **kwargs) -> str:
        scenario = pop_scenario(kwargs)
        system, user_messages = self._split_messages(messages)
        kwargs.setdefault("max_tokens", self._max_tokens)

        def _op() -> Any:
            return self._sync_client.messages.create(
                model=self._model,
                system=system,
                messages=user_messages,  # type: ignore[arg-type]
                **kwargs,
            )

        response = call_with_resilience_sync("llm:anthropic", _op)
        record_usage(scenario, getattr(response, "usage", None))
        return self._extract_text(response.content)

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict | None = None,
        **kwargs,
    ) -> str:
        """Constrained JSON output via a forced ``tool_use`` block.

        Anthropic has no ``response_format``; its structured-output idiom is
        a tool whose ``input_schema`` the model must fill.  Because that
        schema must be an *object* schema (our callers often want a top-level
        array), the caller's schema is wrapped in a ``{"result": <schema>}``
        envelope and unwrapped here.

        Wired to the circuit breaker only — transport retries for this path
        live in ``structured.py`` (semantic retry), so tenacity is not
        layered here.
        """
        scenario = pop_scenario(kwargs)
        system, user_messages = self._split_messages(messages)
        kwargs.setdefault("max_tokens", self._max_tokens)
        schema = json_schema or {"type": "object"}

        async def _op() -> Any:
            return await self._async_client.messages.create(
                model=self._model,
                system=system,
                messages=user_messages,  # type: ignore[arg-type]
                tools=[
                    {
                        "name": "emit_json",
                        "description": (
                            'Return the requested data as a JSON value in the "result" field.'
                        ),
                        "input_schema": {
                            "type": "object",
                            "properties": {"result": schema},
                            "required": ["result"],
                        },
                    }
                ],
                tool_choice={"type": "tool", "name": "emit_json"},
                **kwargs,
            )

        async with circuit_breaker_guard("llm:anthropic"):
            response = await _op()
        record_usage(scenario, getattr(response, "usage", None))
        for block in response.content:  # type: ignore[attr-defined]
            if getattr(block, "type", "") == "tool_use":
                return json.dumps(
                    block.input.get("result", block.input), ensure_ascii=False
                )
        return ""

    async def chat_raw_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, object]] | None = None,
        **kwargs,
    ) -> AsyncIterator[dict[str, object]]:
        """True token streaming for ``chat_raw`` (Anthropic SDK streams).

        Yields ``{"type": "content", "text": <delta>}`` as the model
        generates, then a single ``{"type": "tool_calls", "tool_calls":
        [...]}`` event when the turn ends with tool calls.  The circuit
        breaker guards the whole stream lifecycle — Anthropic's
        ``messages.stream()`` is lazy, so the HTTP connection opens during
        iteration, not at call time.

        Token usage is not recorded on this path (streamed ``usage`` is
        provider-inconsistent) — the non-streaming calls still report it.
        """
        scenario = pop_scenario(kwargs)
        system, user_messages = self._split_messages(messages)
        kwargs.setdefault("max_tokens", self._max_tokens)
        create_kwargs: dict[str, object] = {
            "model": self._model,
            "system": system,
            "messages": self._to_anthropic_messages(user_messages),
            **kwargs,
        }
        if tools:
            create_kwargs["tools"] = self._to_anthropic_tools(tools)

        tool_buf: dict[int, dict[str, object]] = {}
        async with circuit_breaker_guard("llm:anthropic"):
            async with self._async_client.messages.stream(**create_kwargs) as stream:  # type: ignore[arg-type]
                async for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if getattr(block, "type", "") == "tool_use":
                            tool_buf[event.index] = {
                                "id": block.id,
                                "name": block.name,
                                "input_parts": [],
                            }
                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield {"type": "content", "text": delta.text}
                        elif delta.type == "input_json_delta" and event.index in tool_buf:
                            parts = tool_buf[event.index]["input_parts"]
                            assert isinstance(parts, list)
                            parts.append(delta.partial_json)
                    elif event.type == "message_stop":
                        break

        if tool_buf:
            tool_calls: list[dict[str, object]] = []
            for index in sorted(tool_buf):
                info = tool_buf[index]
                raw = "".join(info["input_parts"])  # type: ignore[arg-type]
                try:
                    args: object = json.loads(raw) if raw else {}
                except json.JSONDecodeError:
                    args = {"raw": raw}
                tool_calls.append({"id": info["id"], "name": info["name"], "args": args})
            yield {"type": "tool_calls", "tool_calls": tool_calls}

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream response text token-by-token (no tools)."""
        async for event in self.chat_raw_stream(messages, **kwargs):
            if event.get("type") == "content":
                yield str(event.get("text", ""))

    @staticmethod
    def _to_anthropic_tools(tools: list[dict[str, object]]) -> list[dict[str, object]]:
        """Convert OpenAI-format tool schemas to Anthropic's ``input_schema`` shape.

        The agent layer emits OpenAI function-calling schemas
        (``{"type": "function", "function": {"name", "description",
        "parameters"}}``); Anthropic's Messages API wants
        ``{"name", "description", "input_schema"}``.  Schemas that already
        use the ``name``/``input_schema`` shape are passed through.
        """
        converted: list[dict[str, object]] = []
        for tool in tools:
            fn = tool.get("function") if isinstance(tool, dict) else None
            if isinstance(fn, dict):
                converted.append(
                    {
                        "name": str(fn.get("name", "")),
                        "description": str(fn.get("description", "")),
                        "input_schema": fn.get("parameters")
                        or {"type": "object", "properties": {}},
                    }
                )
            else:
                converted.append(tool)
        return converted

    @staticmethod
    def _to_anthropic_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-compatible message dicts to Anthropic's message shape.

        Anthropic has no ``role: "tool"`` and no ``tool_calls`` field: tool
        results are delivered as ``tool_result`` content blocks inside a
        ``user`` message, and a tool request is a ``tool_use`` content block
        on the assistant message.  Both the OpenAI wire shape produced by
        ``_messages_to_dicts`` and LangChain's native ``tool_call`` dicts are
        accepted for the assistant ``tool_calls`` list.

        Consecutive ``tool`` messages (parallel tool calls executed in one
        assistant turn) are coalesced into a single ``user`` message carrying
        all their ``tool_result`` blocks — Anthropic rejects consecutive
        same-role messages and expects one combined result message per turn.
        """
        converted: list[dict[str, Any]] = []
        pending_results: list[dict[str, Any]] = []

        def _flush_results() -> None:
            if pending_results:
                # Copy the list: appending the live object then clearing it
                # would empty the already-emitted message's content blocks.
                converted.append({"role": "user", "content": list(pending_results)})
                pending_results.clear()

        for msg in messages:
            role = msg.get("role")
            content = str(msg.get("content", "") or "")

            if role == "tool":
                pending_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": str(msg.get("tool_call_id", "")),
                        "content": content,
                    }
                )
                continue

            _flush_results()

            if role == "assistant" and msg.get("tool_calls"):
                blocks: list[dict[str, Any]] = []
                if content:
                    blocks.append({"type": "text", "text": content})
                for tc in msg["tool_calls"]:
                    if not isinstance(tc, dict):
                        continue
                    if isinstance(tc.get("function"), dict):
                        # OpenAI wire shape: {"id", "function": {"name", "arguments"}}
                        fn = tc["function"]
                        name = str(fn.get("name", ""))
                        args_raw = fn.get("arguments")
                        if isinstance(args_raw, str):
                            try:
                                args: Any = json.loads(args_raw)
                            except (json.JSONDecodeError, TypeError):
                                args = {}
                        else:
                            args = args_raw or {}
                        tool_id = str(tc.get("id", ""))
                    else:
                        # LangChain tool_call shape: {"id", "name", "args"}
                        name = str(tc.get("name", ""))
                        args = tc.get("args") or {}
                        tool_id = str(tc.get("id", ""))
                    blocks.append(
                        {"type": "tool_use", "id": tool_id, "name": name, "input": args}
                    )
                converted.append({"role": "assistant", "content": blocks})
                continue

            converted.append({"role": role, "content": content})

        _flush_results()
        return converted

    @staticmethod
    def _split_messages(
        messages: list[dict[str, str]],
    ) -> tuple[str, list[dict[str, str]]]:
        """Anthropic requires system prompt as a top-level param, not a message."""
        system = ""
        if messages and messages[0].get("role") == "system":
            system = messages[0]["content"]
            messages = messages[1:]
        return system, messages

    @staticmethod
    def _extract_text(content: list) -> str:
        """Extract text from Anthropic content blocks."""
        parts: list[str] = []
        for block in content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    @property
    def model(self) -> str:
        return self._model


_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    """Return a singleton LLM provider based on config."""
    global _provider
    if _provider is not None:
        return _provider

    if config.llm.provider == "anthropic":
        _provider = AnthropicProvider(
            api_key=config.llm.api_key,
            model=config.llm.model,
            max_tokens=config.llm.max_tokens,
            timeout=config.llm.timeout,
        )
    elif config.llm.provider in ("deepseek", "openai"):
        _provider = OpenAICompatibleProvider(
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            model=config.llm.model,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            timeout=config.llm.timeout,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm.provider}")
    return _provider
