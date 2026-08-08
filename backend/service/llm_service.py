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
    CircuitOpenError,
    call_with_resilience,
    call_with_resilience_sync,
    circuit_breaker_guard,
    is_retryable,
)
from backend.service.usage import begin_call, record_call

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """Generic provider for any OpenAI-compatible API (DeepSeek, OpenAI, etc.).

    Prompt caching is provider-side and automatic: OpenAI-compatible APIs
    (OpenAI, DeepSeek) cache repeated input prefixes server-side once the
    request clears the provider's minimum token threshold — no API parameter
    is involved.  ``cache_control`` is an Anthropic-only mechanism, handled in
    :class:`AnthropicProvider`.
    """

    PROVIDER_NAME = "openai-compatible"

    def _record(self, ctx: dict, **kwargs) -> None:
        """Record one observed call into the usage buffer (see usage.py)."""
        record_call(
            ctx, model=self.model, provider=self.PROVIDER_NAME, **kwargs
        )

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
        self._base_url = base_url
        # Per-endpoint breaker name: a fallback provider (different base_url
        # and/or model) must not share the primary's breaker, or an open
        # primary breaker would fast-fail the healthy fallback too.
        self._breaker_name = f"llm:openai:{base_url}|{model}"

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
        ctx = begin_call(messages)
        kwargs.setdefault("temperature", self._temperature)
        kwargs.setdefault("max_tokens", self._max_tokens)

        async def _op() -> Any:
            return await self._async_client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            )

        try:
            response = await call_with_resilience(self._breaker_name, _op)
        except Exception as exc:
            self._record(
                ctx, scenario=scenario, status="error", error=str(exc)
            )
            raise
        usage = getattr(response, "usage", None)
        record_usage(scenario, usage)
        text = response.choices[0].message.content or ""
        self._record(
            ctx, scenario=scenario, usage=usage, response_text=text
        )
        return text

    async def chat_raw(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, object]] | None = None,
        **kwargs,
    ) -> dict[str, object]:
        scenario = pop_scenario(kwargs)
        ctx = begin_call(messages)
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

        try:
            response = await call_with_resilience(self._breaker_name, _op)
        except Exception as exc:
            self._record(
                ctx, scenario=scenario, status="error", error=str(exc)
            )
            raise
        usage = getattr(response, "usage", None)
        record_usage(scenario, usage)
        msg = response.choices[0].message
        result: dict[str, object] = {"content": msg.content or ""}

        if msg.tool_calls:
            # Same JSON guard as the streaming path (chat_raw_stream): a
            # malformed arguments blob must not fail the whole call.  Empty
            # arguments mean no args; unparsable text is surfaced as
            # {"raw": ...} so the caller still sees what the model produced.
            tool_calls: list[dict[str, object]] = []
            for tc in msg.tool_calls:
                try:
                    args: object = (
                        json.loads(tc.function.arguments)
                        if tc.function.arguments
                        else {}
                    )
                except json.JSONDecodeError:
                    args = {"raw": tc.function.arguments}
                tool_calls.append(
                    {"id": tc.id, "name": tc.function.name, "args": args}
                )
            result["tool_calls"] = tool_calls
        self._record(
            ctx, scenario=scenario, usage=usage,
            response_text=str(msg.content or ""),
        )
        return result

    def chat_sync(self, messages: list[dict[str, str]], **kwargs) -> str:
        scenario = pop_scenario(kwargs)
        ctx = begin_call(messages)
        kwargs.setdefault("temperature", self._temperature)
        kwargs.setdefault("max_tokens", self._max_tokens)

        def _op() -> Any:
            return self._sync_client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                **kwargs,
            )

        try:
            response = call_with_resilience_sync(self._breaker_name, _op)
        except Exception as exc:
            self._record(
                ctx, scenario=scenario, status="error", error=str(exc)
            )
            raise
        usage = getattr(response, "usage", None)
        record_usage(scenario, usage)
        text = response.choices[0].message.content or ""
        self._record(
            ctx, scenario=scenario, usage=usage, response_text=text
        )
        return text

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
        ctx = begin_call(messages)
        kwargs.setdefault("temperature", self._temperature)
        kwargs.setdefault("max_tokens", self._max_tokens)

        async def _op() -> Any:
            return await self._async_client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                response_format={"type": "json_object"},
                **kwargs,
            )

        try:
            async with circuit_breaker_guard(self._breaker_name):
                response = await _op()
        except Exception as exc:
            self._record(
                ctx, scenario=scenario, status="error", error=str(exc)
            )
            raise
        usage = getattr(response, "usage", None)
        record_usage(scenario, usage)
        text = response.choices[0].message.content or ""
        self._record(
            ctx, scenario=scenario, usage=usage, response_text=text
        )
        return text

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

        Transport retry + the circuit breaker guard connection establishment;
        token iteration is not retried (an established stream is consumed
        once).
        Token usage is recorded when the provider returns it: the OpenAI
        SDK surfaces usage on the stream object (or on the final chunk) once
        the stream is fully consumed.  Providers whose stream carries no
        usage are silently skipped — accounting never fails the stream.
        """
        scenario = pop_scenario(kwargs)
        ctx = begin_call(messages)
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

        content_parts: list[str] = []
        tool_calls_map: dict[int, dict[str, str]] = {}
        tool_call_order: list[int] = []
        last_chunk: Any = None
        try:
            # Same transport retry + circuit breaker as the non-streaming
            # paths: a 429 / 5xx / timeout at create time is retried by
            # tenacity before the stream is established.  Once the first
            # token has been consumed the stream is not retried — the client
            # already saw the prefix.
            response = await call_with_resilience(self._breaker_name, _op)

            async for chunk in response:
                last_chunk = chunk
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = choices[0].delta
                if delta is None:
                    continue
                if delta.content:
                    content_parts.append(delta.content)
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

            # After full consumption the SDK surfaces usage on the stream object
            # (some versions) or on the final chunk (most versions); either way a
            # missing usage just means this provider's stream didn't report it.
            usage = getattr(response, "usage", None)
            if usage is None and last_chunk is not None:
                usage = getattr(last_chunk, "usage", None)
            if usage is not None:
                record_usage(scenario, usage)
            self._record(
                ctx, scenario=scenario, usage=usage,
                response_text="".join(content_parts),
            )

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
        except Exception as exc:
            self._record(
                ctx, scenario=scenario, status="error", error=str(exc)
            )
            raise

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

    PROVIDER_NAME = "anthropic"

    def _record(self, ctx: dict, **kwargs) -> None:
        """Record one observed call into the usage buffer (see usage.py)."""
        record_call(
            ctx, model=self.model, provider=self.PROVIDER_NAME, **kwargs
        )

    def __init__(
        self,
        api_key: str,
        model: str,
        max_tokens: int = 4096,
        timeout: int = 60,
        prompt_caching: bool = True,
    ) -> None:
        from anthropic import Anthropic, AsyncAnthropic

        self._model = model
        self._max_tokens = max_tokens
        self._prompt_caching = prompt_caching
        # Per-instance breaker name (model distinguishes a same-account
        # fallback from the primary so they don't share one breaker).
        self._breaker_name = f"llm:anthropic:{model}"

        self._async_client = AsyncAnthropic(
            api_key=api_key,
            timeout=timeout,
        )
        self._sync_client = Anthropic(
            api_key=api_key,
            timeout=timeout,
        )
        logger.info(
            "Anthropic provider ready: %s (prompt_caching=%s)",
            model,
            prompt_caching,
        )

    def _maybe_cache_system(
        self, system: str
    ) -> str | list[dict[str, object]]:
        """Return *system* as a content block with a cache breakpoint when
        prompt caching is enabled; the bare string otherwise.

        The persona system prompt is the stable prefix every agent call
        repeats, and Anthropic requires ``system`` as a list of blocks to
        carry ``cache_control``.
        """
        if not self._prompt_caching or not system:
            return system
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    def _maybe_cache_tools(
        self, tools: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Mark the last tool schema with a cache breakpoint when prompt
        caching is enabled.

        Anthropic caches the whole prefix up to a ``cache_control`` marker,
        so a single marker on the last tool covers system + every tool
        schema.
        """
        if not self._prompt_caching or not tools:
            return tools
        out = [dict(t) for t in tools]
        out[-1] = {**out[-1], "cache_control": {"type": "ephemeral"}}
        return out

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        scenario = pop_scenario(kwargs)
        ctx = begin_call(messages)
        system, user_messages = self._split_messages(messages)
        kwargs.setdefault("max_tokens", self._max_tokens)

        async def _op() -> Any:
            return await self._async_client.messages.create(
                model=self._model,
                system=self._maybe_cache_system(system),
                messages=user_messages,  # type: ignore[arg-type]
                **kwargs,
            )

        try:
            response = await call_with_resilience(self._breaker_name, _op)
        except Exception as exc:
            self._record(
                ctx, scenario=scenario, status="error", error=str(exc)
            )
            raise
        usage = getattr(response, "usage", None)
        record_usage(scenario, usage)
        text = self._extract_text(response.content)
        self._record(
            ctx, scenario=scenario, usage=usage, response_text=text
        )
        return text

    async def chat_raw(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, object]] | None = None,
        **kwargs,
    ) -> dict[str, object]:
        scenario = pop_scenario(kwargs)
        ctx = begin_call(messages)
        system, user_messages = self._split_messages(messages)
        kwargs.setdefault("max_tokens", self._max_tokens)
        create_kwargs: dict[str, object] = {
            "model": self._model,
            "system": self._maybe_cache_system(system),
            "messages": self._to_anthropic_messages(user_messages),
            **kwargs,
        }
        if tools:
            create_kwargs["tools"] = self._maybe_cache_tools(
                self._to_anthropic_tools(tools)
            )

        async def _op() -> Any:
            return await self._async_client.messages.create(**create_kwargs)  # type: ignore[arg-type]

        try:
            response = await call_with_resilience(self._breaker_name, _op)
        except Exception as exc:
            self._record(
                ctx, scenario=scenario, status="error", error=str(exc)
            )
            raise
        usage = getattr(response, "usage", None)
        record_usage(scenario, usage)
        content_blocks: list[object] = response.content  # type: ignore[assignment]
        text = self._extract_text(content_blocks)
        result: dict[str, object] = {"content": text}

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
        self._record(
            ctx, scenario=scenario, usage=usage, response_text=text
        )
        return result

    def chat_sync(self, messages: list[dict[str, str]], **kwargs) -> str:
        scenario = pop_scenario(kwargs)
        ctx = begin_call(messages)
        system, user_messages = self._split_messages(messages)
        kwargs.setdefault("max_tokens", self._max_tokens)

        def _op() -> Any:
            return self._sync_client.messages.create(
                model=self._model,
                system=self._maybe_cache_system(system),
                messages=user_messages,  # type: ignore[arg-type]
                **kwargs,
            )

        try:
            response = call_with_resilience_sync(self._breaker_name, _op)
        except Exception as exc:
            self._record(
                ctx, scenario=scenario, status="error", error=str(exc)
            )
            raise
        usage = getattr(response, "usage", None)
        record_usage(scenario, usage)
        text = self._extract_text(response.content)
        self._record(
            ctx, scenario=scenario, usage=usage, response_text=text
        )
        return text

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
        ctx = begin_call(messages)
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

        try:
            async with circuit_breaker_guard(self._breaker_name):
                response = await _op()
        except Exception as exc:
            self._record(
                ctx, scenario=scenario, status="error", error=str(exc)
            )
            raise
        usage = getattr(response, "usage", None)
        record_usage(scenario, usage)
        for block in response.content:  # type: ignore[attr-defined]
            if getattr(block, "type", "") == "tool_use":
                text = json.dumps(
                    block.input.get("result", block.input), ensure_ascii=False
                )
                self._record(
                    ctx, scenario=scenario, usage=usage, response_text=text
                )
                return text
        self._record(
            ctx, scenario=scenario, usage=usage, response_text=""
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

        Token usage is recorded after the stream completes: Anthropic's
        stream object accumulates the final ``Message`` (whose ``usage``
        carries ``input_tokens``/``output_tokens``) as events are consumed.
        Providers that omit usage are silently skipped — accounting never
        fails the stream.
        """
        scenario = pop_scenario(kwargs)
        ctx = begin_call(messages)
        system, user_messages = self._split_messages(messages)
        kwargs.setdefault("max_tokens", self._max_tokens)
        create_kwargs: dict[str, object] = {
            "model": self._model,
            "system": self._maybe_cache_system(system),
            "messages": self._to_anthropic_messages(user_messages),
            **kwargs,
        }
        if tools:
            create_kwargs["tools"] = self._maybe_cache_tools(
                self._to_anthropic_tools(tools)
            )

        content_parts: list[str] = []
        tool_buf: dict[int, dict[str, object]] = {}
        usage: Any = None
        try:
            async with circuit_breaker_guard(self._breaker_name):
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
                                content_parts.append(delta.text)
                                yield {"type": "content", "text": delta.text}
                            elif delta.type == "input_json_delta" and event.index in tool_buf:
                                parts = tool_buf[event.index]["input_parts"]
                                assert isinstance(parts, list)
                                parts.append(delta.partial_json)
                        elif event.type == "message_stop":
                            break

                    # The final message snapshot (populated by ``message_stop``)
                    # carries the token usage.  Best-effort: a stream that never
                    # materialised the snapshot (or a provider that omits usage)
                    # must not fail the stream.
                    try:
                        usage = getattr(stream.current_message_snapshot, "usage", None)
                    except Exception:
                        usage = None
                    if usage is not None:
                        record_usage(scenario, usage)

            self._record(
                ctx, scenario=scenario, usage=usage,
                response_text="".join(content_parts),
            )

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
        except Exception as exc:
            self._record(
                ctx, scenario=scenario, status="error", error=str(exc)
            )
            raise

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


def _should_failover(exc: Exception) -> bool:
    """True when a primary-provider failure warrants retrying on the fallback.

    Retryable transport errors (429 / >=500 / timeouts) and an open circuit
    breaker both mean "the provider is having trouble" — a different provider
    may well succeed.  Non-retryable 4xx errors are request problems the
    fallback would hit identically, so they propagate untouched.
    """
    return is_retryable(exc) or isinstance(exc, CircuitOpenError)


class FallbackLLMProvider(LLMProvider):
    """Route calls to a secondary provider when the primary is unhealthy.

    The primary already retries and circuit-breaks internally
    (``resilience.py``); this wrapper adds cross-provider failover.  When the
    primary's call fails with a *retryable* error after its retries are
    exhausted, or its circuit breaker is open, the call is retried once
    against the fallback.

    Streaming is failover-safe too: if the primary's stream raises a
    retryable error *before the first event* (connection establishment), the
    fallback's stream is used instead.  Once any token has been yielded the
    stream is consumed — mid-stream failures are not retried (the client
    already saw the prefix).

    Off by default: only constructed when ``LLM_FALLBACK_PROVIDER`` is set.
    """

    def __init__(self, primary: LLMProvider, fallback: LLMProvider) -> None:
        self._primary = primary
        self._fallback = fallback
        logger.warning(
            "LLM failover active: primary=%s (%s) -> fallback=%s (%s)",
            primary.model,
            getattr(primary, "PROVIDER_NAME", "?"),
            fallback.model,
            getattr(fallback, "PROVIDER_NAME", "?"),
        )

    @property
    def model(self) -> str:
        return f"{self._primary.model}|{self._fallback.model}"

    async def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        try:
            return await self._primary.chat(messages, **kwargs)
        except Exception as exc:
            if not _should_failover(exc):
                raise
            logger.warning(
                "Primary LLM failed (%s) — failing over to %s",
                exc, self._fallback.model,
            )
            return await self._fallback.chat(messages, **kwargs)

    async def chat_raw(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, object]] | None = None,
        **kwargs,
    ) -> dict[str, object]:
        try:
            return await self._primary.chat_raw(messages, tools=tools, **kwargs)
        except Exception as exc:
            if not _should_failover(exc):
                raise
            logger.warning(
                "Primary LLM failed (%s) — failing over to %s",
                exc, self._fallback.model,
            )
            return await self._fallback.chat_raw(messages, tools=tools, **kwargs)

    def chat_sync(self, messages: list[dict[str, str]], **kwargs) -> str:
        try:
            return self._primary.chat_sync(messages, **kwargs)
        except Exception as exc:
            if not _should_failover(exc):
                raise
            logger.warning(
                "Primary LLM failed (%s) — failing over to %s",
                exc, self._fallback.model,
            )
            return self._fallback.chat_sync(messages, **kwargs)

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        json_schema: dict | None = None,
        **kwargs,
    ) -> str:
        try:
            return await self._primary.chat_json(
                messages, json_schema=json_schema, **kwargs
            )
        except Exception as exc:
            if not _should_failover(exc):
                raise
            logger.warning(
                "Primary LLM failed (%s) — failing over to %s",
                exc, self._fallback.model,
            )
            return await self._fallback.chat_json(
                messages, json_schema=json_schema, **kwargs
            )

    async def chat_raw_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, object]] | None = None,
        **kwargs,
    ) -> AsyncIterator[dict[str, object]]:
        primary_gen = self._primary.chat_raw_stream(messages, tools=tools, **kwargs)
        try:
            first = await anext(primary_gen)
        except StopAsyncIteration:
            return
        except Exception as exc:
            if not _should_failover(exc):
                raise
            logger.warning(
                "Primary LLM stream failed before first token (%s) — "
                "failing over to %s",
                exc, self._fallback.model,
            )
            async for event in self._fallback.chat_raw_stream(
                messages, tools=tools, **kwargs
            ):
                yield event
            return
        yield first
        async for event in primary_gen:
            yield event

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        **kwargs,
    ) -> AsyncIterator[str]:
        async for event in self.chat_raw_stream(messages, **kwargs):
            if event.get("type") == "content":
                yield str(event.get("text", ""))


def _build_provider(
    provider: str,
    api_key: str,
    base_url: str,
    model: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    timeout: int = 60,
    prompt_caching: bool = True,
) -> LLMProvider:
    """Construct one provider from explicit settings (primary or fallback)."""
    if provider == "anthropic":
        return AnthropicProvider(
            api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            prompt_caching=prompt_caching,
        )
    if provider in ("deepseek", "openai"):
        return OpenAICompatibleProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")


_provider: LLMProvider | None = None


def primary_breaker_name() -> str:
    """Circuit-breaker name for the configured primary LLM provider.

    Mirrors the per-instance names the provider classes compute in ``__init__``
    (endpoint | model) so ``/health`` can report the primary's breaker state
    without reaching into provider internals.  Keeping the derivation here and
    in the providers in lockstep is deliberate — the health endpoint must read
    the *same* breaker the provider guards with.
    """
    if config.llm.provider == "anthropic":
        return f"llm:anthropic:{config.llm.model}"
    return f"llm:openai:{config.llm.base_url}|{config.llm.model}"


def get_llm_provider() -> LLMProvider:
    """Return a singleton LLM provider based on config.

    When ``LLM_FALLBACK_PROVIDER`` is set, returns a :class:
    ``FallbackLLMProvider`` wrapping the primary and the configured fallback;
    otherwise the primary provider alone.
    """
    global _provider
    if _provider is not None:
        return _provider

    llm = config.llm
    _provider = _build_provider(
        llm.provider,
        llm.api_key,
        llm.base_url,
        llm.model,
        temperature=llm.temperature,
        max_tokens=llm.max_tokens,
        timeout=llm.timeout,
        prompt_caching=llm.prompt_caching_enabled,
    )
    if llm.fallback_provider:
        fallback = _build_provider(
            llm.fallback_provider,
            llm.fallback_api_key,
            llm.fallback_base_url,
            llm.fallback_model,
            max_tokens=llm.fallback_max_tokens,
            timeout=llm.fallback_timeout,
        )
        _provider = FallbackLLMProvider(_provider, fallback)
    return _provider
