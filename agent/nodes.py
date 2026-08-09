"""Agent graph nodes — call_llm_node, check_approval_node, and generate_final_node.

Plain functions with no class wrappers, matching the project's
service-layer conventions.  Each node receives ``AgentState`` and
returns a partial state dict or ``Command`` for dynamic routing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.config import get_stream_writer
from langgraph.types import Command, interrupt

from agent.state import AgentState
from backend.service.extraction import extract_memory
from backend.service.llm_service import get_llm_provider
from backend.service.memory import resolve_conflict, write_memory
from backend.service.prompts import get_prompt

logger = logging.getLogger(__name__)


# ── Helper: message conversion ───────────────────────────────────────


def _stream_writer():
    """Return the LangGraph custom-stream writer, or a no-op outside a run.

    ``get_stream_writer()`` raises outside a graph execution context (e.g.
    when a node is invoked directly in a unit test).  Inside a run it
    forwards token deltas to ``stream_mode="custom"`` consumers; outside,
    streaming is simply disabled.
    """
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _payload: None


# Tool schemas are serialised from each tool's pydantic args schema.  The
# roster is fixed at graph-build time, so the serialised OpenAI function-
# calling schemas are cached per tool-name tuple and reused — one
# ``model_json_schema()`` per tool per process instead of per LLM turn.
_tool_schema_cache: dict[tuple[str, ...], list[dict[str, Any]]] = {}
_tool_schema_cache_lock = threading.Lock()


def _to_openai_tools(tools: list) -> list[dict[str, Any]]:
    """Convert LangChain tool objects to OpenAI function-calling schemas.

    Serialisation is cached by tool-name tuple: the roster is fixed for the
    process lifetime, so repeated agent turns reuse the same schemas instead
    of re-running ``model_json_schema()`` every call.  A shallow copy of the
    list is returned so callers cannot mutate the cache.
    """
    key = tuple(t.name for t in tools)
    with _tool_schema_cache_lock:
        cached = _tool_schema_cache.get(key)
        if cached is not None:
            return list(cached)

    schemas: list[dict[str, Any]] = []
    for t in tools:
        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
            },
        }
        if t.args_schema and hasattr(t.args_schema, "model_json_schema"):
            try:
                json_schema = t.args_schema.model_json_schema()
                schema["function"]["parameters"] = json_schema
            except Exception:
                logger.warning(
                    "Failed to serialize tool schema for %s, falling back to empty parameters",
                    t.name,
                )
                schema["function"]["parameters"] = {"type": "object", "properties": {}}
        else:
            schema["function"]["parameters"] = {"type": "object", "properties": {}}
        schemas.append(schema)

    with _tool_schema_cache_lock:
        _tool_schema_cache[key] = schemas
    return list(schemas)


# An AIMessage appended when ``call_llm`` hits a provider error is a terminal
# error stub — not real assistant output.  It is marked so downstream history
# building (tool-selection dicts, synthesis prompts, compaction transcripts)
# never re-sends it to the model as assistant text on a later turn; it still
# stays in the checkpoint so the thread history / client shows the error.
_LLM_ERROR_MARKER = "__ema_llm_error"


def _is_llm_error_message(message: BaseMessage) -> bool:
    """True for the marked error stub appended after a failed LLM call."""
    return isinstance(message, AIMessage) and bool(
        getattr(message, "additional_kwargs", {}).get(_LLM_ERROR_MARKER)
    )


def _messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, object]]:
    """Convert LangChain messages to OpenAI-compatible dicts.

    Preserves ``tool_calls`` on AIMessages and ``tool_call_id`` on
    ToolMessages so the LLM can track the ReAct conversation loop.

    Orphaned tool_calls (no ToolMessage response) are stripped to
    prevent OpenAI 400 errors when resuming an interrupted conversation.
    """
    # Collect all tool_call_ids that have a ToolMessage response
    responded_ids: set[str] = set()
    for m in messages:
        if isinstance(m, ToolMessage) and m.tool_call_id:
            responded_ids.add(m.tool_call_id)

    dicts: list[dict[str, object]] = []
    # tool_call ids actually emitted on retained assistant messages.  A
    # ToolMessage whose tool-call AIMessage was windowed/compacted out must
    # not be emitted — OpenAI-compatible APIs reject a ``role=tool`` message
    # whose ``tool_call_id`` has no preceding assistant ``tool_calls``.
    emitted_tool_call_ids: set[str] = set()
    for m in messages:
        if _is_llm_error_message(m):
            # A marked error stub from a previous failed LLM call must not be
            # resent to the model as assistant output.
            continue
        # 1. Determine role
        if isinstance(m, SystemMessage):
            role = "system"
        elif isinstance(m, AIMessage):
            role = "assistant"
        elif isinstance(m, ToolMessage):
            role = "tool"
        elif isinstance(m, HumanMessage):
            role = "user"
        else:
            logger.debug("Skipping unknown message type: %s", type(m).__name__)
            continue

        # 2. Extract content
        content: str = ""
        if isinstance(m.content, str):
            content = m.content
        elif isinstance(m.content, list):
            parts: list[str] = []
            for block in m.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif hasattr(block, "text"):
                    parts.append(str(block.text))  # type: ignore[union-attr]
            content = " ".join(parts)

        # 3. Bound tool-result size — full search output is too much context.
        if role == "tool":
            content = _truncate_tool_content(content)

        entry: dict[str, object] = {"role": role, "content": content or ""}

        # 3. Preserve tool_calls on assistant messages (OpenAI API requirement)
        #    but only those that have a corresponding ToolMessage response —
        #    otherwise OpenAI rejects with "insufficient tool messages" 400.
        if isinstance(m, AIMessage) and m.tool_calls:
            answered = [
                tc for tc in m.tool_calls
                if tc.get("id") in responded_ids
            ]
            if answered:
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"], ensure_ascii=False),
                        },
                    }
                    for tc in answered
                ]
                emitted_tool_call_ids.update(tc["id"] for tc in answered)
            elif not content.strip():
                # Orphaned tool_calls with no content — supply a placeholder
                # so the API sees a valid assistant message.
                entry["content"] = "(tool call was interrupted)"

        # 4. Preserve tool_call_id on tool messages (OpenAI API requirement),
        #    but only when the tool-call AIMessage survived windowing.
        if isinstance(m, ToolMessage) and m.tool_call_id:
            if m.tool_call_id not in emitted_tool_call_ids:
                # The parent assistant message (with this tool_call) was
                # windowed or compacted out — emitting the result would make
                # the API reject the whole turn, so drop the orphaned result.
                logger.debug(
                    "Dropping orphaned ToolMessage (tool_call %s not in window)",
                    m.tool_call_id,
                )
                continue
            entry["tool_call_id"] = m.tool_call_id

        dicts.append(entry)
    return dicts


def _has_tool_results_this_turn(messages: list[BaseMessage]) -> bool:
    """True if a ToolMessage appeared since the most recent HumanMessage.

    A tool result in the current turn means the final-answer synthesis must
    account for it (the synthesis prompt folds tool output into the system
    context); a plain-chat turn has none, so the last ``call_llm`` output is
    already a complete answer.  Only the *current* turn counts — tool
    results from earlier turns in the same thread don't force synthesis.
    """
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return False
        if isinstance(m, ToolMessage):
            return True
    return False


def _is_new_user_turn(messages: list[BaseMessage]) -> bool:
    """True when the most recent message is a fresh HumanMessage.

    ``step_count`` bounds the ReAct loop within ONE user turn, but the
    checkpointer persists it across turns — without a reset, a thread whose
    first turn exhausted ``MAX_AGENT_STEPS`` would route every later turn's
    first ``call_llm`` straight to ``generate_final`` (tools silently
    disabled).  A new turn is exactly when the latest message is a
    HumanMessage; loopbacks within a turn end with a ToolMessage (tool
    result, approval rejection, conflict-resolution note), so they never
    match — those continue the current turn's count.
    """
    return bool(messages) and isinstance(messages[-1], HumanMessage)


# ── Context bounding ───────────────────────────────────────────────
# A long-lived thread must not resend unbounded history plus every full
# ToolMessage on each turn.  The window is sized by a *token budget* (see
# ``_estimate_tokens``) rather than a fixed message count, and tool-result
# content is truncated per message.
_MAX_TOOL_CONTENT_CHARS = 800  # per-ToolMessage content cap

# Token estimation: tiktoken (o200k_base) when available — a real BPE count,
# far closer to what the OpenAI-compatible / Anthropic providers charge than
# a char heuristic — with the CJK-aware heuristic as the offline fallback.
# tiktoken fetches its encoding file once and caches it; a process that could
# not reach the blob store at first use stays on the heuristic for its life.
_ESTIMATE_ENCODING = "o200k_base"

_tokenizer: Any | None = None
_tokenizer_attempted = False


def _get_tokenizer() -> Any | None:
    """Lazily load the tiktoken encoding, at most once.  None when unavailable."""
    global _tokenizer, _tokenizer_attempted
    if _tokenizer is not None or _tokenizer_attempted:
        return _tokenizer
    _tokenizer_attempted = True
    try:
        import tiktoken

        _tokenizer = tiktoken.get_encoding(_ESTIMATE_ENCODING)
        logger.info("Token estimation: tiktoken %s active", _ESTIMATE_ENCODING)
    except Exception:
        _tokenizer = None
        logger.warning(
            "tiktoken unavailable — falling back to heuristic token estimate",
            exc_info=True,
        )
    return _tokenizer


def reset_token_estimator() -> None:
    """Drop the cached tokenizer and the load-attempt flag (tests / config)."""
    global _tokenizer, _tokenizer_attempted
    _tokenizer = None
    _tokenizer_attempted = False


def _estimate_tokens(text: str) -> int:
    """Token count for *text* — tiktoken when available, else the heuristic."""
    enc = _get_tokenizer()
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:
            logger.warning(
                "tiktoken encode failed — using heuristic estimate", exc_info=True
            )
    return _heuristic_tokens(text)


def _heuristic_tokens(text: str) -> int:
    """Rough CJK-aware fallback estimate — no tokenizer dependency.

    Non-ASCII (CJK) characters cost ~1 token each; ASCII letters/digits and
    whitespace run ~4 chars per token; ASCII punctuation/symbols — the
    token-dense part of code and JSON — are weighted ~2 chars per token so a
    symbols-heavy tool result never under-budgets.  Used when tiktoken is
    unavailable, and as the deterministic reference in tests.  Over-approximates
    rather than overflows the model's real context.
    """
    if not text:
        return 0
    non_ascii = 0
    ascii_alnum = 0
    ascii_sym = 0
    for ch in text:
        if ord(ch) > 0x7F:
            non_ascii += 1
        elif ch.isalnum() or ch.isspace():
            ascii_alnum += 1
        else:
            ascii_sym += 1
    return non_ascii + (ascii_alnum + 3) // 4 + (ascii_sym + 1) // 2


def _message_tokens(message: BaseMessage) -> int:
    """Estimated tokens a message will cost the LLM context.

    ToolMessage content is truncated before it is sent (see
    ``_truncate_tool_content``), so the estimate uses the truncated length —
    windowing must budget for what the LLM actually receives.
    """
    text = _message_text(message)
    if isinstance(message, ToolMessage):
        text = _truncate_tool_content(text)
    return _estimate_tokens(text)


def _context_budget() -> int:
    """Token budget for the agent context window, from config."""
    from backend.shared.config import config

    return max(config.context_token_budget, 1)


def _window_messages(
    messages: list[BaseMessage], max_tokens: int | None = None
) -> list[BaseMessage]:
    """Keep the newest non-system messages that fit the *max_tokens* budget.

    System prompts are pinned to the front; history is retained from the
    newest message backwards until the token budget is exhausted.  Long
    messages shrink the window and short ones widen it — the budget, not a
    fixed message count, is what bounds context.  The newest message always
    survives even if it alone exceeds the budget.  Only what gets *sent to
    the LLM* is windowed — nodes that inspect the full state (plain-chat
    shortcut, approval/conflict gates) use the raw list, so routing decisions
    are unaffected.
    """
    if max_tokens is None:
        max_tokens = _context_budget()
    system = [m for m in messages if isinstance(m, SystemMessage)]
    rest = [m for m in messages if not isinstance(m, SystemMessage)]
    if not rest:
        return messages

    tail: list[BaseMessage] = []
    used = 0
    for m in reversed(rest):
        t = _message_tokens(m)
        if tail and used + t > max_tokens:
            break
        tail.append(m)
        used += t
    retained = list(reversed(tail))

    if len(retained) == len(rest):
        return messages  # everything fits — return the caller's list unchanged
    return system + retained


# ── Conversation compaction (B4) ─────────────────────────────────
# Long threads previously dropped everything outside the window.  When
# CONVERSATION_COMPACTION_ENABLED=true, the overflow is folded into one
# running-summary SystemMessage (a single LLM call) so the retained tail
# keeps its conversational context.  The summary is later coalesced into
# the persona system by ``_merge_system_messages`` so both OpenAI-compatible
# and Anthropic providers see exactly one system message.  Default on.


# Cap the compaction call's input: an oversized overflow (a long thread's
# entire early history) must not blow past the compaction call's own budget.
_COMPACTION_TRANSCRIPT_CHARS = 12000

# ── Compaction memoization ───────────────────────────────────────────
# A tool turn bounds the conversation twice — ``call_llm_node`` (for the
# tool-selection call) and ``generate_final_node`` (for the synthesis call)
# — over nearly identical message lists with the SAME overflow prefix.  Each
# previously paid its own compaction LLM call and could produce a different,
# non-deterministic summary.  Memoize the summary on the exact transcript
# (the sole input the summarisation LLM sees, plus the prompt version) so the
# second call reuses the first's output.  Bounded LRU; eviction drops the
# oldest entry.  Failures are not cached — a transient LLM error retries next
# turn instead of poisoning the cache with an empty summary.
#
# The lock (``_compaction_cache_lock``) guards both the completed-cache and
# the in-flight map.  ``_compaction_inflight`` holds a key -> future for a
# summary currently being generated: a concurrent bound of the same overflow
# awaits that future instead of firing a second LLM call.  All critical
# sections are non-blocking dict ops — no ``await`` happens under the lock,
# so a joining coroutine can never stall the event loop waiting on a leader
# that needs the same lock to finish.
_compaction_cache: dict[tuple[str, str], str] = {}
_compaction_cache_lock = threading.Lock()
_COMPACTION_CACHE_MAX = 32
_compaction_inflight: dict[tuple[str, str], "asyncio.Future[str]"] = {}


def reset_compaction_cache() -> None:
    """Drop cached compaction summaries and in-flight calls (tests / debugging)."""
    with _compaction_cache_lock:
        _compaction_cache.clear()
        _compaction_inflight.clear()


# Cap per-ToolMessage content in the compaction transcript — a single huge
# search result would otherwise crowd out the rest of the overflow prefix
# (the transcript has a hard char cap below).
_TRANSCRIPT_TOOL_CHAR_CAP = 200


def _tool_display_text(message: ToolMessage) -> str:
    """The display text of a tool result, unwrapped from its JSON envelope.

    Tools return ``{"display": ..., "sources": [...]}`` JSON; parsing the
    envelope *before* truncating (the same ordering generate_final_node uses)
    avoids cutting the raw JSON into a form the summariser can't read.
    """
    raw = _message_text(message)
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "display" in parsed:
            return str(parsed["display"])
    except (json.JSONDecodeError, TypeError):
        pass
    return raw


def _overflow_transcript(messages: list[BaseMessage]) -> str:
    """The exact transcript *messages* collapse to for the compaction prompt.

    Human/assistant turns carry their raw text; tool results are included
    (display-unwrapped, truncated) so a running summary of an early
    retrieval turn keeps the context those tools surfaced — without them the
    summary would lose every memory/doc chunk windowed out of the tail.
    """
    lines: list[str] = []
    for m in messages:
        if _is_llm_error_message(m):
            continue  # a failed-call error stub is not real assistant output
        if isinstance(m, HumanMessage):
            lines.append(f"user: {_message_text(m)}")
        elif isinstance(m, AIMessage):
            lines.append(f"assistant: {_message_text(m)}")
        elif isinstance(m, ToolMessage):
            name = getattr(m, "name", "") or "tool"
            content = _truncate_tool_content(
                _tool_display_text(m), limit=_TRANSCRIPT_TOOL_CHAR_CAP
            )
            lines.append(f"tool ({name}): {content}")
    transcript = "\n".join(lines)
    if len(transcript) > _COMPACTION_TRANSCRIPT_CHARS:
        transcript = transcript[: _COMPACTION_TRANSCRIPT_CHARS] + "\n…[truncated]"
    return transcript


async def _summarize_overflow(
    messages: list[BaseMessage], transcript: str | None = None
) -> str:
    """Collapse *messages* into a one-line running summary (one LLM call).

    The transcript is truncated before the call so a huge overflow can't
    exceed the compaction call's own context budget.  Fails safe: any error
    returns ``""`` so the caller falls back to the existing truncation
    behaviour.
    """
    try:
        provider = get_llm_provider()
        if transcript is None:
            transcript = _overflow_transcript(messages)
        version, prompt = get_prompt("agent.compaction")
        logger.info(
            "Compacting %d early messages (%d chars) — prompt agent.compaction v%s",
            len(messages),
            len(transcript),
            version,
        )
        summary = await provider.chat(
            [{"role": "user", "content": prompt.format(transcript=transcript)}],
            scenario="conversation_compaction",
            temperature=0.3,
        )
        return summary.strip()
    except Exception:
        logger.exception("Conversation compaction failed, falling back to truncation")
        return ""


async def _memoized_summarize_overflow(messages: list[BaseMessage]) -> str:
    """Summarise *messages*, reusing a cached summary for the same transcript.

    See the module note above the cache — this makes the two bounds in a tool
    turn agree on one summary and costs at most one compaction LLM call per
    distinct overflow.  When two bounds race on the same overflow, the loser
    awaits the winner's in-flight future (``_compaction_inflight``) instead
    of firing a second LLM call.

    No ``await`` happens under ``_compaction_cache_lock``: joining a future
    or running the summariser could stall the event loop (the leader needs
    that same lock in its ``finally`` to publish).  The lock only guards the
    non-blocking dict lookups/inserts, so a coroutine that found the future
    is always the leader and a coroutine that found an existing one joins it.
    """
    version, _ = get_prompt("agent.compaction")
    transcript = _overflow_transcript(messages)
    key = (version, transcript)
    with _compaction_cache_lock:
        cached = _compaction_cache.get(key)
        if cached is not None:
            return cached
        inflight = _compaction_inflight.get(key)
        if inflight is None:
            # We are the leader — register our future under the lock so a
            # concurrent bound joins it, then do the summarising outside.
            inflight = asyncio.get_running_loop().create_future()
            _compaction_inflight[key] = inflight
            leader = True
        else:
            leader = False

    if not leader:
        # Joiner — await the leader's future outside the lock.  A leader that
        # failed surfaces as an exception here; fails safe like the summariser
        # itself (no summary cached, caller falls back to truncation) rather
        # than parking the caller on a dead future.
        try:
            return await inflight
        except asyncio.CancelledError:
            # Our own cancellation must propagate — the caller's
            # ``asyncio.timeout`` (AGENT_TIMEOUT) may have cancelled us while
            # we waited on a slow leader.  Swallowing it would let the node
            # run past its deadline and only fail on the next await.
            raise
        except BaseException:
            return ""

    # ── Leader path ──────────────────────────────────────────────────
    try:
        summary = await _summarize_overflow(messages, transcript=transcript)
    except BaseException:
        # Let waiters resolve too — a cancelled/errored leader must not leave
        # them parked on a future that never resolves.  An empty result keeps
        # the joiner's behaviour identical to a summariser that returned ""
        # (nothing is cached); the leader re-raises the original exception so
        # cancellation/failure still propagates to its own caller.
        if not inflight.done():
            inflight.set_result("")
        raise
    else:
        if not inflight.done():
            inflight.set_result(summary)
        if summary:
            with _compaction_cache_lock:
                _compaction_cache[key] = summary
                if len(_compaction_cache) > _COMPACTION_CACHE_MAX:
                    _compaction_cache.popitem(last=False)
        return summary
    finally:
        with _compaction_cache_lock:
            if _compaction_inflight.get(key) is inflight:
                _compaction_inflight.pop(key, None)


def _split_overflow(
    messages: list[BaseMessage], tail_budget: int
) -> tuple[list[BaseMessage], list[BaseMessage]]:
    """Split non-system history into (overflow, tail) by token budget.

    The oldest messages become *overflow* (candidates for compaction); the
    retained *tail* is the newest suffix that fits *tail_budget*.  At least
    the newest message always stays in the tail, even if it alone exceeds
    the budget.
    """
    tail: list[BaseMessage] = []
    used = 0
    for m in reversed(messages):
        t = _message_tokens(m)
        if tail and used + t > tail_budget:
            break
        tail.append(m)
        used += t
    tail = list(reversed(tail))
    overflow = messages[: len(messages) - len(tail)]
    return overflow, tail


async def _maybe_compact(
    messages: list[BaseMessage], max_tokens: int | None = None
) -> list[BaseMessage]:
    """Fold history older than the token budget into a running-summary SystemMessage.

    Only active when ``config.conversation_compaction_enabled`` is set
    (default on).  When the non-system history's estimated token count
    exceeds *max_tokens*, the overflow prefix (the portion that can't fit
    alongside a retained tail) is summarised in one LLM call and prepended
    as a ``SystemMessage`` so the retained tail keeps its conversational
    context.  The tail is budgeted to ~60% of the window so the summary fits
    inside the same total budget.  On any failure it returns *messages*
    unchanged and the caller's windowing still truncates — compaction never
    loses more context than the existing behaviour.
    """
    from backend.shared.config import config

    if max_tokens is None:
        max_tokens = _context_budget()
    if not config.conversation_compaction_enabled:
        return messages
    system = [m for m in messages if isinstance(m, SystemMessage)]
    rest = [m for m in messages if not isinstance(m, SystemMessage)]
    if sum(_message_tokens(m) for m in rest) <= max_tokens:
        return messages
    # Reserve ~60% of the window for the retained tail; the running summary
    # gets the rest.  If even the tail can't fit (a single oversized message),
    # the split still keeps the newest message and compacts everything older.
    tail_budget = max(int(max_tokens * 0.6), 1)
    overflow, tail = _split_overflow(rest, tail_budget)
    if not overflow:
        return messages
    summary = await _memoized_summarize_overflow(overflow)
    if not summary:
        return messages
    # Pinned system messages stay first, then the running summary, then the
    # retained tail.  The summary may coexist with the persona system in the
    # LangChain message list — ``call_llm_node`` coalesces all system messages
    # into one before serialization, so Anthropic's single-top-level-system
    # constraint is met (see ``_merge_system_messages``).
    return system + [SystemMessage(content=summary)] + tail


def _merge_system_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Coalesce multiple SystemMessages into a single leading one.

    Compaction (B4) injects its running summary as its own SystemMessage,
    which can sit beside the pinned persona system.  Anthropic's Messages
    API accepts exactly one top-level ``system`` and ``_split_messages``
    lifts only the first ``role=system`` entry, so a second one would be
    emitted as a message — which Anthropic rejects.  OpenAI-compatible
    providers tolerate several, but joining them is semantically identical
    and keeps both paths to a single system message.

    The first system entry is the persona (instructions); every later one is
    conversation-derived content (the compaction running summary), so it is
    fenced as ``<summary>`` data — the same treatment ``generate_final_node``
    gives it — keeping it out of the executable instruction section.

    Messages with at most one system message (the common case) are returned
    unchanged; otherwise all system content is joined (in order) into one
    ``SystemMessage`` pinned at the front.
    """
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
    if len(system_msgs) <= 1:
        return messages
    parts = [str(system_msgs[0].content or "").strip()]
    for m in system_msgs[1:]:
        content = str(m.content or "").strip()
        if content:
            parts.append(f"<summary>\n{content}\n</summary>")
    merged = SystemMessage(content="\n\n".join(parts))
    return [merged] + [m for m in messages if not isinstance(m, SystemMessage)]


async def _bounded_messages(
    messages: list[BaseMessage], max_tokens: int | None = None
) -> list[BaseMessage]:
    """Compact (when enabled) then window *messages* for the LLM.

    With compaction disabled this is exactly ``_window_messages`` — the
    default behaviour is unchanged.
    """
    compacted = await _maybe_compact(messages, max_tokens)
    return _window_messages(compacted, max_tokens)


def _truncate_tool_content(text: str, limit: int = _MAX_TOOL_CONTENT_CHARS) -> str:
    """Cap a ToolMessage's content before it is resent to the LLM.

    Search/retrieval results can be thousands of characters; the model only
    needs the relevant head.  The truncation marker keeps the model aware the
    result was longer than shown.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + "\n…[truncated]"


# ── Retrieved-context tagging (B2) ───────────────────────────────
# Tool results folded into the final system prompt come from the knowledge
# base and must be framed as untrusted data.  Each item is wrapped in a
# fixed marker tag so the LLM can distinguish retrieved content from its own
# instructions (see the agent.system template's isolation declaration).
_CONTEXT_DOC_TOOLS: frozenset[str] = frozenset({
    "retrieve_chunks_tool",
    "query_rewrite_and_search_tool",
    "ingest_document_tool",
})


def _context_tag(tool_name: str) -> str:
    """Retrieval-source marker: ``doc`` for document/chunk tools, else ``memory``."""
    return "doc" if tool_name in _CONTEXT_DOC_TOOLS else "memory"


def _wrap_context_item(tool_name: str, content: str) -> str:
    """Wrap one retrieved context item in its fixed source marker tag."""
    tag = _context_tag(tool_name)
    return f'<{tag} source="{tool_name}">\n{content}\n</{tag}>'


# ── Automatic knowledge capture (B3) ──────────────────────────────
# When AUTO_MEMORY_ENABLED=true (default), substantive user turns are written
# to the memory store automatically (unless the agent already wrote this turn
# via write_memory_tool).  Set AUTO_MEMORY_ENABLED=false to restore explicit
# write-on-request behaviour.
_AUTO_MEMORY_MIN_SUMMARY_LEN = 15  # summaries shorter than this = no substance

# Each capture costs 3 LLM extractions (summary + entities + relations) plus
# embedding and a similarity scan, so a quality gate decides *before the
# expensive extraction pipeline* whether a turn is durable knowledge.  With
# AUTO_MEMORY_LLM_GATE=true (default) the gate is one cheap structured LLM
# call after a zero-cost fast-path; with it false, a free keyword heuristic
# is the sole gate.  Deliberately conservative: a missed capture is
# recoverable, a junk memory pollutes retrieval forever.
_AUTO_MEMORY_MIN_CONTENT_LEN = 12  # raw user message must be this long
# After stripping chatty words and symbols, this many informative characters
# (CJK hanzi / ASCII letters & digits) must remain — emoji spam, symbol runs
# and bare links pass the raw length gate but carry zero knowledge.
_AUTO_MEMORY_MIN_INFORMATIVE_CHARS = 6
_AUTO_MEMORY_QUESTION_SUFFIXES = ("？", "?", "吗", "呢", "吧", "啊")
_AUTO_MEMORY_QUESTION_MARKERS = (
    "什么", "怎么", "如何", "为什么", "哪些", "哪个", "能否", "能不能",
    "是不是", "有没有", "是否", "请问",
)
_AUTO_MEMORY_REQUEST_PREFIXES = (
    "帮我", "请帮我", "查一下", "搜索", "搜一下", "找一下", "找找",
    "介绍一下", "解释", "讲讲", "看看", "分析一下", "评估一下",
    "总结一下", "列出", "推荐", "对比一下",
)
_AUTO_MEMORY_CHATTY = frozenset({
    "你好", "您好", "谢谢", "感谢", "辛苦了", "再见", "拜拜", "好的",
    "嗯", "收到", "在吗", "hello", "hi", "ok", "okay",
    # Expanded — common acknowledgements / filler / pleasantries that carry
    # no durable knowledge.  Short tokens are redundant with the length gate,
    # so these are the multi-word or multi-char forms worth matching.
    "明白了", "知道了", "了解了", "没问题", "可以的", "好的呢", "好的哟",
    "好的好的", "好的没问题", "收到收到", "哈哈", "呵呵", "好的谢谢",
    "谢谢谢谢", "非常感谢", "辛苦了辛苦了", "早上好", "中午好", "下午好",
    "晚上好", "早安", "晚安", "加油", "欢迎欢迎", "恭喜恭喜", "好的呀",
    "好嘞", "行吧", "嗯嗯", "嗯呢", "thx", "thanks", "tks", "okay",
    "ok ok", "great", "fine", "sure", "yes", "no", "got it", "nice",
})


# Polite acknowledgement phrases stripped before the informative-char count —
# a pure-acknowledgement turn ("好的，明白了，收到，谢谢！") otherwise keeps
# 9 CJK hanzi and passes the symbol-noise check.  Longer phrases first so a
# short one never shadows a longer match.
_AUTO_MEMORY_POLITE_PHRASES = (
    "好的好的", "好的没问题", "明白了", "知道了", "没问题", "辛苦了",
    "可以的", "好的呢", "好的哟", "好的呀", "收到收到", "谢谢谢谢",
    "好的", "收到", "谢谢", "感谢", "哈哈", "呵呵", "嗯嗯", "好嘞",
)


def _is_symbol_noise(text: str) -> bool:
    """True when *text* carries fewer than ``_AUTO_MEMORY_MIN_INFORMATIVE_CHARS``
    informative characters after stripping polite filler.

    Polite phrases (谢谢 / 收到 / 明白了 …) are stripped first, then CJK
    hanzi and ASCII letters/digits count as informative; emoji, other
    non-CJK symbols, punctuation and whitespace are ignored.  A turn that is
    only emoji ("🎉"×12), symbol runs ("！！！！！"), or a bare polite
    acknowledgement ("好的，明白了，收到，谢谢！") collapses to near-zero —
    treated as noise.
    """
    stripped = text
    for phrase in _AUTO_MEMORY_POLITE_PHRASES:
        stripped = stripped.replace(phrase, "")
    informative = 0
    for ch in stripped:
        if ord(ch) > 0x7F:
            if "一" <= ch <= "鿿" or ch.isalnum():
                informative += 1
        elif ch.isalnum():
            informative += 1
    return informative < _AUTO_MEMORY_MIN_INFORMATIVE_CHARS


def _auto_memory_fast_worthy(content: str) -> bool:
    """Cheap zero-cost pre-filter: length, exact chatty set, symbol noise.

    Deliberately does NOT apply the question/request heuristics — they misfire
    on complex declarative statements ("为什么 X 会 Y？后来查明是……" carries a
    question marker but is durable knowledge).  In LLM-gate mode (the default)
    those cases are judged by ``_llm_gate_worthy`` instead.
    """
    text = content.strip()
    if len(text) < _AUTO_MEMORY_MIN_CONTENT_LEN:
        return False
    if text.lower() in _AUTO_MEMORY_CHATTY:
        return False
    if _is_symbol_noise(text):
        return False
    return True


def _is_auto_memory_worthy(content: str) -> bool:
    """Full keyword heuristic — the quality gate when AUTO_MEMORY_LLM_GATE=false.

    Fast path plus question/request rejection.  Free but coarse, so it misfires
    on complex phrasing (kills a declarative statement that contains a question
    marker, lets long chatty filler through).  With the LLM gate enabled (the
    default) only ``_auto_memory_fast_worthy`` runs before the LLM judge and
    this full heuristic is bypassed.  ``记住…`` requests are deliberately NOT
    filtered — they are the explicit-remember path, normally handled by
    ``write_memory_tool``, and capturing them here when the tool was not
    invoked is correct.
    """
    if not _auto_memory_fast_worthy(content):
        return False
    text = content.strip()
    if text.endswith(_AUTO_MEMORY_QUESTION_SUFFIXES):
        return False
    if any(marker in text for marker in _AUTO_MEMORY_QUESTION_MARKERS):
        return False
    lowered = text.lower()
    if any(text.startswith(p) for p in _AUTO_MEMORY_REQUEST_PREFIXES):
        return False
    if any(lowered.startswith(p) for p in ("please ", "can you ", "could you ")):
        return False
    return True


# ── LLM quality gate (B3) ─────────────────────────────────────────
# The keyword heuristic alone is free but coarse — it lets chatty-but-long
# filler through and can misfire on complex phrasing.  AUTO_MEMORY_LLM_GATE=
# true (default) routes every fast-path-passing turn through one cheap
# structured call asking whether the message is durable knowledge; setting it
# false restores the zero-LLM-cost keyword gate.
_AUTO_MEMORY_GATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["worthy"],
    "properties": {"worthy": {"type": "boolean"}},
}


async def _llm_gate_worthy(content: str) -> bool:
    """Ask the LLM whether *content* is durable knowledge (best-effort).

    Returns True when the gate is unavailable (LLM failure, schema-valid but
    missing verdict) so a gate outage never drops a heuristic-passing turn —
    the later ``_has_substance`` check still guards the write.  *content* is
    truncated so a long message doesn't inflate the gate call's prompt, and
    braces in it are escaped so a code snippet can't break the template's
    ``.format()`` interpolation (which would otherwise raise KeyError and
    fail the gate open on the exact messages it exists to judge).
    """
    try:
        from backend.service.prompts import get_prompt
        from backend.service.structured import chat_structured

        version, prompt = get_prompt("agent.auto_memory_gate")
        logger.debug("Auto-memory gate: prompt agent.auto_memory_gate v%s", version)
        # Escape the content's braces so ``.format`` interpolates them as
        # literal text (the template's own ``{{``/``}}`` escapes still render
        # as the JSON example braces).
        content_snippet = content[:500].replace("{", "{{").replace("}", "}}")
        data = await chat_structured(
            [{"role": "user", "content": prompt.format(content=content_snippet)}],
            json_schema=_AUTO_MEMORY_GATE_SCHEMA,
            scenario="auto_memory_gate",
            temperature=0.0,
        )
        return bool(data.get("worthy", False))
    except Exception:
        logger.warning(
            "Auto-memory LLM gate failed — defaulting to allow", exc_info=True
        )
        return True


# ── Auto-memory frequency control ───────────────────────────────────
# Capture is throttled before extraction runs (a throttled turn costs zero
# LLM calls): a per-thread minimum interval, a per-thread lifetime cap, an
# exact-repeat skip, and a process-wide rolling-window cap.  State is
# in-memory (same as the circuit breaker / token-usage counters) and resets
# on process restart.
_AUTO_MEMORY_WINDOW_SECONDS = 3600  # rolling window for the process-wide cap

_auto_memory_lock = threading.Lock()
_auto_memory_last_write: dict[str, float] = {}  # thread_id -> monotonic ts
_auto_memory_write_count: dict[str, int] = {}   # thread_id -> lifetime count
_auto_memory_last_content: dict[str, str] = {}  # thread_id -> last content
_auto_memory_recent_writes: deque = deque()     # monotonic ts, process-wide


def reset_auto_memory_throttle() -> None:
    """Drop all auto-memory throttle state — tests use this for isolation."""
    with _auto_memory_lock:
        _auto_memory_last_write.clear()
        _auto_memory_write_count.clear()
        _auto_memory_last_content.clear()
        _auto_memory_recent_writes.clear()


def _auto_memory_throttled(thread_id: str, content: str) -> bool:
    """True when a new auto-memory capture should be skipped (throttled)."""
    from backend.shared.config import config

    now = time.monotonic()
    with _auto_memory_lock:
        if _auto_memory_last_content.get(thread_id) == content:
            return True
        last = _auto_memory_last_write.get(thread_id)
        if last is not None and now - last < config.auto_memory_min_interval:
            return True
        if _auto_memory_write_count.get(thread_id, 0) >= config.auto_memory_max_per_thread:
            return True
        while (
            _auto_memory_recent_writes
            and now - _auto_memory_recent_writes[0] > _AUTO_MEMORY_WINDOW_SECONDS
        ):
            _auto_memory_recent_writes.popleft()
        return len(_auto_memory_recent_writes) >= config.auto_memory_max_per_window


def _record_auto_memory_write(thread_id: str, content: str) -> None:
    """Note a completed auto-memory capture for the throttle windows."""
    now = time.monotonic()
    with _auto_memory_lock:
        _auto_memory_last_write[thread_id] = now
        _auto_memory_write_count[thread_id] = _auto_memory_write_count.get(thread_id, 0) + 1
        _auto_memory_last_content[thread_id] = content
        _auto_memory_recent_writes.append(now)


def _message_text(message: BaseMessage) -> str:
    """Plain-text content of a message (falls back to str for block content)."""
    content = message.content
    return content if isinstance(content, str) else str(content)


def _last_human_content(messages: list[BaseMessage]) -> str:
    """Return the most recent HumanMessage's non-empty text, or ``""``."""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return _message_text(m).strip()
    return ""


def _write_tool_used_this_turn(messages: list[BaseMessage]) -> bool:
    """True if ``write_memory_tool`` was invoked or executed this turn.

    Scans from the latest HumanMessage forward for either an assistant
    tool_call naming the tool or a ToolMessage it produced.  Only the
    current turn counts — earlier turns' writes don't suppress auto memory.
    """
    start = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            start = i
            break
    for m in messages[start:]:
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "write_memory_tool":
            return True
        if isinstance(m, AIMessage) and any(
            str(tc.get("name", "")) == "write_memory_tool"
            for tc in getattr(m, "tool_calls", None) or []
        ):
            return True
    return False


def _has_substance(extracted: dict) -> bool:
    """Heuristic: does the extracted memory carry real knowledge?"""
    summary = str(extracted.get("summary") or "").strip()
    return len(summary) >= _AUTO_MEMORY_MIN_SUMMARY_LEN or bool(extracted.get("entities"))


async def _maybe_auto_memory(state: AgentState) -> None:
    """Best-effort automatic knowledge capture at the end of a turn.

    Runs only when both ``auto_memory_enabled`` and ``memory_enabled`` are set
    (the memory pipeline as a whole is opt-out via ``MEMORY_ENABLED=false`` —
    with no memory tools the agent is pure chat, so it must not keep writing
    memories behind the scenes).  A turn is captured only when (1) it passes
    the quality gate — the cheap fast-path plus, with ``AUTO_MEMORY_LLM_GATE``
    on (the default), one LLM judge call; with it off, the full keyword
    heuristic — (2) capture is not throttled (``_auto_memory_throttled``),
    (3) the agent did not already call ``write_memory_tool`` this turn, and
    (4) extraction yields substantive content.  Any failure is logged and
    swallowed — auto memory must never break the chat response.
    """
    from backend.shared.config import config, current_thread_id

    if not (config.auto_memory_enabled and config.memory_enabled):
        return

    user_content = _last_human_content(state["messages"])
    if not user_content:
        return
    if _write_tool_used_this_turn(state["messages"]):
        return

    # Quality gate — with the LLM gate enabled (default) only the cheap
    # fast-path applies here (length / chatty / symbol noise), so complex
    # declarative statements that carry a question marker or request phrasing
    # are NOT killed — they proceed to the LLM judge below.  With the LLM
    # gate disabled, the full keyword heuristic is the sole gate (zero LLM
    # cost per turn, coarser).
    if config.auto_memory_llm_gate:
        worthy = _auto_memory_fast_worthy(user_content)
    else:
        worthy = _is_auto_memory_worthy(user_content)
    if not worthy:
        logger.info("Auto-memory: not a knowledge statement, skipping")
        return

    # Frequency control — before the LLM gate, so a throttled turn never
    # pays for the judge call.
    thread_id = current_thread_id.get("") or "_"
    if _auto_memory_throttled(thread_id, user_content):
        logger.info("Auto-memory: throttled (interval/cap/window), skipping")
        return

    # LLM gate (AUTO_MEMORY_LLM_GATE=true, the default) — one cheap structured
    # call judging durable knowledge, after the zero-cost throttle so a
    # throttled turn pays nothing.
    if config.auto_memory_llm_gate and not await _llm_gate_worthy(user_content):
        logger.info("Auto-memory: LLM gate judged not worthy, skipping")
        return

    try:
        extracted = await extract_memory(user_content)
    except Exception:
        logger.exception("Auto-memory extraction failed for user message")
        return
    if not _has_substance(extracted):
        logger.info(
            "Auto-memory: user message carries no substantive knowledge, skipping"
        )
        return

    try:
        result = await write_memory(user_content, source_type="conversation")
        logger.info(
            "Auto-memory: wrote memory %s (action=%s)",
            result.get("id"),
            result.get("action"),
        )
        _record_auto_memory_write(thread_id, user_content)
    except Exception:
        logger.exception("Auto-memory write failed")


# ── Auto-memory background execution ─────────────────────────────────
# Auto-memory is best-effort knowledge capture that runs AFTER the answer
# is delivered.  It must not gate the response on 4-7 more LLM calls (gate +
# summary + entities + relations + embedding + optional conflict check) plus
# DB writes — a substantive turn previously hung the request for seconds
# after the SSE stream finished and held the agent-concurrency slot the whole
# time.  It now runs as a fire-and-forget background task; the context
# variables (``current_thread_id`` / ``current_trace_id``) are copied into
# the task at ``create_task``, so the capture keeps its thread/trace linkage.
#
# Strong references are held so the event loop can't garbage-collect a task
# at its first await (same pattern as ``_schedule_normalization`` in
# ``backend/service/memory.py``).  Concurrency is bounded by a semaphore:
# every agent run can spawn one capture per turn, and each fires up to 4-7
# LLM calls — unbounded, concurrent sessions would stack that onto the
# provider rate limit on top of the interactive traffic.  The per-thread /
# per-window throttle inside ``_maybe_auto_memory`` limits *writes*; this
# bounds the *calls*.
_AUTO_MEMORY_MAX_CONCURRENCY = 4
_auto_memory_semaphore = asyncio.Semaphore(_AUTO_MEMORY_MAX_CONCURRENCY)
_auto_memory_tasks: set[asyncio.Task] = set()


def _schedule_auto_memory(state: AgentState) -> None:
    """Fire-and-forget auto-memory capture — never blocks the response.

    Runs ``_maybe_auto_memory`` in a background task so the answer stream is
    not held open on the capture's LLM + embedding + DB calls.  Failures are
    logged inside ``_maybe_auto_memory`` (best-effort, swallowed) and never
    propagate; the task reference is held until it finishes so it can't be
    garbage-collected at its first await.
    """

    async def _run() -> None:
        try:
            async with _auto_memory_semaphore:
                await _maybe_auto_memory(state)
        except Exception:
            logger.exception("Auto-memory background capture failed")
        finally:
            _auto_memory_tasks.discard(asyncio.current_task())

    try:
        task = asyncio.create_task(_run())
    except RuntimeError:
        # No running event loop (e.g. synchronous test context) — skip.
        logger.debug("No event loop available; skipping auto-memory")
        return
    _auto_memory_tasks.add(task)


async def wait_auto_memory_tasks() -> None:
    """Wait for all in-flight auto-memory background tasks to finish.

    Tests call this after exercising a node that schedules auto-memory,
    before asserting on its side effects; production never needs it.
    """
    while _auto_memory_tasks:
        await asyncio.gather(*list(_auto_memory_tasks), return_exceptions=True)


# ── Nodes ────────────────────────────────────────────────────────────


async def call_llm_node(state: AgentState, *, tools: list) -> dict[str, Any]:
    """Send messages + tool definitions to the LLM, return an AIMessage.

    The *tools* parameter is injected at graph-construction time via
    ``functools.partial`` so the node has access to tool schemas without
    global state.

    Streaming: the LLM call is streamed, and text deltas are forwarded to
    the SSE client live through the LangGraph custom stream (``stream_mode=
    "custom"``).  Tool calls are accumulated and emitted at the end of the
    turn.  Under a non-streaming invocation (``ainvoke``, resume) the stream
    writer is a no-op and behaviour is unchanged.
    """
    provider = get_llm_provider()

    # A fresh HumanMessage starts a new conversation turn — restart the
    # ReAct step budget from 0.  The checkpointer otherwise persists
    # ``step_count`` session-wide, so a turn that exhausted MAX_AGENT_STEPS
    # would silently disable tools for every later turn in the thread.
    step_base = (
        0
        if _is_new_user_turn(state["messages"])
        else (state.get("step_count", 0) or 0)
    )

    # Prepend the system prompt if this is the first call, then bound the
    # history sent to the LLM (compact when enabled, then window to the
    # recent tail + pinned system prompts).
    messages = list(state["messages"])
    has_system = any(isinstance(m, SystemMessage) for m in messages)
    messages = await _bounded_messages(messages)
    if not has_system:
        version, system_text = get_prompt("agent.system")
        logger.info("call_llm_node: using agent.system prompt v%s", version)
        messages.insert(0, SystemMessage(content=system_text.format(context="")))
    # Coalesce the persona system and compaction's running-summary SystemMessage
    # into one before serialization — Anthropic only accepts a single top-level
    # ``system``, and a second ``role=system`` entry would be rejected.  The
    # conversation-derived summary is fenced as <summary> data, out of the
    # instruction section.
    messages = _merge_system_messages(messages)

    tool_schemas = _to_openai_tools(tools)
    dicts = _messages_to_dicts(messages)

    writer = _stream_writer()
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] | None = None
    try:
        async for event in provider.chat_raw_stream(
            messages=dicts, tools=tool_schemas, scenario="agent_chat"
        ):
            if event.get("type") == "content":
                text = str(event.get("text", ""))
                content_parts.append(text)
                writer({"type": "token", "content": text})
            elif event.get("type") == "tool_calls":
                tool_calls = event.get("tool_calls")
    except Exception as exc:
        logger.exception("LLM call failed in call_llm_node")
        # Stream a generic error text: it becomes this turn's final_response
        # (the plain-chat shortcut in generate_final_node reuses it), and the
        # SSE path only renders what the nodes stream — an unstreamed error
        # would leave the client's assistant message empty.  Provider details
        # stay in the log and state.error; the client sees no exception text.
        error_text = "抱歉，当前回答生成失败，请稍后重试。"
        writer({"type": "token", "content": error_text})
        # The error stub is marked so later turns never re-send it as assistant
        # history (see ``_is_llm_error_message``); ``state.error`` keeps the
        # exception detail.
        return {
            "error": str(exc),
            "messages": [
                AIMessage(
                    content=error_text,
                    additional_kwargs={_LLM_ERROR_MARKER: True},
                )
            ],
        }

    content = "".join(content_parts)

    if tool_calls:
        # Build AIMessage with tool_calls — LangChain will handle parsing
        lc_tool_calls: list[dict[str, Any]] = []
        for tc in tool_calls:
            lc_tool_calls.append({
                "id": str(tc["id"]),
                "name": str(tc["name"]),
                "args": tc["args"] if isinstance(tc["args"], dict) else {},
                "type": "tool_call",
            })
        aimessage = AIMessage(content=content or "", tool_calls=lc_tool_calls)  # type: ignore[arg-type]
    else:
        aimessage = AIMessage(content=content)

    return {"messages": [aimessage], "step_count": step_base + 1}


async def generate_final_node(state: AgentState) -> dict[str, Any]:
    """Produce the final answer — reuse ``call_llm``'s output when it is
    already complete, otherwise synthesize one with an LLM call.

    Reads tool-call results from the conversation history (ToolMessages)
    rather than from discrete state fields, so every tool's output is
    automatically included regardless of which tool was called.

    Plain-chat shortcut: when the last message is an AIMessage with text,
    no ``tool_calls``, and no ToolMessage appeared since the latest
    HumanMessage, that output is already a complete answer and is returned
    directly — no second LLM call.  Only tool turns hit the LLM here
    (once), so the response is persisted in the checkpointer state and
    streamed to the client live through the graph's ``custom`` stream
    (``get_stream_writer``).
    """
    # ── Plain-chat shortcut ─────────────────────────────────────────
    # When call_llm_node's output is already a complete answer — the last
    # message is an AIMessage with text and no tool_calls, and no tool
    # results appeared this turn — reuse it instead of synthesizing again.
    # Without this every plain chat turn paid two LLM calls (chat_raw for
    # tool detection + chat for the final answer), discarding the first.
    messages = state["messages"]
    last = messages[-1] if messages else None
    if (
        isinstance(last, AIMessage)
        and not getattr(last, "tool_calls", None)
        and last.content
        and not _has_tool_results_this_turn(messages)
    ):
        logger.info(
            "Reusing call_llm output as final answer (no tool results this turn)"
        )
        _schedule_auto_memory(state)
        return {"final_prompt": None, "final_response": str(last.content)}

    # ── Harvest context from ToolMessages in the recent conversation ──
    # Both this loop and the role loop below walk the *bounded* history so a
    # long thread doesn't resend unbounded tool results into the synthesis
    # prompt on every turn.  When compaction is enabled the overflow is folded
    # into a running-summary SystemMessage, which is folded into the context
    # block below so the synthesis LLM still sees the compressed history.
    windowed = await _bounded_messages(state["messages"])
    context_parts: list[str] = []
    # Compaction summaries (SystemMessages) become <summary> context items.
    for m in windowed:
        if isinstance(m, SystemMessage) and (m.content or "").strip():
            context_parts.append(f"<summary>\n{str(m.content).strip()}\n</summary>")
    for m in windowed:
        if not isinstance(m, ToolMessage):
            continue
        tool_name = getattr(m, "name", "unknown")
        # Chunk retrieval is noisy; the LLM's answer should rely on
        # memory search results (which have stable IDs and summaries).
        if tool_name == "retrieve_chunks_tool":
            continue
        raw = str(m.content) if m.content else ""
        if not raw.strip():
            continue
        # Parse the full envelope *before* truncating: tools return a
        # {"display": ..., "sources": [...]} JSON envelope, and the sources
        # array alone can exceed the truncation cap — cutting the raw text
        # first would corrupt the JSON and fall back to sending the LLM
        # truncated raw JSON instead of the clean display text.
        content = raw
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "display" in parsed:
                content = str(parsed["display"])
        except (json.JSONDecodeError, TypeError):
            content = raw
        content = _truncate_tool_content(content)
        if not content.strip():
            continue
        context_parts.append(_wrap_context_item(tool_name, content))

    context_str = "\n\n".join(context_parts) if context_parts else ""

    # Build the final prompt — single system message (Anthropic only
    # accepts one top-level system param, so context is folded in).  The
    # agent system template is the merged persona + answer guidance; the
    # retrieved-context block is injected via the ``{context}`` placeholder.
    version, system_text = get_prompt("agent.system")
    logger.info("generate_final_node: using agent.system prompt v%s", version)
    context_block = f"\n\nContext:\n{context_str}" if context_str else ""
    system_content = system_text.format(context=context_block)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content},
    ]

    # Include the recent conversation history (skip tool & system messages),
    # windowed so a long thread doesn't resend unbounded history every turn.
    for m in windowed:
        if _is_llm_error_message(m):
            continue  # a failed-call error stub must not enter the prompt
        if isinstance(m, (ToolMessage, SystemMessage)):
            continue
        role = "assistant" if isinstance(m, AIMessage) else "user"
        content = m.content if isinstance(m.content, str) else str(m.content)
        if role == "assistant" and content == "":
            continue  # skip tool_call-only AIMessages
        messages.append({"role": role, "content": content})

    # ── Call LLM here (once) so the response is persisted ──
    # Streamed: text deltas are forwarded to the SSE client live via the
    # LangGraph custom stream.  The full text is aggregated so the persisted
    # final_response matches what the client saw.
    provider = get_llm_provider()
    writer = _stream_writer()
    response_parts: list[str] = []
    try:
        async for token in provider.chat_stream(messages, scenario="agent_final"):
            response_parts.append(token)
            writer({"type": "token", "content": token})
        response = "".join(response_parts)
    except Exception:
        logger.exception("Final answer LLM call failed")
        # Generic text only — the exception details stay in the log.  The
        # message must still be streamed so the client isn't left with an
        # empty assistant message (the answer's tokens normally arrive via
        # the custom stream).
        response = "抱歉，生成回复时出现错误，请稍后重试。"
        writer({"type": "token", "content": response})

    aimessage = AIMessage(content=response)

    _schedule_auto_memory(state)

    return {
        "final_prompt": messages,
        "final_response": response,
        "messages": [aimessage],
    }


# ── Tools requiring human approval before execution ────────────────────
# Search/retrieval tools are safe — they only read.  Write tools must
# be approved because they modify the memory store.

APPROVAL_REQUIRED_TOOLS: frozenset[str] = frozenset({
    "write_memory_tool",
    "ingest_git_repo_tool",
    "ingest_document_tool",
})

# The interactive chat path additionally gates the external-notification tool
# (posting to the team's 飞书 group is a side effect an injected instruction
# in retrieved content could otherwise trigger).  Automated flows (patrol,
# scenarios) keep the default set so they can still notify the team
# autonomously — see ``build_agent_graph(approval_required_tools=...)``.
CHAT_APPROVAL_TOOLS: frozenset[str] = APPROVAL_REQUIRED_TOOLS | {"notify_feishu_tool"}


def _tool_call_id(call: dict) -> str:
    """Extract the stable tool_call id from a raw tool_call dict.

    Tool_calls appear in two shapes — LangChain's native
    ``{"id", "name", "args", ...}`` and OpenAI's
    ``{"id", "function": {"name", "arguments"}, ...}`` — so the id may live
    at the top level or nested under ``function``.
    """
    return str(call.get("id", call.get("function", {}).get("id", "")))


def _build_partial_approval_command(
    messages: list[BaseMessage],
    safe: list[dict],
    sensitive: list[dict],
    approved_ids: set[str],
    reason: str,
) -> Command:
    """Route to ToolNode with only the approved tool calls, rejecting the rest.

    ToolNode executes whatever ``tool_calls`` the latest AIMessage carries, so
    we replace that message (keeping its id so ``add_messages`` swaps it in
    place) with one holding only the approved subset plus every safe call, and
    append a ``[REJECTED]`` ToolMessage per rejected call.  The LLM then sees
    exactly which writes actually ran — previously the per-row buttons were a
    lie: the whole batch was routed to the reject branch and tools never ran
    while approved rows still got a fake ``[APPROVED]`` result.
    """
    safe_ids = {_tool_call_id(c) for c in safe}
    rejection_msgs: list[ToolMessage] = []
    exec_calls: list[dict] = list(safe)
    for call in sensitive:
        if _tool_call_id(call) in approved_ids:
            exec_calls.append(call)
        else:
            rejection_msgs.append(
                ToolMessage(
                    content=f"[REJECTED] {reason}",
                    tool_call_id=_tool_call_id(call),
                    name=str(call.get("name", call.get("function", {}).get("name", ""))),
                )
            )

    last_ai = next(
        (
            m
            for m in reversed(messages)
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
        ),
        None,
    )
    exec_tool_calls = [
        tc
        for tc in (last_ai.tool_calls if last_ai else [])
        if _tool_call_id(tc) in approved_ids or _tool_call_id(tc) in safe_ids
    ]
    exec_msg = AIMessage(
        content=last_ai.content if last_ai else "",
        tool_calls=exec_tool_calls,
        id=last_ai.id if last_ai else None,
    )

    logger.info(
        "Partial approval: executing %d/%d sensitive tool(s)",
        sum(1 for c in sensitive if _tool_call_id(c) in approved_ids),
        len(sensitive),
    )

    return Command(
        goto="tools",
        update={
            "messages": [*rejection_msgs, exec_msg],
            "pending_approval": None,
        },
    )


async def check_approval_node(
    state: AgentState,
    *,
    approval_required_tools: frozenset[str] = APPROVAL_REQUIRED_TOOLS,
) -> Command[Literal["tools", "call_llm"]]:
    """Gate sensitive tool calls before they reach ``ToolNode``.

    Inspects the last AIMessage's ``tool_calls`` and classifies each as
    *safe* (search / retrieval) or *sensitive* (write / ingest, plus any
    tool in ``approval_required_tools`` — default ``APPROVAL_REQUIRED_TOOLS``,
    chat passes ``CHAT_APPROVAL_TOOLS`` to add the notification tool).

    - All-safe → passes through to ``tools`` directly (no interrupt).
    - Any sensitive → ``interrupt()`` pauses the graph, surfacing the
      tool name and arguments for human review.  On resume the
      ``Command(resume=...)`` value decides:

        * ``{"approved": true}``  → route to ``tools`` (execute).
        * ``{"approved": false}`` → inject a rejection ``ToolMessage``
          and route back to ``call_llm`` so the LLM can explain why it
          skipped the action.
    """
    # Locate the AIMessage with tool_calls that tools_condition just matched
    messages = state["messages"]
    last_ai: AIMessage | None = None
    for m in reversed(messages):
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            last_ai = m
            break

    if last_ai is None:
        logger.warning(
            "tools_condition matched but no AIMessage with tool_calls found — "
            "injecting empty assistant message to break potential loop"
        )
        return Command(
            goto="call_llm",
            update={"messages": [AIMessage(content="(internal: no tool calls to process)")]},
        )

    tool_calls = last_ai.tool_calls

    # Separate safe from sensitive
    safe: list[dict] = []
    sensitive: list[dict] = []
    for tc in tool_calls:
        name = str(tc.get("name", tc.get("function", {}).get("name", "")))
        if name in approval_required_tools:
            sensitive.append(dict(tc))
        else:
            safe.append(dict(tc))

    # If everything is safe, go straight to ToolNode
    if not sensitive:
        return Command(goto="tools")

    # Build approval payloads for all sensitive calls
    calls_payload: list[dict[str, Any]] = []
    for call in sensitive:
        tname = str(call.get("name", call.get("function", {}).get("name", "unknown")))
        targs = call.get("args", call.get("function", {}).get("args", {}))
        if isinstance(targs, str):
            try:
                targs = json.loads(targs)
            except json.JSONDecodeError:
                targs = {"raw": targs}

        entry: dict[str, Any] = {
            # The stable tool_call id lets the resume payload (and the
            # frontend batch card) address this exact call — tool_name alone
            # collides when the same tool is called twice in one turn.
            "id": _tool_call_id(call),
            "tool_name": tname,
            "tool_args": targs,
        }
        if tname == "write_memory_tool":
            entry["summary"] = str(targs.get("content", ""))[:200]
        elif tname == "ingest_git_repo_tool":
            entry["summary"] = f"Repo: {targs.get('repo_path', '?')}"
        elif tname == "ingest_document_tool":
            entry["summary"] = f"Doc: {targs.get('document_id', '?')}"

        calls_payload.append(entry)

    if len(calls_payload) == 1:
        payload: dict[str, Any] = dict(calls_payload[0])
    else:
        payload = {"type": "batch", "calls": calls_payload}

    logger.info("Requesting human approval for %d tool(s)", len(calls_payload))

    # Pause and wait for human decision
    decision: dict[str, Any] = interrupt(payload)

    if len(calls_payload) == 1:
        approved = decision.get("approved") is True
    else:
        # Batch mode: check if ALL calls were approved
        batch_decisions = decision.get("calls", [])
        approved = bool(batch_decisions) and all(
            d.get("approved") is True for d in batch_decisions
        )

    if approved:
        logger.info("All %d tool(s) approved, executing", len(calls_payload))
        return Command(goto="tools", update={"pending_approval": None})

    reason = decision.get("reason", "Tool call was rejected by the user.")

    # Partial approval in batch mode — at least one call approved but not all:
    # execute the approved subset (plus every safe call) via ToolNode and
    # inject a [REJECTED] ToolMessage for each call the human did not approve.
    # Without this the "approve this row" button had no effect — the whole
    # batch either ran or was faked as approved.
    if len(calls_payload) > 1 and isinstance(decision.get("calls"), list):
        approved_ids = {
            str(d.get("id", "")) for d in decision["calls"] if d.get("approved") is True
        }
        if approved_ids:
            return _build_partial_approval_command(
                messages, safe, sensitive, approved_ids, reason
            )

    # Some or all rejected — inject a ToolMessage for every tool_call in this turn
    rejection_msgs: list[ToolMessage] = []
    for call in sensitive + safe:
        cid = _tool_call_id(call)
        tname = str(call.get("name", call.get("function", {}).get("name", "")))
        if call in sensitive:
            # In batch mode check whether THIS call (matched by id — the same
            # tool may legitimately appear twice in one turn) was rejected.
            call_rejected = True
            if len(calls_payload) > 1 and isinstance(decision.get("calls"), list):
                for bd in decision["calls"]:
                    if str(bd.get("id", "")) == cid:
                        call_rejected = bd.get("approved") is not True
                        break
            content = f"[REJECTED] {reason}" if call_rejected else "[APPROVED]"
        else:
            content = "[CANCELLED] A related write operation was rejected by the user."
        rejection_msgs.append(
            ToolMessage(content=content, tool_call_id=cid, name=tname)
        )

    # Log with names from calls_payload (always available, regardless of path)
    rejected_names = ", ".join(c["tool_name"] for c in calls_payload)
    logger.info("Tool(s) rejected [%s]: %s", rejected_names, reason)
    return Command(
        goto="call_llm",
        update={
            "messages": rejection_msgs,
            "pending_approval": None,
        },
    )


async def check_conflict_node(
    state: AgentState,
) -> Command[Literal["call_llm"]]:
    """Inspect ``write_memory_tool`` results — if conflict, pause for human.

    This node runs **after** ``tools`` and before ``call_llm``.  It scans
    the last ToolMessage from ``write_memory_tool`` for an ``action:
    "conflict"`` result.  When one is found it calls ``interrupt()`` with
    the conflict details so the human can choose:

    - ``keep_existing`` — discard the new memory
    - ``overwrite``     — replace existing with new
    - ``merge``         — LLM-merges both into existing
    - ``keep_both``     — insert new alongside existing

    On resume the human's ``resolution`` is applied via
    ``resolve_conflict()`` and a note is injected as a ToolMessage so the
    LLM is aware of what happened.
    """
    messages = state["messages"]

    # Find the last write_memory_tool result — but only within the current
    # turn.  Scanning the whole history could hit a conflict ToolMessage from
    # an earlier turn (still present after an interrupt/rollback) and pause
    # for approval on a turn that never wrote anything.  Stop at the first
    # HumanMessage, mirroring ``_has_tool_results_this_turn``.
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return Command(goto="call_llm")
        if (
            isinstance(m, ToolMessage)
            and getattr(m, "name", "") == "write_memory_tool"
        ):
            break
    else:
        return Command(goto="call_llm")

    content = str(m.content)  # type: ignore[possibly-used-before-assignment]
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        return Command(goto="call_llm")

    if result.get("action") != "conflict":
        return Command(goto="call_llm")

    # ── Conflict detected — pause for human ──
    conflicting_summary = str(result.get("existing_summary", "")[:300])
    new_summary = str(result.get("summary", "")[:300])

    payload: dict[str, Any] = {
        "type": "conflict",
        "new_summary": new_summary,
        "existing_id": str(result.get("existing_id", "")),
        "existing_summary": conflicting_summary,
        "options": ["keep_existing", "overwrite", "merge", "keep_both"],
        "deferred": result.get("_deferred"),
    }

    logger.info(
        "Conflict detected — pausing for human. New='%s' vs Existing='%s'",
        new_summary[:80],
        conflicting_summary[:80],
    )

    decision: dict[str, Any] = interrupt(payload)

    resolution = decision.get("resolution", "keep_existing")
    deferred = result.get("_deferred") or {}
    existing_id = str(result.get("existing_id", ""))

    try:
        outcome = await resolve_conflict(resolution, existing_id, deferred)
    except Exception as exc:
        logger.exception("Conflict resolution failed")
        outcome = {"id": existing_id, "action": "conflict_resolved", "resolution": "keep_existing"}

    action_label = _RESOLUTION_LABELS.get(resolution, resolution)
    logger.info("Conflict resolved: %s → %s (%s)", existing_id, resolution, outcome.get("id", "?"))

    # Replace the conflict ToolMessage with a resolution note (same id so
    # add_messages deduplicates it) — avoids two ToolMessages with the same
    # tool_call_id, which OpenAI-compatible APIs reject.
    note = ToolMessage(
        content=(
            f"Conflict resolved — {action_label}. "
            f"Memory id: {outcome.get('id', '?')}."
        ),
        tool_call_id=str(getattr(m, "tool_call_id", "")),
        name="write_memory_tool",
        id=getattr(m, "id", None),  # same id → replaces conflict msg
    )
    return Command(
        goto="call_llm",
        update={"messages": [note], "pending_approval": None},
    )


_RESOLUTION_LABELS: dict[str, str] = {
    "keep_existing": "kept the existing memory",
    "overwrite": "overwrote the existing memory with the new one",
    "merge": "merged both memories together",
    "keep_both": "kept both memories",
}
