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
from collections.abc import Mapping
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.config import get_stream_writer
from langgraph.types import Command, interrupt

from backend.agent.state import AgentState
from backend.agent.tool_envelope import envelope_display, truncate_tool_content
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


def _to_openai_tools(tools: list) -> list[dict[str, Any]]:
    """Convert LangChain tool objects to OpenAI function-calling schemas.

    The roster is fixed at graph-build time, so ``build_agent_graph``
    serialises the schemas once when it builds the graph and passes them into
    ``call_llm_node`` via partial — this function is not on the LLM hot path.
    Pure (no module state) and safe to call freely; ``call_llm_node`` falls
    back to it only when invoked without a pre-computed ``tool_schemas``
    argument (direct node invocation in tests).
    """
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

    return schemas


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
                emitted_tool_call_ids.update(str(tc["id"]) for tc in answered)
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


def _patrol_synthesis_max_tokens() -> int | None:
    """Larger output ceiling for a patrol's final synthesis, else None.

    A patrol report is generated in one LLM call and must arrive as complete
    JSON.  Once the model spends part of its interactive ``LLM_MAX_TOKENS``
    budget on internal reasoning the final message is truncated and the run
    fails contract validation — the budget that produced the 2026-08-10
    daily/weekly failures.  Patrol threads are recognised by their thread_id
    prefix (set by ``run_patrol``); the ``PATROL_MAX_TOKENS`` ceiling gives the
    report headroom while the patrol prompts cap per-category counts.
    """
    from backend.shared.config import config, current_thread_id

    if current_thread_id.get("").startswith("patrol-"):
        return config.patrol_max_tokens
    return None


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

# ── Compaction: two independent bounds ────────────────────────────
# A tool turn bounds the conversation twice — ``call_llm_node`` (tool
# selection) and ``generate_final_node`` (synthesis) — and each bound
# compacts independently.  The second bound's message list always differs
# from the first (the tool-call AIMessage and its ToolMessage are new), so
# sharing one running summary through AgentState rarely matched on exactly
# the turns that need compaction — the extra state cost more than the
# occasional saved call.  The worst case is one extra summariser call per
# turn on oversized history.


# Cap per-ToolMessage content in the compaction transcript — a single huge
# search result would otherwise crowd out the rest of the overflow prefix
# (the transcript has a hard char cap below).
_TRANSCRIPT_TOOL_CHAR_CAP = 200


def _tool_display_text(message: ToolMessage) -> str:
    """The display text of a tool result, unwrapped from its JSON envelope.

    Delegates to :func:`backend.agent.tool_envelope.envelope_display` — the shared
    unwrapper — so this path and the other consumers agree on what an
    envelope is.  Non-envelope results return their raw text.
    """
    return envelope_display(_message_text(message))


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
            # Envelope-aware: unwraps the display text before truncating.
            content = _truncate_tool_content(
                _message_text(m), limit=_TRANSCRIPT_TOOL_CHAR_CAP
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
    messages: list[BaseMessage],
    max_tokens: int | None = None,
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

    Each caller compacts independently — the tool-selection and synthesis
    bounds of a turn do not share a summary (see "Compaction: two
    independent bounds" above).
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
    transcript = _overflow_transcript(overflow)
    summary = await _summarize_overflow(overflow, transcript=transcript)
    if not summary:
        return messages
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
    messages: list[BaseMessage],
    max_tokens: int | None = None,
) -> list[BaseMessage]:
    """Compact (when enabled) then window *messages* for the LLM.

    With compaction disabled this is exactly ``_window_messages`` — the
    default behaviour is unchanged.
    """
    compacted = await _maybe_compact(messages, max_tokens)
    return _window_messages(compacted, max_tokens)


def _truncate_tool_content(text: str, limit: int = _MAX_TOOL_CONTENT_CHARS) -> str:
    """Cap a ToolMessage's content before it is resent to the LLM.

    Envelope-aware: a ``{"display", "sources"}`` tool result is unwrapped to
    its display text *before* truncating, so the model never re-reads a
    half-cut JSON blob as history.  Non-envelope text is truncated verbatim.
    The logic lives in :mod:`backend.agent.tool_envelope`; this alias keeps the
    existing call sites and tests working.
    """
    return truncate_tool_content(text, limit)


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
# expensive extraction pipeline* whether a turn is durable knowledge.  The
# gate is one cheap structured LLM call, preceded by a zero-cost length
# fast-path; the keyword heuristics this pipeline once carried are gone —
# "is this durable knowledge?" is judged by the LLM, not by phrase tables.
# Deliberately conservative: a missed capture is recoverable, a junk memory
# pollutes retrieval forever.
_AUTO_MEMORY_MIN_CONTENT_LEN = 12  # raw user message must be this long
_AUTO_MEMORY_GATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["worthy"],
    "properties": {"worthy": {"type": "boolean"}},
}


async def _llm_gate_worthy(content: str) -> bool:
    """Ask the LLM whether *content* is durable knowledge (best-effort).

    Returns True when the gate is unavailable (LLM failure, schema-valid but
    missing verdict) so a gate outage never drops a length-passing turn — the
    later ``_has_substance`` check still guards the write.  *content* is
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
        return bool(data.get("worthy", False)) if isinstance(data, dict) else False
    except Exception:
        logger.warning(
            "Auto-memory LLM gate failed — defaulting to allow", exc_info=True
        )
        return True


# ── Auto-memory frequency control ───────────────────────────────────
# Capture is throttled before extraction runs (a throttled turn costs zero
# LLM calls) by a single per-thread minimum interval.  The per-thread lifetime
# cap, process-wide rolling-window cap and exact-repeat skip this once carried
# all defended the same "don't write too much" goal — the interval alone
# bounds the steady-state write rate, and the content-hash idempotency in
# ``write_memory`` (backend/service/memory.py) already makes an exact repeat
# a no-op.  State is in-memory (same as the circuit breaker / token-usage
# counters) and resets on process restart.
_auto_memory_lock = threading.Lock()
_auto_memory_last_write: dict[str, float] = {}  # thread_id -> monotonic ts


def _auto_memory_throttled(thread_id: str) -> bool:
    """True when a new auto-memory capture should be skipped (throttled)."""
    from backend.shared.config import config

    now = time.monotonic()
    with _auto_memory_lock:
        last = _auto_memory_last_write.get(thread_id)
        if last is not None and now - last < config.auto_memory_min_interval:
            return True
    return False


def _record_auto_memory_write(thread_id: str) -> None:
    """Note a completed auto-memory capture for the interval throttle."""
    with _auto_memory_lock:
        _auto_memory_last_write[thread_id] = time.monotonic()


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


def _write_succeeded_this_turn(messages: list[BaseMessage]) -> bool:
    """True if this turn produced a *successful* ``write_memory_tool`` result.

    Success means the tool's ToolMessage is JSON carrying an ``action`` key
    (inserted / merged / duplicate).  An injected write whose execution
    failed leaves a non-JSON error ToolMessage, and a resolved conflict's
    ToolMessage is replaced by a plain-text resolution note — neither counts,
    so auto-memory stays free to capture the turn's knowledge (a failed or
    discarded write must not silently lose the content).
    """
    start = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            start = i
            break
    for m in messages[start:]:
        if isinstance(m, ToolMessage) and getattr(m, "name", "") == "write_memory_tool":
            try:
                data = json.loads(str(m.content))
            except (TypeError, ValueError):
                continue  # error ToolMessage / resolution note — not a write
            if isinstance(data, dict) and "action" in data:
                return True
    return False


def _has_substance(extracted: dict, source_content: str | None = None) -> bool:
    """Heuristic: does the extracted memory carry real knowledge?

    ``source_content`` (when supplied) lets the check reject a *degraded*
    extraction: when the LLM is unavailable, ``extract_summary`` falls back
    to the first 200 chars of the source verbatim (see
    ``backend.service.extraction.extract_summary``) and entity/relation
    extraction degrades to empty lists — a combination that easily clears
    the length gate below.  A summary that is exactly the verbatim truncation
    AND carries no entities is a failure artifact, not knowledge; refusing it
    keeps an LLM outage from polluting the memory store with raw truncations.
    """
    summary = str(extracted.get("summary") or "").strip()
    if source_content:
        stripped = source_content.strip()
        if summary and summary == stripped[:200] and not extracted.get("entities"):
            return False
    return len(summary) >= _AUTO_MEMORY_MIN_SUMMARY_LEN or bool(extracted.get("entities"))


async def _maybe_auto_memory(state: AgentState) -> None:
    """Best-effort automatic knowledge capture at the end of a turn.

    Runs only when both ``auto_memory_enabled`` and ``memory_enabled`` are set
    (the memory pipeline as a whole is opt-out via ``MEMORY_ENABLED=false`` —
    with no memory tools the agent is pure chat, so it must not keep writing
    memories behind the scenes).  A turn is captured only when (1) it passes
    the length fast-path, (2) capture is not throttled
    (``_auto_memory_throttled``), (3) the LLM quality gate judges it durable
    knowledge, (4) the turn did not already *successfully* write a memory via
    ``write_memory_tool`` (a failed or conflict-aborted write still lets
    auto-memory capture the knowledge, so content is never lost twice), and
    (5) extraction yields substantive content.  Any failure is
    logged and swallowed — auto memory must never break the chat response.
    """
    from backend.shared.config import config, current_thread_id

    if not (config.auto_memory_enabled and config.memory_enabled):
        return

    user_content = _last_human_content(state["messages"])
    if not user_content:
        return
    if _write_succeeded_this_turn(state["messages"]):
        return

    # Length fast-path — the only zero-cost pre-filter.  Everything deeper
    # (is this durable knowledge?) is judged by the LLM gate below; the
    # keyword heuristics that once sat here are gone.
    if len(user_content) < _AUTO_MEMORY_MIN_CONTENT_LEN:
        logger.info("Auto-memory: message too short, skipping")
        return

    # Frequency control — before the LLM gate, so a throttled turn never
    # pays for the judge call.
    thread_id = current_thread_id.get("") or "_"
    if _auto_memory_throttled(thread_id):
        logger.info("Auto-memory: throttled (interval), skipping")
        return

    # LLM quality gate — one cheap structured call judging durable knowledge.
    if not await _llm_gate_worthy(user_content):
        logger.info("Auto-memory: LLM gate judged not worthy, skipping")
        return

    try:
        extracted = await extract_memory(user_content)
    except Exception:
        logger.exception("Auto-memory extraction failed for user message")
        return
    if not _has_substance(extracted, user_content):
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
        _record_auto_memory_write(thread_id)
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
# provider rate limit on top of the interactive traffic.  The interval
# throttle inside ``_maybe_auto_memory`` limits *writes*; this bounds the
# *calls*.
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


# ── Nodes ────────────────────────────────────────────────────────────


async def call_llm_node(
    state: AgentState,
    *,
    tools: list,
    tool_schemas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Send messages + tool definitions to the LLM, return an AIMessage.

    *tools* and the pre-serialised *tool_schemas* are injected at
    graph-construction time via ``functools.partial`` — schema serialisation
    is a per-build cost, not a per-turn one.  ``tool_schemas=None`` falls
    back to serialising *tools* here (direct node invocation in tests).

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

    if tool_schemas is None:
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
        if content_parts:
            # Partial tokens already reached the client — appending the error
            # text here would glue "half an answer + apology" together.  Emit
            # it as a distinct error event instead (the SSE layer forwards it
            # as a separate ``error`` event); non-streaming callers have a
            # no-op writer and simply keep the error stub in state.
            writer({"type": "error", "message": error_text})
        else:
            writer({"type": "token", "content": error_text})
        # The error stub is marked so later turns never re-send it as assistant
        # history (see ``_is_llm_error_message``); ``state.error`` keeps the
        # exception detail.  The force-write flag is cleared so a failed turn
        # can't leak it into the next user turn.
        return {
            "error": str(exc),
            "force_write": False,
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
        lc_tool_calls = None
        aimessage = AIMessage(content=content)

    # ── Force-write injection ────────────────────────────────────────────
    # When the user checked 强制写入记忆, inject a write_memory_tool call into
    # this AIMessage's tool_calls.  The write runs through the normal ReAct
    # pipeline — ToolNode executes it, check_conflict gates a conflict for
    # on-the-spot resolution, and _write_succeeded_this_turn suppresses
    # auto-memory only once the write actually succeeded (no double
    # extraction; a failed write still lets auto-memory capture the turn).
    # write_memory itself runs the LLM extraction, so what lands in the store
    # is the distilled knowledge, not the raw user text.  The auto-memory
    # throttle is recorded by write_memory_tool itself on success, not here —
    # recording at injection would suppress auto-memory even when the write
    # fails.  Skipped when the model already called the write tool this turn
    # (e.g. MEMORY_ENABLED=false restores the explicit write tool) or when the
    # tool is not on the execution roster (chat-only mode).  The flag is
    # cleared so it fires exactly once.
    if (
        state.get("force_write")
        and not _write_tool_used_this_turn(state["messages"])
        and any(t.name == "write_memory_tool" for t in tools)
    ):
        fw_content = _last_human_content(state["messages"])
        if fw_content:
            lc_tool_calls = (lc_tool_calls or []) + [{
                "id": f"force_write_{len(lc_tool_calls or []) + 1}",
                "name": "write_memory_tool",
                "args": {
                    "content": fw_content,
                    "source_type": "conversation",
                    "metadata": {"forced": True},
                },
                "type": "tool_call",
            }]
            aimessage = AIMessage(content=content or "", tool_calls=lc_tool_calls)  # type: ignore[arg-type]

    update: dict[str, Any] = {"messages": [aimessage], "step_count": step_base + 1}
    if state.get("force_write"):
        update["force_write"] = False
    return update


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
        return {
            "final_prompt": None,
            "final_response": str(last.content),
        }

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
        if isinstance(m, SystemMessage) and str(m.content or "").strip():
            context_parts.append(f"<summary>\n{str(m.content).strip()}\n</summary>")
    for m in windowed:
        if not isinstance(m, ToolMessage):
            continue
        tool_name = getattr(m, "name", "unknown")
        # Chunk retrieval results are folded into the synthesis context like
        # memory results.  Both ``retrieve_chunks_tool`` and
        # ``query_rewrite_and_search_tool`` return document chunks (and both
        # are tagged ``doc`` in ``_CONTEXT_DOC_TOOLS``); previously only
        # ``retrieve_chunks_tool`` was skipped here, so a turn whose *last*
        # tool call was a chunk search synthesized the answer without ever
        # seeing the chunks it retrieved.  Content is envelope-truncated
        # below, so a noisy dump can't crowd out the prompt.
        raw = str(m.content) if m.content else ""
        if not raw.strip():
            continue
        # Envelope-aware truncation: the display text is unwrapped *before*
        # capping (see backend.agent.tool_envelope.truncate_tool_content) — the
        # sources array alone can exceed the cap, and cutting the raw JSON
        # first would send the LLM truncated raw JSON instead of the clean
        # display text.
        content = _truncate_tool_content(raw)
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

    final_messages: list[dict[str, str]] = [
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
        final_messages.append({"role": role, "content": content})

    # ── Call LLM here (once) so the response is persisted ──
    # Streamed: text deltas are forwarded to the SSE client live via the
    # LangGraph custom stream.  The full text is aggregated so the persisted
    # final_response matches what the client saw.
    provider = get_llm_provider()
    writer = _stream_writer()
    response_parts: list[str] = []
    max_tokens = _patrol_synthesis_max_tokens()
    try:
        # Patrol threads raise the output ceiling (see
        # ``_patrol_synthesis_max_tokens``) so a complete report fits; the
        # interactive chat path keeps the provider default.
        final_kwargs: dict[str, Any] = {}
        if max_tokens:
            final_kwargs["max_tokens"] = max_tokens
        async for token in provider.chat_stream(
            final_messages, scenario="agent_final", **final_kwargs
        ):
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
        if response_parts:
            # Partial tokens already reached the client — emitting the error
            # text as a token would glue "half an answer + apology" together.
            # A distinct error event lets the SSE layer surface it separately.
            writer({"type": "error", "message": response})
        else:
            writer({"type": "token", "content": response})

    # The error stub is marked so later turns never re-send it as assistant
    # history (same marker as call_llm_node's failed-call stub).
    aimessage = AIMessage(
        content=response,
        additional_kwargs={_LLM_ERROR_MARKER: True},
    )

    _schedule_auto_memory(state)

    return {
        "final_prompt": final_messages,
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
# in retrieved content could otherwise trigger).  ``write_memory_tool`` is
# deliberately NOT gated in chat: the only way it fires there is the
# force-write injection, which the user already confirmed by checking
# 强制写入记忆 — asking for a second approval would be double confirmation.
# (Automated flows keep the default set so ``write_memory_tool`` still pauses
# if the model ever chooses it.)  Automated flows (patrol, scenarios) keep
# the default set so they can still notify the team autonomously — see
# ``build_agent_graph(approval_required_tools=...)``.
CHAT_APPROVAL_TOOLS: frozenset[str] = (
    APPROVAL_REQUIRED_TOOLS - {"write_memory_tool"}
) | {"notify_feishu_tool"}


def _tool_call_id(call: Mapping[str, Any]) -> str:
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

    The rejection feedback is folded into ``exec_msg.content`` as well: the
    ``[REJECTED]`` ToolMessages are orphans (their ``tool_call_id`` is on no
    retained AIMessage), which the serialization layer drops — without the
    note the next ``call_llm`` would never see which writes were declined and
    could silently retry them.  ToolNode re-executes nothing rejected (those
    calls stay out of ``exec_msg.tool_calls``).
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
    # The [REJECTED] ToolMessages are orphans for serialization — their
    # tool_call_id is on no retained AIMessage (exec_msg only carries approved
    # + safe calls), so ``_messages_to_dicts`` drops them and the next
    # call_llm would never learn which writes were declined.  Fold a summary
    # into exec_msg's content instead: the model sees the rejected calls and
    # reason, while ToolNode still only executes the approved subset (the
    # rejected calls stay out of exec_msg.tool_calls).
    rejection_note = ""
    if rejection_msgs:
        rejected_desc = "; ".join(
            f"{getattr(m, 'name', '') or 'tool'} (id: {m.tool_call_id})"
            for m in rejection_msgs
        )
        rejection_note = (
            f"\n\n[REJECTED] These tool call(s) were rejected by the user and "
            f"NOT executed: {rejected_desc}. Reason: {reason}"
        )
    exec_msg = AIMessage(
        content=(str(last_ai.content) if last_ai else "") + rejection_note,
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
        tcd: dict[str, Any] = dict(tc)
        name = str(tcd.get("name", tcd.get("function", {}).get("name", "")))
        if name in approval_required_tools:
            sensitive.append(tcd)
        else:
            safe.append(tcd)

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
    except Exception:
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
