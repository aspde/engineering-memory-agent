"""EMA — Engineering Memory Agent — Streamlit MVP.

Entry point with shared state initialisation, sidebar, and multipage
navigation via ``st.navigation``. Individual pages live in ``pages/``.
"""

from __future__ import annotations

import json
import time
import uuid

import httpx
import streamlit as st

BACKEND_URL = "http://localhost:8000"


@st.cache_resource(show_spinner=False)
def _get_client() -> httpx.Client:
    """Return a shared httpx.Client with connection pooling.

    Cached via ``@st.cache_resource`` so TCP connections are reused
    across Streamlit reruns instead of paying a ~200ms handshake on
    every request.
    """
    return httpx.Client(
        base_url=BACKEND_URL,
        timeout=httpx.Timeout(300, connect=10),
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=20),
    )


# ═══════════════════════════════════════════════════════════════════════
# Session state (shared across all pages)
# ═══════════════════════════════════════════════════════════════════════


def _init_session() -> None:
    """Seed ``st.session_state`` with default values once per session."""
    defaults: dict[str, object] = {
        "thread_id": str(uuid.uuid4()),
        "messages": [],  # list[dict] with role, content, metadata
        "pending_interrupt": None,  # dict or None
        "waiting_for_approval": False,
        "_stream_interrupt": None,  # internal: interrupt caught during streaming
        "_threads": None,  # cached list[ThreadInfo] from backend
        "_threads_fetched_at": 0.0,  # timestamp of last fetch
        "_loaded_thread_id": None,  # thread_id whose messages are in session
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ═══════════════════════════════════════════════════════════════════════
# Backend communication (shared across pages)
# ═══════════════════════════════════════════════════════════════════════


def _node_labels() -> dict[str, str]:
    return {
        "call_llm": "思考中…",
        "tools": "执行工具…",
        "check_approval": "检查权限…",
        "check_conflict": "检查冲突…",
    }


def _stream_response(user_input: str = ""):
    """Generator that yields tokens and status strings from the SSE stream."""
    thread_id = st.session_state["thread_id"]
    payload: dict = {"message": user_input, "thread_id": thread_id}

    try:
        client = _get_client()
        with client.stream(
            "POST",
            "/api/agent/chat/stream",
            json=payload,
        ) as resp:
            if resp.status_code != 200:
                yield f"*后端错误 ({resp.status_code})*"
                return

            yielded_nodes: set[str] = set()
            for line in resp.iter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")

                if etype == "node":
                    node = event.get("node", "")
                    if node != "generate_final" and node not in yielded_nodes:
                        yielded_nodes.add(node)
                        label = _node_labels().get(node, node)
                        yield f"\n\n> {label}\n\n"

                elif etype == "token":
                    yield event["content"]

                elif etype == "interrupt":
                    st.session_state["_stream_interrupt"] = event["data"]
                    return

                elif etype == "error":
                    yield f"\n\n*错误: {event.get('message', 'unknown')}*"

    except httpx.RequestError as exc:
        yield f"\n\n*连接失败: {exc}*"


def _call_agent_nonstream(message: str = "", resume_data: dict | None = None) -> dict:
    """Fallback non-streaming call for simple resume flows."""
    thread_id = st.session_state["thread_id"]
    payload: dict = {"message": message, "thread_id": thread_id}
    if resume_data is not None:
        payload["resume_data"] = resume_data

    try:
        client = _get_client()
        resp = client.post(
            "/api/agent/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except httpx.RequestError as exc:
        return {
            "thread_id": thread_id,
            "status": "error",
            "response": f"Backend error: {exc}",
            "interrupt": None,
            "tool_calls": [],
            "sources": [],
        }


# ═══════════════════════════════════════════════════════════════════════
# UI helpers (shared across pages)
# ═══════════════════════════════════════════════════════════════════════


def _render_message(msg: dict) -> None:
    """Render a single chat message bubble."""
    role = msg["role"]
    content = msg["content"]

    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)
    elif role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(content)
            meta = msg.get("_meta", {})
            tool_calls = meta.get("tool_calls", [])
            sources = meta.get("sources", [])
            if tool_calls:
                with st.expander("🔧 Tool calls", expanded=False):
                    for tc in tool_calls:
                        st.caption(f"`{tc['tool']}`")
                        st.text(tc["content"][:300])
            if sources:
                with st.expander("📚 Sources", expanded=False):
                    for s in sources:
                        st.caption(f"`{s['type']}` — {s['snippet'][:200]}")
    elif role == "system":
        with st.chat_message("assistant", avatar="⚠️"):
            st.info(content)


def _render_approval(interrupt: dict) -> None:
    """Render the approval or conflict-resolution card."""
    itype = interrupt.get("type", "")
    if itype == "conflict":
        _render_conflict_resolution(interrupt)
    else:
        _render_tool_approval(interrupt)


def _render_tool_approval(interrupt: dict) -> None:
    """Approval card for write/ingest tools."""
    tool_name = interrupt.get("tool_name", "unknown")
    args = interrupt.get("tool_args", {})
    summary = interrupt.get("summary", str(args)[:200])

    labels: dict[str, str] = {
        "write_memory_tool": "Write Memory",
        "ingest_git_repo_tool": "Ingest Git Repository",
        "ingest_document_tool": "Ingest Document",
    }
    label = labels.get(tool_name, tool_name)

    with st.chat_message("assistant", avatar="🛡️"):
        st.markdown(f"**Pending Approval: {label}**")
        st.info(summary)

        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✅ Approve", key="approve_btn", use_container_width=True):
                _handle_approval(approved=True)
        with col2:
            if st.button("❌ Reject", key="reject_btn", use_container_width=True):
                _handle_approval(approved=False)


def _render_conflict_resolution(interrupt: dict) -> None:
    """Conflict resolution card."""
    new_summary = interrupt.get("new_summary", "")
    existing_summary = interrupt.get("existing_summary", "")

    with st.chat_message("assistant", avatar="⚖️"):
        st.markdown("**⚠️ Memory Conflict Detected**")
        st.caption("A new memory contradicts an existing one. How should I resolve this?")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**New memory**")
            st.info(new_summary or "(empty)")
        with col_b:
            st.markdown("**Existing memory**")
            st.warning(existing_summary or "(empty)")

        st.divider()

        cols = st.columns(4)
        with cols[0]:
            if st.button("📌 Keep Existing", key="conflict_keep_existing", use_container_width=True):
                _handle_conflict("keep_existing")
        with cols[1]:
            if st.button("✏️ Overwrite", key="conflict_overwrite", use_container_width=True):
                _handle_conflict("overwrite")
        with cols[2]:
            if st.button("🔀 Merge", key="conflict_merge", use_container_width=True):
                _handle_conflict("merge")
        with cols[3]:
            if st.button("📋 Keep Both", key="conflict_keep_both", use_container_width=True):
                _handle_conflict("keep_both")


def _handle_conflict(resolution: str) -> None:
    """Resume the agent with a conflict resolution choice."""
    st.session_state["waiting_for_approval"] = False
    st.session_state["pending_interrupt"] = None
    _resume({"resolution": resolution})


def _resume(resume_data: dict) -> None:
    """Resume the agent after approval/rejection/conflict resolution."""
    result = _call_agent_nonstream(resume_data=resume_data)

    if result["status"] == "interrupted":
        st.session_state["pending_interrupt"] = result.get("interrupt")
        st.session_state["waiting_for_approval"] = True
        st.session_state["messages"].append({
            "role": "system",
            "content": "Another action needs attention.",
        })
    else:
        response_text = result.get("response", "")
        st.session_state["messages"].append({
            "role": "assistant",
            "content": response_text or "(no response)",
            "_meta": {
                "tool_calls": result.get("tool_calls", []),
                "sources": result.get("sources", []),
            },
        })

    st.rerun()


def _handle_approval(approved: bool) -> None:
    """Resume the agent with the user's approval decision."""
    st.session_state["waiting_for_approval"] = False
    st.session_state["pending_interrupt"] = None

    resume = {"approved": approved}
    if not approved:
        resume["reason"] = "User rejected the tool call."

    _resume(resume)


# ═══════════════════════════════════════════════════════════════════════
# Chat input handler (shared by chat page)
# ═══════════════════════════════════════════════════════════════════════


def _handle_chat_input() -> None:
    """Handles a single chat turn without full-page rerun."""
    disabled = st.session_state["waiting_for_approval"]
    user_input = st.chat_input(
        "Ask EMA anything…" if not disabled else "Waiting for approval…",
        disabled=disabled,
    )

    if not user_input or not user_input.strip():
        return

    _perform_stream(user_input.strip())


def _perform_stream(user_input: str) -> None:
    """Execute streaming for the given user input (no input capture)."""
    # Render user message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Store user message in session state
    st.session_state["messages"].append({
        "role": "user",
        "content": user_input,
    })

    # Remember that this thread's messages are loaded so chat won't re-fetch
    st.session_state["_loaded_thread_id"] = st.session_state["thread_id"]

    # Stream the response
    with st.chat_message("assistant"):
        st.session_state["_stream_interrupt"] = None
        full_response = st.write_stream(
            _stream_response(user_input=user_input)
        )

    # Check if streaming was interrupted by an approval gate
    interrupt = st.session_state.get("_stream_interrupt")
    if interrupt is not None:
        st.session_state["pending_interrupt"] = interrupt
        st.session_state["waiting_for_approval"] = True
        st.session_state["_stream_interrupt"] = None
        itype = interrupt.get("type", "")
        note = (
            "A memory conflict was detected. Please choose how to resolve it."
            if itype == "conflict"
            else "The agent wants to perform a write operation. Please approve or reject."
        )
        st.session_state["messages"].append({
            "role": "system",
            "content": note,
        })
    else:
        st.session_state["messages"].append({
            "role": "assistant",
            "content": full_response or "(no response)",
            "_meta": {"tool_calls": [], "sources": []},
        })

    # Remember that this thread's messages are loaded so chat won't re-fetch
    st.session_state["_loaded_thread_id"] = st.session_state["thread_id"]


def _render_sidebar() -> None:
    """Render the sidebar with conversation list and new-conversation button."""
    st.html(
        "<style>"
        "[data-testid='stSidebarHeader'] { height: 36px !important; min-height: 36px !important; }"
        "[data-testid='stSidebarCollapseButton'] { position: relative !important; top: 8px !important; }"
        "[data-testid='stExpandSidebarButton'] { position: relative !important; top: 8px !important; }"
        "[data-testid='stSidebarUserContent'] { padding-top: 0 !important; }"
        ".stSidebar h1 { padding-top: 0 !important; }"
        ".stSidebar hr { margin-top: 0.5rem; }"
        "[data-testid='stToolbar'] { height: 36px !important; min-height: 36px !important; }"
        "header[data-testid='stHeader'] { height: 36px !important; min-height: 36px !important; }"
        "</style>"
    )
    st.title("🧠 EMA")
    st.caption("Engineering Memory Agent")

    # ── New conversation button → reset and switch to chat page ──
    if st.button("新建对话", use_container_width=True, key="sb_new_conv"):
        st.session_state["thread_id"] = str(uuid.uuid4())
        st.session_state["messages"] = []
        st.session_state["pending_interrupt"] = None
        st.session_state["waiting_for_approval"] = False
        st.session_state["_threads_fetched_at"] = 0.0
        st.session_state["_loaded_thread_id"] = None
        st.switch_page("pages/chat.py")

    if st.button("📚 记忆库", use_container_width=True, key="sb_memories"):
        st.switch_page("pages/memories.py")

    st.caption("**对话历史**")

    # ── Thread list (cached — 30 s TTL) ──
    _THREAD_CACHE_TTL = 30.0
    now = time.monotonic()
    threads: list[dict] = st.session_state.get("_threads") or []
    if now - st.session_state.get("_threads_fetched_at", 0.0) > _THREAD_CACHE_TTL:
        try:
            client = _get_client()
            resp = client.get("/api/agent/threads", timeout=5)
            if resp.status_code == 200:
                threads = resp.json()
                st.session_state["_threads"] = threads
                st.session_state["_threads_fetched_at"] = now
        except Exception:
            pass

    if threads:
        for t in threads:
            tid = t["thread_id"]
            title = t.get("title", tid[:8])
            active = tid == st.session_state["thread_id"]
            label = f"{'● ' if active else ''}{title}"
            if st.button(
                label,
                key=f"thread_{tid}",
                use_container_width=True,
                disabled=active,
            ):
                st.session_state["thread_id"] = tid
                st.session_state["messages"] = []
                st.session_state["pending_interrupt"] = None
                st.session_state["waiting_for_approval"] = False
                st.switch_page("pages/chat.py")
    else:
        st.caption("暂无历史对话")


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════


def main() -> None:
    st.set_page_config(page_title="EMA", page_icon="frontend/static/brain_favicon.png", layout="wide")
    _init_session()

    # ── Navigation ──
    pages = [
        st.Page("pages/chat.py", title="对话", icon=":material/chat:", default=True),
        st.Page("pages/memories.py", title="记忆库", icon=":material/book:"),
    ]
    pg = st.navigation(pages, position="hidden")
    pg.run()

    # ── Sidebar (our custom sidebar, shown on all pages) ──
    with st.sidebar:
        _render_sidebar()


main()
