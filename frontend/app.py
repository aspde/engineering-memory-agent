"""EMA — Engineering Memory Agent — Streamlit MVP.

Chat interface with Human-in-the-Loop approval: when the agent wants
to write or ingest, it pauses and asks for confirmation before
modifying the memory store.
"""

from __future__ import annotations

import uuid

import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"


# ── Session state initialisation ─────────────────────────────────────


def _init_session() -> None:
    """Seed ``st.session_state`` with default values once per session."""
    defaults: dict[str, object] = {
        "thread_id": str(uuid.uuid4()),
        "messages": [],  # list[dict] with role, content, metadata
        "pending_interrupt": None,  # dict or None
        "waiting_for_approval": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ── API helpers ──────────────────────────────────────────────────────


def _call_agent(
    message: str = "",
    resume_data: dict | None = None,
) -> dict:
    """Send a message (or resume) to the agent and return the parsed response."""
    thread_id = st.session_state["thread_id"]
    payload: dict = {
        "message": message,
        "thread_id": thread_id,
    }
    if resume_data is not None:
        payload["resume_data"] = resume_data

    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/agent/chat",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        return {
            "thread_id": thread_id,
            "status": "error",
            "response": f"Backend error: {exc}",
            "interrupt": None,
            "tool_calls": [],
            "sources": [],
        }


# ── UI helpers ───────────────────────────────────────────────────────


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
            # Show tool traces and sources if present
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
    """Render the approval or conflict-resolution card when the agent is paused."""
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
    """Conflict resolution card — human judges contradicting memories."""
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
    """Call the backend with resume data and handle the result."""
    result = _call_agent(resume_data=resume_data)

    if result["status"] == "interrupted":
        st.session_state["pending_interrupt"] = result.get("interrupt")
        st.session_state["waiting_for_approval"] = True
        st.session_state["messages"].append({
            "role": "system",
            "content": f"Another action needs attention.",
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


# ── Main UI ──────────────────────────────────────────────────────────


def main() -> None:
    st.set_page_config(page_title="EMA", page_icon="🧠", layout="wide")
    _init_session()

    # ── Sidebar ──
    with st.sidebar:
        st.title("🧠 EMA")
        st.caption("Engineering Memory Agent")
        st.divider()

        st.write("**Thread ID**")
        st.code(st.session_state["thread_id"], language=None)

        if st.button("🔄 New Conversation"):
            st.session_state["thread_id"] = str(uuid.uuid4())
            st.session_state["messages"] = []
            st.session_state["pending_interrupt"] = None
            st.session_state["waiting_for_approval"] = False
            st.rerun()

        st.divider()
        st.caption(
            "Tools requiring approval: write memory, ingest repo, ingest document. "
            "Search and retrieval run automatically."
        )

    # ── Chat area ──
    st.title("EMA — Engineering Memory Agent")

    # Render message history
    for msg in st.session_state["messages"]:
        _render_message(msg)

    # Render pending approval card (if agent is waiting)
    if st.session_state["waiting_for_approval"] and st.session_state["pending_interrupt"]:
        _render_approval(st.session_state["pending_interrupt"])

    # ── Input area ──
    disabled = st.session_state["waiting_for_approval"]
    user_input = st.chat_input(
        "Ask EMA anything…" if not disabled else "Waiting for approval…",
        disabled=disabled,
    )

    if user_input and user_input.strip():
        # Add user message to history
        st.session_state["messages"].append({
            "role": "user",
            "content": user_input.strip(),
        })

        # Send to backend
        result = _call_agent(message=user_input.strip())

        if result["status"] == "interrupted":
            st.session_state["pending_interrupt"] = result.get("interrupt")
            st.session_state["waiting_for_approval"] = True
            itype = result.get("interrupt", {}).get("type", "")
            if itype == "conflict":
                note = "A memory conflict was detected. Please choose how to resolve it."
            else:
                note = "The agent wants to perform a write operation. Please approve or reject."
            st.session_state["messages"].append({
                "role": "system",
                "content": note,
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


if __name__ == "__main__":
    main()
