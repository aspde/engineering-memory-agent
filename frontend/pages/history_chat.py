"""History chat page — messages + input pinned to viewport bottom."""

from __future__ import annotations

import streamlit as st

from frontend.app import (
    _get_client,
    _handle_chat_input,
    _render_approval,
    _render_message,
)

# ── CSS: input fixed to bottom ──
st.markdown(
    "<style>"
    "  hr { display: none !important; }"
    "  .stMainBlockContainer + div { display: none !important; }"
    "  .stMainBlockContainer { border-bottom: none !important;"
    "    padding-top: 0.5rem !important;"
    "    padding-bottom: 100px !important; }"
    "  [data-testid='stChatInput'] {"
    "    position: fixed !important; bottom: 0.1rem !important;"
    "    z-index: 100 !important;"
    "    left: 50% !important; transform: translateX(-50%) !important;"
    "    max-width: 720px !important; width: calc(100vw - 21rem) !important;"
    "    background: var(--default-backgroundColor) !important;"
    "    padding: 0.75rem 0 0.5rem 0 !important; }"
    "</style>",
    unsafe_allow_html=True,
)

_MAX_VISIBLE = 50
msgs: list[dict] = st.session_state.get("messages", [])

# ── Lazy-load messages on thread switch ──
tid = st.session_state["thread_id"]
if st.session_state.get("_loaded_thread_id") != tid:
    known_tids = {t["thread_id"] for t in (st.session_state.get("_threads") or [])}
    if not known_tids or tid not in known_tids:
        st.session_state["messages"] = []
        st.session_state["_loaded_thread_id"] = tid
        msgs = []
    else:
        try:
            r = _get_client().get(f"/api/agent/thread/{tid}", timeout=5)
            if r.status_code == 200:
                msgs = r.json().get("messages", [])
                st.session_state["messages"] = msgs
                st.session_state["_loaded_thread_id"] = tid
        except Exception:
            pass

# ── Chat input ──
@st.fragment
def _chat_fragment() -> None:
    _handle_chat_input()

_chat_fragment()

# ── Messages ──
visible = msgs[-_MAX_VISIBLE:]
if len(msgs) > _MAX_VISIBLE:
    st.caption(f"*… {len(msgs) - _MAX_VISIBLE} older messages hidden*")
for msg in visible:
    _render_message(msg)

# ── Pending approval ──
if st.session_state["waiting_for_approval"] and st.session_state["pending_interrupt"]:
    _render_approval(st.session_state["pending_interrupt"])
