"""History chat page — messages + input pinned to viewport bottom."""

from __future__ import annotations

import streamlit as st

from frontend.app import (
    _get_client,
    _handle_chat_input,
    _render_approval,
    _render_message,
)

# ── CSS: input fixed to bottom, user messages right-aligned ──
st.html(
    "<style>"
    "  hr { display: none !important; }"
    "  .stMainBlockContainer + div { display: none !important; }"
    "  .stMainBlockContainer { border-bottom: none !important;"
    "    padding-top: 0.5rem !important;"
    "    padding-bottom: 100px !important; }"
    "  /* input follows the same margin as the main content (sidebar-driven) */"
    "  [data-testid='stChatInput'] {"
    "    position: fixed !important; bottom: 0.1rem !important;"
    "    z-index: 100 !important;"
    "    max-width: 720px !important;"
    "    margin: 0 auto !important;"
    "    background: var(--default-backgroundColor) !important;"
    "    padding: 0.75rem 0 0.5rem 0 !important;"
    "    transition: none !important; }"
    "  [data-testid='stLayoutWrapper'] {"
    "    max-width: 720px !important;"
    "    margin: 0 auto !important; }"
    "  [data-testid='stChatMessage']:has([data-testid='stChatMessageAvatarUser']) {"
    "    flex-direction: row-reverse !important; }"
    "  [data-testid='stChatMessage']:has([data-testid='stChatMessageAvatarUser']) "
    "  .stMarkdown,"
    "  [data-testid='stChatMessage']:has([data-testid='stChatMessageAvatarUser']) "
    "  [data-testid='stMarkdownContainer'] {"
    "    text-align: right !important; }"
    "</style>"
)

# JS: dynamically sync input position/width with stLayoutWrapper
st.components.v1.html(
    "<script>"
    "var lastL='',lastW='';"
    "setInterval(function(){"
    "  var w=parent.document.querySelector('[data-testid=\"stLayoutWrapper\"]');"
    "  var c=parent.document.querySelector('[data-testid=\"stChatInput\"]');"
    "  if(!w||!c)return;"
    "  var r=w.getBoundingClientRect();"
    "  var nl=r.left+'px',nw=r.width+'px';"
    "  if(nl!==lastL||nw!==lastW){"
    "    lastL=nl;lastW=nw;"
    "    c.style.left=nl;c.style.width=nw;"
    "  }"
    "},16);"
    "</script>",
    height=1,
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
