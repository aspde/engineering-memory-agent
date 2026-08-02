"""Chat page — centred title when empty, conversation when messages exist."""

from __future__ import annotations

import streamlit as st

from frontend.app import (
    _get_client,
    _handle_chat_input,
    _render_approval,
    _render_message,
)

_MAX_VISIBLE = 50
msgs: list[dict] = st.session_state.get("messages", [])

# ── Lazy-load messages on thread switch ──
tid = st.session_state["thread_id"]
if st.session_state.get("_loaded_thread_id") != tid:
    # Always try the backend — _threads may not have been fetched yet
    # (sidebar renders after pages in st.navigation).
    try:
        r = _get_client().get(f"/api/agent/thread/{tid}", timeout=5)
        if r.status_code == 200:
            msgs = r.json().get("messages", [])
            st.session_state["messages"] = msgs
            st.session_state["_loaded_thread_id"] = tid
    except Exception:
        pass

# ── CSS ──
st.html(
    "<style>"
    "  hr { display: none !important; }"
    "  .stMainBlockContainer + div { display: none !important; }"
    "  .stMainBlockContainer { border-bottom: none !important;"
    "    padding-top: 0.5rem !important;"
    "    padding-bottom: 100px !important; }"
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
    "},8);"
    "</script>",
    height=1,
)


# ── Chat area: messages + title + approval + input (single fragment) ──
@st.fragment
def _chat_fragment() -> None:
    msgs: list[dict] = st.session_state.get("messages", [])

    if not msgs:
        col1, col2, col3 = st.columns([0.5, 4, 0.5])
        with col2:
            st.markdown(
                "<h1 style='text-align: center; margin-bottom: 0.75rem;'>"
                "EMA — Engineering Memory Agent</h1>",
                unsafe_allow_html=True,
            )

    visible = msgs[-_MAX_VISIBLE:]
    if len(msgs) > _MAX_VISIBLE:
        st.caption(f"*... {len(msgs) - _MAX_VISIBLE} older messages hidden*")
    for msg in visible:
        _render_message(msg)

    if st.session_state["waiting_for_approval"] and st.session_state["pending_interrupt"]:
        _render_approval(st.session_state["pending_interrupt"])

    _handle_chat_input()

_chat_fragment()
