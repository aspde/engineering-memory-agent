"""New conversation page — title + centred input, no messages."""

from __future__ import annotations

import streamlit as st

from frontend.app import _handle_chat_input

# ── CSS: input centred vertically, no fixed positioning ──
st.markdown(
    "<style>"
    "  hr { display: none !important; }"
    "  .stMainBlockContainer + div { display: none !important; }"
    "  .stMainBlockContainer { border-bottom: none !important; }"
    "  [data-testid='stChatInput'] {"
    "    max-width: 720px !important;"
    "    margin: 0 auto !important; }"
    "</style>",
    unsafe_allow_html=True,
)

# ── Title ──
col1, col2, col3 = st.columns([0.5, 4, 0.5])
with col2:
    st.markdown(
        "<h1 style='text-align: center; margin-bottom: 0.75rem;'>"
        "EMA — Engineering Memory Agent</h1>",
        unsafe_allow_html=True,
    )

# ── Chat input ──
@st.fragment
def _chat_fragment() -> None:
    _handle_chat_input()

_chat_fragment()
