"""EMA — Engineering Memory Agent — Streamlit MVP."""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="EMA", page_icon="🧠", layout="wide")

# Center the title and caption
st.markdown(
    """
    <style>
    .ema-header {
        text-align: center;
        padding-top: 10vh;
    }
    .ema-header h1 {
        font-size: 2.5rem;
    }
    .ema-header p {
        color: #888;
        font-size: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="ema-header">'
    '<h1>🧠 EMA — Engineering Memory Agent</h1>'
    '<p>研发团队长期记忆智能体</p>'
    '</div>',
    unsafe_allow_html=True,
)
