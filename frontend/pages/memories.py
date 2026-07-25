"""Memory library page — browse, search, and inspect stored memories."""

from __future__ import annotations

import httpx
import streamlit as st

from frontend.app import _get_client

BACKEND_URL = "http://localhost:8000"


def _render_memory_card(mem: dict) -> None:
    """Render a single memory as an expandable card."""
    summary = mem.get("summary", "(no summary)")
    source_type = mem.get("source_type", "unknown")
    decay = mem.get("decay_factor", 1.0)
    created = mem.get("created_at", "")

    with st.container(border=True):
        # Header row
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{summary[:120]}{'…' if len(summary) > 120 else ''}**")
        with col2:
            st.caption(f"`{source_type}`")

        # Meta row
        cols = st.columns(3)
        cols[0].metric("Decay", f"{decay:.2f}")
        if created:
            cols[1].caption(f"Created: {created[:19]}")
        mem_id = mem.get("id", "")
        if mem_id:
            cols[2].caption(f"ID: `{str(mem_id)[:8]}…`")

        # Expand for full details
        if len(summary) > 120:
            with st.expander("Full summary"):
                st.markdown(summary)


def main() -> None:
    # Wider content area for the memories page
    st.markdown(
        "<style>"
        "  .stMainBlockContainer { max-width: 960px !important; }"
        "</style>",
        unsafe_allow_html=True,
    )
    st.title("📚 记忆库")
    st.caption("浏览和搜索 EMA 存储的长期记忆")

    # ── Search bar ──
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input("搜索记忆…", placeholder="输入关键词搜索记忆库", key="mem_search", label_visibility="collapsed")
    with col2:
        top_k = st.selectbox("结果数", [5, 10, 20, 30, 50], index=2, format_func=lambda x: f"{x} 条", key="mem_topk", label_visibility="collapsed")

    if st.button("🔍 搜索", use_container_width=True, key="mem_search_btn"):
        if not query.strip():
            query = ""  # show all / default

        with st.spinner("搜索中…"):
            try:
                client = _get_client()
                resp = client.post(
                    "/api/memory/memories/search",
                    json={"query": query.strip() or "*", "top_k": top_k},
                    timeout=30,
                )
                if resp.status_code == 200:
                    results = resp.json().get("results", [])
                else:
                    results = []
            except Exception as exc:
                st.error(f"搜索失败: {exc}")
                results = []

        if not results:
            st.info("没有找到匹配的记忆。尝试换个关键词，或者在聊天中让 EMA 记录一些内容。")
        else:
            st.caption(f"找到 {len(results)} 条记忆")
            for mem in results:
                _render_memory_card(mem)


main()
