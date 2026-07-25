"""Memory library page — stats dashboard, browse, and search."""

from __future__ import annotations

import httpx
import streamlit as st

from frontend.app import _get_client


# ═══════════════════════════════════════════════════════════════════════
# Stats (cached, shared across reruns)
# ═══════════════════════════════════════════════════════════════════════


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_stats() -> dict | None:
    """Fetch memory statistics from the backend (cached 60 s)."""
    try:
        client = _get_client()
        resp = client.get("/api/memory/stats", timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        pass
    return None


def _render_stats(stats: dict) -> None:
    """Render the stats dashboard panel."""
    # ── KPI row ──
    cols = st.columns(4)
    cols[0].metric("总记忆数", stats["total_memories"])
    cols[1].metric("文档块数", stats["total_chunks"])
    cols[2].metric("对话数", stats["total_conversations"])
    cols[3].metric("近 7 天新增", stats["recent_count_7d"])

    col_a, col_b = st.columns(2)

    # ── Source distribution (left) ──
    with col_a:
        st.caption("**来源分布**")
        by_source = stats.get("by_source_type", [])
        if by_source:
            max_count = max(item["count"] for item in by_source)
            for item in by_source:
                label = item["source_type"]
                count = item["count"]
                ratio = count / max_count if max_count > 0 else 0
                st.markdown(
                    f"`{label}`  "
                    f"<progress value='{ratio}' "
                    f"style='width:120px;height:14px;vertical-align:middle'></progress> "
                    f"**{count}**",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("暂无数据")

    # ── Top entities (right) ──
    with col_b:
        st.caption("**高频实体**")
        top_entities = stats.get("top_entities", [])
        if top_entities:
            badges = " ".join(
                f"<code style='background:#e8e8e8;padding:2px 8px;border-radius:10px;"
                f"margin:2px;display:inline-block'>{e['name']}</code>"
                for e in top_entities
            )
            st.markdown(badges, unsafe_allow_html=True)
        else:
            st.caption("暂无数据")


# ═══════════════════════════════════════════════════════════════════════
# Search & memory cards
# ═══════════════════════════════════════════════════════════════════════


def _render_memory_card(mem: dict) -> None:
    """Render a single memory as an expandable card."""
    summary = mem.get("summary", "(no summary)")
    source_type = mem.get("source_type", "unknown")
    decay = mem.get("decay_factor", 1.0)
    created = mem.get("created_at", "")

    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{summary[:120]}{'…' if len(summary) > 120 else ''}**")
        with col2:
            st.caption(f"`{source_type}`")

        cols = st.columns(3)
        cols[0].metric("Decay", f"{decay:.2f}")
        if created:
            cols[1].caption(f"Created: {created[:19]}")
        mem_id = mem.get("id", "")
        if mem_id:
            cols[2].caption(f"ID: `{str(mem_id)[:8]}…`")

        if len(summary) > 120:
            with st.expander("Full summary"):
                st.markdown(summary)


def main() -> None:
    st.markdown(
        "<style>"
        "  .stMainBlockContainer { max-width: 960px !important; }"
        "</style>",
        unsafe_allow_html=True,
    )
    st.title("📚 记忆库")

    # ── Stats dashboard ──
    stats = _fetch_stats()
    if stats:
        _render_stats(stats)
    else:
        st.caption("无法加载统计数据，请确认后端服务已启动。")

    st.divider()
    st.caption("**搜索记忆**")

    # ── Search bar ──
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "搜索记忆…", placeholder="输入关键词搜索记忆库",
            key="mem_search_main", label_visibility="collapsed",
        )
    with col2:
        top_k = st.selectbox(
            "结果数", [5, 10, 20, 30, 50], index=2,
            format_func=lambda x: f"{x} 条",
            key="mem_topk_main", label_visibility="collapsed",
        )

    if st.button("🔍 搜索", use_container_width=True, key="mem_search_btn_main"):
        if not query.strip():
            query = ""

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
