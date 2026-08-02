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

    # ── Chat jump filter (set by chat page when a memory source is clicked) ──
    filter_id = st.session_state.get("mem_filter_id")
    if filter_id:
        st.info(f"🔗 从聊天跳转 — 正在查找记忆 `{str(filter_id)[:8]}…`")
        found_mem: dict | None = None
        with st.spinner("正在定位记忆…"):
            try:
                client = _get_client()
                resp = client.get(
                    f"/api/memory/memories/{filter_id}",
                    timeout=10,
                )
                if resp.status_code == 200:
                    found_mem = resp.json()
                # 404 or other errors → found_mem stays None, warning shown below
            except Exception as exc:
                st.error(f"查找记忆失败: {exc}")

        if found_mem:
            _render_memory_card(found_mem)
        else:
            st.warning("未找到该记忆，可能已被删除或尚未建立索引。")

        if st.button("清除筛选", key="mem_filter_clear_btn", use_container_width=True):
            del st.session_state["mem_filter_id"]
            st.rerun()

    st.divider()

    # ── Ingest section ──
    with st.expander("📥 摄入文档", expanded=False):
        tab1, tab2 = st.tabs(["粘贴文本", "上传文件"])

        with tab1:
            ingest_text = st.text_area(
                "文本内容", placeholder="粘贴要摄入的文档、代码或任意文本…",
                height=180, key="ingest_text_area",
                label_visibility="collapsed",
            )
            ingest_name = st.text_input(
                "文档名称", placeholder="用于标识这段内容（如 README、app.py）",
                key="ingest_text_name", label_visibility="collapsed",
            )
            if st.button("摄入文本", key="ingest_text_btn", use_container_width=True,
                         disabled=not (ingest_text.strip() and ingest_name.strip())):
                with st.spinner("正在分块、嵌入、入库…"):
                    try:
                        client = _get_client()
                        r = client.post(
                            "/api/memory/ingest",
                            json={
                                "document_id": ingest_name.strip(),
                                "content": ingest_text,
                            },
                            timeout=60,
                        )
                        if r.status_code == 200:
                            data = r.json()
                            st.toast(f"✅ 已摄入：{data['chunks_written']} 个块", icon="✅")
                            _fetch_stats.clear()
                        else:
                            st.toast(f"摄入失败 ({r.status_code})", icon="❌")
                    except Exception as exc:
                        st.toast(f"摄入失败: {exc}", icon="❌")

        with tab2:
            uploaded = st.file_uploader(
                "选择文件", type=["txt", "md", "py", "js", "ts", "json", "yaml", "yml",
                                     "toml", "cfg", "ini", "sql", "html", "css", "sh",
                                     "java", "go", "rs", "c", "cpp", "h", "rb", "php"],
                key="ingest_file_uploader", label_visibility="collapsed",
            )
            if uploaded is not None:
                st.caption(f"已选择: `{uploaded.name}` ({uploaded.size:,} bytes)")
                if st.button("摄入文件", key="ingest_file_btn", use_container_width=True):
                    with st.spinner("正在分块、嵌入、入库…"):
                        try:
                            raw = uploaded.read()
                            content = raw.decode("utf-8")
                        except UnicodeDecodeError:
                            try:
                                content = raw.decode("latin-1")
                            except Exception:
                                st.toast("无法解码文件内容，请使用 UTF-8 编码的文本文件", icon="❌")
                                content = None
                        if content:
                            try:
                                client = _get_client()
                                r = client.post(
                                    "/api/memory/ingest",
                                    json={
                                        "document_id": uploaded.name,
                                        "content": content,
                                    },
                                    timeout=120,
                                )
                                if r.status_code == 200:
                                    data = r.json()
                                    st.toast(f"✅ 已摄入: {uploaded.name} → {data['chunks_written']} 个块", icon="✅")
                                    _fetch_stats.clear()
                                else:
                                    st.toast(f"摄入失败 ({r.status_code})", icon="❌")
                            except Exception as exc:
                                st.toast(f"摄入失败: {exc}", icon="❌")

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
            st.info("请输入搜索关键词。")
        else:
            with st.spinner("搜索中…"):
                try:
                    client = _get_client()
                    resp = client.post(
                        "/api/memory/memories/search",
                        json={"query": query.strip(), "top_k": top_k},
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
