"""PingCode 连接器 — 将 PingCode 工作项变更转化为结构化记忆。

PingCode 是国内常用的研发管理平台，涵盖需求、缺陷、任务、迭代等。
当工作项状态变更（如缺陷修复、需求完成）时，通过 webhook 推送到 EMA。
"""

from __future__ import annotations

from typing import Any

from backend.connectors.base import Connector


class PingCodeConnector(Connector):
    """接收 PingCode webhook，将工作项转化为 EMA 结构化记忆。"""

    display_name = "PingCode"

    @property
    def source_type(self) -> str:
        return "pingcode"

    # ── Connector ABC ─────────────────────────────────────────────────

    def validate(self, payload: dict[str, Any]) -> bool:
        """Payload 需包含 workitem.id 和 workitem.title。"""
        item = payload.get("workitem") if isinstance(payload.get("workitem"), dict) else None
        if item is None:
            return False
        if not item.get("id") or not item.get("title"):
            return False
        return True

    def normalize(self, payload: dict[str, Any]) -> str:
        """将 PingCode 工作项 payload 转换为 EMA 标准文本。"""
        item: dict[str, Any] = payload["workitem"]
        item_id: str = item.get("id", "")
        title: str = item.get("title", "")
        item_type: str = item.get("type", item.get("item_type", ""))  # 需求/缺陷/任务
        status: str = item.get("status", "")
        description: str = item.get("description", "") or ""
        resolution: str = item.get("resolution", "") or ""
        iteration: str = item.get("iteration", item.get("sprint", "")) or ""
        assignee: str = item.get("assignee", item.get("owner", "")) or ""
        priority: str = item.get("priority", "") or ""

        parts: list[str] = [f"PingCode {item_type}: #{item_id} — {title}"]
        if status:
            parts.append(f"状态: {status}")
        if priority:
            parts.append(f"优先级: {priority}")
        if assignee:
            parts.append(f"负责人: {assignee}")
        if iteration:
            parts.append(f"迭代: {iteration}")
        if description:
            parts.append(f"描述:\n{description.strip()}")
        if resolution:
            parts.append(f"解决方案: {resolution}")

        return "\n\n".join(parts)

    def build_metadata(self, payload: dict[str, Any]) -> dict[str, Any]:
        """提取 PingCode 工作项的可溯源元数据。"""
        item: dict[str, Any] = payload.get("workitem", {})
        item_id: str = item.get("id", "")
        item_type: str = item.get("type", item.get("item_type", ""))

        meta: dict[str, Any] = {
            "item_id": item_id,
            "item_type": item_type,
            "status": item.get("status", ""),
        }

        # 生成 PingCode 工作项链接
        import os

        base = os.getenv("WEBHOOK_PINGCODE_BASE_URL", "")
        if base and item_id:
            meta["source_url"] = f"{base.rstrip('/')}/workitem/{item_id}"

        return meta

    async def process(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """写入记忆，缺陷类型使用 pingcode_bug 标记。"""
        from backend.service.memory import write_memory

        meta = metadata or {}
        item_type: str = meta.get("item_type", "")
        effective_source = (
            "pingcode_bug" if item_type in ("缺陷", "bug", "故障") else "pingcode"
        )

        return await write_memory(content, source_type=effective_source, metadata=meta)
