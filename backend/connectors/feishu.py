"""飞书（Feishu/Lark）连接器 — 将飞书消息和讨论转化为结构化记忆。

支持两种模式：
- 单条消息：用户在群聊中 @机器人 或发送特定关键词
- 话题回复：一个话题下的多条消息，格式化为对话文本

飞书机器人收到的消息通过 webhook 转发到 EMA，消息中如含
"记一下"、"备忘"、"记住" 等关键词，自动标记为高优先级摄入。
"""

from __future__ import annotations

from typing import Any

from backend.connectors.base import Connector

# 自动摄入触发词
_AUTO_INGEST_KEYWORDS = ("记一下", "备忘", "记住", "记录下来", "mark")

# 要过滤掉的系统消息类型
_SYSTEM_SENDER_PREFIXES = ("bot_", "system_", "service_")


class FeishuConnector(Connector):
    """接收飞书消息 webhook，转化为 EMA 结构化记忆。"""

    display_name = "飞书"

    @property
    def source_type(self) -> str:
        return "feishu"

    # ── Connector ABC ─────────────────────────────────────────────────

    def validate(self, payload: dict[str, Any]) -> bool:
        """Payload 需包含 event，event type 为 message 或 mention。"""
        event = payload.get("event") if isinstance(payload.get("event"), dict) else None
        if event is None:
            return False
        event_type: str = event.get("type", "")
        return event_type in ("message", "mention", "im_message_receive_v1")

    def normalize(self, payload: dict[str, Any]) -> str:
        """将飞书事件 payload 转换为 EMA 标准文本。"""
        event: dict[str, Any] = payload.get("event", {})

        # 话题模式：多条消息
        messages: list[dict[str, Any]] = event.get("messages") or []
        if messages:
            return self._normalize_thread(messages)

        # 单条消息
        return self._normalize_single(event)

    def build_metadata(self, payload: dict[str, Any]) -> dict[str, Any]:
        """提取飞书消息的可溯源元数据。"""
        event: dict[str, Any] = payload.get("event", {})
        chat_id: str = event.get("chat_id", "")
        thread_id: str = event.get("thread_id", "") or event.get("message_id", "")

        meta: dict[str, Any] = {
            "chat_id": chat_id,
            "thread_id": thread_id,
        }

        # 检测自动摄入关键词
        text = self._collect_text(event)
        for kw in _AUTO_INGEST_KEYWORDS:
            if kw in text:
                meta["auto_ingest"] = True
                meta["trigger_keyword"] = kw
                break

        # 生成飞书消息链接
        import os

        tenant = os.getenv("WEBHOOK_FEISHU_TENANT", "")
        if tenant and chat_id and thread_id:
            meta["source_url"] = (
                f"https://{tenant}.feishu.cn/chat/{chat_id}?thread_id={thread_id}"
            )

        return meta

    # ── normalize helpers ──────────────────────────────────────────────

    def _normalize_thread(self, messages: list[dict[str, Any]]) -> str:
        """将话题消息列表格式化为对话文本。"""
        filtered: list[dict[str, Any]] = []
        for m in messages:
            sender: str = m.get("sender", m.get("sender_name", ""))
            content: str = m.get("content", m.get("text", "")).strip()
            # 过滤系统消息
            if any(sender.startswith(p) for p in _SYSTEM_SENDER_PREFIXES):
                continue
            # 过滤空消息和纯表情
            if not content or len(content) < 2:
                continue
            filtered.append({"sender": sender, "content": content})

        if not filtered:
            return "飞书话题（过滤后无实质内容）"

        lines: list[str] = ["飞书话题讨论："]
        for m in filtered:
            lines.append(f"  [{m['sender']}]: {m['content']}")

        return "\n".join(lines)

    def _normalize_single(self, event: dict[str, Any]) -> str:
        """格式化单条飞书消息。"""
        sender: str = event.get("sender", event.get("sender_name", ""))

        # 过滤系统消息
        if any(sender.startswith(p) for p in _SYSTEM_SENDER_PREFIXES):
            return ""

        content: str = event.get("content", event.get("text", "")).strip()
        if not content:
            return ""

        chat_name: str = event.get("chat_name", "")
        location = f"（来自 {chat_name}）" if chat_name else ""
        return f"飞书消息 [{sender}]{location}：\n  {content}"

    @staticmethod
    def _collect_text(event: dict[str, Any]) -> str:
        """收集事件中所有文本内容，用于关键词检测。"""
        parts: list[str] = []
        for m in event.get("messages") or []:
            t = m.get("content", m.get("text", ""))
            if t:
                parts.append(t)
        if not parts:
            t = event.get("content", event.get("text", ""))
            if t:
                parts.append(t)
        return " ".join(parts)
