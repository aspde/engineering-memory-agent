"""Tests for notify_feishu_tool — message formatting and error handling."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class _FakeAsyncClient:
    """A fake httpx.AsyncClient that supports ``async with``."""

    def __init__(self, **kwargs):
        self.post = MagicMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class TestNotifyFeishu:
    """Verify the notify_feishu tool behaviour."""

    @pytest.mark.asyncio
    async def test_notify_feishu_formats_text_message(self, monkeypatch) -> None:
        """The tool should POST to the Feishu webhook URL with correct text payload."""
        from agent.tools import notify_feishu_tool

        monkeypatch.setattr(
            "backend.shared.config.config.feishu_webhook_url",
            "https://open.feishu.cn/open-apis/bot/v2/hook/test-token",
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = lambda: None
        mock_response.json.return_value = {"code": 0, "msg": "success"}
        fake_client = _FakeAsyncClient()
        fake_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=fake_client):
            result = await notify_feishu_tool.ainvoke({
                "message": "🔴 发现高危模式匹配",
            })

        data = json.loads(result)
        assert data["ok"] is True
        assert data["msg_type"] == "text"

        fake_client.post.assert_called_once()
        call_args = fake_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["msg_type"] == "text"
        assert "🔴" in payload["content"]["text"]

    @pytest.mark.asyncio
    async def test_notify_feishu_formats_interactive_message(self, monkeypatch) -> None:
        """Interactive cards should include header and markdown elements."""
        from agent.tools import notify_feishu_tool

        monkeypatch.setattr(
            "backend.shared.config.config.feishu_webhook_url",
            "https://open.feishu.cn/open-apis/bot/v2/hook/test-token",
        )

        mock_response = MagicMock()
        mock_response.raise_for_status = lambda: None
        mock_response.json.return_value = {"code": 0, "msg": "success"}
        fake_client = _FakeAsyncClient()
        fake_client.post = AsyncMock(return_value=mock_response)

        with patch("httpx.AsyncClient", return_value=fake_client):
            result = await notify_feishu_tool.ainvoke({
                "message": "CI 构建失败 — 匹配到历史问题",
                "msg_type": "interactive",
                "title": "🔴 CI 告警",
            })

        data = json.loads(result)
        assert data["ok"] is True
        assert data["msg_type"] == "interactive"

        call_args = fake_client.post.call_args
        payload = call_args[1]["json"]
        assert payload["msg_type"] == "interactive"
        assert payload["card"]["header"]["title"]["content"] == "🔴 CI 告警"

    @pytest.mark.asyncio
    async def test_notify_feishu_returns_error_when_not_configured(self) -> None:
        """When FEISHU_WEBHOOK_URL is empty, return ok=false."""
        from agent.tools import notify_feishu_tool

        result = await notify_feishu_tool.ainvoke({
            "message": "test",
        })

        data = json.loads(result)
        assert data["ok"] is False
        assert "FEISHU_WEBHOOK_URL" in data["error"]

    @pytest.mark.asyncio
    async def test_notify_feishu_handles_timeout(self, monkeypatch) -> None:
        """A timeout should return ok=false, not crash."""
        from agent.tools import notify_feishu_tool

        monkeypatch.setattr(
            "backend.shared.config.config.feishu_webhook_url",
            "https://open.feishu.cn/open-apis/bot/v2/hook/test-token",
        )

        import httpx

        fake_client = _FakeAsyncClient()
        fake_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))

        with patch("httpx.AsyncClient", return_value=fake_client):
            result = await notify_feishu_tool.ainvoke({
                "message": "test",
            })

        data = json.loads(result)
        assert data["ok"] is False
        assert "timed out" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_tool_schema_has_required_params(self) -> None:
        """Verify the tool has the expected parameter schema."""
        from agent.tools import notify_feishu_tool

        schema = notify_feishu_tool.args_schema.model_json_schema()
        props = schema.get("properties", {})
        assert "message" in props
        assert "msg_type" in props
        assert "title" in props
        required = schema.get("required", [])
        assert "message" in required
