"""Tests for the shared Feishu notification service (notification.py).

The agent tool and the event-analysis notifier both delegate here, so
the payload building and error degradation are pinned once.  The fake
httpx client is imported from ``test_notify_feishu`` — one fake per
behaviour, not one per test module.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.service.notification import (
    build_feishu_payload,
    send_feishu_message,
)
from tests.unit.test_notify_feishu import _FakeAsyncClient


def _ok_response() -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = lambda: None
    resp.json.return_value = {"code": 0}
    return resp


class TestBuildPayload:
    def test_text_payload(self):
        payload = build_feishu_payload("hello")
        assert payload == {"msg_type": "text", "content": {"text": "hello"}}

    def test_interactive_card_payload(self):
        payload = build_feishu_payload("**body**", msg_type="interactive", title="T")
        assert payload["msg_type"] == "interactive"
        assert payload["card"]["header"]["title"]["content"] == "T"
        assert payload["card"]["elements"] == [{"tag": "markdown", "content": "**body**"}]

    def test_default_title_when_none(self):
        payload = build_feishu_payload("m", msg_type="interactive", title=None)
        assert "EMA" in payload["card"]["header"]["title"]["content"]


class TestSendFeishuMessage:
    @pytest.mark.asyncio
    async def test_no_webhook_url_fails_closed_with_detail(self, monkeypatch):
        monkeypatch.setattr("backend.shared.config.config.feishu_webhook_url", "")
        ok, detail = await send_feishu_message("m")
        assert ok is False
        assert "FEISHU_WEBHOOK_URL" in detail

    @pytest.mark.asyncio
    async def test_success_returns_code(self, monkeypatch):
        monkeypatch.setattr(
            "backend.shared.config.config.feishu_webhook_url",
            "https://open.feishu.cn/open-apis/bot/v2/hook/t",
        )
        client = _FakeAsyncClient()
        client.post = AsyncMock(return_value=_ok_response())
        with patch("httpx.AsyncClient", return_value=client):
            ok, detail = await send_feishu_message("m", msg_type="interactive", title="T")
        assert ok is True
        # int, matching the historical feishu_status contract of the tool.
        assert detail == 0
        assert isinstance(detail, int)
        # The interactive card went out as one POST with the built payload.
        sent = client.post.await_args.kwargs["json"]
        assert sent["msg_type"] == "interactive"

    @pytest.mark.asyncio
    async def test_timeout_degrades_to_failure_not_raise(self, monkeypatch):
        import httpx

        monkeypatch.setattr(
            "backend.shared.config.config.feishu_webhook_url",
            "https://open.feishu.cn/open-apis/bot/v2/hook/t",
        )
        client = _FakeAsyncClient()

        async def _timeout(*args, **kwargs):
            raise httpx.TimeoutException("timed out")

        client.post = _timeout
        with patch("httpx.AsyncClient", return_value=client):
            ok, detail = await send_feishu_message("m")
        assert ok is False
        assert "timed out" in detail.lower() or "timeout" in detail.lower()

    @pytest.mark.asyncio
    async def test_unexpected_error_degrades_to_failure(self, monkeypatch):
        monkeypatch.setattr(
            "backend.shared.config.config.feishu_webhook_url",
            "https://open.feishu.cn/open-apis/bot/v2/hook/t",
        )
        client = _FakeAsyncClient()

        async def _boom(*args, **kwargs):
            raise RuntimeError("dns exploded")

        client.post = _boom
        with patch("httpx.AsyncClient", return_value=client):
            ok, detail = await send_feishu_message("m")
        assert ok is False
        assert "dns exploded" in detail
