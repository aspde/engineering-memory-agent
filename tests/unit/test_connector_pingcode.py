"""Unit tests for PingCodeConnector — pure data transformation, no IO."""

import pytest

from backend.connectors.pingcode import PingCodeConnector


def _make_payload(
    item_id="WI-123",
    title="登录超时问题",
    item_type="缺陷",
    status="已解决",
    description="用户登录后5分钟自动退出",
    resolution="增加session超时到30分钟",
):
    return {
        "event_type": "workitem.updated",
        "workitem": {
            "id": item_id,
            "title": title,
            "type": item_type,
            "status": status,
            "description": description,
            "resolution": resolution,
            "iteration": "Sprint 12",
            "assignee": "张三",
            "priority": "高",
        },
    }


class TestPingCodeValidate:
    def test_valid_payload_accepted(self):
        conn = PingCodeConnector()
        assert conn.validate(_make_payload()) is True

    def test_missing_workitem_rejected(self):
        conn = PingCodeConnector()
        assert conn.validate({}) is False

    def test_workitem_not_dict_rejected(self):
        conn = PingCodeConnector()
        assert conn.validate({"workitem": "string"}) is False

    def test_missing_id_rejected(self):
        conn = PingCodeConnector()
        p = _make_payload()
        del p["workitem"]["id"]
        assert conn.validate(p) is False

    def test_missing_title_rejected(self):
        conn = PingCodeConnector()
        p = _make_payload()
        del p["workitem"]["title"]
        assert conn.validate(p) is False


class TestPingCodeNormalize:
    def test_includes_item_id_and_title(self):
        conn = PingCodeConnector()
        result = conn.normalize(_make_payload())
        assert "WI-123" in result
        assert "登录超时问题" in result

    def test_includes_type_and_status(self):
        conn = PingCodeConnector()
        result = conn.normalize(_make_payload())
        assert "缺陷" in result
        assert "已解决" in result

    def test_includes_description(self):
        conn = PingCodeConnector()
        result = conn.normalize(_make_payload())
        assert "用户登录后5分钟自动退出" in result

    def test_includes_resolution(self):
        conn = PingCodeConnector()
        result = conn.normalize(_make_payload())
        assert "增加session超时到30分钟" in result

    def test_includes_priority_and_assignee(self):
        conn = PingCodeConnector()
        result = conn.normalize(_make_payload())
        assert "高" in result
        assert "张三" in result

    def test_includes_iteration(self):
        conn = PingCodeConnector()
        result = conn.normalize(_make_payload())
        assert "Sprint 12" in result

    def test_handles_missing_optional_fields(self):
        conn = PingCodeConnector()
        p = _make_payload(description="", resolution="")
        p["workitem"]["priority"] = ""
        del p["workitem"]["iteration"]
        result = conn.normalize(p)
        # Should not crash — just check key fields present
        assert "WI-123" in result


class TestPingCodeSourceType:
    @pytest.mark.asyncio
    async def test_bug_uses_pingcode_bug(self, monkeypatch):
        from backend.service import memory as mem_module

        calls: list[dict] = []

        async def _fake_write(content, source_type, metadata):
            calls.append({"source_type": source_type})
            return {"id": "x", "action": "inserted", "summary": content}

        monkeypatch.setattr(mem_module, "write_memory", _fake_write)

        conn = PingCodeConnector()
        await conn.process("content", {"item_type": "缺陷"})

        assert calls[0]["source_type"] == "pingcode_bug"

    @pytest.mark.asyncio
    async def test_requirement_uses_pingcode(self, monkeypatch):
        from backend.service import memory as mem_module

        calls: list[dict] = []

        async def _fake_write(content, source_type, metadata):
            calls.append({"source_type": source_type})
            return {"id": "x", "action": "inserted", "summary": content}

        monkeypatch.setattr(mem_module, "write_memory", _fake_write)

        conn = PingCodeConnector()
        await conn.process("content", {"item_type": "需求"})

        assert calls[0]["source_type"] == "pingcode"


class TestPingCodeBuildMetadata:
    def test_includes_item_info(self):
        conn = PingCodeConnector()
        p = _make_payload()
        # Fix key to "type" since that's what _make_payload produces
        p["workitem"]["type"] = "需求"
        meta = conn.build_metadata(p)
        assert meta["item_id"] == "WI-123"
        assert meta["item_type"] == "需求"

    def test_source_url_when_base_url_set(self, monkeypatch):
        monkeypatch.setenv("WEBHOOK_PINGCODE_BASE_URL", "https://team.pingcode.com")
        conn = PingCodeConnector()
        meta = conn.build_metadata(_make_payload())
        assert meta["source_url"] == "https://team.pingcode.com/workitem/WI-123"
