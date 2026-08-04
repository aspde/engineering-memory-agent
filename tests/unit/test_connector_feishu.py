"""Unit tests for FeishuConnector — pure data transformation, no IO."""

from backend.connectors.feishu import FeishuConnector


def _make_thread_payload(messages=None, chat_id="oc_abc"):
    if messages is None:
        messages = [
            {"sender": "张三", "content": "登录接口超时了怎么修？"},
            {"sender": "李四", "content": "连接池不够了，加大点"},
            {"sender": "王五", "content": "记一下，这个方案加到运维手册"},
        ]
    return {
        "event_type": "message",
        "event": {
            "type": "message",
            "chat_id": chat_id,
            "thread_id": "ot_123",
            "messages": messages,
        },
    }


def _make_single_payload(sender="张三", content="部署步骤确认一下", chat_id="oc_abc"):
    return {
        "event_type": "message",
        "event": {
            "type": "message",
            "chat_id": chat_id,
            "sender": sender,
            "content": content,
            "message_id": "om_456",
        },
    }


class TestFeishuValidate:
    def test_valid_message_accepted(self):
        conn = FeishuConnector()
        assert conn.validate(_make_single_payload()) is True

    def test_mention_accepted(self):
        conn = FeishuConnector()
        p = _make_single_payload()
        p["event"]["type"] = "mention"
        assert conn.validate(p) is True

    def test_im_message_receive_accepted(self):
        conn = FeishuConnector()
        p = _make_single_payload()
        p["event"]["type"] = "im_message_receive_v1"
        assert conn.validate(p) is True

    def test_unknown_type_rejected(self):
        conn = FeishuConnector()
        p = _make_single_payload()
        p["event"]["type"] = "reaction_added"
        assert conn.validate(p) is False

    def test_missing_event_rejected(self):
        conn = FeishuConnector()
        assert conn.validate({}) is False


class TestFeishuNormalizeThread:
    def test_formats_thread_as_conversation(self):
        conn = FeishuConnector()
        result = conn.normalize(_make_thread_payload())
        assert "飞书话题讨论" in result
        assert "[张三]: 登录接口超时了怎么修？" in result
        assert "[李四]: 连接池不够了，加大点" in result

    def test_preserves_message_order(self):
        conn = FeishuConnector()
        msgs = [
            {"sender": "A", "content": "第一条"},
            {"sender": "B", "content": "第二条"},
        ]
        result = conn.normalize(_make_thread_payload(messages=msgs))
        assert result.index("第一条") < result.index("第二条")

    def test_filters_system_sender(self):
        conn = FeishuConnector()
        msgs = [
            {"sender": "bot_notify", "content": "自动提醒"},
            {"sender": "张三", "content": "真实消息"},
        ]
        result = conn.normalize(_make_thread_payload(messages=msgs))
        assert "真实消息" in result
        assert "自动提醒" not in result

    def test_filters_short_messages(self):
        conn = FeishuConnector()
        msgs = [
            {"sender": "张三", "content": "嗯"},
            {"sender": "李四", "content": "具体方案是这样的"},
        ]
        result = conn.normalize(_make_thread_payload(messages=msgs))
        assert "具体方案" in result
        assert "嗯" not in result


class TestFeishuNormalizeSingle:
    def test_formats_single_message(self):
        conn = FeishuConnector()
        result = conn.normalize(_make_single_payload(content="部署步骤"))
        assert "飞书消息" in result
        assert "[张三]" in result
        assert "部署步骤" in result

    def test_filters_system_sender(self):
        conn = FeishuConnector()
        result = conn.normalize(_make_single_payload(sender="bot_helper", content="提醒"))
        assert result == ""

    def test_filters_empty_content(self):
        conn = FeishuConnector()
        result = conn.normalize(_make_single_payload(content=""))
        assert result == ""


class TestFeishuBuildMetadata:
    def test_detects_remember_keywords(self):
        conn = FeishuConnector()
        for kw in ("记一下", "备忘", "记住", "记录下来"):
            meta = conn.build_metadata(_make_single_payload(content=f"这个方案要{kw}"))
            assert meta.get("auto_ingest") is True, f"keyword '{kw}' not detected"

    def test_no_auto_ingest_for_normal_message(self):
        conn = FeishuConnector()
        meta = conn.build_metadata(_make_single_payload(content="普通消息"))
        assert "auto_ingest" not in meta

    def test_keyword_in_thread_triggers_auto_ingest(self):
        conn = FeishuConnector()
        msgs = [
            {"sender": "张三", "content": "讨论"},
            {"sender": "李四", "content": "这个记一下"},
        ]
        meta = conn.build_metadata(_make_thread_payload(messages=msgs))
        assert meta.get("auto_ingest") is True

    def test_source_url_when_tenant_configured(self, monkeypatch):
        monkeypatch.setenv("WEBHOOK_FEISHU_TENANT", "mycorp")
        conn = FeishuConnector()
        meta = conn.build_metadata(_make_single_payload())
        assert "source_url" in meta
        assert "mycorp.feishu.cn" in meta["source_url"]
