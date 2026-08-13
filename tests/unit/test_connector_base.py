"""Unit tests for the Connector ABC and registry."""

import pytest

from backend.connectors.base import Connector
from backend.connectors.registry import (
    CONNECTOR_REGISTRY,
    get_connector,
    list_connectors,
    register_connector,
)

# ── Minimal concrete connector for testing ────────────────────────────


class _TestConnector(Connector):
    """A minimal, fully implemented connector for testing the ABC."""

    display_name = "Test Source"

    @property
    def source_type(self) -> str:
        return "test_source"

    def validate(self, payload: dict) -> bool:
        return "data" in payload

    def normalize(self, payload: dict) -> str:
        return f"Test: {payload.get('data', '')}"


# ── ABC tests ─────────────────────────────────────────────────────────


class TestConnectorABC:
    def test_cannot_instantiate_abstract_class(self):
        """Instantiating the ABC directly should raise TypeError."""
        with pytest.raises(TypeError):
            Connector()  # type: ignore[abstract]

    def test_concrete_subclass_instantiates(self):
        """A concrete subclass with all abstract methods can be created."""
        conn = _TestConnector()
        assert conn.source_type == "test_source"
        assert conn.display_name == "Test Source"

    def test_validate_delegates_to_subclass(self):
        conn = _TestConnector()
        assert conn.validate({"data": "hello"}) is True
        assert conn.validate({"other": "nope"}) is False

    def test_normalize_delegates_to_subclass(self):
        conn = _TestConnector()
        result = conn.normalize({"data": "hello world"})
        assert result == "Test: hello world"

    def test_default_build_metadata_returns_empty(self):
        """build_metadata() default returns an empty dict."""
        conn = _TestConnector()
        assert conn.build_metadata({"data": "x"}) == {}

    def test_default_supports_batch_is_false(self):
        conn = _TestConnector()
        assert conn.supports_batch is False

    def test_default_batch_mode_is_pending(self):
        conn = _TestConnector()
        assert conn.batch_mode == "pending"

    def test_batch_mode_supported_when_supports_batch_true(self):
        class _BatchConn(_TestConnector):
            @property
            def source_type(self) -> str:
                return "batch"

            @property
            def supports_batch(self) -> bool:
                return True

        conn = _BatchConn()
        assert conn.batch_mode == "supported"

    def test_batch_mode_not_applicable_when_overridden(self):
        class _NoBatchConn(_TestConnector):
            @property
            def source_type(self) -> str:
                return "no_batch"

            @property
            def batch_mode(self) -> str:
                return "not_applicable"

        conn = _NoBatchConn()
        assert conn.batch_mode == "not_applicable"

    def test_default_normalize_batch_loops_normalize(self):
        conn = _TestConnector()
        payloads = [{"data": "a"}, {"data": "b"}, {"data": "c"}]
        results = conn.normalize_batch(payloads)
        assert results == ["Test: a", "Test: b", "Test: c"]


class TestConnectorProcess:
    """The default process() calls write_memory() — test that wiring."""

    @pytest.mark.asyncio
    async def test_default_process_calls_write_memory(self, monkeypatch):
        """Default process() delegates to write_memory with correct params."""
        from backend.service import memory as mem_module

        calls: list[dict] = []

        async def _fake_write_memory(content, source_type, metadata):
            calls.append(
                {"content": content, "source_type": source_type, "metadata": metadata}
            )
            return {"id": "fake-id", "action": "inserted", "summary": content}

        monkeypatch.setattr(mem_module, "write_memory", _fake_write_memory)

        conn = _TestConnector()
        result = await conn.process("hello", {"key": "val"})

        assert result["id"] == "fake-id"
        assert len(calls) == 1
        assert calls[0]["content"] == "hello"
        assert calls[0]["source_type"] == "test_source"
        assert calls[0]["metadata"] == {"key": "val"}

    @pytest.mark.asyncio
    async def test_default_process_metadata_none(self, monkeypatch):
        """Default process() handles metadata=None gracefully."""
        from backend.service import memory as mem_module

        calls: list[dict] = []

        async def _fake_write_memory(content, source_type, metadata):
            calls.append(
                {"content": content, "source_type": source_type, "metadata": metadata}
            )
            return {"id": "id2", "action": "inserted", "summary": content}

        monkeypatch.setattr(mem_module, "write_memory", _fake_write_memory)

        conn = _TestConnector()
        await conn.process("content only")

        assert calls[0]["metadata"] is None


# ── Registry tests ────────────────────────────────────────────────────


class TestRegistry:
    def teardown_method(self):
        """Clean up registry between tests."""
        CONNECTOR_REGISTRY.clear()

    def test_register_and_get_connector(self):
        conn = _TestConnector()
        register_connector("test", conn)
        retrieved = get_connector("test")
        assert retrieved is conn

    def test_get_unregistered_returns_none(self):
        assert get_connector("nonexistent") is None

    def test_list_connectors_returns_metadata(self):
        register_connector("test", _TestConnector(), status="active")
        items = list_connectors()
        assert len(items) == 1
        assert items[0]["source_type"] == "test"
        assert items[0]["display_name"] == "Test Source"
        assert items[0]["status"] == "active"
        assert items[0]["batch_mode"] == "pending"

    def test_list_connectors_multiple(self):
        register_connector("a", _TestConnector(), status="active")

        class _PendingConnector(_TestConnector):
            @property
            def source_type(self) -> str:
                return "b"

        register_connector("b", _PendingConnector(), status="pending")

        items = list_connectors()
        assert len(items) == 2
        statuses = {i["source_type"]: i["status"] for i in items}
        assert statuses == {"a": "active", "b": "pending"}

    def test_batch_mode_supported(self):
        class _BatchConnector(_TestConnector):
            @property
            def source_type(self) -> str:
                return "batch"

            @property
            def supports_batch(self) -> bool:
                return True

        register_connector("batch", _BatchConnector())
        items = list_connectors()
        assert items[0]["batch_mode"] == "supported"
