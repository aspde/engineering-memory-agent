"""End-to-end best-effort contract for the CI connector's GitHub enrichment.

With ``CI_GITHUB_TOKEN`` set and GitHub identifiers in the payload, the
connector attempts enrichment; when GitHub is entirely unreachable every
enrichment step degrades and the webhook still delivers ``processed`` with
a plain ``ci_build`` memory — a GitHub outage must never lose a CI failure
memory.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient, ConnectError

from backend.connectors.ci import CIConnector
from backend.connectors.registry import register_connector
from backend.main import app
from tests.api.test_webhook_routes import _signed_post, _wait_for_status


@pytest.fixture(autouse=True)
async def _ensure_tables() -> None:
    from backend.db.schema import init_db

    await init_db()


class _DownGHClient:
    """GitHub Actions client whose every call raises a transport error."""

    def __init__(self, *, token: str, api_base: str, timeout) -> None:
        pass

    async def get_job(self, *args, **kwargs):
        raise ConnectError("gh down")

    async def baseline(self, *args, **kwargs):
        raise ConnectError("gh down")

    async def job_log(self, *args, **kwargs):
        raise ConnectError("gh down")


@pytest.fixture(autouse=True)
def _register_ci_connector(monkeypatch):
    """Register the real CIConnector with GitHub enrichment forced on.

    ``GitHubActionsClient`` is swapped for one that always raises, and the
    real ``write_memory`` is stubbed so the test asserts on the delivery
    outcome rather than running the LLM extraction pipeline.
    """
    import backend.connectors.ci as ci_module
    from backend.shared.config import config

    monkeypatch.setenv("WEBHOOK_CI_SECRET", "testsecret123")
    monkeypatch.setattr(config.ci_github, "token", "test-token")
    monkeypatch.setattr(ci_module, "GitHubActionsClient", _DownGHClient)
    register_connector("ci", CIConnector(), status="active")
    with patch(
        "backend.api.routes.webhook_routes.write_memory",
        new_callable=AsyncMock,
    ) as mock_write:
        mock_write.return_value = {
            "id": "11111111-2222-3333-4444-555555555555",
            "action": "inserted",
            "summary": "Build unit-tests failed",
            "entity_ids": [],
        }
        yield
    from backend.connectors.registry import CONNECTOR_REGISTRY

    CONNECTOR_REGISTRY.clear()


@pytest.fixture
async def async_client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_github_down_webhook_still_processed(async_client: AsyncClient) -> None:
    """Enrichment failing end-to-end degrades to a plain ci_build delivery."""
    payload = {
        "job_name": "unit-tests",
        "status": "failure",
        "commit_sha": "abc123def456",
        "branch": "main",
        "github": {"owner": "octo", "repo": "repo", "run_id": 12345, "job_id": 67890},
    }
    raw, headers = _signed_post(payload)
    resp = await async_client.post("/api/webhook/ci", content=raw, headers=headers)

    assert resp.status_code == 202
    delivery_id = resp.json()["delivery_id"]

    row = await _wait_for_status(delivery_id, "processed")
    assert str(row.memory_id) == "11111111-2222-3333-4444-555555555555"
    assert row.error is None


@pytest.mark.asyncio
async def test_github_down_memory_receives_plain_content(
    async_client: AsyncClient,
) -> None:
    """The memory written on full GitHub failure is the plain ci_build content."""
    captured: dict = {}

    async def _capture(content, source_type="conversation", metadata=None):
        captured["content"] = content
        captured["source_type"] = source_type
        captured["metadata"] = metadata
        return {
            "id": "11111111-2222-3333-4444-555555555555",
            "action": "inserted",
            "summary": content,
        }

    # Override the autouse stub for this test (the background task runs after
    # the response returns, so the override must outlive the POST).
    with patch(
        "backend.api.routes.webhook_routes.write_memory",
        new_callable=AsyncMock,
        side_effect=_capture,
    ):
        payload = {
            "job_name": "unit-tests",
            "status": "failure",
            "commit_sha": "abc123def456",
            "github": {"owner": "octo", "repo": "repo", "run_id": 1, "job_id": 2},
        }
        raw, headers = _signed_post(payload)
        resp = await async_client.post("/api/webhook/ci", content=raw, headers=headers)
        assert resp.status_code == 202
        await _wait_for_status(resp.json()["delivery_id"], "processed")

    assert captured["source_type"] == "ci_build"
    assert "GitHub Actions Log:" not in captured["content"]
    assert "github_owner" in captured["metadata"]  # traceability kept
    assert "baseline_duration_seconds" not in captured["metadata"]
