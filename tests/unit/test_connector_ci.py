"""Unit tests for CIConnector — pure data transformation, no IO."""

import pytest

from backend.connectors.ci import CIConnector


# ── Sample payloads ───────────────────────────────────────────────────


def _make_payload(
    job_name: str = "unit-tests",
    status: str = "failure",
    commit_sha: str = "abc123def456",
    branch: str = "main",
    error_summary: str = "3 tests failed in test_auth.py",
    duration_seconds: float = 45.0,
    build_url: str = "https://ci.example.com/build/123",
) -> dict:
    return {
        "job_name": job_name,
        "status": status,
        "commit_sha": commit_sha,
        "branch": branch,
        "error_summary": error_summary,
        "duration_seconds": duration_seconds,
        "build_url": build_url,
    }


# ── validate ──────────────────────────────────────────────────────────


class TestCIValidate:
    def test_valid_payload_accepted(self):
        conn = CIConnector()
        assert conn.validate(_make_payload()) is True

    def test_missing_job_name_rejected(self):
        conn = CIConnector()
        p = _make_payload()
        del p["job_name"]
        assert conn.validate(p) is False

    def test_empty_job_name_rejected(self):
        conn = CIConnector()
        assert conn.validate(_make_payload(job_name="")) is False
        assert conn.validate(_make_payload(job_name="   ")) is False

    def test_missing_status_rejected(self):
        conn = CIConnector()
        p = _make_payload()
        del p["status"]
        assert conn.validate(p) is False

    def test_empty_status_rejected(self):
        conn = CIConnector()
        assert conn.validate(_make_payload(status="")) is False

    def test_missing_commit_sha_rejected(self):
        conn = CIConnector()
        p = _make_payload()
        del p["commit_sha"]
        assert conn.validate(p) is False

    def test_empty_commit_sha_rejected(self):
        conn = CIConnector()
        assert conn.validate(_make_payload(commit_sha="")) is False

    def test_success_status_rejected(self):
        conn = CIConnector()
        assert conn.validate(_make_payload(status="success")) is False

    def test_passed_status_rejected(self):
        conn = CIConnector()
        assert conn.validate(_make_payload(status="passed")) is False

    def test_failure_status_accepted(self):
        conn = CIConnector()
        assert conn.validate(_make_payload(status="failure")) is True

    def test_error_status_accepted(self):
        conn = CIConnector()
        assert conn.validate(_make_payload(status="error")) is True

    def test_failed_status_accepted(self):
        conn = CIConnector()
        assert conn.validate(_make_payload(status="failed")) is True

    def test_empty_payload_rejected(self):
        conn = CIConnector()
        assert conn.validate({}) is False


# ── normalize ─────────────────────────────────────────────────────────


class TestCINormalize:
    def test_normalize_includes_job_name_and_status(self):
        conn = CIConnector()
        result = conn.normalize(_make_payload())
        assert "unit-tests" in result
        assert "FAILURE" in result

    def test_normalize_includes_commit_sha(self):
        conn = CIConnector()
        result = conn.normalize(_make_payload())
        assert "abc123def456" in result

    def test_normalize_includes_branch(self):
        conn = CIConnector()
        result = conn.normalize(_make_payload(branch="feature/login"))
        assert "feature/login" in result

    def test_normalize_includes_duration(self):
        conn = CIConnector()
        result = conn.normalize(_make_payload(duration_seconds=120.5))
        assert "120.5s" in result

    def test_normalize_includes_error_summary(self):
        conn = CIConnector()
        result = conn.normalize(
            _make_payload(error_summary="ImportError: no module named 'xyz'")
        )
        assert "ImportError" in result

    def test_normalize_includes_build_url(self):
        conn = CIConnector()
        result = conn.normalize(_make_payload())
        assert "https://ci.example.com/build/123" in result

    def test_normalize_omits_empty_optional_fields(self):
        conn = CIConnector()
        p = _make_payload(branch="", error_summary="", build_url="")
        result = conn.normalize(p)
        assert "Branch:" not in result
        assert "Error:" not in result
        assert "Build URL:" not in result

    def test_normalize_error_status(self):
        conn = CIConnector()
        result = conn.normalize(_make_payload(status="error"))
        assert "ERROR" in result

    def test_normalize_no_duration(self):
        conn = CIConnector()
        p = _make_payload()
        del p["duration_seconds"]
        result = conn.normalize(p)
        assert "Duration:" not in result


# ── build_metadata ────────────────────────────────────────────────────


class TestCIBuildMetadata:
    def test_includes_core_fields(self):
        conn = CIConnector()
        meta = conn.build_metadata(_make_payload())
        assert meta["job_name"] == "unit-tests"
        assert meta["commit_sha"] == "abc123def456"
        assert meta["branch"] == "main"
        assert meta["ci_status"] == "failure"

    def test_includes_source_url(self):
        conn = CIConnector()
        meta = conn.build_metadata(_make_payload(build_url="https://ci.example.com/99"))
        assert meta["source_url"] == "https://ci.example.com/99"

    def test_includes_duration(self):
        conn = CIConnector()
        meta = conn.build_metadata(_make_payload(duration_seconds=42.0))
        assert meta["duration_seconds"] == 42.0


# ── regression detection via process ──────────────────────────────────


class TestCIRegression:
    @pytest.mark.asyncio
    async def test_normal_build_uses_ci_build_source(self, monkeypatch):
        from backend.service import memory as mem_module

        calls: list[dict] = []

        async def _fake_write(content, source_type, metadata):
            calls.append({"source_type": source_type, "content": content})
            return {"id": "x", "action": "inserted", "summary": content}

        monkeypatch.setattr(mem_module, "write_memory", _fake_write)

        conn = CIConnector()
        await conn.process("content", {"duration_seconds": 30, "baseline_duration_seconds": 45})

        assert calls[0]["source_type"] == "ci_build"
        assert "[DURATION REGRESSION" not in calls[0]["content"]

    @pytest.mark.asyncio
    async def test_regression_uses_ci_regression_source(self, monkeypatch):
        from backend.service import memory as mem_module

        calls: list[dict] = []

        async def _fake_write(content, source_type, metadata):
            calls.append({"source_type": source_type, "content": content})
            return {"id": "x", "action": "inserted", "summary": content}

        monkeypatch.setattr(mem_module, "write_memory", _fake_write)

        conn = CIConnector()
        await conn.process(
            "original content",
            {"duration_seconds": 120, "baseline_duration_seconds": 30},
        )

        assert calls[0]["source_type"] == "ci_regression"
        assert "[DURATION REGRESSION" in calls[0]["content"]
        assert "4.0×" in calls[0]["content"]
        assert "120.0s" in calls[0]["content"]
        assert "30.0s" in calls[0]["content"]

    @pytest.mark.asyncio
    async def test_no_baseline_no_regression(self, monkeypatch):
        from backend.service import memory as mem_module

        calls: list[dict] = []

        async def _fake_write(content, source_type, metadata):
            calls.append({"source_type": source_type})
            return {"id": "x", "action": "inserted", "summary": content}

        monkeypatch.setattr(mem_module, "write_memory", _fake_write)

        conn = CIConnector()
        await conn.process("content", {"duration_seconds": 999})

        assert calls[0]["source_type"] == "ci_build"

    @pytest.mark.asyncio
    async def test_zero_baseline_no_division_error(self, monkeypatch):
        from backend.service import memory as mem_module

        calls: list[dict] = []

        async def _fake_write(content, source_type, metadata):
            calls.append({"source_type": source_type})
            return {"id": "x", "action": "inserted", "summary": content}

        monkeypatch.setattr(mem_module, "write_memory", _fake_write)

        conn = CIConnector()
        # baseline=0 should not trigger regression (division by zero avoided)
        await conn.process("content", {"duration_seconds": 100, "baseline_duration_seconds": 0})

        assert calls[0]["source_type"] == "ci_build"

    @pytest.mark.asyncio
    async def test_exactly_at_threshold_no_regression(self, monkeypatch):
        from backend.service import memory as mem_module

        calls: list[dict] = []

        async def _fake_write(content, source_type, metadata):
            calls.append({"source_type": source_type})
            return {"id": "x", "action": "inserted", "summary": content}

        monkeypatch.setattr(mem_module, "write_memory", _fake_write)

        conn = CIConnector()
        # Exactly 2.0× (not strictly greater) → no regression
        await conn.process(
            "content", {"duration_seconds": 60, "baseline_duration_seconds": 30}
        )

        assert calls[0]["source_type"] == "ci_build"
