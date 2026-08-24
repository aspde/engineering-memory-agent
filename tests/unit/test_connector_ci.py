"""Unit tests for CIConnector — pure data transformation, no IO."""

import pytest

from backend.connectors.ci import CIConnector

# ── Sample payloads ───────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _no_github_token(monkeypatch):
    """Default: GitHub enrichment off.

    A developer's real ``CI_GITHUB_TOKEN`` in ``.env`` must not make the
    existing process tests hit the network; the identifiers gate in
    ``_enrich_github`` is a second guard.  Tests that enable enrichment
    re-set the token via the ``gh_enabled`` fixture.
    """
    from backend.shared.config import config

    monkeypatch.setattr(config.ci_github, "token", "")


class _FakeGHClient:
    """Injected stand-in for ``GitHubActionsClient``.

    Records calls and returns configured values per method; each method can
    be set to raise to exercise the best-effort degradation paths.
    """

    def __init__(self, *, token: str, api_base: str, timeout) -> None:
        self.token = token
        self.api_base = api_base
        self.timeout = timeout
        self.calls: list[tuple] = []
        self.get_job_result = None
        self.baseline_result = None
        self.job_log_result = ""
        self.get_job_error = None
        self.baseline_error = None
        self.job_log_error = None

    async def get_job(self, owner, repo, job_id):
        self.calls.append(("get_job", owner, repo, job_id))
        if self.get_job_error:
            raise self.get_job_error
        if self.get_job_result is not None:
            return self.get_job_result
        return {
            "started_at": "2026-01-01T00:00:00Z",
            "completed_at": "2026-01-01T00:02:00Z",  # 120s
        }

    async def baseline(self, owner, repo, **kwargs):
        self.calls.append(("baseline", owner, repo, kwargs))
        if self.baseline_error:
            raise self.baseline_error
        return self.baseline_result

    async def job_log(self, owner, repo, job_id, max_chars=8000):
        self.calls.append(("job_log", owner, repo, job_id, max_chars))
        if self.job_log_error:
            raise self.job_log_error
        return self.job_log_result


@pytest.fixture
def gh_enabled(monkeypatch) -> _FakeGHClient:
    """Turn GitHub enrichment on and inject a fake client."""
    from backend.connectors import ci as ci_module
    from backend.shared.config import config

    monkeypatch.setattr(config.ci_github, "token", "test-token")
    fake = _FakeGHClient(
        token="test-token", api_base="https://api.github.com", timeout=10
    )
    monkeypatch.setattr(ci_module, "GitHubActionsClient", lambda **kw: fake)
    return fake


# ── GitHub enrichment via prepare ─────────────────────────────────────


class TestGithubEnrichment:
    _CONTENT = (
        "CI Build: unit-tests — FAILURE\n"
        "Commit: abc123\n"
        "Branch: main\n"
        "Duration: 5.0s\n"
        "Error:\nboom"
    )

    def _meta(self, **extra) -> dict:
        return {
            "job_name": "unit-tests",
            "branch": "main",
            "github_owner": "octo",
            "github_repo": "repo",
            "github_run_id": 123,
            "github_job_id": 456,
            **extra,
        }

    @pytest.mark.asyncio
    async def test_no_token_client_never_constructed(self, monkeypatch):
        """With the default empty token, enrichment stays fully off."""

        def _boom(**kw):
            raise AssertionError("GitHubActionsClient constructed without a token")

        from backend.connectors import ci as ci_module

        monkeypatch.setattr(ci_module, "GitHubActionsClient", _boom)

        conn = CIConnector()
        content, source_type, meta = await conn.prepare("content", self._meta())

        assert source_type == "ci_build"
        assert content == "content"

    @pytest.mark.asyncio
    async def test_token_with_ids_enriches_to_regression(self, gh_enabled):
        fake = gh_enabled
        fake.baseline_result = 30.0
        fake.job_log_result = "Run some step\n"

        conn = CIConnector()
        content, source_type, meta = await conn.prepare(self._CONTENT, self._meta())

        assert source_type == "ci_regression"
        assert "4.0×" in content           # 120 / 30
        assert "Duration: 120.0s" in content  # authoritative duration
        assert "GitHub Actions Log:" in content
        assert "Run some step" in content
        assert meta["duration_seconds"] == 120.0
        assert meta["baseline_duration_seconds"] == 30.0
        assert meta["log_enrichment"] is True
        assert meta["github_owner"] == "octo"  # original meta preserved

    @pytest.mark.asyncio
    async def test_baseline_error_keeps_log_and_duration(self, gh_enabled):
        fake = gh_enabled
        fake.baseline_error = RuntimeError("gh down")
        fake.job_log_result = "log line"

        conn = CIConnector()
        content, source_type, meta = await conn.prepare(self._CONTENT, self._meta())

        # No baseline → no regression, but the duration + log enrichment survived.
        assert source_type == "ci_build"
        assert "GitHub Actions Log:" in content
        assert meta["duration_seconds"] == 120.0
        assert "baseline_duration_seconds" not in meta

    @pytest.mark.asyncio
    async def test_all_calls_fail_returns_inbound_content(self, gh_enabled):
        fake = gh_enabled
        fake.get_job_error = RuntimeError("gh down")
        fake.baseline_error = RuntimeError("gh down")
        fake.job_log_error = RuntimeError("gh down")

        conn = CIConnector()
        content, source_type, _ = await conn.prepare(self._CONTENT, self._meta())

        # Content is byte-for-byte identical to the inbound input.
        assert source_type == "ci_build"
        assert content == self._CONTENT

    @pytest.mark.asyncio
    async def test_empty_log_no_log_section(self, gh_enabled):
        fake = gh_enabled
        fake.baseline_result = 30.0
        fake.job_log_result = ""

        conn = CIConnector()
        content, source_type, meta = await conn.prepare(self._CONTENT, self._meta())

        assert source_type == "ci_regression"
        assert "GitHub Actions Log:" not in content
        assert "log_enrichment" not in meta

    @pytest.mark.asyncio
    async def test_baseline_none_with_duration_is_ci_build(self, gh_enabled):
        fake = gh_enabled
        fake.baseline_result = None

        conn = CIConnector()
        _, source_type, meta = await conn.prepare(self._CONTENT, self._meta())

        assert source_type == "ci_build"
        assert "baseline_duration_seconds" not in meta

    @pytest.mark.asyncio
    async def test_no_job_id_baseline_still_runs(self, gh_enabled):
        """owner/repo/run_id without job_id → baseline runs, job calls don't."""
        fake = gh_enabled
        fake.baseline_result = 30.0

        conn = CIConnector()
        _, _, meta = await conn.prepare(
            "CI Build: unit-tests — FAILURE\nCommit: abc",
            self._meta(github_job_id=None),
        )

        kinds = [c[0] for c in fake.calls]
        assert "baseline" in kinds
        assert "get_job" not in kinds
        assert "job_log" not in kinds
        assert meta["baseline_duration_seconds"] == 30.0

    @pytest.mark.asyncio
    async def test_github_build_url_resolves_identifiers(self, gh_enabled):
        """Owner/repo/run_id/job_id parsed from a GitHub build_url."""
        fake = gh_enabled
        fake.baseline_result = 30.0
        fake.job_log_result = "log"

        conn = CIConnector()
        meta = {
            "job_name": "unit-tests",
            "branch": "main",
            "source_url": "https://github.com/o/r/actions/runs/5/job/6",
        }
        _, _, out_meta = await conn.prepare(self._CONTENT, meta)

        job_calls = [c for c in fake.calls if c[0] == "get_job"]
        assert job_calls, "get_job must be called with identifiers from build_url"
        assert job_calls[0][1:] == ("o", "r", "6")
        assert out_meta["duration_seconds"] == 120.0

    @pytest.mark.asyncio
    async def test_jenkins_build_url_skips_enrichment(self, gh_enabled):
        fake = gh_enabled

        conn = CIConnector()
        meta = {
            "job_name": "unit-tests",
            "branch": "main",
            "source_url": "https://jenkins.example.com/job/x/42",
        }
        content, source_type, _ = await conn.prepare("content", meta)

        assert fake.calls == []  # no GitHub identifiers → no client calls
        assert source_type == "ci_build"
        assert content == "content"


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
    async def test_normal_build_uses_ci_build_source(self):
        conn = CIConnector()
        content, source_type, _ = await conn.prepare(
            "content", {"duration_seconds": 30, "baseline_duration_seconds": 45}
        )

        assert source_type == "ci_build"
        assert "[DURATION REGRESSION" not in content

    @pytest.mark.asyncio
    async def test_regression_uses_ci_regression_source(self):
        conn = CIConnector()
        content, source_type, _ = await conn.prepare(
            "original content",
            {"duration_seconds": 120, "baseline_duration_seconds": 30},
        )

        assert source_type == "ci_regression"
        assert "[DURATION REGRESSION" in content
        assert "4.0×" in content
        assert "120.0s" in content
        assert "30.0s" in content

    @pytest.mark.asyncio
    async def test_no_baseline_no_regression(self):
        conn = CIConnector()
        _, source_type, _ = await conn.prepare("content", {"duration_seconds": 999})

        assert source_type == "ci_build"

    @pytest.mark.asyncio
    async def test_zero_baseline_no_division_error(self):
        conn = CIConnector()
        # baseline=0 should not trigger regression (division by zero avoided)
        _, source_type, _ = await conn.prepare(
            "content", {"duration_seconds": 100, "baseline_duration_seconds": 0}
        )

        assert source_type == "ci_build"

    @pytest.mark.asyncio
    async def test_exactly_at_threshold_no_regression(self):
        conn = CIConnector()
        # Exactly 2.0× (not strictly greater) → no regression
        _, source_type, _ = await conn.prepare(
            "content", {"duration_seconds": 60, "baseline_duration_seconds": 30}
        )

        assert source_type == "ci_build"


# ── build_metadata: GitHub identifiers ────────────────────────────────


class TestCIBuildMetadataGithub:
    def test_flat_github_fields_captured(self):
        conn = CIConnector()
        meta = conn.build_metadata(
            {
                **_make_payload(),
                "owner": "octo", "repo": "repo", "run_id": 5, "job_id": 6,
                "workflow_id": 9,
            }
        )
        assert meta["github_owner"] == "octo"
        assert meta["github_repo"] == "repo"
        assert meta["github_run_id"] == 5
        assert meta["github_job_id"] == 6
        assert meta["github_workflow_id"] == 9

    def test_nested_github_dict_captured(self):
        conn = CIConnector()
        meta = conn.build_metadata(
            {
                **_make_payload(),
                "github": {"owner": "octo", "repo": "repo", "run_id": 5, "job_id": 6},
            }
        )
        assert meta["github_owner"] == "octo"
        assert meta["github_repo"] == "repo"
        assert meta["github_run_id"] == 5
        assert meta["github_job_id"] == 6

    def test_github_build_url_kept_for_identifier_parse(self):
        """A GitHub build_url is kept as source_url; identifiers resolve later."""
        conn = CIConnector()
        meta = conn.build_metadata(
            _make_payload(build_url="https://github.com/o/r/actions/runs/5/job/6")
        )
        assert meta["source_url"] == "https://github.com/o/r/actions/runs/5/job/6"
        assert "github_owner" not in meta

    def test_jenkins_build_url_no_github_keys(self):
        conn = CIConnector()
        meta = conn.build_metadata(
            _make_payload(build_url="https://jenkins.example.com/job/x/42")
        )
        assert meta["source_url"] == "https://jenkins.example.com/job/x/42"
        assert not any(key.startswith("github_") for key in meta)
