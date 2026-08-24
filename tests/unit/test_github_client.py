"""Unit tests for the GitHub Actions client and its pure helpers."""

from __future__ import annotations

import io
import zipfile
from datetime import datetime, timedelta

import httpx
import pytest

from backend.connectors.github_client import (
    GitHubActionsClient,
    extract_log_text,
    job_duration_seconds,
)


@pytest.fixture(autouse=True)
def _clear_baseline_cache() -> None:
    """Isolate the module-level baseline cache between tests."""
    from backend.connectors import github_client as ghm

    ghm._baseline_cache.clear()


def _make_zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def _job(start: datetime, seconds: float, **extra) -> dict:
    return {
        "started_at": start.isoformat(),
        "completed_at": (start + timedelta(seconds=seconds)).isoformat(),
        **extra,
    }


# ── job_duration_seconds ──────────────────────────────────────────────


class TestJobDurationSeconds:
    def test_normal_job(self):
        start = datetime(2026, 1, 1, 0, 0, 0)
        assert job_duration_seconds(_job(start, 90)) == pytest.approx(90.0)

    def test_missing_completed_at_returns_none(self):
        assert job_duration_seconds({"started_at": "2026-01-01T00:00:00Z"}) is None

    def test_null_completed_at_returns_none(self):
        job = {"started_at": "2026-01-01T00:00:00Z", "completed_at": None}
        assert job_duration_seconds(job) is None

    def test_end_before_start_returns_none(self):
        start = datetime(2026, 1, 1, 0, 2, 0)
        assert job_duration_seconds(_job(start, -60)) is None

    def test_end_equal_start_returns_none(self):
        start = datetime(2026, 1, 1, 0, 0, 0)
        assert job_duration_seconds(_job(start, 0)) is None

    def test_malformed_timestamp_returns_none(self):
        job = {"started_at": "not-a-date", "completed_at": "also-bad"}
        assert job_duration_seconds(job) is None

    def test_missing_keys_returns_none(self):
        assert job_duration_seconds({}) is None


# ── extract_log_text ──────────────────────────────────────────────────


class TestExtractLogText:
    def test_steps_ordered_by_numeric_index(self):
        z = _make_zip(
            {
                "job_2_second.txt": "step two\n",
                "job_10_tenth.txt": "step ten\n",
                "job_1_first.txt": "step one\n",
            }
        )
        text = extract_log_text(z)
        assert text.index("step one") < text.index("step two") < text.index("step ten")

    def test_ansi_escapes_stripped(self):
        z = _make_zip({"job_1_x.txt": "\x1b[31mred text\x1b[0m ok\n"})
        text = extract_log_text(z)
        assert "\x1b" not in text
        assert "red text ok" in text

    def test_macosx_and_directories_ignored(self):
        z = _make_zip(
            {
                "job_1_x.txt": "real\n",
                "__MACOSX/job_1_x.txt": "junk\n",
            }
        )
        text = extract_log_text(z)
        assert "real" in text
        assert "junk" not in text

    def test_truncation_keeps_tail_and_marker_within_budget(self):
        z = _make_zip({"job_1_x.txt": "a" * 5000})
        text = extract_log_text(z, max_chars=100)
        assert len(text) <= 100
        assert "[log truncated]" in text
        assert text.endswith("a" * 50)

    def test_short_log_untouched(self):
        z = _make_zip({"job_1_x.txt": "short\n"})
        assert extract_log_text(z, max_chars=100) == "short\n"

    def test_empty_zip_returns_empty(self):
        assert extract_log_text(_make_zip({})) == ""

    def test_garbage_bytes_returns_empty(self):
        assert extract_log_text(b"not a zip") == ""

    def test_unparseable_entries_tolerated(self):
        z = _make_zip({"README.txt": "no step index", "job_1_x.txt": "ok"})
        text = extract_log_text(z)
        assert "ok" in text
        assert "no step index" not in text


# ── HTTP client methods ───────────────────────────────────────────────


class TestClientGetJob:
    @pytest.mark.asyncio
    async def test_auth_header_and_path(self):
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["auth"] = request.headers.get("Authorization")
            captured["accept"] = request.headers.get("Accept")
            captured["version"] = request.headers.get("X-GitHub-Api-Version")
            return httpx.Response(200, json={"id": 123, "name": "unit-tests"})

        client = GitHubActionsClient(
            token="t", transport=httpx.MockTransport(handler)
        )
        job = await client.get_job("owner", "repo", 123)
        assert captured["path"] == "/repos/owner/repo/actions/jobs/123"
        assert captured["auth"] == "Bearer t"
        assert captured["accept"] == "application/vnd.github+json"
        assert captured["version"] == "2022-11-28"
        assert job["id"] == 123


class TestBaseline:
    """Baseline = median duration of the same job across successful runs."""

    def _handler(
        self,
        *,
        run_durations: dict[int, float],
        missing_name: list[int] | None = None,
        non_success: list[int] | None = None,
        missing_duration: list[int] | None = None,
    ) -> httpx.MockTransport:
        missing_name = missing_name or []
        non_success = non_success or []
        missing_duration = missing_duration or []

        def handler(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path.endswith("/actions/workflows/99/runs"):
                return httpx.Response(
                    200,
                    json={"workflow_runs": [{"id": rid} for rid in run_durations]},
                )
            if "/actions/runs/" in path and path.endswith("/jobs"):
                run_id = int(path.rsplit("/", 2)[-2])
                start = datetime(2026, 1, 1, 0, 0, 0)
                job = {
                    "name": "unit-tests",
                    "conclusion": "success",
                    "started_at": start.isoformat(),
                    "completed_at": start.isoformat(),  # 0s → duration None
                }
                if run_id in missing_name:
                    job["name"] = "other-job"
                if run_id in non_success:
                    job["conclusion"] = "failure"
                if run_id not in missing_duration:
                    job["completed_at"] = (
                        start + timedelta(seconds=run_durations.get(run_id, 60))
                    ).isoformat()
                return httpx.Response(200, json={"jobs": [job]})
            return httpx.Response(404)

        return httpx.MockTransport(handler)

    @pytest.mark.asyncio
    async def test_median_of_successful_runs(self):
        # durations: 60s / 120s / 90s → median 90
        client = GitHubActionsClient(
            token="t",
            transport=self._handler(
                run_durations={1: 60, 2: 120, 3: 90}
            ),
        )
        base = await client.baseline("o", "r", job_name="unit-tests", workflow_id=99)
        assert base == pytest.approx(90.0)

    @pytest.mark.asyncio
    async def test_non_matching_job_skipped(self):
        client = GitHubActionsClient(
            token="t",
            transport=self._handler(run_durations={1: 60, 2: 120}, missing_name=[1]),
        )
        base = await client.baseline("o", "r", job_name="unit-tests", workflow_id=99)
        assert base == pytest.approx(120.0)

    @pytest.mark.asyncio
    async def test_non_success_run_skipped(self):
        client = GitHubActionsClient(
            token="t",
            transport=self._handler(run_durations={1: 60, 2: 120}, non_success=[1]),
        )
        base = await client.baseline("o", "r", job_name="unit-tests", workflow_id=99)
        assert base == pytest.approx(120.0)

    @pytest.mark.asyncio
    async def test_fewer_successful_runs_than_lookback_ok(self):
        client = GitHubActionsClient(
            token="t",
            transport=self._handler(run_durations={1: 60}),
        )
        base = await client.baseline(
            "o", "r", job_name="unit-tests", workflow_id=99, lookback=5
        )
        assert base == pytest.approx(60.0)

    @pytest.mark.asyncio
    async def test_all_durations_missing_returns_none(self):
        client = GitHubActionsClient(
            token="t",
            transport=self._handler(run_durations={1: 60, 2: 120}, missing_duration=[1, 2]),
        )
        base = await client.baseline("o", "r", job_name="unit-tests", workflow_id=99)
        assert base is None

    @pytest.mark.asyncio
    async def test_no_workflow_id_no_run_id_returns_none(self):
        client = GitHubActionsClient(token="t", transport=httpx.MockTransport(lambda r: httpx.Response(404)))
        base = await client.baseline("o", "r", job_name="unit-tests")
        assert base is None

    @pytest.mark.asyncio
    async def test_workflow_id_resolved_from_run(self):
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            seen.append(url)
            path = request.url.path
            if path.endswith("/actions/runs/123"):
                return httpx.Response(
                    200,
                    json={"workflow_id": 99, "head_branch": "main"},
                )
            if path.endswith("/actions/workflows/99/runs"):
                return httpx.Response(
                    200,
                    json={"workflow_runs": [{"id": 1}, {"id": 2}, {"id": 3}]},
                )
            if "/actions/runs/" in path and path.endswith("/jobs"):
                return httpx.Response(
                    200,
                    json={
                        "jobs": [
                            {
                                "name": "unit-tests",
                                "conclusion": "success",
                                "started_at": "2026-01-01T00:00:00Z",
                                "completed_at": "2026-01-01T00:01:00Z",
                            }
                        ]
                    },
                )
            return httpx.Response(404)

        client = GitHubActionsClient(token="t", transport=httpx.MockTransport(handler))
        base = await client.baseline("o", "r", job_name="unit-tests", run_id=123)
        assert base == pytest.approx(60.0)
        assert seen[0].endswith("/actions/runs/123")
        assert "/actions/workflows/99/runs" in seen[1]
        # head_branch from the run was passed as the branch filter
        assert "branch=main" in seen[1]

    @pytest.mark.asyncio
    async def test_result_cached_second_call_skips_listing(self):
        runs_seen = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal runs_seen
            path = request.url.path
            if path.endswith("/actions/workflows/99/runs"):
                runs_seen += 1
                return httpx.Response(
                    200,
                    json={"workflow_runs": [{"id": 1}]},
                )
            if "/actions/runs/" in path and path.endswith("/jobs"):
                return httpx.Response(
                    200,
                    json={
                        "jobs": [
                            {
                                "name": "unit-tests",
                                "conclusion": "success",
                                "started_at": "2026-01-01T00:00:00Z",
                                "completed_at": "2026-01-01T00:01:00Z",
                            }
                        ]
                    },
                )
            return httpx.Response(404)

        client = GitHubActionsClient(token="t", transport=httpx.MockTransport(handler))
        first = await client.baseline("o", "r", job_name="unit-tests", workflow_id=99)
        second = await client.baseline("o", "r", job_name="unit-tests", workflow_id=99)
        assert first == second == pytest.approx(60.0)
        assert runs_seen == 1


class TestClientJobLog:
    @pytest.mark.asyncio
    async def test_follows_redirect_and_extracts_zip(self):
        zipped = _make_zip({"1_1_setup.txt": "hello from setup\n"})
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            if request.url.path.endswith("/logs"):
                return httpx.Response(
                    302, headers={"Location": "https://codeload.example.com/log.zip"}
                )
            return httpx.Response(200, content=zipped)

        client = GitHubActionsClient(token="t", transport=httpx.MockTransport(handler))
        text = await client.job_log("o", "r", 1, max_chars=1000)
        assert "hello from setup" in text
        assert seen == [
            "https://api.github.com/repos/o/r/actions/jobs/1/logs",
            "https://codeload.example.com/log.zip",
        ]


class TestClientErrorPropagation:
    @pytest.mark.asyncio
    async def test_404_raises_and_handler_called_once(self):
        """Non-retryable 4xx propagates immediately — one call, no retry.

        The transport retry behaviour itself is covered by test_resilience.py;
        this pins that a client error surfaces as ``HTTPStatusError``.
        """
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            return httpx.Response(404, json={"message": "Not Found"})

        client = GitHubActionsClient(token="t", transport=httpx.MockTransport(handler))
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_job("o", "r", 999)
        assert calls == 1
