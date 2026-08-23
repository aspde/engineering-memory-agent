"""Small async GitHub Actions API client for CI connector enrichment.

Best-effort outbound calls that power ``ci_regression`` detection:

- authoritative job duration (``job_duration_seconds`` from GitHub's own
  timestamps),
- a median-duration baseline across recent successful runs of the same job
  (``baseline``),
- a bounded full job log appended to the memory content (``job_log``).

Every HTTP call goes through the shared resilience layer
(``call_with_resilience``), so a failing GitHub is retried on transient
errors and trips a ``"github"`` circuit breaker instead of hammering the
API; non-retryable 4xx (bad token, 404) propagate immediately.  The caller
in ``ci.py`` wraps each enrichment step in its own try/except, so any
GitHub-side failure degrades back to the plain inbound ``ci_build`` memory
— a GitHub outage never loses a CI failure memory.
"""

from __future__ import annotations

import io
import logging
import re
import statistics
import threading
import time
import zipfile
from datetime import datetime
from typing import Any

import httpx

from backend.shared.resilience import call_with_resilience

logger = logging.getLogger(__name__)

# Baseline cache — keyed by (workflow_id, branch, job_name), TTL 600s.
# Mirrors the ``_circuit_breakers`` pattern in ``resilience.py``: under a
# failure storm each webhook would otherwise re-fan out N run listings to
# recompute the same median.
_BASELINE_TTL_SECONDS = 600.0
_baseline_cache: dict[tuple[str, str, str], tuple[float, float]] = {}
_baseline_lock = threading.Lock()

# ANSI CSI escape sequences (color / cursor control) in CI logs.
_ANSI_CSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _get_baseline_cache(key: tuple[str, str, str]) -> float | None:
    now = time.monotonic()
    with _baseline_lock:
        entry = _baseline_cache.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if now >= expires_at:
            del _baseline_cache[key]
            return None
        return value


def _set_baseline_cache(key: tuple[str, str, str], value: float) -> None:
    with _baseline_lock:
        _baseline_cache[key] = (value, time.monotonic() + _BASELINE_TTL_SECONDS)


def job_duration_seconds(job: dict[str, Any]) -> float | None:
    """Job duration in seconds from authoritative GitHub timestamps.

    Returns ``None`` when either timestamp is missing, they are out of order
    (``completed_at <= started_at``), or the format is malformed — the caller
    then keeps the webhook-provided duration instead.
    """
    start_raw = job.get("started_at")
    end_raw = job.get("completed_at")
    if not start_raw or not end_raw:
        return None
    try:
        start = datetime.fromisoformat(start_raw)
        end = datetime.fromisoformat(end_raw)
    except (ValueError, TypeError):
        return None
    duration = (end - start).total_seconds()
    return duration if duration > 0 else None


def extract_log_text(zip_bytes: bytes, max_chars: int = 8000) -> str:
    """Extract and concatenate step logs from a GitHub Actions log zip.

    GitHub returns one ``{job}_{idx}_{step}.txt`` file per step; steps are
    ordered by their numeric index (a lexicographic sort would place ``_2_``
    after ``_10_``).  ANSI CSI escapes are stripped and non-UTF-8 bytes are
    replaced.  Output is capped at ``max_chars`` keeping the *tail* — failed
    steps usually log last — with a ``…[log truncated]…`` marker when
    something is dropped, so the result never exceeds ``max_chars``.

    Returns ``""`` for an empty zip, garbage bytes, or a zip whose entries
    can't be parsed as step files — the caller treats that as "no log".
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        return ""

    entries: list[tuple[int, str]] = []
    for info in zf.infolist():
        name = info.filename
        if info.is_dir() or name.startswith("__MACOSX/"):
            continue
        base = name.rsplit("/", 1)[-1]
        # GitHub format: {job}_{idx}_{step}.txt.  The step index is the first
        # numeric segment after the job id — a plain lexicographic sort would
        # place ``_2_`` after ``_10_``.  Splitting (instead of anchoring a
        # regex at the start) tolerates both ``7497345_1_setup.txt`` and a
        # bare ``job_1_x.txt``.
        parts = base.split("_")
        step_idx: int | None = None
        for part in parts[1:]:
            if part.isdigit():
                step_idx = int(part)
                break
        if step_idx is None:
            continue
        entries.append((step_idx, name))
    if not entries:
        return ""

    entries.sort(key=lambda item: item[0])
    chunks: list[str] = []
    for _, name in entries:
        try:
            raw = zf.read(name)
        except (KeyError, zipfile.BadZipFile):
            continue
        text = _ANSI_CSI_RE.sub("", raw.decode("utf-8", errors="replace"))
        if text:
            chunks.append(text)

    joined = "\n".join(chunks)
    if len(joined) <= max_chars:
        return joined
    marker = "…[log truncated]…\n"
    keep = max_chars - len(marker)
    if keep <= 0:
        return joined[-max_chars:]
    return marker + joined[-keep:]


class GitHubActionsClient:
    """Small async client for the GitHub Actions REST API.

    ``transport`` is injectable for tests (``httpx.MockTransport``); real
    usage leaves it ``None``.  Every public method runs its HTTP operation
    through ``call_with_resilience`` under the ``"github"`` circuit breaker.
    """

    def __init__(
        self,
        token: str,
        api_base: str = "https://api.github.com",
        timeout: float = 10.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._token = token
        self._api_base = api_base.rstrip("/")
        self._timeout = timeout
        self._transport = transport

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def _get_json(
        self, path: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        async def op() -> dict[str, Any]:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                transport=self._transport,
                follow_redirects=True,
                headers=self._headers(),
            ) as client:
                resp = await client.get(f"{self._api_base}{path}", params=params)
                resp.raise_for_status()
                return resp.json()

        return await call_with_resilience("github", op)

    async def get_job(self, owner: str, repo: str, job_id) -> dict[str, Any]:
        """GET /repos/{o}/{r}/actions/jobs/{job_id} — current job detail."""
        return await self._get_json(f"/repos/{owner}/{repo}/actions/jobs/{job_id}")

    async def get_run(self, owner: str, repo: str, run_id) -> dict[str, Any]:
        """GET /repos/{o}/{r}/actions/runs/{run_id} — resolves workflow_id."""
        return await self._get_json(f"/repos/{owner}/{repo}/actions/runs/{run_id}")

    async def list_jobs(self, owner: str, repo: str, run_id) -> dict[str, Any]:
        """GET /repos/{o}/{r}/actions/runs/{run_id}/jobs — job list page."""
        return await self._get_json(
            f"/repos/{owner}/{repo}/actions/runs/{run_id}/jobs",
            params={"per_page": 100},
        )

    async def list_successful_runs(
        self,
        owner: str,
        repo: str,
        workflow_id,
        branch: str | None = None,
        per_page: int = 5,
    ) -> dict[str, Any]:
        """GET …/actions/workflows/{id}/runs — recent successful runs.

        ``branch`` goes through httpx params, which percent-encodes ``/`` to
        ``%2F`` so branch names like ``feature/login`` resolve correctly.
        """
        params: dict[str, Any] = {"per_page": per_page, "status": "success"}
        if branch:
            params["branch"] = branch
        return await self._get_json(
            f"/repos/{owner}/{repo}/actions/workflows/{workflow_id}/runs",
            params=params,
        )

    async def baseline(
        self,
        owner: str,
        repo: str,
        *,
        job_name: str,
        branch: str | None = None,
        workflow_id=None,
        run_id=None,
        lookback: int = 5,
    ) -> float | None:
        """Median duration of *job_name* across recent successful runs.

        Returns ``None`` when there is no way to scope the search (neither
        ``workflow_id`` nor ``run_id``), no successful run of the same job
        exists, or none of them carries a usable duration.  The result is
        cached per (workflow, branch, job) for the TTL.

        ``workflow_id`` resolution: the GitHub ``workflow_job`` webhook
        payload does not carry the workflow id, so when it is absent the
        ``get_run`` response is used.  ``branch`` falls back to the run's
        ``head_branch`` when the webhook didn't provide one.
        """
        if not workflow_id:
            if not run_id:
                return None
            run = await self.get_run(owner, repo, run_id)
            workflow_id = run.get("workflow_id")
            branch = branch or run.get("head_branch")
            if not workflow_id:
                return None

        key = (str(workflow_id), str(branch or ""), job_name)
        if (cached := _get_baseline_cache(key)) is not None:
            return cached

        page = await self.list_successful_runs(
            owner,
            repo,
            workflow_id,
            branch=branch or None,
            per_page=min(lookback, 100),
        )
        durations: list[float] = []
        for candidate in page.get("workflow_runs", []):
            jobs_page = await self.list_jobs(owner, repo, candidate["id"])
            for job in jobs_page.get("jobs", []):
                candidate_name = job.get("name") or job.get("job_name")
                if candidate_name != job_name:
                    continue
                # Same-named job that did not succeed → skip this run so a
                # regression-prone job never pollutes its own baseline.
                if job.get("conclusion") != "success":
                    break
                d = job_duration_seconds(job)
                if d is not None and d > 0:
                    durations.append(d)
                break  # one same-named job per run counts at most once

        if not durations:
            return None
        baseline = statistics.median(durations)
        _set_baseline_cache(key, baseline)
        return baseline

    async def job_log(
        self, owner: str, repo: str, job_id, max_chars: int = 8000
    ) -> str:
        """Download the job's log zip and return bounded extracted text.

        The log endpoint 302s to a codeload download; ``follow_redirects=True``
        follows it.  Returns ``""`` for an empty/unparseable log so the caller
        can distinguish "no log" without an error.
        """

        async def op() -> bytes:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                transport=self._transport,
                follow_redirects=True,
                headers=self._headers(),
            ) as client:
                resp = await client.get(
                    f"{self._api_base}/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"
                )
                resp.raise_for_status()
                return resp.content

        zip_bytes = await call_with_resilience("github", op)
        return extract_log_text(zip_bytes, max_chars)
