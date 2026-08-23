"""CI connector — transforms CI build webhooks into structured memories."""

from __future__ import annotations

import logging
import re
from typing import Any

from backend.connectors.base import Connector
from backend.connectors.github_client import (
    GitHubActionsClient,
    job_duration_seconds,
)
from backend.shared.config import config

logger = logging.getLogger(__name__)

# Duration multiplier threshold: when the current duration exceeds
# baseline × REGRESSION_RATIO, the build is flagged as a regression.
REGRESSION_RATIO = 2.0

# GitHub Actions run URL — ``build_url`` from a GitHub-native webhook, e.g.
# https://github.com/{owner}/{repo}/actions/runs/{run_id}(/job/{job_id})
_GITHUB_RUN_URL_RE = re.compile(
    r"https://github\.com/(?P<owner>[^/]+)/(?P<repo>[^/]+)/actions/runs/"
    r"(?P<run_id>\d+)(?:/job/(?P<job_id>\d+))?"
)

# The ``Duration: Xs`` line rendered by ``normalize()`` — anchored to the
# line start so the GitHub-authoritative value can replace the webhook's.
_DURATION_LINE_RE = re.compile(r"^Duration: .*$", re.MULTILINE)


def _replace_duration_line(content: str, duration: float) -> str:
    """Synchronize the ``Duration: Xs`` line in *content* with *duration*.

    Only the first occurrence is replaced; the webhook may omit the line
    entirely (the enrichment then still drives the regression via metadata).
    """
    return _DURATION_LINE_RE.sub(f"Duration: {duration:.1f}s", content, count=1)


class CIConnector(Connector):
    """Ingest CI build results via webhook.

    Normalizes build payloads into EMA-standard content text.  Failed
    builds are stored with ``source_type="ci_build"``.  When a build's
    duration significantly exceeds a provided baseline, it is flagged as
    ``"ci_regression"`` instead.
    """

    display_name = "CI/CD"

    @property
    def source_type(self) -> str:
        return "ci_build"

    @property
    def triggers_event_analysis(self) -> bool:
        # Failed builds trigger the Phase 3 event-driven analysis (search
        # for similar historical failures, push a summary).  Gated by
        # EVENT_ANALYSIS_ENABLED at the runner level.
        return True

    @property
    def event_analysis_cooldown_key(self) -> str | None:
        # Repeated failures of the same job within the cooldown window are
        # analysed once (spec story 10 — notification fatigue).
        return "job_name"

    @property
    def event_analysis_prompt_key(self) -> str | None:
        return "event.ci_failure"

    @property
    def event_analysis_display(self) -> dict[str, Any]:
        # Card title shows the job; the user message carries branch + URL
        # context lines.
        return {
            "title_entity": "job_name",
            "context_fields": ["branch", "source_url"],
        }

    # ── Connector ABC ─────────────────────────────────────────────────

    def validate(self, payload: dict[str, Any]) -> bool:
        """Payload must have job_name, a failure/error status, and commit_sha.

        Only failed builds are ingested — successful builds are rejected
        so the CI system should be configured to send webhooks only on
        failure.
        """
        if not isinstance(payload.get("job_name"), str) or not payload["job_name"].strip():
            return False
        status: str = (payload.get("status") or "").lower()
        if status not in ("failure", "error", "failed"):
            return False
        if not isinstance(payload.get("commit_sha"), str) or not payload["commit_sha"].strip():
            return False
        return True

    def normalize(self, payload: dict[str, Any]) -> str:
        """Transform a CI build webhook payload into structured text."""
        job_name: str = payload.get("job_name", "")
        status: str = payload.get("status", "")
        error_summary: str = payload.get("error_summary", "") or ""
        commit_sha: str = payload.get("commit_sha", "")
        branch: str = payload.get("branch", "") or ""
        duration: float | None = payload.get("duration_seconds")
        build_url: str = payload.get("build_url", "") or ""

        parts: list[str] = [
            f"CI Build: {job_name} — {status.upper()}",
            f"Commit: {commit_sha}",
        ]
        if branch:
            parts.append(f"Branch: {branch}")
        if duration is not None:
            parts.append(f"Duration: {duration:.1f}s")
        if error_summary:
            parts.append(f"Error:\n{error_summary.strip()}")
        if build_url:
            parts.append(f"Build URL: {build_url}")

        return "\n".join(parts)

    def build_metadata(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Extract traceability metadata from a CI webhook payload."""
        meta: dict[str, Any] = {
            "job_name": payload.get("job_name", ""),
            "commit_sha": payload.get("commit_sha", ""),
            "branch": payload.get("branch", ""),
            "ci_status": payload.get("status", ""),
        }

        build_url = payload.get("build_url", "")
        if build_url:
            meta["source_url"] = build_url

        duration = payload.get("duration_seconds")
        if duration is not None:
            meta["duration_seconds"] = duration

        # Optional GitHub Actions identifiers — flat fields or a nested
        # "github" dict.  Stored under a ``github_`` prefix so they can't
        # collide with connector-native keys.  Owner/repo may still be absent
        # here: ``_github_identifiers`` falls back to parsing ``build_url``.
        github_fields = ("owner", "repo", "run_id", "job_id", "workflow_id")
        nested = payload.get("github")
        if isinstance(nested, dict):
            for key in github_fields:
                if nested.get(key) is not None:
                    meta[f"github_{key}"] = nested[key]
        for key in github_fields:
            if payload.get(key) is not None:
                meta[f"github_{key}"] = payload[key]

        return meta

    async def process(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Write to memory, enriching with GitHub Actions data.

        When ``config.ci_github.token`` is set and the metadata carries
        GitHub identifiers, ``_enrich_github`` fetches the authoritative job
        duration, a median-duration baseline, and a bounded full job log —
        each step degrading independently on GitHub-side failure.  The
        regression check then runs on the enriched values; without a
        baseline the memory stays ``ci_build``.
        """
        from backend.service.memory import write_memory

        meta = metadata or {}
        content, meta = await self._enrich_github(content, meta)

        duration = meta.get("duration_seconds")
        baseline = meta.get("baseline_duration_seconds")

        effective_source = "ci_build"
        if (
            isinstance(duration, (int, float))
            and isinstance(baseline, (int, float))
            and baseline > 0
            and duration > baseline * REGRESSION_RATIO
        ):
            effective_source = "ci_regression"
            # Enrich content with regression context
            ratio = duration / baseline
            content = (
                f"[DURATION REGRESSION — {ratio:.1f}× baseline]\n"
                f"Baseline: {baseline:.1f}s → Current: {duration:.1f}s\n\n"
                + content
            )

        return await write_memory(content, source_type=effective_source, metadata=meta)

    def _github_identifiers(
        self, meta: dict[str, Any]
    ) -> tuple[str, str, str | None, str | None] | None:
        """Extract ``(owner, repo, run_id, job_id)`` from metadata.

        Prefers the explicit ``github_*`` fields captured by
        ``build_metadata``; when owner/repo are absent it falls back to
        parsing a GitHub Actions ``build_url``.  Returns ``None`` when
        owner/repo can't be determined — enrichment then stays off.
        """
        owner = meta.get("github_owner")
        repo = meta.get("github_repo")
        run_id = meta.get("github_run_id")
        job_id = meta.get("github_job_id")

        if not owner or not repo:
            match = _GITHUB_RUN_URL_RE.match(meta.get("source_url", "") or "")
            if not match:
                return None
            owner = owner or match.group("owner")
            repo = repo or match.group("repo")
            run_id = run_id or match.group("run_id")
            job_id = job_id or match.group("job_id")
            if not owner or not repo:
                return None

        return (
            str(owner),
            str(repo),
            str(run_id) if run_id is not None else None,
            str(job_id) if job_id is not None else None,
        )

    async def _enrich_github(
        self, content: str, meta: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Best-effort GitHub Actions enrichment of content and metadata.

        Each step is independently guarded so a partial failure keeps what
        succeeded (a job-log failure does not discard the computed baseline).
        The inputs are never mutated — every update rebinds via ``{**meta,
        ...}`` and the enriched pair is returned.  With no token configured
        or no identifiers in the payload, the input is returned unchanged —
        today's purely inbound behaviour.
        """
        gh = config.ci_github
        ids = self._github_identifiers(meta)
        if not gh.token or not ids:
            return content, meta

        owner, repo, run_id, job_id = ids
        client = GitHubActionsClient(
            token=gh.token, api_base=gh.api_base, timeout=gh.timeout
        )
        warns: list[str] = []

        if job_id:
            try:
                job = await client.get_job(owner, repo, job_id)
                d = job_duration_seconds(job)
                if d is not None:
                    meta = {**meta, "duration_seconds": d}
                    content = _replace_duration_line(content, d)
            except Exception as exc:
                warns.append(f"job duration: {exc}")

        try:
            baseline = await client.baseline(
                owner,
                repo,
                job_name=meta.get("job_name", ""),
                branch=meta.get("branch"),
                workflow_id=meta.get("github_workflow_id"),
                run_id=run_id,
                lookback=gh.lookback_runs,
            )
            if baseline is not None:
                meta = {**meta, "baseline_duration_seconds": baseline}
        except Exception as exc:
            warns.append(f"baseline: {exc}")

        if job_id:
            try:
                log_text = await client.job_log(
                    owner, repo, job_id, max_chars=gh.log_max_chars
                )
                if log_text:
                    content = content.rstrip("\n") + f"\n\nGitHub Actions Log:\n{log_text}"
                    meta = {**meta, "log_enrichment": True}
            except Exception as exc:
                warns.append(f"job log: {exc}")

        if warns:
            logger.warning(
                "GitHub Actions enrichment partial failure for %s/%s (run=%s job=%s): "
                "%s — storing what succeeded",
                owner, repo, run_id, job_id, "; ".join(warns),
            )
        return content, meta
