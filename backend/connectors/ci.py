"""CI connector — transforms CI build webhooks into structured memories."""

from __future__ import annotations

from typing import Any

from backend.connectors.base import Connector

# Duration multiplier threshold: when the current duration exceeds
# baseline × REGRESSION_RATIO, the build is flagged as a regression.
REGRESSION_RATIO = 2.0


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

        return meta

    async def process(
        self, content: str, metadata: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Write to memory, detecting duration regressions.

        If *metadata* contains both ``duration_seconds`` and
        ``baseline_duration_seconds``, and the current duration exceeds
        ``baseline × REGRESSION_RATIO``, the memory is stored as
        ``source_type="ci_regression"``.
        """
        from backend.service.memory import write_memory

        meta = metadata or {}
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
