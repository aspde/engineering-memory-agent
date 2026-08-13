"""Git ingestion — walk repo history via pygit2, feed commits into memory.

Each commit becomes a structured memory through the existing
:func:`backend.service.memory.write_memory` pipeline (extract → grade → merge).

This module is *only* responsible for reading Git history; all LLM extraction,
similarity grading, and persistence is deferred to the memory service.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pygit2

from backend.service.memory import write_memory

logger = logging.getLogger(__name__)

_MAX_DIFF_BYTES = 16_384


def _check_repo_allowed(repo_path: Path) -> None:
    """Reject ingesting a repository outside the configured allow-list.

    ``ingest_git_repo_tool`` is approval-gated, but the HITL gate is not the
    only defense — a prompt-injected tool call (or a mis-guided agent) must
    not be able to read an arbitrary local git repository.  Empty allow-list
    (the default) fails closed: nothing is readable until ``REPO_ALLOW_ROOT``
    is set.  ``repo_path`` is resolved (symlinks, ``..``) before the check so
    ``/allowed/../etc/repo`` cannot sneak past a root boundary.
    """
    from backend.shared.config import config

    allow_roots = config.repo_allow_roots
    if not allow_roots:
        raise ValueError(
            "Git ingestion is disabled: no REPO_ALLOW_ROOT configured. "
            "Set REPO_ALLOW_ROOT to the directory(ies) whose git "
            "repositories may be ingested."
        )

    resolved = repo_path.resolve()
    for root in allow_roots:
        root_path = Path(root).expanduser().resolve()
        try:
            resolved.relative_to(root_path)
        except ValueError:
            continue
        return  # inside an allowed root

    raise ValueError(
        f"Repository {resolved} is outside the allowed ingestion roots "
        f"({', '.join(allow_roots)}). Add its parent to REPO_ALLOW_ROOT "
        "to allow ingestion."
    )


async def ingest_repo(
    repo_path: str | Path,
    *,
    max_commits: int = 50,
    branch: str | None = None,
) -> list[dict[str, Any]]:
    """Walk *repo_path* history and feed each commit into the memory pipeline.

    Args:
        repo_path: Path to a local Git repository.
        max_commits: Maximum number of recent commits to process.
        branch: Branch to walk (default: HEAD).

    Returns:
        List of ``write_memory()`` result dicts.

    Raises:
        ValueError: if *repo_path* is outside ``config.repo_allow_roots``
            (the ingestion sandbox), or if no allow-list is configured.
        FileNotFoundError: if *repo_path* is not a git repository.
    """
    repo_path = Path(repo_path).expanduser()
    _check_repo_allowed(repo_path)
    repo_path = repo_path.resolve()
    if not (repo_path / ".git").exists():
        raise FileNotFoundError(f"Not a git repository: {repo_path}")

    repo = pygit2.Repository(str(repo_path))
    head = _resolve_head(repo, branch)

    results: list[dict[str, Any]] = []
    walker = repo.walk(head.id, pygit2.GIT_SORT_TIME | pygit2.GIT_SORT_TOPOLOGICAL)  # type: ignore[arg-type]

    for i, commit in enumerate(walker):
        if i >= max_commits:
            break
        result = await _ingest_commit(repo, commit, repo_path)
        if result:
            results.append(result)

    logger.info("Ingested %d/%d commits from %s", len(results), max_commits, repo_path)
    return results


# ── helpers ─────────────────────────────────────────────────────────


async def _ingest_commit(
    repo: pygit2.Repository, commit: pygit2.Commit, repo_path: Path
) -> dict[str, Any] | None:
    """Process a single commit through the memory pipeline."""
    # Assembling the LLM-ready commit block walks the full diff (pygit2
    # C code, plus line-by-line string building) — CPU/IO bound, so run it
    # in the thread pool instead of blocking the event loop.  ingest_repo
    # walks commits sequentially, so at most one diff is in flight at a time
    # and the shared ``repo`` object is never touched concurrently.
    content = await asyncio.to_thread(_format_commit, repo, commit)
    ts = _commit_time(commit)

    try:
        return await write_memory(
            content,
            source_type="git_commit",
            metadata={
                "commit_id": str(commit.id),
                "author": commit.author.name,
                "email": commit.author.email,
                "timestamp": ts.isoformat(),
                "repo": str(repo_path),
            },
        )
    except Exception:
        logger.exception("Failed to ingest commit %s", str(commit.id)[:8])
        return None


def _resolve_head(repo: pygit2.Repository, branch: str | None) -> pygit2.Commit:
    """Resolve HEAD or a named branch to a commit."""
    if branch and branch in repo.branches:
        ref = repo.branches[branch].peel()
    else:
        ref = repo.head.peel()

    if not isinstance(ref, pygit2.Commit):
        raise ValueError(f"Expected a commit, got {type(ref).__name__}")
    return ref


def _commit_time(commit: pygit2.Commit) -> datetime:
    """Convert pygit2 timestamp + offset to a timezone-aware datetime."""
    offset = timedelta(minutes=commit.author.offset)
    tz = timezone(offset)
    return datetime.fromtimestamp(commit.author.time, tz=tz)


def _format_commit(repo: pygit2.Repository, commit: pygit2.Commit) -> str:
    """Build a single LLM-ready text block for a commit."""
    msg = commit.message.strip()
    author_line = f"{commit.author.name} <{commit.author.email}>"

    parts = [
        f"Commit: {str(commit.id)[:8]}",
        f"Author: {author_line}",
        f"Message: {msg}",
    ]

    changed = _changed_files(repo, commit)
    if changed:
        parts.append("Files changed:")
        parts.extend(f"  {c}" for c in changed)

    diff_text = _format_diff(repo, commit)
    if diff_text:
        parts.append(f"\nDiff:\n{diff_text}")

    return "\n".join(parts)


def _changed_files(repo: pygit2.Repository, commit: pygit2.Commit) -> list[str]:
    """Return a compact list of ``path (+adds/-dels)`` for changed files."""
    diff = _get_diff(repo, commit)
    if diff is None:
        return []
    return [
        f"{p.delta.new_file.path} (+{p.line_stats[1]}/-{p.line_stats[2]})"
        for p in diff
    ]


def _format_diff(repo: pygit2.Repository, commit: pygit2.Commit) -> str:
    """Return a unified-diff string, truncated to ``_MAX_DIFF_BYTES``."""
    diff = _get_diff(repo, commit)
    if diff is None:
        return ""

    lines: list[str] = []
    for patch in diff:
        lines.append(
            f"--- a/{patch.delta.old_file.path}\n+++ b/{patch.delta.new_file.path}"
        )
        for hunk in patch.hunks:
            lines.append(hunk.header.rstrip())
            for line in hunk.lines:
                lines.append(line.content.rstrip("\n"))

    full = "\n".join(lines)
    full_bytes = full.encode("utf-8")
    if len(full_bytes) <= _MAX_DIFF_BYTES:
        return full

    # Truncate by bytes, not characters, to respect _MAX_DIFF_BYTES
    ellipsis = b"\n... (truncated)\n"
    half = (_MAX_DIFF_BYTES - len(ellipsis)) // 2
    head = full_bytes[:half]
    tail = full_bytes[-half:]
    return head.decode("utf-8", errors="replace") + ellipsis.decode() + tail.decode("utf-8", errors="replace")


def _get_diff(repo: pygit2.Repository, commit: pygit2.Commit):
    """Return the diff object for *commit* against its first parent."""
    if commit.parents:
        return repo.diff(commit.parents[0], commit)
    return commit.tree.diff_to_tree(swap=True)
