"""Tests for the git-ingestion sandbox (REPO_ALLOW_ROOT).

``ingest_git_repo_tool`` reads a local git repository's history — a real file
access primitive.  The allow-list check is what keeps a prompt-injected tool
call (or a mis-guided agent) from reading an arbitrary local repo, so the
path-validation logic is locked down independently of pygit2 (no real
repository is opened here).
"""

from __future__ import annotations

import pytest

from backend.service.ingestion import _check_repo_allowed


@pytest.fixture(autouse=True)
def _config():
    """Expose the shared config singleton so tests can set the allow-list."""
    from backend.shared.config import config

    return config


class TestCheckRepoAllowed:
    def test_fails_closed_when_no_roots_configured(self, _config, tmp_path) -> None:
        """Empty allow-list (the default) rejects every ingest."""
        import backend.service.ingestion as mod

        _config.repo_allow_roots = ()
        with pytest.raises(ValueError, match="REPO_ALLOW_ROOT"):
            mod._check_repo_allowed(tmp_path)

    def test_allows_repo_inside_configured_root(self, _config, tmp_path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        (root / "repo").mkdir()

        _config.repo_allow_roots = (str(root),)
        # No exception → inside the allow-list.
        _check_repo_allowed(root / "repo")

    def test_allows_root_itself(self, _config, tmp_path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        _config.repo_allow_roots = (str(root),)
        _check_repo_allowed(root)

    def test_rejects_repo_outside_configured_root(self, _config, tmp_path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "other"
        outside.mkdir()

        _config.repo_allow_roots = (str(root),)
        with pytest.raises(ValueError, match="outside the allowed ingestion roots"):
            _check_repo_allowed(outside)

    def test_rejects_when_no_root_matches(self, _config, tmp_path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        (root_b / "repo").mkdir()

        _config.repo_allow_roots = (str(root_a),)
        with pytest.raises(ValueError):
            _check_repo_allowed(root_b / "repo")

    def test_parent_escape_is_resolved_before_check(self, _config, tmp_path) -> None:
        """``/allowed/sub/../repo`` must resolve inside the root, not escape it."""
        root = tmp_path / "root"
        root.mkdir()
        (root / "sub").mkdir()
        (root / "repo").mkdir()

        _config.repo_allow_roots = (str(root),)
        # Path textually contains ".." but resolves to root/repo → allowed.
        _check_repo_allowed(root / "sub" / ".." / "repo")

    def test_escape_outside_root_rejected(self, _config, tmp_path) -> None:
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        _config.repo_allow_roots = (str(root),)
        # A path that resolves outside the root is rejected even though its
        # textual prefix starts with the root.
        with pytest.raises(ValueError, match="outside"):
            _check_repo_allowed(root / ".." / "outside")

    def test_multiple_roots_any_match_allows(self, _config, tmp_path) -> None:
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()
        (root_b / "repo").mkdir()

        _config.repo_allow_roots = (str(root_a), str(root_b))
        _check_repo_allowed(root_b / "repo")


class TestIngestRepoSandbox:
    @pytest.mark.asyncio
    async def test_ingest_repo_rejects_outside_sandbox_before_fs_access(
        self, _config, tmp_path
    ) -> None:
        """``ingest_repo`` fails at the sandbox check, before pygit2 touches
        the path — no real repository is needed."""
        from backend.service.ingestion import ingest_repo

        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        _config.repo_allow_roots = (str(root),)

        with pytest.raises(ValueError, match="outside the allowed ingestion roots"):
            await ingest_repo(outside / "any-repo-path")

    @pytest.mark.asyncio
    async def test_ingest_repo_fails_closed_without_config(self, _config, tmp_path) -> None:
        from backend.service.ingestion import ingest_repo

        _config.repo_allow_roots = ()
        with pytest.raises(ValueError, match="REPO_ALLOW_ROOT"):
            await ingest_repo(tmp_path)
