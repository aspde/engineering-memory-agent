"""Verify docs declarations match actual code state.

These tests are assertions, not suggestions — a failure means docs are stale.
They stay lightweight and fast: no DB, no LLM, no network.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = PROJECT_ROOT / "docs"


# ── helpers ──────────────────────────────────────────────────────────────────

class Failure(NamedTuple):
    check: str
    detail: str


def _requirements() -> set[str]:
    """Parse requirements.txt into a set of package names (lowercased)."""
    path = PROJECT_ROOT / "requirements.txt"
    if not path.exists():
        return set()
    pkgs: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # strip extras and version: "foo[bar]>=1.0" → "foo"
        pkg = re.split(r"[\[<>=!~;]", line)[0].strip().lower()
        if pkg:
            pkgs.add(pkg)
    return pkgs


def _read_doc(filename: str) -> str:
    path = DOCS_DIR / filename
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _failures_to_message(failures: list[Failure]) -> str:
    lines = [f"{len(failures)} doc inconsistency found:"]
    for f in failures:
        lines.append(f"  [{f.check}] {f.detail}")
    return "\n".join(lines)


# ── checks ───────────────────────────────────────────────────────────────────

def test_docs_directory_exists() -> None:
    """docs/ dir and its key files exist."""
    missing = []
    required = ["README.md", "architecture.md", "agent-design.md", "memory-system.md", "deployment.md"]
    for name in required:
        if not (DOCS_DIR / name).exists():
            missing.append(name)
    assert not missing, f"Missing docs: {missing}"


def test_tech_stack_matches_requirements() -> None:
    """Key packages declared in docs/architecture.md appear in requirements.txt."""
    failures: list[Failure] = []

    architecture_text = _read_doc("architecture.md")
    reqs = _requirements()

    # packages the architecture doc claims we use
    claims = [
        ("fastapi", "FastAPI"),
        ("langgraph", "LangGraph"),
        ("pgvector", "pgvector"),
        ("sqlalchemy", "SQLAlchemy"),
        ("streamlit", "Streamlit"),
        ("pytest", "pytest"),
        ("gitpython", "GitPython"),
    ]

    for pkg, label in claims:
        if label.lower() in architecture_text.lower():
            if pkg not in reqs:
                failures.append(Failure("tech-stack", f"docs claims {label} but '{pkg}' not in requirements.txt"))

    assert not failures, _failures_to_message(failures)


def test_database_matches_docker_compose() -> None:
    """docs claims PostgreSQL + pgvector; docker-compose must agree."""
    compose = PROJECT_ROOT / "docker-compose.yml"
    if not compose.exists():
        return  # no docker-compose means nothing to check

    compose_text = compose.read_text()
    failures: list[Failure] = []

    if "pgvector" not in compose_text:
        failures.append(Failure("storage", "docker-compose.yml does not use pgvector image"))
    if "postgres" not in compose_text.lower():
        failures.append(Failure("storage", "docker-compose.yml does not reference postgres"))

    assert not failures, _failures_to_message(failures)


def test_project_structure_matches_docs() -> None:
    """Key directories docs reference actually exist."""
    failures: list[Failure] = []

    expected = ["backend", "frontend", "agent", "tests", "docs"]
    for d in expected:
        if not (PROJECT_ROOT / d).is_dir():
            failures.append(Failure("structure", f"docs references '{d}/' but directory does not exist"))

    # agent/ sub-modules must exist
    agent_dir = PROJECT_ROOT / "agent"
    agent_files = ["state.py", "tools.py", "nodes.py", "graph.py"]
    for f in agent_files:
        if not (agent_dir / f).exists():
            failures.append(Failure("structure", f"agent/{f} does not exist"))

    assert not failures, _failures_to_message(failures)


def test_agent_api_route_declared() -> None:
    """docs/architecture.md lists /api/agent/chat endpoint → router must register it."""
    architecture_text = _read_doc("architecture.md")
    if "/api/agent/chat" not in architecture_text:
        return  # not claimed in docs, skip

    router = PROJECT_ROOT / "backend" / "api" / "router.py"
    if not router.exists():
        return

    router_text = router.read_text()
    if "agent_routes" not in router_text or "agent_router" not in router_text:
        assert False, "docs claim /api/agent/chat but router does not register agent routes"


def test_pyproject_config_consistent() -> None:
    """pyproject.toml declares Python version; architecture.md should agree."""
    pyproject = PROJECT_ROOT / "pyproject.toml"
    if not pyproject.exists():
        return

    pyproject_text = pyproject.read_text()
    architecture_text = _read_doc("architecture.md")

    m = re.search(r"requires-python.*?3\.(\d+)", pyproject_text, re.IGNORECASE)
    if not m:
        return

    expected = f"3.{m.group(1)}"
    if f"Python {expected}" in architecture_text or f"python {expected}" in architecture_text.lower():
        return

    assert False, f"docs/architecture.md claims Python version but pyproject.toml requires {expected}"


def test_no_deleted_docs_referenced() -> None:
    """No file in the project references docs files that were deleted."""
    deleted = ["tech-stack.md", "rag-design.md", "git-knowledge.md"]
    failures: list[Failure] = []

    # Only check project-owned markdown files, not .venv
    for md in PROJECT_ROOT.glob("**/*.md"):
        if ".venv" in str(md):
            continue
        text = md.read_text(encoding="utf-8", errors="ignore")
        for name in deleted:
            if name in text:
                failures.append(Failure("stale-ref", f"{md.relative_to(PROJECT_ROOT)} still references deleted {name}"))

    assert not failures, _failures_to_message(failures)
