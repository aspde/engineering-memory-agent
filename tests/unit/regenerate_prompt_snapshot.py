"""Regenerate the prompt-registry snapshot after a *versioned* prompt change.

The snapshot (``tests/unit/_prompt_snapshot.json``) pins every registered
prompt's ``(version, sha256)``.  ``test_prompts.py`` fails when the current
registry drifts from it — that is what forces a version bump on any text
edit.  This script rewrites the snapshot for the *current* registry.

Usage (after bumping the version of every changed prompt):

    python -m tests.unit.regenerate_prompt_snapshot

It refuses to write (exit 1) when a prompt's text changed but its version
did not — bumping the version first is the point of the mechanism, and a
snapshot that absorbs an un-versioned edit would silently re-enable the
behaviour the test exists to catch.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from backend.service.prompts import _PROMPTS

_SNAPSHOT_PATH = Path(__file__).resolve().parent / "_prompt_snapshot.json"


def main() -> int:
    current = {
        key: {
            "version": spec.version,
            "sha256": hashlib.sha256(spec.text.encode("utf-8")).hexdigest(),
        }
        for key, spec in sorted(_PROMPTS.items())
    }

    previous = json.loads(_SNAPSHOT_PATH.read_text(encoding="utf-8")) if _SNAPSHOT_PATH.exists() else {}

    # Guard: a changed text must carry a changed version.  New prompts are
    # exempt (their first registration has no prior version to bump).
    violations: list[str] = []
    for key, entry in current.items():
        prev = previous.get(key)
        if prev is None:
            continue
        if entry["sha256"] != prev["sha256"] and entry["version"] == prev["version"]:
            violations.append(
                f"{key}: text changed but version is still v{entry['version']} — "
                "bump the version before regenerating"
            )
    if violations:
        print("Snapshot regeneration blocked:", file=sys.stderr)
        for v in violations:
            print(f"  ✗ {v}", file=sys.stderr)
        return 1

    _SNAPSHOT_PATH.write_text(
        json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"✓ Prompt snapshot updated ({len(current)} prompts) → {_SNAPSHOT_PATH.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
