"""Labeled evaluation set for EMA retrieval — 70 queries, 5 categories.

Design choices:
    - **Content fingerprints** instead of UUIDs: ``relevant_fingerprints`` are
      distinctive substrings that must appear in the relevant memory's
      ``summary`` (and, for the chunks-table path, its ``content``). This
      keeps the labeled set portable across DB rebuilds and reproducible in CI.
    - ``seed_ids`` cross-references ``seed_memories.jsonl`` so
      ``dataset.validate()`` can assert every fingerprint is (a) present in
      exactly one seed memory and (b) present in the seed(s) it claims.
    - ``difficulty`` tags how much semantic lifting the retriever must do:
        easy   — query shares surface terms with the target summary
        medium — query is paraphrased or uses synonyms
        hard   — query is conceptual / asks "why" with no lexical overlap
    - Categories are chosen to mirror the five memory-buckets EMA actually
      stores, so per-category numbers map directly to product quality.

The corpus is intentionally EMA's own engineering history: every query is a
real question a new contributor would ask. This makes the eval set double as
onboarding material — the numbers answer "how well does EMA remember its own
decisions?".

The labeled queries themselves live in ``data/ground_truth.jsonl`` (one JSON
row per query); this module keeps the accessor, constants and the import-time
consistency guard.
"""

from __future__ import annotations

from collections.abc import Sequence

from tests.eval.core import load_jsonl_items

CATEGORIES: tuple[str, ...] = (
    "技术决策",
    "故障复盘",
    "架构设计",
    "代码实现",
    "历史背景",
)

# Difficulty buckets — mirrored by runner.by_difficulty and report._difficulty_table.
# Keep as a tuple so callers can iterate in stable display order.
DIFFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")


class GroundTruthItem(dict):
    """Typed dict-like accessor for a single labeled query.

    Subclassing ``dict`` keeps JSON serialization trivial while providing
    attribute-style access for the runner.
    """

    @property
    def id(self) -> str:
        return str(self["id"])

    @property
    def query(self) -> str:
        return str(self["query"])

    @property
    def seed_ids(self) -> list[str]:
        return list(self["seed_ids"])

    @property
    def relevant_fingerprints(self) -> list[str]:
        return list(self["relevant_fingerprints"])

    @property
    def category(self) -> str:
        return str(self["category"])

    @property
    def difficulty(self) -> str:
        return str(self.get("difficulty", "medium"))

    @property
    def notes(self) -> str:
        return str(self.get("notes", ""))


# Loaded once at import from the JSONL data file.  Kept as a module-level name
# (rather than only inside the loaders) because tests monkeypatch it and the
# locust query pool reads it directly.
GROUND_TRUTH: list[GroundTruthItem] = load_jsonl_items(
    "ground_truth.jsonl", GroundTruthItem
)


def by_category(items: Sequence[GroundTruthItem]) -> dict[str, list[GroundTruthItem]]:
    """Group labeled items by category, preserving order."""
    out: dict[str, list[GroundTruthItem]] = {c: [] for c in CATEGORIES}
    for it in items:
        out.setdefault(it.category, []).append(it)
    return out


def difficulty_distribution(items: Sequence[GroundTruthItem]) -> dict[str, int]:
    """Count items per difficulty bucket — used in the report header."""
    out: dict[str, int] = {d: 0 for d in DIFFICULTIES}
    for it in items:
        out[it.difficulty] = out.get(it.difficulty, 0) + 1
    return out


def assert_complete() -> None:
    """Sanity-check the labeled set at import time of the CLI.

    Cheap invariants that catch typos before a 20-minute eval run:
        - IDs are unique
        - Every category has ≥1 item
        - Every item has ≥1 fingerprint and ≥1 seed_id
    """
    ids = [it.id for it in GROUND_TRUTH]
    if len(set(ids)) != len(ids):
        dupes = {i for i in ids if ids.count(i) > 1}
        raise ValueError(f"duplicate query ids: {sorted(dupes)}")
    for cat in CATEGORIES:
        if not any(it.category == cat for it in GROUND_TRUTH):
            raise ValueError(f"category has no items: {cat}")
    for it in GROUND_TRUTH:
        if not it.relevant_fingerprints:
            raise ValueError(f"{it.id}: empty relevant_fingerprints")
        if not it.seed_ids:
            raise ValueError(f"{it.id}: empty seed_ids")
