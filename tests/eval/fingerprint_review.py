"""Fingerprint extraction for approved query candidates — review helper.

Query candidates produced by ``generate_queries.py`` carry query / seed_ids
/ difficulty but NOT the ``relevant_fingerprints`` the retrieval eval needs
(a retrieved memory counts as relevant iff its text contains one).  Choosing
fingerprints is deliberately manual — a wrong anchor silently distorts eval
scores — so this script's job is to make that review fast, not to replace it.

For each approved candidate it extracts candidate fingerprints from the
*target* memory (for hard negatives: the memory the query truly intends, i.e.
``seed_ids[0]`` — never the distractor) and flags each one:

- ``unique`` — appears in exactly one seed (otherwise relevance is ambiguous;
  ``validate_dataset`` warns on it);
- ``in_summary`` — appears in the seed's ``summary`` (the memory retriever's
  match field).  A content-only fingerprint can never be hit by the memory
  retrieval path, so it is de-emphasised;
- ``source`` — ``entity`` (from the seed's ``entities`` field) or ``phrase``
  (an n-gram of the summary).

``suggested_fingerprints`` is the top-scoring unique, summary-matching
fingerprint(s) — a starting point the reviewer edits, never an auto-promotion.

Usage::

    # Review mode: approved candidates → fingerprint review file
    python -m tests.eval.fingerprint_review

    # Include candidate rows regardless of their review status
    python -m tests.eval.fingerprint_review --all

    # Apply mode: approved + chosen rows → ground_truth_additions.jsonl
    python -m tests.eval.fingerprint_review --apply

    # Code mode: render the same additions as paste-ready GroundTruthItem(...)
    # entries (grouped by category), ready for ground_truth.py's GROUND_TRUTH
    python -m tests.eval.fingerprint_review --code
    python -m tests.eval.fingerprint_review --code --out /tmp/entries.txt

The apply/code steps re-validate every chosen fingerprint against the seed
corpus (substring + uniqueness) and refuse rows that fail — a fingerprint
that cannot match is a fingerprint that would silently zero out the query's
score.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tests.eval.dataset import SeedMemory, load_seed_memories

DEFAULT_CANDIDATES = Path(__file__).parent / "query_candidates.jsonl"
DEFAULT_REVIEW = Path(__file__).parent / "fingerprint_review.jsonl"
DEFAULT_ADDITIONS = Path(__file__).parent / "ground_truth_additions.jsonl"

# Fingerprints that are too long break under paraphrase (a rewrite drops the
# tail); too short loses discrimination.  Prefer the 6-14 band.
_IDEAL_MIN, _IDEAL_MAX = 6, 14
_ACCEPTABLE_MIN, _ACCEPTABLE_MAX = 4, 40

# Words that carry no retrieval signal when they open/close a phrase.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "的", "了", "是", "在", "和", "与", "或", "为", "用", "做", "对", "就", "都",
        "也", "而", "更", "很", "被", "把", "从", "到", "中", "上", "下", "里", "于",
        "其", "这", "那", "因", "所以", "但是", "以及", "通过", "进行", "一个",
        "我们", "你们", "他们", "这个", "那个", "之后", "之前", "当时", "现在",
        "已经", "会", "能", "要", "可以", "需要", "应该", "可能", "比较", "主要",
        "目前", "则", "并", "且", "跟", "给", "往", "以", "对", "让", "使", "将",
        "作为", "其中", "这些", "那些", "这里", "那里", "一些", "还是", "是否",
        "如何", "为什么", "怎么", "怎样", "存在", "出现", "相关", "有关", "涉及",
        "以及", "所有", "任何", "每个", "各自", "分别",
    }
)
_EN_WORDS: frozenset[str] = frozenset(
    {"the", "a", "an", "of", "to", "in", "for", "with", "on", "and", "or", "as", "is", "are"}
)

# Token types: english identifier (with optional key=value suffix),
# number/ratio/version, CJK run.
_WORD_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9_.\-]*(?:=[A-Za-z0-9._\-]+)?"   # identifiers / key=value / versions
    r"|\d+(?:[./%\-]\d+)*"                              # numbers, ratios, ranges
    r"|[一-鿿]+"                                # CJK runs
)

_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9_]")
_NUMERIC_ONLY_RE = re.compile(r"^\d+[./%\-]?\d*$")


# ── Tokenisation & n-gram phrase candidates ────────────────────────


def _tokens(text: str) -> list[str]:
    """Split *text* into word tokens (identifiers, numbers, CJK runs)."""
    return _WORD_RE.findall(text)


def _looks_like_identifier(token: str) -> bool:
    """Single tokens that carry signal alone: identifiers / numbers."""
    return bool(_IDENTIFIER_RE.search(token))


def _is_stopword(token: str) -> bool:
    if token in _STOPWORDS:
        return True
    low = token.lower()
    if low in _EN_WORDS:
        return True
    return token.isascii() and len(token) <= 2 and not _looks_like_identifier(token)


def _clean_phrase(tokens: Sequence[str]) -> str | None:
    """Join *tokens* into a fingerprint candidate, or None when unusable.

    Trims leading/trailing stopwords, requires a meaningful core, and returns
    a phrase that is guaranteed to be a substring of the source once checked
    against it (both space-joined and glued forms are tried by the caller).
    """
    parts = [t for t in tokens if t]
    while parts and _is_stopword(parts[0]):
        parts = parts[1:]
    while parts and _is_stopword(parts[-1]):
        parts = parts[:-1]
    if not parts:
        return None
    if all(_is_stopword(p) for p in parts):
        return None
    # A single token is only usable when it is an identifier/number — a lone
    # CJK word ("选型") is too weak to anchor a relevance match.
    if len(parts) == 1 and not _looks_like_identifier(parts[0]):
        return None
    # Pure numeric fragments ("0.92" alone, no context) are too ambiguous.
    if len(parts) == 1 and _NUMERIC_ONLY_RE.match(parts[0]):
        return None
    return " ".join(parts)


def _as_substrings(phrase: str, text: str) -> list[str]:
    """Return forms of *phrase* that actually occur in *text*.

    Tries the space-joined form first (English phrases like "pgvector 扩展")
    then the glued form (CJK/identifier runs like "LangGraph而非").  Only
    forms that are true substrings are returned — a fingerprint that does not
    occur in the memory can never be matched.
    """
    forms: list[str] = []
    for form in (phrase, phrase.replace(" ", "")):
        if form in text and form not in forms:
            forms.append(form)
    return forms


def _phrase_candidates(text: str, max_span: int = 4) -> list[str]:
    """All clean substrings of *text* built from 1..max_span tokens."""
    toks = _tokens(text)
    out: list[str] = []
    seen: set[str] = set()
    for start in range(len(toks)):
        for width in range(1, max_span + 1):
            if start + width > len(toks):
                break
            phrase = _clean_phrase(toks[start : start + width])
            if phrase is None:
                continue
            for form in _as_substrings(phrase, text):
                if form not in seen:
                    seen.add(form)
                    out.append(form)
    return out


# ── Per-candidate extraction ───────────────────────────────────────


def extract_fingerprint_candidates(
    target: SeedMemory,
    all_seeds: Sequence[SeedMemory],
    *,
    max_phrase_span: int = 4,
) -> list[dict[str, Any]]:
    """Candidate fingerprints for *target*, annotated for review.

    Each entry: ``{"fingerprint", "source", "unique", "in_summary"}``.
    ``unique`` means the text appears in no *other* seed's summary or content;
    ``in_summary`` means it appears in the target's summary (the memory
    retriever's match field).
    """
    others = [s for s in all_seeds if s.id != target.id]
    other_texts = [f"{s.summary}\n{s.content}" for s in others]

    def _hits(text: str) -> int:
        return sum(1 for other in other_texts if text in other)

    candidates: list[dict[str, Any]] = []
    # Dedupe by fingerprint TEXT (not by source): an entity name also spans a
    # phrase n-gram ("pgvector" is both seed-001's entity and a phrase token).
    # Entity wins the slot — its annotation is richer.
    seen: set[str] = set()

    # Entity names first — they are the highest-signal anchors.
    for e in target.entities or []:
        name = str(e.get("name", "")).strip()
        if not name:
            continue
        if not (len(name) >= 2 or _looks_like_identifier(name)):
            continue
        if name in seen:
            continue
        seen.add(name)
        candidates.append(
            {
                "fingerprint": name,
                "source": "entity",
                "unique": _hits(name) == 0,
                "in_summary": name in target.summary,
            }
        )

    # Phrase n-grams from the summary (content-only phrases are weaker — the
    # memory retriever matches against the summary).
    for text, summary_only in ((target.summary, True), (target.content, False)):
        for phrase in _phrase_candidates(text, max_phrase_span):
            if not (len(phrase) >= _ACCEPTABLE_MIN and len(phrase) <= _ACCEPTABLE_MAX):
                continue
            if phrase in seen:
                continue
            seen.add(phrase)
            candidates.append(
                {
                    "fingerprint": phrase,
                    "source": "phrase",
                    "unique": _hits(phrase) == 0,
                    "in_summary": summary_only or phrase in target.summary,
                }
            )
    candidates.sort(
        key=lambda c: (
            not c["unique"],           # unique candidates first
            not c["in_summary"],       # then those that hit the summary
            abs(len(c["fingerprint"]) - 10),  # then near the ideal length
        )
    )
    return candidates


def _score(c: dict[str, Any]) -> float:
    """Rank a candidate fingerprint for the ``suggested`` selection."""
    if not c["unique"] or not c["in_summary"]:
        return -1.0
    length = len(c["fingerprint"])
    score = 0.0
    if _IDEAL_MIN <= length <= _IDEAL_MAX:
        score += 3.0
    elif length >= _ACCEPTABLE_MIN and length <= _ACCEPTABLE_MAX:
        score += 1.0
    if c["source"] == "entity":
        score += 1.5
    if re.search(r"[A-Za-z0-9]", c["fingerprint"]):
        score += 0.5
    return score


def select_suggested(candidates: Sequence[dict[str, Any]], max_suggested: int = 2) -> list[str]:
    """Top ``max_suggested`` fingerprint strings, or [] when none qualify."""
    scorable = [c for c in candidates if _score(c) >= 0]
    scorable.sort(key=lambda c: (-_score(c), len(c["fingerprint"])))
    return [c["fingerprint"] for c in scorable[:max_suggested]]


# ── Review file ────────────────────────────────────────────────────


def build_review_rows(
    approved_items: Sequence[dict[str, Any]],
    seeds: Sequence[SeedMemory],
) -> list[dict[str, Any]]:
    """One review row per candidate with extracted fingerprint options."""
    seed_by_id: dict[str, SeedMemory] = {s.id: s for s in seeds}
    rows: list[dict[str, Any]] = []
    for item in approved_items:
        target_id = next(iter(item.get("seed_ids") or []), None)
        target = seed_by_id.get(target_id) if target_id else None
        if target is None:
            # The candidate points at a seed the corpus does not have — the
            # underlying candidate is unusable; record the row so the reviewer
            # sees why.
            rows.append(
                {
                    "candidate_id": item.get("id"),
                    "query": item.get("query", ""),
                    "kind": item.get("kind", "positive"),
                    "seed_ids": item.get("seed_ids", []),
                    "category": item.get("category", ""),
                    "difficulty": item.get("difficulty", "medium"),
                    "target_missing": True,
                    "candidate_fingerprints": [],
                    "suggested_fingerprints": [],
                    "chosen_fingerprints": None,
                    "status": "pending",
                }
            )
            continue

        candidates = extract_fingerprint_candidates(target, seeds)
        suggested = select_suggested(candidates)
        rows.append(
            {
                "candidate_id": item.get("id"),
                "query": item.get("query", ""),
                "kind": item.get("kind", "positive"),
                "seed_ids": item.get("seed_ids", []),
                "category": item.get("category", ""),
                "difficulty": item.get("difficulty", "medium"),
                "target_missing": False,
                "candidate_fingerprints": candidates[:12],
                "suggested_fingerprints": suggested,
                "chosen_fingerprints": None,
                "status": "pending",
            }
        )
    return rows


# ── Apply: review → ground-truth additions ─────────────────────────


def _validate_chosen(
    fingerprints: Sequence[str], target: SeedMemory, seeds: Sequence[SeedMemory]
) -> list[str]:
    """Refuse fingerprints that cannot match under the eval's matching rules.

    Returns a list of error messages (empty == all fingerprints usable).  A
    fingerprint that is not a substring of the target's summary or content can
    never be matched by either retriever; one that appears in another seed
    makes relevance ambiguous (validate_dataset warns, the eval degrades).
    """
    errors: list[str] = []
    others = [s for s in seeds if s.id != target.id]
    for fp in fingerprints:
        if not (fp in target.summary or fp in target.content):
            errors.append(
                f"'{fp}' is not a substring of {target.id} summary/content — "
                "no retriever can ever match it"
            )
        elif any(fp in s.summary or fp in s.content for s in others):
            errors.append(
                f"'{fp}' appears in another seed — relevance would be ambiguous"
            )
    return errors


def finalize_review(
    review_rows: Sequence[dict[str, Any]],
    seeds: Sequence[SeedMemory],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Approved + chosen rows → ground_truth_additions entries.

    Returns ``(additions, warnings)``.  Rows with no chosen fingerprints or
    chosen fingerprints that fail validation are skipped with a warning —
    never written half-validated.  Duplicate candidate ids are dropped
    (a re-applied review would otherwise emit the same query twice).
    """
    seed_by_id: dict[str, SeedMemory] = {s.id: s for s in seeds}
    additions: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()

    for row in review_rows:
        if row.get("status") != "approved":
            continue
        cid = row.get("candidate_id")
        if cid in seen_ids:
            warnings.append(f"{cid}: duplicate approved row — skipped")
            continue
        chosen = row.get("chosen_fingerprints") or []
        if not chosen:
            warnings.append(f"{cid}: approved but no chosen fingerprints — skipped")
            continue
        target_id = next(iter(row.get("seed_ids") or []), None)
        target = seed_by_id.get(target_id) if target_id else None
        if target is None:
            warnings.append(f"{cid}: target seed {target_id} not found — skipped")
            continue
        errors = _validate_chosen(chosen, target, seeds)
        if errors:
            for err in errors:
                warnings.append(f"{cid}: {err}")
            continue
        seen_ids.add(cid)
        additions.append(
            {
                "id": cid,
                "query": row.get("query", ""),
                "seed_ids": list(row.get("seed_ids") or []),
                "relevant_fingerprints": list(chosen),
                "category": row.get("category", ""),
                "difficulty": row.get("difficulty", "medium"),
                "notes": f"generated from {target_id} (reviewed)",
            }
        )
    return additions, warnings


# ── Python-code rendering (--code) ───────────────────────────────
# ``--apply`` writes machine-readable JSONL; ``--code`` renders the same
# additions as paste-ready ``GroundTruthItem(...)`` entries matching the
# style of ``ground_truth.py``, so the manual merge step is copy-paste.


def _py_str(value: str) -> str:
    """Render *value* as a double-quoted Python string literal.

    Escapes exactly the characters Python requires (``\\``, ``\"``, and
    whitespace/control chars); everything else — CJK in particular — stays
    verbatim so the snippet reads like the hand-written ground truth.
    """
    out: list[str] = ['"']
    for ch in value:
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 32:
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def render_python_code(additions: Sequence[dict[str, Any]]) -> str:
    """Render *additions* as a paste-ready block for ``ground_truth.py``.

    Entries are grouped by category (first-appearance order), each group
    under the same ``# ── category (N) ──`` divider the file uses.  The
    block is a self-contained list of ``GroundTruthItem(...)`` calls with
    nothing before it but a comment — paste it inside ``GROUND_TRUTH`` and
    run ``--validate-only``.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for a in additions:
        cat = str(a.get("category", ""))
        if cat not in grouped:
            grouped[cat] = []
            order.append(cat)
        grouped[cat].append(a)

    lines: list[str] = [
        "# ── Generated by tests.eval.fingerprint_review --code ──",
        "# Paste the following entries into the GROUND_TRUTH list in",
        "# tests/eval/ground_truth.py, then validate:",
        "#   python -m tests.eval.run_eval --validate-only",
    ]
    for cat in order:
        rows = grouped[cat]
        lines.append("")
        lines.append(f"    # ── {cat} ({len(rows)}) ──" + "─" * max(0, 10 - len(cat)))
        for a in rows:
            lines.append("    GroundTruthItem(")
            lines.append(f"        id={_py_str(str(a.get('id', '')))},")
            lines.append(f"        query={_py_str(str(a.get('query', '')))},")
            seeds = ", ".join(_py_str(str(s)) for s in (a.get("seed_ids") or []))
            lines.append(f"        seed_ids=[{seeds}],")
            fps = ", ".join(_py_str(str(f)) for f in (a.get("relevant_fingerprints") or []))
            lines.append(f"        relevant_fingerprints=[{fps}],")
            lines.append(f"        category={_py_str(str(a.get('category', '')))},")
            lines.append(f"        difficulty={_py_str(str(a.get('difficulty', 'medium')))},")
            notes = a.get("notes")
            if notes:
                lines.append(f"        notes={_py_str(str(notes))},")
            lines.append("    ),")
    lines.append("")
    return "\n".join(lines)


# ── File IO ────────────────────────────────────────────────────────


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    path.write_text(text, encoding="utf-8")


# ── CLI ────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.eval.fingerprint_review",
        description="Extract fingerprint candidates for approved query "
        "candidates (review helper) or finalize chosen ones into additions.",
    )
    p.add_argument(
        "--candidates", default=str(DEFAULT_CANDIDATES),
        help=f"Candidates file (JSONL). Default: {DEFAULT_CANDIDATES}.",
    )
    p.add_argument(
        "--review", default=str(DEFAULT_REVIEW),
        help=f"Review file to write/read. Default: {DEFAULT_REVIEW}.",
    )
    p.add_argument(
        "--additions", default=str(DEFAULT_ADDITIONS),
        help=f"Output additions file. Default: {DEFAULT_ADDITIONS}.",
    )
    p.add_argument(
        "--all", action="store_true",
        help="Review-mode: include candidates regardless of their status "
        "(default: approved only).",
    )
    p.add_argument(
        "--apply", action="store_true",
        help="Apply-mode: read the review file, write approved+chosen rows "
        "as ground_truth_additions (JSONL).",
    )
    p.add_argument(
        "--code", action="store_true",
        help="Render approved+chosen rows as a paste-ready Python snippet "
        "for ground_truth.py (printed to stdout unless --out is set). "
        "Mutually exclusive with --apply.",
    )
    p.add_argument(
        "--out", default=None,
        help="Where to write the --code snippet. Default: stdout.",
    )
    return p


def _review(args: argparse.Namespace, seeds: Sequence[SeedMemory]) -> int:
    candidates = _read_jsonl(Path(args.candidates))
    if not candidates:
        print(
            f"No candidates found in {args.candidates} — run "
            "tests.eval.generate_queries first.",
            file=sys.stderr,
        )
        return 1
    items = list(candidates) if args.all else [c for c in candidates if c.get("status") == "approved"]
    if not items:
        print(
            "No approved candidates (status='approved') in the candidates file. "
            "Run with --all to include un-reviewed rows.",
            file=sys.stderr,
        )
        return 1
    rows = build_review_rows(items, seeds)
    _write_jsonl(Path(args.review), rows)
    missing = sum(1 for r in rows if r.get("target_missing"))
    print(
        f"✓ wrote {len(rows)} review rows → {args.review} "
        f"({len(rows) - missing} with fingerprint candidates, {missing} missing target seed)",
        file=sys.stderr,
    )
    print(
        "Review: edit chosen_fingerprints (or accept suggested) and set "
        "status to approved/rejected, then run --apply.",
        file=sys.stderr,
    )
    return 0


def _apply(args: argparse.Namespace, seeds: Sequence[SeedMemory]) -> int:
    review_path = Path(args.review)
    rows = _read_jsonl(review_path)
    if not rows:
        print(f"No review rows in {args.review}.", file=sys.stderr)
        return 1
    additions, warnings = finalize_review(rows, seeds)
    for w in warnings:
        print(f"  ⚠ {w}", file=sys.stderr)
    if additions:
        _write_jsonl(Path(args.additions), additions)
        print(
            f"✓ wrote {len(additions)} additions → {args.additions}",
            file=sys.stderr,
        )
    else:
        print("No usable additions (approve rows and set chosen_fingerprints first).", file=sys.stderr)
    return 0 if additions else 1


def _code(args: argparse.Namespace, seeds: Sequence[SeedMemory]) -> int:
    review_path = Path(args.review)
    rows = _read_jsonl(review_path)
    if not rows:
        print(f"No review rows in {args.review}.", file=sys.stderr)
        return 1
    additions, warnings = finalize_review(rows, seeds)
    for w in warnings:
        print(f"  ⚠ {w}", file=sys.stderr)
    if not additions:
        print(
            "No usable additions (approve rows and set chosen_fingerprints first).",
            file=sys.stderr,
        )
        return 1
    snippet = render_python_code(additions)
    if args.out:
        Path(args.out).write_text(snippet, encoding="utf-8")
        print(f"✓ wrote Python snippet → {args.out}", file=sys.stderr)
    else:
        print(snippet)
    return 0


def main() -> None:
    args = _build_parser().parse_args()
    if args.apply and args.code:
        print("--apply and --code are mutually exclusive", file=sys.stderr)
        sys.exit(2)
    seeds = load_seed_memories()
    if args.code:
        rc = _code(args, seeds)
    elif args.apply:
        rc = _apply(args, seeds)
    else:
        rc = _review(args, seeds)
    sys.exit(rc)


if __name__ == "__main__":
    main()
