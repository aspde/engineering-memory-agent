"""LLM query-generation for the retrieval eval — cold-start data expansion.

The retrieval eval's labeled set (``tests/eval/ground_truth.py``, 30
queries) is hand-authored and static.  This script is the cold-start
expansion path: given the seed corpus (``tests/eval/seed_memories.jsonl``,
30 real memories) it asks the LLM to generate, for each memory:

- 3 **positive queries** (easy / medium / hard) — how a real user asking
  for this memory would phrase it;
- 1 **hard negative** — a query whose true intent is a *different* memory
  but whose wording makes the retriever likely to rank this one first
  (the discriminator case that extra positives never exercise).

Output is a candidate file (default ``tests/eval/experiments/query_candidates.jsonl``)
for HUMAN REVIEW — LLM-labeled queries are not trusted blindly, they are a
first-pass generator feeding a manual filter (the review step is
deliberately manual; there is no auto-promotion into ground_truth.py).

Candidates carry a ``status`` field ("candidate"); a reviewer flips it to
"approved" / "rejected" and fills ``review``.  Approved rows can then be
hand-converted into ``ground_truth.py`` entries.

The ``source_seed_id`` field records which memory generated each candidate,
so a re-run skips seeds already in the output file (idempotent resume) and
a reviewer knows where each query came from.

Usage::

    # Plan only — print how many seeds would be processed, call no LLM
    python -m tests.eval.experiments.generate_queries --dry-run

    # Generate for the first 3 memories (3 LLM calls), merge into the file
    python -m tests.eval.experiments.generate_queries --limit 3

    # Narrow to specific seeds
    python -m tests.eval.experiments.generate_queries --seed seed-001,seed-007

    # Re-generate one seed, replacing its old candidates
    python -m tests.eval.experiments.generate_queries --seed seed-001 --force
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from tests.eval.dataset import SeedMemory, load_seed_memories
from tests.eval.ground_truth import DIFFICULTIES

DEFAULT_OUT = Path(__file__).parent / "query_candidates.jsonl"

# A generator turns one seed (+ its peers) into a raw LLM response dict.
# Injectable so unit tests run the parse/merge/validation logic without the
# real LLM (mirrors the runner's executor-injection pattern in llm_runner).
QueryGenerator = Callable[[SeedMemory, Sequence[SeedMemory]], Awaitable[dict[str, Any]]]


# ── Prompt ───────────────────────────────────────────────────────
# Structured-output note: the schema is passed to chat_structured for
# validation/retry, so the prompt only has to describe the task — it does
# not embed the JSON Schema (same convention as extraction prompts).

_QUERY_GEN_PROMPT = """\
你是一名信息检索评测数据生成器，为 EMA 工程记忆系统生成检索查询。

知识库中有一条记忆 A：

[记忆 {seed_id}]（类别：{category}）
摘要：{summary}
详细：{content}

语料中还有以下其他记忆（编号 + 摘要，供选择混淆目标）：

{peers_text}

请为记忆 A 生成：

1. 三条"正例查询"（positive_queries），对应三个难度：
   - easy：与摘要表面词重合，用户直接使用原文中的词汇提问；
   - medium：改写或用同义表达，词面部分重合；
   - hard：概念式提问（如"为什么…"），与摘要几乎没有词面重合，只能靠语义召回。
   每条查询要像真实开发者或新人的提问，口语化、自然，不要照抄摘要句子。

2. 一条"hard negative"：
   从上面的其他记忆中选一条与 A 主题最接近、检索器最容易把两者混淆的记忆 B。
   生成一条查询 Q：用户的真实意图是查找记忆 B，但 Q 的措辞（关键词、问法）
   会让检索器把 A 排在 B 前面。必须说明选择 B 的理由。

只输出 JSON，不要输出任何其他文字：
{{
  "positive_queries": [
    {{"query": "...", "difficulty": "easy"}},
    {{"query": "...", "difficulty": "medium"}},
    {{"query": "...", "difficulty": "hard"}}
  ],
  "hard_negative": {{
    "query": "...",
    "target_memory_id": "seed-XXX",
    "reason": "为什么选 B、为什么这个查询会误命中 A"
  }}
}}"""


# ── LLM output schema (validated + retried by chat_structured) ─────

_QUERY_GEN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["positive_queries", "hard_negative"],
    "properties": {
        "positive_queries": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": {
                "type": "object",
                "required": ["query", "difficulty"],
                "properties": {
                    "query": {"type": "string", "minLength": 3},
                    "difficulty": {
                        "type": "string",
                        "enum": list(DIFFICULTIES),
                    },
                },
            },
        },
        "hard_negative": {
            "type": "object",
            "required": ["query", "target_memory_id", "reason"],
            "properties": {
                "query": {"type": "string", "minLength": 3},
                "target_memory_id": {"type": "string", "minLength": 1},
                "reason": {"type": "string", "minLength": 1},
            },
        },
    },
}

# Caps keep the prompt inside the compaction-style budget: peers are the
# dominant cost (29 × summary) and content is context, not signal.
_PEER_SUMMARY_CAP = 150
_SEED_CONTENT_CAP = 400


def _build_prompt(seed: SeedMemory, peers: Sequence[SeedMemory]) -> str:
    """Assemble the QG prompt for *seed* with its *peers* as context."""
    peer_lines: list[str] = []
    for p in peers:
        summary = p.summary.strip().replace("\n", " ")
        if len(summary) > _PEER_SUMMARY_CAP:
            summary = summary[: _PEER_SUMMARY_CAP] + "…"
        peer_lines.append(f"[{p.id}] {summary}")
    peers_text = "\n".join(peer_lines) if peer_lines else "（无其他记忆）"

    content = (seed.content or seed.summary).strip().replace("\n", " ")
    if len(content) > _SEED_CONTENT_CAP:
        content = content[: _SEED_CONTENT_CAP] + "…"

    return _QUERY_GEN_PROMPT.format(
        seed_id=seed.id,
        category=seed.category,
        summary=seed.summary.strip(),
        content=content,
        peers_text=peers_text,
    )


def make_default_generator() -> QueryGenerator:
    """Executor that calls the production structured-output path."""

    async def _generate(seed: SeedMemory, peers: Sequence[SeedMemory]) -> dict[str, Any]:
        from backend.service.structured import chat_structured

        return await chat_structured(
            [{"role": "user", "content": _build_prompt(seed, peers)}],
            json_schema=_QUERY_GEN_SCHEMA,
            scenario="eval_query_gen",
            # Query generation benefits from variety, not determinism —
            # override the low structured temperature.
            temperature=0.7,
        )

    return _generate


# ── Parse LLM output → candidate rows ───────────────────────────
# The raw LLM dict is *not* trusted: hard-negative targets are cross-checked
# against the actual peer set (a hallucinated memory id is dropped, never
# silently written into the candidates file).


def parse_generation(
    seed: SeedMemory,
    peers: Sequence[SeedMemory],
    data: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Turn one LLM response into candidate rows + human-readable warnings.

    Returns ``(candidates, warnings)``.  ``candidates`` may be empty when
    every piece of the response fails validation (the caller records the
    warnings instead of writing junk rows).
    """
    peer_by_id: dict[str, SeedMemory] = {p.id: p for p in peers}
    candidates: list[dict[str, Any]] = []
    warnings: list[str] = []

    for item in data.get("positive_queries") or []:
        query = str(item.get("query", "")).strip()
        difficulty = str(item.get("difficulty", ""))
        if not query:
            warnings.append(f"{seed.id}: empty positive query dropped")
            continue
        if difficulty not in DIFFICULTIES:
            warnings.append(
                f"{seed.id}: unknown difficulty {difficulty!r}, defaulting to medium"
            )
            difficulty = "medium"
        candidates.append(
            {
                "id": f"qg-{seed.id}-{difficulty}",
                "source_seed_id": seed.id,
                "seed_ids": [seed.id],
                "query": query,
                "category": seed.category,
                "difficulty": difficulty,
                "kind": "positive",
                "status": "candidate",
                "review": None,
            }
        )

    hn = data.get("hard_negative") or {}
    query = str(hn.get("query", "")).strip()
    target = str(hn.get("target_memory_id", "")).strip()
    reason = str(hn.get("reason", "")).strip()
    if not query or not target:
        warnings.append(f"{seed.id}: hard negative missing query or target — skipped")
    elif target not in peer_by_id:
        warnings.append(
            f"{seed.id}: hard negative targets unknown memory {target!r} — skipped"
        )
    else:
        target_seed = peer_by_id[target]
        candidates.append(
            {
                "id": f"qg-{seed.id}-hardneg",
                "source_seed_id": seed.id,
                # seed_ids = the query's TRUE intent (memory B); the
                # distractor is the memory it risks outranking (A).
                "seed_ids": [target],
                "distractor_seed_ids": [seed.id],
                "query": query,
                "category": target_seed.category,
                "difficulty": "hard",
                "kind": "hard_negative",
                "reason": reason,
                "status": "candidate",
                "review": None,
            }
        )

    return candidates, warnings


# ── Output file merge (idempotent resume) ────────────────────────


def _load_rows(path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows, silently skipping malformed lines."""
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _processed_seed_ids(path: Path) -> set[str]:
    """Seed ids that already have candidates in *path* (for resume skip)."""
    return {str(r.get("source_seed_id", "")) for r in _load_rows(path) if r.get("source_seed_id")}


def _merge_candidates(path: Path, new_entries: list[dict[str, Any]]) -> int:
    """Rewrite *path*: keep old rows, replace rows for re-generated seeds.

    Old candidates for any seed in *new_entries* are dropped (``--force``
    re-generation) and the new rows appended; every other seed's rows are
    preserved.  Returns the total row count after the merge.
    """
    fresh_seeds = {str(e["source_seed_id"]) for e in new_entries}
    kept = [
        r for r in _load_rows(path) if str(r.get("source_seed_id", "")) not in fresh_seeds
    ]
    all_rows = kept + new_entries
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in all_rows) + "\n"
    path.write_text(text, encoding="utf-8")
    return len(all_rows)


# ── Orchestration ────────────────────────────────────────────────


async def generate_candidates(
    seeds: Sequence[SeedMemory],
    *,
    out_path: Path,
    force: bool = False,
    generator: QueryGenerator | None = None,
    corpus: Sequence[SeedMemory] | None = None,
) -> tuple[int, list[str]]:
    """Generate candidates for *seeds*, merging into *out_path*.

    ``seeds`` is the subset to process; ``corpus`` is the FULL memory set
    used to compute each seed's peers.  They are distinct: a ``--limit`` /
    ``--seed`` run processes a subset but the LLM must still pick its
    hard-negative target from the whole corpus — otherwise a lone seed gets
    an empty peer list and the model invents a target (which validation
    correctly drops).

    Skips seeds already processed (unless ``force``).  A per-seed LLM
    failure is logged and skipped — one bad memory must not abort the batch.
    Returns ``(total_new_candidates, warnings)``.
    """
    gen = generator or make_default_generator()
    peer_pool = list(corpus) if corpus is not None else list(seeds)
    processed = set() if force else _processed_seed_ids(out_path)
    warnings: list[str] = []
    new_entries: list[dict[str, Any]] = []

    for seed in seeds:
        if seed.id in processed:
            print(f"  skip {seed.id} (already in {out_path.name})", file=sys.stderr)
            continue
        peers = [s for s in peer_pool if s.id != seed.id]
        try:
            data = await gen(seed, peers)
        except Exception as exc:
            warnings.append(f"{seed.id}: generation failed ({exc}) — skipped")
            print(f"  ✗ {seed.id}: generation failed", file=sys.stderr)
            continue
        candidates, seed_warnings = parse_generation(seed, peers, data)
        warnings.extend(seed_warnings)
        if candidates:
            new_entries.extend(candidates)
            kinds = ",".join("p" if c["kind"] == "positive" else "h" for c in candidates)
            print(f"  ✓ {seed.id}: {len(candidates)} candidates ({kinds})", file=sys.stderr)
        else:
            print(f"  ✗ {seed.id}: no usable candidates", file=sys.stderr)

    if new_entries:
        total = _merge_candidates(out_path, new_entries)
        print(f"✓ wrote {len(new_entries)} new candidates → {out_path} ({total} total rows)", file=sys.stderr)
    else:
        print("No new candidates generated.", file=sys.stderr)

    return len(new_entries), warnings


# ── CLI ───────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.eval.experiments.generate_queries",
        description="Generate retrieval-eval query candidates from the seed "
        "corpus (3 positives + 1 hard negative per memory) for human review.",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N seeds.",
    )
    p.add_argument(
        "--offset", type=int, default=0,
        help="Skip the first N seeds (resume after a partial run).",
    )
    p.add_argument(
        "--seed", default=None,
        help="Comma-separated seed ids to process (overrides --limit/--offset).",
    )
    p.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help=f"Output candidates file (JSONL). Default: {DEFAULT_OUT}.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-generate seeds already present in the output file, replacing "
        "their old candidates.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan (seeds to process, LLM calls it would make) "
        "without calling the LLM.",
    )
    return p


def _select_seeds(seeds: Sequence[SeedMemory], args: argparse.Namespace) -> list[SeedMemory]:
    if args.seed:
        wanted = {s.strip() for s in args.seed.split(",") if s.strip()}
        selected = [s for s in seeds if s.id in wanted]
        missing = wanted - {s.id for s in selected}
        if missing:
            print(f"⚠ unknown seed ids: {sorted(missing)}", file=sys.stderr)
        return selected
    stop = args.offset + args.limit if args.limit is not None else None
    return list(seeds[args.offset:stop])


async def _run(args: argparse.Namespace) -> int:
    seeds = load_seed_memories()
    selected = _select_seeds(seeds, args)
    out_path = Path(args.out)

    if args.dry_run:
        print(f"Loaded {len(seeds)} seed memories; would process {len(selected)}:", file=sys.stderr)
        for s in selected:
            print(f"  [{s.category}] {s.id}: {s.summary[:60]}…", file=sys.stderr)
        print(f"Each memory costs 1 LLM call (→ {len(selected)} calls total).", file=sys.stderr)
        return 0

    print(f"Generating queries for {len(selected)} memories → {out_path} …", file=sys.stderr)
    n_new, warnings = await generate_candidates(
        selected, out_path=out_path, force=args.force, corpus=seeds
    )
    for w in warnings:
        print(f"  ⚠ {w}", file=sys.stderr)
    print(f"Done: {n_new} candidates generated.", file=sys.stderr)
    return 0 if n_new or not selected else 1


def main() -> None:
    import asyncio

    args = _build_parser().parse_args()
    rc = asyncio.run(_run(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
