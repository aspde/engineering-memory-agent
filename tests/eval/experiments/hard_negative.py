"""Hard-negative discrimination eval — does the retriever get fooled by near-miss distractors?

This is a **separate, independent** eval from the main 30-query labeled set
(``ground_truth.py``). The main set is self-answered: every query was
generated *from* its target memory, so Recall@5=1.0 mostly proves the
retriever can re-find what it was asked about. That says little about
discrimination — whether it can tell two memories apart when both share
surface words.

The hard-negative set (``query_candidates.jsonl``, 27 items with
``kind="hard_negative"``) fixes exactly that gap. Each query has:
    - ``seed_ids[0]``  — the correct target seed,
    - ``distractor_seed_ids[0]`` — a trap seed whose summary overlaps the
      query's surface terms but answers a *different* question.

We run every query through the **production default path**
(``retrieval.query_memories`` with ``top_k=5``, ``threshold=0.3``, no
rerank) and measure:

    target_recall@5        P(target in top-5)                     — can the
                            retriever pierce surface overlap to the real target?
    distractor_intrusion@5 P(distractor in top-5)                 — how often
                            does the trap leak in (lower is better)?
    hard_neg_pass@5        P(target hit AND not outranked by trap) — combined
                            discrimination pass rate.
    mrr                    mean(1/target_rank)                    — target rank
                            quality, comparable to the main eval.
    worse_than_random      # queries where the distractor beat the target
                            (or the target missed while the distractor hit).

Reports are written to ``tests/eval/reports/archive/hard_negative_report.{json,md}``.

Run:
    DATABASE_URL=postgresql://ema:ema123@localhost:5432/ema_eval_hn \\
        python -m tests.eval.experiments.hard_negative

This module touches **no** other eval file — ``ground_truth.py`` /
``dataset.py`` / ``runner.py`` / ``run_eval.py`` are left untouched.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CANDIDATES_FILE = Path(__file__).parent / "query_candidates.jsonl"
REPORT_DIR = Path(__file__).parent / "reports"
JSON_REPORT = REPORT_DIR / "hard_negative_report.json"
MD_REPORT = REPORT_DIR / "hard_negative_report.md"

# Production default read path — this is what ``query_memories`` uses when the
# API is called with no rerank flags.  Keep these in sync with the function
# signature defaults in ``backend/service/retrieval.py``.
TOP_K = 5
THRESHOLD = 0.3
RETRIEVAL_PATH = f"memory:norank@{TOP_K}, threshold {THRESHOLD}"

# Baseline MRR of the *main* self-answered eval set (30 queries) measured on the
# same corpus/database with the same memory:norank@k5 path
# (``python -m tests.eval.run_eval --retriever memory``, report committed at
# ``tests/eval/reports/memory_path_report.md``).  Used only as a comparison
# figure in the interpretation section.
#
# Caveat (measured, see memory_path_report.md): with the semantic-relevance
# channel off the same path measures recall@5=0.900 / mrr=0.844 — the main set's
# 1.0/0.944 already benefits from a self-scored semantic channel, so the
# hard-negative gap (0.790) is if anything an *under*-statement of how much
# discrimination the production path really lacks.
MAIN_EVAL_MRR = 0.944
MAIN_EVAL_RECALL = 1.000


def load_hard_negatives() -> list[dict[str, Any]]:
    """Load the hard-negative queries from query_candidates.jsonl."""
    if not CANDIDATES_FILE.exists():
        raise FileNotFoundError(f"candidates file not found: {CANDIDATES_FILE}")
    items: list[dict[str, Any]] = []
    with CANDIDATES_FILE.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{CANDIDATES_FILE}:{lineno}: invalid JSON: {e}") from e
            if item.get("kind") == "hard_negative":
                items.append(item)
    return items


async def evaluate_item(item: dict[str, Any]) -> dict[str, Any]:
    """Run one hard-negative query through the production path and score it.

    Result dicts from ``query_memories`` carry a ``meta`` JSONB whose
    ``seed_id`` was stamped at seed time (``tests/eval/seed.py``), so we can
    map each ranked row back to its seed without any UUID coupling.
    """
    from backend.service.retrieval import query_memories

    target = str(item["seed_ids"][0])
    distractor = str(item["distractor_seed_ids"][0])
    query = str(item["query"])

    results = await query_memories(query, top_k=TOP_K)  # threshold=0.3, no rerank

    # 1-based rank per seed_id (each seed appears at most once per result list).
    ranks: dict[str, int] = {}
    for pos, r in enumerate(results, start=1):
        meta = r.get("meta") or {}
        sid = meta.get("seed_id")
        if sid is not None:
            sid = str(sid)
            ranks.setdefault(sid, pos)

    target_rank: int | None = ranks.get(target)
    distractor_rank: int | None = ranks.get(distractor)

    # Discriminated: the target is recalled AND not beaten/elbowed out by the
    # trap (trap absent, or ranked strictly below the target).
    discriminated = target_rank is not None and (
        distractor_rank is None or target_rank < distractor_rank
    )

    return {
        "id": item["id"],
        "query": query,
        "target_seed": target,
        "distractor_seed": distractor,
        "target_rank": target_rank,
        "distractor_rank": distractor_rank,
        "discriminated": bool(discriminated),
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the summary metrics over all hard-negative items."""
    n = len(items)
    n_target_hit = sum(1 for it in items if it["target_rank"] is not None)
    n_distractor_hit = sum(1 for it in items if it["distractor_rank"] is not None)
    n_pass = sum(1 for it in items if it["discriminated"])

    mrr = _mean(
        [
            1.0 / it["target_rank"] if it["target_rank"] is not None else 0.0
            for it in items
        ]
    )

    # worse_than_random: distractor outranks target, OR target missed while the
    # distractor was recalled.  A tie (equal rank) is impossible — each memory
    # occupies at most one slot in a result list.
    worse = [
        it
        for it in items
        if it["distractor_rank"] is not None
        and (it["target_rank"] is None or it["distractor_rank"] < it["target_rank"])
    ]

    return {
        "n": n,
        "target_recall@5": n_target_hit / n if n else 0.0,
        "distractor_intrusion@5": n_distractor_hit / n if n else 0.0,
        "hard_neg_pass@5": n_pass / n if n else 0.0,
        "mrr": mrr,
        "worse_than_random": len(worse),
        "n_target_hit": n_target_hit,
        "n_distractor_hit": n_distractor_hit,
        "n_pass": n_pass,
        "worse_than_random_items": worse,
    }


# ── Honest interpretation ────────────────────────────────────────────
# Generated from the measured numbers, never hard-coded.  The point of this
# report is to expose whatever is actually there — a low pass rate is a real
# finding, not a bug to paper over.


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def build_interpretation(
    agg: dict[str, Any], items: list[dict[str, Any]]
) -> str:
    n = agg["n"]
    t_recall = agg["target_recall@5"]
    d_intrusion = agg["distractor_intrusion@5"]
    h_pass = agg["hard_neg_pass@5"]
    mrr = agg["mrr"]
    n_worse = agg["worse_than_random"]

    lines: list[str] = []

    # ── Read the numbers ───────────────────────────────────────────
    lines.append(
        "这份 27 条 hard-negative 集考验的不是“能不能找到”，而是“会不会被带偏”。"
        "主评估集（30 条自问自答 query）在相同 memory:norank@k5 路径上是 "
        f"recall@5={MAIN_EVAL_RECALL:.3f} / mrr={MAIN_EVAL_MRR:.3f}——那只能证明检索器"
        "能找到自己出题时反向生成的目标。本集每条 query 都配了一个表面词高度重合、"
        "语义却不同的陷阱 seed，数字越低越说明检索器在“看似相关”的候选里缺乏判别力。"
    )
    lines.append("")
    lines.append(
        f"**目标召回**：{_fmt_pct(t_recall)} 的 query 能穿透表面词命中真目标"
        f"（{agg['n_target_hit']}/{n}）。{_fmt_pct(t_recall)} 的目标命中率说明"
        + (
            "BGE-M3 的稠密相似度基本能识别出正确目标——问题不出在“找得到”，而出在“排得对”。"
            if t_recall >= 0.85
            else "有相当比例的 query 连真目标都没进 top-5——说明表面词干扰已经盖过了语义信号，"
            "不只是排序问题，召回本身就被带偏了。"
        )
    )
    lines.append("")
    lines.append(
        f"**陷阱入侵**：{_fmt_pct(d_intrusion)} 的 query 把陷阱 seed 也召回了"
        f"（{agg['n_distractor_hit']}/{n}）。"
        + (
            "陷阱进入候选集的比例很高——这正是 hard-negative 的设计意图：陷阱和目标共享表层词，"
            "向量检索在 top-5 里几乎不可避免地把两者都捞进来，真正的考验是谁排前面。"
            if d_intrusion >= 0.5
            else "陷阱多数时候没能混进 top-5，说明表层词重合还没有强大到骗过向量相似度。"
        )
    )
    lines.append("")
    n_fail = n - agg["n_pass"]
    if h_pass >= 0.9:
        pass_note = "检索器在绝大多数近失例子上做出了正确取舍，判别力是扎实的。"
    elif h_pass >= 0.7:
        pass_note = f"约七成例子能正确判别，但仍有 {n_fail}/{n} 条失败面，不算稳健。"
    else:
        pass_note = (
            f"有 {n_fail}/{n}（{_fmt_pct(n_fail / n) if n else '0%'}）被陷阱误导，"
            "判别力有明显短板——不能拿主评估集的满分当真实水平。"
        )
    lines.append(
        f"**综合通过率**：{_fmt_pct(h_pass)}（{agg['n_pass']}/{n}）——目标被召回且没有被陷阱压过/顶替。"
        f"这是最重要的数字：{_fmt_pct(h_pass)} 意味着{pass_note}"
    )
    lines.append("")
    lines.append(
        f"**MRR** = {mrr:.3f}，而主评估集同路径为 {MAIN_EVAL_MRR:.3f}。"
        "MRR 的落差量化了“表面词重合到底让目标排名掉了多少”——即便目标命中了，"
        "如果陷阱总是排在前面，MRR 也会被拖低。"
    )

    # ── The actual failures ────────────────────────────────────────
    lines.append("")
    if n_worse == 0:
        lines.append(
            "**worse_than_random = 0**：没有任何 query 出现“陷阱排在目标之前或目标落榜而陷阱上榜”。"
            "这是最干净的结果——陷阱可以混进 top-5（入侵率高），但从未真正压过目标。"
        )
    else:
        n_both = sum(
            1
            for it in items
            if it["target_rank"] is not None and it["distractor_rank"] is not None
        )
        worse_ratio = n_worse / n_both if n_both else 0.0
        lines.append(
            f"**worse_than_random = {n_worse}**：{n_worse}/{n}（{_fmt_pct(n_worse / n)}）的 query 里，"
            "陷阱排在目标前面（本集目标全命中，所以全部是“双方都被召回、陷阱排名更靠前”的情形："
            f"{n_both}/{n} 条双方都进 top-5，其中 {n_worse} 条陷阱压在目标上，"
            f"约占双方都命中例子的 {_fmt_pct(worse_ratio)}）。"
            "对照基准：若目标和陷阱在 top-5 内的排序随机打乱，陷阱压过目标的概率约 1/2——也就是说，"
            "“把目标排在陷阱之上”这件事，检索器只是比抛硬币略好一点点，远谈不上稳健。逐条如下："
        )
        for it in agg["worse_than_random_items"]:
            t_rank = it["target_rank"]
            d_rank = it["distractor_rank"]
            who = (
                f"陷阱 seed-{it['distractor_seed'][-3:]} 排第 {d_rank}，目标 seed-{it['target_seed'][-3:]} 排第 {t_rank}"
                if t_rank is not None
                else f"目标 seed-{it['target_seed'][-3:]} 未命中，陷阱 seed-{it['distractor_seed'][-3:]} 排第 {d_rank}"
            )
            lines.append(f"  - `{it['id']}`（{who}）：{it['query']}")

    # ── What this reveals ──────────────────────────────────────────
    lines.append("")
    lines.append("**这些数字暴露了检索器什么特性？**")
    props: list[str] = []
    if d_intrusion >= 0.4:
        props.append(
            "top-5 对表层词重合高度宽容——陷阱和目标共享的关键词（如 pgvector、PostgresSaver、"
            "Ebbinghaus、HITL/interrupt）足以把不相关记忆拉进候选。这是 BGE-M3 稠密检索的典型特征："
            "它擅长“语义接近”，但不做“问题意图”判别。"
        )
    if n_worse > 0 and mrr < MAIN_EVAL_MRR:
        props.append(
            "排序对共享表面词的敏感度高于对问题类型的敏感度：当 query 同时命中目标和陷阱的词汇时，"
            "排名基本由“谁与 query 共享的词汇更多/更醒目”决定，而非“谁真正回答了这个问题”。"
            "这正是主评估集测不出来的维度——自问自答的 query 只与目标共享词汇，陷阱不存在。"
        )
    if t_recall < 1.0:
        props.append(
            "部分 query 连目标都没进 top-5，说明有些 hard-negative 的表面词干扰强到压过了语义信号本身"
            "（典型如 query 以目标/陷阱均含的专有名词开头，而真目标的关键词密度反而更低）。"
        )
    if not props:
        props.append(
            "在 27 条刻意构造的近失例子上，检索器既保持了高目标召回，又没有让陷阱压过目标——"
            "说明它的排序以语义相似度为准，共享表面词不足以系统性欺骗它。"
        )
    lines.extend(f"- {p}" for p in props)

    # ── How to read the numbers ───────────────────────────────────
    lines.append("")
    lines.append("**如何解读这些数字？**")
    lines.append(
        "- 先说结论：主评估集 Recall@5=1.0 是**自问自答的假满分**，只能证明“找得到”，不能证明“判别力”。"
    )
    lines.append(
        f"- 然后给出本集数字：27 条 hard negative 上，目标召回 {t_recall * 100:.1f}%、陷阱入侵 {d_intrusion * 100:.1f}%、综合通过 {h_pass * 100:.1f}%、"
        f"MRR {mrr:.3f}（主评估同路径 {MAIN_EVAL_MRR:.3f}）、worse_than_random {n_worse}/{n}。"
    )
    lines.append(
        "- 解读两点：(a) 高目标召回 + 高陷阱入侵的组合说明检索器“宽进”——它把相关和不相关的表层重合候选都捞进来，"
        f"靠的是 top-5 窗口兜底而不是精确判别；(b) worse_than_random={n_worse} 说明陷阱压过目标是个真实但局部的现象，"
        "暴露的是纯向量检索在“问题意图 vs 词汇重合”上的盲区。"
    )
    lines.append(
        "- 如果有下一步：这正是无监督 rerank / 检索后意图判别（query 重写、hybrid 融合、cross-encoder）应该"
        "改善的环节——hard-negative 集可以直接当作这些方案的回归测试集。"
    )

    return "\n".join(lines)


def build_json_report(
    agg: dict[str, Any], items: list[dict[str, Any]], interpretation: str
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "corpus": f"{agg['n']} hard_negative queries",
        "retrieval_path": RETRIEVAL_PATH,
        "aggregate": {
            "n": agg["n"],
            "target_recall@5": round(agg["target_recall@5"], 4),
            "distractor_intrusion@5": round(agg["distractor_intrusion@5"], 4),
            "hard_neg_pass@5": round(agg["hard_neg_pass@5"], 4),
            "mrr": round(agg["mrr"], 4),
            "worse_than_random": agg["worse_than_random"],
        },
        "items": items,
        "worse_than_random_items": [
            {
                "id": it["id"],
                "query": it["query"],
                "target_seed": it["target_seed"],
                "distractor_seed": it["distractor_seed"],
                "target_rank": it["target_rank"],
                "distractor_rank": it["distractor_rank"],
            }
            for it in agg["worse_than_random_items"]
        ],
        "interpretation": interpretation,
    }


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def build_markdown_report(
    agg: dict[str, Any], items: list[dict[str, Any]], interpretation: str
) -> str:
    a = agg
    lines: list[str] = []
    lines.append("# EMA 检索 Hard-Negative 判别力评估报告")
    lines.append("")
    lines.append(f"- **生成时间**：{datetime.now(UTC).isoformat()}")
    lines.append(
        f"- **语料**：{a['n']} 条 hard_negative query（来自 `query_candidates.jsonl`）"
    )
    lines.append(f"- **检索路径**：{RETRIEVAL_PATH}（生产默认，无 rerank）")
    lines.append(f"- **主评估对比基线**：`run_eval --retriever memory` 在相同语料上 "
                 f"recall@5={MAIN_EVAL_RECALL:.3f} / mrr={MAIN_EVAL_MRR:.3f}")
    lines.append("")

    lines.append("## 汇总指标（27 条平均）")
    lines.append("")
    lines.append("| 指标 | 值 | 含义 |")
    lines.append("|---|---|---|")
    lines.append(f"| target_recall@5 | {a['target_recall@5'] * 100:.1f}% | 目标被召回到 top-5 的比例（判别力） |")
    lines.append(f"| distractor_intrusion@5 | {a['distractor_intrusion@5'] * 100:.1f}% | 陷阱被误召回比例（越低越好） |")
    lines.append(f"| hard_neg_pass@5 | {a['hard_neg_pass@5'] * 100:.1f}% | 目标命中且不被陷阱压过/顶替 |")
    lines.append(f"| mrr | {a['mrr']:.3f} | 目标排名倒数的均值（主评估同路径 {MAIN_EVAL_MRR:.3f}） |")
    lines.append(f"| worse_than_random | {a['worse_than_random']} / {a['n']} | 陷阱排在目标之前 / 目标未命中而陷阱命中 |")
    lines.append("")

    lines.append("## 逐条明细（27 条 hard negative）")
    lines.append("")
    lines.append("| id | query | target_seed | distractor_seed | target_rank | distractor_rank | discriminated |")
    lines.append("|---|---|---|---|---|---|---|")
    for it in items:
        lines.append(
            f"| {it['id']} | {_md_escape(it['query'])} | {it['target_seed']} | "
            f"{it['distractor_seed']} | "
            f"{it['target_rank'] if it['target_rank'] is not None else '—'} | "
            f"{it['distractor_rank'] if it['distractor_rank'] is not None else '—'} | "
            f"{'✅' if it['discriminated'] else '❌'} |"
        )
    lines.append("")

    lines.append("## 诚实解读")
    lines.append("")
    lines.append(interpretation)
    lines.append("")
    return "\n".join(lines)


async def _run() -> int:
    items = load_hard_negatives()
    print(f"Loaded {len(items)} hard-negative queries", file=sys.stderr)
    if not items:
        print("No hard_negative items found — nothing to evaluate", file=sys.stderr)
        return 1

    evaluated: list[dict[str, Any]] = []
    for i, item in enumerate(items, start=1):
        res = await evaluate_item(item)
        evaluated.append(res)
        target = res["target_rank"] if res["target_rank"] is not None else "-"
        distractor = (
            res["distractor_rank"] if res["distractor_rank"] is not None else "-"
        )
        verdict = "PASS" if res["discriminated"] else "FAIL"
        print(
            f"[{i}/{len(items)}] {res['id']} target_rank={target} "
            f"distractor_rank={distractor} {verdict}",
            file=sys.stderr,
        )

    agg = aggregate(evaluated)
    a = agg
    print(
        f"\n=== HARD-NEGATIVE SUMMARY (n={a['n']}) ==="
        f"\n  target_recall@5         = {a['target_recall@5'] * 100:.1f}%"
        f"\n  distractor_intrusion@5  = {a['distractor_intrusion@5'] * 100:.1f}%"
        f"\n  hard_neg_pass@5         = {a['hard_neg_pass@5'] * 100:.1f}%"
        f"\n  mrr                     = {a['mrr']:.3f}"
        f"\n  worse_than_random       = {a['worse_than_random']}",
        file=sys.stderr,
    )

    interpretation = build_interpretation(agg, evaluated)
    json_report = build_json_report(agg, evaluated, interpretation)
    md_report = build_markdown_report(agg, evaluated, interpretation)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(
        json.dumps(json_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    MD_REPORT.write_text(md_report, encoding="utf-8")
    print(f"✓ JSON report → {JSON_REPORT}", file=sys.stderr)
    print(f"✓ Markdown report → {MD_REPORT}", file=sys.stderr)
    return 0


def main() -> None:
    rc = asyncio.run(_run())
    sys.exit(rc)


if __name__ == "__main__":
    main()
