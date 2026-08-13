"""四级相似度阈值标定 — 收集真实摘要对的相似度分布。

The four-level dedup/conflict thresholds in ``backend/service/memory.py``
(MERGE 0.85 / CONFLICT 0.72 / SUPPLEMENT 0.60 / below → new) were set by
calibration against this script's data plus judgement.  This script gathers
similarity data for the three shapes of summary-pair the thresholds must
separate:

- **duplicate pairs** (should MERGE): a seed summary plus an LLM
  paraphrase of it — simulates the same knowledge extracted by different
  sources.
- **same-category pairs** (should NOT merge, often supplement/conflict):
  different seeds from the same category — related but distinct memories.
- **cross-category pairs** (should be NEW): seeds from different
  categories — unrelated knowledge.

It embeds all pairs with the production BGE-M3 provider, reports per-group
quantiles, and evaluates how each threshold would classify the pairs.  The
output is a **first calibration pass on the seed corpus**, not production
ground truth — real dedup pairs must be collected from production to
confirm (the report says so).  The thresholds are the production values
(0.85 / 0.72 / 0.60); an earlier calibration pass under the old constants
(0.92 / 0.75 / 0.60) motivated the change to 0.85 — re-running this script
regenerates ``threshold_calibration_report.md`` under the current
classification.

Usage:
    python -m tests.eval.experiments.threshold_calibration --report-md tests/eval/reports/archive/threshold_calibration_report.md
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import sys
from datetime import datetime
from pathlib import Path

from tests.eval.dataset import load_seed_memories

MERGE, CONFLICT, SUPPLEMENT = 0.85, 0.72, 0.60

_N_PARAPHRASES = 8  # seeds to LLM-paraphrase into duplicate pairs
_RANDOM_SEED = 42   # deterministic cross-category sampling


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


async def _paraphrase_summaries(summaries: list[str]) -> list[str | None]:
    """LLM-paraphrase each summary into a same-meaning variant.

    Returns ``None`` for a seed whose paraphrase failed — an LLM error, a
    response too short to be a real paraphrase, or the model returning the
    original verbatim.  The caller drops such pairs and reports the drop
    count.

    A failed paraphrase must NOT fall back to the original: the original vs
    its own embedding scores cosine 1.0, a fake "perfect duplicate" that
    inflates the merge band and skews the very distribution this calibration
    is measuring.
    """
    from backend.service.llm_service import get_llm_provider

    llm = get_llm_provider()
    out: list[str | None] = []
    for s in summaries:
        try:
            resp = await llm.chat(
                [{
                    "role": "user",
                    "content": (
                        "Paraphrase the following technical summary into a "
                        "different wording that states the SAME facts (the "
                        "way another engineer might write it).  Keep the key "
                        "terms and entities.  Output ONLY the paraphrase, no "
                        "labels.\n\nSummary: "
                        + s
                    ),
                }],
                scenario="eval_threshold_calibration",
                temperature=0.4,
            )
            text = str(resp).strip()
            out.append(text if len(text) > 10 and text != s else None)
        except Exception:
            out.append(None)
    return out


def _quantiles(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"min": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "max": 0.0}
    s = sorted(vals)
    n = len(s)

    def pct(p: float) -> float:
        return s[min(n - 1, int(p * n))]

    return {"min": s[0], "p25": pct(0.25), "p50": pct(0.50),
            "p75": pct(0.75), "max": s[-1]}


def _band(value: float) -> str:
    if value >= MERGE:
        return "merge(≥0.85)"
    if value >= CONFLICT:
        return "conflict(0.72-0.85)"
    if value >= SUPPLEMENT:
        return "supplement(0.60-0.72)"
    return "new(<0.60)"


_BAND_ORDER = ("merge(≥0.85)", "conflict(0.72-0.85)", "supplement(0.60-0.72)", "new(<0.60)")


def _band_counts(vals: list[float]) -> list[int]:
    """Per-band counts for *vals*, in ``_BAND_ORDER`` order."""
    counts = {b: 0 for b in _BAND_ORDER}
    for v in vals:
        counts[_band(v)] += 1
    return [counts[b] for b in _BAND_ORDER]


def _build_conclusion(groups: dict[str, list[float]]) -> list[str]:
    """Generate the conclusion section from this run's measured distributions.

    Every line is derived from the actual numbers in *groups* — never
    hard-coded — so a re-run with more (or different) data produces an
    up-to-date conclusion instead of a stale snapshot.
    """
    names = list(groups)
    dup = groups.get(names[0], []) if names else []
    same = groups.get(names[1], []) if len(names) > 1 else []
    cross = groups.get(names[2], []) if len(names) > 2 else []

    lines: list[str] = []
    if dup:
        n_merge, n_conflict = _band_counts(dup)[:2]
        lines.append(
            f"- **merge 阈值合理性**：{n_merge}/{len(dup)}（{n_merge / len(dup):.0%}）的同义改写对"
            f"落在 merge 带（≥0.85）"
            + (
                "，全部正确合并，对改写对分离良好。"
                if n_conflict == 0
                else f"，另有 {n_conflict}/{len(dup)} 落入 conflict 带（0.72-0.85）——会走冲突检测/补充"
                "而非合并（内容不丢，但多一次裁决或补充处理）。"
            )
        )
        dup_min = min(dup)
        same_max = max(same) if same else 0.0
        if same and dup_min < same_max:
            lines.append(
                f"- **分离边界**：同义改写对下限 {dup_min:.3f} 低于同类对上限 {same_max:.3f}，两类分布"
                "在单个 merge 阈值上无法完全分离——conflict 带（0.72-0.85）承担中间裁决，靠 HITL 兜底"
                "而非再调阈值。"
            )
        elif same:
            lines.append(
                f"- **分离边界**：同义改写对下限 {dup_min:.3f} 高于同类对上限 {same_max:.3f}，"
                "两组在 merge 阈值处分离良好。"
            )
        if len(dup) < 20:
            lines.append(
                f"- **样本量**：改写对仅 {len(dup)} 条，结论是初步方向；需生产收集真实"
                "「不同来源抽取同一记忆」的对后复标。"
            )
    else:
        lines.append("- **merge 阈值**：本次无可用同义改写对（全部改写失败被剔除），阈值无法评估，需重跑。")

    if cross:
        cross_new = _band_counts(cross)[3]
        if cross_new / len(cross) >= 0.9:
            lines.append(
                f"- **new 阈值**：{cross_new}/{len(cross)}（{cross_new / len(cross):.0%}）异类对落入 "
                "new 带（<0.60），<0.60 判定为新记忆基本可靠。"
            )
    return lines


async def _main(report_md: str | None) -> int:
    seeds = load_seed_memories()
    by_cat: dict[str, list] = {}
    for s in seeds:
        by_cat.setdefault(s.category, []).append(s)

    from backend.service.embedding_service import get_embedding_provider

    provider = get_embedding_provider()

    # 1. Duplicate pairs: LLM paraphrases of a sample of seeds.  Paraphrases
    # that failed (LLM error / degenerate verbatim return) are dropped before
    # embedding — a failed pair would score cosine 1.0 and fake-pollute the
    # merge band (see _paraphrase_summaries).
    sample = seeds[:_N_PARAPHRASES]
    paraphrases = await _paraphrase_summaries([s.summary for s in sample])
    pairs = [
        (orig, para) for orig, para in zip(sample, paraphrases)
        if para is not None
    ]
    n_failed = len(sample) - len(pairs)
    dup_pairs: list[float] = []
    if pairs:
        dup_vecs = await provider.embed(
            [s.summary for s, _ in pairs] + [para for _, para in pairs]
        )
        dup_pairs = [
            _cosine(dup_vecs[i], dup_vecs[len(pairs) + i])
            for i in range(len(pairs))
        ]

    # 2. Same-category pairs: all C(6,2)=15 per category.
    same_pairs: list[float] = []
    for cat, members in by_cat.items():
        vecs = await provider.embed([s.summary for s in members])
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                same_pairs.append(_cosine(vecs[i], vecs[j]))

    # 3. Cross-category pairs: deterministic sample.
    rng = random.Random(_RANDOM_SEED)
    cats = list(by_cat)
    cross_pairs: list[float] = []
    vec_cache: dict[str, list[float]] = {}
    seen_pairs: set[tuple[str, str]] = set()
    while len(cross_pairs) < 40:
        a_cat, b_cat = rng.sample(cats, 2)
        a = rng.choice(by_cat[a_cat])
        b = rng.choice(by_cat[b_cat])
        key = tuple(sorted((a.id, b.id)))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        if a.id not in vec_cache:
            vec_cache[a.id] = (await provider.embed([a.summary]))[0]
        if b.id not in vec_cache:
            vec_cache[b.id] = (await provider.embed([b.summary]))[0]
        cross_pairs.append(_cosine(vec_cache[a.id], vec_cache[b.id]))

    groups = {
        "duplicate (LLM 同义改写，应 merge)": dup_pairs,
        "same-category (同类不同 seed，应 独立/补充)": same_pairs,
        "cross-category (异类，应 new)": cross_pairs,
    }

    print("\n=== 四级阈值标定（seed 语料 + LLM 改写对，BGE-M3 cosine）===")
    print(f"  {'group':<42} {'min':>6} {'p25':>6} {'p50':>6} {'p75':>6} {'max':>6}")
    for name, vals in groups.items():
        q = _quantiles(vals)
        print(
            f"  {name:<42} {q['min']:>6.3f} {q['p25']:>6.3f} {q['p50']:>6.3f} "
            f"{q['p75']:>6.3f} {q['max']:>6.3f}"
        )

    print("\n=== 各阈值分组误判（band 分布）===")
    for name, vals in groups.items():
        bands = [_band(v) for v in vals]
        counts = {b: bands.count(b) for b in set(bands)}
        print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    if report_md:
        lines = [
            "# 四级相似度阈值标定",
            "",
            f"> {datetime.now():%Y-%m-%d} · 基于 seed 语料 + LLM 同义改写对，BGE-M3 cosine 相似度。"
            f"这是**初步标定**（{len(seeds)} 条 seed），非生产真实去重对；生产上需收集真实对确认。",
            "",
            "## 方法",
            "",
            "三类摘要对（`tests/eval/experiments/threshold_calibration.py`）：",
            "",
            f"- **duplicate**：{len(pairs)} 条可用 seed 的 LLM 同义改写（应 merge；"
            f"{n_failed} 条改写失败已剔除）",
            "- **same-category**：同类 15 对/类（应 独立或补充，不 merge）",
            "- **cross-category**：40 对随机异类（应 new）",
            "",
            "## 相似度分布",
            "",
            "| 组 | min | p25 | p50 | p75 | max |",
            "|----|-----|-----|-----|-----|-----|",
        ]
        for name, vals in groups.items():
            q = _quantiles(vals)
            lines.append(
                f"| {name} | {q['min']:.3f} | {q['p25']:.3f} | {q['p50']:.3f} "
                f"| {q['p75']:.3f} | {q['max']:.3f} |"
            )
        lines += [
            "",
            "## 各阈值 band 分布",
            "",
            "| 组 | merge(≥0.85) | conflict(0.72-0.85) | supplement(0.60-0.72) | new(<0.60) |",
            "|----|------|------|------|------|",
        ]
        band_order = _BAND_ORDER
        for name, vals in groups.items():
            counts = dict(zip(band_order, _band_counts(vals)))
            lines.append(
                f"| {name} | {counts[band_order[0]]} | {counts[band_order[1]]} | "
                f"{counts[band_order[2]]} | {counts[band_order[3]]} |"
            )
        lines += [
            "",
            "## 结论（初步）",
            "",
        ]
        lines += _build_conclusion(groups)
        lines += [
            "",
            "## 边界",
            "",
            f"- seed 语料 {len(seeds)} 条、可用 LLM 改写对 {len(pairs)} 条"
            f"（{n_failed} 条改写失败剔除）——样本小，结论是初步方向而非生产标定。",
            "- 改写对是「同一知识的不同表述」，与真实「不同来源抽取同一记忆」的摘要还有差距。",
            "- 未考虑 embedding 温度、类别内主题多样性对分布的偏置。",
        ]
        Path(report_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\n✓ Markdown report → {report_md}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.eval.experiments.threshold_calibration",
        description="Calibrate the four-level similarity thresholds on seed data.",
    )
    p.add_argument("--report-md", default=None, help="Write a Markdown report.")
    return p


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _build_parser().parse_args()
    rc = asyncio.run(_main(args.report_md))
    sys.exit(rc)


if __name__ == "__main__":
    main()
