"""Decay A/B — does Ebbinghaus decay weighting change memory retrieval?

The production memory read path ranks by ``similarity × decay_factor``
(``search_memories``) and writes recall state on every search.  The gap in
the interview materials was: **"记忆衰减加权有没有让检索变好？" had never
been measured**.  This script closes it.

It runs the same 30-query labeled set twice from *identical* decay state:

- **decay on**  (production default): rank by ``weighted_score``.
- **decay off** (control): rank by raw similarity, no decay write.

A fresh seed corpus makes the A/B vacuous — every factor is ≈1.0 (0 hours
old), so both arms return the same ranking.  The script therefore applies a
deterministic synthetic aging profile first (hot / medium / cold bands), so
``decay_factor`` actually discriminates.  The profile is synthetic and the
report says so — the measurement isolates the *formula's* effect on ranking,
not a claim about real production decay distributions.

The decay-on arm writes recall state (that is the production behavior); a
snapshot/restore of the decay columns between arms guarantees both see the
same starting point, and the script restores the original state before
exiting so the CI-gated baseline corpus is left untouched.

Usage:
    # seeds must exist first (30 eval_seed memories)
    python -m tests.eval.seed --memories --clear
    python -m tests.eval.decay_ab --report-md tests/eval/reports/decay_ab_report.md
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any

from tests.eval.runner import EvalConfig, run_eval


# ── Synthetic decay profile ─────────────────────────────────────────

HOT_HOURS = 10  # band A: created 1..10h ago
MED_HOURS = 40  # band B: created 12..39h ago
COLD_HOURS = 120  # band C: created 48..120h ago


def _profile_for(index: int, n: int) -> dict[str, Any]:
    """Deterministic decay profile for eval seed ``index`` of ``n``.

    Three roughly-equal bands simulate hot (recent + well-recalled), medium,
    and cold (old + rarely-recalled) memories.  Under the production formula
    (``S = 1 + (recall+1)·12`` with a 0.10 retention floor) the hot band
    stays near factor 1.0, the cold band saturates at the 0.10 floor, and
    only the medium band spreads — enough that ranking changes without
    collapsing either arm.
    """
    band = index / n
    if band < 0.34:
        hours = 1 + index  # 1..10h
        recall = 10 + (index % 6)  # 10..15
        recalled_ago = max(1, hours // 3)
    elif band < 0.67:
        hours = 12 + (index % 10) * 3  # 12..39h
        recall = 3 + (index % 5)  # 3..7
        recalled_ago = max(2, hours // 2)
    else:
        hours = 48 + (index % 10) * 8  # 48..120h
        recall = index % 3  # 0..2
        recalled_ago = max(24, hours)
    return {
        "created_hours_ago": hours,
        # recalled_at = now - recalled_ago; NULL when never recalled
        "recalled_hours_ago": recalled_ago if recall > 0 else None,
        "recall_count": recall,
    }


# ── DB helpers ──────────────────────────────────────────────────────


async def _snapshot_decay_state() -> list[dict[str, Any]]:
    """Capture the decay columns (created_at / recalled_at / recall_count /
    stored decay_factor) for eval_seed rows."""
    from sqlalchemy import text

    from backend.db import get_session_factory

    async with get_session_factory()() as session:
        rows = await session.execute(
            text(
                "SELECT id, created_at, recalled_at, recall_count, decay_factor "
                "FROM memories WHERE source_type = 'eval_seed'"
            )
        )
        return [dict(r._mapping) for r in rows]


async def _restore_decay_state(snapshot: list[dict[str, Any]]) -> None:
    """Reset the eval_seed decay columns — including the stored decay_factor,
    which ``update_decay_batch`` writes — to a captured state."""
    from sqlalchemy import text

    from backend.db import get_session_factory

    async with get_session_factory()() as session:
        for row in snapshot:
            await session.execute(
                text(
                    "UPDATE memories SET created_at = :created_at, "
                    "recalled_at = :recalled_at, recall_count = :recall_count, "
                    "decay_factor = :decay_factor "
                    "WHERE id = :id"
                ),
                {
                    "id": row["id"],
                    "created_at": row["created_at"],
                    "recalled_at": row["recalled_at"],
                    "recall_count": row["recall_count"],
                    "decay_factor": row["decay_factor"],
                },
            )
        await session.commit()


async def _apply_decay_profile() -> int:
    """Backdate created_at / set recalled_at / recall_count on eval_seed rows."""
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import text

    from backend.db import get_session_factory

    ids: list[Any] = []
    async with get_session_factory()() as session:
        rows = await session.execute(
            text("SELECT id FROM memories WHERE source_type = 'eval_seed' ORDER BY id")
        )
        ids = [r[0] for r in rows.fetchall()]
        if not ids:
            raise SystemExit(
                "No eval_seed memories — run `python -m tests.eval.seed --memories --clear` first."
            )

        now = datetime.now(timezone.utc)
        for idx, mid in enumerate(ids):
            prof = _profile_for(idx, len(ids))
            created_at = now - timedelta(hours=prof["created_hours_ago"])
            recalled_at = (
                now - timedelta(hours=prof["recalled_hours_ago"])
                if prof["recalled_hours_ago"] is not None
                else None
            )
            await session.execute(
                text(
                    "UPDATE memories SET created_at = :created_at, "
                    "recalled_at = :recalled_at, recall_count = :recall_count "
                    "WHERE id = :id"
                ),
                {
                    "id": mid,
                    "created_at": created_at,
                    "recalled_at": recalled_at,
                    "recall_count": prof["recall_count"],
                },
            )
        await session.commit()
    return len(ids)


# ── A/B ─────────────────────────────────────────────────────────────


@dataclass
class ArmResult:
    label: str
    result: Any


def _overview(result: Any) -> dict[str, float]:
    o = result.overall
    return {
        "recall@5": o.get("recall@5", 0.0),
        "mrr": o.get("mrr", 0.0),
        "ndcg@5": o.get("ndcg@5", 0.0),
        "avg_latency_ms": o.get("latency_ms", 0.0),
    }


async def _ranking_churn(snapshot: list[dict[str, Any]]) -> tuple[int, int]:
    """Count queries whose top-5 memory-id lists differ between the arms.

    Re-runs both adapters per query.  Fairness: the decay-off arm ranks by
    raw similarity and writes nothing, so its output is independent of the
    decay state — the decay-on arm's per-query writes can't bias the
    comparison.  Embeddings are LRU-cached from the eval runs, so this is
    cheap.
    """
    from tests.eval.dataset import build_adapter, load_ground_truth

    on = build_adapter("memory", use_decay=True)
    off = build_adapter("memory", use_decay=False)
    items = load_ground_truth()
    await _restore_decay_state(snapshot)
    churned = 0
    for it in items:
        a = [str(r["id"]) for r in await on.fn(it.query, 5)]
        b = [str(r["id"]) for r in await off.fn(it.query, 5)]
        if a != b:
            churned += 1
    return churned, len(items)


async def main(report_md: str | None) -> int:
    # Snapshot the FRESH post-seed decay state first — the script must leave
    # the corpus exactly as it found it (the CI-gated baseline assumes fresh
    # seeds; an aged corpus would shift the memory-path numbers).
    fresh_snapshot = await _snapshot_decay_state()

    n = await _apply_decay_profile()
    print(f"[OK] Applied synthetic decay profile to {n} eval_seed memories")

    # Profiled state — the starting point both arms (and churn) restore to,
    # so every arm sees identical decay columns.
    profiled_snapshot = await _snapshot_decay_state()

    def _cfg(use_decay: bool) -> EvalConfig:
        return EvalConfig(
            name="memory:norank@k5" if use_decay else "memory:norank:nodecay@k5",
            retriever="memory",
            top_k=5,
            use_decay=use_decay,
        )

    print("Running decay-ON arm (production default)...")
    on = await run_eval(_cfg(use_decay=True))
    await _restore_decay_state(profiled_snapshot)

    print("Running decay-OFF arm (raw similarity control)...")
    off = await run_eval(_cfg(use_decay=False))
    await _restore_decay_state(profiled_snapshot)

    on_sum = _overview(on)
    off_sum = _overview(off)
    churned, total = await _ranking_churn(profiled_snapshot)

    # Leave the corpus fresh.
    await _restore_decay_state(fresh_snapshot)

    print("\n=== Decay A/B (30 queries, same decay state, synthetic aging profile) ===")
    print(f"  {'metric':<14} {'decay on':>10} {'decay off':>10} {'d':>10}")
    for k in ("recall@5", "mrr", "ndcg@5"):
        print(f"  {k:<14} {on_sum[k]:>10.3f} {off_sum[k]:>10.3f} {on_sum[k]-off_sum[k]:>+10.3f}")
    print(f"  {'queries reordered':<14} {churned:>10}/{total}")

    if report_md:
        lines = [
            "# Decay A/B — does Ebbinghaus decay weighting change memory retrieval?",
            "",
            f"> 日期 2026-08-11 · 30 条标注 query · memory 路径 · 同一衰减状态下运行双臂",
            "",
            "## 方法",
            "",
            "生产 memory 读路径用 `similarity × decay_factor` 排序并写衰减状态。本 A/B 从**相同衰减状态**"
            "（快照/恢复保证）跑同一 30 条 query 两次：",
            "",
            "- **decay on**（生产默认）：`weighted_score` 排序 + 写 recall 状态",
            "- **decay off**（对照）：纯相似度排序，零衰减写入",
            "",
            "新鲜 seed 语料下所有 factor≈1.0，A/B 无意义，故先施加一个**合成老化分布**"
            "（hot 1-10h / medium 12-39h / cold 48-120h，recall 0-15），让 decay_factor 真正区分行。"
            "分布是合成的——本测量隔离的是**公式对排序的影响**，不代表真实生产的衰减分布。",
            "",
            "## 结果",
            "",
            "| 指标 | decay on | decay off | Δ |",
            "|------|---------|-----------|-----|",
        ]
        for k in ("recall@5", "mrr", "ndcg@5"):
            lines.append(
                f"| {k} | {on_sum[k]:.3f} | {off_sum[k]:.3f} | "
                f"{on_sum[k]-off_sum[k]:+.3f} |"
            )
        lines += [
            f"| 排序变化的 query 数 | {churned}/{total} | — | — |",
            "",
            "## 解读",
            "",
            "（跑完回填：哪个指标升/降、多少条 query 的目标记忆被衰减压出 top-5、MRR 变化。）",
            "",
            "## 边界",
            "",
            "- 合成老化分布，非真实生产衰减状态；测量的是公式效应而非真实语料行为。",
            "- 双臂起点一致（快照恢复），但 decay-on 自身随每次运行改变状态——跨天重跑数字可能漂移。",
            "- 未调 `S=1+2·recall` 的系数；这是公式本身的行为，不是调参后的行为。",
        ]
        with open(report_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n✓ Markdown report → {report_md}")

    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tests.eval.decay_ab",
        description="A/B the Ebbinghaus decay weighting on the memory retrieval eval.",
    )
    p.add_argument(
        "--report-md",
        default=None,
        help="Write a Markdown report to this path.",
    )
    return p


def main_entry() -> None:
    # Windows consoles default to GBK, which cannot encode the ✓/— chars this
    # script prints — reconfigure stdout so a console-encoding error can't
    # kill the run (the DB state restore must always run).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _build_parser().parse_args()
    import asyncio

    rc = asyncio.run(main(args.report_md))
    sys.exit(rc)


if __name__ == "__main__":
    main_entry()
