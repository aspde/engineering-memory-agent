"""Measure real end-to-end chat P95 latency and per-turn token cost.

Drives the real FastAPI app (real LLM via DeepSeek, real retrieval via
local BGE-M3, real DB) through ``httpx.ASGITransport`` — no server, but
the full request → agent ReAct loop → response path is exercised exactly
as ``/api/agent/chat`` would in production.

Usage:
    python -m tests.perf.measure_chat_p95 [--turns 10]

Reads ``.env`` for credentials.  Writes nothing except the conversation
rows the app itself persists.  Run with ``AUTO_MEMORY_ENABLED`` unset —
this script forces it off so the chat turns don't trigger background
memory extraction (which would pollute the corpus and burn tokens).

Output: per-turn elapsed seconds + token counts (read back from the
``llm_usage`` table after the run, cost via ``usage.estimate_cost``),
then the P95 of the end-to-end latencies.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import statistics
import time
import uuid
from pathlib import Path

# ── Environment must be finalised before any backend import reads it ──
# Force these OFF regardless of .env: a running patrol would steal LLM
# slots and pollute both the latency P95 and the token/cost summary.
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")
os.environ["AUTO_MEMORY_ENABLED"] = "false"
os.environ["PATROL_ENABLED"] = "false"
os.environ["PATROL_WEEKLY_ENABLED"] = "false"
os.environ["ALERTS_ENABLED"] = "false"

# Quiet the app's own logging during the run.
logging.getLogger("backend").setLevel(logging.WARNING)
logging.getLogger("backend.agent").setLevel(logging.WARNING)

# Realistic queries built from the seeded corpus (EMA's own engineering
# history), a mix of memory-retrieval, chunk-retrieval and direct answers.
DEFAULT_QUERIES = [
    "BGE-M3 嵌入模型选型时考虑了哪些因素？",
    "之前那次 502 事故的根因是什么？",
    "PostgresSaver 在 Windows 上为什么不可用？",
    "为什么选 PostgreSQL 而不是 Elasticsearch？",
    "Agent 状态图有几个节点，分别是什么？",
    "记忆去重的四分支逻辑是怎么实现的？",
    "ivfflat 索引在小数据量下有什么坑？",
    "chunks 和 memories 表是怎么分工的？",
    "BGE-M3 的向量维度是多少？",
    "为什么不用 LangChain Agent？",
]


async def _run_turn(client, api_key: str, message: str, idx: int) -> tuple[str, float]:
    """One real chat turn.  Returns (thread_id, end-to-end seconds)."""
    thread_id = f"p95-{uuid.uuid4().hex[:12]}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"message": message, "thread_id": thread_id}

    t0 = time.perf_counter()
    resp = await client.post("/api/agent/chat", headers=headers, json=payload)
    elapsed = time.perf_counter() - t0
    resp.raise_for_status()
    body = resp.json()
    status = body.get("status")
    print(
        f"  [{idx+1:2d}] {elapsed:6.2f}s  status={status}  "
        f"resp={len(body.get('response') or '')}ch  q={message[:24]}…"
    )
    if status != "completed":
        print(f"      ⚠ interrupted/error: {body.get('interrupt') or body.get('detail')}")
    return thread_id, elapsed


async def _summarize_usage() -> None:
    """After the lifespan exits (flusher drained on shutdown), read the rows
    this run wrote to ``llm_usage`` and print the token/cost breakdown."""
    import asyncpg

    from backend.service.usage import estimate_cost

    conn = await asyncpg.connect(os.environ["DATABASE_URL"])
    try:
        rows = await conn.fetch(
            """
            SELECT scenario, provider, model,
                   sum(input_tokens) i, sum(output_tokens) o,
                   sum(cache_read_tokens) cr, sum(cache_creation_tokens) cc,
                   sum(total_tokens) t, count(*) n
            FROM llm_usage
            WHERE thread_id LIKE 'p95-%%'
            GROUP BY scenario, provider, model
            ORDER BY t DESC
            """
        )
    finally:
        await conn.close()

    print("\n=== llm_usage: this run (thread_id LIKE 'p95-%') ===")
    total = 0.0
    for r in rows:
        cost = estimate_cost(
            r["model"], r["i"], r["o"], r["cr"], r["cc"], provider=r["provider"]
        )
        total += cost
        print(
            f"  {r['scenario']:16s} {r['model']:22s} n={r['n']:3d} "
            f"tokens={r['t']:7d} (i={r['i']} o={r['o']} cr={r['cr']} cc={r['cc']}) "
            f"est=${cost:.6f}"
        )
    print(f"  ───────────────────────────── 合计估算成本: ${total:.6f}")


async def main(turns: int) -> None:
    from httpx import ASGITransport, AsyncClient

    from backend.main import app, lifespan

    queries = (DEFAULT_QUERIES * (turns // len(DEFAULT_QUERIES) + 1))[:turns]
    api_key = os.environ.get("EMA_API_KEY", "")

    async with lifespan(app):
        transport = ASGITransport(app=app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Warm the embedding model (lifespan already does, but a second
            # pass is cheap and guards against lazy first-request load).
            try:
                from backend.service.embedding_service import get_embedding_provider_async
                await get_embedding_provider_async()
            except Exception:
                pass

            print(f"=== 实测 {len(queries)} 轮真实 /api/agent/chat（AUTO_MEMORY 关闭）===")
            results: list[tuple[str, float]] = []
            for idx, q in enumerate(queries):
                thread_id, elapsed = await _run_turn(client, api_key, q, idx)
                results.append((thread_id, elapsed))

    latencies = sorted(s for _, s in results)
    n = len(latencies)
    p95 = latencies[min(n - 1, int(n * 0.95))] if n else 0.0
    p50 = latencies[min(n - 1, int(n * 0.50))]
    print(f"\n=== 端到端对话延迟（{n} 轮）===")
    print(f"  P50   = {p50:6.2f}s")
    print(f"  P95   = {p95:6.2f}s")
    print(f"  P99   = {latencies[min(n - 1, int(n * 0.99))]:6.2f}s")
    print(f"  min   = {min(latencies):6.2f}s   max = {max(latencies):6.2f}s")
    print(f"  mean  = {statistics.mean(latencies):6.2f}s")

    await _summarize_usage()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=10)
    args = parser.parse_args()
    asyncio.run(main(args.turns))
