"""Locust load test for EMA /api/memory/search.

Usage (from repo root, with the backend running):
    locust -f tests/perf/locustfile.py --host http://127.0.0.1:8000
    # then open http://localhost:8089, set user count / spawn rate / duration

The API-key guard (backend/api/auth.py) requires `Authorization: Bearer
<EMA_API_KEY>` on every /api call; the key is read from the repo-root .env
(via python-dotenv).  Queries are drawn from the eval ground-truth set so
every request exercises real retrieval (BGE-M3 embed + pgvector + sparse),
not an empty result set.

Two modes:
- default (cold): a ~750-query pool generated from the seed corpus's named
  entities × natural question templates, so consecutive requests almost never
  repeat a query text and the query-embedding LRU cache (1024 entries, in the
  *backend* process) is missed most of the time → measures the true
  per-query embed cost on BGE-M3 CPU.
- `HOT=1 locust ...`: the original 10 fixed queries, all of which hit the
  backend's LRU after the first pass → measures the cached hot path.

The pool is built at import time from ``tests/eval/seed_memories.jsonl``
(entities) + ``tests/eval/ground_truth.py`` (natural queries), so it stays
in sync with the seeded corpus.
"""

import json
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from locust import HttpUser, between, task

load_dotenv()  # repo-root .env → EMA_API_KEY etc.

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Real queries from tests/eval/ground_truth.py — all hit seeded corpus rows.
_HOT_QUERIES = [
    "PostgreSQL 连接池配置多少合适",
    "之前怎么修的 OOM 问题",
    "LangGraph 为什么不用 LangChain Agent",
    "实体归一化怎么做",
    "记忆去重相似度阈值",
    "艾宾浩斯衰减公式",
    "HITL 审批流程",
    "max_steps 防循环",
    "连接器支持哪些平台",
    "Phase 分几个阶段",
]

# Natural question templates × seed-corpus entities → a large pool whose
# texts are almost all unique, so the backend's query-embedding LRU misses.
_QUESTION_TEMPLATES = [
    "{e} 是怎么实现的",
    "{e} 之前出过什么问题",
    "为什么要用 {e}",
    "{e} 有哪些坑",
    "{e} 性能怎么样",
    "{e} 和替代方案怎么选",
    "如何优化 {e}",
    "{e} 的配置怎么调",
    "{e} 为什么不能用",
    "{e} 有什么优缺点",
    "{e} 遇到过哪些问题",
    "怎么解决 {e} 的问题",
    "{e} 是怎么选的",
    "之前怎么处理的 {e}",
    "{e} 的原理是什么",
]


def _build_cold_pool() -> list[str]:
    """Seed-entity × template queries + the ground-truth set → ~750 unique texts."""
    entities: list[str] = []
    seed_path = _REPO_ROOT / "tests" / "eval" / "seed_memories.jsonl"
    if seed_path.exists():
        with open(seed_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                for e in data.get("entities", []):
                    name = (e.get("name") or "").strip()
                    if name and name not in entities:
                        entities.append(name)

    pool = [
        tpl.format(e=e)
        for e in entities
        for tpl in _QUESTION_TEMPLATES
    ]
    # Add the natural ground-truth queries so the pool isn't all templates.
    try:
        from tests.eval.ground_truth import GROUND_TRUTH
        pool.extend(g["query"] for g in GROUND_TRUTH)
    except Exception:
        pass  # ground_truth import is optional (noisy cwd) — templates suffice
    # Drop duplicates while preserving order.
    seen: set[str] = set()
    deduped = [q for q in pool if not (q in seen or seen.add(q))]
    return deduped


_COLD_QUERIES = _build_cold_pool()
QUERIES = _HOT_QUERIES if os.getenv("HOT") == "1" else _COLD_QUERIES


class MemorySearchUser(HttpUser):
    wait_time = between(1, 3)
    host = "http://127.0.0.1:8000"

    @task
    def search_memories(self):
        self.client.post(
            "/api/memory/search",
            json={"query": random.choice(QUERIES), "top_k": 5},
            headers={"Authorization": f"Bearer {self.ema_api_key}"},
        )

    def on_start(self):
        import os

        self.ema_api_key = os.getenv("EMA_API_KEY", "")
