"""Locust load test for EMA /api/memory/search.

Usage (from repo root, with the backend running):
    locust -f tests/perf/locustfile.py --host http://127.0.0.1:8000
    # then open http://localhost:8089, set user count / spawn rate / duration

The API-key guard (backend/api/auth.py) requires `Authorization: Bearer
<EMA_API_KEY>` on every /api call; the key is read from the repo-root .env
(via python-dotenv).  Queries are drawn from the eval ground-truth set so
every request exercises real retrieval (BGE-M3 embed + pgvector + sparse),
not an empty result set.
"""

import random

from dotenv import load_dotenv
from locust import HttpUser, between, task

load_dotenv()  # repo-root .env → EMA_API_KEY etc.

# Real queries from tests/eval/ground_truth.py — all hit seeded corpus rows.
QUERIES = [
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
