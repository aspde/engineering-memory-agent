"""Simulate an N-chunk corpus: verify Recall@5 drop and rerank crossover at scale.

Scale hypothesis (docs/interview/gap-remediation.md §11.5.1): rerank's value
is scale-dependent — harmful on the 30-chunk corpus (candidate pool covers
~73%, dense ranking is already near-perfect), marginal at ~1000 (recall
starts dropping), and expected to turn *positive* at 10k+ (candidate pool
~0.4%, dense scores compress).  This probe tests that on real chunks:

Flow:
  1. Verify seed chunks exist; run a baseline eval on the *current* corpus
  2. Generate distractors: LLM-refined adjacent-topic entries + template fill
     (LLM provides the retrieval competition, templates provide the scale)
  3. Insert via write_chunks (batched, document_id="distractor")
  4. Run eval: hybrid_norerank on the N-chunk corpus
  5. (--rerank) Also run hybrid with cross-encoder rerank for comparison
  6. Compare baseline vs scaled corpus, report deltas
  7. Cleanup: DELETE FROM chunks WHERE document_id = 'distractor'

Usage:
  python -m tests.eval.probe_scale_1000            # 1000 chunks, no rerank
  python -m tests.eval.probe_scale_1000 --target 10000
  python -m tests.eval.probe_scale_1000 --target 10000 --rerank
  python -m tests.eval.probe_scale_1000 --target 200 --template-only  # smoke
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

DISTRACTOR_DOCUMENT_ID = "distractor"
SEED_DOCUMENT_ID = "ema-eval-seed"
DEFAULT_TARGET = 1000
BATCH_SIZE = 50

# LLM-refined topics: ADJACENT to the seed topics to create realistic
# retrieval competition.  Each yields ~30 entries via one LLM call.  The
# remaining entries (to reach the target) come from the template generator —
# templates give scale without burning API budget; the LLM entries give the
# competition that actually tests whether recall holds up.
LLM_REFINED_TOPICS = [
    "PostgreSQL 性能调优、索引优化、查询计划分析、VACUUM、WAL 配置",
    "FastAPI 中间件顺序、依赖注入、异步路由、后台任务、WebSocket",
    "LangGraph 图编译、子图、条件边、状态通道、流式输出",
    "LangChain Agent memory、ToolNode、create_react_agent、回调机制",
    "BGE-M3 量化、批量推理、ONNX 导出、多 GPU 推理、模型蒸馏",
    "pgvector HNSW 参数调优、IVF 索引、PQ 压缩、混合搜索融合",
    "RAG 架构模式、parent-child chunk、HyDE、多路召回融合",
    "Embedding 模型对比、Cohere、OpenAI、BGE、M3E 选型",
    "Cross-encoder vs Bi-encoder、rerank 策略、蒸馏加速",
    "Agent 工具调用、function calling、tool selection 策略",
    "HITL 人机协同、审批流、状态机、工作流引擎",
    "记忆系统设计、遗忘曲线、记忆合并、冲突检测策略",
    "实体链接与消歧、命名实体识别、知识图谱构建",
    "PostgreSQL 连接池、PgBouncer、事务隔离级别、锁管理",
    "Python asyncio 事件循环、asyncio.gather、任务取消、线程池",
    "Python GIL 影响、多进程 vs 多线程、concurrent.futures",
    "Python 类型注解、Pydantic 模型、dataclass、TypeVar",
    "pytest fixtures 作用域、参数化测试、conftest、并行测试",
    "Docker 镜像分层、多阶段构建、缓存优化、健康检查",
    "CI/CD 流水线设计、蓝绿部署、金丝雀发布、回滚策略",
    "监控告警体系、Prometheus、Grafana、SLO/SLI 定义",
    "日志聚合架构、ELK 栈、Loki、结构化日志、链路追踪",
    "Redis 缓存策略、LRU、TTL、缓存穿透与雪崩",
    "Kafka 消息队列、RabbitMQ、至少一次语义、分区与消费组",
    "微服务架构、服务发现、负载均衡、熔断降级",
    "Git 工作流、rebase vs merge、cherry-pick、子模块管理",
    "代码审查流程、PR 模板、CI 检查、静态分析工具",
    "模型版本管理、MLflow、模型注册表、A/B 测试框架",
    "特征存储设计、在线推理、离线特征、特征漂移检测",
    "Elasticsearch 集群搭建、分片策略、映射设计、聚合查询",
    "Qdrant 向量库部署、集合管理、过滤索引、Payload 索引",
    "Milvus 分布式部署、段管理、索引构建、数据导入",
    "向量检索评测、Recall/MRR/NDCG、hard-negative、LLM-as-judge",
]

LLM_PROMPT_TEMPLATE = """Generate {n} distinct short engineering memory entries about: {topic}

Requirements:
- Each entry: 2-4 sentences, technical, specific, in Chinese
- Cover different aspects/subtopics within the topic
- Sound like real engineering notes/decisions/lessons learned
- One entry per line, no numbering, no markdown
- Do NOT mention "EMA" or "Engineering Memory Agent"

Generate exactly {n} entries:"""


# ── Template fallback (no LLM) ──────────────────────────────────

_TEMPLATE_SENTENCES = [
    "在{project}项目中，{tech}的{aspect}经过压测发现{issue}，最终通过{solution}解决。",
    "{tech}的{aspect}需要关注{metric}指标，实测在{condition}下{result}。",
    "决策记录：选用{tech}而非{alternative}，原因是{reason}，代价是{tradeoff}。",
    "故障复盘：{tech}在{scenario}下出现{issue}，根因是{root_cause}，修复方案是{fix}。",
    "架构设计：{tech}采用{pattern}模式，{component}负责{responsibility}，{advantage}。",
    "代码实现：{tech}的{function}函数通过{approach}实现，注意{caveat}。",
    "性能优化：{tech}的{bottleneck}是主要延迟来源，优化后{metric}从{before}降到{after}。",
    "测试策略：{tech}的{aspect}用{test_type}覆盖，重点验证{scenario}。",
]

_PROJECTS = ["数据平台", "推荐系统", "风控引擎", "内容审核", "搜索服务", "支付网关", "用户中心", "配置中心"]
_TECHS = ["PostgreSQL", "Redis", "Kafka", "FastAPI", "Django", "Celery", "Nginx", "Docker", "Kubernetes", "gRPC"]
_ASPECTS = ["连接池配置", "缓存策略", "序列化", "并发控制", "错误处理", "日志规范", "监控埋点", "性能调优"]
_ISSUES = ["P95 延迟超标", "内存泄漏", "连接耗尽", "数据不一致", "死锁", "OOM", "超时频发", "吞吐瓶颈"]
_SOLUTIONS = ["增加连接池大小", "引入 LRU 缓存", "批量合并请求", "异步化改造", "读写分离", "加索引", "分区表"]
_METRICS = ["QPS", "P95 延迟", "内存占用", "CPU 利用率", "错误率", "吞吐量", "GC 时间"]
_CONDITIONS = ["10 并发", "100 并发", "高峰流量", "批量导入", "长时间运行", "跨可用区"]
_RESULTS = ["延迟翻倍", "吞吐下降 40%", "内存增长 3 倍", "错误率升至 2%", "P99 超过 5s"]
_ALTERNATIVES = ["MySQL", "MongoDB", "Elasticsearch", "RabbitMQ", "Flask", "gunicorn"]
_REASONS = ["运维成本低", "生态成熟", "性能更好", "团队熟悉", "事务支持", "水平扩展"]
_TRADEOFFS = ["牺牲部分性能", "增加复杂度", "需要额外组件", "学习成本高"]
_PATTERNS = ["分层", "事件驱动", "CQRS", "微服务", "管道", "插件"]
_COMPONENTS = ["调度器", "执行器", "路由层", "存储层", "缓存层", "监控层"]
_RESPONSIBILITIES = ["任务分发", "结果聚合", "流量路由", "数据持久化", "缓存加速", "指标采集"]
_ADVANTAGES = ["解耦清晰", "可独立扩展", "故障隔离", "延迟可控"]
_FUNCTIONS = ["process_request", "validate_input", "transform_data", "cache_get", "retry_with_backoff"]
_APPROACHES = ["装饰器模式", "策略模式", "责任链", "模板方法", "观察者"]
_CAVEATS = ["线程安全", "异常透传", "资源释放", "幂等性", "超时控制"]
_BOTTLENECKS = ["网络 IO", "磁盘读写", "CPU 计算", "锁竞争", "序列化"]
_BEFORES = ["800ms", "500ms", "2s", "1.5s", "300ms"]
_AFTERS = ["200ms", "120ms", "500ms", "300ms", "80ms"]
_TEST_TYPES = ["单元测试", "集成测试", "压测", "混沌工程", "契约测试"]
_SCENARIOS = ["高并发", "网络分区", "磁盘满", "依赖宕机", "慢查询"]
_ROOT_CAUSES = ["连接池泄漏", "配置错误", "资源未释放", "竞态条件", "缓存失效"]


def _template_generate(n: int) -> list[str]:
    """Generate n distractor chunks using template substitution."""
    import random
    random.seed(42)  # reproducible
    pools = {
        "project": _PROJECTS, "tech": _TECHS, "aspect": _ASPECTS,
        "issue": _ISSUES, "solution": _SOLUTIONS, "metric": _METRICS,
        "condition": _CONDITIONS, "result": _RESULTS, "alternative": _ALTERNATIVES,
        "reason": _REASONS, "tradeoff": _TRADEOFFS, "pattern": _PATTERNS,
        "component": _COMPONENTS, "responsibility": _RESPONSIBILITIES,
        "advantage": _ADVANTAGES, "function": _FUNCTIONS, "approach": _APPROACHES,
        "caveat": _CAVEATS, "bottleneck": _BOTTLENECKS, "before": _BEFORES,
        "after": _AFTERS, "test_type": _TEST_TYPES, "scenario": _SCENARIOS,
        "root_cause": _ROOT_CAUSES, "fix": _SOLUTIONS,
    }
    entries = []
    for i in range(n):
        tmpl = _TEMPLATE_SENTENCES[i % len(_TEMPLATE_SENTENCES)]
        fills = {k: random.choice(v) for k, v in pools.items()}
        entry = tmpl.format(**fills)
        # Add a second sentence for realism
        tmpl2 = _TEMPLATE_SENTENCES[(i + 3) % len(_TEMPLATE_SENTENCES)]
        fills2 = {k: random.choice(v) for k, v in pools.items()}
        entry += " " + tmpl2.format(**fills2)
        entries.append(entry)
    return entries


async def _llm_generate(n_per_topic: int = 30, concurrency: int = 8) -> list[str]:
    """Generate adjacent-topic distractors via LLM, topics in parallel.

    One provider call per topic (bounded by a semaphore so a slow/rate-limited
    provider doesn't stall the whole probe).  Raises on failure so the caller
    can fall back to templates.
    """
    from backend.service.llm_service import get_llm_provider

    llm = get_llm_provider()
    sem = asyncio.Semaphore(concurrency)

    def _parse(resp: object) -> list[str]:
        return [
            line.strip()
            for line in str(resp).strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
            and not line.strip().startswith("```")
            and len(line.strip()) > 10
        ]

    async def one(topic: str, idx: int) -> list[str]:
        async with sem:
            prompt = LLM_PROMPT_TEMPLATE.format(n=n_per_topic, topic=topic)
            resp = await llm.chat(messages=[{"role": "user", "content": prompt}])
            lines = _parse(resp)[:n_per_topic]
            logger.info(
                "LLM topic %d/%d: %s → %d entries",
                idx + 1, len(LLM_REFINED_TOPICS), topic[:20], len(lines),
            )
            return lines

    results = await asyncio.gather(
        *[one(topic, i) for i, topic in enumerate(LLM_REFINED_TOPICS)]
    )
    return [line for chunk in results for line in chunk]


# ── DB helpers ──────────────────────────────────────────────────


async def _count_chunks(document_id: str) -> int:
    from sqlalchemy import text
    from backend.db import get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            text("SELECT count(*) FROM chunks WHERE document_id = :did"),
            {"did": document_id},
        )
        return int(result.scalar() or 0)


async def _insert_distractors(entries: list[str], concurrency: int = 6) -> int:
    """Insert distractors in batches, embedding batches concurrently.

    ``write_chunks`` embeds via ``asyncio.to_thread`` (the default thread
    pool), so running several batches at once uses more CPU cores than the
    single-threaded loop — the BGE-M3 CPU embed (0.4-0.5s/chunk) is the
    dominant cost and parallelising it is what keeps a 10k corpus feasible.
    Each ``write_chunks`` call opens its own session, so concurrent calls are
    isolated.  Returns total inserted.
    """
    from backend.service.retrieval import write_chunks

    batches = [
        entries[i : i + BATCH_SIZE]
        for i in range(0, len(entries), BATCH_SIZE)
    ]
    sem = asyncio.Semaphore(concurrency)
    inserted = 0

    async def one(batch: list[str], idx: int) -> None:
        nonlocal inserted
        async with sem:
            count = await write_chunks(
                document_id=DISTRACTOR_DOCUMENT_ID,
                chunks=batch,
                meta={"source": "distractor", "batch": idx},
            )
            inserted += count
            logger.info("Inserted distractor batch %d: +%d (total %d)", idx, count, inserted)

    await asyncio.gather(*[one(b, i) for i, b in enumerate(batches)])
    return inserted


async def _cleanup_distractors() -> int:
    from sqlalchemy import text
    from backend.db import get_session_factory

    factory = get_session_factory()
    async with factory() as session:
        result = await session.execute(
            text("DELETE FROM chunks WHERE document_id = :did RETURNING id"),
            {"did": DISTRACTOR_DOCUMENT_ID},
        )
        count = len(result.fetchall())
        await session.commit()
        return count


def _run_eval(retriever: str, report_path: Path, extra: list[str] | None = None) -> dict:
    """Run one eval via the CLI and parse the overall summary.

    The subprocess runs with cwd=repo root so ``python -m tests.eval.run_eval``
    resolves imports.  Returns a dict with overall metrics + per-query misses.
    """
    cmd = [
        sys.executable, "-m", "tests.eval.run_eval",
        "--retriever", retriever,
        "--report-json", str(report_path),
    ] + (extra or [])
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_REPO_ROOT))
    if result.returncode != 0:
        logger.error("Eval stderr:\n%s", result.stderr[-2000:])
        return {}
    try:
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        overall = report["results"][0]["overall"]
        per_query = report["results"][0].get("per_query", [])
        return {
            "recall@5": overall["recall@5"],
            "mrr": overall["mrr"],
            "ndcg@5": overall["ndcg@5"],
            "latency_ms": overall.get("latency_ms", 0),
            "misses": [q for q in per_query if q.get("recall@5", 1.0) < 1.0],
        }
    except Exception as e:
        logger.error("Failed to parse report %s: %s", report_path, e)
        return {}


def _print_table(title: str, baseline: dict, scaled: dict, rerank: dict | None = None) -> None:
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)
    header = f"{'Metric':<16}{'baseline':>10}{'scaled':>10}{'Δ':>9}"
    if rerank:
        header += f"{'rerank':>14}"
    print(header)
    print("-" * 60)
    for key, label in (("recall@5", "Recall@5"), ("mrr", "MRR"), ("ndcg@5", "NDCG@5"), ("latency_ms", "Latency(ms)")):
        b = baseline.get(key, 0.0)
        s = scaled.get(key, 0.0)
        delta = s - b
        r = rerank.get(key, 0.0) if rerank else None
        if key == "latency_ms":
            b = int(b); s = int(s); delta = int(s - b)
            r = int(r) if rerank else None
        row = f"{label:<16}{b:>10}{s:>10}{delta:>+9}"
        if rerank:
            row += f"{r:>14}"
        print(row)
    b_miss = len(baseline.get("misses", []))
    s_miss = len(scaled.get("misses", []))
    r_miss = len(rerank.get("misses", [])) if rerank else None
    print(f"{'Misses':<16}{b_miss:>10}{s_miss:>10}{s_miss - b_miss:>+9}"
          + (f"{r_miss:>14}" if rerank else ""))
    if scaled.get("misses"):
        print("\nScaled-corpus missed queries:")
        for q in scaled["misses"]:
            print(f"  {q.get('id')}: {q.get('query', '')[:50]} (recall={q.get('recall@5'):.3f})")


async def main(target: int, template_only: bool, with_rerank: bool) -> None:
    t0 = time.time()

    seed_count = await _count_chunks(SEED_DOCUMENT_ID)
    logger.info("Seed chunks in DB: %d", seed_count)
    if seed_count == 0:
        logger.error("No seed chunks! Run `python -m tests.eval.seed` first.")
        return

    leftover = await _cleanup_distractors()
    if leftover:
        logger.info("Cleaned up %d leftover distractors", leftover)

    # 1. Baseline on the current corpus (before inserting distractors).
    baseline_path = _REPO_ROOT / "tests" / "eval" / "scale_baseline.json"
    logger.info("Running baseline eval on %d-chunk corpus...", seed_count)
    baseline = _run_eval("hybrid_norerank", baseline_path)
    if not baseline:
        logger.error("Baseline eval failed — aborting.")
        return

    # 2. Generate distractors.
    n_needed = max(target - seed_count, 0)
    logger.info("Generating %d distractors (target %d, %d seeds)...", n_needed, target, seed_count)
    llm_entries: list[str] = []
    if not template_only:
        try:
            llm_entries = await _llm_generate(n_per_topic=30)
            logger.info("LLM generated %d entries", len(llm_entries))
        except Exception as e:
            logger.warning("LLM generation failed (%s), using templates only", e)
            llm_entries = []
    n_template = max(n_needed - len(llm_entries), 0)
    tmpl_entries = _template_generate(n_template) if n_template else []
    entries = (llm_entries + tmpl_entries)[:n_needed]
    if len(entries) < n_needed:
        logger.error("Only %d/%d entries generated", len(entries), n_needed)
        return
    logger.info("Distractor mix: %d LLM-refined + %d template", len(llm_entries), len(tmpl_entries))

    # 3. Insert.
    logger.info("Inserting %d distractors in batches of %d...", len(entries), BATCH_SIZE)
    inserted = await _insert_distractors(entries)
    total_chunks = seed_count + inserted
    logger.info("Inserted %d distractors. Total chunks: %d", inserted, total_chunks)

    # 4-5. Scaled evals.
    norerank_path = _REPO_ROOT / "tests" / "eval" / f"scale_{target}_norerank.json"
    logger.info("Running eval: hybrid_norerank on %d-chunk corpus...", total_chunks)
    scaled = _run_eval("hybrid_norerank", norerank_path)
    if not scaled:
        # A failed scaled eval must not print an all-zero "catastrophic
        # regression" table — that would read as a real result.
        logger.error("Scaled eval failed — aborting.")
        deleted = await _cleanup_distractors()
        logger.info("Cleaned up %d distractors after failure.", deleted)
        return
    rerank: dict | None = None
    if with_rerank and scaled:
        rerank_path = _REPO_ROOT / "tests" / "eval" / f"scale_{target}_rerank.json"
        logger.info("Running eval: hybrid + cross-encoder on %d-chunk corpus (~20s/query)...", total_chunks)
        rerank = _run_eval("hybrid", rerank_path, ["--cross-encoder"])

    _print_table(
        f"SCALE COMPARISON: {seed_count} chunks vs {total_chunks} chunks (hybrid_norerank)",
        baseline, scaled, rerank,
    )
    print(f"\nTotal time: {time.time() - t0:.1f}s")

    # 6. Cleanup.
    deleted = await _cleanup_distractors()
    logger.info("Cleaned up %d distractors. DB restored to %d seed chunks.", deleted, seed_count)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    target = DEFAULT_TARGET
    template_only = "--template-only" in args
    with_rerank = "--rerank" in args
    if "--target" in args:
        target = int(args[args.index("--target") + 1])
    asyncio.run(main(target=target, template_only=template_only, with_rerank=with_rerank))
