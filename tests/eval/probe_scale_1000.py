"""Simulate 1000-chunk corpus: verify Recall@5 drop without rerank at scale.

Flow:
  1. Verify 30 seed chunks exist in DB
  2. Generate 970 distractor chunks (LLM or template fallback)
  3. Insert distractors via write_chunks (batched, document_id="distractor")
  4. Run eval: hybrid_norerank on 1000-chunk corpus
  5. Compare to 30-chunk baseline (Recall@5=1.000, MRR=0.983)
  6. Cleanup: DELETE FROM chunks WHERE document_id = 'distractor'

Usage:
  python -u probe_scale_1000.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import subprocess
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DISTRACTOR_DOCUMENT_ID = "distractor"
SEED_DOCUMENT_ID = "ema-eval-seed"
TARGET_TOTAL = 1000
BATCH_SIZE = 50

# 33 topics × ~30 entries = ~990 distractors.
# Topics are ADJACENT to seed topics to create realistic retrieval competition.
DISTRACTOR_TOPICS = [
    "PostgreSQL 性能调优、索引优化、查询计划分析、VACUUM、WAL 配置",
    "FastAPI 中间件顺序、依赖注入、异步路由、后台任务、WebSocket",
    "React 组件设计、状态管理、Suspense、错误边界、性能优化",
    "LangGraph 图编译、子图、条件边、状态通道、流式输出",
    "LangChain Agent memory、ToolNode、create_react_agent、回调机制",
    "BGE-M3 量化、批量推理、ONNX 导出、多 GPU 推理、模型蒸馏",
    "pgvector HNSW 参数调优、IVF 索引、PQ 压缩、混合搜索融合",
    "Elasticsearch 集群搭建、分片策略、映射设计、聚合查询",
    "Qdrant 向量库部署、集合管理、过滤索引、Payload 索引",
    "Milvus 分布式部署、段管理、索引构建、数据导入",
    "Python asyncio 事件循环、asyncio.gather、任务取消、线程池",
    "Python GIL 影响、多进程 vs 多线程、concurrent.futures",
    "Python 类型注解、Pydantic 模型、dataclass、TypeVar",
    "pytest fixtures 作用域、参数化测试、conftest、并行测试",
    "Docker 镜像分层、多阶段构建、缓存优化、健康检查",
    "CI/CD 流水线设计、蓝绿部署、金丝雀发布、回滚策略",
    "日志聚合架构、ELK 栈、Loki、结构化日志、链路追踪",
    "监控告警体系、Prometheus、Grafana、SLO/SLI 定义",
    "模型版本管理、MLflow、模型注册表、A/B 测试框架",
    "特征存储设计、在线推理、离线特征、特征漂移检测",
    "RAG 架构模式、parent-child chunk、HyDE、多路召回融合",
    "Embedding 模型对比、Cohere、OpenAI、BGE、M3E 选型",
    "Cross-encoder vs Bi-encoder、rerank 策略、蒸馏加速",
    "Agent 工具调用、function calling、tool selection 策略",
    "HITL 人机协同、审批流、状态机、工作流引擎",
    "记忆系统设计、遗忘曲线、记忆合并、冲突检测策略",
    "实体链接与消歧、命名实体识别、知识图谱构建",
    "PostgreSQL 连接池、PgBouncer、事务隔离级别、锁管理",
    "Redis 缓存策略、LRU、TTL、缓存穿透与雪崩",
    "Git 工作流、rebase vs merge、cherry-pick、子模块管理",
    "代码审查流程、PR 模板、CI 检查、静态分析工具",
    "微服务架构、服务发现、负载均衡、熔断降级",
    "消息队列设计、Kafka、RabbitMQ、至少一次语义",
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


async def _llm_generate(n_per_topic: int = 30) -> list[str]:
    """Generate distractors via LLM. Raises on failure."""
    from backend.service.llm_service import get_llm_provider

    llm = get_llm_provider()
    all_entries: list[str] = []
    for i, topic in enumerate(DISTRACTOR_TOPICS):
        prompt = LLM_PROMPT_TEMPLATE.format(n=n_per_topic, topic=topic)
        resp = await llm.chat(messages=[{"role": "user", "content": prompt}])
        lines = [
            line.strip()
            for line in str(resp).strip().split("\n")
            if line.strip() and not line.strip().startswith("#")
            and not line.strip().startswith("```")
            and len(line.strip()) > 10
        ]
        all_entries.extend(lines[:n_per_topic])
        logger.info("LLM topic %d/%d: %s → %d entries", i + 1, len(DISTRACTOR_TOPICS), topic[:20], len(lines[:n_per_topic]))
    return all_entries


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


async def _insert_distractors(entries: list[str]) -> int:
    """Insert distractors in batches. Returns total inserted."""
    from backend.service.retrieval import write_chunks

    total = 0
    for i in range(0, len(entries), BATCH_SIZE):
        batch = entries[i : i + BATCH_SIZE]
        count = await write_chunks(
            document_id=DISTRACTOR_DOCUMENT_ID,
            chunks=batch,
            meta={"source": "distractor", "batch": i // BATCH_SIZE},
        )
        total += count
        logger.info("Inserted distractor batch %d: +%d (total %d)", i // BATCH_SIZE, count, total)
    return total


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


async def main(template_only: bool = False):
    t0 = time.time()

    # 1. Verify seeds exist
    seed_count = await _count_chunks(SEED_DOCUMENT_ID)
    logger.info("Seed chunks in DB: %d", seed_count)
    if seed_count == 0:
        logger.error("No seed chunks! Run `python -m tests.eval.seed` first.")
        return

    # 2. Clean any leftover distractors
    leftover = await _cleanup_distractors()
    if leftover:
        logger.info("Cleaned up %d leftover distractors", leftover)

    # 3. Generate distractors
    n_needed = TARGET_TOTAL - seed_count
    n_per_topic = max(n_needed // len(DISTRACTOR_TOPICS) + 1, 25)
    logger.info("Generating %d distractors (%d per topic × %d topics)...",
                n_needed, n_per_topic, len(DISTRACTOR_TOPICS))

    if template_only:
        entries = _template_generate(n_needed)
        logger.info("Template generated %d entries (--template-only)", len(entries))
    else:
        try:
            entries = await _llm_generate(n_per_topic=n_per_topic)
            logger.info("LLM generated %d entries", len(entries))
        except Exception as e:
            logger.warning("LLM generation failed (%s), falling back to templates", e)
            entries = _template_generate(n_needed)
            logger.info("Template generated %d entries", len(entries))

    entries = entries[:n_needed]
    if len(entries) < n_needed:
        logger.warning("Only %d entries generated, topping up with templates", len(entries))
        entries.extend(_template_generate(n_needed - len(entries)))

    # 4. Insert distractors
    logger.info("Inserting %d distractors in batches of %d...", len(entries), BATCH_SIZE)
    inserted = await _insert_distractors(entries)
    total_chunks = seed_count + inserted
    logger.info("Inserted %d distractors. Total chunks: %d", inserted, total_chunks)

    # 5. Run eval (hybrid_norerank)
    logger.info("Running eval: hybrid_norerank on %d-chunk corpus...", total_chunks)
    report_path = os.path.join(os.path.dirname(__file__), "scale_1000_report.json")
    result = subprocess.run(
        [
            sys.executable, "-m", "tests.eval.run_eval",
            "--retriever", "hybrid_norerank",
            "--report-json", report_path,
        ],
        capture_output=True, text=True, cwd=os.path.dirname(__file__),
    )
    logger.info("Eval exit code: %d", result.returncode)
    if result.returncode != 0:
        logger.error("Eval stderr:\n%s", result.stderr[-2000:])

    # 6. Parse results
    try:
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        overall = report["results"][0]["overall"]
        recall = overall["recall@5"]
        mrr = overall["mrr"]
        ndcg = overall["ndcg@5"]
        latency = overall["latency_ms"]
        per_query = report["results"][0].get("per_query", [])
        misses = [q for q in per_query if q["recall@5"] < 1.0]
    except Exception as e:
        logger.error("Failed to parse report: %s", e)
        recall = mrr = ndcg = latency = 0.0
        misses = []

    # 7. Report comparison
    print("\n" + "=" * 70)
    print("SCALE COMPARISON: 30 chunks vs 1000 chunks (hybrid_norerank)")
    print("=" * 70)
    print(f"{'Metric':<20} {'30 chunks':>15} {'1000 chunks':>15} {'Delta':>10}")
    print("-" * 60)
    print(f"{'Recall@5':<20} {'1.000':>15} {recall:>15.3f} {recall - 1.000:>+10.3f}")
    print(f"{'MRR':<20} {'0.983':>15} {mrr:>15.3f} {mrr - 0.983:>+10.3f}")
    print(f"{'NDCG@5':<20} {'0.988':>15} {ndcg:>15.3f} {ndcg - 0.988:>+10.3f}")
    print(f"{'Latency (ms)':<20} {'235':>15} {latency:>15.0f} {latency - 235:>+10.0f}")
    print(f"{'Misses':<20} {'0':>15} {len(misses):>15}")
    if misses:
        print(f"\nMissed queries ({len(misses)}):")
        for q in misses:
            print(f"  {q['id']}: {q['query']} (recall={q['recall@5']:.3f}, "
                  f"difficulty={q.get('difficulty', '?')})")
    print("=" * 70)
    print(f"\nTotal time: {time.time() - t0:.1f}s")

    # 8. Cleanup
    deleted = await _cleanup_distractors()
    logger.info("Cleaned up %d distractors. DB restored to %d seed chunks.", deleted, seed_count)


if __name__ == "__main__":
    import sys
    asyncio.run(main(template_only="--template-only" in sys.argv))
