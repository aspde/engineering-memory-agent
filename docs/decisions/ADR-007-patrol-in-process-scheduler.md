# ADR-007: 巡检调度器内嵌主进程，不引入独立任务队列

**日期**: 2026-08-04
**更新**: 2026-08-07（补充 missed-slot catch-up 机制）

**状态**: 已接受

## 背景

Phase 3 主动 Agent 需要定时触发巡检（每日 / 每周 / 技术债扫描），巡检调度器需要决定用哪种实现方式。候选方案：

1. **内嵌主进程的轻量 asyncio 循环**（`asyncio.sleep` + loop）
2. **APScheduler**（进程内定时库）
3. **Celery / Redis / Bull**（独立 worker + 持久化任务队列 / broker）

调度规则本身很简单——每天固定小时、每周固定天 + 小时，没有秒级 / 分钟级定时需求，也不需要任务的持久化重放。

## 决策

**巡检调度器内嵌 FastAPI 主进程**（`backend/service/scheduler.py` 的 `PatrolScheduler`），在 `backend/main.py` 的 lifespan 中启动、shutdown 时取消。不引入 APScheduler、Celery、Redis、Bull 等任何任务调度依赖，不使用持久化任务队列。

实现形态：

```python
# 核心是 background task + asyncio.sleep 循环（scheduler.py）
async def _loop():
    while True:
        next_run = calculate_next_run(now, schedule)
        await asyncio.sleep((next_run - now).total_seconds())
        await callback()
```

## 理由

1. **调度规则简单，复杂度不匹配**：每天固定小时、每周固定天 + 小时，`asyncio.sleep` + 循环就能精确表达。APScheduler 的 cron 表达式、Celery 的 broker / worker / 重试体系在这里没有用武之地——引入它们是把不存在的需求提前实现。

2. **零新依赖**：Celery 强依赖 Redis / RabbitMQ 做 broker，Redis 本身又是一套要部署、备份、监控的服务。APScheduler 虽轻，但仍是新增第三方运行时。EMA 的约束是"简单优先、禁止随意增加依赖"，当前方案只复用 asyncio 标准库。

3. **错过一次巡检可接受**：巡检是低频、重量级的养护任务（每天 / 每周一次），不是用户请求路径。进程重启最多丢一个调度点，下次定时仍会跑——丢失的代价是"少一次扫描"，不是数据损坏或功能不可用，不需要任务持久化来兜底。

4. **巡检只是触发方式，与 Agent 层解耦**：巡检复用同一个 `build_agent_graph()` 产物，只是 System Prompt 与触发方式不同（对话 / Cron / Webhook / 手动）。调度器只负责"到点调 `run_patrol()`"，不承载任何业务逻辑，后续要换成独立 worker 不影响 Agent 层。

## 代价

以下代价已认知并接受，每个都有缓解措施。

### 代价 1：进程重启丢失调度点

调度循环只在进程存活期间起作用——服务在计划时间点停机（重启 / 部署 / 崩溃）时，那个 slot 不会被触发。

**缓解**：启动时用 `should_catch_up()`（[scheduler.py](../../backend/service/scheduler.py)）对比最近调度 slot 与 `patrol_logs` 历史——有历史且该 slot 之后没跑过，就在后台补跑一次（trigger 记为 `cron_catchup`）。历史 guard 保证全新安装不会在首次启动时误触发巡检。补跑与定时跑之间的并发重叠由 `run_patrol` 的 overlap guard 防止。

### 代价 2：单进程故障导致巡检不可用

主进程挂了，巡检就停了——与 Web 服务同生命周期。

**缓解**：巡检是养护任务，短暂缺失可接受。兜底手段：`PATROL_ENABLED` 开关可关停；`POST /api/patrol/trigger` 可手动触发；服务恢复后 catch-up 机制自动补跑；遗留的 mid-run 巡检在下次启动时由 `mark_stale_patrols_failed()` 标记 failed，日志不撒谎。

### 代价 3：多实例部署会重复触发

内嵌调度器意味着每个实例各跑一个循环——多实例部署时同一 slot 会被触发多次。

**缓解**：当前按单实例设计部署（Dockerfile 单后端进程）。多实例是拐点之一（见下），到那时引入 DB 锁 / 租约选主，或改独立 worker。

### 代价 4：调度逻辑与 Web 进程生命周期耦合

调度任务和请求处理共享同一事件循环，长巡检会占用循环资源。

**缓解**：巡检全程异步（`await run_patrol()`，IO 密集），事件循环在 await 期间让出，不阻塞请求处理。lifespan 在 shutdown 时统一 cancel 所有调度与 catch-up task，不留 detached task。

## 拐点

当以下任一条件出现时，重新评估是否引入独立 worker / Celery / APScheduler：

1. 巡检频率提升到分钟级，或出现需要精确到秒的调度需求——`asyncio.sleep` 循环会变得脆弱
2. 需要任务持久化重放——进程崩溃后未跑的任务要排队补执行，而不只是一次 catch-up
3. 多实例 / 水平扩展部署——内嵌调度器无法选主，需要 DB 锁、租约或独立 worker
4. 巡检成为用户可见的关键路径（如 SLA 保证），单进程故障不可接受

到那时再按实测需求引入，而不是现在预置。

## 后果

- 巡检调度零新增依赖，`PatrolScheduler`（约 120 行）+ asyncio 实现，主进程内启动 / 优雅关闭
- 重启丢一个调度点可接受，且已有 catch-up 兜底；技术债扫描复用同一调度器（`main.py` 里 `schedule_weekly`）
- 后续迁移独立 worker 时，只需把 `main.py` 的调度段移到独立进程，Agent / 巡检业务代码不变
- 本决策对应 [ADR-006](./ADR-006-extension-roadmap.md) 的 Phase 3（主动 Agent），详细设计与巡检 Prompt 模板见 [phase-3-proactive-agent-spec.md](../design/phase-3-proactive-agent-spec.md)
