# ADR-008: 冲突解析统一为单流水线（peer_id 区分巡检仲裁）

**日期**: 2026-08-11

**状态**: 已接受

## 背景

记忆冲突有三条来源，但解析端曾经维护着两条独立流水线：

| 来源 | 入队/入口 | 原解析器 |
|------|-----------|---------|
| Agent 交互 | `check_conflict_node` → `interrupt()`（不入队） | `resolve_conflict` |
| Webhook / 连接器 | `persist_pending_conflict` | `resolve_conflict` |
| Patrol 巡检 | `persist_patrol_conflict`（`conflict_type='patrol'`） | `resolve_patrol_conflict` |

关键观察：**入队端已经统一了 payload 形状**。`persist_patrol_conflict` 会把败方 B 的完整内容打包成与 ingestion 冲突一模一样的 `_deferred` 载荷（`conflicts.py`），也就是说一条 patrol 矛盾入队后，本质上就是"一条带了 `peer_id` 的 ingestion 冲突"。但解析端却保留了两条流水线：`resolve_patrol_conflict` 约 250 行，把 keep_existing / overwrite / merge 四个分支各自实现了一遍，与 `resolve_conflict` 的对应分支逐行重复（读 A 的 meta + content_hash → 写 A → 捕获 `IntegrityError` → 重查 winner 返回 duplicate）。

两条流水线的**唯一实质差异**是：patrol 的败方 B 已经在库，解析时要在同一事务内先软删 B，再让 A 接管 B 的 content_hash——因为 `(content_hash) WHERE deleted_at IS NULL` 部分唯一索引按语句逐条校验，B 必须先消失 A 才能采用它的 hash。

## 决策

**把解析统一为单一 `resolve_conflict()`**（`backend/service/memory.py`），给函数增加可选参数 `peer_id: str | None = None`：

- `peer_id` 为 `None`（默认）：ingestion / webhook / agent 语义，行为与之前完全一致，含未知 resolution 宽容降级到 keep_existing。
- `peer_id` 非空（patrol 矛盾，A = `existing_id` 存活、B = `peer_id` 落败）：
  - 解析前先做存活方 A 的 liveness 检查，防止对已删除的记忆仲裁
  - keep_existing / overwrite / merge 在同一事务里先软删 B、再写 A（B 必须先删，约束同上）
  - overwrite / merge 写事务带 `rowcount != 1` 守卫，堵住"A 在检查与写入之间被删"的竞态
  - keep_both 为 no-op（两边都留），resolved 的队列记录即仲裁台账

**删除整个 `resolve_patrol_conflict`**（`backend/service/conflicts.py`）。`resolve_pending_conflict` 不再按 `conflict_type` 分派，直接 `resolve_conflict(resolution, existing_id, deferred, peer_id=...)` 一个出口。`conflict_type` 列保留，但**降级为显示 / 过滤标签，不再参与路由**。

## 理由

1. **同一决策逻辑写两份，双份正确性负担**：软删顺序、content-hash 竞态、prior_hashes 语义在两条流水线里各自正确——任何一处修复都必须同步另一处，否则两条路径行为漂移。合并后这套最微妙的部分只存在一份。

2. **入队端已经统一，解析端分叉没有数据基础**：`persist_patrol_conflict` 产出的 `_deferred` 与 ingestion 相同，`peer_id` 只是附加信息。分叉是历史演进（patrol 先做、复用后做）的产物，不是需求差异。

3. **`peer_id` 参数是纯增量**：默认 `None` 时 ingestion / agent / webhook 三路径零行为变化，patrol 语义完全由参数承载。不需要新的抽象、类或路由。

4. **附带的 DB 往返减少**：patrol 的 merge 解析从 5 次 DB 调用降为 4 次——原来 liveness 检查 + meta 预读是两次 SELECT，合并后存活检查一次即可（overwrite / merge 的 meta 读取在写前查询中完成）。

## 代价与保留的机器

以下正确性约束**不是过度设计**，合并后依然保留：

- **B 先软删、A 后接管 hash 的事务顺序**：live-content-hash 部分唯一索引按语句校验，顺序错误会产生违反约束的写入。
- **`IntegrityError` → 重查 winner 的竞态处理**：内容在排队期间被并发写入时，解析不 500，而是报告已存储的那条。
- **peer 形状的存活方 liveness 检查 + 写事务 rowcount 守卫**：防止对已删除记忆的过期仲裁。
- **keep_both 的双重语义**（ingestion = 插入新侧，patrol = 两边都留）：这是保留下来的**语义分叉**，靠 `peer_id` 区分，不再靠 `conflict_type`。它是 reopen 机制存在的原因——记住这个关联，评估删除 reopen 时必须一并处理。

## 拐点与后续

- **Tier 2（未实施，候选）**：patrol 的 keep_both 语义本质上等价于 findings 的 dismiss——不是真矛盾，两个都留。已有 `POST /api/patrol/findings/{log_id}/dismiss` 通道。若把 patrol 仲裁改走 finding-dismiss，可让 patrol 彻底离开 `pending_conflicts` 队列，届时可删 `reopen_patrol_conflict`、LEAST/GREATEST 成对去重索引、already-arbitrated 检查。这动 schema 与前端，单独评估，不并入本次。
- 当 patrol 需要比"四选项"更丰富的仲裁语义（如部分采纳、逐条合并）时，重开此决策——届时队列模型可能不再适用。

## 后果

- `backend/service/conflicts.py` 删除约 250 行（`resolve_patrol_conflict`），`conflicts.py` 从 ~770 行降到 ~500 行
- 冲突解析只有一个入口 `resolve_conflict`，`peer_id` 参数区分两种语义；`conflict_type` 仅作显示 / 过滤
- `docs/memory-system.md` 的巡检矛盾描述同步为"解析走与 ingestion 相同的 `resolve_conflict()`（以 `peer_id=B` 传入）"
- 对应测试重写：`tests/unit/test_patrol_conflicts.py` 的 `TestResolvePatrolConflict` 改为测共享函数，`tests/api/test_conflict_routes.py` 的 keep_both 集成用例改调 `resolve_conflict(..., peer_id=...)`
