# ADR-011: 广度层默认关闭（connectors / scenarios / patrol）

**日期**: 2026-08-11

**状态**: 已接受

## 背景

EMA 的广度层——PingCode/CI/飞书连接器与 webhook 入口、4 个垂直场景（复盘/审查/Onboarding/技术债）、自主巡检调度器——按 [ADR-006](./ADR-006-extension-roadmap.md) 的 Phase 2/3/4 路线图实现完毕。但这些模块服务的输入源目前都不存在：**生产记忆库为空（`memories` 表 0 行），连接器没有真实数据源，巡检每周空跑扫描空库**。

现状下这些模块的 API 路由全部无条件挂载（`backend/api/router.py` 9 个 router 无差别 include），connectors 在 lifespan 里无条件注册，patrol 调度器默认启动。也就是说：**项目在为不存在的输入运行完整的管道，活跃面（可被调用/可被执行的入口）覆盖了所有模块，而非核心闭环**。

## 决策

**把广度层挂到 `*_ENABLED` flag 后面、默认关闭**，活跃面收敛到核心闭环（chat + 记忆读写 + 实体 + 冲突仲裁 + 用量观测）：

| 模块 | Flag | 默认 | 生效位置 |
|------|------|------|---------|
| 连接器 / webhook | `CONNECTORS_ENABLED` | `false` | router 不挂载 webhook/connector 路由；main lifespan 不注册 connector |
| 垂直场景 | `SCENARIOS_ENABLED` | `false` | router 不挂载 scenario 路由 |
| 巡检调度 | `PATROL_ENABLED` | `false`（原 `true`） | main lifespan 不启动调度器；router 不挂载 patrol 路由 |

**测试豁免**：`APP_ENV=test` 时三个 flag 视为开启（config 的 `*_active` 属性折叠 `app_env == "test"`），API 测试套件继续完整驱动广度层路由——与 auth / 限流的 test 豁免惯例一致。

**不删代码**：模块、路由、连接器实现、调度器全部保留在仓库，通过 `*_ENABLED=true` 逐个启用。

## 理由

1. **不为不存在的输入暴露入口**：路由未挂载时返回 404（入口不存在），而非 401（入口存在但拒绝）——"未启用"比"已启用但空转"更真实。前端对未启用模块显示错误态即"该功能不可用"，无需额外业务逻辑。
2. **避免空跑烧钱**：patrol 每周空跑会调真实 LLM（空库扫描仍有 LLM 调用），默认关闭直接消除。
3. **活跃面 = 可维护面**：默认只挂核心闭环后，攻击面、路由表、启动副作用全部收窄；广度层代码仍是**编译进进程**的（import 存在），不是死代码——与"简单优先"约束一致：不做删除，做收敛。
4. **决策纪律一致**：项目每个决策都是"评估驱动 + ADR 记录"。生产空库是实测事实（`gap-remediation.md` §3.1），据此收敛活跃面是同样的"先测量、再决策"。
5. **可逆**：flag 是旋钮，任一模块有真实数据源时 `*_ENABLED=true` 即恢复，无需回滚。

## 代价与保留

以下代价已认知并接受：

- **前端部分页面 off 时显示错误态**（连接器/巡检/场景页在 flag 关闭时调接口拿 404）。当前前端启动不拉这些 API（[App.tsx](../../frontend/src/App.tsx) 仅定义路由），挂载后才调用；导航按探活隐藏属可选增强，不并入本次。
- **`PATROL_ENABLED` 默认值变更影响部署**：原默认 `true`，显式设置过该变量的环境不受影响；依赖默认值的环境行为改变（调度器不再自启）。`.env.example` 与 `deployment.md` 同步更新。
- **webhook 关闭后冲突队列的输入收窄**：`conflict_router` 恒挂载（冲突仲裁是核心闭环），但 webhook 端点 off 后"来自连接器的冲突入队"入口不存在，冲突只能走 agent 路径——这正是收敛意图。
- **连接器注册随 flag 门控**：flag 关闭时 `CONNECTOR_REGISTRY` 为空，但 `notify_feishu_tool` 走 `config.feishu_webhook_url` 不经 registry，不受影响。

## 拐点

当任一模块出现**真实数据源或真实用户**时，逐个评估并开启对应 flag：

1. 有团队接入连接器（PingCode/CI/飞书推送真实事件）→ `CONNECTORS_ENABLED=true`
2. 知识库积累到场景值得跑（复盘/审查/Onboarding 有真实输入）→ `SCENARIOS_ENABLED=true`
3. 记忆库有真实内容、巡检有价值 → `PATROL_ENABLED=true`（同时恢复默认值讨论）

开启时机由"有没有真实输入"决定，不由"功能是否实现完毕"决定。

## 后果

- `backend/shared/config.py` 新增 `connectors_enabled` / `scenarios_enabled`（默认 `false`），`patrol_enabled` 默认改 `false`；新增 `connectors_active` / `scenarios_active` / `patrol_active` 属性（含 test 豁免）
- `backend/api/router.py`：核心闭环（memory/agent/entity/conflict/usage）恒挂载；webhook+connector / scenario / patrol 路由按 flag 条件挂载
- `backend/main.py`：connector 注册与 patrol 调度器按 `*_active` 门控
- `docs/architecture.md` / `docs/deployment.md` / `.env.example` 同步 flag 与默认值
- 对应测试：`tests/unit/test_config.py` 新增 `TestBreadthLayerFlags`（默认关闭 / test 豁免 / 显式开启三态）
