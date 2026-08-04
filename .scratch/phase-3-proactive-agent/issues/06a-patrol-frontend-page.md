# 06a — 巡检前端页面

**What to build:** 前端新增 Patrol 页面（`/patrol` 路由），包含巡检历史列表（分页，显示 patrol_type、trigger、status、时间、finding 数量）、单次巡检详情视图（findings 按类别分组展示，含 dismiss 按钮）、手动触发按钮（下拉选择 patrol_type 和 scope）。Finding 按类别（pattern_match / knowledge_gap / new_entity / contradiction / decay_alert）分组渲染，每组不同颜色标识。Finding dismiss 操作即时反馈 UI（按钮 disabled，无需刷新页面）。空状态（无巡检记录）和加载态（skeleton）均有处理。

**Blocked by:** 02（需要 Patrol API 端点）

**Status:** ready-for-agent

- [ ] `frontend/src/types/index.ts` 新增 Patrol 相关类型：`PatrolType`（'daily' | 'weekly' | 'event_driven' | 'manual'）、`PatrolTrigger`（'cron' | 'webhook' | 'manual'）、`PatrolLog`（id, patrol_type, trigger, status, findings, dismissed_findings, started_at, completed_at）、`PatrolFinding`（id, type, title, description, severity, related_memory_ids）
- [ ] `frontend/src/api/patrol.ts`：API 客户端函数 `triggerPatrol(patrol_type, scope?)` → POST /api/patrol/trigger、`listPatrolLogs(params?)` → GET /api/patrol/logs、`getPatrolLog(id)` → GET /api/patrol/logs/{id}、`dismissFinding(logId, findingId)` → POST /api/patrol/findings/{id}/dismiss
- [ ] `frontend/src/pages/PatrolPage.tsx`：页面组件，包含三个区域——(1) 顶部操作栏：手动触发按钮 + patrol_type 下拉（每日巡检/每周巡检/矛盾扫描）+ scope 输入；(2) 左侧/主区域巡逻日志列表（分页，点击行选中）；(3) 右侧/展开区域 selected patrol detail（findings 分组渲染）
- [ ] Finding 卡片组件：按类别分组（pattern_match/knowledge_gap/new_entity/contradiction/decay_alert），每组不同颜色左边框（红/黄/蓝/橙/灰），每条展示 title + description，右侧 dismiss 按钮
- [ ] Dismiss 按钮行为：点击 → API 调用 → 成功后按钮变灰 + "已忽略"文字，失败 toast 提示。不刷新页面，不重新拉取列表
- [ ] 空状态：无巡检记录时显示 "暂无巡检记录，点击上方按钮手动触发一次巡检" 提示
- [ ] 加载态：列表加载中显示骨架屏（skeleton），详情加载中显示 loading spinner
- [ ] `frontend/src/App.tsx` 注册 `/patrol` 路由，lazy import PatrolPage
- [ ] 前端测试：`frontend/src/pages/PatrolPage.test.tsx` — 列表渲染（Mock API 返回多条记录）、分页交互（点击下一页）、手动触发按钮点击后 API 调用、dismiss 按钮点击后按钮 disabled、空列表 empty state 展示、loading skeleton 渲染
