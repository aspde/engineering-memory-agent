# 06 — 前端实体图谱页面

**What to build:** 新路由 `/graph`，交互式实体关系图谱可视化。搜索实体后以节点-边图展示一度关系，可点击展开、查看记忆详情。

**Blocked by:** 03 — 需要实体查询 API 可用。

**Status:** ready-for-agent

- [ ] 新增 `/graph` 路由，在 `App.tsx` 中注册，Sidebar 中增加导航入口
- [ ] `EntityGraph` 组件：Canvas 或 SVG 渲染节点-边图，中心节点为当前实体，周围为关联实体节点
- [ ] 实体搜索框：输入实体名 → 调用 `GET /api/entities/search` → 选择实体 → 以该实体为中心渲染图谱
- [ ] 点击节点展开：点击关联实体节点 → 以该实体为新的中心重新渲染（调用 `GET /api/entities/{id}/relations`）
- [ ] 点击记忆：点击实体与记忆之间的连线或记忆节点 → 展示记忆完整内容（弹窗或侧面板，调用 `GET /api/memory/memories/{id}`）
- [ ] 实体节点按类型区分颜色和大小（technology/person/project/decision/event/file/concept 各自不同样式）
- [ ] 连线标注关系类型（depends_on、causes、part_of、contradicts、supersedes、relates_to）
- [ ] 聊天页「在图谱中查看」链接：当 Agent 回复中出现实体名时，提供跳转到 `/graph?entity=PostgreSQL` 的链接
- [ ] 纯 Canvas/SVG 实现，不引入 D3 或任何图表框架依赖
- [ ] 新建 `src/api/entities.ts` API 模块（getEntity、getEntityRelations、searchEntities）
- [ ] 组件测试：EntityGraph 渲染、搜索交互、节点点击展开
