# Phase 4: 垂直场景孵化 — 功能规格

## Problem Statement

Phase 1-3 建好了三样东西：实体关系图谱（理解知识的关联）、多源连接器（持续接入外部数据）、主动 Agent（定时巡检+事件响应）。这些是通用基础设施。但团队的实际使用场景不是"查询通用基础设施"——而是"帮我写复盘"、"审查这个 PR"、"带新人了解项目"、"找出哪些临时方案该清理了"。

每个场景本质上都是把已有的检索、分析、推理、写入能力**按特定流程和格式组装起来**。如果没有这些预组装，用户需要在聊天框里手写长 prompt 来告诉 EMA 怎么做事——这就像有了乐高积木但每次都要从零开始搭。

Phase 4 的目标不是增加新能力，而是**为高频场景预组装解决方案**。

## Solution

新增四个垂直场景——故障复盘、代码审查、新人 Onboarding、技术债雷达。每个场景 = 一个专用 System Prompt + 一个 compose 函数 + 一个前端入口。底层能力全部复用已有工具，架构零变更。

场景按需孵化——哪个场景被团队成员先提出、先验证、先打磨，就先上线哪个。同时建立场景孵化的轻量流程：有人要 → 检查是否需要新能力 → 不需要就直接写 prompt → 上线 → 观察使用 → 有用就留、没用就删。

## User Stories

### 故障复盘 Agent

1. As an on-call engineer, I want EMA to automatically generate a postmortem draft when a Jira incident issue is marked "Resolved", so that I don't start the postmortem from a blank page.
2. As an on-call engineer, I want the postmortem draft to include a timeline assembled from Jira timestamps + CI failure logs + related Slack discussions, so that I don't have to manually piece together the chronology.
3. As an on-call engineer, I want the postmortem draft to include a "similar incidents" section — matching the current incident to historical ones by affected entities and symptom patterns — so that I can identify recurring root causes.
4. As an on-call engineer, I want the postmortem draft to suggest potential root causes based on the code diff of the fix commit + entity relation graph (e.g., "the changed file DBConfig.java has been involved in 2 previous connection-pool incidents"), so that the analysis is grounded in project history.
5. As a team, I want the finalized postmortem to be saved as a structured memory linked to all related entities, so that it enriches the knowledge base for future queries.

### 代码审查助手

6. As a developer reviewing a PR, I want EMA to analyze the changed files and tell me what I should watch out for, based on historical faults and decisions associated with those files, so that I don't miss a known risk that the author and I may have both forgotten.
7. As a developer reviewing a PR, I want EMA to flag when the changed file has been involved in past incidents — "⚠️ DBConfig.java: changed the connection pool config — this file was involved in 2 production incidents in the last 6 months" — so that high-risk changes get extra scrutiny.
8. As a PR author, I want EMA to check whether the PR's stated goal (from the PR description) aligns with the historical decision record — "this PR proposes a microservice split, but we have ADR-007 documenting a decision to consolidate" — so that contradictions with past decisions are surfaced before merge.
9. As a developer, I want the review context to be delivered as a comment on the PR (via GitHub/GitLab API) or as a Slack message, so that I get it in my existing workflow without switching to EMA's Web UI.

### 新人 Onboarding 向导

10. As a new team member, I want EMA to give me a structured overview of the project — core modules ranked by memory count and incident history, key architectural decisions, and recent fault patterns — so that I can build a mental model without reading months of chat history.
11. As a new team member, I want EMA to recommend a reading order for onboarding documents, based on which documents are most referenced by other memories and which cover entities with the highest incident count, so that I read the most impactful material first.
12. As a new team member, I want EMA to answer "why was X done this way?" by tracing decisions back through the entity graph — "PostgreSQL was chosen over MySQL because..." with links to the decision memory and the ADR — so that I understand the reasoning, not just the outcome.
13. As a team lead, I want the onboarding guide to be refreshable — "generate a fresh overview for the new hire starting next week" — so that it reflects the current state of the project, not a stale document written six months ago.

### 技术债雷达

14. As a tech lead, I want EMA to produce a weekly report of unresolved temporary solutions — memories marked with "workaround" or "temporary" that are older than 3 months with no follow-up — so that shortcuts don't become permanent.
15. As a tech lead, I want EMA to flag documentation gaps — modules associated with many memories but zero formal documentation — so that I know where tribal knowledge is most concentrated and most fragile.
16. As a tech lead, I want EMA to detect when a temporary solution has been replaced by a proper fix (matching a new commit memory against the workaround memory) and automatically mark the workaround as resolved, so that the debt radar stays accurate without manual curation.
17. As a tech lead, I want the tech debt report to be shareable — "send this to #tech-debt in Slack" — so that the whole team sees it without another meeting.

### 场景孵化流程

18. As a team member, I want to be able to request a new vertical scenario by describing the workflow I want — "when I do X, EMA should Y" — and have EMA's keeper evaluate whether it needs new infrastructure or is just a prompt assembly.
19. As the EMA maintainer, I want scenarios that aren't used to be removable with zero side effects — delete the prompt file, delete the frontend entry, done — so that failed experiments don't leave cruft in the codebase.

## Implementation Decisions

### 场景架构

每个场景是一个自包含的 `.py` 文件，结构统一：

```
backend/service/scenarios/
  __init__.py          # 注册所有场景
  base.py               # Scenario 抽象（可选——四个场景足够简单，直接函数即可）
  postmortem.py         # compose_postmortem(incident_memory_id) → prompt + tools
  code_review.py        # compose_review_context(pr_diff, pr_description) → prompt + tools
  onboarding.py         # compose_onboarding_guide(scope) → prompt + tools
  tech_debt.py          # compose_tech_debt_report() → prompt + tools
```

每个 `compose_*()` 函数：
1. 接收场景参数（incident ID、PR diff、scope 等）
2. 从已有管线拉取上下文（search_memories、query_entity_relations）
3. 组装专用 System Prompt + 用户消息
4. 调用 `get_default_agent().ainvoke()` 执行
5. 返回格式化结果

**不改 `agent/graph.py`。不改已有 tool。不新增数据库表。** 场景是纯消费层。

### 场景注册

```python
# scenarios/__init__.py
# 不是插件系统——是一个显式的 dict
SCENARIOS: dict[str, dict] = {
    "postmortem": {
        "name": "故障复盘",
        "compose": "backend.service.scenarios.postmortem.compose_postmortem",
        "triggers": ["jira_issue_resolved", "manual"],
        "status": "active",
    },
    "code_review": {
        "name": "代码审查助手",
        "compose": "backend.service.scenarios.code_review.compose_review_context",
        "triggers": ["pr_opened", "manual"],
        "status": "active",
    },
    "onboarding": {
        "name": "新人 Onboarding",
        "compose": "backend.service.scenarios.onboarding.compose_onboarding_guide",
        "triggers": ["manual"],
        "status": "active",
    },
    "tech_debt": {
        "name": "技术债雷达",
        "compose": "backend.service.scenarios.tech_debt.compose_tech_debt_report",
        "triggers": ["weekly_patrol", "manual"],
        "status": "active",
    },
}
```

`status` 可以是 `active`（已上线）、`beta`（实验性，默认隐藏）、`inactive`（已下线，保留代码但不暴露）。切换 status 不需要删除代码。

### 与已有 Phase 的关系

每个场景复用的能力：

| 场景 | 用到 Phase 1（实体查询） | 用到 Phase 2（连接器数据） | 用到 Phase 3（主动触发） |
|------|----------------------|--------------------------|------------------------|
| 故障复盘 | ✅ 关联实体历史故障 | ✅ Jira + CI + Git 拼时间线 | ✅ Jira resolved 事件触发 |
| 代码审查 | ✅ 文件→实体→历史故障 | ✅ Git PR webhook | ✅ PR opened 事件触发 |
| Onboarding | ✅ 核心实体 + 关系全景 | — | — |
| 技术债雷达 | ✅ 临时方案→关联模块 | — | ✅ 每周巡检自动运行 |

### 前端

每个场景一个独立入口——不是场景间自动路由，是用户选择：

- 聊天页侧边栏增加"场景"快捷入口（下拉菜单或按钮组）
- 故障复盘：在故障详情页一键触发 "生成复盘草稿"
- 代码审查：PR 页面嵌入审查提示
- Onboarding / 技术债：独立页面或仪表盘卡片

### API

- `POST /api/scenarios/{name}/run` — 手动触发某个场景，传入场景参数
- `GET /api/scenarios` — 列出可用场景及状态

## Testing Decisions

### 测试原则

- 场景的 `compose()` 函数测试只验证：给定输入 → 生成正确的 prompt 模板和 tool 选择 → 调用 Agent。Mock Agent.ainvoke()
- Prompt 模板测试验证：模板包含必要的 instructions、output format、关键约束
- 不测试 Agent 的实际输出内容（那是 LLM 的行为，不是场景的逻辑）

### 测试模块

| 测试文件 | 内容 | 参考 |
|---------|------|------|
| `tests/unit/test_scenario_postmortem.py` | compose 逻辑、prompt 模板 | 新——纯逻辑 |
| `tests/unit/test_scenario_code_review.py` | compose 逻辑、prompt 模板 | 同上 |
| `tests/unit/test_scenario_onboarding.py` | compose 逻辑、prompt 模板 | 同上 |
| `tests/unit/test_scenario_tech_debt.py` | compose 逻辑、prompt 模板 | 同上 |
| `tests/api/test_scenario_routes.py` | /run 端点 + /list 端点 | test_memory_routes.py |
| `src/components/ScenarioPanel.test.tsx` | 场景入口渲染、选择 | 已有组件测试模式 |

### 测试用例速览

- `test_compose_postmortem_includes_timeline_section` — 复盘 prompt 包含时间线指令
- `test_compose_postmortem_includes_similar_incidents_section` — 包含相似故障指令
- `test_compose_code_review_flags_high_risk_files` — 审查 prompt 要求标记高风险文件
- `test_compose_onboarding_ranks_modules_by_memory_count` — Onboarding 按记忆数排序
- `test_compose_tech_debt_flags_workarounds_older_than_3m` — 技术债扫描时间阈值
- `test_scenario_registry_includes_all_four_scenarios` — 注册完整性
- `test_scenario_status_inactive_not_exposed_to_ui` — 非活跃场景不暴露
- `test_api_scenario_unknown_returns_404` — 未知场景处理
- `test_api_scenario_run_without_params_returns_422` — 参数校验

## Out of Scope

- 场景间自动路由/编排——用户自己选场景
- 场景配置 UI——先硬编码，场景数量 > 10 再考虑可视化配置
- 场景 Marketplace / 社区共享——不做。这些场景是团队内部的
- 场景 A/B 测试——不需要，场景不是产品功能，是实用工具
- 场景性能优化——每个场景一次 Agent 调用 + 若干 tool 调用，性能瓶颈在 LLM，不在场景层
- Agent 扮演角色切换——不引入"你是复盘专家"vs"你是代码审查专家"的角色分离。同一个 Agent，不同的 prompt

## Further Notes

- 本 spec 对应 [ADR-006](./decisions/ADR-006-extension-roadmap.md) 的 Phase 4
- 场景按需孵化——不提前建四个。先用"故障复盘"验证模式，其余三个陆续按需上线
- 场景可以失败——如果某个场景上线后没人用，删掉 prompt 文件即可，零残留。这和其他 Phase 的基础设施变更不同
- 场景是最接近用户的一层——如果某个场景的需求反馈需要新 tool 或新数据，那是回到 Phase 1-3 去补的，不是场景层的职责
- 场景文件 100-200 行，不引入新的抽象层次（Scenario 基类、注册机制复杂度 > 实际场景数）。四个场景，dict 注册足够
