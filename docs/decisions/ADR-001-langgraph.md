# ADR-001: Use LangGraph

**日期**: 2026-07-18
**更新**: 2026-08-04（补充实施后的设计决策与经验）

**状态**: 已接受

## Context

系统需要构建支持状态管理、多步骤执行、Tool Calling、Memory 的 AI Agent。选择编排框架时，需要在 LangChain Agent、裸 async 循环、LangGraph 三者之间做出选择。

## Decision

使用 **LangGraph** 作为 Agent 编排框架。手动构建 `StateGraph` ReAct 循环，单 Agent 架构，不使用任何预建 Agent（`create_react_agent` 已废弃，`create_agent` 禁止引入）。

## Alternatives Considered

### LangChain Agent

已拒绝。黑盒程度高，状态控制能力不足，调试困难。具体而言：内部状态不可见，自定义节点插入困难，错误恢复路径不透明。

### 裸 async while 循环

没有采用。当前 Agent 流程本质是一个 while 循环——调 LLM → 检查 tool_calls → 执行 tool → 回到 LLM。裸循环完全可以胜任当前需求，且更简单。

保留 LangGraph 是为以下场景预留入口：

| 未来场景 | LangGraph 能力 | 何时启用 |
|---------|---------------|---------|
| 记忆冲突人工审批 | `interrupt()` / `Command(resume=…)` | 已启用——`check_approval_node`、`check_conflict_node` |
| 对话持久化 | `PostgresSaver` / LangGraph checkpoint | 已启用——`thread_id` 跨轮次上下文恢复 |
| 多步自主工作流 | 条件边 + 并行节点 | Phase 3 主动 Agent 启用 |
| 进度流式输出 | `graph.astream()` | 已启用——SSE 流式聊天 |

### create_react_agent（LangGraph 预建）

LangGraph 曾提供 `create_react_agent` 预建函数。已废弃且不采用。EMA 手动构建 StateGraph 以获得对节点行为、路由逻辑、错误处理的完全控制。

## Architecture

### Graph Structure

```
START → call_llm ──(无 tool_calls)──→ generate_final → END
         │
         └──(有 tool_calls)──→ check_approval ──→ tools ──→ check_conflict
                                   │                            │
                                   └──(拒绝)────────────────────┘
                                                              │
                                   call_llm ←─────────────────┘
```

五个节点：

| 节点 | 职责 | 路由 |
|------|------|------|
| `call_llm` | 将对话历史 + tool schema 发给 LLM，返回 AIMessage（含 tool_calls） | `tools_condition` → check_approval 或 generate_final |
| `check_approval` | HITL 门：写/摄入工具执行前暂停，等待用户审批。读工具直接放行 | 通过 → tools；拒绝 → call_llm（注入拒绝 ToolMessage） |
| `tools` | `ToolNode` 自动执行 tool_calls 并产生 ToolMessage | → check_conflict |
| `check_conflict` | HITL 门：write_memory 返回 conflict 时暂停，等待用户选择 resolution | → call_llm |
| `generate_final` | 从 ToolMessage 中提取检索上下文，调用 LLM 生成最终回答 | → END |

### State

使用 `AgentState(TypedDict)`，字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| `messages` | `Annotated[list[BaseMessage], add_messages]` | 对话历史，ID 去重 |
| `step_count` | `int \| None` | ReAct 循环步数，配合 MAX_AGENT_STEPS 限流 |
| `final_response` | `str \| None` | 最终答案，generate_final_node 写入 |
| `final_prompt` | `list[dict] \| None` | 最终 LLM 调用的 prompt，SSE 流式层读取 |
| `error` | `str \| None` | 错误状态，任一节点捕获异常时写入 |
| `pending_approval` | `dict \| None` | HITL 暂停载荷 |

### AgentState 设计原则

**只用 `messages` 传递工具结果，不用离散 state 字段。** @tool 函数通过 `ToolNode` 只能写入 `messages`，不能直接写 state 的其他字段。`generate_final_node` 从消息历史中提取 `ToolMessage` 作为上下文——这比维护专门的 `retrieved_chunks`/`retrieved_memories` 字段更健壮，且对任意 tool 通用。

## Design Decisions Within LangGraph

### 为什么 Tool 返回 string

`ToolNode` 的标准约定是 `ToolMessage.content = str`，LLM 通过它读取工具结果。结构化 `list[dict]` 需要额外的 state channel，复杂度增加但没有实际收益。

EMA 的工具返回 JSON 字符串，内含 `display`（给 LLM 看的）和 `sources`（给前端展示的）。`generate_final_node` 解析 JSON envelope 提取 display 文本喂给 LLM，前端 API 层提取 sources 渲染可点击的来源链接。

### 为什么不做意图分类

LLM 通过 tools 自主决定调用哪个 tool。添加分类器只会增加一个出错点，不增加能力。Agent 看到的是 6 个 tool 的 schema——LLM 自己判断该调哪个。

### 为什么 max_steps 是环境变量而非硬编码

`MAX_AGENT_STEPS` 通过 `.env` 注入，默认 5。信任 LLM 自然停止——大多数对话在 1-2 步内结束（搜记忆 → 回答）。步数限制是安全网，不是策略。

### 为什么 check_approval 在 tools 之前

LangGraph 常规模式是 `call_llm → tools → call_llm`。EMA 在中间插入 `check_approval`：读操作（search/retrieve）直接放行到 tools，写操作（write/ingest）暂停等待审批。这保证了用户对任何内存修改都有最终话语权。

### check_approval 拒绝时的处理

不用 `Command(goto="__end__")` 终止，而是注入一条 `[REJECTED]` ToolMessage 并回到 `call_llm`。这样 LLM 知道操作被拒绝，可以解释原因或建议替代方案——而不是静默吞掉拒绝。

### check_conflict 为什么单独成节点

`check_conflict` 在 `tools` **之后**——冲突只有在 tool 执行完（write_memory 返回了 conflict 结果）才能检测到。分离成独立节点让审批流（事前）和冲突流（事后）的边界清晰，各管各的。

### HITL 的容错设计

当 `tools_condition` 匹配到 tool_calls 但找不到对应的 AIMessage（边界情况），不抛异常，注入一条占位 AIMessage 并路由回 `call_llm`。这是防御性编程——LangGraph 的状态管理很可靠，但 EMA 不假设它永远不会出边界错误。

## Tool System

6 个 tool，分层清晰：

| 类别 | Tool | 操作类型 | HITL |
|------|------|---------|------|
| 检索 | `search_memories_tool` | 读 | 直接放行 |
| 检索 | `retrieve_chunks_tool` | 读 | 直接放行 |
| 写入 | `write_memory_tool` | 写 | 审批 + 冲突检测 |
| 提取 | `extract_memory_tool` | 读（不持久化） | 直接放行 |
| 摄入 | `ingest_git_repo_tool` | 写 | 审批 |
| 摄入 | `ingest_document_tool` | 写 | 审批 |

Tool 与 Agent 解耦——每个 tool 只是现有 backend/service 函数的 `@tool` 装饰器包装，零逻辑重复。Tool 可以独立测试、独立替换，不依赖 Agent 层。

## Consequences

### 已确认的收益

- ✅ **Workflow 清晰**：5 个节点 + 2 个 HITL 门，每个节点职责单一
- ✅ **状态显式管理**：AgentState 字段含义明确，checkpointer 自动持久化
- ✅ **HITL 工作**：`interrupt()` / `Command(resume=…)` 实现了审批暂停、冲突仲裁两个独立的人机协同流程
- ✅ **SSE 流式**：`graph.astream()` 输出逐节点事件，前端展示 ReAct 循环过程
- ✅ **对话持久化**：InMemorySaver 支持跨轮次 thread_id 上下文恢复
- ✅ **错误恢复**：节点内异常不终止图执行，error 字段 + 降级响应
- ✅ **Tool 解耦**：Tool 可独立测试，新增 tool 只需注册到 ALL_TOOLS 列表

### 已知代价

- ⚠️ **比裸循环重**：当前 ReAct 循环本质是一个 while 循环，LangGraph 的大部分能力（条件边、并行节点、子图）尚未用到。裸循环的代码量更少、理解门槛更低
- ⚠️ **概念负担**：StateGraph、Command、interrupt、add_messages reducer——这些概念对首次接触的开发者需要学习成本
- ⚠️ **PostgresSaver 未启用**：Windows ProactorEventLoop 与 asyncpg 的兼容性问题导致 PostgresSaver 回退到 InMemorySaver，对话在服务重启后丢失。这是部署问题而非架构问题，但实际体验打折扣
- ⚠️ **LangChain 依赖**：`@tool` 装饰器、`ToolNode`、`BaseMessage` 来自 LangChain，不是 LangGraph 核心。未来 LangGraph 脱离 LangChain 独立演进时，这些依赖可能需要迁移

### 不会做的事

- **不引入 Multi-Agent 架构**：EMA 永久保持单 Agent。需要不同行为时换 System Prompt，不换 Agent——这是 Phase 4 垂直场景的设计基础
- **不使用 LangChain Agent / create_agent**：任何 LangChain 黑盒都不进入项目
- **不提前启用并行节点**：等到 Phase 3 主动 Agent 真正需要"同时检索多个来源 + 同时推理多条线索"时再启用

## References

- [Agent 设计文档](../agent-design.md) — 节点实现细节、System Prompt 设计
- [ADR-006: 扩展路线图](ADR-006-extension-roadmap.md) — Phase 3 主动 Agent 将启用更多 LangGraph 能力
- [LangGraph 文档](https://langchain-ai.github.io/langgraph/) — 上游框架参考
