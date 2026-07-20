# Agent Design

## 架构

手动构建 `StateGraph` ReAct 循环 —— 不使用任何预建 Agent（`create_react_agent` 已废弃，`create_agent` 禁止引入）。

```
START → call_llm ──(有 tool_calls)──→ tools ──→ call_llm (循环)
         │
         └──(无 tool_calls)──→ generate_final → END
```

三个节点：

| 节点 | 实现 | 职责 |
|------|------|------|
| `call_llm` | `agent/nodes.py` | 将对话历史 + tool schema 发给 LLM，解析返回的 `AIMessage`（含 `tool_calls`，如有） |
| `tools` | `ToolNode(tools, handle_tool_errors=True)` | LangGraph 内置，自动执行 tool_calls 并产生 `ToolMessage`；执行失败返回错误 `ToolMessage`，不终止图执行 |
| `generate_final` | `agent/nodes.py` | 从 `ToolMessage` 中提取检索上下文，调用 LLM（无 tools）生成最终自然语言回答 |

路由：`tools_condition`（LangGraph 内置）—— AIMessage 有 `tool_calls` 则去 `tools`（形成循环），无则去 `generate_final`（终止）。

## 设计决策

### 为什么用 LangGraph

当前 Agent 流程是一个简单的 while 循环，LangGraph 确实比它需要的东西更重。保留 LangGraph 是为以下场景预留入口：

| 未来场景 | LangGraph 能力 | 何时启用 |
|---------|---------------|---------|
| 记忆冲突人工审批 | `interrupt()` / `Command(resume=...)` | 当 `write_memory()` 检测到矛盾时暂停并等待用户确认 |
| 对话持久化 | `PostgresSaver` | 替换 `InMemorySaver`，利用已有 PostgreSQL 实现跨重启对话恢复 |
| 多步自主工作流 | 条件边 + 并行节点 | 当需要"摄取→检索→分析→写入"的多阶段管线时 |
| 进度流式输出 | `graph.astream()` | 接入 Streamlit 前端展示实时进度 |

### 为什么不做意图分类

LLM 通过 tools 自主决定调用哪个 tool。添加分类器只会增加一个出错点，不增加能力。

### 为什么不做 `_max_iter_guard`

信任 LLM 自然停止。`.env` 中 `MAX_AGENT_STEPS=10` 备用。如果实际运行中出现循环，再在路由中加迭代计数限制。

### 为什么 Tool 返回 string

`ToolNode` 的标准约定是 `ToolMessage.content = str`，LLM 通过它读取工具结果。结构化 `list[dict]` 需要额外的 state channel，复杂度增加但没有实际收益。

### 为什么工具结果从 ToolMessage 而不是 state 字段读取

`@tool` 函数通过 `ToolNode` 只能写入 `messages`，不能直接写 state 的其他字段。`generate_final_node` 从消息历史中提取 `ToolMessage` 作为上下文——比维护专门的 `retrieved_chunks`/`retrieved_memories` 字段更健壮，且对任意 tool 通用。

## 文件结构

```
agent/
  state.py    # AgentState TypedDict (messages, final_response, error)
  tools.py    # 6 个 @tool 薄封装 → 调用 backend/service/
  nodes.py    # call_llm_node, generate_final_node
  graph.py    # build_agent_graph(), get_default_agent()

backend/
  api/routes/agent_routes.py    # POST /api/agent/chat
  service/agent_service.py      # get_agent(), get_agent_for_thread()
```

## 关键约束

- **不引入 LangChain Agent**：手动 `StateGraph`，不经过 `create_agent` 或任何 LangChain 黑盒
- **Tool 与 Agent 解耦**：Tool 只是现有 service 函数的 `@tool` 包装，零逻辑重复
- **LLM 调用通过 `LLMProvider` 抽象接口**：节点内调用 `get_llm_provider().chat_raw()` 和 `get_llm_provider().chat()`，不直接依赖 `openai`/`anthropic`
- **异步优先**：所有节点和 tool 都是 `async` 函数
