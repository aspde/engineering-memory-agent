# Agent Design

## 架构

手动构建 `StateGraph` ReAct 循环 —— 不使用任何预建 Agent（`create_react_agent` 已废弃，`create_agent` 禁止引入）。

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

| 节点 | 实现 | 职责 |
|------|------|------|
| `call_llm` | `backend/agent/nodes.py` | 将对话历史 + tool schema 发给 LLM，解析返回的 `AIMessage`（含 `tool_calls`，如有） |
| `check_approval` | `backend/agent/nodes.py` | Human-in-the-Loop：写工具执行前暂停等待用户审批。审批集合由 `approval_required_tools` 参数化：默认集写/摄入供自动化流程（巡检/场景）自主执行；交互式 chat 路径用 `CHAT_APPROVAL_TOOLS`，额外把 `notify_feishu_tool`（外发到团队飞书群）纳入审批 |
| `tools` | `ToolNode(tools, handle_tool_errors=True)` | LangGraph 内置，自动执行 tool_calls 并产生 `ToolMessage` |
| `check_conflict` | `backend/agent/nodes.py` | Human-in-the-Loop：检测记忆冲突，暂停等待用户选择解决方案 |
| `generate_final` | `backend/agent/nodes.py` | 从 `ToolMessage` 中提取检索上下文，调用 LLM（无 tools）生成最终回答；本轮无工具结果（纯聊天）时直接复用 `call_llm` 输出，不重复调用 LLM |

路由：`tools_condition`（LangGraph 内置）—— AIMessage 有 `tool_calls` 则进入 `check_approval`（HITL 审批），无则去 `generate_final`（终止）。

## 设计决策

### 为什么用 LangGraph

当前 Agent 流程是一个简单的 while 循环，LangGraph 确实比它需要的东西更重。保留 LangGraph 是为以下场景预留入口：

| 未来场景 | LangGraph 能力 | 何时启用 |
|---------|---------------|---------|
| 记忆冲突人工审批 | `interrupt()` / `Command(resume=...)` | 当 `write_memory()` 检测到矛盾时暂停并等待用户确认 |
| 对话持久化 | `PostgresSaver` | 已实现，利用已有 PostgreSQL 实现跨重启对话恢复 |
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

### 为什么上下文窗口按 token 预算而不是消息条数

发送给 LLM 的对话历史由 `context_token_budget`（默认 12000，`CONTEXT_TOKEN_BUDGET` 可调）约束，而不是旧的固定 12 条窗口。长消息自动收窄窗口、短消息放宽，避免"12 条都超长逼近上下文上限"的硬编码问题。token 数量用 `_estimate_tokens` 估算：tiktoken（`o200k_base`）可用时按真实 BPE 计数，更接近各 provider 的实际计费；tiktoken 取不到编码时（离线环境）回退到 CJK 感知的启发式（中文 ~1 token/字符、ASCII 字母数字 ~4 字符/token、ASCII 标点符号按 ~2 字符/token 加权——代码/JSON 里符号密集，低估会溢出预算）。对话压缩（`CONVERSATION_COMPACTION_ENABLED`）的触发条件同样是预算：超预算的早期历史被折叠为一条 running-summary，保留尾部约占预算 60%，压缩调用自身的输入也被截断（`_COMPACTION_TRANSCRIPT_CHARS`），防止溢出历史把压缩调用本身撑爆。压缩的 transcript 除对话文本外还纳入被窗口化的工具结果（`tool (名称): 内容`，按 display 解包并截断），这样早期检索回合的摘要不会丢掉工具返回的记忆/文档上下文。压缩摘要按"prompt 版本 + 逐字 transcript"有界记忆化（加锁收口）：工具回合会两次界定量（`call_llm_node` 与 `generate_final_node`），相同的溢出前缀复用同一次压缩结果，既省一次 LLM 调用，也让同一回合内两处注入的摘要一致。

## 文件结构

```
backend/
  agent/
    state.py    # AgentState TypedDict (messages, final_response, final_prompt, error, pending_approval)
    tools.py    # 9 个 @tool 薄封装 → 调用 backend/service/
    nodes.py    # call_llm_node, check_approval_node, check_conflict_node, generate_final_node
    graph.py    # build_agent_graph(), get_default_agent()
  api/routes/agent_routes.py    # POST /api/agent/chat
  service/agent_service.py      # get_agent(), get_agent_for_thread()
```

## 关键约束

- **不引入 LangChain Agent**：手动 `StateGraph`，不经过 `create_agent` 或任何 LangChain 黑盒
- **Tool 与 Agent 解耦**：Tool 只是现有 service 函数的 `@tool` 包装，零逻辑重复
- **LLM 调用通过 `LLMProvider` 抽象接口**：节点内调用 `get_llm_provider().chat_raw()` 和 `get_llm_provider().chat()`，不直接依赖 `openai`/`anthropic`
- **异步优先**：所有节点和 tool 都是 `async` 函数
- **简单优先**：当前 ReAct 循环本质是一个 while 循环，LangGraph 的大部分能力尚未用到。保留 LangGraph 仅是为人机协同审批 (`interrupt`)、对话持久化 (`PostgresSaver`) 等未来场景预留入口，不做提前设计
