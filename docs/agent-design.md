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

### 为什么迭代计数限制是「到达上限即强制收束」

每个用户轮次的 ReAct 循环有硬性上限：`step_count` 达到 `MAX_AGENT_STEPS`（默认 5，`config.max_agent_steps`）时，`_make_route_after_call_llm`（`backend/agent/graph.py`）强制把路由导向 `generate_final`，而不是继续循环。`step_count` 在每一轮新用户消息到达时重置（`call_llm_node` 通过 `_is_new_user_turn` 归零），所以上限约束的是**单轮内的工具循环**，不跨轮次累积。

选择「到顶即收束」而不是「到顶报错」：LLM 偶尔会在复杂任务上多转几圈，强行报错会中断本来可以完成的回答；导向 `generate_final` 则把已拿到的工具结果收束成最终回答。默认 5 是交互场景的折中——大多数轮次 1-2 步即完成，5 步足以覆盖多工具链路（搜索 → 实体查询 → 写入），同时把单轮 LLM 调用成本控制在有界范围。自动化巡检（patrol）按类型放宽到 15/20 步（`backend/service/patrol.py` 的 `_PATROL_MAX_STEPS`），因为全量扫描需要的搜索步数远超交互轮次。

### 为什么 Tool 返回 string

`ToolNode` 的标准约定是 `ToolMessage.content = str`，LLM 通过它读取工具结果。结构化 `list[dict]` 需要额外的 state channel，复杂度增加但没有实际收益。

### 为什么工具结果从 ToolMessage 而不是 state 字段读取

`@tool` 函数通过 `ToolNode` 只能写入 `messages`，不能直接写 state 的其他字段。`generate_final_node` 从消息历史中提取 `ToolMessage` 作为上下文——比维护专门的 `retrieved_chunks`/`retrieved_memories` 字段更健壮，且对任意 tool 通用。

### 为什么上下文窗口按 token 预算而不是消息条数

发送给 LLM 的对话历史由 `context_token_budget`（默认 12000，`CONTEXT_TOKEN_BUDGET` 可调）约束，而不是旧的固定 12 条窗口。长消息自动收窄窗口、短消息放宽，避免"12 条都超长逼近上下文上限"的硬编码问题。token 数量用 `_estimate_tokens` 估算：tiktoken（`o200k_base`）可用时按真实 BPE 计数，更接近各 provider 的实际计费；tiktoken 取不到编码时（离线环境）回退到 CJK 感知的启发式（中文 ~1 token/字符、ASCII 字母数字 ~4 字符/token、ASCII 标点符号按 ~2 字符/token 加权——代码/JSON 里符号密集，低估会溢出预算）。对话压缩（`CONVERSATION_COMPACTION_ENABLED`）的触发条件同样是预算：超预算的早期历史被折叠为一条 running-summary，保留尾部约占预算 60%，压缩调用自身的输入也被截断（`_COMPACTION_TRANSCRIPT_CHARS`），防止溢出历史把压缩调用本身撑爆。压缩的 transcript 除对话文本外还纳入被窗口化的工具结果（`tool (名称): 内容`，按 display 解包并截断），这样早期检索回合的摘要不会丢掉工具返回的记忆/文档上下文。工具回合会两次界定量（`call_llm_node` 与 `generate_final_node`），**两次各自独立压缩**——不共享摘要：第二次界定的消息列表必然多了工具调用的 AIMessage 和 ToolMessage，切分点随之改变，逐字 transcript 匹配在真正需要压缩的回合反而经常失效，跨 state 共享摘要的复杂度不划算，最坏情况只是超长会话多付一次压缩调用。

## 自动知识捕获（Auto-Memory）

对话中涌现的实质性知识默认被**自动**写入记忆库，不需要用户显式说"记住这个"。这是 `generate_final_node` 的收尾行为：最终回答交付后，`_schedule_auto_memory` 把捕获排入后台任务，不阻塞、不强 gate 响应。开关 `AUTO_MEMORY_ENABLED`（默认 `true`）；关闭后恢复"仅按请求写入"（`write_memory_tool`）的行为。

### 捕获管线

一个用户轮次只有在**全部**满足以下条件时才被捕获（`_maybe_auto_memory`，`backend/agent/nodes.py`）：

1. **`auto_memory_enabled` && `memory_enabled`**——记忆管线整体 opt-out（`MEMORY_ENABLED=false`）时 agent 是纯聊天，不后台写记忆；
2. **长度快路径**——原始用户消息 ≥ `_AUTO_MEMORY_MIN_CONTENT_LEN`（12 字符），唯一零成本的预过滤；
3. **未触发节流**——per-thread 最小写入间隔 `AUTO_MEMORY_MIN_INTERVAL`（默认 60s）内不重复捕获；
4. **LLM 质量门**——一次结构化调用（`agent.auto_memory_gate` prompt，输出 `{"worthy": bool}`）判断该内容是否值得入库：技术决策、项目事实、经验教训、解决方案为 YES；闲聊、感谢、观点、提问、行动请求为 NO；
5. **本 turn 未显式写入**——`_write_tool_used_this_turn` 检测到 `write_memory_tool` 已被调用则跳过，避免重复；
6. **提取有实质**——`_has_substance` 要求摘要长度 ≥ `_AUTO_MEMORY_MIN_SUMMARY_LEN`（15）或含实体；**拒绝降级提取产物**（摘要恰好等于原文前 200 字符且无实体——那是 LLM 不可用时的 fallback 产物，不是知识）。

### 为什么需要 LLM 质量门

每次捕获成本是 4-7 次 LLM 调用（门控 1 + 摘要/实体/关系提取 3 + 嵌入 + 相似度扫描 + 可选冲突检测）。质量门在**昂贵的提取管线之前**用一次廉价结构化调用决定"值不值得"，取代了早先的关键词启发式——"这是否是持久知识"交给 LLM 判断而非短语表。门控**故意保守**：漏捕获可恢复，垃圾记忆永久污染检索。门控失败时默认放行（异常返回 `true`），由条件 6 的实质检查兜底，保证门控故障不吞掉任何一条长度合格的轮次。

### 为什么后台执行

实质轮次的捕获若同步执行，会在 SSE 流结束后继续拖住请求数秒（4-7 次 LLM 调用 + 写入），并占用 agent 并发槽位。改为 fire-and-forget 后台任务后，contextvar（`current_thread_id` / `current_trace_id`）在 `create_task` 时复制进任务，捕获仍保留线程/追踪归属；并发由 `_AUTO_MEMORY_MAX_CONCURRENCY`（4）信号量限制——每个 agent 轮次可 spawn 一个捕获，每个捕获最多 7 次 LLM 调用，不加并发上限会在 provider 限流上叠加后台流量。任务持有 strong reference 防止在首个 await 被 GC。

### 节流设计

稳态写入率由 **per-thread 最小间隔 + `write_memory` 的内容哈希幂等**共同界定（精确重复在 `write_memory` 直接返回 `duplicate`，不花提取成本）。早先同目标的两道闸——per-thread 生命周期上限、进程级滚动窗口——已移除：间隔单一即可界定额率，多一道滚动窗口是重复防御。节流状态在内存（与熔断器、usage 计数同级），进程重启即重置；多副本部署时各自计数（见 deployment.md「单实例部署约束」）。

## 显式写入（强制写入记忆）

自动记忆是"该记的自动记"；显式写入是"我要它记"——用户在发送框勾选「强制写入记忆」时，本轮消息**无条件**进入记忆管线，不经过 LLM 质量门（用户判断优先于模型判断）。旧的"请记住 X"交互已移除：LLM 不再自主选择写入工具，"记住"类请求退回自动记忆判断。

### 写入路径

勾选后的写入走 **Agent 消息流**，而不是独立的 REST 调用：`force_write` 随 chat 请求进入 state，`call_llm_node` 把一次 `write_memory_tool` tool_call **注入**该轮的 AIMessage，写入经正常的 ReAct 管线完成——

1. `ToolNode` 执行 `write_memory_tool`，`write_memory` 的三阶段提取（摘要/实体/关系）+ 相似度 merge + 冲突检测照常运行——库里存的是**提炼后的知识**，不是用户原文。
2. 命中冲突时 `check_conflict` **当场中断**，用户直接裁决（keep_existing / overwrite / merge / keep_both），不复用 `pending_conflicts` 队列（那是无交互路径 webhook/连接器的出口）。
3. 写入结果（inserted / merged / conflict）经 `/chat` 响应或 SSE meta 事件回到前端，Toast 反馈。

### 工具可见性分离

chat 传给 LLM 的 tool schema 不包含 `write_memory_tool`（`CHAT_LLM_TOOLS`，`build_agent_graph(llm_tools=...)` 与执行注册表 `ALL_TOOLS` 分离）——模型**无法自主选择**写入，显式写入只能由用户勾选触发。但工具仍留在执行注册表，注入的调用照常被 `ToolNode` 执行、`check_conflict` 照常拦截冲突、`_write_tool_used_this_turn` 照常抑制同轮自动记忆。对应地，prompt 里"用户要求记住时立即调用 write_memory_tool"的指令已删除。

### 审批与节流

勾选本身就是确认，故 chat 审批集（`CHAT_APPROVAL_TOOLS`）不再包含 `write_memory_tool`——再问一次批准是双重确认。`ingest_*` 与飞书通知在 chat 仍须审批；`write_memory_tool` 在**默认审批集**（`APPROVAL_REQUIRED_TOOLS`，patrol/scenario 使用）里仍保留，自主流程若模型误选仍会暂停。节流与抑制都只在**写入成功**后生效：`write_memory_tool` 成功（inserted / merged / duplicate）时把该线程记入自动记忆节流表，同轮自动记忆被 `_write_succeeded_this_turn` 抑制——同一内容不会双写；写入失败或冲突未决时两者都保持让位，自动记忆兜底捕获，内容不会因"强制写入失败 + 自动记忆被压"而双重丢失。

## 工具结果信封（Tool Envelope）

检索类工具返回一个 JSON 信封，统一承载"给 LLM 的展示文本"和"给前端的结构化来源"两件事（`backend/agent/tool_envelope.py`）：

```json
{"display": "Found 3 relevant memories:\n[1] (memory: a1b2c3d4, ...) ...", "sources": [{"id": "...", "type": "memory", "summary": "...", "relevance": 0.87, "entities": [...]}]}
```

- **`display`**：LLM 读取的干净文本——记忆/文档的展示行（含召回次数、实体名等）。这是模型在 ReAct 循环和历史里看到的唯一内容。
- **`sources`**：结构化引用（memory id、document_id、chunk_index、relevance、entities），供前端 API 层渲染可点击的来源链接，不进模型上下文。

**单一入口原则**：`build_tool_envelope` 是唯一构造函数（`search_memories_tool` / `retrieve_chunks_tool` / `query_rewrite_and_search_tool` 调用它），`parse_tool_envelope` / `envelope_display` 是唯一解析路径。非信封结果（纯文本、write/ingest/entity/notify 的普通 JSON）解析失败时原样透传，**绝不抛异常**——信封只是检索类工具的约定，不是全局强制。

**Envelope-aware 截断**：`truncate_tool_content`（默认上限 `MAX_TOOL_CONTENT_CHARS` = 800）**先解包 display 再截断**。原因：`sources` 数组本身就可能超过 800 字符，直接截原始 JSON 会把半截 JSON blob 喂回 ReAct 历史（后续轮次的 `_messages_to_dicts` 会把它当工具结果重发）。截断标记 `…[truncated]` 让模型知道结果比显示的更长。消费点（`_messages_to_dicts`、`generate_final_node` 的 Context 折叠、compaction transcript）全部经 `envelope_display` 解包后再处理，保证各路径对"什么是信封"的认知一致。

一次约定同时解决三件事：模型只读干净的 display 文本、前端拿到结构化来源、历史截断不破坏 JSON 结构。

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
