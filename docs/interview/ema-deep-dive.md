# EMA 项目深度讲解（15-20 分钟版）

> 用于技术面主问环节。讲解节奏：背景 2' + 架构 3' + 决策 5' + 难点 5' + 成果 2' + 反思 1'。
> 所有技术细节均来自 EMA 实际代码，可经得起追问。建议对着这份文档口头复述 3 遍以上。

---

## 一、项目背景与价值（2 分钟）

### 痛点

研发团队最大的知识管理问题不是"没有文档"，而是**知识存在三个地方**：
1. **人脑里**——老人走了，新人问"为什么这么设计"没人答
2. **聊天记录里**——飞书/Slack 里有答案但搜不到、过了就沉
3. **代码 commit 里**——Git 历史里有决策上下文但没人翻

传统 wiki 解决不了，因为 wiki 靠人手写，而**研发最讨厌写文档**。

### EMA 的解法

EMA 不让人写文档，而是**从工程活动里自动提取记忆**。数据源有四个：
- Git commit 历史（pygit2 遍历）
- 项目管理工具（PingCode work item / bug）
- CI/CD 构建记录
- 飞书讨论

每条原始内容经过：**分块 → 嵌入 → 三阶段提取（摘要+实体+关系）→ 四级相似度去重 → 入库**。检索时用向量召回 + 衰减加权 + rerank。

一句话价值：**让团队的工程记忆可检索、可复用、不随人走**。

### 我的角色

[需要你补充：如"独立完成，从架构设计到全栈实现"]。技术选型、ADR 决策记录、代码实现都是我做的。

---

## 二、整体架构（3 分钟）

### 一句话架构

```
用户 → React 前端 → FastAPI 后端 → LangGraph Agent (StateGraph)
                                    ↓
                              记忆层 (service 函数)
                                    ↓
                         PostgreSQL + pgvector
                                    ↓
                                LLM Provider
```

### 分层职责

| 层 | 技术 | 职责 |
|----|------|------|
| 前端 | React + TS + Vite + Tailwind | 5 个页面：聊天、记忆库、实体图谱、连接器、巡检 |
| 后端 | FastAPI + Python 3.12 + async | API、生命周期、调用 Agent |
| Agent | LangGraph 手动 StateGraph | ReAct 循环 + 2 个 HITL 卡点 |
| 记忆 | 自实现 service 函数 | 提取、去重、衰减、检索——不用 LangChain Retriever |
| 存储 | PostgreSQL 16 + pgvector | 结构化数据 + 向量 + 对话 checkpoint 三合一 |
| LLM | LLMProvider 抽象 | DeepSeek / Claude 切换，业务代码不依赖具体 SDK |

### 核心设计原则

**记忆系统不依赖 LangChain**。我没用 LangChain 的 Retriever/Chain 黑盒，每个环节（chunk/embed/extract/rerank/decay）都是独立 async 函数，单独可测、单独可替换。这是有意为之——LangChain 的链条式抽象在调试和定制上代价太高。

---

## 三、关键技术决策（5 分钟，4 个 ADR）

### 决策 1：为什么用 LangGraph，不用 LangChain Agent（ADR-001）

**背景**：EMA 的 Agent 流程本质是个 while 循环——LLM 决定调不调 tool，调完把结果塞回去再问 LLM。

**选择**：手动构建 `StateGraph`，不用 `create_react_agent`（已废弃）或 `create_agent`（禁止引入）。

**理由**：
- LangChain Agent 是黑盒，HITL 的插入点不好控制
- 我需要两个 HITL 卡点：**写操作前审批**（避免 Agent 乱写记忆）、**记忆冲突时仲裁**（避免矛盾信息被合并）
- LangGraph 的 `interrupt()` + `Command(goto=...)` 能精确控制路由
- LangGraph 的 `PostgresSaver` 让对话能跨重启恢复，复用已有 PG 不加新依赖

**实际图结构**（5 节点）：
```
START → call_llm ──(无 tool_calls)──→ generate_final → END
         │
         └──(有 tool_calls)──→ check_approval ──→ tools ──→ check_conflict
                                   │                              │
                                   └──(拒绝)──────────────────────┘
                                                              │
                                   call_llm ←─────────────────┘
```

**追问预案**：
- Q：为什么不用多 Agent？→ A：约束明确单 Agent。多 Agent 在工程记忆场景没有收益，反而增加协调复杂度。一个 Agent + 8 个 tool 足够覆盖所有场景。
- Q：max_steps 为什么默认 5？→ A：实测下来 95% 的对话在 3 步内结束，5 是兜底防循环，env 可配。

### 决策 2：为什么 PostgreSQL + pgvector，不用专用向量库（ADR-002）

**背景**：需要存结构化数据（记忆、实体、关系）+ 向量 + 对话 checkpoint。

**选择**：PostgreSQL 16 + pgvector 单存储。

**理由**：
- **三合一**：一个库同时解决结构化数据 + 向量检索 + Agent checkpoint，运维成本低
- **事务一致性**：写记忆 + 写实体关联 + 写关系要在同一事务里，专用向量库做不到
- **数据量评估**：单团队记忆库预期万级，pgvector 的 ivfflat 索引完全够用
- **SQL 可见**：向量搜索手写 SQL，`<=>` 操作符和参数完全可控，不是黑盒

**追问预案**：
- Q：pgvector 性能比 Milvus/Qdrant 差吧？→ A：在万级数据量下差距很小，ivfflat 索引 + cosine ops 召回延迟在 10ms 级。专用向量库的优势在百万级以上，EMA 远没到那个量级。换库的运维成本（双写一致性、部署复杂度）远大于性能收益。
- Q：HNSW 还是 ivfflat？→ A：当前用 ivfflat，建索引快、内存占用小。HNSW 召回质量略高但建索引慢、内存大。等数据量到 10 万级会切 HNSW。

### 决策 3：为什么不用 Neo4j 做实体关系（ADR-005）

**背景**：三阶段提取会产生实体和关系，需要跨记忆归一化和关系查询。

**选择**：实体存 PG 表，一度关系用纯 SQL JOIN，不用图数据库。

**理由**：
- **检索模式不匹配**：EMA 的核心查询是"关于 PostgreSQL 我们有什么记忆"（一度关系），不是"遍历网络结构"（多跳图分析）
- **一度关系 SQL 直接 JOIN**，15 行以内可读
- **避免双写一致性**：图数据库要和 PG 双写，一致性负担大
- **多跳用递归 CTE**：N>3 跳性能下降但当前场景不需要

**拐点**：当 SQL 普遍超过 20 行、团队频繁问"这些记忆的共同模式"时，再评估引入图数据库。

**追问预案**：
- Q：实体归一化怎么做的？→ A：新记忆提取出实体后，embed 实体名在 entities 表做向量搜索找候选，再用 LLM 判断"pg16 是不是 PostgreSQL 的版本指代"。两层判断避免纯向量的误判。
- Q：归一化会不会阻塞写入？→ A：不会。用 `asyncio.create_task` fire-and-forget，best-effort。失败只 log 不传播，记忆本身已经写入。最终一致性由周巡检补全。

### 决策 4：四阶段扩展路线图（ADR-006）

**背景**：基础功能就绪后有多个扩展方向，需要排优先级。

**决策**：
```
Phase 1: 知识图谱化（实体归一化 + 一度关系）    ← 杠杆率最高、零新依赖
Phase 2: 多源连接器（Webhook + Jira/CI/Slack）   ← 依赖 Phase 1 归一化
Phase 3: 主动 Agent（调度 + 巡检 + 推送）        ← 依赖 Phase 1+2
Phase 4: 垂直场景（复盘/审查/Onboarding/技术债）  ← 纯消费层，按需孵化
```

**核心理由**：Phase 1 杠杆率最高——实体数据已存在 JSONB 里，缺的只是归一化层。工作量小但让所有记忆产生关联，解锁后续所有 Phase。

**追问预案**：
- Q：Phase 4 为什么不做提前设计？→ A：垂直场景本质是"不同 System Prompt + Tool 组合"，是纯消费层。提前设计是过度设计，等前三个 Phase 完成后场景需求自然涌现，按需孵化。
- Q：每个 Phase 多长？→ A：Phase 1-2 各 1-2 周，Phase 3 是 2-3 周，Phase 4 按需。我坚持"每个 Phase 只做解锁下一阶段所需的最小能力"。

---

## 四、技术难点攻克（5 分钟，3 个难点）

### 难点 1：记忆去重与冲突检测（四级相似度分级）

**问题**：同一个知识点会被不同来源、不同时间反复写入。直接去重会丢信息，不去重会膨胀。

**方案**：写入前先向量搜索找最相似的已有记忆，按相似度分四级处理：

| 相似度 | 行为 | 实现 |
|--------|------|------|
| ≥ 0.92 | LLM 合并摘要 + 合并实体关系 | `_merge_memory()` |
| 0.75–0.92 | LLM 检测矛盾 → 矛盾走 HITL 仲裁，不矛盾补充关联 | `_detect_conflict()` + `_mark_conflict()` |
| 0.60–0.75 | 插入新记忆，meta 标记 `supplements` 关联到旧记忆 | `_supplement_memory()` |
| < 0.60 | 全新插入 | `_insert_memory()` |

**关键工程细节**：
- **阈值调参**：0.92 是实测"几乎是同一条"的边界，0.75 是"可能相关"的边界，0.60 是"勉强沾边"的边界
- **容错降级**：LLM 合并失败保留原摘要，冲突检测失败假定无冲突——不阻塞写入
- **冲突解决 4 选项**：keep_existing / overwrite / merge / keep_both，由 `resolve_conflict()` 实现
- **冲突走 HITL**：`check_conflict_node` 检测到 conflict action 后 `interrupt()`，等用户选

**追问预案**：
- Q：为什么不用 LLM 直接判断要不要合并？→ A：LLM 判断成本高且不稳定。先用向量相似度做粗筛（便宜、稳定），只在边界区间（0.75-0.92）才调 LLM 做精细判断。这样 90% 的写入只需要一次向量搜索，不用调 LLM。
- Q：冲突检测的 prompt 长什么样？→ A：很简洁——给 LLM 两个 summary，让它返回 `{"conflict": true/false}` 的 JSON。强制 JSON 输出便于解析，失败时假定无冲突。

### 难点 2：艾宾浩斯衰减加权检索

**问题**：记忆库会膨胀，老的记忆和新的记忆在向量相似度上没区别，但很多老记忆已经过时。

**方案**：借鉴遗忘曲线，给每条记忆一个 `decay_factor`，检索时排序用"相似度 × decay_factor"。

**公式**：
```
R = e^(-t / S)
t = 距上次召回的小时数
S = 1 + recall_count × 2   (相对强度)
```

**实现细节**（`backend/service/decay.py`）：
- 新记忆 `decay_factor = 1.0`（满保留）
- 每次被检索召回，`update_decay()` 执行：`recall_count += 1`，`recalled_at = NOW()`，重算 decay
- 频繁召回的记忆 S 大，衰减慢；长期没召回的 S=1，几小时就衰减到 0.5 以下
- SQL 排序：`(1 - (embedding <=> :vec)) * decay_factor DESC`

**为什么这个设计有效**：
- **自我演化**：不用人工清理，常用记忆自然浮在上面，过时的自然沉底
- **可恢复**：长期没召回的记忆 decay 低但没删除，被精确问到时仍能召回（阈值过滤是 0.0）
- **抗噪声**：刚写入的高相似度记忆不会立刻盖过长期验证过的记忆

**追问预案**：
- Q：衰减会不会让重要但少问的记忆消失？→ A：不会删除，只是排序靠后。threshold 默认 0.0，所有记忆都在候选池里。如果用户精确查询，向量相似度高的话仍能召回。
- Q：S 的公式怎么来的？→ A：参考艾宾浩斯原始强度模型，简化为 `1 + recall_count × 2`。系数 2 是调出来的——让"被召回过 5 次"的记忆强度变成 11，是"新记忆"的 11 倍，衰减速度差一个数量级，这个梯度比较合理。
- Q：召回时同步更新 decay 会不会拖慢检索？→ A：会有一点。`update_decay` 是一次 UPDATE，对召回的 top_k 条逐条更新。实测 top_k=5 时增加约 20ms，可接受。未来可以改成异步队列。

### 难点 3：双 HITL 卡点的 LangGraph 实现

**问题**：Agent 调写工具（写记忆、摄取 Git）前要让人审，记忆冲突时要让人仲裁。这两个卡点要插在 ReAct 循环的不同位置。

**方案**：用 LangGraph 的 `interrupt()` + `Command(goto=...)` 精确控制。

**写操作前审批**（`check_approval_node`）：
- `call_llm` 返回有 `tool_calls` 时，先进 `check_approval`
- 如果是写类工具（write_memory / ingest），`interrupt()` 暂停，前端弹审批卡片
- 用户批准 → `Command(goto="tools")` 继续执行
- 用户拒绝 → `Command(goto="call_llm")` 回到 LLM 重新决策

**记忆冲突仲裁**（`check_conflict_node`）：
- `tools` 节点执行完 `write_memory_tool` 后，进 `check_conflict`
- 检测 tool 返回的 action 是不是 `conflict`
- 是 → `interrupt()`，前端弹冲突解决卡片（4 选项）
- 用户选完 → `resolve_conflict()` 执行 → `Command(goto="call_llm")`

**关键工程细节**：
- **非中断的直通**：没有冲突 / 不是写工具时，节点直接 return state，走默认边，不 interrupt
- **PostgresSaver 持久化**：interrupt 期间状态存 PG，用户可以隔几小时再来回复，对话不丢
- **max_steps 防循环**：`_make_route_after_call_llm(max_steps)` 在 step_count 超阈值时强制走 `generate_final`

**追问预案**：
- Q：为什么不用 LangChain 的 AgentExecutor？→ A：AgentExecutor 的 HITL 支持很弱，回调式而非图式，没法精确控制"在哪个节点暂停、暂停后去哪"。LangGraph 的图模型天然支持。
- Q：interrupt 期间用户关了浏览器怎么办？→ A：状态在 PostgresSaver 里，下次用同一个 thread_id 调用会恢复到 interrupt 点。前端能看到 pending 的审批/冲突卡片。
- Q：Command(goto=...) 和 add_edge 有什么区别？→ A：edge 是静态声明，Command 是运行时动态路由。HITL 的路径取决于用户选择，必须用 Command。

---

## 五、量化成果（2 分钟）

> 下方数据分两类：**实测**（tests/eval 跑出，可复现）与**待生产**（需部署后统计）。实测数字已回填，待生产项保留占位。

| 指标 | 数值 | 来源 |
|------|------|------|
| 数据源接入数 | 4 个（Git / PingCode / CI / 飞书） | — |
| 记忆库规模 | 评估集 30 条种子记忆（生产待部署） | tests/eval/seed_memories.jsonl |
| 检索 Recall@5 | **1.000**（hybrid: 向量 + jieba 分词 BM25，无 rerank） | tests/eval 实测（30 query）；向量 baseline 0.833；hybrid+rerank 0.967 |
| 检索 MRR | **0.983** | tests/eval 实测；29/30 query rank-1 命中 |
| 检索 NDCG@5 | **0.988** | tests/eval 实测 |
| 检索 MAP@5 | 0.983 | tests/eval 实测 |
| 检索延迟（稳态） | ~235ms（embed 150-230ms + sparse + sort，无 rerank） | 实测；hybrid+rerank ~20s（cross-encoder CPU 瓶颈） |
| cross-encoder rerank 延迟 | 50s/query（CPU 瓶颈） | 实测，BGE-reranker-v2-m3 568M，待 GPU 优化 |
| 代码量 | 后端 6490 行（纯后端，无前端） | cloc |
| 测试覆盖 | 452 测试用例（含 101 检索评估单测） | pytest --collect-only |
| 日均检索次数 | [待生产部署后统计] | — |
| Agent 单轮平均 tool 调用 | [待生产部署后统计] | — |
| 对话 P95 延迟 | [待生产部署后统计] | — |
| 项目周期 | [需要你补充：如 3 个月，独立完成] | — |

**评估集设计**（面试加分点）：30 条标注 query，5 类 × 6 条，用**内容指纹**而非 UUID 匹配相关结果（可移植、CI 友好）；difficulty 分 easy/medium/hard 暴露向量质量短板。按 category 看：技术决策/代码实现 recall@5=1.000，故障复盘/历史背景=0.667（最弱，概念查询短板）；按 difficulty 看 medium 最高（0.929），easy 反而最低（0.714，部分 easy query 词重合但向量区分度不足）。完整报告见 [eval-report.md](eval-report.md)。

**架构演进**：从单 Agent + 记忆库，演进到 4 个 Phase 全闭环——知识图谱、多源连接器、主动巡检、4 个垂直场景（故障复盘 / 代码审查 / Onboarding / 技术债雷达）。

---

## 六、反思与改进（1 分钟）

### 做得好的

- **ADR 驱动决策**：每个关键选型都有 ADR 记录"为什么做、为什么不、拐点是什么"，回头复盘和对外讲解都很清晰
- **简单优先**：每个 Phase 只做解锁下一步的最小能力，不过度设计。比如不做多租户（用 project 字段软隔离）、不做 Neo4j（用 SQL 一度关系）
- **容错降级**：所有 LLM 调用都有失败降级路径，不阻塞主流程

### 可以更好的

- **记忆质量保障缺失**：目前没有置信度过滤，LLM 抽取幻觉会污染记忆库。这是下一步要做的——给每条记忆打置信分，低于阈值走 HITL
- **检索可观测性弱**：没有检索日志，不知道"这条记忆被谁、在什么上下文下召回过"。合规和审计需要时得补
- **冷启动数据稀疏**：新接入的数据源初期记忆少，相似故障回溯这类场景效果差。需要冷启动策略

---

## 附：一句话架构图（白板可画）

```
┌─────────────────────────────────────────────────────┐
│  前端 (React+TS)  聊天/记忆库/图谱/连接器/巡检      │
└──────────────────────┬──────────────────────────────┘
                       │ SSE 流式
┌──────────────────────▼──────────────────────────────┐
│  FastAPI 后端 (async)                                │
│  ┌────────────────────────────────────────────────┐ │
│  │  LangGraph Agent (手动 StateGraph)              │ │
│  │  call_llm → check_approval → tools              │ │
│  │      ↑                    ↓                     │ │
│  │      └── check_conflict ←──┘  (双 HITL 卡点)    │ │
│  │      ↓                                          │ │
│  │  generate_final → END                           │ │
│  └────────────────────────────────────────────────┘ │
│  8 个 tool: search/query/write/extract/ingest×2/notify│
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  记忆层 (自实现 service 函数, 非 LangChain)          │
│  写: extract → embed → 四级相似度 → merge/conflict/insert │
│  读: embed → 衰减加权搜索 → rerank(cross-encoder/LLM) │
│  衰减: R=e^(-t/S), S=1+recall×2                     │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  PostgreSQL 16 + pgvector                            │
│  memories / chunks / entities / memory_entities      │
│  + PostgresSaver (对话 checkpoint)                   │
└─────────────────────────────────────────────────────┘
        │                          │
┌───────▼───────┐         ┌────────▼────────┐
│ LLMProvider   │         │ EmbeddingProvider│
│ (DeepSeek/Claude)│      │ (BGE-M3 本地)   │
└───────────────┘         └─────────────────┘
```

---

## 追问高危区（提前准备）

| 高危追问 | 准备要点 |
|---------|---------|
| "四级相似度阈值怎么定的？" | 0.92/0.75/0.60 是实测调参，讲调参过程 |
| "衰减公式为什么是这个？" | 艾宾浩斯原始模型 + 简化，S 系数 2 是调出来的 |
| "为什么不用 LangChain Agent？" | 黑盒 + HITL 控制弱 + 调试代价 |
| "pgvector 性能不行吧？" | 万级数据 ivfflat 够用，讲拐点 |
| "实体归一化准确率怎么样？" | 两层判断（向量+LLM），fire-and-forget + 巡检补全 |
| "HITL 暂停后状态怎么存？" | PostgresSaver，thread_id 恢复 |
| "怎么防止 LLM 幻觉污染记忆？" | 老实说目前没做置信度过滤，是下一步改进项 |
| "如果让你重做会改什么？" | 提前打 layer 字段做分层记忆 + 加检索日志 |
