# EMA 架构深潜

> EMA 的系统架构、分层职责、关键技术决策与技术难点剖析。所有技术细节均来自实际代码，可按文件路径追溯验证。文档结构：背景 → 架构 → 决策 → 难点 → 成果 → 反思。

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

### 项目推进方式

技术选型、ADR 决策记录、代码实现均端到端完成。项目没有团队兜底，所以从第一天就用团队工程的纪律约束自己：每个架构决策写 ADR 记录依据、每次提交过 CI、每个检索改动跑评估集对比——用可追溯的流程替代缺失的 code review。

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
| 前端 | React + TS + Vite + Tailwind | 6 个页面：聊天、记忆库、实体图谱、连接器、巡检、冲突解决 |
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

**设计权衡**：
- Q：为什么不用多 Agent？→ A：约束明确单 Agent。多 Agent 在工程记忆场景没有收益，反而增加协调复杂度。一个 Agent + 9 个 tool 足够覆盖所有场景。
- Q：max_steps 为什么默认 5？→ A：实测下来 95% 的对话在 3 步内结束，5 是兜底防循环，env 可配。

### 决策 2：为什么 PostgreSQL + pgvector，不用专用向量库（ADR-002）

**背景**：需要存结构化数据（记忆、实体、关系）+ 向量 + 对话 checkpoint。

**选择**：PostgreSQL 16 + pgvector 单存储。

**理由**：
- **三合一**：一个库同时解决结构化数据 + 向量检索 + Agent checkpoint，运维成本低
- **事务一致性**：写记忆 + 写实体关联 + 写关系要在同一事务里，专用向量库做不到
- **数据量评估**：单团队记忆库预期万级，pgvector 的 hnsw 索引完全够用
- **SQL 可见**：向量搜索手写 SQL，`<=>` 操作符和参数完全可控，不是黑盒

**设计权衡**：
- Q：pgvector 性能比 Milvus/Qdrant 差吧？→ A：在万级数据量下差距很小，hnsw 索引 + cosine ops 召回延迟在 10ms 级。专用向量库的优势在百万级以上，EMA 远没到那个量级。换库的运维成本（双写一致性、部署复杂度）远大于性能收益。
- Q：HNSW 还是 ivfflat？→ A：当前用 hnsw（pgvector ≥ 0.5）。早期 ivfflat 的 lists 参数依赖数据量——小库（< 千条）时 lists=100 把探针散布到大量空聚类，召回差（记忆 seed-010 的教训）；hnsw 无聚类依赖，小库也稳，且建索引参数用默认。数据量到百万级才会重新评估专用向量库。

### 决策 3：为什么不用 Neo4j 做实体关系（ADR-005）

**背景**：三阶段提取会产生实体和关系，需要跨记忆归一化和关系查询。

**选择**：实体存 PG 表，一度关系用纯 SQL JOIN，不用图数据库。

**理由**：
- **检索模式不匹配**：EMA 的核心查询是"关于 PostgreSQL 我们有什么记忆"（一度关系），不是"遍历网络结构"（多跳图分析）
- **一度关系 SQL 直接 JOIN**，15 行以内可读
- **避免双写一致性**：图数据库要和 PG 双写，一致性负担大
- **多跳用递归 CTE**：N>3 跳性能下降但当前场景不需要

**拐点**：当 SQL 普遍超过 20 行、团队频繁问"这些记忆的共同模式"时，再评估引入图数据库。

**设计权衡**：
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

**设计权衡**：
- Q：Phase 4 为什么不做提前设计？→ A：垂直场景本质是"不同 System Prompt + Tool 组合"，是纯消费层。提前设计是过度设计，等前三个 Phase 完成后场景需求自然涌现，按需孵化。
- Q：每个 Phase 多长？→ A：Phase 1-2 各 1-2 周，Phase 3 是 2-3 周，Phase 4 按需。我坚持"每个 Phase 只做解锁下一阶段所需的最小能力"。

---

## 四、技术难点攻克（5 分钟，3 个难点）

### 难点 1：记忆去重与冲突检测（四级相似度分级）

**问题**：同一个知识点会被不同来源、不同时间反复写入。直接去重会丢信息，不去重会膨胀。

**方案**：写入前先向量搜索找最相似的已有记忆，按相似度分四级处理：

| 相似度 | 行为 | 实现 |
|--------|------|------|
| ≥ 0.85 | LLM 合并摘要 + 合并实体关系 | `_merge_memory()` |
| 0.72–0.85 | LLM 检测矛盾 → 矛盾走 HITL 仲裁，不矛盾补充关联 | `_detect_conflict()` + `_mark_conflict()` |
| 0.60–0.72 | 插入新记忆，meta 标记 `supplements` 关联到旧记忆 | `_supplement_memory()` |
| < 0.60 | 全新插入 | `_insert_memory()` |

**关键工程细节**：
- **阈值标定**：用 `threshold_calibration.py` 收集三类摘要对的 BGE-M3 相似度分布——同义改写（应 merge）0.84-0.97、同类不同记忆 ≤0.79、异类 ≤0.72。**旧值 0.92 高到一半该 merge 的同义对被漏成冲突检测，0.85 是分离点**，已从 0.92/0.75 改为 0.85/0.72（见 `tests/eval/reports/threshold_calibration_report.md`）
- **容错降级**：LLM 合并失败保留原摘要，冲突检测失败假定无冲突——不阻塞写入
- **冲突解决 4 选项**：keep_existing / overwrite / merge / keep_both，由 `resolve_conflict()` 实现
- **冲突走 HITL**：`check_conflict_node` 检测到 conflict action 后 `interrupt()`，等用户选

**设计权衡**：
- Q：为什么不用 LLM 直接判断要不要合并？→ A：LLM 判断成本高且不稳定。先用向量相似度做粗筛（便宜、稳定），只在边界区间（0.72-0.85）才调 LLM 做精细判断。这样 90% 的写入只需要一次向量搜索，不用调 LLM。
- Q：冲突检测的 prompt 长什么样？→ A：很简洁——给 LLM 两个 summary，让它返回 `{"conflict": true/false}` 的 JSON。强制 JSON 输出便于解析，失败时假定无冲突。

### 难点 2：艾宾浩斯衰减加权检索

**问题**：记忆库会膨胀，老的记忆和新的记忆在向量相似度上没区别，但很多老记忆已经过时。

**方案**：借鉴遗忘曲线，给每条记忆一个 `decay_factor`，检索时排序用"相似度 × decay_factor"。

**公式**：
```
R = e^(-t / S)
t = 距上次召回的小时数
S = 1 + (recall_count + 1) × 2   (相对强度)
```

> recall_count 是召回前的存储值，`+1` 得到召回后的计数（post-recall 约定，`compute_decay_factor` / `update_decay_batch` / `search_memories` 三处调用点统一）。

**实现细节**（`backend/service/decay.py`）：
- 新记忆 `decay_factor = 1.0`（满保留）
- 每次被检索召回，`update_decay_batch()` 执行：`recall_count += 1`，`recalled_at = NOW()`，重算 decay
- 频繁召回的记忆 S 大，衰减慢；长期没召回的 S=1，几小时就衰减到 0.5 以下
- SQL 排序：`(1 - (embedding <=> :vec)) * decay_factor DESC`

**为什么这个设计有效**：
- **自我演化**：不用人工清理，常用记忆自然浮在上面，过时的自然沉底
- **可恢复**：长期没召回的记忆 decay 低但没删除，被精确问到时仍能召回。decay 层 `search_memories` 默认 `threshold=0.0`；生产入口 `query_memories` 默认 `threshold=0.3` 作为垃圾过滤门槛（作用于原始相似度、衰减加权之前）——低于 0.3 的记忆不进候选池，衰减无法把它们救回
- **抗噪声**：刚写入的高相似度记忆不会立刻盖过长期验证过的记忆

**设计权衡**：
- Q：衰减会不会让重要但少问的记忆消失？→ A：不会删除，只是排序靠后。生产入口 `query_memories` 默认 `threshold=0.3` 作为垃圾过滤门槛（作用于原始相似度、衰减加权之前）——低于 0.3 的记忆不进候选池，衰减救不回；但 0.3 以上、只是排序靠后的记忆，精确查询时向量相似度高仍能召回。decay 层 `search_memories` 自身默认 `threshold=0.0`
- Q：S 的公式怎么来的？→ A：参考艾宾浩斯原始强度模型，简化为 `1 + (recall_count + 1) × 12`（`recall_count` 是召回前的存储值，`+1` 得到召回后的计数）。系数最初是 2，但 decay A/B 显示 `×2` 半衰期太短、把低频相关旧记忆整体埋掉（合成老化分布下 recall@5 只有 0.367）；调大到 12 并加 0.10 保留下限后同分布重测 recall 回升到 0.667，保留"过时沉底"偏好的同时不埋掉旧知识（见 decay_ab_report.md）。
- Q：召回时同步更新 decay 会不会拖慢检索？→ A：会有一点。`update_decay_batch` 是一条原子 `UPDATE ... RETURNING`，对召回的 top_k 条批量更新（替代了早期逐条 UPDATE 的 N+1 写法）。实测 top_k=5 时增加约 20ms，可接受。未来可以改成异步队列。

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

**设计权衡**：
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
| 检索 Recall@5 | **0.900（默认确定性基线）**；语义通道 opt-in 时 1.000 | tests/eval 实测（30 query，memory:norank@k5；默认纯子串匹配，见 [memory-path-report.md](../../tests/eval/reports/memory_path_report.md)） |
| 检索 MRR | **0.844（默认确定性基线）**；语义通道 opt-in 时 0.944 | 同上；此前的 0.983 是 chunk:vector 路径，非生产默认路径 |
| 语义通道贡献 | 30 条中 3 条（q018/q024/q026）仅靠 embedding 相似度判相关（0.900→1.000、0.844→0.944）；**语义通道是显式 opt-in，非默认** | 语义通道用被评测的 BGE-M3 自评（见 dataset.py），因自证故默认关闭、显式开启，贡献已如实披露 |
| 检索判别力（hard-negative） | **纯向量 27 条陷阱集：综合通过仅 59.3%、MRR 0.790、11 条陷阱压过目标 → bounded cross-encoder top-3 重排后 81.5%、MRR 0.889、5 条** | [hard-negative-report.md](../../tests/eval/reports/hard_negative_report.md) 实测；纯向量对表层词重合高度宽容（陷阱与目标共享关键词时排序靠词面而非意图）；已实现 bounded-CE 重排修复（`query_memories(use_cross_encoder=True)`，默认关） |
| 检索 NDCG@5 | 0.959 | tests/eval 实测（memory:norank@k5） |
| 检索延迟（稳态） | ~190ms（hybrid 无 rerank：embed + sparse + sort） | tests/eval 实测（hybrid:norank@k5 190ms）；hybrid+rerank 17.5s（cross-encoder CPU 瓶颈） |
| cross-encoder rerank 延迟 | 17.5s/query（CPU 瓶颈） | tests/eval 实测，BGE-reranker-v2-m3 568M，待 GPU 优化 |
| 代码量 | backend 13916 行 + agent 2497 行（纯 Python，无前端） | wc -l |
| 测试覆盖 | 1416 测试用例 | pytest --collect-only |
| Agent 任务级完成率 | **completed 0.500** / tool_recall 0.938 / within_budget 0.875 | run_task_eval 8 任务实测（DeepSeek，deterministic judge，2026-08-09） |
| Agent 任务轨迹质量 | groundedness 1.000 / citation 0.875 / **0 执行错误**；unexpected_rate 0.375 暴露过度调用 | 同上次运行，详见 [llm-eval.md](llm-eval.md) |
| 日均检索次数 | [待生产部署后统计] | — |
| Agent 单轮平均 tool 调用 | 2.6 次（任务级实测，task 轨迹均值） | run_task_eval 8 任务 n_steps 均值 |
| 对话 P95 延迟 | **73.6s（10 轮真实对话实测，P50 43.2s / mean 35.9s）** | tests/perf/measure_chat_p95.py（2026-08-11，DeepSeek deepseek-v4-flash + 本地 BGE-M3；主要耗时在每轮约 19 次 rerank_llm 约 40s） |
| 单轮对话 token / 成本 | **≈28.6k tokens/轮 ≈¥0.06**（10 轮合计 285.9k tokens / 估 $0.081） | 同上；成本大头 rerank_llm 75.6k + agent_chat 165.8k（含 144k cache_read 折扣价，见 [usage.py](../../backend/service/usage.py) `estimate_cost`） |
| 项目周期 | 3 个月，端到端推进 | — |

**任务级评估设计**：8 个真实多步任务驱动**完整 Agent 图**（ReAct 循环 + 真实工具执行 + HITL 自动放行），测的不是单次决策而是整条轨迹——`completed`（调齐必备工具 + 实质答案）/ `tool_recall` / `within_budget`（未撞 max_steps 强制终止）/ 答案接地（judge 对 Agent 实际看到的工具上下文判定）。HITL 自动放行是为了隔离"人的决策"与"Agent 能力"。**实测最有价值的不是分数，而是它暴露了组件级评测看不到的轨迹级问题**：DeepSeek 对单检索任务过度调用工具（task-006 回答一个记忆问题调了 4 次），概念查询甚至撞 max_steps——`unexpected_rate 0.375` 是 completed 掉到 0.5 的主因，改进方向是强化工具描述边界与轨迹节流。这套评估还顺带抓出并修复了一个生产 HITL 正确性 bug（拒绝审批后写操作仍被静态边路由执行），见 [llm-eval.md](llm-eval.md) 末尾。

**评估集设计**：30 条标注 query，5 类 × 6 条，用**内容指纹**而非 UUID 匹配相关结果（可移植、CI 友好）；difficulty 分 easy/medium/hard。**三个如实披露的点**：
1. **主评估集是"自问自答"构造的**——每条 query 由目标记忆反向生成、每条只有 1 条相关记忆，Recall@5=1.0 只能证明"找得到"，不能证明"判别力"。它测的是"记住答案"而非"检索能力"，是回归基线不是能力上限。
2. **语义通道已改为显式 opt-in（非默认）**：30 条里 3 条靠"被评测的 BGE-M3 给自己打分"（embedding 相似度 ≥0.80）判为相关（0.900→1.000、0.844→0.944）。这部分是模型"认出自已"，不是独立判据——所以默认评估是**确定性纯子串基线**（recall 0.90/MRR 0.84，无自证），语义通道显式开启才启用，贡献已在报告里如实披露。
3. **hard-negative 判别力才是真实水平，且已用它改进检索**：`query_candidates.jsonl` 里 27 条陷阱集（每条配一个表面词重合但语义不同的陷阱记忆）实测——纯向量目标召回 100%（找得到），但陷阱入侵 96.3%（几乎都混进 top-5）、综合通过仅 **59.3%**、11 条陷阱排在目标前。**这个集驱动了真实改进**：A/B 实验对比 query 重写 / hybrid 融合 / 检索后意图判别三个方向后，落地了 bounded cross-encoder top-3 重排——只对 decay 排序前 3 名（竞争区）用 cross-encoder 打分重排、其余保持原序，不 floor 过滤，把综合通过提至 **81.5%**、MRR 0.790→0.889、worse 11→5（默认关、显式启用，避免 CPU 延迟）。完整数字见 [hard-negative-report.md](../../tests/eval/reports/hard_negative_report.md)。

**设计选择**：对抗性审查要求"hard negative 跑一遍还 1.0 吗"必须能回答——与其让 1.0 被当自证拆穿，不如主动把真实判别力数字摆出来：1.0 的局限是什么、用 27 条陷阱集量化了真实判别力、下一步怎么改进。诚实暴露比完美数字可信。

**架构演进**：从单 Agent + 记忆库，演进到 4 个 Phase 全闭环——知识图谱、多源连接器、主动巡检、4 个垂直场景（故障复盘 / 代码审查 / Onboarding / 技术债雷达）。

---

## 六、反思与改进（1 分钟）

### 做得好的

- **ADR 驱动决策**：每个关键选型都有 ADR 记录"为什么做、为什么不、拐点是什么"，回头复盘和对外讲解都很清晰
- **简单优先**：每个 Phase 只做解锁下一步的最小能力，不过度设计。比如不做多租户（用 project 字段软隔离）、不做 Neo4j（用 SQL 一度关系）
- **容错降级**：所有 LLM 调用都有失败降级路径，不阻塞主流程
- **可观测性分层**：成本（`llm_usage` 表持久化）+ 健康（`/metrics` Prometheus 时序：HTTP 延迟/状态、LLM 调用/token、熔断器、Agent 并发槽位、ReAct 步数分布）两套通道，埋点都收敛在咽喉点（provider 调用、熔断器、槽位、路由结束）——task eval 测到的过度调用信号在产线用 `ema_agent_steps` 直方图持续观测

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
│  9 个 tool: search×3/entity/write/extract/ingest×2/notify│
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  记忆层 (自实现 service 函数, 非 LangChain)          │
│  写: extract → embed → 四级相似度 → merge/conflict/insert │
│  读: embed → 衰减加权搜索 → rerank(cross-encoder/LLM 默认关) │
│  衰减: R=max(e^(-t/S),0.1), S=1+(recall+1)×12                │
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
| "四级相似度阈值怎么定的？" | 已标定：收集三类摘要对的相似度分布（同义改写 0.84-0.97 / 同类 ≤0.79 / 异类 ≤0.72），发现旧值 0.92 高到一半该 merge 的同义对被漏成冲突检测；0.85 是分离点，已改 MERGE 0.85 / CONFLICT 0.72（见 threshold_calibration_report.md）。诚实边界：8 对改写样本小，真实生产分布待部署后确认 |
| "衰减公式为什么是这个？" | 艾宾浩斯简化 `R = max(e^(-t/S), 0.10)`，`S=1+(recall+1)×12`（post-recall 约定，三处调用点统一）。**参数经 decay A/B 校准**：原 `×2` 半衰期太短，合成老化分布下 recall 0.367；调 `×12` + 0.10 floor 后 0.667，保留"过时沉底"偏好（见 decay_ab_report.md）。另说明"从未召回按 created_at 衰减"（否则旧记忆永不沉底） |
| "评估数字 1.0 是自证吧？" | 主动认 + 给诚实口径：主评估集 30 条**默认是确定性纯子串基线**（recall 0.90 / MRR 0.84，无自证）；语义通道是显式 opt-in（救回 3/30 → 1.00/0.94，用被评测模型自评故非默认）；真实判别力看 27 条陷阱集——纯向量综合通过仅 59.3%、11 条陷阱压过目标。**已用这个集落地改进**：bounded cross-encoder top-3 重排提至 81.5%（默认关、显式启用），排除 query 重写/hybrid（会放大表层词问题） |
| "为什么不用 LangChain Agent？" | 黑盒 + HITL 控制弱 + 调试代价 |
| "pgvector 性能不行吧？" | 万级数据 hnsw 够用，讲拐点 |
| "实体归一化准确率怎么样？" | 两层判断（向量+LLM），fire-and-forget + 巡检补全 |
| "HITL 暂停后状态怎么存？" | PostgresSaver，thread_id 恢复 |
| "怎么防止 LLM 幻觉污染记忆？" | 老实说目前没做置信度过滤，是下一步改进项 |
| "如果让你重做会改什么？" | 提前打 layer 字段做分层记忆 + 加检索日志 |
