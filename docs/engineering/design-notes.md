# EMA 技术设计笔记

> EMA 各技术主题的设计与权衡记录，按六个板块整理：Agent 架构、RAG 与检索、向量检索与 pgvector、Prompt 工程、AI 工程化、系统设计。每条对应真实代码，记录设计取舍与实现细节。

---

## 板块一：Agent 架构

### LangGraph 和 LangChain Agent 有什么区别？为什么选 LangGraph？

1. **抽象层级不同**：LangChain Agent（create_react_agent / create_agent）是预建黑盒，内部是固定流程；LangGraph 的 StateGraph 是图原语，节点和边自己定义
2. **可控性**：LangGraph 能精确控制"在哪个节点暂停、暂停后去哪"，LangChain Agent 的 HITL 是回调式，插入点受限
3. **场景需求**：EMA 需要两个 HITL 卡点——写操作前审批、记忆冲突仲裁，要插在 ReAct 循环的不同位置，LangGraph 的 `interrupt()` + `Command(goto=...)` 天然支持
4. **副作用**：LangGraph 还能用 PostgresSaver 做对话持久化，复用已有 PG 不加新依赖

**权衡**：LangGraph 学习曲线集中在 interrupt/Command 的状态恢复机制——interrupt 后 state 里的字段不会自动同步，要手动管理。工程记忆场景不需要多 Agent 角色分离，一个 Agent + 9 个 tool 覆盖所有场景，多 Agent 只增加协调复杂度。

### ReAct 循环是什么？怎么防止死循环？

1. ReAct = Reasoning + Acting：LLM 先思考要不要调 tool，调完把结果塞回 context 再问 LLM，直到 LLM 不再要 tool 生成最终回答
2. 防死循环：`_make_route_after_call_llm(max_steps)` 在 step_count >= max_steps 时强制路由到 generate_final
3. 默认 max_steps=5，env 可配 MAX_AGENT_STEPS；实测 95% 对话在 3 步内结束，5 是兜底

**权衡**：信任 LLM 自然停止的前提是 tool schema 写清楚用途，LLM 看到 ToolMessage 内容后通常知道该生成最终答案了。真出循环再调 max_steps，不提前优化。

### Tool calling 怎么实现？Tool 返回什么格式？

1. 用 LangChain 的 `@tool` 装饰器定义 async 函数，自动生成 schema
2. `ToolNode(tools, handle_tool_errors=True)` 自动执行 tool_calls 并产生 ToolMessage
3. **Tool 返回 string**——ToolNode 标准约定，LLM 通过 ToolMessage.content 读取
4. 设计：返回 `json.dumps({"display": "...", "sources": [...]})`，display 给 LLM 看，sources 给前端渲染用
5. LLM 调用通过 `LLMProvider.chat_raw(messages, tools)` 返回 `{content, tool_calls}` 结构化响应

**权衡**：ToolNode 只能写 messages，不能直接写 state 其他字段，所以从 ToolMessage 提取结果比维护专门字段更健壮，且对任意 tool 通用。tool 报错由 `handle_tool_errors=True` 包成 ToolMessage 返回给 LLM，LLM 自己重试或换 tool。

### Agent 有几个 tool？粒度怎么定？

9 个 tool，分四类：
- 检索类：search_memories_tool（记忆）、retrieve_chunks_tool（文档分块）、query_entity_tool（实体关系）、query_rewrite_and_search_tool（LLM 改写多路召回）
- 写入类：write_memory_tool（含冲突检测）、extract_memory_tool（仅提取不持久化）
- 摄取类：ingest_git_repo_tool（Git 历史）、ingest_document_tool（文档 AST chunk）
- 通知类：notify_feishu_tool（飞书推送）

粒度原则：**一个 tool 对应一个 service 函数，零逻辑重复**。tool 是薄封装，业务在 service 层。

---

## 板块二：RAG 与检索

### RAG 管道长什么样？和标准 LangChain RAG 有什么区别？

1. 不用 LangChain Retriever/Chain，是**一组独立 async 函数**
2. 写路径：`chunk_text() / chunk_code()` → `embed()` → `write_chunks()` 入 chunks 表
3. 读路径：`embed_query()` → `vector_search()` →（默认无 rerank，直接相似度排序；`rerank_cross_encoder()` / `rerank_llm()` 均为 opt-in）→ `assemble()` → LLM
4. **区别**：每个环节单独可测、单独可替换，不通过框架链粘合。调试时能精确定位是 embed 还是 rerank 的问题

**权衡**：Retriever 是黑盒，定制 rerank 和过滤要继承重写，代价高。`retrieve()` 函数 20 行，所有逻辑可见。双 reranker 默认都不开（opt-in）——eval 实测小语料下 cross-encoder rerank 掉 recall 且慢 ~90 倍，所以关闭；需要时 `use_cross_encoder=True` 走本地 BGE-Reranker-v2-m3（零 API 成本），或 `use_llm_rerank=True` 走 LLM。

### chunk 策略怎么设计？代码和文本不一样吗？

1. **通用文本** `chunk_text(text, max_size=512, overlap=64)`：递归分隔符切分——段落 → 行 → 句子 → 词，保证不截断语义单元
2. **Python 代码** `chunk_code(code, max_lines=80)`：AST 感知，按函数/类/模块边界切分，不会把 200 行函数切两半
3. 非 Python 代码回退到按行切分
4. overlap 后重新检查大小，超限时继续细分
5. 全部自实现，不依赖外部库

**权衡**：`max_size=512` 是精度和召回的平衡点——BGE-M3 窗口支持到 8192，但 512 太大会稀释向量语义、太小会丢失上下文。`overlap=64` 大约一两句话，覆盖一个完整语义单元足够。

### 怎么做 rerank？cross-encoder 和 LLM rerank 怎么选？

| Reranker | 引擎 | 成本 | 适用 |
|----------|------|------|------|
| cross-encoder | BGE-Reranker-v2-m3 本地 | 零 API | 默认，90% 场景 |
| LLM | LLMProvider | API 费用 | 需精细语义判断 |

cross-encoder 是双塔模型，query 和 doc 一起进 transformer 输出相似度，比向量召回的 dot product 准但慢——所以先用向量召回 top 20，再用 cross-encoder rerank 到 top 5。实测 top 20 rerank 约 100-200ms（CPU），可批处理优化。LLM rerank 让模型给每个候选打 0-1 分，成本高但能理解"虽然字面不像但语义相关"的情况。

### 三阶段记忆提取是什么？为什么分三阶段？

```
extract_summary(content) ──┐
                           ├─ asyncio.gather (并行)
extract_entities(content) ─┘
                           │
                           └─ extract_relations(summary, entities)
```

1. **摘要**：2-5 句简洁段落
2. **实体**：`[{name, type}]`，类型 person/project/technology/decision/event/file/concept
3. **关系**：`[{from, to, type}]`，类型 depends_on/causes/part_of/contradicts/supersedes/relates_to

**为什么分阶段**：摘要和实体无依赖，并行省一半时间；关系提取依赖摘要和实体，必须串行；每阶段独立调 LLM，一个失败不影响其他（容错）。

---

## 板块三：向量检索与 pgvector

### pgvector 怎么用？索引选了什么？

1. 表结构：`embedding vector(1024)`（BGE-M3 输出 1024 维）
2. 索引：`hnsw` on `embedding vector_cosine_ops`（pgvector ≥ 0.5）
3. 相似度计算：`1 - (embedding <=> :vec ::vector)`，`<=>` 是 cosine distance，1 减一下变成相似度
4. 排序：**纯相似度**，单段 HNSW 直接按 `embedding <=> :vec` 返回。原实现曾用「相似度 × 实时 decay_factor」加权排序，decay A/B 实测掉 recall（0.667 vs 0.900）后已移除（见 decision-faq 第 7 节）；`recall_count`/`recalled_at` 只作元数据记录，不参与排序

**HNSW vs ivfflat**：当前用 hnsw。早期用 ivfflat 时 lists 参数依赖数据量，小库（< 千条）下 lists=100 把探针散布到大量空聚类，召回差；hnsw 无聚类依赖、小库也稳，参数用默认。数据量到百万级才重新评估专用向量库。PQ 是有损压缩、召回精度下降，当前数据量内存够用不需要。

### 向量检索的 SQL 长什么样？

```sql
SELECT id, source_type, summary, entities, relations,
       recall_count, meta, created_at, recalled_at,
       1 - (embedding <=> :vec ::vector) AS similarity
FROM memories
WHERE embedding IS NOT NULL
  AND deleted_at IS NULL
  AND 1 - (embedding <=> :vec ::vector) > :threshold
ORDER BY embedding <=> :vec ::vector
LIMIT :limit
```

要点：`embedding IS NOT NULL` 防止空向量；`deleted_at IS NULL` 软删除过滤；threshold 过滤低相似度（`search_memories` 默认 0.0；生产入口 `query_memories` 默认 0.3 作垃圾过滤门槛）；排序键是原始相似度——HNSW 索引只服务按 `embedding <=> :vec` 排序的扫描，所以不需要两段候选窗 + Python 重排。召回统计由 `record_recalls` 在搜索后单独批量记录。

### 实体归一化怎么做的？

两层判断：
1. 新记忆提取出实体 `[{name: "pg16", type: "technology"}]`
2. `embed("pg16")` 在 entities 表做向量搜索找候选
3. 找到候选 `{name: "PostgreSQL", similarity: 0.88}`
4. **LLM 判断**："pg16 是不是 PostgreSQL 的版本指代？" → 是同一实体
5. 把 memory 链接到已有 entity，而非创建新实体

**为什么两层**：纯向量会误判（"pg16" 和 "PostgreSQL" 向量相似但不一定是同一实体）；纯 LLM 成本高。向量粗筛 + LLM 精判，平衡准确率和成本。归一化不阻塞写入——`asyncio.create_task` fire-and-forget，best-effort，失败只 log 不传播，记忆本身已写入，最终一致性由周巡检补全。

---

## 板块四：Prompt 工程

### 冲突检测 prompt 长什么样？怎么保证输出可解析？

```
You are a conflict detector. Compare two summaries and determine if the new one
CONTRADICTS the existing one.

Existing summary: {existing_summary}
New summary: {new_summary}

Reply with ONLY a JSON object: {{"conflict": true}} or {{"conflict": false}}
```

设计要点：
1. **强制 JSON 输出**：`Reply with ONLY a JSON object`，避免自由文本
2. **明确判断标准**：用 CONTRADICTS 而不是 "related"，避免误判相关为冲突
3. **容错解析**：`json.loads` 失败时 `except` 假定无冲突，不阻塞写入

**权衡**：`json.loads` 失败走 except 假定无冲突，是 "fails safe" 原则——宁可漏报冲突也不要阻塞写入。冲突检测频率不高（只在 0.72-0.85 区间），走结构化 JSON 够用；提取侧的 entity/relation 已经用函数调用通道（enum 生成期约束 + 降级 chat_structured）。

### 三阶段提取的 prompt 怎么设计？

1. **摘要 prompt**：要求 2-5 句简洁段落，保留关键事实
2. **实体 prompt**：输出 `[{name, type}]`，type 枚举固定（person/project/technology/...）
3. **关系 prompt**：依赖前两步输出，输出 `[{from, to, type}]`，type 枚举固定（depends_on/causes/...）

关键设计：**type 枚举固定**避免 LLM 自由发挥，便于后续归一化和查询；每个阶段独立 prompt，一个失败不影响其他（容错）；输出 JSON 便于程序解析。

---

## 板块五：AI 工程化

### 流式输出怎么做的？SSE 怎么实现？

1. 后端用 FastAPI 的 `StreamingResponse` + SSE 格式（`data: ...\n\n`）
2. Agent 用 `graph.astream()` 逐 token 输出
3. 前端用 EventSource 接收
4. HITL interrupt 时流式暂停，前端弹审批卡片，用户操作后恢复

**权衡**：SSE 而非 WebSocket——单向推送够用，比 WebSocket 简单，浏览器原生支持自动重连。interrupt 时 astream 会 yield interrupt 标志，前端识别后停止流式渲染、显示审批 UI，用户操作后 `Command(resume=...)` 恢复继续流式。

### HITL 人机协同怎么实现？状态怎么存？

1. **interrupt 机制**：在节点里调 `interrupt()`，图执行暂停，状态序列化到 checkpointer
2. **PostgresSaver**：状态存 PG，用 thread_id 关联，跨重启可恢复
3. **恢复机制**：用户操作后 `graph.ainvoke(Command(resume=value), config={"configurable": {"thread_id": tid}})`
4. **路由**：节点返回 `Command(goto="...")` 动态指定下一步

两个 HITL 卡点：check_approval_node（写操作前审批）、check_conflict_node（记忆冲突仲裁）。interrupt 期间用户关浏览器，状态在 PG，下次用同 thread_id 调用恢复到 interrupt 点，前端能看到 pending 卡片。PostgresSaver 每次 interrupt 序列化整个 state，state 不大（messages + 几个字段），实测 < 50ms。

### LLM 调用怎么做容错？

fails safe 原则：
1. **记忆合并**：LLM 合并失败 → 保留原摘要，不丢数据
2. **冲突检测**：LLM 检测失败 → 假定无冲突，不阻塞写入
3. **实体归一化**：失败 → fire-and-forget，记忆已写入，归一化后续巡检补
4. **Agent 节点**：LLM 调用失败不终止图执行，错误信息进 state，generate_final 兜底

**权衡**：LLM 调用不稳定是常态（限流、超时、格式错），抛错会让一次写入失败。fails safe 让主流程总能走通，副作用由巡检补偿。

### LLM 成本怎么控制？

1. **粗筛省 LLM**：90% 写入只走向量搜索（便宜），只在 0.72-0.85 边界区间才调 LLM 做冲突检测
2. **双 reranker**：默认关闭（opt-in）——小语料下 cross-encoder rerank 实测掉 recall 且慢 ~90 倍；启用时优先本地 cross-encoder（零 API 成本），需精细语义判断才走 LLM rerank
3. **Provider 切换**：DeepSeek 便宜，Claude 贵但质量高，按场景配
4. **max_steps 限制**：防止 Agent 无限调 tool 烧 token
5. **传输韧性**：LLM/Embedding 调用统一走 tenacity 指数退避重试 + 熔断器（[resilience.py](../../backend/shared/resilience.py)），429/5xx 自动重试、连续失败熔断快速失败；重复投递靠 content-hash 幂等去重

成本监控：provider 层是唯一咽喉点，每次调用写 `llm_usage` 表（scenario / tokens / latency / 成本估算），`/api/usage/*` 按天、按场景、按模型汇总，trace_id 能回放单轮 agent 的调用链。缓存没做——记忆检索是动态的，相同 query 召回结果会随写入变化，缓存意义不大。

### 系统健康怎么监控？

1. **分工**：成本（`llm_usage` 表，历史查询）与健康（Prometheus 时序，实时抓取）是两个独立通道——`GET /metrics` 暴露文本格式，`METRICS_ENABLED` 开关
2. **HTTP 层**：ASGI 中间件按**路由模板路径**记录请求数、延迟直方图、状态码分布——用 route path 不用原始 URL，标签基数有界
3. **LLM 层**：和成本行同一咽喉点打点——按 scenario 的调用数（success/error）、延迟、token
4. **韧性层**：熔断器状态 gauge + 打开次数 + open 期间快速拒绝次数
5. **Agent 层**：并发槽位占用 + 503 拒绝计数；每次 chat 完成记录 ReAct 步数分布——task eval 发现的过度调用问题，产线用这个直方图持续观测
6. **实现原则**：所有埋点异常吞掉、开关关闭时 no-op，绝不给热路径加阻塞

**权衡**：单实例部署 + 需要的指标就这几类，prometheus_client 轻量直接暴露文本格式，一个库搞定；要接 otel 迁移成本低（埋点已收敛在咽喉点）。`ema_agent_steps` 直方图把 task eval 的过度调用信号搬到产线——不用等每周评测就能看到平均步数/长尾分布有没有恶化。

---

## 板块六：系统设计

### 企业知识库 Agent 怎么设计？

1. **存储**：PostgreSQL + pgvector（结构化 + 向量 + checkpoint 三合一）
2. **Agent**：LangGraph StateGraph + ReAct + HITL 卡点
3. **记忆写入**：三阶段提取 + 四级相似度去重 + 冲突 HITL
4. **检索**：向量召回 + rerank + 召回统计（命中记忆记 `recall_count`，过期归档由巡检读访问历史判断）
5. **数据源**：连接器抽象（Webhook + 适配器），按需接入
6. **交互**：Web UI + ChatOps Bot + MCP Server（给其他 Agent 用）

**扩展权衡**：万级数据 pgvector 切 HNSW 索引、记忆库分表（按 project 或时间）、检索加 pre-filter（先按 metadata 过滤再向量搜索）。多租户先用 project 字段软隔离（默认过滤 + 允许跨项目搜索），团队数 5+ 且有安全需求再硬隔离。知识不过时靠周巡检扫描陈旧记忆 + 知识时效性检测 + HITL 复核。

### 记忆库到百万级，架构怎么演进？

1. **存储**：pgvector 可能不够，切 Milvus/Qdrant 做向量库，PG 只存结构化数据。双写一致性用 outbox pattern
2. **检索**：加 pre-filter（先按 metadata + 时间过滤缩小候选集），再向量搜索
3. **召回统计**：批量更新 recall 元数据，不在线上召回路径同步影响排序（排序是纯相似度）
4. **分层记忆**：Working / Task / Long-term 三层，向量检索只走 Long-term
5. **缓存**：高频 query 的检索结果缓存（带 TTL，因为新写入会改召回集合）

**双写一致性**：PG 先写 + outbox 表，异步 worker 消费 outbox 写向量库，失败重试，最终一致。向量库坏了 PG 是 source of truth，可用 outbox 重放重建。

### 怎么评估 RAG 系统的质量？

1. **检索质量**：人工标注 query-doc 相关性，算 Recall@K、MRR、NDCG@K——EMA 有标注集（70 条，5 类 × 14，含 easy/medium/hard 难度分级），生产路径默认确定性基线 0.886 / 0.767
2. **生成质量**：LLM-as-judge 从准确性、完整性、相关性打分；EMA 用 `chat_structured`（JSON Schema 校验）输出覆盖事实/忠实度/幻觉论断，四套件（工具选择 / 知识抽取 / 最终答案 / 端到端）见 [llm-eval.md](llm-eval.md)
3. **端到端**：用户反馈（点赞/点踩）+ 后续追问率（追问多说明没答好）
4. **现状**：检索与生成质量都有自动化评估（`python -m tests.eval.run_eval` / `run_llm_eval`），评估集 CI 每周自动跑

**权衡**：LLM 评估有偏差，只做粗筛——裁判输出结构化事实而非 1-5 分，覆盖率/忠实度从判决直接算；关键改动仍人工抽检。检索与生成分开测——检索跑 run_eval 看 Recall/MRR，生成跑 run_llm_eval 看答案覆盖度与幻觉，前端 sources 面板展示检索来源方便定位。

### Agent 整体效果怎么量化？

1. **任务级端到端评测**（`run_task_eval`）：8 个真实多步任务驱动**完整 Agent 图**——ReAct 循环 + 真实工具执行 + HITL 门，测的是整条轨迹而非单次决策
2. **指标**：`completed`（调齐必备工具 + 实质答案 + 无错误）/ `tool_recall` / `within_budget`（未撞 max_steps 强制终止）/ 答案接地（judge 对 Agent 实际看到的工具上下文判定）
3. **实测**（DeepSeek）：tool_recall 0.94、groundedness 1.00、0 执行错误；但 **completed 只有 0.5**——`unexpected_rate 0.375` 说明模型过度调用工具，这是组件级评测永远看不到的轨迹级问题
4. **HITL 处理**：评测自动放行（审批通过、冲突 keep_existing），隔离"人的决策"与"Agent 能力"；要测拒绝路径就注入自定义 resume 策略

**completed 为什么只有 0.5**：不是工具选不对（tool_recall 0.94 说明该调的几乎都调了），而是过度调用——task-006 回答一个记忆问题调了 4 次工具（search×2 + retrieve_chunks + query_entity），task-004 概念查询循环 8 次撞 max_steps。这是真实的行为短板。改进方向：强化工具描述边界、轨迹级节流（检索结果已覆盖就停手）、max_steps 兜底但 5 步对概念查询仍偏松。

**这个评估的额外收获**：抓出一个生产 bug——拒绝审批后写操作仍被执行。根因是 LangGraph 1.2.10 resume 被 interrupt 暂停的节点时，Command(goto) 和节点的静态边会同时生效，拒绝路径的静态边把路由拉到 tools。删掉静态边、Command 唯一路由，加了图级回归测试。单测和组件评测都发现不了这种跨节点编排问题。

---

## 技术速查

| 项 | 值 |
|------|---------|
| BGE-M3 维度 | 1024 |
| pgvector 索引 | hnsw + cosine_ops |
| 召回统计 | record_recalls 批量记 recall_count/recalled_at（元数据，不参与排序） |
| 四级阈值 | 0.85合并 / 0.72冲突 / 0.60关联 / <0.60新增 |
| Agent 节点 | 5 个：call_llm/check_approval/tools/check_conflict/generate_final |
| HITL 卡点 | 2 个：写前审批 + 冲突仲裁 |
| Agent 任务级指标 | 8 任务 completed 0.5 / tool_recall 0.94 / grounded 1.0 |
| 工具数 | 9 个 |
| 冲突解决选项 | 4 个：keep_existing/overwrite/merge/keep_both |
| LangGraph 持久化 | PostgresSaver，thread_id 关联 |
| 为什么不用 Neo4j | 一度关系 SQL 够用，避免双写一致性 |
| 为什么不做多租户 | project 字段软隔离，等拐点 |
| LLM 容错原则 | fails safe，不阻塞主流程 |
