# 技术问题演练 Q&A 库

> 按 AI/LLM 应用工程师岗位高频考点分 6 大板块。每题给出「答题要点」+「追问预案」。
> 练习方式：先盖住答案自己讲一遍，再对照要点查漏。标记 ⭐ 的是高频必练题。

---

## 板块一：LLM Agent 架构

### ⭐ Q1.1 LangGraph 和 LangChain Agent 有什么区别？你为什么选 LangGraph？

**答题要点**（30-60 秒）：
1. **抽象层级不同**：LangChain Agent（如 create_react_agent / create_agent）是预建黑盒，内部是固定流程；LangGraph 的 StateGraph 是图原语，节点和边自己定义
2. **可控性**：LangGraph 能精确控制"在哪个节点暂停、暂停后去哪"，LangChain Agent 的 HITL 是回调式，插入点受限
3. **我的场景**：EMA 需要两个 HITL 卡点——写操作前审批、记忆冲突仲裁，要插在 ReAct 循环的不同位置，LangGraph 的 `interrupt()` + `Command(goto=...)` 天然支持
4. **副作用**：LangGraph 还能用 PostgresSaver 做对话持久化，复用已有 PG 不加新依赖

**追问预案**：
- Q：LangGraph 学习曲线陡吗？→ A：图模型概念简单，难点在 interrupt/Command 的状态恢复机制，文档不算特别清晰，我踩过坑——interrupt 后 state 里的字段不会自动同步，要手动管理
- Q：为什么不用多 Agent？→ A：工程记忆场景不需要角色分离，一个 Agent + 8 个 tool 覆盖所有场景。多 Agent 增加协调复杂度，没有收益

### Q1.2 ReAct 循环是什么？你的实现里怎么防止死循环？

**答题要点**：
1. ReAct = Reasoning + Acting，LLM 先思考要不要调 tool，调完把结果塞回 context 再问 LLM，直到 LLM 不再要 tool 生成最终回答
2. 防死循环：`_make_route_after_call_llm(max_steps)` 在 step_count >= max_steps 时强制路由到 generate_final
3. 默认 max_steps=5，env 可配 MAX_AGENT_STEPS
4. 实测 95% 对话在 3 步内结束，5 是兜底

**追问预案**：
- Q：为什么信任 LLM 自然停止？→ A：tool schema 里写清楚每个 tool 的用途，LLM 看到 ToolMessage 内容后通常知道该生成最终答案了。真出循环再调 max_steps，不提前优化

### ⭐ Q1.3 Tool calling 怎么实现的？Tool 返回什么格式？

**答题要点**：
1. 用 LangChain 的 `@tool` 装饰器定义 async 函数，自动生成 schema
2. `ToolNode(tools, handle_tool_errors=True)` 自动执行 tool_calls 并产生 ToolMessage
3. **Tool 返回 string**——这是 ToolNode 标准约定，LLM 通过 ToolMessage.content 读取
4. 我的设计：返回 `json.dumps({"display": "...", "sources": [...]})`，display 给 LLM 看，sources 给前端渲染用
5. LLM 调用通过 `LLMProvider.chat_raw(messages, tools)` 返回 `{content, tool_calls}` 结构化响应

**追问预案**：
- Q：为什么不用结构化 state 字段传 tool 结果？→ A：ToolNode 只能写 messages，不能直接写 state 其他字段。从 ToolMessage 提取比维护专门字段更健壮，且对任意 tool 通用
- Q：tool 报错怎么办？→ A：`handle_tool_errors=True` 让 ToolNode 把异常包成 ToolMessage 返回给 LLM，LLM 会自己重试或换 tool

### Q1.4 你的 Agent 有几个 tool？怎么决定 tool 粒度？

**答题要点**：8 个 tool，分四类：
- 检索类：search_memories_tool（记忆）、retrieve_chunks_tool（文档分块）、query_entity_tool（实体关系）
- 写入类：write_memory_tool（含冲突检测）、extract_memory_tool（仅提取不持久化）
- 摄取类：ingest_git_repo_tool（Git 历史）、ingest_document_tool（文档 AST chunk）
- 通知类：notify_feishu_tool（飞书推送）

粒度原则：**一个 tool 对应一个 service 函数，零逻辑重复**。tool 是薄封装，业务在 service 层。

---

## 板块二：RAG 与检索

### ⭐ Q2.1 你的 RAG 管道长什么样？和标准 LangChain RAG 有什么区别？

**答题要点**：
1. 我的 RAG 不用 LangChain Retriever/Chain，是**一组独立 async 函数**
2. 写路径：`chunk_text() / chunk_code()` → `embed()` → `write_chunks()` 入 chunks 表
3. 读路径：`embed_query()` → `vector_search()` → `rerank_cross_encoder()`（默认）或 `rerank_llm()`（可选）→ `assemble()` → LLM
4. **区别**：每个环节单独可测、单独可替换，不通过框架链粘合。调试时能精确定位是 embed 还是 rerank 的问题

**追问预案**：
- Q：为什么不用 LangChain 的 Retriever？→ A：Retriever 是黑盒，定制 rerank 和过滤要继承重写，代价高。我的 `retrieve()` 函数 20 行，所有逻辑可见
- Q：双 reranker 怎么切换？→ A：参数 `use_llm_rerank=False` 默认走 cross-encoder（BGE-Reranker-v2-m3 本地，零 API 成本），需要精细语义判断时传 True 走 LLM

### ⭐ Q2.2 chunk 策略怎么设计的？代码和文本不一样吗？

**答题要点**：
1. **通用文本** `chunk_text(text, max_size=512, overlap=64)`：递归分隔符切分——段落 → 行 → 句子 → 词的优先级，保证不截断语义单元
2. **Python 代码** `chunk_code(code, max_lines=80)`：AST 感知，按函数/类/模块边界切分，不会把 200 行函数切两半
3. 非Python 代码回退到按行切分
4. overlap 后重新检查大小，超限时继续细分
5. 全部自实现，不依赖外部库

**追问预案**：
- Q：为什么 max_size=512？→ A：BGE-M3 的窗口支持到 8192，但 512 是精度和召回的平衡点。太大会稀释向量语义，太小会丢失上下文
- Q：overlap=64 够吗？→ A：够。overlap 是防止边界信息丢失，64 token 大概是一两句话，覆盖一个完整语义单元足够

### Q2.3 怎么做 rerank？cross-encoder 和 LLM rerank 怎么选？

**答题要点**：
| Reranker | 引擎 | 成本 | 适用 |
|----------|------|------|------|
| cross-encoder | BGE-Reranker-v2-m3 本地 | 零 API | 默认，90% 场景 |
| LLM | LLMProvider | API 费用 | 需精细语义判断 |

cross-encoder 是双塔模型，query 和 doc 一起进 transformer，输出相似度。比向量召回的 dot product 准，但慢——所以先用向量召回 top 20，再用 cross-encoder rerank 到 top 5。

**追问预案**：
- Q：cross-encoder rerank 多慢？→ A：top 20 rerank 大概 100-200ms（CPU）。可以批处理优化
- Q：LLM rerank 的 prompt 长什么样？→ A：让 LLM 给每个候选打 0-1 分，返回 JSON。成本高但能理解"虽然字面不像但语义相关"的情况

### Q2.4 三阶段记忆提取是什么？为什么分三阶段？

**答题要点**：
`extract_memory(content)` 三个阶段，前两个并行：
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

**为什么分阶段**：
- 摘要和实体无依赖，并行省一半时间
- 关系提取依赖摘要和实体，必须串行
- 每阶段独立调 LLM，一个失败不影响其他（容错）

---

## 板块三：向量检索与 pgvector

### ⭐ Q3.1 pgvector 怎么用的？索引选了什么？

**答题要点**：
1. 表结构：`embedding vector(1024)`（BGE-M3 输出 1024 维）
2. 索引：`ivfflat` on `embedding vector_cosine_ops`
3. 相似度计算：`1 - (embedding <=> :vec ::vector)`，`<=>` 是 cosine distance，1 减一下变成相似度
4. 加权排序：`(1 - cosine_distance) * decay_factor DESC`

**追问预案**：
- Q：ivfflat 和 HNSW 怎么选？→ A：ivfflat 建索引快、内存小，万级数据够用。HNSW 召回质量略高但建索引慢、内存大。10 万级以上我会切 HNSW
- Q：ivfflat 的 lists 参数怎么设？→ A：经验值是 `sqrt(rows)`，比如 1 万条数据 lists=100。需要在建索引时调
- Q：为什么不用乘积量化（PQ）？→ A：PQ 是有损压缩，召回精度下降。当前数据量内存够用，不需要压缩

### Q3.2 向量检索的 SQL 长什么样？

**答题要点**（衰减加权检索）：
```sql
SELECT id, source_type, summary, entities, relations,
       decay_factor, recall_count, meta, created_at,
       (1 - (embedding <=> :vec ::vector)) * decay_factor AS weighted_score
FROM memories
WHERE embedding IS NOT NULL
  AND deleted_at IS NULL
  AND 1 - (embedding <=> :vec ::vector) > :threshold
ORDER BY (1 - (embedding <=> :vec ::vector)) * decay_factor DESC
LIMIT :limit
```

要点：
- `embedding IS NOT NULL` 防止空向量
- `deleted_at IS NULL` 软删除过滤
- threshold 过滤低相似度（默认 0.0，可调）
- 排序键是"相似度 × 衰减因子"

**追问预案**：
- Q：为什么 threshold 默认 0.0 不过滤？→ A：让所有记忆都在候选池，由 decay 和 rerank 负责排序。如果用户精确查询，即使老记忆也能召回
- Q：`::vector` 是什么语法？→ A：pgvector 的类型转换，把字符串参数转成 vector 类型

### Q3.3 实体归一化怎么做的？

**答题要点**（两层判断）：
1. 新记忆提取出实体 `[{name: "pg16", type: "technology"}]`
2. `embed("pg16")` 在 entities 表做向量搜索找候选
3. 找到候选 `{name: "PostgreSQL", similarity: 0.88}`
4. **LLM 判断**："pg16 是不是 PostgreSQL 的版本指代？" → 是同一实体
5. 把 memory 链接到已有 entity，而非创建新实体

**为什么两层**：
- 纯向量会误判（"pg16" 和 "PostgreSQL" 向量相似但不一定是同一实体）
- 纯 LLM 成本高
- 向量粗筛 + LLM 精判，平衡准确率和成本

**追问预案**：
- Q：归一化会不会阻塞写入？→ A：不会。`asyncio.create_task` fire-and-forget，best-effort。失败只 log 不传播，记忆本身已写入。最终一致性由周巡检补全
- Q：LLM 判断错了怎么办？→ A：目前没有自动纠错。周巡检会扫描"疑似归一化错误"的实体对，标记人工复核。这是已知改进项

---

## 板块四：Prompt 工程

### Q4.1 你的冲突检测 prompt 长什么样？怎么保证输出可解析？

**答题要点**：
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

**追问预案**：
- Q：LLM 不按格式输出怎么办？→ A：`json.loads` 失败走 except，假定无冲突。这是"fails safe"原则——宁可漏报冲突也不要阻塞写入
- Q：为什么不用 function calling 强制结构化？→ A：可以，但冲突检测频率不高（只在 0.75-0.92 区间），简单 prompt + JSON 解析够用

### Q4.2 三阶段提取的 prompt 怎么设计的？

**答题要点**：
1. **摘要 prompt**：要求 2-5 句简洁段落，保留关键事实
2. **实体 prompt**：输出 `[{name, type}]`，type 枚举固定（person/project/technology/...）
3. **关系 prompt**：依赖前两步输出，输出 `[{from, to, type}]`，type 枚举固定（depends_on/causes/...）

关键设计：
- **type 枚举固定**：避免 LLM 自由发挥，便于后续归一化和查询
- **每个阶段独立 prompt**：一个失败不影响其他（容错）
- **输出 JSON**：便于程序解析

---

## 板块五：AI 工程化

### ⭐ Q5.1 流式输出怎么做的？SSE 怎么实现？

**答题要点**：
1. 后端用 FastAPI 的 `StreamingResponse` + SSE 格式（`data: ...\n\n`）
2. Agent 用 `graph.astream()` 逐 token 输出
3. 前端用 EventSource 接收
4. HITL interrupt 时流式暂停，前端弹审批卡片，用户操作后恢复

**追问预案**：
- Q：interrupt 期间流式怎么处理？→ A：interrupt 时 astream 会 yield 一个 interrupt 标志，前端识别后停止流式渲染、显示审批 UI。用户操作后调 `Command(resume=...)` 恢复，继续流式
- Q：SSE 还是 WebSocket？→ A：SSE。单向推送够用，比 WebSocket 简单，浏览器原生支持自动重连

### ⭐ Q5.2 HITL 人机协同怎么实现的？状态怎么存？

**答题要点**：
1. **interrupt 机制**：在节点里调 `interrupt()`，图执行暂停，状态序列化到 checkpointer
2. **PostgresSaver**：状态存 PG，用 thread_id 关联，跨重启可恢复
3. **恢复机制**：用户操作后调 `graph.ainvoke(Command(resume=value), config={"configurable": {"thread_id": tid}})`
4. **路由**：节点返回 `Command(goto="...")` 动态指定下一步

**两个 HITL 卡点**：
- check_approval_node：写操作前审批
- check_conflict_node：记忆冲突仲裁

**追问预案**：
- Q：interrupt 期间用户关浏览器怎么办？→ A：状态在 PG，下次用同 thread_id 调用恢复到 interrupt 点。前端能看到 pending 卡片
- Q：PostgresSaver 性能怎么样？→ A：每次 interrupt 序列化整个 state 到 PG，state 不大（messages + 几个字段），实测 < 50ms

### Q5.3 LLM 调用怎么做容错的？

**答题要点**（fails safe 原则）：
1. **记忆合并**：LLM 合并失败 → 保留原摘要，不丢数据
2. **冲突检测**：LLM 检测失败 → 假定无冲突，不阻塞写入
3. **实体归一化**：失败 → fire-and-forget，记忆已写入，归一化后续巡检补
4. **Agent 节点**：LLM 调用失败不终止图执行，错误信息进 state，generate_final 兜底

**追问预案**：
- Q：为什么不直接抛错？→ A：LLM 调用不稳定是常态（限流、超时、格式错）。抛错会让一次写入失败，用户体验差。fails safe 让主流程总能走通，副作用由巡检补偿

### Q5.4 LLM 成本怎么控制？

**答题要点**：
1. **粗筛省 LLM**：90% 写入只走向量搜索（便宜），只在 0.75-0.92 边界区间才调 LLM 做冲突检测
2. **双 reranker**：默认 cross-encoder 本地（零 API 成本），需要时才走 LLM rerank
3. **Provider 切换**：DeepSeek 便宜，Claude 贵但质量高，按场景配
4. **max_steps 限制**：防止 Agent 无限调 tool 烧 token
5. **传输韧性**：LLM/Embedding 调用统一走 tenacity 指数退避重试 + 熔断器（[resilience.py](../../backend/shared/resilience.py)），429/5xx 自动重试、连续失败熔断快速失败；重复投递靠 content-hash 幂等去重，不重复入库

**追问预案**：
- Q：怎么知道哪次调用贵？→ A：目前没有精细的成本监控。改进项是加 token 计数日志，按 tool 和场景统计
- Q：缓存做了吗？→ A：没做。记忆检索是动态的，相同 query 召回结果会因 decay 变化，缓存意义不大

---

## 板块六：系统设计

### ⭐ Q6.1 如果让你设计一个企业知识库 Agent，你怎么做？

**答题要点**（套用 EMA 架构）：
1. **存储**：PostgreSQL + pgvector（结构化 + 向量 + checkpoint 三合一）
2. **Agent**：LangGraph StateGraph + ReAct + HITL 卡点
3. **记忆写入**：三阶段提取 + 四级相似度去重 + 冲突 HITL
4. **检索**：向量召回 + rerank + 衰减加权（让常用知识浮上来）
5. **数据源**：连接器抽象（Webhook + 适配器），按需接入
6. **交互**：Web UI + ChatOps Bot + MCP Server（给其他 Agent 用）

**追问预案**：
- Q：万级数据怎么扩展？→ A：pgvector 切 HNSW 索引；记忆库分表（按 project 或时间）；检索加 pre-filter（先按 metadata 过滤再向量搜索）
- Q：多租户怎么做？→ A：先用 project 字段软隔离（默认过滤+允许跨项目搜索），等团队数 5+ 且有安全需求再硬隔离
- Q：怎么保证知识不过时？→ A：周巡检扫描陈旧记忆 + 知识时效性检测（标记可能与最新代码冲突的记忆）+ HITL 复核

### Q6.2 如果记忆库到百万级，架构怎么演进？

**答题要点**：
1. **存储**：pgvector 可能不够，切 Milvus/Qdrant 做向量库，PG 只存结构化数据。双写一致性用 outbox pattern
2. **检索**：加 pre-filter（先按 metadata + 时间过滤缩小候选集），再向量搜索
3. **衰减**：批量异步更新 decay，不在线上召回路径同步更新
4. **分层记忆**：Working / Task / Long-term 三层，向量检索只走 Long-term
5. **缓存**：高频 query 的检索结果缓存（带 TTL，因为 decay 会变）

**追问预案**：
- Q：双写一致性怎么保证？→ A：PG 先写 + outbox 表，异步 worker 消费 outbox 写向量库。失败重试。最终一致
- Q：向量库坏了怎么办？→ A：PG 是 source of truth，向量库可以重建。重建用 outbox 重放

### Q6.3 怎么评估 RAG 系统的质量？

**答题要点**：
1. **检索质量**：人工标注 query-doc 相关性，算 Recall@K、MRR
2. **生成质量**：人工评估答案准确性、完整性、是否有幻觉
3. **端到端**：用户反馈（点赞/点踩）+ 后续追问率（追问多说明没答好）
4. **EMA 现状**：老实说没有系统化评估，主要靠用户反馈和 case by case 调优。这是改进项

**追问预案**：
- Q：有没有自动化评估？→ A：可以用 LLM-as-judge，让 LLM 评估答案质量。但 LLM 评估有偏差，只能做粗筛
- Q：怎么知道是检索问题还是生成问题？→ A：分开测——先看检索召回的 doc 相关不相关，再看 LLM 基于这些 doc 的答案准不准。EMA 我加了 sources 面板前端展示检索来源，方便定位

---

## 高频快问快答（30 秒内答完）

| 问题 | 答案要点 |
|------|---------|
| BGE-M3 维度？ | 1024 |
| pgvector 索引？ | ivfflat + cosine_ops |
| 衰减公式？ | R=e^(-t/S), S=1+recall×2 |
| 四级阈值？ | 0.92合并 / 0.75冲突 / 0.60关联 / <0.60新增 |
| Agent 几个节点？ | 5 个：call_llm/check_approval/tools/check_conflict/generate_final |
| 几个 HITL 卡点？ | 2 个：写前审批 + 冲突仲裁 |
| 几个 tool？ | 8 个 |
| 冲突解决几选项？ | 4 个：keep_existing/overwrite/merge/keep_both |
| LangGraph 持久化？ | PostgresSaver，thread_id 关联 |
| 为什么不用 Neo4j？ | 一度关系 SQL 够用，避免双写一致性 |
| 为什么不做多租户？ | 用 project 字段软隔离，等拐点 |
| LLM 容错原则？ | fails safe，不阻塞主流程 |
