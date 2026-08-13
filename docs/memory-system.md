# Memory System Design

## 设计理念

EMA 的记忆系统不是 LangChain 那样的黑盒 RAG 管道，而是**一组独立可替换的异步函数**。每个环节单独调用、单独测试、单独替换，不通过框架链粘合。

```
Write Path:
  Raw Content → chunk_text() → write_chunks() → pgvector (chunks 表)
  Raw Content → extract_memory() → write_memory() → pgvector (memories 表)
  Git Repo   → ingest_repo() → write_memory() × N → pgvector (memories 表)

Read Path:
  Query → embed_query() → vector_search() → rerank() → assemble() → LLM
  Query → query_memories() → record_recalls() → rerank
```

每步一个函数，没有 class wrapper，没有 LangChain retriever/chain。

---

## 核心功能

### 1. 文档索引与检索

**写入**：`write_chunks(document_id, chunks)` 将文本片段 embed 后批量写入 `chunks` 表。

`write_chunks` 是幂等的：每个 chunk 计算 SHA-256（`content_hash`），按 `(document_id, content_hash)` 唯一索引去重——重复摄取同一文档会跳过已存在的 chunk（不重新嵌入、不产生重复行），`ON CONFLICT DO NOTHING` 兜底并发竞争。文档内容变更时，仅新增/变更的 chunk 会被写入。

**检索**：`retrieve(query, top_k, use_llm_rerank=False, use_cross_encoder=False)` 执行完整管道：
- `embed_query()` → `vector_search()` → 默认按相似度直接排序返回 `RetrievalResult` 列表；`rerank_cross_encoder()` 与 `rerank_llm()` 均为显式 opt-in（默认关闭）

`vector_search()` 支持可选的 `filters` 参数（如 `{"document_id": "repo.py"}`）限定搜索范围。

两种 reranker 均可通过参数显式开启（默认关闭，opt-in）：
| Reranker | 引擎 | 成本 | 适用 |
|----------|------|------|------|
| `rerank_cross_encoder()` | BGE-Reranker-v2-m3 本地 | 零 API 成本 | opt-in；小语料下实测掉 recall 且慢 ~90 倍（eval-report），默认关闭 |
| `rerank_llm()` | 现有 LLMProvider | API 调用费用 | opt-in；需精细语义判断时 |

**hybrid 检索默认跳过 rerank**：`retrieve_hybrid(query, top_k, use_llm_rerank=False, skip_rerank=True)` 默认按 **RRF（reciprocal rank fusion）** 融合 dense 与 sparse 两个列表的名次直接排序——以名次而非原始分数融合，规避 cosine 与 jaccard 两种分布不可比、`max(dense, sparse)` 会被分布更热的检索器主导的问题；同时被两个检索器召回的 chunk 会获得交叉验证的加权。融合分数按最大可达值（双列表 #1）归一化到 0-1 相似度刻度，语义是**相对排序信号**而非绝对相似度。eval 报告（`tests/eval/reports/eval-report.md`）显示当前语料下 cross-encoder rerank 延迟高 ~90 倍且 recall 反而更低（0.967 vs 1.000，0.15 floor 误伤 q015），故 rerank 为 opt-in（传 `skip_rerank=False` 走 cross-encoder，或 `use_llm_rerank=True` 走 LLM）。rerank 收益 scale-dependent，万级语料候选池覆盖率下降时需重新评估。

### 2. 三阶段记忆提取

`extract_memory(content)` 将原始内容转化为结构化记忆，三个阶段两个并行：

```
extract_summary(content) ──┐
                           ├─ asyncio.gather (并行)
extract_entities(content) ─┘
                           │
                           └─ extract_relations(summary, entities)
```

- **摘要**：2-5 句简洁段落
- **实体**：JSON 数组 `[{name, type}]`，类型包括 person/project/technology/decision/event/file/concept
- **关系**：JSON 数组 `[{from, to, type}]`，类型包括 depends_on/causes/part_of/contradicts/supersedes/relates_to

每个阶段独立调用 LLM，一个失败不影响其他（各阶段的失败语义见「分层容错」）。

### 3. 智能写入与去重

写入是**内容哈希幂等**的：`write_memory` 先对原始内容计算 SHA-256（`content_hash`，memories 表唯一索引）。若该内容已存在（重新摄取同一 commit、同一文档、重放的 webhook），直接返回 `action="duplicate"`，不执行任何 LLM 提取。此处的"完全重复"去重与下面的相似度去重（"近似重复"）互补，顺序为：先哈希精确去重，再相似度分级。

`write_memory(content, source_type)` 在写入前查询已有记忆：

| 相似度 | 行为 |
|--------|------|
| ≥ 0.85 | LLM 合并摘要，合并实体和关系 |
| 0.72–0.85 | LLM 检测矛盾 → 矛盾则标记冲突，否则补充关联 |
| 0.60–0.72 | 插入为新记忆，关联到最相似记忆 |
| < 0.60 | 作为全新记忆插入 |

阈值经标定（`tests/eval/reports/archive/threshold_calibration_report.md`）：同义改写对的相似度 p25 为 0.878、同类不同记忆上限 0.792，0.85 是自然分离点。旧值 0.92 高到「同一知识被不同来源写出来」的 merge 一半不触发。

合并和矛盾检测均为结构化 LLM 调用（JSON-schema 校验 + 重试，见「分层容错」）。合并失败保留原有摘要（合并是自由文本，失败成本是"少合并一次"）；矛盾检测失败则**降级为 supplement 关联写入**（`write_memory` 的 failsafe）——不假定矛盾（那会丢弃内容或把非矛盾误路由到 HITL），也不把新内容无标记写进冲突记忆，内容以补充关联保留；检测遗漏的矛盾由每周巡检的全量矛盾扫描兜底。实体/关系提取失败（增强类）在重试耗尽后降级为 `[]`，但会记 ERROR 日志 + 失败计数（`ema_structured_failures_total`），写入继续。

检测到矛盾时，两种路径都进入人工处理（HITL），绝不静默丢弃：**agent 路径**通过 `interrupt()` 暂停对话等待选择；**webhook/连接器路径**没有交互会话，冲突内容落入 `pending_conflicts` 队列（保存 `_deferred` 载荷），由人通过 `GET/POST /api/conflicts` 用相同的四种选项（keep_existing / overwrite / merge / keep_both）经同一个 `resolve_conflict()` 解决。队列对同一冲突（同一 `existing_id` + 同一内容哈希）只保留一条 pending 记录——webhook 至少一次投递的重放不会堆叠重复行；已解决的行离开去重范围，同类内容再次出现会正常重新入队。

**巡检矛盾**是第三条来源（`conflict_type='patrol'`）：每周巡检的全量矛盾扫描发现两条**都已在库**的记忆（A、B）结论相反，但写入时检测没拦住（embedding 距离远、或检测上线前就存在的旧矛盾）。这类矛盾由 Patrol 页手动逐条「转入仲裁」（入队成功自动忽略该 finding），入队到同一个 `pending_conflicts` 队列，`peer_id` 列记录被放弃方（B），`_deferred` 载荷把 B 的完整内容作为"新侧"。解析走与 ingestion 相同的 `resolve_conflict()`（以 `peer_id=B` 传入）——四选项语义重解释为「保留侧 A + 被放弃方 B」：keep_existing / overwrite / merge 都软删 B（`memories.deleted_at`），B 从检索与巡检中消失，矛盾不再重报；keep_both 不改任何记忆行，靠队列里已 resolved 的记录抑制重复入队。`resolve_pending_conflict` 按 `peer_id` 是否有值进入同一流水线的两个语义，仍是同一个裁决出口。

已仲裁的巡检对子（`status='resolved'`）可通过 `GET /api/conflicts?status=resolved&conflict_type=patrol` 查看台账，误选的 keep_both 可用 `POST /api/conflicts/{id}/reopen` 重置回待处理重新仲裁——仅 patrol 冲突可重新打开，且被放弃方 B 仍存活时才允许（keep_existing / overwrite / merge 已软删 B，重开会在缺失记忆上仲裁，故拒绝）。

### 4. 召回统计（替代原艾宾浩斯衰减）

> 衰减加权的移除决策见 [ADR-009](./decisions/ADR-009-decay-weighting-removed.md)，A/B 数据见 `tests/eval/reports/decay_ab_report.md`。

记忆检索按**纯相似度**排序；每次检索会把命中的记忆记一次召回（`recall_count` + 1、`recalled_at = NOW()`），作为**元数据**而非排序信号。

原实现曾用艾宾浩斯遗忘曲线把 `decay_factor` 乘进相似度排序，但 decay A/B 实测（`tests/eval/reports/decay_ab_report.md`）显示衰减加权 recall@5 0.667、无衰减 0.900——唯一的测量数据表明衰减让检索变差，且其前提「近期/高频=相关」建立在合成老化分布上、没有真实语料支撑。调参三轮（S=2x→8x→12x）本质是把曲线调得越来越接近 no-op。故从排序路径移除衰减，保留召回计数作访问历史。

`search_memories(query_vector)` 单段 HNSW 按 `embedding <=> :vec` 直接返回相似度排序，不再需要两段检索（候选窗 + Python 按 `similarity × decay_factor` 重排）。`query_memories(query)` 封装完整管道：embed → 纯相似度搜索 →（默认无 rerank，cross-encoder/LLM 为 opt-in）→ `record_recalls()`（单条 `UPDATE ... WHERE id = ANY(:ids)` 批量递增 `recall_count`/`recalled_at`，无 N+1 提交、并发不丢计数；从未召回的记忆记首次召回）→ 返回。

记忆「是否过期」由人/LLM 基于召回历史判断：`search_memories_tool` 的展示行带 `recalls` 与 `last_recalled`，每周 patrol 的过期记忆扫描让 LLM 直接读这两个字段（从未召回或久未召回 → 归档候选），不再依赖机器算出的衰减因子。

### 5. Chunk 策略

全部自实现，不依赖外部库。

**通用文本** `chunk_text(text, max_size=512, overlap=64)`：递归分隔符切分——按段落 → 行 → 句子 → 词的优先级切分，保证不截断语义单元。句子边界同时识别拉丁标点（`[.!?]` 后跟空白）与中文标点（`。！？` 零宽切分——中文句子以这些标记结尾且通常无尾随空格，故单独一条规则），长中文段落（commit 消息、讨论、文档）按句切分而非硬切在句中。overlap 后重新检查大小，超限时继续细分。

**代码文件** `chunk_code(code, max_lines=80)`：AST 感知，按函数/类/模块边界切分。非 Python 代码回退到按行切分。

### 6. Git 历史摄取

`ingest_repo(repo_path)` 通过 pygit2 遍历 git 历史，每个 commit 格式化为结构化文本后经 `write_memory()` 进入记忆管线。Diff 截断到 16KB。

---

## 数据库 Schema

### chunks — 文档片段 + 向量

| 列 | 类型 | 说明 |
|----|------|------|
| id | UUID PK | gen_random_uuid() |
| document_id | TEXT | 来源文档标识 |
| content | TEXT | chunk 文本 |
| embedding | vector(N) | 向量，维度由 EMBEDDING_MODEL 决定（默认 BGE-M3 = 1024） |
| meta | JSONB | 来源、行号、语言等 |
| chunk_index | INT | 在原文档中的顺序 |
| created_at | TIMESTAMPTZ | now() |

索引：`hnsw` on `embedding vector_cosine_ops`（pgvector ≥ 0.5；HNSW 无聚类质心依赖，小库召回稳定——替代了早期 `ivfflat lists=100` 在小数据量下聚类不稳的问题）

### memories — 结构化长期记忆

| 列 | 类型 | 说明 |
|----|------|------|
| id | UUID PK | gen_random_uuid() |
| source_type | TEXT | git_commit / doc / conversation |
| summary | TEXT | LLM 摘要 |
| entities | JSONB | `[{name, type}]` |
| relations | JSONB | `[{from, to, type}]` |
| embedding | vector(N) | 摘要向量，维度同 embedding 配置 |
| recalled_at | TIMESTAMPTZ | 最后一次被检索的时间 |
| recall_count | INT | 累计检索次数，默认 0 |
| meta | JSONB | 冲突标记、补充关联等 |
| created_at | TIMESTAMPTZ | now() |
| content_hash | TEXT | 原始内容 SHA-256，精确去重键（未软删行唯一，见 §3 内容哈希幂等） |
| updated_at | TIMESTAMPTZ | 最近一次更新（合并/覆写时刷新） |
| deleted_at | TIMESTAMPTZ | 软删除时间戳；非 NULL 的记忆退出检索与巡检（见 §3 巡检 peer_id 软删） |

### 维度（切换 embedding 模型）

`embedding` 列维度由 `config.embedding.dimension` 决定（模型名 → 已知维度映射，见 `backend/shared/config.py`），不再硬编码 1024。schema 由 Alembic 迁移统一管理（baseline 用 1024 作占位维度），`init_db()` 启动时**校验**实时列维度与配置维度是否一致：不一致直接启动失败（错误信息给出对齐 `EMBEDDING_MODEL` 或重建库两条出路），绝不自动清空。

选择「校验而非自动迁移」：pgvector 的向量只能放进「维度 ≤ 自身」的列，改大改小都要求列先为空，所以自动迁移的本质是清空整个向量列——破坏性操作不该在每次启动的静默路径上发生。对开发工具，换模型 = 重建库（数据可重摄取）：

```bash
python -m scripts.recreate_db        # 清空 public schema + alembic upgrade head
```

若出现 `embedding IS NULL` 的行（写入中断或手动清空后），可用重嵌脚本补齐：

```bash
python -m scripts.reembed_embeddings --dry-run   # 先查看有多少行待重嵌
python -m scripts.reembed_embeddings             # chunks + memories 分批重嵌（可重复执行）
```

脚本只处理 `embedding IS NULL` 的行，幂等且安全。

---

## 文件结构

```
backend/
  db/
    __init__.py       ← asyncpg + SQLAlchemy 异步引擎，连接池 5+10
    schema.py         ← Chunk / Memory 表 (raw SQL)
  service/
    chunk.py          ← chunk_text(), chunk_code()
    extraction.py     ← extract_summary(), extract_entities(), extract_relations(), extract_memory()
    rerank.py         ← rerank_cross_encoder(), rerank_llm()
    recall.py         ← record_recalls() 批量记召回（N+1 修复）
    retrieval.py      ← write_chunks(), vector_search(), search_memories(), query_memories(), assemble()
    memory.py         ← write_memory() + 四级相似度判断 + merge/conflict/supplement
    ingestion.py      ← ingest_repo() via pygit2
```

---

## 设计原则

- **函数优先**：每个功能一个函数，没有不必要的 class wrapper
- **独立可测**：每个函数单独 mock LLM/embedding 即可测试
- **分层容错**：结构化输出经 `response_format`/forced tool 强制 + `jsonschema` 校验 + 有界重试（`chat_structured`）。重试耗尽后，增强类（实体/关系提取）**大声降级**——ERROR 日志 + 失败计数（`ema_structured_failures_total`），写入继续；矛盾检测（正确性关键类）失败则**降级为 supplement 关联写入**——不假定矛盾（不丢弃内容、不误路由 HITL），也不把新内容无标记写进冲突记忆，检测遗漏由每周巡检的全量矛盾扫描兜底
- **不依赖 LangChain**：chunk、retrieval、rerank 全部自实现，不引入链条式黑盒
- **SQL 可见**：向量搜索手写 SQL，`<=>` 操作符和参数完全可控
