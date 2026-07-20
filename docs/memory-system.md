# Memory System Design

## 设计理念

EMA 的记忆系统不是 LangChain 那样的黑盒 RAG 管道，而是**一组独立可替换的异步函数**。每个环节单独调用、单独测试、单独替换，不通过框架链粘合。

```
Write Path:
  Raw Content → chunk_text() → write_chunks() → pgvector (chunks 表)
  Raw Content → extract_memory() → write_memory() → pgvector (memories 表)

Read Path:
  Query → embed_query() → vector_search() → rerank() → assemble() → LLM
  Query → query_memories() → decay_weighted_search → update_decay() → rerank
```

每步一个函数，没有 class wrapper，没有 LangChain retriever/chain。

---

## 核心功能

### 1. 文档索引与检索

**写入**：`write_chunks(document_id, chunks)` 将文本片段 embed 后批量写入 `chunks` 表。

**检索**：`retrieve(query, top_k, use_llm_rerank=False)` 执行完整管道：
- `embed_query()` → `vector_search()` → `rerank_cross_encoder()`（默认）或 `rerank_llm()`（可选）→ 返回 `RetrievalResult` 列表

`vector_search()` 支持可选的 `filters` 参数（如 `{"document_id": "repo.py"}`）限定搜索范围。

两种 reranker 可通过参数切换：
| Reranker | 引擎 | 成本 | 适用 |
|----------|------|------|------|
| `rerank_cross_encoder()` | BGE-Reranker-v2-m3 本地 | 零 API 成本 | 默认 |
| `rerank_llm()` | 现有 LLMProvider | API 调用费用 | 需精细语义判断时 |

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

每个阶段独立调用 LLM，一个失败不影响其他。

### 3. 智能写入与去重

`write_memory(content, source_type)` 在写入前查询已有记忆：

| 相似度 | 行为 |
|--------|------|
| ≥ 0.92 | LLM 合并摘要，合并实体和关系 |
| 0.75–0.92 | LLM 检测矛盾 → 矛盾则标记冲突，否则补充关联 |
| 0.60–0.75 | 插入为新记忆，关联到最相似记忆（复用已有 embedding，不重复调用） |
| < 0.60 | 作为全新记忆插入 |

合并和矛盾检测的 LLM 调用失败时降级处理：合并失败保留原有摘要，矛盾检测失败假定无矛盾——不会阻塞写入。

### 4. 艾宾浩斯遗忘衰减

每次检索记忆时更新衰减因子：

```
R = e^(-t / S)
t = 距上次召回的小时数
S = 1 + recall_count × 2
```

- 新记忆 `decay_factor = 1.0`
- 频繁召回的记忆衰减慢
- 长期未召回的记忆自然沉底

`search_memories(query_vector)` 用 `相似度 × decay_factor` 加权排序。`query_memories(query)` 封装了完整管道：embed → 衰减加权搜索 → rerank（默认 cross-encoder，可选 llm）→ 更新 decay 并返回。

### 5. Chunk 策略

全部自实现，不依赖外部库。

**通用文本** `chunk_text(text, max_size=512, overlap=64)`：递归分隔符切分——按段落 → 行 → 句子 → 词的优先级切分，保证不截断语义单元。overlap 后重新检查大小，超限时继续细分。

**代码文件** `chunk_code(code, max_lines=80)`：AST 感知，按函数/类/模块边界切分。非 Python 代码回退到按行切分。

---

## 数据库 Schema

### chunks — 文档片段 + 向量

| 列 | 类型 | 说明 |
|----|------|------|
| id | UUID PK | gen_random_uuid() |
| document_id | TEXT | 来源文档标识 |
| content | TEXT | chunk 文本 |
| embedding | vector(1024) | BGE-M3 向量 |
| metadata | JSONB | 来源、行号、语言等 |
| chunk_index | INT | 在原文档中的顺序 |
| created_at | TIMESTAMPTZ | now() |

索引：`ivfflat` on `embedding vector_cosine_ops`

### memories — 结构化长期记忆

| 列 | 类型 | 说明 |
|----|------|------|
| id | UUID PK | gen_random_uuid() |
| source_type | TEXT | git_commit / doc / conversation |
| summary | TEXT | LLM 摘要 |
| entities | JSONB | `[{name, type}]` |
| relations | JSONB | `[{from, to, type}]` |
| embedding | vector(1024) | 摘要的 BGE-M3 向量 |
| decay_factor | FLOAT | 艾宾浩斯衰减因子，默认 1.0 |
| recalled_at | TIMESTAMPTZ | 最后一次被检索的时间 |
| recall_count | INT | 累计检索次数，默认 0 |
| metadata | JSONB | 冲突标记、补充关联等 |
| created_at | TIMESTAMPTZ | now() |

---

## 文件结构

```
backend/
  db/
    __init__.py       ← asyncpg + SQLAlchemy 异步引擎，连接池 5+10
    schema.py         ← Chunk / Memory 表 (SQLAlchemy ORM)
  service/
    chunk.py          ← chunk_text(), chunk_code()
    extraction.py     ← extract_summary(), extract_entities(), extract_relations(), extract_memory()
    rerank.py         ← rerank_cross_encoder(), rerank_llm()
    retrieval.py      ← write_chunks(), vector_search(), retrieve(), query_memories(), assemble()
    memory.py         ← write_memory() + 四级相似度判断 + merge/conflict/supplement
    decay.py          ← compute_decay_factor(), update_decay(), search_memories()
```

---

## 设计原则

- **函数优先**：每个功能一个函数，没有不必要的 class wrapper
- **独立可测**：每个函数单独 mock LLM/embedding 即可测试
- **容错降级**：LLM 调用失败不阻塞写入，merge/conflict 检测失败走安全路径
- **不依赖 LangChain**：chunk、retrieval、rerank 全部自实现，不引入链条式黑盒
- **SQL 可见**：向量搜索手写 SQL，`<=>` 操作符和参数完全可控
