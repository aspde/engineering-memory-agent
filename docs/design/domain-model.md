# EMA 领域模型

> 本文档定义 EMA 的核心领域概念及其关系。是 ADR 的补充——ADR 记录"为什么做这个决策"，领域模型记录"系统由什么组成、它们怎么关联"。

## 核心领域术语

| 术语 | 定义 | 当前状态 |
|------|------|---------|
| **Memory（记忆）** | 从原始内容中提取的结构化长期知识单元。包含摘要、实体、关系、向量、衰减因子。 | ✅ 已实现（`memories` 表） |
| **Chunk（文档片段）** | 文档的向量化文本片段，用于语义检索。轻量级——只有文本和向量，不提取实体。 | ✅ 已实现（`chunks` 表） |
| **Entity（实体）** | 从记忆中提取的命名概念：技术、人、项目、决策、事件、文件。 | ⚠️ 已提取但仅存于 memory JSONB 中，未跨记忆归一化 |
| **Relation（关系）** | 两个实体之间的有向关系：`depends_on`、`causes`、`part_of`、`contradicts`、`supersedes`、`relates_to`。 | ⚠️ 已提取但仅存于 memory JSONB 中，不可跨记忆查询 |
| **Conversation（对话）** | 用户与 EMA Agent 的一次或多次交互，通过 thread_id 关联。 | ✅ 已实现（`conversations` 表 + LangGraph checkpoints） |
| **Source（来源）** | 记忆的出处类型：`git_commit`、`doc`、`conversation`、`api`。 | ✅ 已实现（`memories.source_type`） |
| **Decay（衰减）** | 艾宾浩斯遗忘曲线：`R = e^(-t/S)`，`S = 1 + (recall_count + 1) × 2`。记忆随时间衰减，召回时恢复。 | ✅ 已实现 |
| **Connector（连接器）** | 对接外部系统的输入适配器，将外部数据（PingCode、CI、飞书）转化为 EMA 可消化的内容。 | ❌ 未实现 |
| **Ingestion Pipeline（摄入管线）** | 将原始内容转化为 Memory 或 Chunk 的完整流程：分块 → 嵌入 → 提取 → 去重 → 存储。 | ✅ 已实现 |
| **HITL（人机协同）** | Human-in-the-Loop：Agent 在执行敏感操作前暂停，等待用户审批；或在记忆冲突时等待用户仲裁。 | ✅ 已实现 |

## 领域关系图

```
                         ┌─────────────────────────────┐
                         │        Connector             │
                         │ (PingCode / CI / 飞书 / …)    │
                         └─────────────┬───────────────┘
                                       │ 产生
                                       ▼
┌──────────┐   摄入     ┌─────────────────────────┐
│  Source  │──────────▶│   Ingestion Pipeline     │
│ (git/doc │           │ chunk → embed → extract  │
│  /chat)  │           └──────────┬──────────────┘
└──────────┘                      │
                       ┌──────────┴──────────┐
                       ▼                     ▼
                ┌──────────┐          ┌──────────┐
                │  Chunk   │          │  Memory  │
                │ (文档片段) │          │ (结构化记忆) │
                └──────────┘          └────┬─────┘
                                           │ 提取
                                     ┌─────┴─────┐
                                     ▼           ▼
                               ┌────────┐  ┌──────────┐
                               │ Entity │◄─│ Relation │
                               │ (实体)  │  │  (关系)   │
                               └────────┘  └──────────┘
                                    │            │
                                    └── 关联 ────┘
                                    （跨 Memory 归一化）
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │  Entity Graph │
                                    │（一度关系查询） │
                                    └──────────────┘

┌──────────────┐     使用      ┌──────────────────┐
│ Conversation │◄─────────────│    EMA Agent     │
│  (对话历史)   │─────────────▶│  (ReAct 循环)    │
└──────────────┘    控制       └────────┬─────────┘
                                        │
                          ┌─────────────┴─────────────┐
                          ▼                           ▼
                   ┌────────────┐             ┌──────────────┐
                   │  检索路径   │             │   写入路径    │
                   │ query →    │             │ content →    │
                   │ embed →    │             │ extract →    │
                   │ search →   │             │ similarity → │
                   │ rerank →   │             │ insert/merge │
                   │ decay      │             │              │
                   └────────────┘             └──────────────┘
```

## 实体模型演进路线

### 当前状态

```
memories 表:
  id: UUID
  summary: TEXT
  entities: JSONB   ← [{name: "PostgreSQL", type: "technology"}, ...]
  relations: JSONB  ← [{from: "PostgreSQL", to: "pgvector", type: "depends_on"}, ...]
  embedding: vector(1024)
  decay_factor: FLOAT
  ...
```

**问题**: 实体名称是自由文本——"PostgreSQL"、"Postgres"、"pg16"是三个不同字符串，查询时无法关联。

### Phase 1 目标状态

新增 `entities` 表，实现实体归一化：

```sql
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,              -- 原始名称
    canonical_name TEXT NOT NULL,    -- 归一化名称（LLM 确定）
    type TEXT NOT NULL,              -- technology | person | project | decision | event | file | concept
    embedding vector(1024),          -- canonical_name 的向量
    first_seen_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(canonical_name, type)
);
```

新增 `memory_entities` 关联表：

```sql
CREATE TABLE memory_entities (
    memory_id UUID REFERENCES memories(id),
    entity_id UUID REFERENCES entities(id),
    PRIMARY KEY (memory_id, entity_id)
);
```

归一化流程：

```
新的 Memory 提取出实体 [{name: "pg16", type: "technology"}]
    │
    ▼
embed("pg16") → 在 entities 表中向量搜索已有实体
    │
    ▼
找到候选: {name: "PostgreSQL", canonical_name: "PostgreSQL", similarity: 0.88}
    │
    ▼
LLM 判断: "pg16" 是 "PostgreSQL" 的版本指代 → 是同一实体
    │
    ▼
将 memory 链接到已有 entity (PostgreSQL)，而非创建新实体
```

### 一度关系查询 API

```http
GET /api/entities/{entity_id}/relations
→ {
    entity: {id, name, type},
    related_memories: [
      {memory_id, summary, source_type, relation_type, created_at},
      ...
    ],
    related_entities: [
      {entity_id, name, type, relation_type, count},
      ...
    ]
  }
```

### Phase 2-3 扩展

- **Phase 2**: 连接器引入的外部数据自动提取实体并链接到归一化实体库
- **Phase 3**: 主动 Agent 可沿实体关系发现知识盲区："我们对 PostgreSQL 的性能调优只有 2 条记忆，但它是核心依赖——建议补充"

## 交互模型

```
                    ┌─────────────────────┐
                    │     EMA Backend     │
                    │   (FastAPI + Agent)  │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
   ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
   │   Web UI    │    │  ChatOps Bot │    │  IDE Plugin  │
   │ (聊天+图谱)  │    │ (Slack/飞书) │    │ (VS Code/…)  │
   └─────────────┘    └──────────────┘    └──────────────┘
   现在做               Phase 3 附近         远期
```

| 界面 | 擅长场景 | 做它的时机 |
|------|---------|-----------|
| Web UI | 深度查询、知识图谱探索、摄入管理、仪表盘 | **现在** |
| ChatOps Bot | "帮我记一下"、"之前怎么修的"——低摩擦沉淀 | API 就绪后一个周末 |
| IDE Plugin | 编辑器内查记忆 | 知识库丰富 + 用户习惯养成后 |

## 不做的事

| 概念 | 为什么现在不做 |
|------|-------------|
| Workspace / Team（团队空间） | EMA 的价值是打破知识孤岛，硬隔离反模式；用 metadata tag 满足基本分组需求 |
| User / Role / Permission（用户权限） | 单团队场景下不需要；SaaS 化时再引入 |
| Graph Database（图数据库） | 详见 ADR-005；一度关系用 SQL，复杂图遍历不是当前需求 |
| Event Bus / Message Queue | Webhook 已在进程内异步处理（后台任务 + 并发上限），持久化消息队列等负载上来再加 |

## 与现有文档的关系

- **架构**: 参见 [`architecture.md`](../architecture.md) — 分层设计和全链路
- **Agent 设计**: 参见 [`agent-design.md`](../agent-design.md) — LangGraph ReAct 循环
- **记忆系统**: 参见 [`memory-system.md`](../memory-system.md) — 提取、去重、衰减
- **决策记录**: 参见 [`decisions/`](../decisions/) — 所有 ADR
