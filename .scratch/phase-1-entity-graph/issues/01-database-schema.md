# 01 — 数据库迁移：entities + memory_entities 表

**What to build:** 创建 `entities` 和 `memory_entities` 两张新表，支持实体归一化存储和记忆-实体多对多关联。不修改现有 `memories` 表结构。

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] `entities` 表创建完成：`id UUID PK`、`name TEXT`、`canonical_name TEXT`、`type TEXT`（technology/person/project/decision/event/file/concept）、`embedding vector(1024)`、`first_seen_at TIMESTAMPTZ`、`UNIQUE(canonical_name, type)`
- [ ] `memory_entities` 关联表创建完成：`memory_id UUID FK → memories(id) ON DELETE CASCADE`、`entity_id UUID FK → entities(id) ON DELETE CASCADE`、`PRIMARY KEY (memory_id, entity_id)`
- [ ] `entities.embedding` 上有 ivfflat 余弦索引（与现有 chunks/memories 索引风格一致）
- [ ] 迁移集成到 `backend/db/schema.py` 的 `init_db()` 中，沿用现有 `CREATE TABLE IF NOT EXISTS` 模式
- [ ] 回溯填充脚本可用（独立 SQL/脚本文件，手动执行），逻辑为：遍历现有 `memories.entities` JSONB → 对每个实体名调用嵌入 → 插入 entities 表 → 填充 memory_entities
