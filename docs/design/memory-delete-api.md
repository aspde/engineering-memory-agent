# 记忆删除 API 设计

## 问题描述

记忆系统当前只支持写入（含去重合并、冲突检测），没有删除能力。写入错误或已过时的记忆无法从系统中清理，前端也无法移除不需要的记忆条目。需要新增软删除 API，所有查询自动排除已删除记忆。

## 方案选择

### 硬删除 vs 软删除

| | 硬删除 | 软删除 (选定) |
|---|--------|---------------|
| 实现 | `DELETE FROM memories WHERE id = :id` | `UPDATE memories SET deleted_at = NOW()` |
| 恢复 | 不可恢复 | 手动 SQL 恢复 |
| 数据安全 | 误删即永久丢失 | 数据保留，可审计 |
| 查询影响 | 自动排除 | 需加 `WHERE deleted_at IS NULL` |
| Schema 变更 | 无 | 加 `deleted_at TIMESTAMPTZ` |

选择软删除：记忆系统对写入有精心设计的四级去重流水线（merge/conflict/supplement/insert），删除也同样应该谨慎。

### 不包含 undo/restore 端点

保持简单。如果真需要恢复，一句 SQL 的事：`UPDATE memories SET deleted_at = NULL WHERE id = '...'`。

---

## 数据模型变更

`memories` 表加一列：

```sql
ALTER TABLE memories ADD COLUMN deleted_at TIMESTAMPTZ;
```

迁移方式沿用 `updated_at` 的模式（`schema.py` 中 `DO $$ ... EXCEPTION WHEN duplicate_column` 幂等块）。无需索引 — `deleted_at IS NULL` 在绝大多数记忆未删除时选择度极低，索引无帮助。

---

## 接口定义

### 新增端点

```
DELETE /api/memory/memories/{memory_id}
```

**正常响应** (200):
```json
{"id": "uuid", "deleted": true}
```

**异常响应** (404):
```json
{"detail": "Memory not found or has been deleted"}
```

**Pydantic 模型**：

```python
class MemoryDeleteResponse(BaseModel):
    id: str
    deleted: bool
```

### 已有端点修改

| 端点 | 变更 |
|------|------|
| `GET /api/memory/memories/{id}` | 查询加 `AND deleted_at IS NULL`，无结果返回 404 |
| `GET /api/memory/stats` | 7 个聚合查询全部加 `WHERE deleted_at IS NULL` |
| `POST /api/memory/memories/search` | 底层 `search_memories()` 加 `AND deleted_at IS NULL` |
| `POST /api/memory/memories/write` | `_find_similar()` 加 `AND deleted_at IS NULL`（不对已删除记忆去重） |

---

## 影响范围

### Backend（4 个文件修改 + 1 个测试新增）

```
backend/db/schema.py                          ← 加 deleted_at 迁移块
backend/service/memory.py                      ← _find_similar() 加过滤条件
backend/service/decay.py                       ← search_memories() 加过滤条件
backend/api/routes/memory_routes.py            ← 新 DELETE 端点 + 7 个统计查询加过滤 + GET 加过滤
tests/api/test_memory_routes.py                ← 新增 DELETE 端点测试
```

**查询过滤需改（3 处）**：

| 函数 | 文件 | 变更 |
|------|------|------|
| `_find_similar()` | memory.py | `WHERE embedding IS NOT NULL AND ...` → 加 `AND deleted_at IS NULL` |
| `search_memories()` | decay.py | `WHERE embedding IS NOT NULL AND ...` → 加 `AND deleted_at IS NULL` |
| `memory_stats()` | memory_routes.py | 7 个聚合查询全部加 `WHERE deleted_at IS NULL` |
| `get_memory_by_id()` | memory_routes.py | 加 `AND deleted_at IS NULL` |

**不需要改的查询**：按 ID 操作的 UPDATE/SELECT（`_merge_memory`、`_insert_memory`、`resolve_conflict`、`update_decay`）——它们操作的是已知未被删除的记忆。

### Frontend（5 个文件）

```
frontend/src/api/client.ts                     ← 加 apiDelete<T>() 方法
frontend/src/api/memory.ts                     ← 加 deleteMemory(id) 函数
frontend/src/types/index.ts                    ← 加 DeleteMemoryResponse 类型
frontend/src/hooks/useMemories.ts              ← 加 deleteMemory 回调，搜索列表本地移除
frontend/src/components/MemoryCard.tsx         ← 加删除按钮 + onDelete prop
frontend/src/components/MemorySearch.tsx       ← 透传 onDelete 给 MemoryCard
```

---

## 实施步骤

1. **Schema**：在 `schema.py` 的 `_STATEMENTS` 末尾追加 `DO $$ ... ADD COLUMN deleted_at` 迁移块
2. **Backend 查询过滤**：在 `_find_similar`、`search_memories`、`get_memory_by_id`、`memory_stats` 的 SQL 中加 `deleted_at IS NULL`
3. **Backend DELETE 端点**：新增 `DELETE /api/memory/memories/{memory_id}`，Pydantic response model `MemoryDeleteResponse`
4. **Frontend API 层**：`apiDelete` helper + `deleteMemory` 函数 + 类型定义
5. **Frontend UI**：`useMemories` 加 `deleteMemory`，`MemoryCard` 加删除按钮（confirm 后调用）
6. **测试**：`tests/api/test_memory_routes.py` 增加 DELETE 的 5 个测试用例

---

## 验证

### pytest

```
正常删除 → 200 {"id": "...", "deleted": true}
删除不存在的 ID → 404
删除已删除的 → 404
删除后 search 不返回该记忆
删除后 stats 计数正确（total 减 1，source_type 分布对应更新）
删除后 GET /memories/{id} → 404
```

### curl

```bash
# 1. 删除
curl -X DELETE http://localhost:8000/api/memory/memories/{id}

# 2. 确认查不到
curl http://localhost:8000/api/memory/memories/{id}

# 3. 搜索不出现
curl -X POST http://localhost:8000/api/memory/memories/search \
  -H 'Content-Type: application/json' -d '{"query":"...","top_k":5}'
```
