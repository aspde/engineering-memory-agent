# 02 — 实体归一化管线

**What to build:** `write_memory()` 写入成功后自动触发实体归一化：embed 实体名 → pgvector 搜索已有相似实体 → LLM 确认匹配 → 链接已有或创建新实体。归一化失败不影响记忆写入。

**Blocked by:** 01 — 需要 entities/memory_entities 表存在。

**Status:** ready-for-agent

- [ ] `normalize_entities(memory_id, extracted_entities)` 服务函数实现（新建 `backend/service/entity.py` 或扩展现有 service）
- [ ] 归一化流程完整：embed(entity_name) → `entities` 表余弦相似度搜索 top-3（阈值 > 0.85）→ LLM 判断（prompt 极简："Are `X` and `Y` the same technology entity? Reply YES or NO."）
- [ ] LLM 确认匹配时：链接到已有 entity，插入 `memory_entities` 记录
- [ ] LLM 拒绝或无候选时：INSERT 新 entity → 插入 `memory_entities` 记录
- [ ] LLM 调用失败时：安全降级，创建新实体（不阻塞记忆写入）
- [ ] `write_memory()` 的 `_insert_memory()` 成功后调用 `normalize_entities()`，失败不影响已有返回
- [ ] `MemoryWriteResponse` 新增 `entity_ids: list[str]` 字段
- [ ] 回溯填充：`normalize_all_existing()` 批量版本，遍历现有 memories 的 entities JSONB 执行归一化（手动运行）
- [ ] 单元测试覆盖：新实体创建、相似实体链接、LLM 确认匹配、LLM 拒绝匹配、LLM 失败降级、空实体列表
