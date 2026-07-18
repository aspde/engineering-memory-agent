# Git Knowledge Extraction

## Library

GitPython — 当前使用，用于解析 Git 仓库。

## Extract Information

支持提取：

- Commit 记录（message、author、date）
- Diff 变更内容
- 文件变更列表
- Branch 信息
- Tag 信息
- 文件历史

## Pipeline

```
Git Repo → GitPython → Parse → Chunk → Embedding → Store → Retrieval
```

## Development Phases

### Phase 1 ✅

获取基础信息：commit message、author、changed files

### Phase 2 🔜

- Commit diff 解析
- LLM 总结 commit 内容
- 自动生成 Memory

### Phase 3 🔜

扩展到 Issue、Pull Request、Code Review、ADR

## Integration with Memory

Git 提取的知识通过 Memory Pipeline 存入长期记忆：

- 结构化信息 → PostgreSQL
- 语义内容 → pgvector Embedding
