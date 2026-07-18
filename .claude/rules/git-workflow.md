# Git 工作流规则

## 提交规范

Commit Message 使用 Conventional Commits 格式：`<type>: <description>`

| 类型 | 用途 | 示例 |
|------|------|------|
| feat | 新功能 | `feat: add memory retrieval workflow` |
| fix | 修复问题 | `fix: fix vector search error` |
| refactor | 重构 | `refactor: extract chunk strategy` |
| test | 测试相关 | `test: add embedding service tests` |
| docs | 文档修改 | `docs: update architecture document` |
| chore | 工程配置 | `chore: update pytest settings` |

一次提交聚焦单一目的，包含完整变更。

禁止以下作为提交说明：

- `update` / `modify` / `test` / `change`
- 提交包含多个无关功能
- 提交大量临时修改

## 分支规范

- 稳定分支：`master`
- 功能开发：`feature/<name>`（如 `feature/memory-retrieval`）
- 修复：`fix/<name>`（如 `fix/vector-search`）

## 提交前检查

- 代码可以运行
- 测试通过（`pytest`）
- 没有敏感信息（API key、密码等）
- 没有临时文件

## Git 操作规则

以下操作执行前需要确认：

- 删除分支
- 修改提交历史（rebase、amend）
- 强制推送

禁止：

- 自动执行 `git reset --hard`
- 自动删除代码历史
- 自动覆盖远程分支
