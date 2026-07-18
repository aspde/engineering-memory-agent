# Git 工作流规则

## 基本原则

Git 用于：

- 代码版本管理
- 变更追踪
- 协作开发


所有代码修改应保持：

- 提交清晰
- 变更可追踪
- 历史可理解


---

# 提交规范

Commit Message 使用 Conventional Commits 格式。


格式：

```
<type>: <description>
```


类型：

- feat: 新功能
- fix: 修复问题
- refactor: 重构
- test: 测试相关
- docs: 文档修改
- chore: 工程配置修改


示例：

```
feat: add memory retrieval workflow

fix: fix vector search error

docs: update architecture document
```


---

# 提交原则

一次提交应该：

- 聚焦单一目的
- 包含完整变更
- 易于理解


避免：

- 一个提交包含多个无关功能
- 提交大量临时修改
- 使用无意义提交信息


禁止：

```
update
modify
test
change
```


作为提交说明。


---

# 分支规范

使用：

```
master
```

作为稳定分支。


功能开发使用：

```
feature/<name>
```


修复使用：

```
fix/<name>
```


示例：

```
feature/memory-retrieval

fix/vector-search
```


---

# 提交前检查

提交前确认：

- 代码可以运行
- 测试通过
- 没有敏感信息
- 没有临时文件


---

# Git 操作规则

执行以下操作前需要确认：

- 删除分支
- 修改提交历史
- rebase
- 强制推送


禁止：

- 自动执行 git reset --hard
- 自动删除代码历史
- 自动覆盖远程分支