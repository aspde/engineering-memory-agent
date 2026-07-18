# Engineering Memory Agent

## 项目目标

构建面向研发团队的长期记忆智能体。

核心目标：

- 沉淀研发过程中的技术知识
- 理解代码、文档和历史记录
- 提供团队知识检索能力
- 提升研发协作效率


---

# Rule 文件

项目规则统一存放：

```
.claude/rules/
```


## 开发规则

详细开发规范：

```
.claude/rules/development.md
```

负责：

- 修改前方案设计
- 用户确认机制
- 依赖管理
- 代码修改规范


## 架构规则

详细架构约束：

```
.claude/rules/architecture.md
```

负责：

- 系统分层
- Agent 设计
- Memory 设计
- 数据存储约束


## 测试规则

详细测试规范：

```
.claude/rules/testing.md
```

负责：

- 单元测试
- API 测试
- Agent 测试
- Memory 测试


## Git 工作流规则

Git 管理规范：

```
.claude/rules/git-workflow.md
```

负责：

- Commit 规范
- 分支管理
- 提交检查


## 文档维护规则

文档管理规范：

```
.claude/rules/documentation.md
```

负责：

- 文档目录结构
- 文档更新规则
- ADR 规范
- Claude 行为要求


---

# 修改要求

详见 [development.md](.claude/rules/development.md)：

- 修改前必须规划 → 输出实施方案
- 修改前分析要求 → 理解实现、分析影响
- 用户确认机制 → 等待确认后方可执行