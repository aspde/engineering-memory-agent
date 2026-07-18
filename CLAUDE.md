# Engineering Memory Agent

## 项目目标

构建面向研发团队的长期记忆智能体。

核心目标：

- 沉淀研发过程中的技术知识
- 理解代码、文档和历史记录
- 提供团队知识检索能力
- 提升研发协作效率


---

# 核心技术栈


## Backend

- Python 3.12
- FastAPI


## Agent

- LangGraph


## LLM

- OpenAI SDK
- DeepSeek API


## Database

- PostgreSQL
- pgvector


## Frontend

MVP:

- Streamlit

Production:

- React


---

# 核心架构约束


必须遵循：

- LangGraph 负责 Agent 工作流编排
- Memory System 作为独立能力模块
- PostgreSQL + pgvector 作为数据和向量存储
- LLM 调用统一封装
- 技术组件保持可替换


禁止：

- ChromaDB
- MongoDB
- LangChain Agent
- Multi-Agent 架构


---

# Rule 文件


## 开发规则

详细开发规范：

```
.claude/rules/development.md
```


包含：

- 修改前方案设计
- 用户确认机制
- 依赖管理
- 代码修改规范


## 架构规则

详细架构约束：

```
.claude/rules/architecture.md
```


包含：

- 系统分层
- Agent 设计
- Memory 设计
- 数据存储约束


## 测试规则

详细测试规范：

```
.claude/rules/testing.md
```


包含：

- 单元测试
- API 测试
- Agent 测试
- Memory 测试


## Git 工作流规则

Git 管理规范：

```
.claude/rules/git-workflow.md
```


包含：

- Commit 规范
- 分支管理
- 提交检查


---

# 修改要求


修改任何代码前必须：

1. 理解当前实现
2. 分析影响范围
3. 输出实施方案
4. 等待用户确认


未经确认禁止：

- 修改文件
- 创建文件
- 删除文件
- 添加依赖
- 调整架构