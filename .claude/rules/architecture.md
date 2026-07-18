# 架构规则

## 项目定位

本项目为 Engineering Memory Agent。

目标：

构建面向研发团队的长期记忆智能体，用于：

- 研发知识沉淀
- 技术经验检索
- 代码与文档理解
- 历史决策查询
- 团队知识复用


---

# 架构原则

系统采用分层架构：

- Frontend Layer
- Backend Layer
- Agent Layer
- Memory Layer
- Storage Layer
- Tool Layer


各层职责必须保持清晰。


---

# Frontend Layer

技术：

MVP:

- Streamlit


Production:

- React


职责：

- 用户交互
- 请求提交
- 结果展示


Frontend 不负责：

- Agent 编排
- Memory 管理
- 数据处理逻辑


---

# Backend Layer

技术：

- Python 3.12
- FastAPI


职责：

- 提供 API 接口
- 管理请求生命周期
- 调用 Agent
- 处理业务逻辑


Backend 不负责：

- 复杂 Agent 推理流程
- Memory 检索逻辑


---

# Agent Layer

技术：

- LangGraph


职责：

- Agent 工作流编排
- 状态管理
- 节点执行
- Tool 调用
- Memory 调用


设计要求：

- 使用单 Agent 架构
- 状态流转明确
- 节点职责清晰


禁止：

- LangChain Agent
- Multi-Agent 架构


---

# Memory Layer

Memory System 是 Agent 的核心能力。


负责：

- 长期记忆管理
- Memory 检索
- 上下文构建
- 历史知识关联


Memory 类型：

## 短期记忆

用于：

- 当前会话状态
- Agent 执行状态


## 长期记忆

用于：

- 技术知识
- 架构决策
- 问题解决方案
- 研发历史记录


---

# Storage Layer

数据库：

- PostgreSQL
- pgvector


PostgreSQL 用于：

- 业务数据
- Memory 元数据
- 结构化信息


pgvector 用于：

- Embedding 存储
- 向量检索


禁止：

- MongoDB
- ChromaDB


---

# Tool Layer

Tools 用于连接研发资源。


包括：

- Git 信息解析
- 代码分析
- 文档读取
- 项目知识提取


要求：

- Tool 与 Agent 解耦
- Tool 通过接口调用
- 避免业务逻辑散落


---

# RAG 架构要求

RAG 流程：

数据输入

→ 文档处理

→ Chunk 切分

→ Embedding

→ 向量存储

→ Retrieval

→ Context 构建

→ LLM 生成


要求：

- Retrieval 独立封装
- Embedding 模型可替换
- 不绑定单一模型


---

# LLM 设计

支持：

- OpenAI SDK
- DeepSeek API


要求：

- 统一封装模型调用
- 支持配置切换模型
- 业务代码不直接依赖具体模型


---

# 架构变更规则

任何架构调整必须先说明：

- 当前问题
- 修改原因
- 设计方案
- 影响范围


未经确认禁止：

- 更换核心技术栈
- 修改 Agent 架构
- 引入 Multi-Agent
- 替换数据库方案
- 替换 Memory 存储方案
- 引入新的核心框架


---

# 设计原则

## 简单优先

优先：

- 简单设计
- 清晰边界
- 易维护方案


避免：

- 过度设计
- 提前复杂化
- 为未来未知需求增加复杂度