# 绝对约束

所有 Claude Code 会话自动加载本文件。以下是不可绕过的硬约束。

---

## 技术栈边界

| 层级 | 必须使用 | 禁止使用 |
|------|---------|---------|
| Agent 框架 | LangGraph 单 Agent | LangChain Agent, Multi-Agent 架构 |
| 数据库 | PostgreSQL + pgvector 作为唯一数据存储方案 | 引入或替换为其他数据库 |
| LLM 调用 | LLMProvider 抽象接口 | 业务代码直接 import openai / anthropic |
| Embedding | EmbeddingProvider 抽象接口 | 业务代码直接依赖具体模型 |
| 后端 | Python 3.12 + FastAPI | — |
| 前端 | Streamlit（MVP 阶段） | MVP 阶段引入 React 框架 |

## 设计原则

- **简单优先**：选择最简单的可行方案。避免过度设计、提前优化、为未来需求增加复杂度
- **Tool 与 Agent 解耦**：Tool 通过接口调用，避免业务逻辑散落在 Agent 中
- **组件可替换**：LLM、Embedding 通过抽象接口切换，不绑定单一模型或 provider
- **分层职责清晰**：Frontend → Backend → Agent → Memory → Storage，每层职责边界明确，不跨层调用

## 架构变更要求

任何架构调整必须先说明：

1. 当前问题
2. 修改原因
3. 设计方案
4. 影响范围

## 绝对禁止

- 未经用户明确确认，不得更换核心技术栈
- 引入新的核心框架

## 工具调用规范

- **每个回复必须产出用户可见的文字**——即使只是状态简报（如"已完成 X，正在做 Y"）
- 工具调用和文字输出必须在同一个 turn 中完成，不允许只调工具不说话
- 任务完成后，用文字列出：做了什么、结果是什么、下一步建议
- 遇到阻塞或不确定时，用文字说明情况并等待用户指示，禁止反复重试同类操作
- 同一 turn 内禁止连续调用超过 5 次同类工具（如连续 5 次 Bash），超出则必须先产出文字说明原因

Git 操作相关禁止项见 [git-workflow.md](git-workflow.md)。
