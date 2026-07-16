# Architecture Rules

系统采用：

FastAPI
    |
LangGraph
    |
Retriever
    |
pgvector


禁止改变核心架构。


新增模块必须说明：
- 输入
- 输出
- 依赖
- 数据流