# Engineering Memory Agent

## 项目目标

构建研发团队长期记忆智能体。


## 核心技术约束

Backend:
- Python 3.12
- FastAPI

Agent:
- LangGraph

LLM:
- OpenAI SDK
- DeepSeek API

Database:
- PostgreSQL + pgvector

Frontend:
- MVP: Streamlit
- Production: React


## 开发原则

1. 优先简单实现
2. 不随意增加依赖
3. 修改架构前先说明方案


## 禁止

禁止：
- ChromaDB
- MongoDB
- LangChain Agent
- 多Agent设计


## 代码修改流程

修改代码前：

说明：
1. 修改目标
2. 涉及文件
3. 实现方案


完成后：

说明：
1. 修改内容
2. 测试结果
3. 风险