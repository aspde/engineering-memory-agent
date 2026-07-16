# Agent Design


## Framework

LangGraph


---

# Agent Workflow


```
用户问题

 ↓

Intent Classification

 ↓

Agent Planning

 ↓

Tool Execution

 ↓

Memory Retrieval

 ↓

Context Assembly

 ↓

LLM Generation
```


---

# Agent Capabilities


## Code Search

查询代码仓库。


## Git History Search

查询：

- Commit
- Diff
- Author
- File History


## Memory Search

查询历史研发知识。


## Answer Generation

结合上下文生成回答。


---

# Design Goals


支持：

- 多步骤任务
- 状态管理
- Tool Calling
- Long-term Memory