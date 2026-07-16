# ADR-004 Model Provider Abstraction


## Status

Accepted


## Context

LLM 和 Embedding 模型会持续变化。


不应该让业务代码绑定具体模型。


---

## Decision


增加抽象层：

```
ModelProvider

EmbeddingService
```


业务代码只依赖接口。


---

## Benefits


支持切换：

- DeepSeek
- OpenAI
- Claude
- Local Model


降低模型迁移成本。


---

## Consequences


增加：

- 接口设计成本


获得：

- 更好的扩展性
- 更强的工程能力展示