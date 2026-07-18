# ADR-004: Model Provider Abstraction

## Status

Accepted

## Context

LLM 和 Embedding 模型会持续变化，不应让业务代码绑定具体模型。

## Decision

通过抽象接口统一封装：

- `LLMProvider` — LLM 调用抽象
- `EmbeddingProvider` — Embedding 抽象

业务代码只依赖接口，不直接依赖具体 SDK。

## Implementation

| Interface | Implementations |
|-----------|----------------|
| `LLMProvider` | `OpenAICompatibleProvider` (DeepSeek/OpenAI), `AnthropicProvider` |
| `EmbeddingProvider` | `BGEEmbeddingProvider` |

通过环境变量 `LLM_PROVIDER` / `EMBEDDING_PROVIDER` 切换实现。

## Consequences

- ✅ 支持多 provider 切换，降低迁移成本，更好的扩展性
- ⚠️ 接口设计需要一定前期投入
