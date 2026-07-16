# Deployment


## Container


使用：

Docker Compose


包含：


## backend

FastAPI 服务


## postgres

PostgreSQL

+

pgvector


## redis

应用缓存


---

# Cache Strategy


## LLM Prefix Cache


由模型服务提供。


作用：

- 减少重复 Prompt 成本
- 提升模型处理速度


无需业务实现。


---

## Application Cache


Redis 实现。


用途：

- Memory 查询缓存
- 重复计算缓存
- Session 状态保存


需要业务代码实现。


---

# Runtime Architecture


```
FastAPI

 ↓

Redis

 ↓

PostgreSQL

 ↓

pgvector
```