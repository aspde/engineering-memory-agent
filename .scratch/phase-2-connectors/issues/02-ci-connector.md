# 02 — CI 连接器

**What to build:** `POST /api/webhook/ci` 接收 CI 构建失败 payload（GitHub Actions / GitLab CI / Jenkins，通过通用 envelope）→ CIConnector 校验并提取 job 名称、错误摘要、commit SHA、分支、耗时 → 记忆以 `source_type="ci_build"` 存储。耗时回归（如测试套件从 3min→12min）被自动检测并标记为 `"ci_regression"`。commit SHA 归一化后可关联 Phase 1 的代码实体。

**Blocked by:** 01 — 连接器基础设施 + Webhook 端点 + Jira 连接器

**Status:** ready-for-agent

- [ ] CIConnector 实现：validate 校验必要字段（job_name, status, commit_sha），normalize 输出结构化文本（job + status + error_summary + commit_sha + branch + duration）
- [ ] 构建失败（status=failure/error）→ `source_type="ci_build"`，写入记忆
- [ ] 耗时回归检测：duration 超过历史基线 N 倍时 → `source_type="ci_regression"`，summary 中注明回归幅度
- [ ] 在 registry 中注册 CIConnector（`WEBHOOK_CI_SECRET` 配置即激活）
- [ ] 单元测试：`test_connector_ci.py`（validate 拒绝缺字段 payload、normalize 输出含 commit SHA、回归检测逻辑）
