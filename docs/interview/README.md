# EMA 面试准备材料索引

> 目标岗位：AI/LLM 应用工程师 | 工作年限：3-5 年中级 | 核心主项目：EMA

## 文档清单

| 文档 | 用途 | 时长 |
|------|------|------|
| [self-introduction.md](./self-introduction.md) | 自我介绍逐字稿，可背诵 | 5-8 分钟 |
| [ema-deep-dive.md](./ema-deep-dive.md) | EMA 项目深度讲解（架构 / 决策 / 难点 / 成果） | 15-20 分钟 |
| [tech-qa-drill.md](./tech-qa-drill.md) | 技术问题演练 Q&A 库（按主题分类） | — |
| [behavioral-star.md](./behavioral-star.md) | 行为面试 STAR 案例库 | — |
| [gap-remediation.md](./gap-remediation.md) | 已知短板诊断 + 评估体系补全方案 | — |
| [script-cards.md](./script-cards.md) | **应答话术卡（13 段必背 + 5 段补充高危口径），临场只带这一份** | 15 分钟/轮 |
| [adversarial-review-evidence.md](./adversarial-review-evidence.md) | 对抗性审查证据档案（每条短板：代码证据 / 状态 / 应答入口），深挖时查这份 | — |

## 评估证据与报告（追问数字来源）

风险矩阵里的关键数字来自以下报告，深挖时可查原始证据：

| 文件 | 内容 |
|------|------|
| [llm-eval.md](./llm-eval.md) | LLM 行为评测文档（工具选择 / 知识抽取 / 最终答案 / 端到端 / task_eval），`task completed 0.5 / tool_recall 0.94` 的出处 |
| [llm-eval-report.md](./llm-eval-report.md) | LLM 行为评测报告（2026-08-09，39 条，LLM judge 通道） |
| [llm-eval-baseline.json](./llm-eval-baseline.json) | 确定性通道基线，CI 门禁校准依据 |
| [eval-report.md](./eval-report.md) / [eval_30_db_sparse.md](./eval_30_db_sparse.md) | 2026-08-07 检索评测历史报告——"自证 1.0"口径的原始证据 |
| `tests/eval/reports/*` | 当前检索/LLM 评测报告（hard-negative、judge 校准、memory 路径、hybrid 对比、**memory LLM rerank vs CE**、**decay A/B**、**extraction A/B**，风险矩阵已引用） |

## 使用方式

1. **第一遍**：通读含占位符的 4 份文档（self-introduction.md / ema-deep-dive.md / behavioral-star.md / adversarial-review-evidence.md），标出与个人经历不符的内容，用 `[需要你补充：XXX]` 占位符填实
2. **第二遍**：背诵 self-introduction.md，对着 ema-deep-dive.md 复述项目讲解
3. **第三遍**：找人模拟面试，用 tech-qa-drill.md 做技术问答，用 behavioral-star.md 做行为问答
4. **临场**：带 script-cards.md（应答话术卡）+ ema-deep-dive.md 的"一句话架构图"和"4 个 ADR 决策"小抄

## 占位符说明

文档中出现 `[需要你补充：XXX]` 的地方是必须由你本人填实的个人信息，包括：
- 具体工作年限、公司名、团队规模
- 个人在项目中的实际角色（端到端主导 / 核心开发 / 技术负责人）
- 量化成果的真实数据（QPS、延迟、记忆条数、用户数等）
- 其他项目经历

含占位符的文档：self-introduction.md（3 处）、ema-deep-dive.md（2 处）、behavioral-star.md（12 处）、adversarial-review-evidence.md（1 处）。

## 核心准备策略

- **EMA 是 AI/LLM 应用工程师岗位的"对口项目"**：覆盖 LangGraph Agent、RAG、向量检索、Prompt 工程、HITL、流式输出全链路，重点讲透
- **3-5 年中级定位**：突出"端到端交付 + 技术深度 + 工程纪律"，不夸大为架构师，但要有"技术选型决策依据"
- **不提"独立完成"**：不强调"一个人写的 demo"，改讲"我端到端主导，并用团队工程的纪律约束自己"——ADR 记录决策、CI 把关提交、评估集量化改动、1416 个测试兜底，用可追溯的流程替代缺失的 code review。被问"怎么没有团队"时，诚实承认 + 讲这个替代流程，不编造团队
- **真实代码背书**：所有技术细节均来自 EMA 实际代码，可经得起追问，不要替换为编造内容

---

## 已知短板与应对

> 基于代码核查的真实诊断。被问到时遵循「诚实承认 + 讲决策依据 + 给改进方案」三段式。

### 风险矩阵

> 注：已补齐项（评估体系 / 成本监控 / CI / Prompt 版本管理 / 运行监控指标 / **LLM rerank vs CE 与衰减两个 A/B** / **提取 few-shot + 函数调用优化** / **数据库备份与恢复** / **Prometheus+Grafana 采集闭环** / **API 限流** / **Caddy TLS 反代与日志结构化** / **Alembic schema 版本化迁移**）不再列为短板，详见 [gap-remediation.md](./gap-remediation.md)。

| 维度 | 风险等级 | 关键短板 | 应对策略 |
|------|---------|---------|---------|
| AI 工程化 | 🟢 低 | 评估体系（Recall@K/MRR + LLM-as-judge + task 级端到端 + hard-negative 判别力）、成本监控（llm_usage 表）、**两个 A/B（LLM rerank vs CE、衰减开关）+ 衰减调参闭环 + 提取优化（few-shot + 函数调用 + precision 回拉）+ 四级阈值标定 + 评估集扩充到 70 条**均已交付，数字按诚实口径披露并驱动改进 | 讲实测数字 + 诚实分层：主评估集 **70 条**（5 类 × 14，hard 占 30%）**默认确定性基线** recall@5 0.886 / MRR 0.767（纯子串匹配，无自证）；语义通道为显式 opt-in（用被评测模型自评故非默认）；**hard-negative 判别力：纯向量 27 条仅 59.3% 通过 / MRR 0.79 / 11 条陷阱压过目标，已用此集落地 bounded cross-encoder top-3 重排提至 81.5%**（见 [hard-negative-report.md](../../tests/eval/reports/hard_negative_report.md)）；**LLM rerank vs bounded-CE：recall@5 相同 0.900、MRR +0.014 但延迟 5.5x**（[memory_llm_vs_ce_report.md](../../tests/eval/reports/memory_llm_vs_ce_report.md)）；**衰减 A/B + 调参闭环：原公式 recall 0.367 → 调 S=12+floor 0.1 后 0.667，保留"过时沉底"偏好**（[decay_ab_report.md](../../tests/eval/reports/decay_ab_report.md)）；**四级阈值已标定：0.92/0.75 高到 merge 不触发，改 0.85/0.72**（[threshold_calibration_report.md](../../tests/eval/reports/threshold_calibration_report.md)）；task completed 0.5 tool_recall 0.94（DeepSeek 过度调工具是真实发现）/ 17.5s→0.19s 瓶颈归因；提取 few-shot + 函数调用 A/B（precision 已回拉 0.609→0.713）见 [extraction_ab_report.md](../../tests/eval/reports/extraction_ab_report.md) |
| 项目可信度 | 🔴 高 | 无团队协作背书 + 无真实用户 | 填实量化数据；传 GitHub；主打"端到端主导 + 工程纪律替代 review"（ADR/CI/评估集/1416 测试）；讲迭代方法而非线性开发；被问团队时诚实承认 + 讲替代流程 |
| 生产成熟度 | 🟢 低 | 巡检内嵌主进程（有意取舍，ADR-007）+ 单实例部署约束（有意取舍）——**纯缺口已全部补齐**（备份/监控/限流/TLS/日志/迁移） | 巡检讲 [ADR-007](../../decisions/ADR-007-patrol-in-process-scheduler.md) 依据 + 拐点；单实例约束明示于 deployment.md（改造方向：熔断/限流/节流计数迁共享存储）；其余已补：备份/恢复（backup 容器 pg_dump + runbook，**已实测恢复一致**）、监控闭环（Prometheus+Grafana 9 面板）、API 限流（per-key 令牌桶，429 + Retry-After）、Caddy TLS 反代（域名自动 Let's Encrypt）、日志结构化（`LOG_FORMAT=json` + trace_id/thread_id）、**Alembic schema 版本化迁移**（`alembic_version` 表 + upgrade/downgrade 成对迁移，embedding 维度按运行时对齐，见 deployment.md Schema Migration） |
| 规模性能 | 🟡 中 | 单机 pgvector / BGE-M3 CPU 推理。**诚实口径**：缓存热路径 10 并发 QPS 4.8 P95 110ms / 160 并发 QPS 63 0 失败；**冷路径（779 条互异真实查询）长尾已修：并发控制（线程信号量 + torch 限线程，乘积=核数）把 10 并发 P95 19s→690ms、Max 920ms、QPS 2.61→4.39；40 并发 P95 9.4s→1.9s（吞吐受限，CPU 密集固有取舍）**（见 gap-remediation.md §5.3.1-5.3.2）。对话 P95 73.6s 中 rerank_llm 占 58.6% 延迟 / 46% 成本（llm_usage 表量化，见 §3.1.1）；**已从工具 schema 锁死 LLM rerank（移除 use_llm_rerank 参数），task_eval 验证质量不掉（completed 持平 / groundedness 1.000 / within_budget 改善），P95 预期降至 ~25s**（见 llm-eval.md）。**万级 rerank 转正已实测：9999 条语料无 rerank recall 0.933/MRR 0.732，+CE 0.967/0.886（延迟 882ms→29.9s）**（见 gap-remediation.md §11.5.3） | 讲拐点思维 + 冷热双口径数字 + 并发控制落地（长尾 19s→0.7s）+ rerank_llm 占比量化 + **万级 scale-dependent 实测曲线（30 有害→1000 临界→10k 转正）**：热路径 P95 110ms 是缓存加速器，冷路径真实能力已修长尾；对话 rerank 已锁死（任务完成率与答案质量验证不掉）；万级开 rerank 的前提是 embed/rerank 服务化上 GPU（P2，未实测） |
| 技术广度 | 🟢 低 | 前端深度 / DevOps / 安全 | 坦承非最强项，重点讲后端 + AI |

### 决策 vs 疏漏（应答时区分）

| 类型 | 短板 | 应对 |
|------|------|------|
| **有意决策**（ADR 撑腰） | 不做多租户 / 不用 Neo4j / 单 Agent / 不做场景自动路由 / 无用户级认证（有 API key 接入认证） | 讲决策依据 + 拐点，是有意取舍不是疏漏 |
| **已知疏漏**（老实认） | 无真实用户流量（对话 P95 73.6s / ≈¥0.06 每轮为实测，非生产流量） | 诚实承认 + 讲改进方案，不狡辩 |

### 面试前 Quick Win（按 ROI 排序）

| 优先级 | 动作 | 耗时 | 详见 |
|--------|------|------|------|
| 🔴 P0 | 填实所有 `[需要你补充]` 占位符 | 1h | 各文档 |
| ✅ 已完成 | 对话 P95 + token 成本实测（10 轮真实对话：P95 73.6s、≈28.6k tokens/轮 ≈¥0.06，见 gap-remediation §3.1） | — | tests/perf/measure_chat_p95.py |
| 🟢 P2 | 传 GitHub + 整理 README | 2h | — |
| ✅ 已完成 | Prometheus 运行监控指标（runtime_metrics.py + `GET /metrics`） | — | architecture.md Observability |
| ✅ 已完成 | 监控采集闭环：Prometheus 抓取 + Grafana 看板（9 面板，compose 一键启动） | — | deployment.md Monitoring |
| ✅ 已完成 | 数据库备份 + 恢复 runbook（backup 容器每小时 pg_dump，已实测恢复一致） | — | deployment.md Backup & Restore |
| ✅ 已完成 | Caddy TLS 反代（默认本地 HTTP，配域名自动 Let's Encrypt，SSE 流式实测透传） | — | deployment.md HTTPS (TLS) |
| ✅ 已完成 | 日志结构化（LOG_FORMAT=json 单行 JSON + trace_id/thread_id） | — | deployment.md 日志（结构化） |
| ✅ 已完成 | Alembic schema 版本化迁移（alembic_version 表 + upgrade/downgrade，基线迁移固化 9 表，已实测新库/旧库/回滚三路径） | — | deployment.md Schema Migration |
| ✅ 已完成 | Dockerfile（后端+前端单镜像）+ docker compose 一键启动 | — | deployment.md |
| ✅ 已完成 | locust 压测 /api/memory/search：10 并发 QPS 4.8 P95 110ms / 160 并发 QPS 63 0 失败 | — | gap-remediation.md §5.3 |

### 高危问题应答模板

被问到短板时：

> 「这块确实是 EMA 当前的不足。[诚实承认现状]。我做的时候是有意识做了取舍——[讲决策依据，引用 ADR]。如果在大规模场景下，我的改进方案是 [讲方案]。这个我目前还没机会实测，是设计预案。」

**四原则**：① 不狡辩 ② 要有方案 ③ 区分决策与疏漏 ④ 不编数据
