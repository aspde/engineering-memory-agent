# EMA 代码审查记录与修复清单

> 基于 5 路对抗性技术审查 + 代码逐条核查（2026-08-10）形成的缺陷清单与处理记录。每条记录代码证据、当前状态与处理方向。代码项已全部修复并提交；待改进项的完整分析见 [decision-faq.md](./decision-faq.md) 对应章节。
>
> 状态标记：✅ 已完成（代码+测试+提交）｜🗣 待改进项（已记录处理方向）

---

## 执行状态总览（2026-08-11）

| 区间 | 状态 | 说明 |
|------|------|------|
| P0 评估与核心卖点 | **8 ✅ 全完成** | P0-1/2/3/4/5/7/8 代码项 + P0-6 处理全部完成。P0-3 judge 校准（一致率 1.000 / coverage F1 0.833，见 [judge_calibration_report.md](../../tests/eval/reports/judge_calibration_report.md)）、P0-4 多次均值门禁（CI 已改 3 次均值 CI 下限判红） |
| P1 崩溃级缺陷 | **10 ✅ 全完成** | `0481dd5` 一次提交修完，全量回归通过 |
| P2 生产/安全/定位 | **4 ✅ 代码项 + 12 🗣 待改进项** | 代码完成：P2-4 沙箱 / P2-5 tsc 门禁 / P2-8 attempts / P2-11 错误渲染。其余为待改进项（分析见 [decision-faq.md](./decision-faq.md)） |

---

## P0 — 评估数字与核心卖点

### ✅ P0-1 评估数字是"自问自答"，Recall@5=1.000 含金量低

**证据**：`tests/eval/ground_truth.py` 的 query 由 30 条种子记忆反向生成（指纹即摘要逐字子串）；`tests/eval/seed.py` 把同一份 seed 灌进库；语料 30 条、每 query 仅 `n_relevant=1`（`tests/eval/reports/eval-report.json` 全部行），随机基线 recall@5=5/30≈0.167。因此 1.0 是"记住答案"而非"检索能力"。

**现状**：已启用 `query_candidates.jsonl` 的 27 条 hard-negative 判别集并重述数字——纯向量综合通过 59.3%，bounded cross-encoder top-3 重排后 81.5%（见 [hard_negative_report.md](../../tests/eval/reports/hard_negative_report.md)）。

→ 完整分析见 [decision-faq.md](./decision-faq.md) 第 4 节（评估数字 1.0 是自证吗）。

### ✅ P0-2 语义相关性通道用"被评测模型给自己打分"

**证据**：`tests/eval/dataset.py:219-252`，`run_eval.py:103-111` 默认开启；用同一个 BGE-M3 嵌入召回结果与目标摘要，cos≥0.80 判"相关"——模型检到自己就是"对"。"hard" 类目 recall=1.0 依赖此通道。

**现状**：已改为 opt-in（`--semantic-relevance`），默认确定性纯子串基线（memory 路径 recall 0.900 / MRR 0.844）；报告区分 `substring_hits` / `semantic_only_hits` 如实披露自证贡献。

### ✅ P0-3 LLM judge 与被测模型同源、无校准

**证据**：`tests/eval/llm_judge.py` judge 用 DeepSeek（免费档 mimo-v2.5-free）；committed 报告 `llm-eval-semantic-baseline.json:46-48` 自记 known_anomaly（ans-006 正确回答被判 grounded=false）；`.github/workflows/eval.yml` 曾承认 judge 限流导致"所有 LLM-judged 指标不可信"。

**现状**：已加 judge 一致性小样本校准——12 条人工标注样本（6 grounded / 6 ungrounded，含 2 条同义改写），真实跑出 **grounded 一致率 1.000 / coverage F1 0.833 / 0 假阴假阳**。ans-006 那类误判未复现（同义改写被正确判 grounded，说明是 judge 模型偶发而非 prompt 缺陷），报告见 [judge_calibration_report.md](../../tests/eval/reports/judge_calibration_report.md)。

→ 完整分析见 [decision-faq.md](./decision-faq.md) 第 5 节（LLM judge 可信吗）。

### ✅ P0-4 单次 LLM 采样方差大、两份基线打架、CI 门禁卡中间

**证据**：同一天 `llm-eval-baseline.json`（relation_recall 0.531, tool_accuracy 0.733）与 `llm-eval-report.json`（0.344 / 0.667）互相矛盾；`eval.yml` 门禁 `--min-tool-accuracy 0.68` 正好夹在两次观测之间；`compare_baseline.py` tolerance=0.01 对 n=15 远小于单样本翻转量 0.067。

**现状**：已加 `tests/eval/multi_run_gate.py`——N 次运行均值 + 95% CI 下限判红（t 值硬编码，无 scipy）；`eval.yml` 的 llm-eval 门禁改为 `--n-runs 3` 均值 CI 下限判红。单次随机低值不再误杀 CI，真实下滑才把 CI 下限推到阈值以下。

→ 完整分析见 [decision-faq.md](./decision-faq.md) 第 6 节（单次跑的数字能代表能力吗）。

### ✅ P0-5 头部数字用非生产路径，生产 `memory` 路径无报告

**证据**：`eval-report.json:15-24` 的 1.0/0.983 来自 `chunk:vector`（裸向量）；生产默认 `query_memories`（memory 路径）无 committed 报告；`eval.yml` 对 memory 的门禁 recall 0.95/mrr 0.87 比报的 0.983 低。

**现状**：已补跑生产 memory 路径报告（[memory_path_report.md](../../tests/eval/reports/memory_path_report.md)），[deep-dive.md](./deep-dive.md) 成果表注明路径口径。

### ✅ P0-6 "四级阈值 0.92/0.75/0.60 实测调参"无任何证据

**证据**：全仓库（docs/tests/eval）找不到相似度分布分析或调参过程；`test_memory.py` 只测分级逻辑不测阈值合理性。0.92 对"不同来源各自生成的 LLM 摘要"高到 merge 大概率从不触发。

**现状（已标定）**：`tests/eval/threshold_calibration.py` 收集三类摘要对的 BGE-M3 相似度分布——同义改写（应 merge）0.842-0.965（p25 0.878）、同类不同记忆（不该 merge）≤0.792、异类 ≤0.724。**旧值 0.92 高到 4/8 同义改写对被漏成冲突检测**，0.85 是自然分离点。已改 `memory.py` MERGE 0.92→0.85、CONFLICT 0.75→0.72（SUPPLEMENT 0.60 不变），报告见 [threshold_calibration_report.md](../../tests/eval/reports/threshold_calibration_report.md)。**边界**：8 对改写样本小、与真实生产摘要对还有差距，部署后收集真实对确认。

→ 完整分析见 [decision-faq.md](./decision-faq.md) 第 1 节（四级阈值怎么定的）。

### ✅ P0-7 核心卖点与实现失配（4 处）

| # | 文档宣称 | 代码实际（证据） |
|---|---------|---------|
| 7a | 衰减公式 `S=1+recall×2` | 实际 `S=1+(recall+1)×2`（`decay.py:70,103-111`），排序用 post-recall 强度（`decay.py:190-192`）。**2026-08-11 再改**：A/B 校准后 `S=1+(recall+1)×12` + 0.10 floor（`decay.py` 常量），文档已同步 |
| 7b | "从未召回的记忆自然沉底" | **从未召回的记忆 `decay_factor` 恒为 1.0 永不衰减**（`decay.py:56-57` 在 `recalled_at is None` 时早退）——真设计漏洞 |
| 7c | "默认 cross-encoder rerank" | 生产 `retrieve()/query_memories()` 全默认 `use_cross_encoder=False`，568M rerank 模型生产零调用（`retrieval.py:395-430,744-778`） |
| 7d | "threshold 0.0 全部进候选池" | 默认 `threshold=0.3` 且作用于**原始相似度、衰减之前**（`retrieval.py:686`；`decay.py:172`） |

**现状**：文档已与代码一致（公式/rerank/threshold 改口）；7b 的"从未召回永不衰减"已修（时间基准 `COALESCE(recalled_at, created_at)`，见 `decay.py`）；7a 公式参数已随 A/B 校准同步。

→ 完整分析见 [decision-faq.md](./decision-faq.md) 第 7 节（衰减公式和"沉底"对得上吗）。

### ✅ P0-8 分块对中文无感知，"不截断语义单元"对中文是假的

**证据**：`chunk.py:15` 句号分隔符 `(?<=[.!?])\s+` 不含中文标点，无换行中文长段 fallthrough 到 `_hard_split`（`chunk.py:164-171`）按 512 字符硬切在句中；overlap 的 `prev[-overlap:]`（`chunk.py:149`）也是字符级硬切。

**现状**：已修——分隔符补中文句末标点零宽规则 `(?<=[。！？])`，中文长段落按句切分；`\s+` 保持在最后一项（overlap 重切依赖）。

---

## P1 — 崩溃级缺陷（全部 ✅ 已修复，`0481dd5` 一次提交）

> 这些是"被要求现场演示就会炸"的缺陷，修复时每条都补了回归测试。证据保留供后续引用。

### ✅ P1-1 REST 写路径遇到 conflict 直接 500

**证据**：`_mark_conflict`（`memory.py:448-466`）返回 dict 无 `"id"` 键，`memory_routes.py:162-171` 直接 `id=result["id"]` → KeyError。agent 路径（`tools.py:223-235`）和 webhook 路径（`webhook_routes.py:244-248`）都正确处理，唯独 REST 直写漏修。冲突时新内容既不入库也不入冲突队列。

### ✅ P1-2 SSE 客户端中途断开不清理 agent 流，重试出现重复 user turn

**证据**：`agent_routes.py:620-623` 断开只 return，不 `aclose()` 底层 `agent.astream` 生成器；`_mark_interrupted_thread` 只在 timeout 时调用，断开不调用。用户网络抖一下重试同一问题 = 两条一模一样的 user turn。

### ✅ P1-3 webhook 后台任务 `asyncio.create_task` 无强引用，可被 GC 丢弃

**证据**：`webhook_routes.py:351-353` create_task 不保存引用（delivery 卡在 `received` 无声无息）；对照 `nodes.py:1108-1136`、`memory.py:485-517` 都持有了 set——作者知道这个坑，这里是漏网之鱼。`patrol_routes.py:107`、`main.py:154` 同样无强引用。

### ✅ P1-4 scenario 执行路径无超时、无并发上限

**证据**：`scenario_routes.py:116-128` 直接 `await compose_func(...)` 无 `asyncio.timeout`；`MAX_AGENT_CONCURRENCY` 只盖 chat 路由（`agent_service.py:64-88`）。`recursion_limit=50` 的完整 agent 无上限并发，卡死只能重启。这是"agent 有 timeout、patrol 有 timeout、scenario 没有"三兄弟里漏网的那个。

### ✅ P1-5 embedding failover 与主 provider 共用熔断器名，breaker 打开时兜底恰好失效

**证据**：主备两个 `OpenAIEmbeddingProvider` 共用硬编码 `"embedding:openai"`（`embedding_service.py:147,172`），failover（`embedding_service.py:222-231`）捕获主失败后调 fallback，fallback 再命中同一 breaker 立刻抛 `CircuitOpenError`。LLM 侧特意按 `base_url|model` 区分熔断器名（`llm_service.py:59-62`）——embedding 侧没做同样处理。

### ✅ P1-6 embedding 预热 `threading.Lock` 在事件循环线程上阻塞，冻结整个事件循环

**证据**：`embedding_service.py:298-300` `with _lock:`（threading.Lock）在 async 路径上阻塞；预热用 `asyncio.to_thread(get_embedding_provider)`（`main.py:144-154`），预热未完成时请求在事件循环线程上 `Lock.acquire()` 硬阻塞最多 30s；预热失败则首次请求同步加载模型卡死健康检查。

### ✅ P1-7 熔断器 `record_success` 不校验恢复探测身份

**证据**：`resilience.py:172-178` 无条件清零 `_open_until`，不检查调用者是否为 half-open 放行的探测（`_probing` 状态机存在但没用它做身份校验）。trip 前发出、现在才成功的旧调用会把已打开的 breaker 重新合上，形成"打开→被旧成功合上→再打满失败→再打开"振荡，持续锤打已挂 provider。

### ✅ P1-8 `CircuitOpenError` 穿透 enrichment 降级路径，与文档降级契约矛盾

**证据**：`extraction.py:71,150` 只 `except LLMStructuredError`；熔断器打开时 `structured.py:109-112` 抛 `CircuitOpenError` 不被捕获 → 实体/关系提取整体失败 → `memory.py:86` 写入失败。文档承诺"增强类失败降级为 []"。对照 `memory.py:145-160` 同时捕获 `(LLMStructuredError, CircuitOpenError)`——三处调用方对同一异常处理不一致。

### ✅ P1-9 `supplements_unchecked` 是死字段——"打了标但没人看"

**证据**：`memory.py:104-116,133-136` 写入该 meta 标记，但全项目（backend+tests）无一处读取。第二条近亲记忆可能和新内容矛盾，既没被检测也没在任何下游暴露。

### ✅ P1-10 部分审批的"拒绝反馈"被序列化层吞掉

**证据**（Agent 层审查实测）：`check_approval_node` 部分审批路径注入 `[REJECTED]` ToolMessage，但 `_messages_to_dicts` 序列化时依赖 `responded_ids`，被拒绝的 tool_call 因无 ToolMessage 对应而被判"孤儿"剔除（`nodes.py:174-209` 附近）——LLM 看不到哪些写操作被拒，下一轮可能重试被拒操作。

---

## P2 — 生产成熟度 / 安全 / 定位

### 🗣 P2-1 全站零速率限制 + 单 key 内嵌前端 bundle

**证据**：全仓库无任何请求限流；`VITE_EMA_API_KEY` 构建期打进 JS bundle（`frontend/src/api/client.ts:13-19`，`.env.example:264-271`），任何打开页面的人都能提取 key 无限调 `/api/agent/chat`、`/api/memory/ingest`；`MAX_AGENT_CONCURRENCY=4` 是并发上限不是限流。

→ 完整分析见 [decision-faq.md](./decision-faq.md) 第 10 节（前端 bundle 里的 key）。

### 🗣 P2-2 `APP_ENV=test` 环境变量即可关闭全部 API 认证

**证据**：`auth.py:40-41` 第一行 `if os.environ.get("APP_ENV") == "test": return`；`config.py:218` 默认 development。测试安全与生产安全共用进程级开关。

→ 完整分析见 [decision-faq.md](./decision-faq.md) 第 11 节（为什么测试环境绕过认证用环境变量）。

### 🗣 P2-3 webhook 非 production 且未配 secret 时接受无签名请求

**证据**：`webhook_routes.py:83-98`。**要点**：production 是 fail-closed（未配 secret 直接拒绝），这层是对的；问题在"dev + 无 secret"两个默认值叠加等于裸奔。

→ 完整分析见 [decision-faq.md](./decision-faq.md) 第 12 节（webhook 没配 secret 怎么处理）。

### ✅ P2-4 `ingest_git_repo_tool` 无路径沙箱

**证据**：`tools.py:319-345` repo_path 任意绝对路径，`ingestion.py:43-45` 仅校验 `.git` 存在。prompt injection 可诱导读取服务器任意本地 Git 仓库。

**现状**：已修——`REPO_ALLOW_ROOT` 白名单沙箱，空=默认 fail-closed（所有 ingest 拒绝），路径 resolve 后校验防 `..` 逃逸（见 `ingestion.py` + `config.py`）。

**待改进**：prompt injection 当前仍是软防御（`prompts.py` untrusted-DATA 声明 + `<doc>/<memory>` 包裹），测试只测结构隔离不测行为隔离（`test_prompt_injection.py`）——已知加固点。

### ✅ P2-5 CI 缺三道门禁：前端无 tsc、后端无 integration、不构建 Docker

**证据**：`ci.yml` 只跑 `pytest tests/unit tests/api` 和 `npm test`（vitest 用 esbuild 不查类型）；无 Docker build job；仓库提交了 `frontend/tsconfig.app.tsbuildinfo`。

**现状**：已加 `npx tsc --noEmit` 门禁（前端类型错误不再静默进 master）；后端 integration 与 Docker build job 仍未加（可选）。

### 🗣 P2-6 双连接池、无迁移工具、启动 DDL 清向量列

**证据**：`db/__init__.py:34`（asyncpg 池 5+10）+ `agent_service.py:123-128`（psycopg 池 max_size=5）互不感知、参数写死；`schema.py:23-47` embedding 维度变更清空全部向量列、无可回滚。

**现状**：已引入 Alembic schema 版本化迁移（`alembic_version` 表 + upgrade/downgrade 成对迁移，基线固化 9 表，新库/旧库/回滚三路径已实测）——"清向量列换维度"的危险路径已可回滚。双池仍是驱动差异技术债。

→ 完整分析见 [decision-faq.md](./decision-faq.md) 第 13 节（为什么有两套连接池）。

### 🗣 P2-7 检索是读路径却写库（每次检索 UPDATE decay）

**证据**：`retrieval.py:786-795` 每次 `query_memories` 对 top-k 结果执行 `update_decay_batch`；`decay.py:103-118`。语义失真：`recall_count` 记录"被任意查询返回过"而非"被使用过"，垃圾查询无差别强化返回的记忆，马太效应自我强化。

→ 完整分析见 [decision-faq.md](./decision-faq.md) 第 2 节（检索为什么每次 UPDATE decay）。

### ✅ P2-8 读路径的 LLM 用量会计丢失瞬时失败（错误率被低估）

**证据**：`llm_service.py:95-107` tenacity 吞掉中间 429/5xx，只有最终成功写一行 usage；熔断器只在整条重试梯耗尽后计数（`resilience.py:297-308`）。provider 一直在抖但监控看不到。

**现状**：已修——`llm_usage` 加 `attempts` 列，被吞掉的瞬时失败如实记录（1=一次成功，3=重试两次）。告警逻辑未改（本次是记录半），后续可在 alerts 读 `attempts>1` 作抖动信号。

### 🗣 P2-9 实体归一化：fire-and-forget 无重放、merge 后旧链接不清、0.85 阈值无测量

**证据**：`memory.py:488-517` create_task fire-and-forget，重启即丢；全项目无 `DELETE FROM memory_entities`，merge/overwrite 后被合并实体在关联表永不清除（`entity.py:137-148` 只 INSERT ON CONFLICT DO NOTHING）；`entity.py:24` `SIMILARITY_THRESHOLD=0.85` 无准确率测量。

→ 完整分析见 [decision-faq.md](./decision-faq.md) 第 14 节（实体归一化可靠吗）。

### 🗣 P2-10 `chat_sync`/`embed_sync` 在生产零调用——每 provider 维护双客户端是死复杂度

**证据**：接口 abstractmethod 强制（`llm.py:28-31`、`embedding.py:14-17`），生产唯一"调用"是包装器互相转发，真实调用方只有 tests。

→ 完整分析见 [decision-faq.md](./decision-faq.md) 第 15 节（chat_sync 用在哪）。

### ✅ P2-11 前端偏薄：无状态管理、单页 150-400 行、SSE 无重连

**证据**：`App.tsx` 仅 28 行；无 redux/zustand/context 状态管理；`useChat.ts` 流中断无指数退避重连，错误与正文混排；`ChatArea.tsx` `MAX_VISIBLE=50` 硬截断。

**现状**：已修错误与正文混排（SSE 错误/网络失败渲染为独立 `kind:'error'` 消息，不再 append 进回复正文）；无状态管理与 SSE 无重连仍为定位取舍。

**定位说明**：前端定位"配角、验证后端能力"（README 已承认前端深度/DevOps/安全低优先）。

### 🗣 P2-12 检索排序的边界近似

**证据**：`retrieval.py:676` multi-query 并集按插入序截断、未先按相关性排序；`retrieval.py:494-516` sparse 候选 `ORDER BY id` 先截断后 Jaccard 过滤；`decay.py:159-196` 两阶段窗口（top_k×8）不保证全局最优。

→ 完整分析见 [decision-faq.md](./decision-faq.md) 第 3 节（检索排序是全局最优吗）。

---

## 项目定位与改进方向

### 🗣 P2-13 无团队协作背书 + 无真实用户

**证据/策略**（README.md 和 [project-overview.md](./project-overview.md)）：定位为"端到端主导 + 工程纪律替代 review"（ADR/CI/评估集/1450+ 测试）。每个数字要有真实来源（QPS 4.8、记忆条数、token 成本），并准备"1450+ 测试没有 code review 怎么保证质量"的答案。

→ 完整分析见 [decision-faq.md](./decision-faq.md) 第 18 节（没有团队怎么保证质量）。

### 🗣 P2-14 规模偏小，属个人项目量级

**证据**：记忆库 30 条（评估集）、10 并发 QPS 4.8、项目周期 3 个月。

→ 完整分析见 [decision-faq.md](./decision-faq.md) 第 17 节（数据量这么小有意义吗）——拐点思维（万级记忆库 pgvector HNSW 够用），诚实承认生产级规模没跑过。

### 🗣 P2-15 LangGraph 是相对冷门框架

**已有分析**（[deep-dive.md](./deep-dive.md) 决策 1）：黑盒 + HITL 控制弱 + 调试代价。补充两个点：公司已有 LangChain 基础设施时的迁移成本与收益（框架是薄封装，核心价值在记忆/检索/韧性层）；LangGraph 与自研 event loop 的边界（价值在 interrupt/checkpoint/流式）。

→ 完整分析见 [decision-faq.md](./decision-faq.md) 第 16 节（为什么不用 LangChain Agent）。

### 🗣 P2-16 生产可靠性是 Agent 工程的重点考查面

**定位策略**：把"评估有水分"主动转为"我做过诚实评估"（P0-1/2/4 口径）；把"无限流"转为"我清楚生产安全基线"（P2-1）；把 task completed=0.5 和 unexpected_rate=0.375（DeepSeek 过度调工具）当成**最有价值的发现**主动讲——比任何完美数字都更能说明对 Agent 生产行为的理解。

→ 完整分析见 [decision-faq.md](./decision-faq.md) 第 4 节（评估数字 1.0 是自证吗）。
