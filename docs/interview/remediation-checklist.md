# EMA 面试短板修复 / 应答清单

> 基于 5 路对抗性技术审查 + 代码逐条核查（2026-08-10）。每条标注动作类型与优先级，面试前逐条对照执行。
>
> **动作类型**：🔧 改代码（含测试）｜📝 改文档｜🗣 仅准备应答口径（不碰代码）
> **优先级**：P0 不修/不备必露馅 → P1 深挖命中 → P2 生产成熟度与定位
>
> 三条总原则：
> 1. **改文档优先于改代码**——"文档和代码不一致"是零成本消除的；先让讲的每一句话都有代码撑腰。
> 2. **诚实暴露比完美数字可信**——task completed=0.5 比 Recall@5=1.0 更能让面试官信服。对抗性审查无法击穿"我如实报告了失败"这个姿态。
> 3. **每个"改代码"都必须补测试**（项目规则强制），每个"🗣"都必须写成一段能背的话术，不要现场组织语言。

---

## 执行状态总览（2026-08-11 更新）

> 状态标记见各条标题：✅ 已完成（代码+测试+提交）｜🟡 部分完成（主项已做，可选加分项未做）｜🗣 纯应答话术已备（无需改代码）

| 区间 | 状态 | 说明 |
|------|------|------|
| P0 评估与核心卖点 | **8 ✅ 全完成** | P0-1/2/3/4/5/7/8 代码项 + P0-6 话术全部完成。P0-3 judge 校准（一致率 1.000 / coverage F1 0.833，见 [judge_calibration_report.md](../../tests/eval/reports/judge_calibration_report.md)）、P0-4 多次均值门禁（CI 已改 3 次均值 CI 下限判红） |
| P1 崩溃级缺陷 | **10 ✅ 全完成** | `0481dd5` 一次提交修完，全量回归通过 |
| P2 生产/安全/定位 | **4 ✅ 代码项 + 12 🗣 话术项** | 代码完成：P2-4 沙箱 / P2-5 tsc 门禁 / P2-8 attempts / P2-11 错误渲染。其余为纯应答话术（已备在文） |

**留给面试前的只剩**：
1. 背熟全部 🗣 话术（P0-6 / P2-1/2/3/6/7/9/10/12/13/14/15/16，见 [script-cards.md](./script-cards.md)）
2. 填实 `ema-deep-dive.md` / `self-introduction.md` 里的 `[需要你补充：XXX]` 个人占位符

---

## P0 — 评估数字与核心卖点（不修必露馅）

### ✅ P0-1 [🔧+🗣] 评估数字是"自问自答"，Recall@5=1.000 含金量低

**证据**：`tests/eval/ground_truth.py` 的 query 由 30 条种子记忆反向生成（指纹即摘要逐字子串）；`tests/eval/seed.py` 把同一份 seed 灌进库；语料 30 条、每 query 仅 `n_relevant=1`（`tests/eval/reports/eval-report.json` 全部行），随机基线 recall@5=5/30≈0.167。因此 1.0 是"记住答案"而非"检索能力"。

**🔧 改代码（高优先）**：启用 `tests/eval/query_candidates.jsonl` 里 34 条 **hard negative**（现全部 status=candidate、review=null）。这是能证明判别力的测试集，跑一遍，把数字重新报告——无论结果好坏都是加分项（好的话含金量翻倍，坏的话你有"诚实校准"的故事）。

**🗣 应答口径（必背）**：
> "Recall@5=1.0 是我在 30 条种子语料上的自评结果，query 是从种子记忆手工标注的，每条只标了一条相关记忆，所以这个数字含金量有限，我把它定位为回归基线而不是能力上限。评估集下一步要做的两件事：引入 hard negative 和混淆负例、扩充到百条级。另外生产默认路径 `query_memories` 我还没有 committed 报告，chunk:vector 的 1.0 不能代表它。"

### ✅ P0-2 [🔧] 语义相关性通道用"被评测模型给自己打分"

**证据**：`tests/eval/dataset.py:219-252`，`run_eval.py:103-111` 默认开启；用同一个 BGE-M3 嵌入召回结果与目标摘要，cos≥0.80 判"相关"——模型检到自己就是"对"。"hard" 类目 recall=1.0 依赖此通道。

**🔧 改代码**：把 `semantic_relevance` 默认改为 False，或改用独立判据（如非重叠词/人工标注子串）。报告里区分 `substring_hits` 与 `semantic_only_hits`，诚实披露哪些 query 是靠语义通道通过的。

### ✅ P0-3 [🔧+🗣] LLM judge 与被测模型同源、无校准

**证据**：`tests/eval/llm_judge.py` judge 用 DeepSeek（免费档 mimo-v2.5-free）；committed 报告 `llm-eval-semantic-baseline.json:46-48` 自记 known_anomaly（ans-006 正确回答被判 grounded=false）；`.github/workflows/eval.yml:128-131` 承认 judge 曾被限流导致"所有 LLM-judged 指标不可信"。

**🔧 改代码**：judge 强制独立 provider 已有（`run_llm_eval.py:208-250` 是对的，保留）；补一个 **judge 一致性小样本校准**（10-20 条人工 verdict vs LLM verdict 的一致率）。**✅ 已完成**：`tests/eval/judge_calibration.py` + 12 条人工标注样本（6 grounded / 6 ungrounded，含 2 条同义改写），真实跑出 **grounded 一致率 1.000 / coverage F1 0.833 / 0 假阴假阳**——ans-006 那类误判未复现（同义改写被正确判 grounded，说明是 judge 模型偶发而非 prompt 缺陷），报告见 [judge_calibration_report.md](../../tests/eval/reports/judge_calibration_report.md)。

**🗣 应答口径**：主动讲 ans-006 那个误判——"judge 把'允许改写'当耳边风，所以我在报告里把它记为已知异常，LLM-judge 的 groundedness 是参考值不是绝对值；deterministic judge 才是门禁"。

### ✅ P0-4 [🔧+🗣] 单次 LLM 采样方差大、两份基线打架、CI 门禁卡中间

**证据**：同一天 `llm-eval-baseline.json`（relation_recall 0.531, tool_accuracy 0.733）与 `llm-eval-report.json`（0.344 / 0.667）互相矛盾；`eval.yml:170-174` 门禁 `--min-tool-accuracy 0.68` 正好夹在两次观测之间；`compare_baseline.py` tolerance=0.01 对 n=15 远小于单样本翻转量 0.067。

**🔧 改代码**：门禁改成**多次运行的均值 ± 置信区间**判红（n≥3 次取均值），tolerance 按 `1/n` 量级设。这份工作本身是很好的面试谈资。**✅ 已完成**：`tests/eval/multi_run_gate.py`（N 次运行均值 + 95% CI 下限判红，t 值硬编码无 scipy）；`eval.yml` 的 llm-eval 门禁已改为 `--n-runs 3` 均值 CI 下限判红；真实聚合两份 committed 报告验证门禁行为（relation_f1 / groundedness 判红暴露 judge-mode 混合，tool_accuracy 等通过）。

**🗣 应答口径**："LLM 行为方差大是真实挑战，我测过同一天跑两次 relation_recall 从 0.53 掉到 0.34。所以我在门禁上做的是 [多次均值]，而不是拿单次数字当能力——单次采样当能力指标是会骗人的。"

### ✅ P0-5 [📝] 头部数字用非生产路径，生产 `memory` 路径无报告

**证据**：`eval-report.json:15-24` 的 1.0/0.983 来自 `chunk:vector`（裸向量）；生产默认 `query_memories`（memory 路径）无 committed 报告；`eval.yml:95` 对 memory 的门禁 recall 0.95/mrr 0.87（校准注释 mrr≈0.94）比报的 0.983 低。

**📝 改文档**：`ema-deep-dive.md:243-245` 的成果表注明"1.0/0.983 为 chunk:vector 路径，生产 memory 路径门禁为 0.95/0.87"。**🔧 补跑**一份 memory 路径报告，让口径统一。

### 🗣 P0-6 [🗣] "四级阈值 0.92/0.75/0.60 实测调参"无任何证据

**证据**：全仓库（docs/tests/eval）找不到相似度分布分析或调参过程；`test_memory.py` 只测分级逻辑不测阈值合理性。0.92 对"不同来源各自生成的 LLM 摘要"高到 merge 大概率从不触发。

**🗣 应答口径（必背，不要硬辩）**：
> "阈值是拍板定的初始值，0.92 是'几乎同句'的边界、0.75 是'可能相关'、0.60 是'勉强沾边'，但我没有做过相似度分布统计来校准它——这是我知道的未验证参数。我在测试里覆盖了分级逻辑，但没有覆盖阈值在不同 embedding 分布下的合理性。改进方案是收集真实写入的摘要对，画相似度直方图重新标定，这个我还没做。"
> 这个回答里"诚实承认 + 改进方案 + 未做的边界"三段式，比"实测调参"的谎经得起追问。

### ✅ P0-7 [🔧] 核心卖点与实现失配（4 处，深挖必露馅）

| # | 文档宣称 | 代码实际 | 修法 |
|---|---------|---------|------|
| 7a | 衰减公式 `S=1+recall×2` | 实际 `S=1+(recall+1)×2`（`decay.py:70,103-111`），排序用 post-recall 强度（`decay.py:190-192`） | 🔧 统一约定并写明注释；或 📝 改文档公式。**至少让文档=代码** |
| 7b | "从未召回的记忆自然沉底" | **从未召回的记忆 `decay_factor` 恒为 1.0 永不衰减**（`decay.py:56-57` 在 `recalled_at is None` 时早退）——真设计漏洞 | 🔧 按 `created_at` 年龄或未命中时长计算基础衰减。这是"沉底机制"的必要条件 |
| 7c | "默认 cross-encoder rerank" | 生产 `retrieve()/query_memories()` 全默认 `use_cross_encoder=False`，568M rerank 模型生产零调用（`retrieval.py:395-430,744-778`） | 📝 文档改口：rerank 默认关、eval 数据支持"小语料 rerank 有害"（`eval-report.md:21`） |
| 7d | "threshold 0.0 全部进候选池" | 默认 `threshold=0.3` 且作用于**原始相似度、衰减之前**（`retrieval.py:686`；`decay.py:172`） | 🔧 或 📝 统一口径：0.3 是垃圾过滤门槛，不是 0.0。能解释为什么 0.3 即可 |

**🗣 通用口径**：这些被追问时主动先讲"哪几个是有意取舍（rerank 关闭有 A/B 数字撑腰）、哪几个是我没校准（阈值）、哪几个是文档没同步（公式）"——**主动分级比被发现漏**。

### ✅ P0-8 [🔧] 分块对中文无感知，"不截断语义单元"对中文是假的

**证据**：`chunk.py:15` 句号分隔符 `(?<=[.!?])\s+` 不含中文标点，无换行中文长段 fallthrough 到 `_hard_split`（`chunk.py:164-171`）按 512 字符硬切在句中；overlap 的 `prev[-overlap:]`（`chunk.py:149`）也是字符级硬切。

**🔧 改代码**：分隔符补 `[。！？]`（无空格情况单独处理），overlap 改为按分隔符边界回溯。目标语料是中文，这是必须修的中文场景 bug。补中文分块测试。

---

## P1 — 崩溃级缺陷（被要求 demo / 复现时必炸）

> 以下全部 🔧，且全部要求"能现场跑通"的修复 + 测试。面试官让现场演示时这些是翻车点。

### ✅ P1-1 [🔧] REST 写路径遇到 conflict 直接 500

**证据**：`_mark_conflict`（`memory.py:448-466`）返回 dict 无 `"id"` 键，`memory_routes.py:162-171` 直接 `id=result["id"]` → KeyError。agent 路径（`tools.py:223-235`）和 webhook 路径（`webhook_routes.py:244-248`）都正确处理，唯独 REST 直写漏修。冲突时新内容既不入库也不入冲突队列。

**修法**：路由侧处理 `action=="conflict"`——落 pending_conflicts 队列并返回 201 + conflict_id（与 webhook 路径一致），或返回 409 + 冲突详情。补 `test_memory_routes.py` 覆盖。**这是 10 分钟的修复，先做。**

### ✅ P1-2 [🔧] SSE 客户端中途断开不清理 agent 流，重试出现重复 user turn

**证据**：`agent_routes.py:620-623` 断开只 return，不 `aclose()` 底层 `agent.astream` 生成器；`_mark_interrupted_thread` 只在 timeout 时调用，断开不调用。用户网络抖一下重试同一问题 = 两条一模一样的 user turn。

**修法**：finally 块里 `aclose()` 生成器 + 调用 `_mark_interrupted_thread`。补断开场景测试。

### ✅ P1-3 [🔧] webhook 后台任务 `asyncio.create_task` 无强引用，可被 GC 丢弃

**证据**：`webhook_routes.py:351-353` create_task 不保存引用（delivery 卡在 `received` 无声无息）；对照 `nodes.py:1108-1136`、`memory.py:485-517` 都持有了 set——作者知道这个坑，这里是漏网之鱼。`patrol_routes.py:107`、`main.py:154` 同样无强引用。

**修法**：模块级 set 持有 task 引用，`done` 回调里 discard。**改代码 + 补测试。**

### ✅ P1-4 [🔧] scenario 执行路径无超时、无并发上限

**证据**：`scenario_routes.py:116-128` 直接 `await compose_func(...)` 无 `asyncio.timeout`；`MAX_AGENT_CONCURRENCY` 只盖 chat 路由（`agent_service.py:64-88`）。`recursion_limit=50` 的完整 agent 无上限并发，卡死只能重启。

**修法**：scenario run 套 `asyncio.timeout`（如 5 分钟）+ 复用并发槽位。**修法**：这是"agent 有 timeout、patrol 有 timeout、scenario 没有"三兄弟里漏网的那个，改代码。

### ✅ P1-5 [🔧] embedding failover 与主 provider 共用熔断器名，breaker 打开时兜底恰好失效

**证据**：主备两个 `OpenAIEmbeddingProvider` 共用硬编码 `"embedding:openai"`（`embedding_service.py:147,172`），failover（`embedding_service.py:222-231`）捕获主失败后调 fallback，fallback 再命中同一 breaker 立刻抛 `CircuitOpenError`。LLM 侧特意按 `base_url|model` 区分熔断器名（`llm_service.py:59-62`）并注释说明——embedding 侧没做同样处理。

**修法**：breaker 名改为 `f"embedding:{base_url}|{model}"`。一行修复。**这是最容易被面试官抓的"不一致"。**

### ✅ P1-6 [🔧] embedding 预热 `threading.Lock` 在事件循环线程上阻塞，冻结整个事件循环

**证据**：`embedding_service.py:298-300` `with _lock:`（threading.Lock）在 async 路径上阻塞；预热用 `asyncio.to_thread(get_embedding_provider)`（`main.py:144-154`），预热未完成时请求在事件循环线程上 `Lock.acquire()` 硬阻塞最多 30s；预热失败则首次请求同步加载模型卡死健康检查。

**修法**：预热用可 await 的 future/asyncio.Lock + 后台线程加载并 `set_result`，事件循环等待时不阻塞。

### ✅ P1-7 [🔧] 熔断器 `record_success` 不校验恢复探测身份

**证据**：`resilience.py:172-178` 无条件清零 `_open_until`，不检查调用者是否为 half-open 放行的探测（`_probing` 状态机存在但没用它做身份校验）。trip 前发出、现在才成功的旧调用会把已打开的 breaker 重新合上，形成"打开→被旧成功合上→再打满失败→再打开"振荡，持续锤打已挂 provider。

**修法**：`record_success` 只有在 `_probing` 状态下由探测调用才关闭 breaker。**补状态转换测试。**

### ✅ P1-8 [🔧] `CircuitOpenError` 穿透 enrichment 降级路径，与文档降级契约矛盾

**证据**：`extraction.py:71,150` 只 `except LLMStructuredError`；熔断器打开时 `structured.py:109-112` 抛 `CircuitOpenError` 不被捕获 → 实体/关系提取整体失败 → `memory.py:86` 写入失败。文档承诺"增强类失败降级为 []"。对照 `memory.py:145-160` 同时捕获 `(LLMStructuredError, CircuitOpenError)` 做了 fail-safe——三处调用方对同一异常处理不一致。

**修法**：`extraction.py` 的 `except` 扩为同时捕获两种异常（或共用异常基类）。

### ✅ P1-9 [🔧] `supplements_unchecked` 是死字段——"打了标但没人看"

**证据**：`memory.py:104-116,133-136` 写入该 meta 标记，但全项目（backend+tests）无一处读取。第二条近亲记忆可能和新内容矛盾，既没被检测也没在任何下游暴露。

**修法**：二选一——对冲突带内所有候选（不只最近一条）跑 `_detect_conflict`；或在 API/前端暴露该字段。**最低成本**：在 `_supplement_memory` 里对全部冲突带候选跑矛盾检测，不再打标了事。

### ✅ P1-10 [🔧] 部分审批的"拒绝反馈"被序列化层吞掉

**证据**（Agent 层审查实测）：`check_approval_node` 部分审批路径注入 `[REJECTED]` ToolMessage，但 `_messages_to_dicts` 序列化时依赖 `responded_ids`，被拒绝的 tool_call 因无 ToolMessage 对应而被判"孤儿"剔除（`nodes.py:174-209` 附近）——LLM 看不到哪些写操作被拒，下一轮可能重试被拒操作。

**修法**：拒绝注入的 ToolMessage 也要正确进入 `responded_ids`，或序列化层对 `[REJECTED]` 内容放行。补部分审批端到端测试。

---

## P2 — 生产成熟度 / 安全 / 定位（深挖命中的靶子）

### 🗣 P2-1 [🗣] 全站零速率限制 + 单 key 内嵌前端 bundle

**证据**：全仓库无任何请求限流；`VITE_EMA_API_KEY` 构建期打进 JS bundle（`frontend/src/api/client.ts:13-19`，`.env.example:264-271`），任何打开页面的人都能提取 key 无限调 `/api/agent/chat`、`/api/memory/ingest`；`MAX_AGENT_CONCURRENCY=4` 是并发上限不是限流。

**🗣 应答口径（主动认 + 30 秒方案）**：
> "这是我清楚的安全短板：单 key 是 ADR-004 无多租户的推论，key 打进前端 bundle 等于公开，且没有 API 限流，成本没有上限。生产上我会做三层：网关限流（按 IP+key 的令牌桶）、服务端签发短期凭证（前端不再持有静态 key）、把 `/api/memory/ingest` 这类重写入路径单独限流。"
> **不要硬辩"demo 取舍"**——主动讲方案分高。可选 🔧：加 slowapi 级 IP 限流，30 分钟可落地。

### 🗣 P2-2 [🗣+🔧] `APP_ENV=test` 环境变量即可关闭全部 API 认证

**证据**：`auth.py:40-41` 第一行 `if os.environ.get("APP_ENV") == "test": return`；`config.py:218` 默认 development。测试安全与生产安全共用进程级开关。

**🗣 应答**：承认是"可审计的隐患"，正确做法是测试用 FastAPI `dependencies_overrides` / fixture 注入而非环境变量。可选 🔧 改造（中等工作量，P2）。

### 🗣 P2-3 [🗣] webhook 非 production 且未配 secret 时接受无签名请求

**证据**：`webhook_routes.py:83-98`。**要点**：production 是 fail-closed（未配 secret 直接拒绝），这层是对的；应答口径 = "dev 便利 + 生产默认拒绝，但我意识到两个默认值（dev + 无 secret）叠加等于裸奔，生产部署时 secret 是强制的"。

### ✅ P2-4 [🔧+🗣] `ingest_git_repo_tool` 无路径沙箱

**证据**：`tools.py:319-345` repo_path 任意绝对路径，`ingestion.py:43-45` 仅校验 `.git` 存在。prompt injection 可诱导读取服务器任意本地 Git 仓库。

**🔧 改代码**：加路径白名单/前缀校验（如必须位于 `REPO_ALLOW_ROOT` 之下）。**🗣 应答**：prompt injection 当前是软防御（`prompts.py` untrusted-DATA 声明 + `<doc>/<memory>` 包裹），测试只测结构隔离不测行为隔离（`test_prompt_injection.py`），这是已知加固点。

### ✅ P2-5 [🗣] CI 缺三道门禁：前端无 tsc、后端无 integration、不构建 Docker

**证据**：`ci.yml:67-68` 只跑 `pytest tests/unit tests/api`；`ci.yml:86-88` 只跑 `npm test`（vitest 用 esbuild 不查类型）；无 Docker build job。仓库还提交了 `frontend/tsconfig.app.tsbuildinfo`。

**🗣 应答**：承认"测试多但 CI 不完整，恰好放过了最贵的两类错误（前端类型错误、镜像构建失败）"。可选 🔧：加 `tsc --noEmit`（10 分钟）、Docker build job。

### 🗣 P2-6 [🗣] 双连接池、无迁移工具、启动 DDL 清向量列

**证据**：`db/__init__.py:34`（asyncpg 池 5+10）+ `agent_service.py:123-128`（psycopg 池 max_size=5）互不感知、参数写死；`schema.py:23-47` embedding 维度变更清空全部向量列、无可回滚。

**🗣 应答**：双池是驱动差异（asyncpg vs psycopg）的技术债不是设计；"每次启动清向量列换维度"已超出简化进入危险区，误配 `EMBEDDING_MODEL` 会清空生产向量——承认这个并讲 Alembic 迁移方案。

### 🗣 P2-7 [🗣] 检索是读路径却写库（每次检索 UPDATE decay）

**证据**：`retrieval.py:786-795` 每次 `query_memories` 对 top-k 结果执行 `update_decay_batch`；`decay.py:103-118`。语义失真：`recall_count` 记录"被任意查询返回过"而非"被使用过"，垃圾查询无差别强化返回的记忆，马太效应自我强化。

**🗣 应答**："批量 UPDATE 解决了 N+1 和并发丢计数，但没解决'读路径写状态'和'曝光=价值'。改进方向：改成显式 confirm 才记 recall，或异步队列合并写。" 承认这是"被包装成优点的折中"。

### ✅ P2-8 [🔧] 读路径的 LLM 用量会计丢失瞬时失败（错误率被低估）

**证据**：`llm_service.py:95-107` tenacity 吞掉中间 429/5xx，只有最终成功写一行 usage；熔断器只在整条重试梯耗尽后计数（`resilience.py:297-308`）。provider 一直在抖但监控看不到。

**🔧 改代码（可选，低成本）**：usage 记录 attempt 级失败（或至少记一个 `retried` 标志）。**🗣 应答**：承认这是 call-level 记账的简化，系统性低估 provider 抖动。

### 🗣 P2-9 [🗣] 实体归一化：fire-and-forget 无重放、merge 后旧链接不清、0.85 阈值无测量

**证据**：`memory.py:488-517` create_task fire-and-forget，重启即丢；全项目无 `DELETE FROM memory_entities`，merge/overwrite 后被合并实体在关联表永不清除（`entity.py:137-148` 只 INSERT ON CONFLICT DO NOTHING）；`entity.py:24` `SIMILARITY_THRESHOLD=0.85` 无准确率测量。

**🗣 应答**：fire-and-forget 不阻塞写入是有意取舍；stale 链接和 0.85 是无据参数——诚实认，讲"周巡检补偿 + 定期 cleanup 脚本"。

### 🗣 P2-10 [🗣] `chat_sync`/`embed_sync` 在生产零调用——每 provider 维护双客户端是死复杂度

**证据**：接口 abstractmethod 强制（`llm.py:28-31`、`embedding.py:14-17`），生产唯一"调用"是包装器互相转发，真实调用方只有 tests。

**🗣 应答**：承认是过度设计，面试官问"sync 路径用在哪"时答"没有生产调用方，是我为对称性加的接口，实际应删"。

### ✅ P2-11 [🗣] 前端偏薄：无状态管理、单页 150-400 行、SSE 无重连

**证据**：`App.tsx` 仅 28 行；无 redux/zustand/context 状态管理；`useChat.ts` 流中断无指数退避重连，错误与正文混排（`useChat.ts:154-161`）；`ChatArea.tsx:62-63` `MAX_VISIBLE=50` 硬截断。

**🗣 应答**：定位"前端是配角、验证后端能力"，坦承非最强项（README 已承认前端深度/DevOps/安全是低优先）。**可选 🔧**：加流错误独立渲染（错误事件不再 append 进正文），这是最影响 demo 观感的一条。

### 🗣 P2-12 [🗣] 检索排序的边界近似（可辩护）

**证据**：`retrieval.py:676` multi-query 并集按插入序截断、未先按相关性排序；`retrieval.py:494-516` sparse 候选 `ORDER BY id` 先截断后 Jaccard 过滤；`decay.py:159-196` 两阶段窗口（top_k×8）不保证全局最优。

**🗣 应答**：承认这些是"确定性偏差的边界近似"，能讲清在什么数据规模下才成问题即可。

---

## P2 — 社招定位风险（代码改不动，纯应答）

### 🗣 P2-13 [🗣] 无团队协作背书 + 无真实用户

**已有策略**（README.md 和 self-introduction.md 已备）：不强调"独立完成"，改讲"端到端主导 + 工程纪律替代 review"（ADR/CI/评估集/1293 测试）。**必须补**：每个数字要有真实来源（QPS 4.8、记忆条数、token 成本），并准备好"1281 个测试没有 code review 怎么保证质量"的答案。

**🗣 应答要点**："这个项目是一个人的团队，所以我用团队工程的纪律约束自己：ADR 记录每个选型为什么、CI 把关每个提交、评估集量化每次检索改动。我承认这替代不了真实的 code review——review 能发现的跨模块问题、视角盲区，是我这个项目最大的风险敞口。"

### 🗣 P2-14 [🗣] 规模太小暴露个人项目量级

**证据**：记忆库 30 条（评估集）、10 并发 QPS 4.8、项目周期 3 个月。

**🗣 应答**：准备"为什么这个量级足够支撑技术决策"的论证（万级记忆库 pgvector hnsw 足够、拐点思维），并诚实说"生产级规模没跑过，这是我的上限"。

### 🗣 P2-15 [🗣] LangGraph 是冷门框架，面试官可能不熟或问"为什么不用 LangChain Agent"

**已有应答**（ema-deep-dive.md 决策 1）：黑盒 + HITL 控制弱 + 调试代价。**必须再补**：
- "如果公司已有 LangChain 基础设施，迁移成本与收益"——框架是薄封装，核心价值在记忆/检索/韧性层，迁移不改这些。
- "LangGraph 和自研 event loop 的边界在哪"——图模型的价值在 interrupt/checkpoint/流式，超出这三点就回归普通编排。

### 🗣 P2-16 [🗣] 岗位匹配度：记忆/RAG 很深，但 Agent 岗位面试官更常考生产可靠性

**应答策略**：把"评估有水分"主动转为"我做过诚实评估"（P0-1/2/4 的口径）；把"无限流"转为"我清楚生产安全基线"（P2-1）；把 task completed=0.5 和 unexpected_rate=0.375（DeepSeek 过度调工具）当成**最有价值的发现**主动讲——这比任何完美数字都证明你懂 Agent 的生产行为。

---

## 执行顺序（按 1 天工作量排序）

| 顺序 | 动作 | 耗时 | 对应条目 |
|------|------|------|---------|
| 1 | 修 P1-1 REST conflict 500 | 15 min | P1-1 |
| 2 | 改 P0-7 四处文档=代码（公式/threshold/rerank） | 30 min | P0-7 |
| 3 | 修 P1-3 webhook 强引用 | 20 min | P1-3 |
| 4 | 修 P1-5 embedding 熔断器名 | 5 min | P1-5 |
| 5 | 修 P0-8 中文分块 | 30 min | P0-8 |
| 6 | 跑 hard negative + memory 路径报告，重述数字 | 1-2 h | P0-1/5 |
| 7 | 修 P1-4 scenario 超时 | 20 min | P1-4 |
| 8 | 修 P1-8 CircuitOpenError 降级 | 15 min | P1-8 |
| 9 | 补 P0-2 语义通道披露 + P0-4 多次均值门禁 | 1 h | P0-2/4 |
| 10 | 写所有 🗣 话术（背熟 8 段关键口径） | 2 h | 见各条 |
| — | 可选：slowapi 限流、tsc 门禁、SSE 断连清理 | 半天 | P2-1/5, P1-2 |

**最重要的三条（只做这三条也值得）**：
1. **修掉能现场演示的崩溃**（P1-1/3/5）——面试官让跑一下就不炸。
2. **统一文档=代码**（P0-7）——消除"深挖露馅"。
3. **诚实重述评估数字**（P0-1/5）——把"1.0 自证"变成"诚实校准"，这是对抗性审查唯一无法击穿的姿态。
