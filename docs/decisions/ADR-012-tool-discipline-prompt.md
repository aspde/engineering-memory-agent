# ADR-012: 工具纪律 prompt（agent.system v6）与检索工具描述边界

**日期**: 2026-08-24

**状态**: 已接受

## 背景

任务级端到端评测（`evals/run_task_eval`，8 个标注任务驱动真实 agent 图）持续暴露同一短板：`tool_recall 0.94`（工具选择意图准）但 `unexpected_rate 0.375`（3/8 任务调了预期外工具），严格 `completed` 被拖到 0.5。三种具体形态：

1. **task-001**（单记忆事实题）：该调 `search_memories_tool` 却调了 `query_rewrite_and_search_tool + retrieve_chunks_tool`，预期工具一次没调；
2. **task-005**（notify）：多带一次不必要的 `retrieve_chunks_tool`；
3. 历史轮次：单记忆问题调 4 个工具、概念查询循环撞 max_steps。

根因在引导文本，不在机制：

- `agent.system` v5 说 "Search relevant memories **and documents** first"——复数、鼓励双库都查；全文无停手纪律；
- `retrieve_chunks_tool` docstring："Use this as the default document search, **or when memory search doesn't return enough context**"——主动邀请链式调用；
- `query_rewrite_and_search_tool` docstring 对 "conceptual or abstract queries" 宽泛自荐。

## 决策

**三层引导文本修复，零机制变更**（不加轨迹节流、不加分类器）：

1. **`agent.system` v5→v6**：把 "Search memories and documents first" 替换为工具纪律段——按答案所在位置选存储（团队知识→长期记忆；文件/文档内容→文档 chunk；措辞空泛才用改写）、每个信息需求一次检索（多段问题每段一次合法）、上下文已覆盖即立即作答（薄结果不是换工具重试的理由）、寒暄免检索直接回复。
2. **三个检索工具 docstring 边界收紧**：
   - `search_memories_tool`：声明为团队知识问题的 DEFAULT first choice，结果已覆盖时停止、不得链式调用其他检索工具；
   - `retrieve_chunks_tool`：删除"memory search 不够再查我"，改为仅当问题指向文档/文件内容；
   - `query_rewrite_and_search_tool`：收窄为 last-resort（措辞空泛到无法匹配任何存储文本），且只能作为该需求的第一次检索、不得作为重试。
3. **tsel-006 重标对齐新策略**：该条标注（与 task-004 同一查询族）原按旧策略期望 query_rewrite，重标为期望记忆检索、允许 rewrite 作为替代；同时修正 query_rewrite docstring 中与重标自相矛盾的示例句。

## 理由

1. **数据支撑**：`unexpected_rate` 三次基线观测全部稳定 0.375（非噪声）；改动后三次复测全部 **0.000**——目标指标归零且跨轮次稳定。
2. **轨迹精确化而非能力损失**：干净轨迹里该调的工具都调了——task-001 精确一次调用，task-002 双库各查一次（多段问题的合法检索被"每段一次"条款保护，未被停手纪律误伤），task-003 检索+写入正确，task-007 寒暄保持零调用。
3. **tool_selection 单决策套件同向改善**：accuracy 0.867→0.933（15 条中 14 对）、unexpected_rate 0.000，高于 CI 门限 0.68。
4. **最小机制**：与 ADR-010 同一哲学——模型行为问题优先修引导文本而不是加运行时约束。max_steps 已兜底失控循环；prompt 纪律解决的是"每次都多转一圈"的系统性浪费。

## 代价与保留

- **completed 干净复测已补（run 4，provider 零故障轮）**：`completed` **0.625**（历史最高，基线 0.375-0.500）、`unexpected_rate` 第 4 次连续归零、groundedness/citation 双 1.000。唯一 `within_budget` miss 是 task-001 撞 AGENT_TIMEOUT=180s 墙钟（provider 慢），非行为回归。此前三轮复测因 provider 503/超时污染答案指标，轨迹指标不受影响（工具调用发生在最终答案生成之前）。
- **task-008 已知取舍**：该任务（问 EMA 自身功能）改动后三次复测连续零调用直接作答（改动前调 3 次 search）。已验证两个事实当前均可检索到——不是检索不到，而是模型判断"问我自己的功能不需要查库"；答案 groundedness 保持 1.000 无捏造。这是停手纪律在 agent 自身功能类问题上的副作用，严格轨迹分归零；消除需在 prompt/docstring 强化"涉及 EMA 自身用法也先搜记忆库"，属后续迭代。
- **tsel-006 重标是策略变更的忠实编码**，不是教测试：标签记录的是新政策下的正确行为，notes 字段与 ADR 均有留痕。

## 拐点

若后续复测出现以下任一信号，升级为机制层修复（轨迹级节流）或 prompt 再迭代：

1. `unexpected_rate` 回升到 >0.125（≥2/8 任务）且 prompt 无回归；
2. task-008 式零调用扩散到其他操作类/知识类任务（过度抑制信号）；
3. 新任务类别引入后停手纪律系统性漏检。

## 后果

- `backend/service/prompts.py` agent.system v6；快照已重生成（`tests/unit/_prompt_snapshot.json`）。
- `backend/agent/tools.py` 三个检索工具 docstring 收紧；函数体零变更。
- `evals/data/tool_selection.jsonl` tsel-006 重标。
- 单测新增 `TestAgentSystemToolDiscipline` / `TestRetrievalToolDescriptionBoundaries` 钉住纪律文本与边界。
- 复测数据见 `evals/reports/task_eval_report.md`「工具纪律 prompt（agent.system v6）复测」节与 `evals/reports/task_eval_{pre,post}_discipline_run*.md`。
