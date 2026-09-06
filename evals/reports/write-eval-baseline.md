# EMA LLM Behavior Evaluation Report

- Generated: 2026-09-05 13:23:55 UTC
- Suites: 3
- Items: 40
- Execution errors: 0
- Judge degradations: 0

## Overall

### 写入冲突检测

| metric | conflict_tp | conflict_fp | conflict_fn | conflict_tn | conflict_precision | conflict_recall | conflict_f1 | conflict_accuracy | conflict_false_positive_rate | conflict_false_negative_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| **overall** | 0.550 | 0.000 | 0.000 | 0.450 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |

### 写入合并质量 (LLM judge)

| metric | merge_fact_coverage | merge_faithfulness | merge_completeness |
|---|---|---|---|
| **overall** | 1.000 | 1.000 | 1.000 |

### 自动记忆门控

| metric | worthy_tp | worthy_fp | worthy_fn | worthy_tn | worthy_precision | worthy_recall | worthy_f1 | worthy_accuracy | worthy_false_positive_rate | worthy_false_negative_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| **overall** | 0.500 | 0.083 | 0.000 | 0.417 | 0.857 | 1.000 | 0.923 | 0.917 | 0.083 | 0.000 |


## 写入冲突检测 by category

| category | conflict_tp | conflict_fp | conflict_fn | conflict_tn | conflict_precision | conflict_recall | conflict_f1 | conflict_accuracy | conflict_false_positive_rate | conflict_false_negative_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| direct_contradiction | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| parameter_change | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| status_reversal | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| refinement | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| supplement | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |

## 写入合并质量 by category

| category | merge_fact_coverage | merge_faithfulness | merge_completeness |
|---|---|---|---|
| paraphrase | 1.000 | 1.000 | 1.000 |
| partial_overlap | 1.000 | 1.000 | 1.000 |

## 自动记忆门控 by category

| category | worthy_tp | worthy_fp | worthy_fn | worthy_tn | worthy_precision | worthy_recall | worthy_f1 | worthy_accuracy | worthy_false_positive_rate | worthy_false_negative_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| technical_decision | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| incident_lesson | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| how_to | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |
| chitchat | 0.000 | 0.250 | 0.000 | 0.750 | 0.000 | 0.000 | 0.000 | 0.750 | 0.250 | 0.000 |
| question | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |
| action_request | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 |

<details><summary>Per-query detail (write_conflict)</summary>

**wc-001** (direct_contradiction)
- conflict: predicted=True (tp=1.000 fp=0.000 fn=0.000 tn=0.000)

**wc-002** (direct_contradiction)
- conflict: predicted=True (tp=1.000 fp=0.000 fn=0.000 tn=0.000)

**wc-003** (parameter_change)
- conflict: predicted=True (tp=1.000 fp=0.000 fn=0.000 tn=0.000)

**wc-004** (status_reversal)
- conflict: predicted=True (tp=1.000 fp=0.000 fn=0.000 tn=0.000)

**wc-005** (supplement)
- conflict: predicted=False (tp=0.000 fp=0.000 fn=0.000 tn=1.000)

**wc-006** (refinement)
- conflict: predicted=False (tp=0.000 fp=0.000 fn=0.000 tn=1.000)

**wc-007** (refinement)
- conflict: predicted=False (tp=0.000 fp=0.000 fn=0.000 tn=1.000)

**wc-008** (supplement)
- conflict: predicted=False (tp=0.000 fp=0.000 fn=0.000 tn=1.000)

**wc-009** (direct_contradiction)
- conflict: predicted=True (tp=1.000 fp=0.000 fn=0.000 tn=0.000)

**wc-010** (supplement)
- conflict: predicted=False (tp=0.000 fp=0.000 fn=0.000 tn=1.000)

**wc-011** (status_reversal)
- conflict: predicted=True (tp=1.000 fp=0.000 fn=0.000 tn=0.000)

**wc-012** (refinement)
- conflict: predicted=False (tp=0.000 fp=0.000 fn=0.000 tn=1.000)

**wc-013** (parameter_change)
- conflict: predicted=True (tp=1.000 fp=0.000 fn=0.000 tn=0.000)

**wc-014** (parameter_change)
- conflict: predicted=True (tp=1.000 fp=0.000 fn=0.000 tn=0.000)

**wc-015** (parameter_change)
- conflict: predicted=True (tp=1.000 fp=0.000 fn=0.000 tn=0.000)

**wc-016** (status_reversal)
- conflict: predicted=True (tp=1.000 fp=0.000 fn=0.000 tn=0.000)

**wc-017** (supplement)
- conflict: predicted=False (tp=0.000 fp=0.000 fn=0.000 tn=1.000)

**wc-018** (refinement)
- conflict: predicted=False (tp=0.000 fp=0.000 fn=0.000 tn=1.000)

**wc-019** (refinement)
- conflict: predicted=False (tp=0.000 fp=0.000 fn=0.000 tn=1.000)

**wc-020** (parameter_change)
- conflict: predicted=True (tp=1.000 fp=0.000 fn=0.000 tn=0.000)

</details>

<details><summary>Per-query detail (write_merge)</summary>

**wm-001** (partial_overlap)
- merge: coverage=1.000 (len=105)
- judge: faithfulness=1.000 completeness=1.000
- merged: `检索服务默认跳过 cross-encoder rerank（A/B 实测小语料下打分噪声会误伤低分相关结果）；生产对话路径则已锁死 LLM rerank——实测占单轮延迟 58%，且不改变召回集合、仅微调排序。…`

**wm-002** (paraphrase)
- merge: coverage=1.000 (len=99)
- judge: faithfulness=1.000 completeness=1.000
- merged: `团队决定用 PostgreSQL + pgvector 做向量检索（而非 Elasticsearch），业务数据与向量同库；选择原因是省去独立 ES 集群的运维成本，且与业务库同库可保证事务一致性。…`

**wm-003** (partial_overlap)
- merge: coverage=1.000 (len=119)
- judge: faithfulness=1.000 completeness=1.000
- merged: `Windows 下 psycopg3 异步驱动与 ProactorEventLoop 不兼容，checkpointer 降级为 InMemorySaver；测试环境则用 NullPool 规避 asyncpg 连接池跨事件循环复用的问题。…`

**wm-004** (partial_overlap)
- merge: coverage=1.000 (len=114)
- judge: faithfulness=1.000 completeness=1.000
- merged: `Agent 单轮 ReAct 循环默认上限 5 步，到达上限强制收束到 generate_final 而非报错中断；巡检任务按类型放宽步数预算（daily 15 步、weekly 20 步），因全量扫描所需搜索步数远超交互轮次。…`

**wm-005** (paraphrase)
- merge: coverage=1.000 (len=90)
- judge: faithfulness=1.000 completeness=1.000
- merged: `自动记忆捕获在最终回答交付后排入后台任务执行，不阻塞 SSE 流；并发由信号量限制为 4；每次捕获最多消耗 7 次 LLM 调用（门控 1 次、提取 3 次、嵌入加相似度扫描等）。…`

**wm-006** (paraphrase)
- merge: coverage=1.000 (len=124)
- judge: faithfulness=1.000 completeness=1.000
- merged: `冲突仲裁提供四个选项：keep_existing 保留旧记忆、overwrite 用新内容覆盖、merge LLM 合并、keep_both 两条并存；无人值守的巡检流程遇到写冲突时自动选择 keep_both，让新旧记忆并存而不卡死等待人…`

**wm-007** (paraphrase)
- merge: coverage=1.000 (len=134)
- judge: faithfulness=1.000 completeness=1.000
- merged: `**Merged summary:**

jieba 分词结果落到 chunks.tokens 列并建 GIN 索引，中文 sparse 检索复杂度从 O(N) 全表扫描降到 O(log N)；1000 条语料下稳态延迟从 641ms 降到…`

**wm-008** (partial_overlap)
- merge: coverage=1.000 (len=150)
- judge: faithfulness=1.000 completeness=1.000
- merged: `实体抽取改进：prompt 加入 few-shot 示例后 entity_recall 从 0.781 升至 0.927；同时函数调用通道在生成期约束 entity type 枚举，从机制上杜绝非法 type 的产生，DeepSeek th…`

</details>

<details><summary>Per-query detail (auto_gate)</summary>

**ag-001** (technical_decision)
- gate: predicted_worthy=True (tp=1.000 fp=0.000 fn=0.000 tn=0.000)

**ag-002** (incident_lesson)
- gate: predicted_worthy=True (tp=1.000 fp=0.000 fn=0.000 tn=0.000)

**ag-003** (how_to)
- gate: predicted_worthy=True (tp=1.000 fp=0.000 fn=0.000 tn=0.000)

**ag-004** (chitchat)
- gate: predicted_worthy=False (tp=0.000 fp=0.000 fn=0.000 tn=1.000)

**ag-005** (question)
- gate: predicted_worthy=False (tp=0.000 fp=0.000 fn=0.000 tn=1.000)

**ag-006** (action_request)
- gate: predicted_worthy=False (tp=0.000 fp=0.000 fn=0.000 tn=1.000)

**ag-007** (chitchat)
- gate: predicted_worthy=False (tp=0.000 fp=0.000 fn=0.000 tn=1.000)

**ag-008** (technical_decision)
- gate: predicted_worthy=True (tp=1.000 fp=0.000 fn=0.000 tn=0.000)

**ag-009** (chitchat)
- gate: predicted_worthy=True (tp=0.000 fp=1.000 fn=0.000 tn=0.000)

**ag-010** (how_to)
- gate: predicted_worthy=True (tp=1.000 fp=0.000 fn=0.000 tn=0.000)

**ag-011** (chitchat)
- gate: predicted_worthy=False (tp=0.000 fp=0.000 fn=0.000 tn=1.000)

**ag-012** (incident_lesson)
- gate: predicted_worthy=True (tp=1.000 fp=0.000 fn=0.000 tn=0.000)

</details>
