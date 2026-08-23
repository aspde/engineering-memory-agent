# EMA 检索 Hard-Negative 判别力评估报告

- **生成时间**：2026-08-10T12:33:37.290585+00:00
- **语料**：27 条 hard_negative query（来自 `query_candidates.jsonl`）
- **检索路径**：memory:norank@5, threshold 0.3（生产默认，无 rerank）
- **主评估对比基线**：`run_eval --retriever memory` 在相同语料上 recall@5=1.000 / mrr=0.944

## 汇总指标（27 条平均）

| 指标 | 值 | 含义 |
|---|---|---|
| target_recall@5 | 100.0% | 目标被召回到 top-5 的比例（判别力） |
| distractor_intrusion@5 | 96.3% | 陷阱被误召回比例（越低越好） |
| hard_neg_pass@5 | 59.3% | 目标命中且不被陷阱压过/顶替 |
| mrr | 0.790 | 目标排名倒数的均值（主评估同路径 0.944） |
| worse_than_random | 11 / 27 | 陷阱排在目标之前 / 目标未命中而陷阱命中 |

## 逐条明细（27 条 hard negative）

| id | query | target_seed | distractor_seed | target_rank | distractor_rank | discriminated |
|---|---|---|---|---|---|---|
| qg-seed-001-hardneg | pgvector 的索引参数怎么调才能避免召回率低？ | seed-010 | seed-001 | 1 | 3 | ✅ |
| qg-seed-002-hardneg | 那个双HITL的checkpoint具体是怎么设计的？interrupt是怎么用的？ | seed-015 | seed-002 | 1 | 3 | ✅ |
| qg-seed-003-hardneg | 嵌入模型选型时，BGE-M3 在 CPU 上的推理速度怎么样？ | seed-009 | seed-003 | 2 | 1 | ❌ |
| qg-seed-004-hardneg | PostgresSaver 在 Windows 上报错，我是不是该直接换成 InMemorySaver？ | seed-008 | seed-004 | 2 | 1 | ❌ |
| qg-seed-005-hardneg | rerank 分数低于多少应该被过滤掉？ | seed-024 | seed-005 | 1 | 2 | ✅ |
| qg-seed-006-hardneg | 去重的相似度阈值是0.92/0.75/0.60，这四个分支具体怎么区分？ | seed-019 | seed-006 | 2 | 1 | ❌ |
| qg-seed-008-hardneg | PostgresSaver 在 Windows 上用 psycopg3 异步模式，因为 ProactorEventLoop 导致连接池初始化抛 RuntimeError，LangGraph checkpoint 写入失败，Agent 第二轮对话就丢了上下文。这种情况是不是只能用 InMemorySaver 降级？生产环境还能用 PostgresSaver 吗？ | seed-004 | seed-008 | 2 | 1 | ❌ |
| qg-seed-009-hardneg | BGE-M3嵌入模型性能怎么样，适不适合我们项目？ | seed-003 | seed-009 | 1 | 2 | ✅ |
| qg-seed-010-hardneg | 为什么我 pgvector 的 ivfflat 索引在小数据量下召回率这么低？我是不是应该把向量检索的 top_k 调大一点，多召回一些候选再让 rerank 处理？ | seed-023 | seed-010 | 2 | 1 | ❌ |
| qg-seed-011-hardneg | 实体归一化时，如果模型输出被markdown代码块包裹导致解析失败，有什么兜底策略吗？ | seed-018 | seed-011 | 1 | 2 | ✅ |
| qg-seed-012-hardneg | Agent 状态图中的 finalize 节点在什么情况下会被强制触发？正常流程和异常流程的区别是什么？ | seed-013 | seed-012 | 2 | 1 | ❌ |
| qg-seed-013-hardneg | HITL 有几个 checkpoint？第一个在 resolve_conflict，第二个在 finalize 前，这两个 interrupt 分别要用户确认什么？ | seed-015 | seed-013 | 1 | 2 | ✅ |
| qg-seed-014-hardneg | 选嵌入模型时，OpenAI和Anthropic的API哪个好？ | seed-003 | seed-014 | 2 | 1 | ❌ |
| qg-seed-016-hardneg | 为什么在 search_memories 里用 decay_factor 加权重？选 Ebbinghaus 衰减而不是简单时间排序的原因是什么？ | seed-027 | seed-016 | 1 | 2 | ✅ |
| qg-seed-017-hardneg | 记忆提取时，某个阶段解析失败会不会阻断其他阶段？ | seed-011 | seed-017 | 3 | 1 | ❌ |
| qg-seed-018-hardneg | 实体归一化的相似度阈值怎么调优？ | seed-006 | seed-018 | 2 | 1 | ❌ |
| qg-seed-019-hardneg | 四分支去重里的 0.92/0.75/0.60 这些阈值是人工标的吗？ | seed-006 | seed-019 | 1 | 2 | ✅ |
| qg-seed-021-hardneg | `_dedup_relations` 函数里的四分支逻辑是怎么写的？它用相似度阈值区分 merge/conflict/insert 吗？ | seed-019 | seed-021 | 1 | 2 | ✅ |
| qg-seed-022-hardneg | 多请求并发时数据被覆盖导致状态错乱怎么解决？ | seed-007 | seed-022 | 1 | — | ✅ |
| qg-seed-023-hardneg | ivfflat 索引在小数据集上 top_k 召回结果顺序不稳定，是不是应该多召回一些候选再精排？ | seed-010 | seed-023 | 1 | 2 | ✅ |
| qg-seed-024-hardneg | 为什么用 cross-encoder 做 rerank 而不是 LLM？它的分数阈值怎么设？ | seed-005 | seed-024 | 1 | 2 | ✅ |
| qg-seed-025-hardneg | EMA 的 Phase 1 到 Phase 4 分别是什么？ | seed-026 | seed-025 | 1 | 2 | ✅ |
| qg-seed-026-hardneg | 想找一下 EMA 从 RAG demo 到记忆 Agent 的演进过程，好像有 Phase 1-4 的完整背景和路线图？ | seed-025 | seed-026 | 1 | 2 | ✅ |
| qg-seed-027-hardneg | 你们用 Ebbinghaus 衰减来排序记忆，具体是怎么做的？ | seed-016 | seed-027 | 2 | 1 | ❌ |
| qg-seed-028-hardneg | chunks 表和 memories 表分开后，retrieve 函数在 chunks 表上的召回流程是直接 vector_search 还是要先 top_k*4 过采样再 rerank？ | seed-023 | seed-028 | 1 | 2 | ✅ |
| qg-seed-029-hardneg | 连接器生态现在发展到哪一步了，还有哪些计划？ | seed-026 | seed-029 | 2 | 1 | ❌ |
| qg-seed-030-hardneg | 本地 BGE-M3 做 embedding 的吞吐和延迟数据有记录吗？如果和 API 请求抢 CPU，接口延迟会受多大影响？ | seed-009 | seed-030 | 1 | 3 | ✅ |

## 诚实解读

这份 27 条 hard-negative 集考验的不是“能不能找到”，而是“会不会被带偏”。主评估集（30 条自问自答 query）在相同 memory:norank@k5 路径上是 recall@5=1.000 / mrr=0.944——那只能证明检索器能找到自己出题时反向生成的目标。本集每条 query 都配了一个表面词高度重合、语义却不同的陷阱 seed，数字越低越说明检索器在“看似相关”的候选里缺乏判别力。

**目标召回**：100.0% 的 query 能穿透表面词命中真目标（27/27）。100.0% 的目标命中率说明BGE-M3 的稠密相似度基本能识别出正确目标——问题不出在“找得到”，而出在“排得对”。

**陷阱入侵**：96.3% 的 query 把陷阱 seed 也召回了（26/27）。陷阱进入候选集的比例很高——这正是 hard-negative 的设计意图：陷阱和目标共享表层词，向量检索在 top-5 里几乎不可避免地把两者都捞进来，真正的考验是谁排前面。

**综合通过率**：59.3%（16/27）——目标被召回且没有被陷阱压过/顶替。这是最重要的数字：59.3% 意味着有 11/27（40.7%）被陷阱误导，判别力有明显短板——不能拿主评估集的满分当真实水平。

**MRR** = 0.790，而主评估集同路径为 0.944。MRR 的落差量化了“表面词重合到底让目标排名掉了多少”——即便目标命中了，如果陷阱总是排在前面，MRR 也会被拖低。

**worse_than_random = 11**：11/27（40.7%）的 query 里，陷阱排在目标前面（本集目标全命中，所以全部是“双方都被召回、陷阱排名更靠前”的情形：26/27 条双方都进 top-5，其中 11 条陷阱压在目标上，约占双方都命中例子的 42.3%）。对照基准：若目标和陷阱在 top-5 内的排序随机打乱，陷阱压过目标的概率约 1/2——也就是说，“把目标排在陷阱之上”这件事，检索器只是比抛硬币略好一点点，远谈不上稳健。逐条如下：
  - `qg-seed-003-hardneg`（陷阱 seed-003 排第 1，目标 seed-009 排第 2）：嵌入模型选型时，BGE-M3 在 CPU 上的推理速度怎么样？
  - `qg-seed-004-hardneg`（陷阱 seed-004 排第 1，目标 seed-008 排第 2）：PostgresSaver 在 Windows 上报错，我是不是该直接换成 InMemorySaver？
  - `qg-seed-006-hardneg`（陷阱 seed-006 排第 1，目标 seed-019 排第 2）：去重的相似度阈值是0.92/0.75/0.60，这四个分支具体怎么区分？
  - `qg-seed-008-hardneg`（陷阱 seed-008 排第 1，目标 seed-004 排第 2）：PostgresSaver 在 Windows 上用 psycopg3 异步模式，因为 ProactorEventLoop 导致连接池初始化抛 RuntimeError，LangGraph checkpoint 写入失败，Agent 第二轮对话就丢了上下文。这种情况是不是只能用 InMemorySaver 降级？生产环境还能用 PostgresSaver 吗？
  - `qg-seed-010-hardneg`（陷阱 seed-010 排第 1，目标 seed-023 排第 2）：为什么我 pgvector 的 ivfflat 索引在小数据量下召回率这么低？我是不是应该把向量检索的 top_k 调大一点，多召回一些候选再让 rerank 处理？
  - `qg-seed-012-hardneg`（陷阱 seed-012 排第 1，目标 seed-013 排第 2）：Agent 状态图中的 finalize 节点在什么情况下会被强制触发？正常流程和异常流程的区别是什么？
  - `qg-seed-014-hardneg`（陷阱 seed-014 排第 1，目标 seed-003 排第 2）：选嵌入模型时，OpenAI和Anthropic的API哪个好？
  - `qg-seed-017-hardneg`（陷阱 seed-017 排第 1，目标 seed-011 排第 3）：记忆提取时，某个阶段解析失败会不会阻断其他阶段？
  - `qg-seed-018-hardneg`（陷阱 seed-018 排第 1，目标 seed-006 排第 2）：实体归一化的相似度阈值怎么调优？
  - `qg-seed-027-hardneg`（陷阱 seed-027 排第 1，目标 seed-016 排第 2）：你们用 Ebbinghaus 衰减来排序记忆，具体是怎么做的？
  - `qg-seed-029-hardneg`（陷阱 seed-029 排第 1，目标 seed-026 排第 2）：连接器生态现在发展到哪一步了，还有哪些计划？

**这些数字暴露了检索器什么特性？**
- top-5 对表层词重合高度宽容——陷阱和目标共享的关键词（如 pgvector、PostgresSaver、Ebbinghaus、HITL/interrupt）足以把不相关记忆拉进候选。这是 BGE-M3 稠密检索的典型特征：它擅长“语义接近”，但不做“问题意图”判别。
- 排序对共享表面词的敏感度高于对问题类型的敏感度：当 query 同时命中目标和陷阱的词汇时，排名基本由“谁与 query 共享的词汇更多/更醒目”决定，而非“谁真正回答了这个问题”。这正是主评估集测不出来的维度——自问自答的 query 只与目标共享词汇，陷阱不存在。

**面试时怎么讲？**
- 先说结论：主评估集 Recall@5=1.0 是**自问自答的假满分**，只能证明“找得到”，不能证明“判别力”。
- 然后给出本集数字：27 条 hard negative 上，目标召回 100.0%、陷阱入侵 96.3%、综合通过 59.3%、MRR 0.790（主评估同路径 0.944）、worse_than_random 11/27。
- 解读两点：(a) 高目标召回 + 高陷阱入侵的组合说明检索器“宽进”——它把相关和不相关的表层重合候选都捞进来，靠的是 top-5 窗口兜底而不是精确判别；(b) worse_than_random=11 说明陷阱压过目标是个真实但局部的现象，暴露的是纯向量检索在“问题意图 vs 词汇重合”上的盲区。
- 如果有下一步：这正是无监督 rerank / 检索后意图判别（query 重写、hybrid 融合、cross-encoder）应该改善的环节——hard-negative 集可以直接当作这些方案的回归测试集。

---

## 改进方向验证（已实测，已落地）

2026-08-10 用本 hard-negative 集做检索改进的 A/B 实验，对比三个方向后选定并实现了 **bounded cross-encoder 重排**（`query_memories(use_cross_encoder=True)`）：

| 检索路径 | pass@5 | target@5 | intrusion@5 | MRR | worse |
|---|---|---|---|---|---|
| 纯向量（baseline） | 59.3% | 100.0% | 96.3% | 0.790 | 11 |
| 全量 cross-encoder 重排（含 `_RERANK_FLOOR`） | 77.8% | 96.3% | 85.2% | 0.874 | 6 |
| **bounded-CE top-3 重排（已实现）** | **81.5%** | **100.0%** | 96.3% | **0.889** | **5** |

**机理**：候选经 `search_memories` 已按 decay 加权相似度排序，竞争区就是前 3 名（它们才与 query 共享表层词）。只对前 3 名调 cross-encoder 打分重排、其余保持原序——**比全量 CE 更好的原因**：不给远端候选“意外翻身”的机会，且不应用 `_RERANK_FLOOR`（memory 候选已过 threshold=0.3 语义门槛，CE 只重排不过滤；全量 CE 的 floor 曾把 q022 的真目标误删）。代价是每 query 增加约 1-2s CPU 推理（3 对打分），故默认关闭、显式启用。

**排除的方向**：sparse/hybrid 融合会**放大**表层词重合（陷阱恰与 query 共享专有名词，BM25 让陷阱排更高）；query 重写主要解决“词面不重合”的召回问题，与 hard-negative 的“词面过重合”病根相反。

**边界（诚实记录）**：bounded-CE 修复 6 条（003/004/006/008/014/017/029），剩余 5 条失败（010/011/012/018/027）都是“真目标排在 top-3 之外、陷阱在 zone 内”——zone 只覆盖前 3 名，目标若跌出竞争区则无从救回。这是 zone 边界的固有取舍，不是回归。
