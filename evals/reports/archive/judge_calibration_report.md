# LLM Judge Calibration Report

- Generated: 2026-08-11 00:09:59 UTC
- Judge provider: openai (mimo-v2.5-free)
- Samples: 12 (judged 12, judge errors 0)
- Human verdicts: 6 grounded / 6 ungrounded (2 synonym-rewrite)

## Summary

| metric | value |
|---|---|
| grounded_agreement | 1.000 |
| coverage_precision | 1.000 |
| coverage_recall | 0.833 |
| coverage_f1 | 0.833 |
| false_negative (human grounded, judge ungrounded) | 0 |
| false_positive (human ungrounded, judge grounded) | 0 |

> `false_negative` is the `ans-006` class of error: a grounded answer judged ungrounded — typically a faithful paraphrase penalized for not repeating the required-fact strings verbatim, despite the judge prompt's "允许同义改写".

## Per-sample

| id | base | human | judge | agree | human covered | judge covered | cov P/R/F1 |
|---|---|---|---|---|---|---|---|
| cal-001 | ans-001 | grounded | grounded | ✓ | pgvector, Elasticsearch, cosine | pgvector, Elasticsearch, cosine | 1.000/1.000/1.000 |
| cal-002 | ans-001 | grounded | grounded | ✓ | pgvector, Elasticsearch, cosine | pgvector, Elasticsearch, cosine | 1.000/1.000/1.000 |
| cal-003 | ans-002 | grounded | grounded | ✓ | 连接池, 占满, 超时 | 连接池, 占满, 超时 | 1.000/1.000/1.000 |
| cal-004 | ans-006 | grounded | grounded | ✓ | psycopg3, 事件循环, InMemorySaver | psycopg3, 事件循环, InMemorySaver | 1.000/1.000/1.000 |
| cal-005 | ans-003 | grounded | grounded | ✓ | InMemorySaver, 丢失 | InMemorySaver, 丢失 | 1.000/1.000/1.000 |
| cal-006 | ans-005 | grounded | grounded | ✓ | BGE-M3, 1024, 本地推理 | BGE-M3, 1024, 本地推理 | 1.000/1.000/1.000 |
| cal-007 | ans-001 | ungrounded | ungrounded | ✓ | Elasticsearch, cosine | Elasticsearch, cosine | 1.000/1.000/1.000 |
| cal-008 | ans-002 | ungrounded | ungrounded | ✓ | 超时 | - | 1.000/0.000/0.000 |
| cal-009 | ans-003 | ungrounded | ungrounded | ✓ | - | - | 1.000/1.000/1.000 |
| cal-010 | ans-005 | ungrounded | ungrounded | ✓ | - | - | 1.000/1.000/1.000 |
| cal-011 | ans-004 | ungrounded | ungrounded | ✓ | - | - | 1.000/1.000/1.000 |
| cal-012 | ans-006 | ungrounded | ungrounded | ✓ | InMemorySaver | - | 1.000/0.000/0.000 |

<details><summary>Per-sample answers & judge forensics</summary>

**cal-001** (base ans-001)
- answer: `最终选用 PostgreSQL 的 pgvector 扩展，而不是 Elasticsearch。原因：pgvector 与业务库同库，事务一致性有保障；原生支持 cosine 距离；还省去了独立 ES 集群的运维成本。`

**cal-002** (base ans-001)
- answer: `我们定的是 PostgreSQL 生态里的向量检索扩展方案，彻底放弃了 Elasticsearch。理由是同库部署、事务一致性有保证、原生支持余弦距离计算，也省掉了单独运维一套检索集群的成本。`

**cal-003** (base ans-002)
- answer: `线上 502 的根因是数据库连接池被占满。链路是：流量上涨 → 连接池没有空闲连接 → 新请求排队超时 → 网关返回 502。修复方式是重启连接池并增加连接泄漏监控。`

**cal-004** (base ans-006)
- answer: `开发环境用 InMemorySaver 是因为 psycopg3 的异步驱动和 Windows 的 ProactorEventLoop 存在兼容性冲突，AsyncPostgresSaver 因此在 Windows 下不可用；生产目标是 Linux，可以正常用 PostgresSaver。`

**cal-005** (base ans-003)
- answer: `在 Windows 开发环境下不行，重启后对话状态会丢失。AsyncPostgresSaver 因为 psycopg3 异步实现的兼容性问题在 Windows 下不可用，所以降级成了 InMemorySaver；生产 Linux 环境用 PostgresSaver，可以跨会话恢复。`

**cal-006** (base ans-005)
- answer: `嵌入模型选的是 BGE-M3，1024 维，支持中英双语，本地推理零 API 成本；对比 OpenAI 的 text-embedding-3-large，准确率相当但成本和延迟更优。`

**cal-007** (base ans-001)
- answer: `最终选型是 Qdrant，而不是 Elasticsearch。[UNGROUNDED: 声称使用了 Qdrant，但 context 只提到 pgvector 与 Elasticsearch，从未提及 Qdrant] 理由是同库部署、原生支持 cosine 距离。`
- judge ungrounded claims: ['最终选型是 Qdrant', '理由是同库部署']

**cal-008** (base ans-002)
- answer: `线上 502 的根因是内存泄漏。[UNGROUNDED: 声称根因是内存泄漏，context 明确记载根因是数据库连接池被占满，且禁止编造「内存泄漏」这个根因] 流量上涨后内存持续增长，最终请求超时返回 502。`
- judge ungrounded claims: ['线上 502 的根因是内存泄漏。', '流量上涨后内存持续增长']

**cal-009** (base ans-003)
- answer: `Windows 开发环境下重启后对话状态不会丢，状态会被完整保存下来。[UNGROUNDED: 声称 Windows 下状态不会丢失，context 明确说降级为 InMemorySaver 后重启状态会丢失]`
- judge ungrounded claims: ['Windows 开发环境下重启后对话状态不会丢，状态会被完整保存下来']

**cal-010** (base ans-005)
- answer: `嵌入模型最终选了 OpenAI 的 text-embedding-3-large。[UNGROUNDED: 声称选用了 OpenAI 的 text-embedding-3-large，context 明确说选用的是 BGE-M3，OpenAI 只是对比项] 因为它的准确率更高。`
- judge ungrounded claims: ['声称选用了 OpenAI 的 text-embedding-3-large', '声称因为它的准确率更高']

**cal-011** (base ans-004)
- answer: `导入本地仓库提交历史很简单，把仓库文件夹拖到网页上传即可，系统会自动解析提交历史。[UNGROUNDED: 声称通过网页上传导入，context 只说明要用 ingest_git_repo_tool 并传入 repo_path 参数，从未提到网页上传]`
- judge ungrounded claims: ['声称通过网页上传导入，但上下文只说明要用 ingest_git_repo_tool 并传入 repo_path 参数，从未提到网页上传']

**cal-012** (base ans-006)
- answer: `开发环境用 InMemorySaver 是因为它比 PostgresSaver 性能更好、启动更快。[UNGROUNDED: 声称选 InMemorySaver 是出于性能优势，context 给出的真实原因是 psycopg3 异步实现与 Windows 事件循环冲突导致 AsyncPostgresSaver 不可用]`
- judge ungrounded claims: ['声称选 InMemorySaver 是出于性能优势']

</details>