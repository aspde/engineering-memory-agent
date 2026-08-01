# Source 引用可点击

## 问题描述

EMA 聊天界面中，Agent 通过 `search_memories_tool` / `retrieve_chunks_tool` 检索到的 Sources 以纯文本 snippet 展示在 `📚 Sources` expander 中，用户无法点击跳转查看详情。

**根因**：两个检索 tool 将结构化结果（memory ID、chunk document_id、relevance score 等）格式化为纯文本字符串返回给 LLM，结构化数据在 tool 层被丢弃，后续的 `_extract_tool_traces` 和前端只能拿到截断的纯文本字符串。

## 数据流追踪

```
search_memories_tool()
  ├── query_memories() → list[dict]   ← 有完整结构化数据（id, summary, score, decay…）
  ├── 格式化为纯文本字符串           ← 结构化数据在此丢弃 ✗
  └── return str                      ← ToolMessage.content = 纯文本

_extract_tool_traces()
  ├── 读取 ToolMessage.content         ← 只有字符串
  ├── 截断为 200 字符 → snippet       ← 无法提取 ID、链接
  └── return [{type, snippet}]

前端 _render_message()
  └── st.caption("memory — How to...") ← 纯文本，不可点击
```

## 方案选择

### 方案 A（推荐）：Tool 输出 JSON 信封

工具返回 `{"display": "<LLM 可读文本>", "sources": [{结构化数据}]}`。`display` 给 LLM 读，`sources` 给前后端渲染链接。

**优点**：
- 改动集中在 tool 层和消费端，不影响 LangGraph 框架逻辑
- `display` 字段保持与当前纯文本一致的可读性
- 已有 3 个 tool（write/extract/conflict）返回 JSON，这是延续现有模式

**缺点**：
- 需要所有消费者（nodes.py, routes.py, frontend）同步更新解析逻辑

### 方案 B：Tool 结果走 State 侧通道

Tool 把 sources 存入 AgentState 的新字段 `retrieved_sources`，不走 ToolMessage.content。

**优点**：
- ToolMessage.content 保持不变，对 LLM 零影响

**缺点**：
- 需要修改 AgentState 定义、graph 结构、checkpoint schema
- 引入隐式状态共享，增加调试复杂度
- 与 ReAct 循环中"通过 ToolMessage 传递信息"的模式不一致

**选择方案 A**。

## 接口定义

### Tool 输出格式

```json
{
  "display": "Found 5 relevant memories:\n[1] (relevance: 0.85, decay: 1.00) How to deploy...",
  "sources": [
    {
      "id": "550e8400-e29b-...",
      "type": "memory",
      "summary": "How to deploy the app to production",
      "relevance": 0.85,
      "decay": 1.0
    }
  ]
}
```

### `ChatResponse.sources` 类型变更

```python
# 旧
sources: list[dict[str, Any]]  # [{type, snippet}]

# 新
sources: list[dict[str, Any]]  # [{id, type, snippet, relevance, document_id?}]
```

### 前端 Source item 渲染

- **memory 类型**：可点击 badge → `st.session_state["mem_filter_id"]` → `st.switch_page("pages/memories.py")`
- **chunk 类型**：expander → 展开显示完整 snippet + document_id + relevance

## 影响文件

| 文件 | 改动 |
|------|------|
| `agent/tools.py` | `search_memories_tool`, `retrieve_chunks_tool` → JSON 输出 |
| `agent/nodes.py` | `generate_final_node` L216-225 → 解析 `display` 字段 |
| `backend/api/routes/agent_routes.py` | `_extract_tool_traces` → 解析 `sources` 字段 |
| `frontend/app.py` | `_format_tool_result` → 提取 `display`；`_render_message` → 可点击 sources |
| `frontend/pages/memories.py` | 接收 `mem_filter_id` session_state 参数 |

## 兼容性

- 所有 JSON 解析处带 try/except → fallback 到当前纯文本逻辑
- 空结果（"No relevant memories found."）保持纯字符串，不改变行为
- LLM 看到的 `display` 字符串与当前纯文本格式一致

## 实施步骤

1. `agent/tools.py` — 检索 tool 改为 JSON 输出
2. `agent/nodes.py` — 更新 LLM 上下文构建逻辑
3. `backend/api/routes/agent_routes.py` — 更新 source 提取
4. `frontend/app.py` — 更新 tool 摘要和 source 渲染
5. `frontend/pages/memories.py` — 接收 session_state 筛选参数
6. 运行 `pytest tests/unit/test_agent_tools.py tests/api/test_agent_routes.py`
