import type { ToolCall } from '../types';
import RichText from './RichText';

interface ToolCallPanelProps {
  toolCalls: ToolCall[];
}

/** Tool name → human-readable Chinese label. */
const TOOL_LABELS: Record<string, string> = {
  search_memories_tool: '搜索记忆',
  retrieve_chunks_tool: '检索片段',
  write_memory_tool: '写入记忆',
  extract_memory_tool: '提取记忆',
  ingest_git_repo_tool: '摄取 Git 仓库',
  ingest_document_tool: '摄取文档',
};

/**
 * Return a human-readable one-line summary of `content` for `toolName`.
 *
 * Mirrors Streamlit's `_format_tool_result` exactly (see frontend/app.py),
 * including the English labels emitted there. Falls back to a truncated
 * snippet when the format is unrecognised.
 */
export function formatToolResult(toolName: string, content: string): string {
  // ── search_memories_tool / retrieve_chunks_tool ──
  // JSON envelope: {"display": "...", "sources": [...]} → use the "display"
  // field for the human-readable summary. Falls back to plain text for
  // legacy/empty results.
  if (toolName === 'search_memories_tool' || toolName === 'retrieve_chunks_tool') {
    let display = content;
    try {
      const data: unknown = JSON.parse(content);
      if (
        typeof data === 'object' &&
        data !== null &&
        'display' in data &&
        typeof (data as Record<string, unknown>).display === 'string'
      ) {
        display = (data as Record<string, unknown>).display as string;
      }
    } catch {
      // keep raw content
    }
    const m = display.match(/Found (\d+) relevant (memories|chunks)/);
    if (m) {
      const count = parseInt(m[1], 10);
      const kind = m[2];
      if (count === 0) {
        return `🔍 No relevant ${kind} found.`;
      }
      const items = display.match(/\[\d+\]\s(.+)/g) ?? [];
      const first = items[0]?.slice(0, 80) ?? '';
      return `🔍 Found **${count}** ${kind}${first ? ` — _${first}…_` : ''}`;
    }
    return display.slice(0, 120);
  }

  // ── write_memory_tool ──
  // JSON: {"id":..., "action":"inserted|merged|conflict", "summary":"..."}
  if (toolName === 'write_memory_tool') {
    let data: Record<string, unknown>;
    try {
      data = JSON.parse(content) as Record<string, unknown>;
    } catch {
      return content.slice(0, 120);
    }
    const action = typeof data.action === 'string' ? data.action : '?';
    const summary = typeof data.summary === 'string' ? data.summary.slice(0, 120) : '';
    const labels: Record<string, string> = {
      inserted: '📝 New memory created',
      merged: '🔗 Merged into existing memory',
      conflict: '⚠️ Conflict detected — needs resolution',
    };
    const label = labels[action] ?? `📝 Memory ${action}`;
    return `${label}${summary ? `: _${summary}_` : ''}`;
  }

  // ── extract_memory_tool ──
  // JSON: {"summary":"...", "entities":[...], "relations":[...]}
  if (toolName === 'extract_memory_tool') {
    let data: Record<string, unknown>;
    try {
      data = JSON.parse(content) as Record<string, unknown>;
    } catch {
      return content.slice(0, 120);
    }
    const nEntities = Array.isArray(data.entities) ? data.entities.length : 0;
    const nRelations = Array.isArray(data.relations) ? data.relations.length : 0;
    const summary = typeof data.summary === 'string' ? data.summary.slice(0, 80) : '';
    return (
      `🧠 Extracted **${nEntities}** entities, **${nRelations}** relations` +
      (summary ? ` — _${summary}…_` : '')
    );
  }

  // ── ingest_git_repo_tool ──
  // "Ingested N commits as memories:" or "No commits were ingested …"
  if (toolName === 'ingest_git_repo_tool') {
    const m = content.match(/Ingested (\d+) commits?/);
    if (m) {
      return `📥 Ingested **${m[1]}** commits from Git repo`;
    }
    return content.slice(0, 120);
  }

  // ── ingest_document_tool ──
  // "Ingested N chunks from document 'X'."
  if (toolName === 'ingest_document_tool') {
    const m = content.match(/Ingested (\d+) chunks? from document '(.+?)'/);
    if (m) {
      return `📥 Ingested **${m[1]}** chunks from \`${m[2]}\``;
    }
    return content.slice(0, 120);
  }

  // ── Fallback ──
  return content.slice(0, 200);
}

/**
 * Renders a compact list of tool-call results, each with a formatted summary
 * and a collapsible raw view (mirrors Streamlit's "🔧 Tool calls" expander).
 */
export default function ToolCallPanel({ toolCalls }: ToolCallPanelProps) {
  if (toolCalls.length === 0) return null;

  return (
    <div className="w-full rounded-xl border border-gray-200 bg-white">
      <div className="rounded-t-xl border-b border-gray-200 px-3 py-2 text-xs font-semibold text-gray-600">
        🔧 工具调用
      </div>
      <ul className="divide-y divide-gray-100">
        {toolCalls.map((tc, i) => {
          const tool = tc.tool ?? 'unknown';
          const label = TOOL_LABELS[tool] ?? tool;
          return (
            <li key={i} className="px-3 py-2">
              <div className="mb-1 flex items-center gap-1.5">
                <span className="text-xs font-medium text-gray-500">{label}</span>
              </div>
              <p className="text-sm leading-relaxed text-gray-800">
                <RichText text={formatToolResult(tool, tc.content ?? '')} />
              </p>
              <details className="mt-1">
                <summary className="cursor-pointer select-none text-xs text-gray-400 hover:text-gray-600">
                  查看原始内容
                </summary>
                <pre className="mt-1 overflow-x-auto whitespace-pre-wrap rounded bg-gray-50 p-2 text-xs text-gray-600">
                  {`${tool}\n${(tc.content ?? '').slice(0, 300)}`}
                </pre>
              </details>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
