import type { Interrupt } from '../types';

interface ApprovalCardProps {
  interrupt: Interrupt;
  onResume: (resumeData: Record<string, unknown>) => void;
  isResolving?: boolean;
}

/** Tool name → human-readable Chinese label. */
const TOOL_LABELS: Record<string, string> = {
  write_memory_tool: '写入记忆',
  ingest_git_repo_tool: '摄取 Git 仓库',
  ingest_document_tool: '摄取文档',
};

/**
 * Approval card shown when the agent wants to perform a write/ingest
 * operation. Mirrors Streamlit's `_render_tool_approval`.
 */
export default function ApprovalCard({
  interrupt,
  onResume,
  isResolving = false,
}: ApprovalCardProps) {
  const toolName = interrupt.tool_name ?? 'unknown';
  const args = interrupt.tool_args ?? {};
  const summary = interrupt.summary ?? JSON.stringify(args).slice(0, 200);
  const label = TOOL_LABELS[toolName] ?? toolName;

  return (
    <div className="rounded-xl border border-amber-300 bg-amber-50 p-4">
      <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-amber-900">
        <span aria-hidden>🛡️</span>
        <span>待批准：{label}</span>
      </div>
      <p className="mb-4 text-sm leading-relaxed text-amber-800">{summary}</p>
      <div className="flex gap-2">
        <button
          type="button"
          disabled={isResolving}
          onClick={() => onResume({ approved: true })}
          className="rounded-lg bg-green-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-green-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          ✅ 批准
        </button>
        <button
          type="button"
          disabled={isResolving}
          onClick={() => onResume({ approved: false, reason: '用户拒绝了工具调用。' })}
          className="rounded-lg bg-red-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          ❌ 拒绝
        </button>
      </div>
    </div>
  );
}
