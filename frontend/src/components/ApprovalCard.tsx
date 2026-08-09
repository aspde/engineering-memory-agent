import type { ApprovalCall, Interrupt } from '../types';

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
 *
 * The backend sends two payload shapes:
 *   - single tool  → flat `{ tool_name, tool_args, summary? }`
 *   - multiple     → `{ type: 'batch', calls: [{ tool_name, tool_args, summary? }, ...] }`
 *
 * A batch is rendered one row per tool (name + arg summary) with per-tool
 * approve/reject buttons, and resumes with
 * `{ calls: [{ id, tool_name, approved, reason }, ...] }`.  The backend
 * executes exactly the approved subset (by tool_call ``id``) and injects a
 * ``[REJECTED]`` ToolMessage for each row the user did not approve — the
 * buttons are honest: "✅ 批准" on a row runs that write and only that write.
 */
function callSummary(call: ApprovalCall): string {
  if (call.summary) return call.summary;
  return JSON.stringify(call.tool_args ?? {}).slice(0, 200);
}

export default function ApprovalCard({
  interrupt,
  onResume,
  isResolving = false,
}: ApprovalCardProps) {
  const isBatch = interrupt.type === 'batch' && Array.isArray(interrupt.calls);
  const calls: ApprovalCall[] = isBatch ? interrupt.calls! : [];

  if (isBatch) {
    return (
      <div className="rounded-xl border border-amber-300 bg-amber-50 p-4">
        <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-amber-900">
          <span aria-hidden>🛡️</span>
          <span>待批准：{calls.length} 个工具操作</span>
        </div>
        <ul className="mb-4 flex flex-col gap-2">
          {calls.map((call, idx) => {
            const label = TOOL_LABELS[call.tool_name] ?? call.tool_name;
            return (
              <li
                key={call.id ?? `${call.tool_name}-${idx}`}
                className="rounded-lg bg-white p-3 ring-1 ring-inset ring-amber-200"
              >
                <p className="mb-1 text-sm font-semibold text-amber-900">{label}</p>
                <p className="mb-2 text-sm leading-relaxed text-amber-800">
                  {callSummary(call)}
                </p>
                <div className="flex gap-2">
                  <button
                    type="button"
                    disabled={isResolving}
                    onClick={() =>
                      onResume({
                        calls: calls.map((c, i) => ({
                          id: c.id,
                          tool_name: c.tool_name,
                          approved: i === idx,
                          reason: i === idx ? undefined : '用户拒绝了工具调用。',
                        })),
                      })
                    }
                    className="rounded-lg bg-green-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-green-700 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    ✅ 批准
                  </button>
                  <button
                    type="button"
                    disabled={isResolving}
                    onClick={() =>
                      onResume({
                        calls: calls.map((c, i) => ({
                          id: c.id,
                          tool_name: c.tool_name,
                          approved: i !== idx,
                          reason: i === idx ? '用户拒绝了工具调用。' : undefined,
                        })),
                      })
                    }
                    className="rounded-lg bg-red-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    ❌ 拒绝
                  </button>
                </div>
              </li>
            );
          })}
        </ul>
        <div className="flex gap-2 border-t border-amber-200 pt-3">
          <button
            type="button"
            disabled={isResolving}
            onClick={() =>
              onResume({
                calls: calls.map((c) => ({ id: c.id, tool_name: c.tool_name, approved: true })),
              })
            }
            className="rounded-lg bg-green-700 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-green-800 disabled:cursor-not-allowed disabled:opacity-50"
          >
            ✅ 全部批准
          </button>
          <button
            type="button"
            disabled={isResolving}
            onClick={() =>
              onResume({
                calls: calls.map((c) => ({ id: c.id, tool_name: c.tool_name, approved: false })),
                reason: '用户拒绝了工具调用。',
              })
            }
            className="rounded-lg bg-red-700 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-red-800 disabled:cursor-not-allowed disabled:opacity-50"
          >
            ❌ 全部拒绝
          </button>
        </div>
      </div>
    );
  }

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
