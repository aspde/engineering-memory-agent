import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

// ── Source badge styling ─────────────────────────────────────────────

const SOURCE_STYLES: Record<string, { icon: string; colors: string }> = {
  pingcode: { icon: '📋', colors: 'bg-blue-100 text-blue-700' },
  pingcode_bug: { icon: '🐛', colors: 'bg-blue-100 text-blue-700' },
  ci_build: { icon: '🔄', colors: 'bg-green-100 text-green-700' },
  ci_regression: { icon: '📉', colors: 'bg-green-100 text-green-700' },
  feishu: { icon: '💬', colors: 'bg-purple-100 text-purple-700' },
  git_commit: { icon: '📦', colors: 'bg-gray-100 text-gray-700' },
  conversation: { icon: '💭', colors: 'bg-indigo-100 text-indigo-700' },
  api: { icon: '📄', colors: 'bg-cyan-100 text-cyan-700' },
};

function sourceStyle(sourceType: string): { icon: string; colors: string } {
  return SOURCE_STYLES[sourceType] ?? { icon: '📌', colors: 'bg-gray-100 text-gray-600' };
}

interface MemoryCardProps {
  /** Raw memory record from searchMemories (or a full MemoryGetResponse). */
  memory: Record<string, unknown>;
  /** Called when the user confirms deletion of this memory. */
  onDelete?: (id: string) => void;
  /** Disables the delete controls while a delete request is in flight. */
  isDeleting?: boolean;
}

export default function MemoryCard({ memory, onDelete, isDeleting }: MemoryCardProps) {
  const navigate = useNavigate();
  const [showConfirm, setShowConfirm] = useState(false);
  const summary =
    typeof memory.summary === 'string' && memory.summary.length > 0
      ? memory.summary
      : '(无摘要)';
  const sourceType = typeof memory.source_type === 'string' ? memory.source_type : 'unknown';
  const decay = typeof memory.decay_factor === 'number' ? memory.decay_factor : 1;
  const createdAt = typeof memory.created_at === 'string' ? memory.created_at : '';
  const rawId = memory.id;
  const id = rawId != null ? String(rawId) : '';

  // Extract thread_id from meta if this memory came from a conversation
  const meta = memory.meta as Record<string, unknown> | undefined;
  const threadId = typeof meta?.thread_id === 'string' ? meta.thread_id : null;

  const isLong = summary.length > 120;
  const shortSummary = isLong ? `${summary.slice(0, 120)}…` : summary;

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      {/* Header: summary + source badge */}
      <div className="flex items-start justify-between gap-3">
        <p className="font-semibold text-gray-900">{shortSummary}</p>
        <span
          className={`shrink-0 rounded-full px-2.5 py-0.5 text-xs font-medium ${sourceStyle(sourceType).colors}`}
        >
          {sourceStyle(sourceType).icon} {sourceType}
        </span>
        {onDelete && (
          <div className="ml-2 shrink-0">
            {!showConfirm ? (
              <button
                type="button"
                onClick={() => setShowConfirm(true)}
                disabled={isDeleting}
                className="rounded p-1 text-gray-400 hover:bg-red-50 hover:text-red-500 transition-colors"
                title="删除此记忆"
              >
                <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            ) : (
              <div className="flex items-center gap-1 rounded bg-red-50 px-2 py-1">
                <span className="text-xs text-red-700">确认删除?</span>
                <button
                  type="button"
                  onClick={() => { setShowConfirm(false); onDelete(id); }}
                  disabled={isDeleting}
                  className="rounded px-1.5 py-0.5 text-xs font-medium bg-red-500 text-white hover:bg-red-600 disabled:opacity-50"
                >
                  删除
                </button>
                <button
                  type="button"
                  onClick={() => setShowConfirm(false)}
                  disabled={isDeleting}
                  className="rounded px-1.5 py-0.5 text-xs font-medium text-gray-600 hover:bg-gray-200"
                >
                  取消
                </button>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Metrics row */}
      <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
        <div>
          <p className="text-gray-400">Decay</p>
          <p className="font-medium text-gray-700">{decay.toFixed(2)}</p>
        </div>
        <div>
          <p className="text-gray-400">创建时间</p>
          <p className="font-medium text-gray-700">{createdAt.slice(0, 19) || '—'}</p>
        </div>
        <div>
          <p className="text-gray-400">ID</p>
          <p className="font-mono text-gray-700">{id ? `${id.slice(0, 8)}…` : '—'}</p>
        </div>
      </div>

      {/* Full summary expander */}
      {isLong && (
        <details className="mt-3">
          <summary className="cursor-pointer text-xs font-medium text-blue-600 hover:underline">
            完整摘要
          </summary>
          <p className="mt-2 whitespace-pre-wrap text-sm text-gray-700">{summary}</p>
        </details>
      )}
    </div>
  );
}
