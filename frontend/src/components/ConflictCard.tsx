import type { Interrupt } from '../types';

interface ConflictCardProps {
  interrupt: Interrupt;
  onResume: (resumeData: Record<string, unknown>) => void;
  isResolving?: boolean;
}

const OPTIONS: { key: string; label: string }[] = [
  { key: 'keep_existing', label: '📌 保留现有' },
  { key: 'overwrite', label: '✏️ 覆盖' },
  { key: 'merge', label: '🔀 合并' },
  { key: 'keep_both', label: '📋 两者都保留' },
];

/**
 * Conflict-resolution card shown when a new memory contradicts an existing
 * one. Mirrors Streamlit's `_render_conflict_resolution`.
 */
export default function ConflictCard({
  interrupt,
  onResume,
  isResolving = false,
}: ConflictCardProps) {
  const newSummary = interrupt.new_summary ?? '';
  const existingSummary = interrupt.existing_summary ?? '';

  return (
    <div className="rounded-xl border border-red-300 bg-red-50 p-4">
      <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-red-900">
        <span aria-hidden>⚖️</span>
        <span>检测到记忆冲突</span>
      </div>
      <p className="mb-3 text-sm text-red-800">新记忆与现有记忆冲突，应如何解决？</p>

      <div className="mb-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
        <div className="rounded-lg bg-white p-3 ring-1 ring-inset ring-red-100">
          <p className="mb-1 text-xs font-semibold text-gray-500">新记忆</p>
          <p className="text-sm leading-relaxed text-gray-800">{newSummary || '(空)'}</p>
        </div>
        <div className="rounded-lg bg-white p-3 ring-1 ring-inset ring-red-100">
          <p className="mb-1 text-xs font-semibold text-gray-500">现有记忆</p>
          <p className="text-sm leading-relaxed text-gray-800">{existingSummary || '(空)'}</p>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        {OPTIONS.map((opt) => (
          <button
            key={opt.key}
            type="button"
            disabled={isResolving}
            onClick={() => onResume({ resolution: opt.key })}
            className="rounded-lg bg-white px-3 py-2 text-sm font-medium text-gray-700 ring-1 ring-inset ring-gray-300 transition-colors hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );
}
