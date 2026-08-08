import type { Interrupt } from '../types';

interface ConflictCardProps {
  interrupt: Interrupt;
  onResume: (resumeData: Record<string, unknown>) => void;
  isResolving?: boolean;
  /** Which pipeline queued this conflict — changes the wording, not the options. */
  variant?: 'ingestion' | 'patrol';
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
 *
 * `variant="patrol"` re-words the card for patrol contradictions: both
 * memories are already stored, so the panels are labelled "记忆 B" (the
 * losing side, peer_id) and "记忆 A" (the surviving side, existing_id)
 * instead of "新记忆 / 现有记忆".
 */
export default function ConflictCard({
  interrupt,
  onResume,
  isResolving = false,
  variant = 'ingestion',
}: ConflictCardProps) {
  const newSummary = interrupt.new_summary ?? '';
  const existingSummary = interrupt.existing_summary ?? '';
  const isPatrol = variant === 'patrol';

  return (
    <div className="rounded-xl border border-red-300 bg-red-50 p-4">
      <div className="mb-2 flex items-center gap-2 text-sm font-semibold text-red-900">
        <span aria-hidden>⚖️</span>
        <span>{isPatrol ? '检测到记忆矛盾（巡检发现）' : '检测到记忆冲突'}</span>
      </div>
      <p className="mb-3 text-sm text-red-800">
        {isPatrol
          ? '巡检发现两条已有记忆相互矛盾，应如何解决？'
          : '新记忆与现有记忆冲突，应如何解决？'}
      </p>

      <div className="mb-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
        <div className="rounded-lg bg-white p-3 ring-1 ring-inset ring-red-100">
          <p className="mb-1 text-xs font-semibold text-gray-500">
            {isPatrol ? '记忆 B' : '新记忆'}
          </p>
          <p className="text-sm leading-relaxed text-gray-800">{newSummary || '(空)'}</p>
        </div>
        <div className="rounded-lg bg-white p-3 ring-1 ring-inset ring-red-100">
          <p className="mb-1 text-xs font-semibold text-gray-500">
            {isPatrol ? '记忆 A' : '现有记忆'}
          </p>
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
