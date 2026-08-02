interface MemoryCardProps {
  /** Raw memory record from searchMemories (or a full MemoryGetResponse). */
  memory: Record<string, unknown>;
}

export default function MemoryCard({ memory }: MemoryCardProps) {
  const summary =
    typeof memory.summary === 'string' && memory.summary.length > 0
      ? memory.summary
      : '(无摘要)';
  const sourceType = typeof memory.source_type === 'string' ? memory.source_type : 'unknown';
  const decay = typeof memory.decay_factor === 'number' ? memory.decay_factor : 1;
  const createdAt = typeof memory.created_at === 'string' ? memory.created_at : '';
  const rawId = memory.id;
  const id = rawId != null ? String(rawId) : '';

  const isLong = summary.length > 120;
  const shortSummary = isLong ? `${summary.slice(0, 120)}…` : summary;

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      {/* Header: summary + source badge */}
      <div className="flex items-start justify-between gap-3">
        <p className="font-semibold text-gray-900">{shortSummary}</p>
        <span className="shrink-0 rounded bg-gray-100 px-2 py-0.5 font-mono text-xs text-gray-600">
          {sourceType}
        </span>
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
