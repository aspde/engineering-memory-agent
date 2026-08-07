import { useCallback, useEffect, useState } from 'react';
import { getConflicts, resolveConflict } from '../api/conflicts';
import ConflictCard from '../components/ConflictCard';
import type { PendingConflict } from '../types';

/**
 * Human-in-the-loop surface for webhook/connector conflicts.
 *
 * Webhook deliveries that contradict an existing memory land in the pending
 * queue; each is rendered with the same ConflictCard (and same four options)
 * the agent interrupt path uses — the resolution just hits the REST API
 * instead of resuming an agent thread.
 */
export default function ConflictsPage() {
  const [conflicts, setConflicts] = useState<PendingConflict[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [resolvingId, setResolvingId] = useState<string | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    getConflicts()
      .then(setConflicts)
      .catch((err) =>
        setError(err instanceof Error ? err.message : '加载失败，请重试'),
      )
      .finally(() => setLoading(false));
  }, []);

  useEffect(load, [load]);

  const handleResolve = async (
    id: string,
    resumeData: Record<string, unknown>,
  ) => {
    const resolution = String(resumeData.resolution ?? 'keep_existing');
    setResolvingId(id);
    setError(null);
    try {
      await resolveConflict(id, resolution);
      setConflicts((prev) => prev.filter((c) => c.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : '解决失败，请重试');
    } finally {
      setResolvingId(null);
    }
  };

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center p-6 text-sm text-gray-500">
        加载中…
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="mb-4 flex items-center justify-between">
        <h1 className="text-lg font-semibold text-gray-900">待处理冲突</h1>
        <button
          type="button"
          onClick={load}
          className="rounded-md px-2 py-1 text-xs text-gray-500 hover:bg-gray-100"
        >
          ↻ 刷新
        </button>
      </div>

      {error && (
        <p className="mb-4 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </p>
      )}

      {conflicts.length === 0 ? (
        <p className="rounded-xl border border-gray-200 bg-white p-6 text-sm text-gray-500">
          暂无待处理的记忆冲突 🎉
        </p>
      ) : (
        <ul className="space-y-4">
          {conflicts.map((c) => (
            <li
              key={c.id}
              className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm"
            >
              <div className="mb-2 text-xs text-gray-400">
                来源：{c.source}
                {c.source_type ? ` / ${c.source_type}` : ''}
                {c.created_at ? ` · ${c.created_at}` : ''}
              </div>
              <ConflictCard
                interrupt={{
                  type: 'conflict',
                  new_summary: c.new_summary,
                  existing_summary: c.existing_summary,
                }}
                isResolving={resolvingId === c.id}
                onResume={(resumeData) => void handleResolve(c.id, resumeData)}
              />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
