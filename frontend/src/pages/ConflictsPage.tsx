import { useCallback, useEffect, useState } from 'react';
import { getConflicts, reopenConflict, resolveConflict } from '../api/conflicts';
import ConflictCard from '../components/ConflictCard';
import type { PendingConflict } from '../types';

type ViewTab = 'pending' | 'arbitrated';
type TypeFilter = '' | 'ingestion' | 'patrol';

/**
 * Human-in-the-loop surface for memory conflicts.
 *
 * Two pipelines feed this page: webhook/connector ingestion conflicts and
 * patrol contradictions (both already-stored memories).  Pending rows render
 * the ConflictCard with four options; the "已仲裁" tab lists resolved patrol
 * conflicts so a mistaken keep_both can be reopened for re-arbitration.
 */
export default function ConflictsPage() {
  const [conflicts, setConflicts] = useState<PendingConflict[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [resolvingId, setResolvingId] = useState<string | null>(null);
  const [reopeningId, setReopeningId] = useState<string | null>(null);
  const [viewTab, setViewTab] = useState<ViewTab>('pending');
  const [typeFilter, setTypeFilter] = useState<TypeFilter>('');

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    const params =
      viewTab === 'arbitrated'
        ? { status: 'resolved', conflict_type: 'patrol' }
        : { status: 'pending', ...(typeFilter ? { conflict_type: typeFilter } : {}) };
    getConflicts(params)
      .then(setConflicts)
      .catch((err) =>
        setError(err instanceof Error ? err.message : '加载失败，请重试'),
      )
      .finally(() => setLoading(false));
  }, [viewTab, typeFilter]);

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

  const handleReopen = async (id: string) => {
    setReopeningId(id);
    setError(null);
    try {
      await reopenConflict(id);
      await load();
    } catch (err) {
      setError(err instanceof Error ? err.message : '重新打开失败，请重试');
    } finally {
      setReopeningId(null);
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
        <h1 className="text-lg font-semibold text-gray-900">冲突处理</h1>
        <button
          type="button"
          onClick={load}
          className="rounded-md px-2 py-1 text-xs text-gray-500 hover:bg-gray-100"
        >
          ↻ 刷新
        </button>
      </div>

      <div className="mb-3 flex items-center gap-1.5 text-xs">
        <button
          type="button"
          onClick={() => setViewTab('pending')}
          className={`rounded px-2.5 py-1 font-medium transition-colors ${
            viewTab === 'pending'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          待处理
        </button>
        <button
          type="button"
          onClick={() => setViewTab('arbitrated')}
          className={`rounded px-2.5 py-1 font-medium transition-colors ${
            viewTab === 'arbitrated'
              ? 'bg-blue-600 text-white'
              : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
          }`}
        >
          已仲裁（巡检）
        </button>
      </div>

      {viewTab === 'pending' && (
        <div className="mb-3 flex items-center gap-1.5 text-xs">
          {(
            [
              { key: '', label: '全部' },
              { key: 'ingestion', label: '写入冲突' },
              { key: 'patrol', label: '巡检矛盾' },
            ] as { key: TypeFilter; label: string }[]
          ).map(({ key, label }) => (
            <button
              key={key}
              type="button"
              onClick={() => setTypeFilter(key)}
              className={`rounded px-2.5 py-1 font-medium transition-colors ${
                typeFilter === key
                  ? 'bg-orange-500 text-white'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      )}

      {error && (
        <p className="mb-4 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </p>
      )}

      {conflicts.length === 0 ? (
        <p className="rounded-xl border border-gray-200 bg-white p-6 text-sm text-gray-500">
          {viewTab === 'pending'
            ? '暂无待处理的记忆冲突 🎉'
            : '暂无已仲裁的巡检矛盾'}
        </p>
      ) : viewTab === 'pending' ? (
        <ul className="space-y-4">
          {conflicts.map((c) => (
            <li
              key={c.id}
              className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm"
            >
              <div className="mb-2 flex items-center gap-2 text-xs text-gray-400">
                {c.conflict_type === 'patrol' ? (
                  <span className="rounded-full bg-orange-100 px-2 py-0.5 text-orange-600">
                    巡检矛盾
                  </span>
                ) : null}
                <span>
                  来源：{c.source}
                  {c.source_type ? ` / ${c.source_type}` : ''}
                  {c.created_at ? ` · ${c.created_at}` : ''}
                </span>
              </div>
              <ConflictCard
                interrupt={{
                  type: 'conflict',
                  new_summary: c.new_summary,
                  existing_summary: c.existing_summary,
                }}
                variant={c.conflict_type === 'patrol' ? 'patrol' : 'ingestion'}
                isResolving={resolvingId === c.id}
                onResume={(resumeData) => void handleResolve(c.id, resumeData)}
              />
            </li>
          ))}
        </ul>
      ) : (
        <ul className="space-y-4">
          {conflicts.map((c) => (
            <li
              key={c.id}
              className="rounded-xl border border-gray-200 bg-white p-4 shadow-sm"
            >
              <div className="mb-2 flex items-center gap-2 text-xs text-gray-400">
                <span className="rounded-full bg-orange-100 px-2 py-0.5 text-orange-600">
                  巡检矛盾
                </span>
                <span>
                  来源：{c.source}
                  {c.created_at ? ` · ${c.created_at}` : ''}
                  {c.resolution ? ` · 已仲裁：${c.resolution}` : ''}
                </span>
              </div>
              <div className="mb-3 grid grid-cols-1 gap-2 sm:grid-cols-2">
                <div className="rounded-lg bg-gray-50 p-3 ring-1 ring-inset ring-gray-200">
                  <p className="mb-1 text-xs font-semibold text-gray-500">记忆 B</p>
                  <p className="text-sm leading-relaxed text-gray-800">
                    {c.new_summary || '(空)'}
                  </p>
                </div>
                <div className="rounded-lg bg-gray-50 p-3 ring-1 ring-inset ring-gray-200">
                  <p className="mb-1 text-xs font-semibold text-gray-500">记忆 A</p>
                  <p className="text-sm leading-relaxed text-gray-800">
                    {c.existing_summary || '(空)'}
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={() => void handleReopen(c.id)}
                disabled={reopeningId === c.id}
                className="rounded-md bg-white px-3 py-1.5 text-xs font-medium text-gray-700 ring-1 ring-inset ring-gray-300 transition-colors hover:bg-amber-50 hover:text-amber-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {reopeningId === c.id ? '重新打开中…' : '↩ 重新打开仲裁'}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
