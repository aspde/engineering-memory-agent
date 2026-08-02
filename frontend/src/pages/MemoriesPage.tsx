import { useEffect, useState } from 'react';
import { ApiError } from '../api/client';
import IngestSection from '../components/IngestSection';
import MemoryCard from '../components/MemoryCard';
import MemorySearch from '../components/MemorySearch';
import StatsDashboard from '../components/StatsDashboard';
import { useAppDispatch, useAppState } from '../context/AppContext';
import { useMemories } from '../hooks/useMemories';
import type { MemoryGetResponse } from '../types';

export default function MemoriesPage() {
  const { stats, isLoading, error, fetchStats, getMemoryById } = useMemories();
  const { memFilterId } = useAppState();
  const dispatch = useAppDispatch();

  // Chat → memories jump filter state.
  const [jumpMemory, setJumpMemory] = useState<MemoryGetResponse | null>(null);
  const [jumpLoading, setJumpLoading] = useState(false);
  const [jumpNotFound, setJumpNotFound] = useState(false);
  const [jumpError, setJumpError] = useState<string | null>(null);

  useEffect(() => {
    if (!memFilterId) return;
    let cancelled = false;
    setJumpLoading(true);
    setJumpNotFound(false);
    setJumpError(null);
    setJumpMemory(null);

    getMemoryById(memFilterId)
      .then((mem) => {
        if (!cancelled) setJumpMemory(mem);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        if (err instanceof ApiError && err.status === 404) {
          setJumpNotFound(true);
        } else {
          setJumpError(err instanceof Error ? err.message : String(err));
        }
      })
      .finally(() => {
        if (!cancelled) setJumpLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [memFilterId, getMemoryById]);

  const handleClearFilter = () => {
    dispatch({ type: 'CLEAR_MEM_FILTER' });
  };

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-4xl px-4 py-6 sm:px-6">
        <h1 className="mb-6 text-2xl font-bold text-gray-900">📚 记忆库</h1>

        {/* Stats dashboard */}
        <StatsDashboard stats={stats} isLoading={isLoading} error={error} onRetry={fetchStats} />

        {/* Chat jump filter */}
        {memFilterId && (
          <div className="mt-6 rounded-lg border border-blue-200 bg-blue-50 p-4">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
              <p className="text-sm text-blue-800">
                🔗 从聊天跳转 — 正在查找记忆{' '}
                <span className="font-mono">{memFilterId.slice(0, 8)}…</span>
              </p>
              <button
                type="button"
                onClick={handleClearFilter}
                className="rounded-lg border border-blue-300 bg-white px-3 py-1.5 text-sm font-medium text-blue-700 transition-colors hover:bg-blue-100"
              >
                清除筛选
              </button>
            </div>

            {jumpLoading ? (
              <p className="text-sm text-blue-700">正在定位记忆…</p>
            ) : jumpMemory ? (
              <MemoryCard memory={jumpMemory as unknown as Record<string, unknown>} />
            ) : jumpNotFound ? (
              <p className="text-sm text-amber-700">未找到该记忆，可能已被删除或尚未建立索引。</p>
            ) : jumpError ? (
              <p className="text-sm text-red-600">查找记忆失败: {jumpError}</p>
            ) : null}
          </div>
        )}

        <hr className="my-8 border-gray-200" />

        {/* Ingest section (collapsible, default collapsed) */}
        <details className="rounded-lg border border-gray-200 bg-white">
          <summary className="cursor-pointer select-none rounded-lg px-4 py-3 text-sm font-medium text-gray-800 transition-colors hover:bg-gray-50">
            📥 摄入文档
          </summary>
          <div className="border-t border-gray-200 p-4">
            <IngestSection onIngest={fetchStats} />
          </div>
        </details>

        <hr className="my-8 border-gray-200" />

        {/* Memory search */}
        <section>
          <h2 className="mb-3 text-sm font-semibold text-gray-900">搜索记忆</h2>
          <MemorySearch />
        </section>
      </div>
    </div>
  );
}
