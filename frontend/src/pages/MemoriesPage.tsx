import { useEffect, useState } from 'react';
import { ApiError } from '../api/client';
import IngestSection from '../components/IngestSection';
import MemoryCard from '../components/MemoryCard';
import MemorySearch from '../components/MemorySearch';
import StatsDashboard from '../components/StatsDashboard';
import { useAppDispatch, useAppState } from '../context/AppContext';
import { useMemories } from '../hooks/useMemories';
import type { MemoryGetResponse } from '../types';

type Tab = 'dashboard' | 'ingest' | 'search';

const TABS: { key: Tab; label: string; icon: string }[] = [
  { key: 'dashboard', label: '仪表盘', icon: '📊' },
  { key: 'ingest', label: '摄入文档', icon: '📥' },
  { key: 'search', label: '搜索记忆', icon: '🔍' },
];

export default function MemoriesPage() {
  const { stats, isLoading, error, fetchStats, getMemoryById } = useMemories();
  const { memFilterId } = useAppState();
  const dispatch = useAppDispatch();

  const [activeTab, setActiveTab] = useState<Tab>('dashboard');

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

    // Switch to search tab so the jumped memory is visible.
    setActiveTab('search');

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

        {/* Tab bar */}
        <nav className="mb-6 flex gap-1 border-b border-gray-200">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              type="button"
              onClick={() => setActiveTab(tab.key)}
              className={`rounded-t-lg border-b-2 px-4 py-2.5 text-sm font-medium transition-colors ${
                activeTab === tab.key
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              <span className="mr-1.5">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </nav>

        {/* Chat jump filter — visible across all tabs */}
        {memFilterId && (
          <div className="mb-6 rounded-lg border border-blue-200 bg-blue-50 p-4">
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

        {/* Tab content */}
        <section>
          {activeTab === 'dashboard' && (
            <StatsDashboard stats={stats} isLoading={isLoading} error={error} onRetry={fetchStats} />
          )}
          {activeTab === 'ingest' && <IngestSection onIngest={fetchStats} />}
          {activeTab === 'search' && <MemorySearch />}
        </section>
      </div>
    </div>
  );
}
