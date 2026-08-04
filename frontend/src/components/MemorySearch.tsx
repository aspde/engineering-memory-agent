import { useMemo, useRef, useState } from 'react';
import { useMemories } from '../hooks/useMemories';
import MemoryCard from './MemoryCard';

const TOP_K_OPTIONS = [5, 10, 20, 30, 50];

const SOURCE_FILTER_LABELS: Record<string, string> = {
  conversation: '💭 对话',
  api: '📄 API',
  git_commit: '📦 Git',
  pingcode: '📋 PingCode',
  pingcode_bug: '🐛 PingCode 缺陷',
  ci_build: '🔄 CI',
  ci_regression: '📉 CI 回归',
  feishu: '💬 飞书',
};

interface ToastState {
  type: 'success' | 'error';
  message: string;
}

const TOAST_DURATION_MS = 4000;

export default function MemorySearch() {
  const { searchResults, isSearching, searchError, search, deleteMemoryById, removeSearchResult, fetchStats } =
    useMemories();
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(20);
  const [sourceFilter, setSourceFilter] = useState<string>('');
  const [prompt, setPrompt] = useState<string | null>(null);
  const [toast, setToast] = useState<ToastState | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  // Collect distinct source_types from search results for the filter dropdown.
  const sourceTypes = useMemo(() => {
    if (!searchResults) return [];
    const seen = new Set<string>();
    for (const r of searchResults) {
      const st = typeof r.source_type === 'string' ? r.source_type : '';
      if (st) seen.add(st);
    }
    return Array.from(seen).sort();
  }, [searchResults]);

  // Client-side source_type filtering.
  const filteredResults = useMemo(() => {
    if (!searchResults) return null;
    if (!sourceFilter) return searchResults;
    return searchResults.filter(
      (r) => String(r.source_type ?? '') === sourceFilter,
    );
  }, [searchResults, sourceFilter]);

  const toastTimerRef = useRef<number | null>(null);

  const showToast = (t: ToastState) => {
    setToast(t);
    if (toastTimerRef.current !== null) {
      window.clearTimeout(toastTimerRef.current);
    }
    toastTimerRef.current = window.setTimeout(() => {
      setToast(null);
      toastTimerRef.current = null;
    }, TOAST_DURATION_MS);
  };

  const handleSearch = () => {
    const q = query.trim();
    if (!q) {
      setPrompt('请输入搜索关键词。');
      return;
    }
    setPrompt(null);
    void search(q, topK);
  };

  const handleDelete = async (id: string) => {
    setDeletingId(id);
    try {
      await deleteMemoryById(id);
      removeSearchResult(id);
      fetchStats(); // 刷新仪表盘统计
      showToast({ type: 'success', message: '✅ 已删除记忆' });
    } catch (err) {
      showToast({
        type: 'error',
        message: `删除失败: ${err instanceof Error ? err.message : String(err)}`,
      });
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div>
      {/* Search bar */}
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') handleSearch();
          }}
          placeholder="输入关键词搜索记忆库"
          className="flex-1 rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-900 placeholder:text-gray-400 focus:border-blue-500 focus:outline-none"
        />
        <div className="flex shrink-0 items-center gap-2">
          {sourceTypes.length > 0 && (
            <select
              value={sourceFilter}
              onChange={(e) => setSourceFilter(e.target.value)}
              className="rounded-lg border border-gray-300 bg-white px-2 py-2 text-sm text-gray-700 focus:border-blue-500 focus:outline-none"
            >
              <option value="">全部来源</option>
              {sourceTypes.map((st) => (
                <option key={st} value={st}>
                  {SOURCE_FILTER_LABELS[st] ?? st}
                </option>
              ))}
            </select>
          )}
          <select
            value={topK}
            onChange={(e) => setTopK(Number(e.target.value))}
            className="rounded-lg border border-gray-300 bg-white px-2 py-2 text-sm text-gray-700 focus:border-blue-500 focus:outline-none"
          >
            {TOP_K_OPTIONS.map((n) => (
              <option key={n} value={n}>
                {n} 条
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={handleSearch}
            disabled={isSearching}
            className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            🔍 搜索
          </button>
        </div>
      </div>

      {/* Results / status */}
      <div className="mt-4">
        {searchError ? (
          <p className="text-sm text-red-600">搜索失败：{searchError}</p>
        ) : isSearching ? (
          <div className="flex items-center gap-2 py-2 text-sm text-gray-500">
            <span className="h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-blue-600" />
            搜索中…
          </div>
        ) : filteredResults !== null ? (
          filteredResults.length === 0 ? (
            <p className="rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-700">
              没有找到匹配的记忆。尝试换个关键词，或者在聊天中让 EMA 记录一些内容。
            </p>
          ) : (
            <div className="space-y-3">
              <p className="text-sm text-gray-500">
                找到 {filteredResults.length} 条记忆
                {sourceFilter && searchResults && searchResults.length > filteredResults.length && (
                  <span className="text-gray-400">（共 {searchResults.length} 条，已按来源筛选）</span>
                )}
              </p>
              {filteredResults.map((mem, idx) => (
                <MemoryCard
                  key={idx}
                  memory={mem}
                  onDelete={handleDelete}
                  isDeleting={deletingId === String(mem.id)}
                />
              ))}
            </div>
          )
        ) : prompt ? (
          <p className="rounded-lg bg-blue-50 px-3 py-2 text-sm text-blue-700">{prompt}</p>
        ) : null}
      </div>

      {/* Toast */}
      {toast && (
        <div
          className={`fixed bottom-4 right-4 z-50 rounded-lg px-4 py-3 text-sm font-medium text-white shadow-lg ${
            toast.type === 'success' ? 'bg-green-600' : 'bg-red-600'
          }`}
        >
          {toast.message}
        </div>
      )}
    </div>
  );
}
