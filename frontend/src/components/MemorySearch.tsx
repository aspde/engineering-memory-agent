import { useState } from 'react';
import { useMemories } from '../hooks/useMemories';
import MemoryCard from './MemoryCard';

const TOP_K_OPTIONS = [5, 10, 20, 30, 50];

export default function MemorySearch() {
  const { searchResults, isSearching, searchError, search } = useMemories();
  const [query, setQuery] = useState('');
  const [topK, setTopK] = useState(20);
  const [prompt, setPrompt] = useState<string | null>(null);

  const handleSearch = () => {
    const q = query.trim();
    if (!q) {
      setPrompt('请输入搜索关键词。');
      return;
    }
    setPrompt(null);
    void search(q, topK);
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
        ) : searchResults !== null ? (
          searchResults.length === 0 ? (
            <p className="rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-700">
              没有找到匹配的记忆。尝试换个关键词，或者在聊天中让 EMA 记录一些内容。
            </p>
          ) : (
            <div className="space-y-3">
              <p className="text-sm text-gray-500">找到 {searchResults.length} 条记忆</p>
              {searchResults.map((mem, idx) => (
                <MemoryCard key={idx} memory={mem} />
              ))}
            </div>
          )
        ) : prompt ? (
          <p className="rounded-lg bg-blue-50 px-3 py-2 text-sm text-blue-700">{prompt}</p>
        ) : null}
      </div>
    </div>
  );
}
