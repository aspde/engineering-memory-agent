import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppDispatch, useAppState } from '../context/AppContext';
import { listThreads } from '../api/agent';

const THREADS_CACHE_TTL_MS = 30_000;

export default function Sidebar() {
  const navigate = useNavigate();
  const { threadId, threads, threadsFetchedAt, isStreaming } = useAppState();
  const dispatch = useAppDispatch();

  useEffect(() => {
    // Skip if we already have a fresh thread list within the cache TTL.
    if (Date.now() - threadsFetchedAt < THREADS_CACHE_TTL_MS) return;

    let cancelled = false;
    listThreads()
      .then((data) => {
        if (!cancelled) {
          dispatch({ type: 'SET_THREADS', threads: data });
        }
      })
      .catch(() => {
        // On error, record an (empty) fetch so we show the empty state
        // rather than a perpetual loading skeleton, and respect the TTL.
        if (!cancelled) {
          dispatch({ type: 'SET_THREADS', threads: [] });
        }
      });

    return () => {
      cancelled = true;
    };
  }, [dispatch, threadsFetchedAt]);

  const isLoading = threadsFetchedAt === 0;

  const handleNewConversation = () => {
    if (isStreaming) return;
    dispatch({ type: 'NEW_CONVERSATION', threadId: crypto.randomUUID() });
    navigate('/');
  };

  const handleOpenThread = (id: string) => {
    if (isStreaming) return;
    dispatch({ type: 'SET_THREAD_ID', threadId: id });
    navigate('/');
  };

  return (
    <aside className="hidden md:flex w-64 shrink-0 h-full flex-col bg-gray-50 border-r border-gray-200">
      {/* Brand */}
      <div className="px-4 pt-5 pb-3">
        <h1 className="text-xl font-bold text-gray-900">🧠 EMA</h1>
        <p className="mt-0.5 text-xs text-gray-500">Engineering Memory Agent</p>
      </div>

      {/* Actions */}
      <div className="space-y-1.5 px-3">
        <button
          type="button"
          onClick={handleNewConversation}
          disabled={isStreaming}
          className="w-full rounded-lg bg-blue-600 px-3 py-2 text-left text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          ＋ 新建对话
        </button>
        <button
          type="button"
          onClick={() => navigate('/memories')}
          className="w-full rounded-lg px-3 py-2 text-left text-sm font-medium text-gray-700 transition-colors hover:bg-gray-200"
        >
          📚 记忆库
        </button>
      </div>

      {/* Thread list */}
      <div className="px-4 pb-2 pt-5 text-xs font-semibold uppercase tracking-wider text-gray-400">
        对话历史
      </div>
      <div className="flex-1 space-y-1 overflow-y-auto px-2 pb-4">
        {isLoading ? (
          Array.from({ length: 4 }, (_, i) => (
            <div key={i} className="animate-pulse rounded-lg px-3 py-2.5">
              <div className="mb-2 h-3 w-3/4 rounded bg-gray-200" />
              <div className="h-2 w-1/2 rounded bg-gray-200" />
            </div>
          ))
        ) : threads.length === 0 ? (
          <p className="px-3 py-2 text-sm text-gray-400">暂无历史对话</p>
        ) : (
          threads.map((t) => {
            const active = t.thread_id === threadId;
            return (
              <button
                key={t.thread_id}
                type="button"
                onClick={() => handleOpenThread(t.thread_id)}
                disabled={isStreaming || active}
                className={`flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm transition-colors ${
                  active
                    ? 'bg-blue-50 font-medium text-blue-700'
                    : 'text-gray-700 hover:bg-gray-100 disabled:opacity-50'
                }`}
              >
                <span
                  className={`h-2 w-2 shrink-0 rounded-full ${active ? 'bg-blue-500' : 'bg-transparent'}`}
                />
                <span className="truncate">{t.title || '未命名对话'}</span>
              </button>
            );
          })
        )}
      </div>
    </aside>
  );
}
