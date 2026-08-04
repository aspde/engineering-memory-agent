import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppDispatch, useAppState } from '../context/AppContext';
import { deleteThread, listThreads } from '../api/agent';

const THREADS_CACHE_TTL_MS = 30_000;

export default function Sidebar() {
  const navigate = useNavigate();
  const { threadId, threads, threadsFetchedAt } = useAppState();
  const dispatch = useAppDispatch();

  // Track which thread (if any) is in the delete-confirm state.
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);

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
    dispatch({ type: 'NEW_CONVERSATION', threadId: crypto.randomUUID() });
    navigate('/');
  };

  const handleOpenThread = (id: string) => {
    dispatch({ type: 'SET_THREAD_ID', threadId: id });
    navigate('/');
  };

  const handleDelete = async (id: string) => {
    setIsDeleting(true);
    setConfirmingId(null);
    setDeleteError(null);
    try {
      await deleteThread(id);
      dispatch({ type: 'REMOVE_THREAD', threadId: id });
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : '删除失败，请重试');
    } finally {
      setIsDeleting(false);
    }
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
          className="w-full rounded-lg bg-blue-600 px-3 py-2 text-left text-sm font-medium text-white transition-colors hover:bg-blue-700"
        >
          ＋ 新建对话
        </button>
        <button
          type="button"
          onClick={() => navigate('/memories')}
          className="w-full rounded-lg bg-violet-600 px-3 py-2 text-left text-sm font-medium text-white transition-colors hover:bg-violet-700"
        >
          📚 记忆库
        </button>
        <button
          type="button"
          onClick={() => navigate('/graph')}
          className="w-full rounded-lg bg-emerald-600 px-3 py-2 text-left text-sm font-medium text-white transition-colors hover:bg-emerald-700"
        >
          🔗 实体图谱
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
            const isConfirming = confirmingId === t.thread_id;

            return (
              <div key={t.thread_id} className="group relative">
                <button
                  type="button"
                  onClick={() => handleOpenThread(t.thread_id)}
                  disabled={active || isDeleting}
                  className={`flex w-full items-center gap-2 rounded-lg px-3 py-2 pr-8 text-left text-sm transition-colors ${
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

                {/* Delete button — visible on hover */}
                <div className="absolute right-1 top-1/2 -translate-y-1/2">
                  {!isConfirming ? (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setConfirmingId(t.thread_id);
                      }}
                      disabled={isDeleting}
                      className="hidden rounded p-1 text-gray-400 hover:bg-red-50 hover:text-red-500 transition-colors group-hover:block"
                      title="删除对话"
                    >
                      <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  ) : (
                    <span className="flex items-center gap-1 rounded bg-red-50 px-2 py-1 text-xs text-red-700">
                      删除?
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          void handleDelete(t.thread_id);
                        }}
                        disabled={isDeleting}
                        className="rounded px-1 py-0.5 font-medium bg-red-500 text-white hover:bg-red-600 disabled:opacity-50"
                      >
                        是
                      </button>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setConfirmingId(null);
                        }}
                        disabled={isDeleting}
                        className="rounded px-1 py-0.5 font-medium text-gray-600 hover:bg-gray-200"
                      >
                        否
                      </button>
                    </span>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>

      {/* Delete error feedback */}
      {deleteError && (
        <div className="mx-3 mb-3 rounded-lg bg-red-50 px-3 py-2 text-xs text-red-700">
          {deleteError}
          <button
            type="button"
            onClick={() => setDeleteError(null)}
            className="ml-2 font-medium underline hover:no-underline"
          >
            关闭
          </button>
        </div>
      )}
    </aside>
  );
}
