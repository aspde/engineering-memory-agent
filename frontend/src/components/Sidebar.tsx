import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppDispatch, useAppState } from '../context/AppContext';
import { deleteThread, listThreads } from '../api/agent';
import { listScenarios } from '../api/scenarios';
import type { ScenarioInfo } from '../types';

const THREADS_CACHE_TTL_MS = 30_000;

const SCENARIO_ICONS: Record<string, string> = {
  postmortem: '🔎',
  code_review: '👀',
  onboarding: '🎓',
  tech_debt: '📋',
};

export default function Sidebar() {
  const navigate = useNavigate();
  const { threadId, threads, threadsFetchedAt } = useAppState();
  const dispatch = useAppDispatch();

  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [scenarios, setScenarios] = useState<ScenarioInfo[]>([]);
  const [scenariosOpen, setScenariosOpen] = useState(false);

  useEffect(() => {
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

  useEffect(() => {
    let cancelled = false;
    listScenarios()
      .then((data) => {
        if (!cancelled) setScenarios(data);
      })
      .catch(() => {
        if (!cancelled) setScenarios([]);
      });
    return () => { cancelled = true; };
  }, []);

  const isLoading = threadsFetchedAt === 0;

  const handleScenarioClick = (key: string) => {
    const newThreadId = crypto.randomUUID();
    const info = scenarios.find((s) => s.key === key);
    dispatch({ type: 'NEW_CONVERSATION', threadId: newThreadId });
    dispatch({ type: 'SET_ACTIVE_SCENARIO', scenario: key });
    dispatch({
      type: 'PREPEND_THREAD',
      threadId: newThreadId,
      title: info?.name ?? key,
    });
    navigate('/');
  };

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
      <div className="px-3 pt-4 pb-2">
        <h1 className="text-lg font-bold text-gray-900">🧠 EMA</h1>
      </div>

      {/* Primary action */}
      <div className="px-3 pb-2">
        <button
          type="button"
          onClick={handleNewConversation}
          className="w-full rounded-lg bg-blue-600 px-3 py-2 text-center text-sm font-medium text-white transition-colors hover:bg-blue-700"
        >
          ＋ 新建对话
        </button>
      </div>

      {/* Scenarios — collapsible */}
      {scenarios.length > 0 && (
        <div className="px-3 pb-2">
          <button
            type="button"
            onClick={() => setScenariosOpen((v) => !v)}
            className="flex w-full items-center gap-1 rounded-md px-1 py-1 text-xs font-semibold text-gray-500 hover:bg-gray-100 hover:text-gray-700 transition-colors"
          >
            <span className="text-[10px] transition-transform" style={{ transform: scenariosOpen ? 'rotate(90deg)' : undefined }}>
              ▶
            </span>
            场景
          </button>
          {scenariosOpen && (
            <div className="mt-1 space-y-0.5">
              {scenarios.map((s) => (
                <button
                  key={s.key}
                  type="button"
                  onClick={() => handleScenarioClick(s.key)}
                  className="w-full rounded-md px-2 py-1 text-left text-xs text-gray-600 hover:bg-gray-100 hover:text-gray-900 transition-colors"
                  title={s.description}
                >
                  <span className="mr-1.5">{SCENARIO_ICONS[s.key] ?? '📌'}</span>
                  {s.name}
                  {s.status === 'beta' && (
                    <span className="ml-1 rounded bg-amber-100 px-1 py-0.5 text-[10px] text-amber-700">beta</span>
                  )}
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Thread list */}
      <div className="px-3 pb-1.5 pt-3 text-sm font-semibold text-gray-400">
        历史对话
      </div>
      <div className="flex-1 space-y-0.5 overflow-y-auto px-2 pb-4">
        {isLoading ? (
          Array.from({ length: 4 }, (_, i) => (
            <div key={i} className="animate-pulse rounded-lg px-3 py-2">
              <div className="mb-1.5 h-3 w-3/4 rounded bg-gray-200" />
              <div className="h-2 w-1/2 rounded bg-gray-200" />
            </div>
          ))
        ) : threads.length === 0 ? (
          <p className="px-3 py-2 text-sm text-gray-400">暂无</p>
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
                  className={`flex w-full items-center gap-2 rounded-lg px-2 py-1.5 pr-7 text-left text-sm transition-colors ${
                    active
                      ? 'bg-blue-50 font-medium text-blue-700'
                      : 'text-gray-700 hover:bg-gray-100 disabled:opacity-50'
                  }`}
                >
                  <span
                    className={`h-1.5 w-1.5 shrink-0 rounded-full ${active ? 'bg-blue-500' : 'bg-transparent'}`}
                  />
                  <span className="truncate">{t.title || '未命名对话'}</span>
                </button>

                {/* Delete button — visible on hover */}
                <div className="absolute right-0.5 top-1/2 -translate-y-1/2">
                  {!isConfirming ? (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setConfirmingId(t.thread_id);
                      }}
                      disabled={isDeleting}
                      className="hidden rounded p-0.5 text-gray-400 hover:bg-red-50 hover:text-red-500 transition-colors group-hover:block"
                      title="删除对话"
                    >
                      <svg className="h-3.5 w-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  ) : (
                    <span className="flex items-center gap-0.5 rounded bg-red-50 px-1.5 py-0.5 text-[11px] text-red-700">
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
        <div className="mx-2 mb-2 rounded-lg bg-red-50 px-2 py-1.5 text-xs text-red-700">
          {deleteError}
          <button
            type="button"
            onClick={() => setDeleteError(null)}
            className="ml-1.5 font-medium underline hover:no-underline"
          >
            关闭
          </button>
        </div>
      )}
    </aside>
  );
}
