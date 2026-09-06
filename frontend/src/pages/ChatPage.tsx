import { useCallback, useEffect, useRef, useState } from 'react';
import { useAppDispatch, useAppState } from '../context/AppContext';
import { getThreadMessages } from '../api/agent';
import { runScenario, saveRunAsMemory } from '../api/scenarios';
import { useChat } from '../hooks/useChat';
import ChatArea from '../components/ChatArea';
import ChatInput from '../components/ChatInput';

/**
 * Chat page: lazy-loads message history for the active thread, then renders
 * the scrollable message area plus a pinned chat input.
 *
 * When "记住这条" is checked, the send carries ``force_write`` so the
 * server injects a write_memory_tool call for this turn — the memory is
 * written inside the agent flow (LLM extraction + on-the-spot conflict
 * handling), and the write's outcome surfaces as a toast via the meta event.
 */
export default function ChatPage() {
  const { threadId, loadedThreadId, messages, pendingInterrupt, waitingForApproval, activeScenario } =
    useAppState();
  const dispatch = useAppDispatch();
  const { sendMessage, resume, isStreaming } = useChat();
  const [isLoading, setIsLoading] = useState(false);
  const [writeToast, setWriteToast] = useState<string | null>(null);

  // Track which thread has already been auto-triggered for a scenario.
  const triggeredRef = useRef<string | null>(null);
  // Map threadId → scenario key for retry support.
  const scenarioForThreadRef = useRef<Record<string, { key: string; label: string }>>({});
  const [retryingScenario, setRetryingScenario] = useState(false);
  const [retryMessage, setRetryMessage] = useState<string | null>(null);

  // Completed postmortem run awaiting the user's save-as-memory decision.
  // Only offered while its thread is open; a reload drops the affordance
  // (the draft stays readable in history) rather than persisting UI state.
  const [postmortemRunId, setPostmortemRunId] = useState<string | null>(null);
  const [savingRun, setSavingRun] = useState(false);

  const SCENARIO_LABELS: Record<string, string> = {
    postmortem: '故障复盘',
    code_review: '代码审查助手',
    onboarding: '新人 Onboarding',
    tech_debt: '技术债雷达',
  };

  // Auto-trigger scenario via API when a new conversation is created with an active scenario.
  useEffect(() => {
    if (!activeScenario) return;
    if (triggeredRef.current === threadId) return;
    if (loadedThreadId !== threadId) return;
    if (isLoading || messages.length > 0) return;

    triggeredRef.current = threadId;
    const scenarioKey = activeScenario;
    const triggeredForThreadId = threadId;
    // Save mapping so we can offer retry if the user returns later
    scenarioForThreadRef.current[triggeredForThreadId] = {
      key: scenarioKey,
      label: SCENARIO_LABELS[scenarioKey] ?? scenarioKey,
    };
    dispatch({ type: 'CLEAR_ACTIVE_SCENARIO' });

    // Add a placeholder that will be replaced when the API returns
    dispatch({
      type: 'ADD_MESSAGE',
      message: { role: 'user', content: `触发场景: ${scenarioKey}` },
    });
    dispatch({
      type: 'ADD_MESSAGE',
      message: { role: 'assistant', content: '正在执行场景…' },
    });

    runScenario(scenarioKey, {}, triggeredForThreadId)
      .then((res) => {
        // Guard: if the user switched threads while waiting, discard the result
        if (triggeredRef.current !== triggeredForThreadId) return;
        // Replace the placeholder with the scenario result
        dispatch({
          type: 'UPDATE_LAST_MESSAGE',
          appendContent: '',
        });
        dispatch({
          type: 'ADD_MESSAGE',
          message: {
            role: 'assistant',
            content: res.result || '(场景返回为空)',
          },
        });
        // Sync sidebar in case the title was updated on the backend
        dispatch({ type: 'INVALIDATE_THREADS' });
        if (scenarioKey === 'postmortem' && res.run_id) {
          setPostmortemRunId(res.run_id);
        }
      })
      .catch((err) => {
        if (triggeredRef.current !== triggeredForThreadId) return;
        dispatch({
          type: 'UPDATE_LAST_MESSAGE',
          appendContent: `\n\n场景执行失败: ${err instanceof Error ? err.message : String(err)}`,
        });
        dispatch({ type: 'INVALIDATE_THREADS' });
      });
  }, [activeScenario, threadId, loadedThreadId, isLoading, messages.length, dispatch]);

  // Auto-dismiss toast after 2.5 s
  useEffect(() => {
    if (!writeToast) return;
    const t = setTimeout(() => setWriteToast(null), 2500);
    return () => clearTimeout(t);
  }, [writeToast]);

  // Lazy-load message history whenever the active thread changes and its
  // messages aren't already in state (e.g. thread switch, page refresh).
  useEffect(() => {
    if (loadedThreadId === threadId) return;

    let cancelled = false;
    setIsLoading(true);

    getThreadMessages(threadId)
      .then((res) => {
        if (cancelled) return;
        dispatch({
          type: 'SET_MESSAGES',
          messages: res.messages.map((m) => {
            const toolCalls = m.tool_calls ?? [];
            const sources = m.sources ?? [];
            const hasMeta = toolCalls.length > 0 || sources.length > 0;
            return {
              role: m.role,
              content: m.content,
              _meta: hasMeta ? { toolCalls, sources } : undefined,
            };
          }),
        });
      })
      .catch(() => {
        // New/empty threads return 404 — fall back to an empty conversation.
        if (!cancelled) {
          dispatch({ type: 'SET_MESSAGES', messages: [] });
        }
      })
      .finally(() => {
        if (!cancelled) {
          dispatch({ type: 'SET_LOADED_THREAD', threadId });
          setIsLoading(false);
          // Reset the scenario trigger ref so stale .then() callbacks
          // from a previous scenario run won't add duplicate messages
          // on top of the checkpoint state we just loaded.
          triggeredRef.current = null;
        }
      });

    return () => {
      cancelled = true;
    };
  }, [threadId, loadedThreadId, dispatch]);

  const handleSend = useCallback(
    (text: string, forceWrite: boolean) => {
      // Send the chat message with the force-write flag; the write runs inside
      // the agent flow (server-injected write_memory_tool call).  sendMessage
      // resolves the force-write outcome from the meta event so the toast
      // reflects what actually happened — including the distilled summary the
      // LLM extracted, so the user can verify what was stored.
      sendMessage(text, forceWrite).then((memoryWrite) => {
        if (!memoryWrite) return;
        const summary = (memoryWrite.summary || '').slice(0, 60);
        if (memoryWrite.action === 'inserted') {
          setWriteToast(summary ? `已写入：${summary}` : '已写入新记忆');
        } else if (memoryWrite.action === 'merged') {
          setWriteToast(summary ? `已合并：${summary}` : '已合并到已有记忆');
        } else if (memoryWrite.action === 'conflict') {
          setWriteToast('检测到冲突，请在记忆库中处理');
        } else {
          setWriteToast(`记忆${memoryWrite.action}`);
        }
      });
    },
    [sendMessage],
  );

  const inputDisabled = isLoading || isStreaming || waitingForApproval || retryingScenario;
  const placeholder = isLoading
    ? '加载中…'
    : waitingForApproval
      ? '等待批准…'
      : isStreaming
        ? '回复生成中…'
        : retryingScenario
          ? '重新执行场景中…'
          : '向 EMA 提问…';

  // Check if the current thread is a scenario that may need retry
  const scenarioInfo = scenarioForThreadRef.current[threadId];
  const lastMsg = messages.length > 0 ? messages[messages.length - 1] : null;
  const lastIsError = lastMsg?.role === 'assistant' && (
    lastMsg.content.includes('场景执行失败') || lastMsg.content.includes('错误')
  );
  const showRetry = !isLoading && scenarioInfo && (messages.length === 0 || lastIsError);

  const handleRetryScenario = useCallback(async () => {
    if (!scenarioInfo || retryingScenario) return;
    setRetryingScenario(true);
    setRetryMessage(null);

    dispatch({
      type: 'ADD_MESSAGE',
      message: { role: 'user', content: `触发场景: ${scenarioInfo.key}` },
    });
    dispatch({
      type: 'ADD_MESSAGE',
      message: { role: 'assistant', content: '正在执行场景…' },
    });

    try {
      const res = await runScenario(scenarioInfo.key, {}, threadId);
      dispatch({ type: 'UPDATE_LAST_MESSAGE', appendContent: '' });
      dispatch({
        type: 'ADD_MESSAGE',
        message: { role: 'assistant', content: res.result || '(场景返回为空)' },
      });
      dispatch({ type: 'INVALIDATE_THREADS' });
      if (scenarioInfo.key === 'postmortem' && res.run_id) {
        setPostmortemRunId(res.run_id);
      }
    } catch (err) {
      dispatch({
        type: 'UPDATE_LAST_MESSAGE',
        appendContent: `\n\n场景执行失败: ${err instanceof Error ? err.message : String(err)}`,
      });
      setRetryMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setRetryingScenario(false);
    }
  }, [scenarioInfo, retryingScenario, dispatch, threadId]);

  const handleSavePostmortem = useCallback(async () => {
    if (!postmortemRunId || savingRun) return;
    setSavingRun(true);
    try {
      const res = await saveRunAsMemory(postmortemRunId);
      if (res.action === 'inserted') {
        setWriteToast(res.summary ? `复盘已写入：${res.summary.slice(0, 60)}` : '复盘已写入记忆库');
      } else if (res.action === 'merged') {
        setWriteToast('复盘已合并到相似记忆');
      } else if (res.action === 'conflict') {
        setWriteToast('检测到相似记忆冲突，请在记忆库页面仲裁');
      } else {
        setWriteToast('该复盘此前已保存过');
      }
    } catch (err) {
      setWriteToast(`保存失败: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSavingRun(false);
      // One deliberate save closes the affordance — a second click would
      // only ever return the duplicate probe.
      setPostmortemRunId(null);
    }
  }, [postmortemRunId, savingRun]);

  return (
    <div className="flex h-full flex-col">
      <ChatArea
        messages={messages}
        isLoading={isLoading}
        isStreaming={isStreaming}
        pendingInterrupt={pendingInterrupt}
        waitingForApproval={waitingForApproval}
        onResume={resume}
      />
      {/* Save-as-memory affordance for a completed postmortem draft */}
      {postmortemRunId && (
        <div className="mx-auto mb-2 max-w-3xl px-4">
          <div className="flex items-center justify-between gap-3 rounded-lg border border-blue-200 bg-blue-50 px-4 py-3">
            <p className="text-sm text-blue-800">复盘草稿已生成，可保存为长期记忆供日后检索。</p>
            <button
              type="button"
              onClick={handleSavePostmortem}
              disabled={savingRun}
              className="shrink-0 rounded bg-blue-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-50"
            >
              {savingRun ? '保存中…' : '💾 保存为记忆'}
            </button>
          </div>
        </div>
      )}
      {/* Retry prompt for incomplete / failed scenario threads */}
      {showRetry && (
        <div className="mx-auto mb-2 max-w-3xl px-4">
          <div className="rounded-lg border border-orange-200 bg-orange-50 px-4 py-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-sm font-medium text-orange-800">
                  {scenarioInfo.label} — {messages.length === 0 ? '场景尚未执行或已被中断' : '上次执行失败'}
                </p>
                {retryMessage && (
                  <p className="mt-0.5 text-xs text-orange-600">{retryMessage}</p>
                )}
              </div>
              <button
                type="button"
                onClick={handleRetryScenario}
                disabled={retryingScenario}
                className="shrink-0 rounded bg-orange-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-orange-700 disabled:opacity-50"
              >
                {retryingScenario ? '执行中…' : '🔄 重试'}
              </button>
            </div>
          </div>
        </div>
      )}

      <ChatInput onSend={handleSend} disabled={inputDisabled} placeholder={placeholder} />

      {/* Toast notification for force-write result */}
      {writeToast && (
        <div className="fixed bottom-20 left-1/2 z-50 -translate-x-1/2 rounded-full bg-gray-800 px-4 py-2 text-sm text-white shadow-lg">
          {writeToast}
        </div>
      )}
    </div>
  );
}
