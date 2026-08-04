import { useCallback, useEffect, useRef, useState } from 'react';
import { useAppDispatch, useAppState } from '../context/AppContext';
import { getThreadMessages } from '../api/agent';
import { writeMemory } from '../api/memory';
import { runScenario } from '../api/scenarios';
import { useChat } from '../hooks/useChat';
import ChatArea from '../components/ChatArea';
import ChatInput from '../components/ChatInput';

/**
 * Chat page: lazy-loads message history for the active thread, then renders
 * the scrollable message area plus a pinned chat input.
 *
 * When "强制写入记忆" is checked, the user's message is also sent directly
 * to the memory write API, bypassing the LLM's tool-calling judgement.
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

  // Auto-trigger scenario via API when a new conversation is created with an active scenario.
  useEffect(() => {
    if (!activeScenario) return;
    if (triggeredRef.current === threadId) return;
    if (loadedThreadId !== threadId) return;
    if (isLoading || messages.length > 0) return;

    triggeredRef.current = threadId;
    const scenarioKey = activeScenario;
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

    runScenario(scenarioKey)
      .then((res) => {
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
      })
      .catch((err) => {
        dispatch({
          type: 'UPDATE_LAST_MESSAGE',
          appendContent: `\n\n场景执行失败: ${err instanceof Error ? err.message : String(err)}`,
        });
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
        }
      });

    return () => {
      cancelled = true;
    };
  }, [threadId, loadedThreadId, dispatch]);

  const handleSend = useCallback(
    (text: string, forceWrite: boolean) => {
      // Always send the chat message (Agent will respond normally)
      sendMessage(text);

      // When force-write is checked, also persist via the direct API
      if (forceWrite) {
        writeMemory(text, 'conversation', { thread_id: threadId })
          .then((res) => {
            const labels: Record<string, string> = {
              inserted: '已写入新记忆',
              merged: '已合并到已有记忆',
              conflict: '检测到冲突，请在记忆库中处理',
            };
            setWriteToast(labels[res.action] ?? `记忆${res.action}`);
          })
          .catch(() => {
            setWriteToast('记忆写入失败');
          });
      }
    },
    [sendMessage],
  );

  const inputDisabled = isLoading || isStreaming || waitingForApproval;
  const placeholder = isLoading
    ? '加载中…'
    : waitingForApproval
      ? '等待批准…'
      : isStreaming
        ? '回复生成中…'
        : '向 EMA 提问…';

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
