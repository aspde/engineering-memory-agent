import { useEffect, useState } from 'react';
import { useAppDispatch, useAppState } from '../context/AppContext';
import { getThreadMessages } from '../api/agent';
import { useChat } from '../hooks/useChat';
import ChatArea from '../components/ChatArea';
import ChatInput from '../components/ChatInput';

/**
 * Chat page: lazy-loads message history for the active thread, then renders
 * the scrollable message area plus a pinned chat input.
 */
export default function ChatPage() {
  const { threadId, loadedThreadId, messages, pendingInterrupt, waitingForApproval } =
    useAppState();
  const dispatch = useAppDispatch();
  const { sendMessage, resume, isStreaming } = useChat();
  const [isLoading, setIsLoading] = useState(false);

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
      <ChatInput onSend={sendMessage} disabled={inputDisabled} placeholder={placeholder} />
    </div>
  );
}
