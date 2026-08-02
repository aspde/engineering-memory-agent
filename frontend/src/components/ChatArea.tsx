import { useEffect, useRef } from 'react';
import type { Interrupt, Message } from '../types';
import MessageBubble from './MessageBubble';
import ApprovalCard from './ApprovalCard';
import ConflictCard from './ConflictCard';

/** Max messages rendered at once (older ones are hidden with a note). */
const MAX_VISIBLE = 50;

interface ChatAreaProps {
  messages: Message[];
  isLoading: boolean;
  isStreaming: boolean;
  pendingInterrupt: Interrupt | null;
  waitingForApproval: boolean;
  onResume: (resumeData: Record<string, unknown>) => void;
}

/**
 * Scrollable chat message list with an empty state and the approval /
 * conflict card pinned to the bottom while awaiting the user's decision.
 */
export default function ChatArea({
  messages,
  isLoading,
  isStreaming,
  pendingInterrupt,
  waitingForApproval,
  onResume,
}: ChatAreaProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({
      behavior: isStreaming ? 'auto' : 'smooth',
      block: 'end',
    });
  }, [messages, isStreaming]);

  if (isLoading) {
    return (
      <div className="flex flex-1 items-center justify-center px-4">
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-blue-600" />
          加载中…
        </div>
      </div>
    );
  }

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center px-4">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900">EMA — Engineering Memory Agent</h1>
          <p className="mt-2 text-sm text-gray-500">向智能体提问，探索你的代码记忆</p>
        </div>
      </div>
    );
  }

  const hiddenCount = Math.max(0, messages.length - MAX_VISIBLE);
  const visible = hiddenCount > 0 ? messages.slice(-MAX_VISIBLE) : messages;

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="mx-auto max-w-3xl px-4 py-4">
        {hiddenCount > 0 && (
          <p className="pb-2 text-center text-xs text-gray-400">
            ... 已隐藏 {hiddenCount} 条更早的消息
          </p>
        )}

        {visible.map((msg, i) => {
          const isLast = i === visible.length - 1;
          return (
            <MessageBubble
              key={i}
              message={msg}
              isStreaming={isStreaming && isLast}
            />
          );
        })}

        {waitingForApproval && pendingInterrupt && (
          <div className="py-2">
            {pendingInterrupt.type === 'conflict' ? (
              <ConflictCard
                interrupt={pendingInterrupt}
                onResume={onResume}
                isResolving={isStreaming}
              />
            ) : (
              <ApprovalCard
                interrupt={pendingInterrupt}
                onResume={onResume}
                isResolving={isStreaming}
              />
            )}
          </div>
        )}

        <div ref={bottomRef} />
      </div>
    </div>
  );
}
