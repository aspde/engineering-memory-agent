import type { Message } from '../types';
import CitationText from './CitationText';
import RichText from './RichText';
import SourcesPanel from './SourcesPanel';

interface MessageBubbleProps {
  message: Message;
  /** True for the in-flight assistant message while tokens are streaming. */
  isStreaming?: boolean;
}

/**
 * Renders a single chat message:
 * - user      → right-aligned blue bubble
 * - assistant → left-aligned grey bubble + optional tool-call / source panels
 * - system    → centred amber notice (interrupt / status notes)
 * - error     → left-aligned red bubble, visually distinct from replies
 */
export default function MessageBubble({ message, isStreaming = false }: MessageBubbleProps) {
  if (message.kind === 'error') {
    // A failed run (SSE error event or network failure). Rendered as a
    // red, bordered, italic bubble so it can't be mistaken for an answer.
    return (
      <div className="flex justify-start px-4 py-2">
        <div className="flex max-w-[85%] items-start gap-2.5">
          <div
            className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-full bg-red-100 text-sm"
            aria-hidden
          >
            ⚠️
          </div>
          <div
            role="alert"
            className="rounded-2xl rounded-bl-sm border border-red-300 bg-red-50 px-4 py-2.5 text-sm leading-relaxed italic text-red-800"
          >
            <p className="whitespace-pre-wrap">
              <RichText text={message.content} />
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (message.role === 'system') {
    return (
      <div className="flex justify-center px-4 py-2">
        <div className="flex max-w-[85%] items-start gap-2 rounded-lg bg-amber-50 px-4 py-2.5 text-sm leading-relaxed text-amber-800">
          <span aria-hidden>⚠️</span>
          <p className="whitespace-pre-wrap">
            <RichText text={message.content} />
          </p>
        </div>
      </div>
    );
  }

  const isUser = message.role === 'user';
  const showTyping =
    !isUser && isStreaming && message.content.length === 0 && !message._meta;

  return (
    <div className={`flex px-4 py-2 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`flex max-w-[85%] flex-col gap-1.5 ${isUser ? 'items-end' : 'items-start'}`}>
        <div className={`flex items-start gap-2.5 ${isUser ? 'flex-row-reverse' : ''}`}>
          <div className="flex h-8 w-8 shrink-0 select-none items-center justify-center rounded-full bg-gray-100 text-sm">
            {isUser ? '🧑' : '🤖'}
          </div>
          <div
            className={`rounded-2xl px-4 py-2.5 text-sm leading-relaxed ${
              isUser
                ? 'rounded-br-sm bg-blue-600 text-white'
                : 'rounded-bl-sm bg-gray-100 text-gray-900'
            }`}
          >
            {showTyping ? (
              <span className="inline-flex items-center gap-1 py-1">
                <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400" />
                <span
                  className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400"
                  style={{ animationDelay: '0.15s' }}
                />
                <span
                  className="h-1.5 w-1.5 animate-bounce rounded-full bg-gray-400"
                  style={{ animationDelay: '0.3s' }}
                />
              </span>
            ) : isUser ? (
              <RichText text={message.content} />
            ) : (
              <CitationText
                text={message.content}
                sources={message._meta?.sources ?? []}
              />
            )}
          </div>
        </div>

        {!isUser && message._meta && (message._meta.toolCalls.length > 0 || message._meta.sources.length > 0) && (
          <div className="w-full space-y-2 pl-10">
            {message._meta.sources.length > 0 && (
              <SourcesPanel sources={message._meta.sources} />
            )}
          </div>
        )}
      </div>
    </div>
  );
}
