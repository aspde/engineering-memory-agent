import { useEffect, useRef, useState, type KeyboardEvent } from 'react';

interface ChatInputProps {
  onSend: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
}

/**
 * Chat input box: Enter to send, Shift+Enter for a newline.
 * Disabled while the agent is busy (streaming / awaiting approval).
 */
export default function ChatInput({
  onSend,
  disabled = false,
  placeholder = '向 EMA 提问…',
}: ChatInputProps) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-grow the textarea up to a sensible max height.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [value]);

  const submit = () => {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue('');
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter sends; Shift+Enter inserts a newline. Ignore while IME composing
    // (Chinese input confirmation) to avoid sending mid-composition.
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="border-t border-gray-200 bg-white px-4 py-3">
      <div className="mx-auto flex max-w-3xl items-end gap-2">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          rows={1}
          placeholder={placeholder}
          aria-label="聊天输入"
          className="max-h-40 min-h-[2.5rem] flex-1 resize-none rounded-xl border border-gray-300 px-3 py-2 text-sm text-gray-900 placeholder-gray-400 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-500"
        />
        <button
          type="button"
          onClick={submit}
          disabled={disabled || value.trim().length === 0}
          className="shrink-0 rounded-xl bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-gray-300"
        >
          发送
        </button>
      </div>
      <p className="mx-auto mt-1 max-w-3xl text-right text-xs text-gray-400">
        Enter 发送 · Shift+Enter 换行
      </p>
    </div>
  );
}
