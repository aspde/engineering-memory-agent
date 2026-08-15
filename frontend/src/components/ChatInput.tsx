import { useEffect, useRef, useState, type KeyboardEvent } from 'react';

interface ChatInputProps {
  onSend: (text: string, forceWrite: boolean) => void;
  disabled?: boolean;
  placeholder?: string;
}

/**
 * Chat input box: Enter to send, Shift+Enter for a newline.
 * Disabled while the agent is busy (streaming / awaiting approval).
 * Optional "记住这条" checkbox writes this message into the memory store
 * (server-side LLM extraction + on-the-spot conflict handling) regardless of
 * the model's own judgement.  It is a single-action toggle — it resets after
 * each send, so only the current message is affected, never future ones.
 */
export default function ChatInput({
  onSend,
  disabled = false,
  placeholder = '向 EMA 提问…',
}: ChatInputProps) {
  const [value, setValue] = useState('');
  const [forceWrite, setForceWrite] = useState(false);
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
    onSend(text, forceWrite);
    setValue('');
    // Single-action toggle: force-write applies to THIS message only, so it
    // resets after sending instead of staying sticky for every later message.
    setForceWrite(false);
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
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
      <div className="mx-auto mt-1 flex max-w-3xl items-center justify-between text-xs text-gray-400">
        <label className="flex items-center gap-1.5 cursor-pointer select-none">
          <input
            type="checkbox"
            checked={forceWrite}
            onChange={(e) => setForceWrite(e.target.checked)}
            className="h-3.5 w-3.5 rounded border-gray-300 text-amber-500 focus:ring-amber-500"
          />
          <span className={forceWrite ? 'font-medium text-amber-600' : ''}>
            记住这条
          </span>
        </label>
        <span>Enter 发送 · Shift+Enter 换行</span>
      </div>
    </div>
  );
}
