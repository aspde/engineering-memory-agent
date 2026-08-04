import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import ChatInput from '../components/ChatInput';

describe('ChatInput', () => {
  it('renders a textarea with the default placeholder', () => {
    render(<ChatInput onSend={vi.fn()} />);
    expect(screen.getByPlaceholderText('向 EMA 提问…')).toBeInTheDocument();
  });

  it('renders a textarea with a custom placeholder', () => {
    render(<ChatInput onSend={vi.fn()} placeholder="自定义提示…" />);
    expect(screen.getByPlaceholderText('自定义提示…')).toBeInTheDocument();
  });

  it('calls onSend with trimmed text when the send button is clicked', async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByPlaceholderText('向 EMA 提问…');

    await user.type(textarea, '  hello  ');
    await user.click(screen.getByRole('button', { name: '发送' }));
    expect(onSend).toHaveBeenCalledWith('hello', false);
  });

  it('calls onSend when Enter is pressed', async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByPlaceholderText('向 EMA 提问…');

    await user.type(textarea, 'hello{enter}');
    expect(onSend).toHaveBeenCalledWith('hello', false);
  });

  it('inserts a newline on Shift+Enter without sending', async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByPlaceholderText('向 EMA 提问…');

    await user.type(textarea, 'line1{shift>}{enter}{/shift}');
    expect(onSend).not.toHaveBeenCalled();
    expect(textarea).toHaveValue('line1\n');
  });

  it('keeps the send button disabled and does nothing on Enter for empty input', async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} />);
    const button = screen.getByRole('button', { name: '发送' });
    expect(button).toBeDisabled();

    const textarea = screen.getByPlaceholderText('向 EMA 提问…');
    await user.type(textarea, '{enter}');
    expect(onSend).not.toHaveBeenCalled();
  });

  it('does not call onSend for whitespace-only input', async () => {
    const user = userEvent.setup();
    const onSend = vi.fn();
    render(<ChatInput onSend={onSend} />);
    const textarea = screen.getByPlaceholderText('向 EMA 提问…');

    await user.type(textarea, '   ');
    await user.click(screen.getByRole('button', { name: '发送' }));
    expect(onSend).not.toHaveBeenCalled();
  });

  it('clears the textarea after sending', async () => {
    const user = userEvent.setup();
    render(<ChatInput onSend={vi.fn()} />);
    const textarea = screen.getByPlaceholderText('向 EMA 提问…');

    await user.type(textarea, 'hello{enter}');
    expect(textarea).toHaveValue('');
  });

  it('disables the textarea and send button when disabled', () => {
    render(<ChatInput onSend={vi.fn()} disabled />);
    expect(screen.getByPlaceholderText('向 EMA 提问…')).toBeDisabled();
    expect(screen.getByRole('button', { name: '发送' })).toBeDisabled();
  });

  it('auto-grows the textarea when content overflows', async () => {
    const user = userEvent.setup();
    render(<ChatInput onSend={vi.fn()} />);
    const textarea = screen.getByPlaceholderText('向 EMA 提问…');
    Object.defineProperty(textarea, 'scrollHeight', { value: 100, configurable: true });

    await user.type(textarea, 'line1\nline2\nline3');
    expect(textarea.style.height).toBe('100px');
  });
});
