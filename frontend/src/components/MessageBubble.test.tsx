import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import MessageBubble from '../components/MessageBubble';
import type { Message, Source, ToolCall } from '../types';

// SourcesPanel depends on React Router + AppContext; mock it so MessageBubble
// can be rendered as a pure presentational component in isolation.
vi.mock('../components/SourcesPanel', () => ({
  default: () => <div data-testid="sources-panel">mock sources panel</div>,
}));
// CitationText likewise depends on React Router + AppContext — mock it so the
// bubble tests stay focused on layout, not citation resolution.
vi.mock('../components/CitationText', () => ({
  default: ({ text }: { text: string }) => <span>{text}</span>,
}));

function makeMessage(overrides: Partial<Message> = {}): Message {
  return { role: 'assistant', content: '你好', ...overrides };
}

const toolCalls: ToolCall[] = [
  {
    tool: 'write_memory_tool',
    content: JSON.stringify({ action: 'inserted', summary: 'New memory' }),
  },
];

const sources: Source[] = [{ type: 'memory', summary: '一篇关于缓存的记忆', relevance: 0.9 }];

describe('MessageBubble', () => {
  it('renders user messages right-aligned with a blue bubble', () => {
    const { container } = render(<MessageBubble message={makeMessage({ role: 'user', content: 'hello' })} />);
    const outer = container.firstElementChild as HTMLElement;
    expect(outer.className).toContain('justify-end');
    expect(container.querySelector('.rounded-2xl.bg-blue-600')).not.toBeNull();
  });

  it('renders assistant messages left-aligned with a grey bubble', () => {
    const { container } = render(<MessageBubble message={makeMessage({ role: 'assistant' })} />);
    const outer = container.firstElementChild as HTMLElement;
    expect(outer.className).toContain('justify-start');
    expect(container.querySelector('.rounded-2xl.bg-gray-100')).not.toBeNull();
  });

  it('renders system messages centred with an amber background', () => {
    const { container } = render(<MessageBubble message={makeMessage({ role: 'system', content: '注意' })} />);
    const outer = container.firstElementChild as HTMLElement;
    expect(outer.className).toContain('justify-center');
    expect(container.querySelector('.bg-amber-50')).not.toBeNull();
  });

  it('renders error messages with distinct red error styling, not a normal bubble', () => {
    const { container } = render(
      <MessageBubble message={makeMessage({ role: 'system', kind: 'error', content: '错误: boom' })} />,
    );
    expect(screen.getByText('错误: boom')).toBeInTheDocument();
    // Red error bubble with a border — clearly distinct from the grey assistant bubble.
    expect(container.querySelector('.bg-red-50')).not.toBeNull();
    expect(container.querySelector('.border-red-300')).not.toBeNull();
    expect(container.querySelector('.italic')).not.toBeNull();
    expect(container.querySelector('.bg-gray-100')).toBeNull();
    expect(container.querySelector('.bg-amber-50')).toBeNull();
  });

  it('does not apply error styling to regular system notices', () => {
    const { container } = render(<MessageBubble message={makeMessage({ role: 'system', content: '注意' })} />);
    expect(container.querySelector('.bg-red-50')).toBeNull();
    expect(container.querySelector('.border-red-300')).toBeNull();
  });

  it('does not apply error styling to regular assistant messages', () => {
    const { container } = render(<MessageBubble message={makeMessage({ content: '正常回复' })} />);
    expect(container.querySelector('.bg-red-50')).toBeNull();
    expect(container.querySelector('.border-red-300')).toBeNull();
    expect(container.querySelector('.bg-gray-100')).not.toBeNull();
  });

  it('shows a typing indicator for an empty streaming assistant message', () => {
    const { container } = render(<MessageBubble message={makeMessage({ content: '' })} isStreaming />);
    expect(container.querySelectorAll('.animate-bounce')).toHaveLength(3);
  });

  it('renders streamed content without a typing indicator', () => {
    const { container } = render(<MessageBubble message={makeMessage({ content: '流式内容' })} isStreaming />);
    expect(screen.getByText('流式内容')).toBeInTheDocument();
    expect(container.querySelectorAll('.animate-bounce')).toHaveLength(0);
  });

  it('does not render SourcesPanel when only toolCalls are present', () => {
    render(<MessageBubble message={makeMessage({ _meta: { toolCalls, sources: [] } })} />);
    expect(screen.queryByTestId('sources-panel')).not.toBeInTheDocument();
  });

  it('renders the SourcesPanel below an assistant message with sources', () => {
    render(<MessageBubble message={makeMessage({ _meta: { toolCalls: [], sources } })} />);
    expect(screen.getByTestId('sources-panel')).toBeInTheDocument();
  });

  it('renders SourcesPanel when both toolCalls and sources are present', () => {
    render(<MessageBubble message={makeMessage({ _meta: { toolCalls, sources } })} />);
    expect(screen.getByTestId('sources-panel')).toBeInTheDocument();
  });

  it('renders no panels when the message has no metadata', () => {
    render(<MessageBubble message={makeMessage()} />);
    expect(screen.queryByTestId('sources-panel')).not.toBeInTheDocument();
  });
});
