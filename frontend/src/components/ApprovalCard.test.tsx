import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi } from 'vitest';
import ApprovalCard from '../components/ApprovalCard';
import type { Interrupt } from '../types';

function makeInterrupt(overrides: Partial<Interrupt> = {}): Interrupt {
  return {
    type: 'approval',
    tool_name: 'write_memory_tool',
    summary: '把用户偏好写入长期记忆',
    tool_args: {},
    ...overrides,
  };
}

describe('ApprovalCard', () => {
  it('renders the tool name label in Chinese via TOOL_LABELS', () => {
    render(
      <ApprovalCard interrupt={makeInterrupt({ tool_name: 'write_memory_tool' })} onResume={vi.fn()} />,
    );
    expect(screen.getByText('待批准：写入记忆')).toBeInTheDocument();
  });

  it('renders the summary from the interrupt', () => {
    const summary = '创建一条关于缓存设计的记忆';
    render(<ApprovalCard interrupt={makeInterrupt({ summary })} onResume={vi.fn()} />);
    expect(screen.getByText(summary)).toBeInTheDocument();
  });

  it('falls back to JSON.stringify(tool_args) when summary is missing', () => {
    const args = { path: 'docs/architecture.md' };
    render(
      <ApprovalCard
        interrupt={makeInterrupt({ summary: undefined, tool_args: args })}
        onResume={vi.fn()}
      />,
    );
    expect(screen.getByText('{"path":"docs/architecture.md"}')).toBeInTheDocument();
  });

  it('calls onResume with { approved: true } when approve is clicked', async () => {
    const user = userEvent.setup();
    const onResume = vi.fn();
    render(<ApprovalCard interrupt={makeInterrupt()} onResume={onResume} />);

    await user.click(screen.getByRole('button', { name: /批准/ }));
    expect(onResume).toHaveBeenCalledWith({ approved: true });
  });

  it('calls onResume with reject data when reject is clicked', async () => {
    const user = userEvent.setup();
    const onResume = vi.fn();
    render(<ApprovalCard interrupt={makeInterrupt()} onResume={onResume} />);

    await user.click(screen.getByRole('button', { name: /拒绝/ }));
    expect(onResume).toHaveBeenCalledWith({ approved: false, reason: '用户拒绝了工具调用。' });
  });

  it('disables both buttons while resolving', () => {
    render(<ApprovalCard interrupt={makeInterrupt()} onResume={vi.fn()} isResolving />);

    expect(screen.getByRole('button', { name: /批准/ })).toBeDisabled();
    expect(screen.getByRole('button', { name: /拒绝/ })).toBeDisabled();
  });

  it('shows the raw tool name for an unknown tool', () => {
    render(<ApprovalCard interrupt={makeInterrupt({ tool_name: 'mystery_tool' })} onResume={vi.fn()} />);
    expect(screen.getByText('待批准：mystery_tool')).toBeInTheDocument();
  });
});
