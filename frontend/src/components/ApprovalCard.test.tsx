import { render, screen, within } from '@testing-library/react';
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

function makeBatchInterrupt(
  calls: { tool_name: string; tool_args?: Record<string, unknown>; summary?: string }[] = [],
): Interrupt {
  return {
    type: 'batch',
    calls,
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

  describe('batch approval (multiple sensitive tools)', () => {
    const calls = [
      { id: 'call_1', tool_name: 'write_memory_tool', tool_args: { content: '记录 A' } },
      { id: 'call_2', tool_name: 'ingest_git_repo_tool', tool_args: { repo_path: '/repo' }, summary: 'Repo: /repo' },
    ];

    it('renders one row per tool with its label and summary', () => {
      render(<ApprovalCard interrupt={makeBatchInterrupt(calls)} onResume={vi.fn()} />);

      expect(screen.getByText('待批准：2 个工具操作')).toBeInTheDocument();
      // Two rows, each with its own approve/reject buttons.
      const rows = screen.getAllByRole('listitem');
      expect(rows).toHaveLength(2);
      expect(within(rows[0]).getByText('写入记忆')).toBeInTheDocument();
      expect(within(rows[0]).getByText(/记录 A/)).toBeInTheDocument();
      expect(within(rows[1]).getByText('摄取 Git 仓库')).toBeInTheDocument();
      expect(within(rows[1]).getByText('Repo: /repo')).toBeInTheDocument();
      // Global approve-all / reject-all actions.
      expect(screen.getByRole('button', { name: /全部批准/ })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /全部拒绝/ })).toBeInTheDocument();
    });

    it('approves only the clicked tool and rejects the rest, addressing by id', async () => {
      const user = userEvent.setup();
      const onResume = vi.fn();
      render(<ApprovalCard interrupt={makeBatchInterrupt(calls)} onResume={onResume} />);

      const rows = screen.getAllByRole('listitem');
      await user.click(within(rows[0]).getByRole('button', { name: /批准/ }));

      expect(onResume).toHaveBeenCalledWith({
        calls: [
          { id: 'call_1', tool_name: 'write_memory_tool', approved: true, reason: undefined },
          { id: 'call_2', tool_name: 'ingest_git_repo_tool', approved: false, reason: '用户拒绝了工具调用。' },
        ],
      });
    });

    it('rejects only the clicked tool and approves the rest', async () => {
      const user = userEvent.setup();
      const onResume = vi.fn();
      render(<ApprovalCard interrupt={makeBatchInterrupt(calls)} onResume={onResume} />);

      const rows = screen.getAllByRole('listitem');
      await user.click(within(rows[1]).getByRole('button', { name: /拒绝/ }));

      expect(onResume).toHaveBeenCalledWith({
        calls: [
          { id: 'call_1', tool_name: 'write_memory_tool', approved: true, reason: undefined },
          { id: 'call_2', tool_name: 'ingest_git_repo_tool', approved: false, reason: '用户拒绝了工具调用。' },
        ],
      });
    });

    it('sends a calls array with all approved when "全部批准" is clicked', async () => {
      const user = userEvent.setup();
      const onResume = vi.fn();
      render(<ApprovalCard interrupt={makeBatchInterrupt(calls)} onResume={onResume} />);

      await user.click(screen.getByRole('button', { name: /全部批准/ }));
      expect(onResume).toHaveBeenCalledWith({
        calls: [
          { id: 'call_1', tool_name: 'write_memory_tool', approved: true },
          { id: 'call_2', tool_name: 'ingest_git_repo_tool', approved: true },
        ],
      });
    });

    it('sends a calls array with all rejected and a reason when "全部拒绝" is clicked', async () => {
      const user = userEvent.setup();
      const onResume = vi.fn();
      render(<ApprovalCard interrupt={makeBatchInterrupt(calls)} onResume={onResume} />);

      await user.click(screen.getByRole('button', { name: /全部拒绝/ }));
      expect(onResume).toHaveBeenCalledWith({
        calls: [
          { id: 'call_1', tool_name: 'write_memory_tool', approved: false },
          { id: 'call_2', tool_name: 'ingest_git_repo_tool', approved: false },
        ],
        reason: '用户拒绝了工具调用。',
      });
    });
  });
});
