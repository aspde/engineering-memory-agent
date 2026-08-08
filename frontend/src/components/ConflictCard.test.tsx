import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import ConflictCard from './ConflictCard';
import type { Interrupt } from '../types';

function createConflictInterrupt(
  overrides: Partial<Interrupt> = {},
): Interrupt {
  return {
    type: 'conflict',
    new_summary: 'New memory about React hooks',
    existing_summary: 'Existing memory about React classes',
    ...overrides,
  };
}

describe('ConflictCard', () => {
  it('renders new and existing summaries', () => {
    const interrupt = createConflictInterrupt();
    render(<ConflictCard interrupt={interrupt} onResume={vi.fn()} />);

    expect(screen.getByText('New memory about React hooks')).toBeInTheDocument();
    expect(screen.getByText('Existing memory about React classes')).toBeInTheDocument();
  });

  it('shows (空) for empty new_summary', () => {
    const interrupt = createConflictInterrupt({ new_summary: '' });
    render(<ConflictCard interrupt={interrupt} onResume={vi.fn()} />);

    expect(screen.getByText('(空)')).toBeInTheDocument();
  });

  it('renders all four resolution buttons', () => {
    render(<ConflictCard interrupt={createConflictInterrupt()} onResume={vi.fn()} />);

    expect(screen.getByText(/保留现有/)).toBeInTheDocument();
    expect(screen.getByText(/覆盖/)).toBeInTheDocument();
    expect(screen.getByText(/合并/)).toBeInTheDocument();
    expect(screen.getByText(/两者都保留/)).toBeInTheDocument();
  });

  it('keep_existing button calls onResume with keep_existing', async () => {
    const onResume = vi.fn();
    render(<ConflictCard interrupt={createConflictInterrupt()} onResume={onResume} />);

    screen.getByText(/保留现有/).click();
    expect(onResume).toHaveBeenCalledWith({ resolution: 'keep_existing' });
  });

  it('overwrite button calls onResume with overwrite', async () => {
    const onResume = vi.fn();
    render(<ConflictCard interrupt={createConflictInterrupt()} onResume={onResume} />);

    screen.getByText(/覆盖/).click();
    expect(onResume).toHaveBeenCalledWith({ resolution: 'overwrite' });
  });

  it('merge button calls onResume with merge', async () => {
    const onResume = vi.fn();
    render(<ConflictCard interrupt={createConflictInterrupt()} onResume={onResume} />);

    screen.getByText(/合并/).click();
    expect(onResume).toHaveBeenCalledWith({ resolution: 'merge' });
  });

  it('keep_both button calls onResume with keep_both', async () => {
    const onResume = vi.fn();
    render(<ConflictCard interrupt={createConflictInterrupt()} onResume={onResume} />);

    screen.getByText(/两者都保留/).click();
    expect(onResume).toHaveBeenCalledWith({ resolution: 'keep_both' });
  });

  it('disables all buttons when isResolving', () => {
    render(
      <ConflictCard
        interrupt={createConflictInterrupt()}
        onResume={vi.fn()}
        isResolving={true}
      />,
    );

    const buttons = screen.getAllByRole('button');
    for (const btn of buttons) {
      expect(btn).toBeDisabled();
    }
  });

  it('re-words the card for patrol contradictions', () => {
    render(
      <ConflictCard
        interrupt={createConflictInterrupt()}
        onResume={vi.fn()}
        variant="patrol"
      />,
    );

    expect(screen.getByText('检测到记忆矛盾（巡检发现）')).toBeInTheDocument();
    expect(screen.getByText('记忆 B')).toBeInTheDocument();
    expect(screen.getByText('记忆 A')).toBeInTheDocument();
    // The default ingestion wording is gone.
    expect(screen.queryByText('检测到记忆冲突')).not.toBeInTheDocument();
    expect(screen.queryByText('新记忆')).not.toBeInTheDocument();
    expect(screen.queryByText('现有记忆')).not.toBeInTheDocument();
  });

  it('keeps the four options identical for patrol contradictions', () => {
    render(
      <ConflictCard
        interrupt={createConflictInterrupt()}
        onResume={vi.fn()}
        variant="patrol"
      />,
    );

    expect(screen.getByText(/保留现有/)).toBeInTheDocument();
    expect(screen.getByText(/覆盖/)).toBeInTheDocument();
    expect(screen.getByText(/合并/)).toBeInTheDocument();
    expect(screen.getByText(/两者都保留/)).toBeInTheDocument();
  });
});
