import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import StatsDashboard from './StatsDashboard';
import type { MemoryStatsResponse } from '../types';

function mockStats(overrides: Partial<MemoryStatsResponse> = {}): MemoryStatsResponse {
  return {
    total_memories: 42,
    total_chunks: 150,
    total_conversations: 7,
    by_source_type: [
      { source_type: 'api', count: 20 },
      { source_type: 'git_commit', count: 15 },
    ],
    avg_decay_factor: 0.85,
    avg_entities_per_memory: 3.2,
    avg_relations_per_memory: 1.5,
    recent_count_7d: 10,
    top_entities: [
      { name: 'React', count: 8 },
      { name: 'TypeScript', count: 5 },
    ],
    ...overrides,
  };
}

describe('StatsDashboard', () => {
  it('shows skeleton cards when loading', () => {
    render(
      <StatsDashboard stats={null} isLoading={true} error={null} onRetry={vi.fn()} />,
    );
    // Skeleton cards have animate-pulse class
    const skeletons = document.querySelectorAll('.animate-pulse');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it('shows error message with retry button', () => {
    const onRetry = vi.fn();
    render(
      <StatsDashboard
        stats={null}
        isLoading={false}
        error="Network error"
        onRetry={onRetry}
      />,
    );

    expect(screen.getByText(/Network error/)).toBeInTheDocument();
    screen.getByText(/重试/).click();
    expect(onRetry).toHaveBeenCalledOnce();
  });

  it('shows empty state when stats is null and not loading and no error', () => {
    render(
      <StatsDashboard stats={null} isLoading={false} error={null} onRetry={vi.fn()} />,
    );
    expect(screen.getByText(/暂无数据/)).toBeInTheDocument();
  });

  it('renders KPI cards with formatted numbers', () => {
    render(
      <StatsDashboard
        stats={mockStats()}
        isLoading={false}
        error={null}
        onRetry={vi.fn()}
      />,
    );

    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText('150')).toBeInTheDocument();
    expect(screen.getByText('7')).toBeInTheDocument();
    expect(screen.getByText('10')).toBeInTheDocument(); // recent 7d
  });

  it('renders top entity tags', () => {
    render(
      <StatsDashboard
        stats={mockStats()}
        isLoading={false}
        error={null}
        onRetry={vi.fn()}
      />,
    );

    expect(screen.getByText('React')).toBeInTheDocument();
    expect(screen.getByText('TypeScript')).toBeInTheDocument();
  });
});
