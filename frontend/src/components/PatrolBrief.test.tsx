import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import PatrolBrief from './PatrolBrief';

// Mock the API module
vi.mock('../api/patrol', () => ({
  listPatrolLogs: vi.fn(),
}));

import { listPatrolLogs } from '../api/patrol';

function mockLogs(items: Array<{
  id: string;
  patrol_type: string;
  trigger: string;
  status: string;
  finding_count: number;
  started_at: string;
  completed_at: string | null;
}>) {
  return { items, total: items.length };
}

describe('PatrolBrief', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows loading skeleton initially', () => {
    (listPatrolLogs as ReturnType<typeof vi.fn>).mockReturnValue(new Promise(() => {})); // never resolves
    render(
      <MemoryRouter>
        <PatrolBrief />
      </MemoryRouter>,
    );
    const skeletons = document.querySelectorAll('.animate-pulse');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it('shows "not yet run" when no logs exist', async () => {
    (listPatrolLogs as ReturnType<typeof vi.fn>).mockResolvedValue({ items: [], total: 0 });
    render(
      <MemoryRouter>
        <PatrolBrief />
      </MemoryRouter>,
    );
    await waitFor(() => {
      expect(screen.getByText(/尚未执行/)).toBeInTheDocument();
    });
  });

  it('shows "no findings" when log has zero findings', async () => {
    (listPatrolLogs as ReturnType<typeof vi.fn>).mockResolvedValue(
      mockLogs([
        {
          id: '1',
          patrol_type: 'daily',
          trigger: 'cron',
          status: 'completed',
          finding_count: 0,
          started_at: '2026-01-15T00:00:00Z',
          completed_at: '2026-01-15T00:01:00Z',
        },
      ]),
    );
    render(
      <MemoryRouter>
        <PatrolBrief />
      </MemoryRouter>,
    );
    await waitFor(() => {
      expect(screen.getByText(/未发现需关注事项/)).toBeInTheDocument();
    });
  });

  it('shows finding count when log has findings', async () => {
    (listPatrolLogs as ReturnType<typeof vi.fn>).mockResolvedValue(
      mockLogs([
        {
          id: '1',
          patrol_type: 'daily',
          trigger: 'cron',
          status: 'completed',
          finding_count: 3,
          started_at: '2026-01-15T00:00:00Z',
          completed_at: '2026-01-15T00:01:00Z',
        },
      ]),
    );
    render(
      <MemoryRouter>
        <PatrolBrief />
      </MemoryRouter>,
    );
    await waitFor(() => {
      expect(screen.getByText(/3 个发现/)).toBeInTheDocument();
      expect(screen.getByText(/查看全部/)).toBeInTheDocument();
    });
  });

  it('handles API error silently', async () => {
    (listPatrolLogs as ReturnType<typeof vi.fn>).mockRejectedValue(new Error('fail'));
    const { container } = render(
      <MemoryRouter>
        <PatrolBrief />
      </MemoryRouter>,
    );
    await waitFor(() => {
      // Component should render nothing on error
      expect(container.innerHTML).toBe('');
    });
  });
});
