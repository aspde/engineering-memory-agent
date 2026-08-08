import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import ConflictsPage from './ConflictsPage';

// Mock the conflicts API
vi.mock('../api/conflicts', () => ({
  getConflicts: vi.fn(),
  reopenConflict: vi.fn(),
  resolveConflict: vi.fn(),
}));

import { getConflicts, reopenConflict, resolveConflict } from '../api/conflicts';

const mockGetConflicts = getConflicts as ReturnType<typeof vi.fn>;
const mockReopenConflict = reopenConflict as ReturnType<typeof vi.fn>;
const mockResolveConflict = resolveConflict as ReturnType<typeof vi.fn>;

function renderPage() {
  return render(
    <MemoryRouter>
      <ConflictsPage />
    </MemoryRouter>,
  );
}

const conflict = {
  id: 'conflict-1',
  source: 'ci',
  source_type: 'ci_build',
  existing_id: 'existing-1',
  existing_summary: 'Existing summary',
  new_summary: 'New summary',
  status: 'pending',
  resolution: null,
  created_at: '2026-08-07T10:00:00Z',
};

describe('ConflictsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows empty state when no pending conflicts', async () => {
    mockGetConflicts.mockResolvedValue([]);
    renderPage();

    await waitFor(() => {
      expect(screen.getByText(/暂无待处理的记忆冲突/)).toBeDefined();
    });
  });

  it('renders a ConflictCard for each pending conflict', async () => {
    mockGetConflicts.mockResolvedValue([conflict]);
    renderPage();

    await waitFor(() => {
      expect(screen.getByText('检测到记忆冲突')).toBeDefined();
    });
    expect(screen.getByText('New summary')).toBeDefined();
    expect(screen.getByText('Existing summary')).toBeDefined();
    expect(screen.getByText(/来源：ci/)).toBeDefined();
  });

  it('resolves a conflict via the API and removes it from the list', async () => {
    mockGetConflicts.mockResolvedValue([conflict]);
    mockResolveConflict.mockResolvedValue({
      id: 'conflict-1',
      resolution: 'keep_existing',
      outcome: {},
    });
    renderPage();

    await waitFor(() => {
      expect(screen.getByText('检测到记忆冲突')).toBeDefined();
    });

    const user = userEvent.setup();
    await user.click(screen.getByText('📌 保留现有'));

    await waitFor(() => {
      expect(mockResolveConflict).toHaveBeenCalledWith('conflict-1', 'keep_existing');
    });
    await waitFor(() => {
      expect(screen.getByText(/暂无待处理的记忆冲突/)).toBeDefined();
    });
  });

  it('renders a patrol badge for patrol contradictions', async () => {
    const patrolConflict = {
      ...conflict,
      id: 'patrol-1',
      source: 'patrol',
      conflict_type: 'patrol',
      peer_id: 'peer-1',
    };
    mockGetConflicts.mockResolvedValue([patrolConflict]);
    renderPage();

    await waitFor(() => {
      // The patrol badge and the type-filter tab both say 巡检矛盾.
      expect(screen.getAllByText('巡检矛盾').length).toBeGreaterThan(0);
    });
    expect(screen.getByText(/来源：patrol/)).toBeDefined();
  });

  it('switches to the arbitrated view and reopens a resolved patrol conflict', async () => {
    const arbitrated = {
      ...conflict,
      id: 'patrol-2',
      source: 'patrol',
      conflict_type: 'patrol',
      peer_id: 'peer-1',
      status: 'resolved',
      resolution: 'keep_both',
    };
    // pending view → empty; arbitrated view → the resolved row
    mockGetConflicts.mockResolvedValueOnce([]);
    mockGetConflicts.mockResolvedValue([arbitrated]);
    mockReopenConflict.mockResolvedValue({ id: 'patrol-2', status: 'pending' });
    renderPage();

    // Wait for the initial pending view to finish loading.
    await waitFor(() => {
      expect(screen.getByText(/暂无待处理的记忆冲突/)).toBeDefined();
    });

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: '已仲裁（巡检）' }));

    await waitFor(() => {
      expect(mockGetConflicts).toHaveBeenCalledWith({
        status: 'resolved',
        conflict_type: 'patrol',
      });
    });
    await waitFor(() => {
      expect(screen.getByText('↩ 重新打开仲裁')).toBeDefined();
    });

    await user.click(screen.getByText('↩ 重新打开仲裁'));
    await waitFor(() => {
      expect(mockReopenConflict).toHaveBeenCalledWith('patrol-2');
    });
  });

  it('shows an error message on load failure', async () => {
    mockGetConflicts.mockRejectedValue(new Error('Network error'));
    renderPage();

    await waitFor(() => {
      expect(screen.getByText(/Network error/)).toBeDefined();
    });
  });
});
