import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import ConflictsPage from './ConflictsPage';

// Mock the conflicts API
vi.mock('../api/conflicts', () => ({
  getConflicts: vi.fn(),
  resolveConflict: vi.fn(),
}));

import { getConflicts, resolveConflict } from '../api/conflicts';

const mockGetConflicts = getConflicts as ReturnType<typeof vi.fn>;
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

  it('shows an error message on load failure', async () => {
    mockGetConflicts.mockRejectedValue(new Error('Network error'));
    renderPage();

    await waitFor(() => {
      expect(screen.getByText(/Network error/)).toBeDefined();
    });
  });
});
