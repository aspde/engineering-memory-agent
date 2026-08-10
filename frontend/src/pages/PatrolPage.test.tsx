import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import PatrolPage from './PatrolPage';

vi.mock('../api/patrol', () => ({
  dismissFinding: vi.fn(),
  getPatrolLog: vi.fn(),
  listPatrolLogs: vi.fn(),
  queuePatrolConflict: vi.fn(),
  triggerPatrol: vi.fn(),
}));

import { dismissFinding, getPatrolLog, listPatrolLogs, queuePatrolConflict } from '../api/patrol';

const mockListPatrolLogs = listPatrolLogs as ReturnType<typeof vi.fn>;
const mockGetPatrolLog = getPatrolLog as ReturnType<typeof vi.fn>;
const mockQueuePatrolConflict = queuePatrolConflict as ReturnType<typeof vi.fn>;
const mockDismissFinding = dismissFinding as ReturnType<typeof vi.fn>;

const logSummary = {
  id: 'log-1',
  patrol_type: 'weekly',
  trigger: 'cron',
  status: 'completed',
  finding_count: 1,
  started_at: '2026-08-08T09:00:00Z',
  completed_at: '2026-08-08T09:01:00Z',
};

const contradictionFinding = {
  memory_a_id: 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
  memory_a_summary: 'Use PostgreSQL for storage',
  memory_b_id: 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb',
  memory_b_summary: 'Migrate away from PostgreSQL',
  conflict_description: 'opposite recommendations',
  severity: 'warning',
};

function renderPage() {
  return render(
    <MemoryRouter>
      <PatrolPage />
    </MemoryRouter>,
  );
}

describe('PatrolPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockListPatrolLogs.mockResolvedValue({ items: [logSummary], total: 1 });
    mockGetPatrolLog.mockResolvedValue({
      ...logSummary,
      findings: { contradictions: [contradictionFinding] },
      dismissed_findings: [],
    });
    mockQueuePatrolConflict.mockResolvedValue({
      conflict_id: 'conflict-1',
      status: 'queued',
    });
  });

  it('queues a contradiction finding for arbitration from the card', async () => {
    renderPage();

    // Load the log list, then open the detail.
    await waitFor(() => {
      expect(screen.getByText(/1 个发现/)).toBeDefined();
    });
    const user = userEvent.setup();
    await user.click(screen.getByText(/1 个发现/));

    await waitFor(() => {
      expect(screen.getByText('转入仲裁')).toBeDefined();
    });

    await user.click(screen.getByText('转入仲裁'));

    await waitFor(() => {
      expect(mockQueuePatrolConflict).toHaveBeenCalledWith('log-1', contradictionFinding);
    });
    await waitFor(() => {
      expect(screen.getByText('已转入仲裁')).toBeDefined();
    });
    // Queuing auto-dismisses the finding from the patrol view.
    await waitFor(() => {
      expect(mockDismissFinding).toHaveBeenCalledWith('log-1', expect.any(String));
    });
  });

  it('shows a notice when the pair was already queued', async () => {
    mockQueuePatrolConflict.mockResolvedValue({
      conflict_id: 'conflict-1',
      status: 'already_pending',
    });
    renderPage();

    const user = userEvent.setup();
    await waitFor(() => {
      expect(screen.getByText(/1 个发现/)).toBeDefined();
    });
    await user.click(screen.getByText(/1 个发现/));
    await waitFor(() => {
      expect(screen.getByText('转入仲裁')).toBeDefined();
    });
    await user.click(screen.getByText('转入仲裁'));

    await waitFor(() => {
      expect(screen.getByText('该矛盾已在待处理列表中')).toBeDefined();
    });
    // Still marked arbitrated so the user does not re-click.
    expect(screen.getByText('已转入仲裁')).toBeDefined();
  });
});
