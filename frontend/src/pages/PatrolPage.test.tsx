import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, act, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import PatrolPage from './PatrolPage';

vi.mock('../api/patrol', () => ({
  dismissFinding: vi.fn(),
  getPatrolLog: vi.fn(),
  listPatrolLogs: vi.fn(),
  mergePatrolFinding: vi.fn(),
  queuePatrolConflict: vi.fn(),
  triggerPatrol: vi.fn(),
}));

import { dismissFinding, getPatrolLog, listPatrolLogs, mergePatrolFinding, queuePatrolConflict, triggerPatrol } from '../api/patrol';

const mockListPatrolLogs = listPatrolLogs as ReturnType<typeof vi.fn>;
const mockGetPatrolLog = getPatrolLog as ReturnType<typeof vi.fn>;
const mockQueuePatrolConflict = queuePatrolConflict as ReturnType<typeof vi.fn>;
const mockDismissFinding = dismissFinding as ReturnType<typeof vi.fn>;
const mockMergePatrolFinding = mergePatrolFinding as ReturnType<typeof vi.fn>;

const logSummary = {
  id: 'log-1',
  patrol_type: 'weekly',
  trigger: 'cron',
  status: 'completed',
  finding_count: 1,
  started_at: '2026-08-08T09:00:00Z',
  completed_at: '2026-08-08T09:01:00Z',
};

/** A patrol log whose started_at falls on *today* in the local timezone —
 *  built fresh so the date matches whenever the test runs. */
function todayLog(overrides: Partial<typeof logSummary> = {}) {
  return {
    ...logSummary,
    started_at: new Date().toISOString(),
    ...overrides,
  };
}

const contradictionFinding = {
  memory_a_id: 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
  memory_a_summary: 'Use PostgreSQL for storage',
  memory_b_id: 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb',
  memory_b_summary: 'Migrate away from PostgreSQL',
  conflict_description: 'opposite recommendations',
  severity: 'warning',
};

/** A daily pattern-match finding: the new memory duplicates a historical one.
 *  Keyed by finding index (no memory-pair key), like the backend emits. */
const patternFinding = {
  matched_memory_id: 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
  matched_summary: 'PostgreSQL is the store',
  new_memory_id: 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb',
  new_summary: 'Use PostgreSQL for persistence',
  reason: 'same content, newer write',
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
    mockMergePatrolFinding.mockResolvedValue({
      ok: true,
      kept_id: 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa',
      merged_id: 'bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb',
      action: 'conflict_resolved',
    });
  });

  it('renders entity coverage findings with their entity name, not placeholders', async () => {
    mockGetPatrolLog.mockResolvedValue({
      ...logSummary,
      findings: {
        entity_coverage: [
          {
            entity_name: 'PostgreSQL',
            total_memories: 30,
            covered_domains: ['deployment', 'monitoring'],
            missing_domains: ['backup', 'security'],
            recommendation: '补充备份与安全相关的文档',
          },
          {
            entity_name: 'CI/CD',
            total_memories: 18,
            missing_domains: [],
            recommendation: '',
          },
        ],
      },
      dismissed_findings: [],
    });
    renderPage();

    const user = userEvent.setup();
    await waitFor(() => {
      expect(screen.getByText(/1 个发现/)).toBeDefined();
    });
    await user.click(screen.getByText(/1 个发现/));

    await waitFor(() => {
      expect(screen.getByText(/实体覆盖/)).toBeDefined();
    });
    // Entity names render as card titles.
    expect(screen.getByText('PostgreSQL')).toBeDefined();
    expect(screen.getByText('CI/CD')).toBeDefined();
    // Missing domains surface in the description.
    expect(screen.getByText(/缺失领域：backup、security/)).toBeDefined();
    // No placeholder titles.
    expect(screen.queryByText(/#1/)).toBeNull();
    expect(screen.queryByText(/#2/)).toBeNull();
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

  it('shows a hint that today’s patrol already ran, without blocking re-trigger', async () => {
    // A completed daily patrol today + an older weekly run (not today).
    mockListPatrolLogs.mockResolvedValue({
      items: [
        todayLog({ id: 'log-daily', patrol_type: 'daily', status: 'completed' }),
        { ...logSummary, id: 'log-weekly-old', patrol_type: 'weekly' },
      ],
      total: 2,
    });
    renderPage();

    await waitFor(() => {
      expect(screen.getByText(/今日每日巡检已执行/)).toBeDefined();
    });

    // The hint is informational — the trigger button stays enabled.
    const dailyButton = screen.getByRole('button', { name: /▶ 每日/ });
    expect((dailyButton as HTMLButtonElement).disabled).toBe(false);
  });

  it('marks a today run as incomplete when the patrol failed', async () => {
    mockListPatrolLogs.mockResolvedValue({
      items: [todayLog({ id: 'log-daily-fail', patrol_type: 'daily', status: 'failed' })],
      total: 1,
    });
    renderPage();

    await waitFor(() => {
      expect(screen.getByText(/未完成（可再次执行）/)).toBeDefined();
    });
  });

  it('shows no hint when no patrol ran today', async () => {
    mockListPatrolLogs.mockResolvedValue({
      items: [{ ...logSummary, id: 'log-weekly-old', patrol_type: 'weekly' }],
      total: 1,
    });
    renderPage();

    await waitFor(() => {
      expect(screen.getByText(/1 个发现/)).toBeDefined();
    });
    expect(screen.queryByText(/今日.*已执行/)).toBeNull();
  });

  it('shows the empty-state notice when every category is an empty array', async () => {
    mockListPatrolLogs.mockResolvedValue({
      items: [{ ...logSummary, finding_count: 0 }],
      total: 1,
    });
    mockGetPatrolLog.mockResolvedValue({
      ...logSummary,
      finding_count: 0,
      findings: { contradictions: [], entity_coverage: [], decay_alerts: [] },
      dismissed_findings: [],
    });
    renderPage();

    const user = userEvent.setup();
    await waitFor(() => {
      expect(screen.getByText(/0 个发现/)).toBeDefined();
    });
    await user.click(screen.getByText(/0 个发现/));

    // All-empty categories are a "nothing to report" scan — the empty state,
    // not a bare heading + timestamp.
    await waitFor(() => {
      expect(screen.getByText('未发现需关注事项')).toBeDefined();
    });
    // No group headings for the empty categories.
    expect(screen.queryByText(/矛盾发现/)).toBeNull();
    expect(screen.queryByText(/实体覆盖/)).toBeNull();
  });

  it('renders the raw report of an unstructured patrol log', async () => {
    mockListPatrolLogs.mockResolvedValue({
      items: [{ ...logSummary, finding_count: 0 }],
      total: 1,
    });
    mockGetPatrolLog.mockResolvedValue({
      ...logSummary,
      finding_count: 0,
      findings: { raw_output: '# 矛盾扫描巡逻报告\n\n未发现矛盾。' },
      dismissed_findings: [],
    });
    renderPage();

    const user = userEvent.setup();
    await waitFor(() => {
      expect(screen.getByText(/0 个发现/)).toBeDefined();
    });
    await user.click(screen.getByText(/0 个发现/));

    // The raw report is shown as-is under its own heading, not the empty state.
    await waitFor(() => {
      expect(screen.getByText('原始报告')).toBeDefined();
    });
    expect(screen.getByText(/# 矛盾扫描巡逻报告/)).toBeDefined();
    expect(screen.queryByText('未发现需关注事项')).toBeNull();
  });

  it('merges a daily pattern finding from the card', async () => {
    mockGetPatrolLog.mockResolvedValue({
      ...logSummary,
      findings: { pattern_matches: [patternFinding] },
      dismissed_findings: [],
    });
    renderPage();

    const user = userEvent.setup();
    await waitFor(() => {
      expect(screen.getByText(/1 个发现/)).toBeDefined();
    });
    await user.click(screen.getByText(/1 个发现/));

    await waitFor(() => {
      expect(screen.getByText('合并')).toBeDefined();
    });
    await user.click(screen.getByText('合并'));

    // The full finding reaches the merge endpoint.
    await waitFor(() => {
      expect(mockMergePatrolFinding).toHaveBeenCalledWith('log-1', patternFinding);
    });
    // The card flips to "已合并" so the pair cannot be re-merged.
    await waitFor(() => {
      expect(screen.getByText('已合并')).toBeDefined();
    });
    // Merging auto-dismisses the finding from the patrol view.
    await waitFor(() => {
      expect(mockDismissFinding).toHaveBeenCalledWith('log-1', expect.any(String));
    });
  });

  it('shows a merge error when the pair was already processed', async () => {
    mockGetPatrolLog.mockResolvedValue({
      ...logSummary,
      findings: { pattern_matches: [patternFinding] },
      dismissed_findings: [],
    });
    mockMergePatrolFinding.mockRejectedValue(new Error('409 Conflict: pair already merged'));
    renderPage();

    const user = userEvent.setup();
    await waitFor(() => {
      expect(screen.getByText(/1 个发现/)).toBeDefined();
    });
    await user.click(screen.getByText(/1 个发现/));

    await waitFor(() => {
      expect(screen.getByText('合并')).toBeDefined();
    });
    await user.click(screen.getByText('合并'));

    await waitFor(() => {
      expect(screen.getByText('合并失败：记忆可能已被处理或删除')).toBeDefined();
    });
  });

  it('keeps the merge label on a merely dismissed pattern finding', async () => {
    // Regression: the button read 已合并 whenever the finding was dismissed.
    // Dismissing is not merging — and since the backend has no merged flag
    // (both actions land in dismissed_findings), only the local merge state may
    // claim 已合并.
    mockGetPatrolLog.mockResolvedValue({
      ...logSummary,
      findings: { pattern_matches: [patternFinding] },
      dismissed_findings: [],
    });
    renderPage();

    const user = userEvent.setup();
    await waitFor(() => {
      expect(screen.getByText(/1 个发现/)).toBeDefined();
    });
    await user.click(screen.getByText(/1 个发现/));

    await waitFor(() => {
      expect(screen.getByText('忽略')).toBeDefined();
    });
    await user.click(screen.getByText('忽略'));

    // The finding is dismissed …
    await waitFor(() => {
      expect(screen.getByText('已忽略')).toBeDefined();
    });
    // … but nothing was merged, so the merge button must not claim otherwise.
    expect(screen.queryByText('已合并')).toBeNull();
    const mergeButton = screen.getByRole('button', { name: '合并' });
    // Still non-actionable — a dismissed finding cannot be merged.
    expect((mergeButton as HTMLButtonElement).disabled).toBe(true);
    expect(mockMergePatrolFinding).not.toHaveBeenCalled();
  });

  it('shows the failure reason of a failed patrol', async () => {
    // A failed run carries no findings — its reason lives in `error`.
    mockListPatrolLogs.mockResolvedValue({
      items: [{ ...logSummary, id: 'log-failed', status: 'failed', finding_count: 0 }],
      total: 1,
    });
    mockGetPatrolLog.mockResolvedValue({
      ...logSummary,
      id: 'log-failed',
      status: 'failed',
      error: 'LLM provider unavailable',
      findings: null,
      dismissed_findings: [],
    });
    renderPage();

    const user = userEvent.setup();
    await waitFor(() => {
      expect(screen.getByText(/0 个发现/)).toBeDefined();
    });
    await user.click(screen.getByText(/0 个发现/));

    // The reason is shown in the detail header, not swallowed as "未发现".
    await waitFor(() => {
      expect(screen.getByText(/失败原因：LLM provider unavailable/)).toBeDefined();
    });
  });

  it('polls an in-flight patrol until it reaches a terminal status', async () => {
    // Regression: nothing re-fetched after the initial load, so a run that was
    // `running` on screen stayed that way even after it had failed server-side.
    let status = 'running';
    mockListPatrolLogs.mockImplementation(async () => ({
      items: [todayLog({ id: 'log-run', patrol_type: 'daily', status })],
      total: 1,
    }));

    vi.useFakeTimers();
    try {
      renderPage();
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      expect(screen.getByText('运行中')).toBeDefined();

      // The patrol fails in the background — no user interaction at all.
      status = 'failed';
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2100);
      });

      expect(screen.getByText('失败')).toBeDefined();
      expect(screen.queryByText('运行中')).toBeNull();
    } finally {
      vi.useRealTimers();
    }
  });

  it('watches a manually triggered patrol through to its terminal status', async () => {
    // POST /trigger returns 202 *before* run_patrol inserts the log row, so the
    // first refresh legitimately sees nothing — the watch has to survive that
    // gap and keep polling once the row shows up.
    let items: ReturnType<typeof todayLog>[] = [];
    mockListPatrolLogs.mockImplementation(async () => ({
      items,
      total: items.length,
    }));

    vi.useFakeTimers();
    try {
      renderPage();
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0);
      });
      expect(screen.getByText(/暂无巡检记录/)).toBeDefined();

      // fireEvent rather than userEvent — no extra fake-timer wiring needed.
      await act(async () => {
        fireEvent.click(screen.getByRole('button', { name: /▶ 每日/ }));
      });
      expect(triggerPatrol).toHaveBeenCalledWith('daily');

      // The row lands a moment later, still running.
      items = [todayLog({ id: 'log-new', patrol_type: 'daily', status: 'running' })];
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2100);
      });
      expect(screen.getByText('运行中')).toBeDefined();

      // …and the watch keeps going until it turns terminal.
      items = [todayLog({ id: 'log-new', patrol_type: 'daily', status: 'failed' })];
      await act(async () => {
        await vi.advanceTimersByTimeAsync(2100);
      });
      expect(screen.getByText('失败')).toBeDefined();
    } finally {
      vi.useRealTimers();
    }
  });
});
