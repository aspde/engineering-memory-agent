import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import ConnectorsPage from './ConnectorsPage';

// Mock the connectors API
vi.mock('../api/connectors', () => ({
  listConnectors: vi.fn(),
  getConnectorLogs: vi.fn(),
}));

import { listConnectors, getConnectorLogs } from '../api/connectors';

const mockListConnectors = listConnectors as ReturnType<typeof vi.fn>;
const mockGetConnectorLogs = getConnectorLogs as ReturnType<typeof vi.fn>;

function renderPage() {
  return render(
    <MemoryRouter>
      <ConnectorsPage />
    </MemoryRouter>,
  );
}

describe('ConnectorsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows loading state initially', () => {
    mockListConnectors.mockReturnValue(new Promise(() => {})); // never resolves
    renderPage();
    expect(screen.getByText('🔌 连接器')).toBeDefined();
  });

  it('renders connector list when loaded', async () => {
    mockListConnectors.mockResolvedValue({
      connectors: [
        {
          source_type: 'jira',
          display_name: 'Jira',
          status: 'active',
          batch_mode: 'pending',
        },
        {
          source_type: 'ci',
          display_name: 'CI/CD',
          status: 'pending',
          batch_mode: 'pending',
        },
      ],
    });

    renderPage();

    await waitFor(() => {
      expect(screen.getByText('Jira')).toBeDefined();
    });
    expect(screen.getByText('CI/CD')).toBeDefined();
    expect(screen.getByText('已激活')).toBeDefined();
    expect(screen.getByText('待配置')).toBeDefined();
  });

  it('shows empty state when no connectors registered', async () => {
    mockListConnectors.mockResolvedValue({ connectors: [] });

    renderPage();

    await waitFor(() => {
      expect(screen.getByText('暂无已注册的连接器')).toBeDefined();
    });
  });

  it('shows error message on API failure', async () => {
    mockListConnectors.mockRejectedValue(new Error('Network error'));

    renderPage();

    await waitFor(() => {
      expect(screen.getByText(/Network error/)).toBeDefined();
    });
  });

  it('can open and view connector logs', async () => {
    mockListConnectors.mockResolvedValue({
      connectors: [
        {
          source_type: 'jira',
          display_name: 'Jira',
          status: 'active',
          batch_mode: 'pending',
        },
      ],
    });
    mockGetConnectorLogs.mockResolvedValue({
      logs: [
        {
          id: 'log-1',
          source: 'jira',
          event_type: 'issue.resolved',
          status: 'processed',
          payload_summary: 'EMA-42 fixed',
          memory_id: null,
          error: null,
          created_at: '2026-08-04T10:00:00Z',
        },
      ],
    });

    renderPage();

    await waitFor(() => {
      expect(screen.getByText('Jira')).toBeDefined();
    });

    const user = userEvent.setup();
    await user.click(screen.getByText('查看投递日志'));

    await waitFor(() => {
      expect(screen.getByText('EMA-42 fixed')).toBeDefined();
    });
  });

  it('handles empty log results', async () => {
    mockListConnectors.mockResolvedValue({
      connectors: [
        {
          source_type: 'jira',
          display_name: 'Jira',
          status: 'active',
          batch_mode: 'pending',
        },
      ],
    });
    mockGetConnectorLogs.mockResolvedValue({ logs: [] });

    renderPage();

    await waitFor(() => {
      expect(screen.getByText('Jira')).toBeDefined();
    });

    const user = userEvent.setup();
    await user.click(screen.getByText('查看投递日志'));

    await waitFor(() => {
      expect(screen.getByText('暂无投递记录')).toBeDefined();
    });
  });
});
