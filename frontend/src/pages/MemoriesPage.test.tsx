import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { ApiError } from '../api/client';
import { useAppDispatch, useAppState } from '../context/AppContext';
import { useMemories } from '../hooks/useMemories';
import { listPatrolLogs } from '../api/patrol';
import MemoriesPage from './MemoriesPage';

// Mock the context and hooks — the page wires them to state, not a server.
vi.mock('../context/AppContext', () => ({
  useAppState: vi.fn(),
  useAppDispatch: vi.fn(),
}));
vi.mock('../hooks/useMemories', () => ({ useMemories: vi.fn() }));
vi.mock('../api/patrol', () => ({ listPatrolLogs: vi.fn() }));

type Mock = ReturnType<typeof vi.fn>;

let state: { memFilterId: string | null };
let dispatch: Mock;
let fetchStats: Mock;
let getMemoryById: Mock;
let deleteMemoryById: Mock;

beforeEach(() => {
  vi.clearAllMocks();
  state = { memFilterId: null };
  (useAppState as unknown as Mock).mockReturnValue(state);
  dispatch = vi.fn();
  (useAppDispatch as unknown as Mock).mockReturnValue(dispatch);

  fetchStats = vi.fn().mockResolvedValue(null);
  getMemoryById = vi.fn().mockRejectedValue({ status: 404 });
  deleteMemoryById = vi.fn().mockResolvedValue({ id: 'm-1', deleted: true });

  (useMemories as unknown as Mock).mockReturnValue({
    stats: null,
    isLoading: false,
    error: null,
    fetchStats,
    searchResults: null,
    isSearching: false,
    searchError: null,
    search: vi.fn(),
    ingestText: vi.fn(),
    ingestFile: vi.fn(),
    getMemoryById,
    deleteMemoryById,
    removeSearchResult: vi.fn(),
  });

  // PatrolBrief renders a skeleton while loading, then the empty state.
  (listPatrolLogs as unknown as Mock).mockResolvedValue({ items: [] });
});

afterEach(() => {
  vi.restoreAllMocks();
});

function renderPage() {
  return render(
    <MemoryRouter>
      <MemoriesPage />
    </MemoryRouter>,
  );
}

describe('MemoriesPage', () => {
  it('renders the dashboard tab with the stats empty state and patrol brief', async () => {
    renderPage();

    expect(screen.getByRole('heading', { name: /记忆库/ })).toBeInTheDocument();
    // The dashboard tab is active by default → fetchStats fires.
    expect(fetchStats).toHaveBeenCalled();
    // PatrolBrief with no logs shows the not-yet-run state.
    expect(await screen.findByText(/今日巡检尚未执行/)).toBeInTheDocument();
  });

  it('loads and renders a memory jumped from the chat page', async () => {
    getMemoryById.mockResolvedValue({
      id: 'm-1',
      source_type: 'conversation',
      summary: '跳转过来的记忆内容',
      entities: [],
      relations: [],
      recall_count: 0,
      meta: {},
      created_at: '2026-08-01T00:00:00Z',
    });
    state.memFilterId = 'm-1';

    renderPage();

    await waitFor(() => expect(getMemoryById).toHaveBeenCalledWith('m-1'));
    expect(await screen.findByText('跳转过来的记忆内容')).toBeInTheDocument();
  });

  it('shows a not-found notice when the jumped memory 404s', async () => {
    getMemoryById.mockRejectedValue(new ApiError(404, 'Not Found', 'missing'));
    state.memFilterId = 'm-missing';

    renderPage();

    expect(await screen.findByText(/未找到该记忆/)).toBeInTheDocument();
  });

  it('clears the filter via the clear button', async () => {
    state.memFilterId = 'm-1';
    renderPage();

    const user = userEvent.setup();
    await user.click(await screen.findByRole('button', { name: '清除筛选' }));

    expect(dispatch).toHaveBeenCalledWith({ type: 'CLEAR_MEM_FILTER' });
  });

  it('deletes a jumped memory and refreshes stats', async () => {
    getMemoryById.mockResolvedValue({
      id: 'm-1',
      source_type: 'conversation',
      summary: '要删除的记忆',
      entities: [],
      relations: [],
      recall_count: 0,
      meta: {},
      created_at: '2026-08-01T00:00:00Z',
    });
    state.memFilterId = 'm-1';

    renderPage();
    await screen.findByText('要删除的记忆');

    const user = userEvent.setup();
    // Deletion is two-step: the trash icon opens a confirm row, then "删除".
    await user.click(screen.getByTitle('删除此记忆'));
    await user.click(screen.getByRole('button', { name: '删除' }));

    await waitFor(() => expect(deleteMemoryById).toHaveBeenCalledWith('m-1'));
    await waitFor(() => expect(dispatch).toHaveBeenCalledWith({ type: 'CLEAR_MEM_FILTER' }));
    expect(fetchStats).toHaveBeenCalled();
  });
});
