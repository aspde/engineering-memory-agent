import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter } from 'react-router-dom';
import { getEntityRelations, searchEntities } from '../api/entities';
import { getMemory } from '../api/memory';
import EntityGraphPage from './EntityGraphPage';

// Mock the API modules — the page is a pure client of the entity graph API.
vi.mock('../api/entities', () => ({
  searchEntities: vi.fn(),
  getEntityRelations: vi.fn(),
}));
vi.mock('../api/memory', () => ({ getMemory: vi.fn() }));

type Mock = ReturnType<typeof vi.fn>;

const ENTITY = { id: 'e-1', canonical_name: 'PostgreSQL', type: 'technology', memory_count: 3 };

const RELATIONS = {
  entity: ENTITY,
  related_entities: [
    { entity_id: 'e-2', name: 'pgvector', type: 'technology', relation_type: 'depends_on', memory_count: 2 },
  ],
  recent_memories: [
    { memory_id: 'm-1', summary: 'PostgreSQL 连接池建议', source_type: 'conversation', created_at: '2026-08-01T00:00:00Z' },
  ],
};

function renderPage(route = '/') {
  return render(
    <MemoryRouter initialEntries={[route]}>
      <EntityGraphPage />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe('EntityGraphPage', () => {
  it('shows the empty state before any search', () => {
    renderPage();

    expect(screen.getByText('搜索实体名称以开始探索知识图谱')).toBeInTheDocument();
  });

  it('searches entities and lists the results', async () => {
    (searchEntities as unknown as Mock).mockResolvedValue({ results: [ENTITY] });
    const user = userEvent.setup();
    renderPage();

    await user.type(screen.getByPlaceholderText('搜索实体名称…'), 'postgres');
    await user.click(screen.getByRole('button', { name: '搜索' }));

    await waitFor(() => expect(searchEntities).toHaveBeenCalledWith('postgres'));
    expect(await screen.findByText('PostgreSQL')).toBeInTheDocument();
  });

  it('loads and renders the graph when an entity is selected', async () => {
    (searchEntities as unknown as Mock).mockResolvedValue({ results: [ENTITY] });
    (getEntityRelations as unknown as Mock).mockResolvedValue(RELATIONS);
    const user = userEvent.setup();
    renderPage();

    await user.type(screen.getByPlaceholderText('搜索实体名称…'), 'postgres');
    await user.click(screen.getByRole('button', { name: '搜索' }));
    await user.click(await screen.findByText('PostgreSQL'));

    await waitFor(() => expect(getEntityRelations).toHaveBeenCalledWith('e-1'));
    // Center node name appears in the graph panel.
    expect(await screen.findByText('PostgreSQL', { selector: 'h2' })).toBeInTheDocument();
    // The related entity and recent memory render.
    expect(await screen.findByText('PostgreSQL 连接池建议')).toBeInTheDocument();
  });

  it('shows an error message when the search fails', async () => {
    (searchEntities as unknown as Mock).mockRejectedValue(new Error('network'));
    const user = userEvent.setup();
    renderPage();

    await user.type(screen.getByPlaceholderText('搜索实体名称…'), 'x');
    await user.click(screen.getByRole('button', { name: '搜索' }));

    expect(await screen.findByText('搜索实体失败')).toBeInTheDocument();
  });

  it('auto-loads the graph from the ?entity= query param', async () => {
    (searchEntities as unknown as Mock).mockResolvedValue({ results: [ENTITY] });
    (getEntityRelations as unknown as Mock).mockResolvedValue(RELATIONS);

    renderPage('/?entity=PostgreSQL');

    await waitFor(() => expect(searchEntities).toHaveBeenCalledWith('PostgreSQL'));
    await waitFor(() => expect(getEntityRelations).toHaveBeenCalledWith('e-1'));
    expect(await screen.findByText('PostgreSQL 连接池建议')).toBeInTheDocument();
  });

  it('loads memory details when a recent memory is clicked', async () => {
    (searchEntities as unknown as Mock).mockResolvedValue({ results: [ENTITY] });
    (getEntityRelations as unknown as Mock).mockResolvedValue(RELATIONS);
    (getMemory as unknown as Mock).mockResolvedValue({
      id: 'm-1',
      source_type: 'conversation',
      summary: '完整的记忆详情',
      entities: [],
      relations: [],
      recall_count: 0,
      meta: {},
      created_at: '2026-08-01T00:00:00Z',
    });
    const user = userEvent.setup();
    renderPage();

    await user.type(screen.getByPlaceholderText('搜索实体名称…'), 'postgres');
    await user.click(screen.getByRole('button', { name: '搜索' }));
    await user.click(await screen.findByText('PostgreSQL'));
    await user.click(await screen.findByText('PostgreSQL 连接池建议'));

    await waitFor(() => expect(getMemory).toHaveBeenCalledWith('m-1'));
    expect(await screen.findByText('完整的记忆详情')).toBeInTheDocument();
  });
});
