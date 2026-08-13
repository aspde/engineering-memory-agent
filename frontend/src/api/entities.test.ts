import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getEntity, getEntityRelations, searchEntities } from './entities';

const fetchMock = vi.fn();

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock);
  // Pin the auth env so assertions don't depend on whether VITE_EMA_API_KEY
  // happens to be injected from a .env file at run time.
  vi.stubEnv('VITE_EMA_API_KEY', '');
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  fetchMock.mockReset();
});

function okJson(body: unknown) {
  return {
    ok: true,
    status: 200,
    statusText: 'OK',
    json: async () => body,
  };
}

describe('getEntity', () => {
  it('GETs the entity endpoint', async () => {
    fetchMock.mockResolvedValue(okJson({ id: 'e/1' }));
    await expect(getEntity('e/1')).resolves.toEqual({ id: 'e/1' });
    expect(fetchMock).toHaveBeenCalledWith('/api/entities/e/1', {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
  });
});

describe('getEntityRelations', () => {
  it('GETs the relations endpoint', async () => {
    fetchMock.mockResolvedValue(okJson({ entity: {}, related_entities: [], recent_memories: [] }));
    await getEntityRelations('e-2');
    expect(fetchMock).toHaveBeenCalledWith('/api/entities/e-2/relations', {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
  });
});

describe('searchEntities', () => {
  it('GETs the search endpoint with the query', async () => {
    fetchMock.mockResolvedValue(okJson({ results: [] }));
    await searchEntities('postgresql');
    expect(fetchMock).toHaveBeenCalledWith('/api/entities/search?q=postgresql', {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
  });

  it('adds a type filter when supplied and URL-encodes the query', async () => {
    fetchMock.mockResolvedValue(okJson({ results: [] }));
    await searchEntities('C++', 'technology');
    expect(fetchMock).toHaveBeenCalledWith('/api/entities/search?q=C%2B%2B&type=technology', {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
  });
});
