import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  getStats,
  ingest,
  searchMemories,
  getMemory,
  deleteMemory,
} from './memory';

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

describe('getStats', () => {
  it('GETs /api/memory/stats', async () => {
    const body = { total_memories: 3 };
    fetchMock.mockResolvedValue(okJson(body));
    await expect(getStats()).resolves.toEqual(body);
    expect(fetchMock).toHaveBeenCalledWith('/api/memory/stats', {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
  });
});

describe('ingest', () => {
  it('POSTs document_id and content to /api/memory/ingest', async () => {
    fetchMock.mockResolvedValue(okJson({ document_id: 'doc.md', chunks_written: 2 }));
    await expect(ingest('doc.md', 'hello world')).resolves.toEqual({
      document_id: 'doc.md',
      chunks_written: 2,
    });
    expect(fetchMock).toHaveBeenCalledWith('/api/memory/ingest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ document_id: 'doc.md', content: 'hello world' }),
    });
  });
});

describe('searchMemories', () => {
  it('POSTs the query with the camelCase topK mapped to top_k', async () => {
    fetchMock.mockResolvedValue(okJson({ results: [] }));
    await searchMemories('pgvector', 7);
    expect(fetchMock).toHaveBeenCalledWith('/api/memory/memories/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ query: 'pgvector', top_k: 7 }),
    });
  });
});

describe('getMemory', () => {
  it('GETs the memory endpoint with the id URL-encoded', async () => {
    fetchMock.mockResolvedValue(okJson({ id: 'm/1' }));
    await expect(getMemory('m/1')).resolves.toEqual({ id: 'm/1' });
    expect(fetchMock).toHaveBeenCalledWith('/api/memory/memories/m%2F1', {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
  });
});

describe('deleteMemory', () => {
  it('DELETEs the memory endpoint with the id URL-encoded', async () => {
    fetchMock.mockResolvedValue(okJson({ id: 'm-2', deleted: true }));
    await expect(deleteMemory('m-2')).resolves.toEqual({ id: 'm-2', deleted: true });
    expect(fetchMock).toHaveBeenCalledWith('/api/memory/memories/m-2', {
      method: 'DELETE',
      headers: { Accept: 'application/json' },
    });
  });
});
