import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getConflicts, reopenConflict, resolveConflict } from './conflicts';

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

describe('getConflicts', () => {
  it('GETs /api/conflicts with no query when no params are given', async () => {
    fetchMock.mockResolvedValue(okJson([]));
    await expect(getConflicts()).resolves.toEqual([]);
    expect(fetchMock).toHaveBeenCalledWith('/api/conflicts', {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
  });

  it('adds query params only for the filters supplied', async () => {
    fetchMock.mockResolvedValue(okJson([]));
    await getConflicts({ conflict_type: 'patrol', status: 'resolved' });
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/conflicts?conflict_type=patrol&status=resolved',
      expect.anything(),
    );
  });

  it('omits the query string when a filter value is empty', async () => {
    fetchMock.mockResolvedValue(okJson([]));
    await getConflicts({ status: '' });
    expect(fetchMock).toHaveBeenCalledWith('/api/conflicts', expect.anything());
  });
});

describe('resolveConflict', () => {
  it('POSTs the resolution to the conflict endpoint', async () => {
    fetchMock.mockResolvedValue(okJson({ id: 'c-1', resolution: 'merge', outcome: {} }));
    await resolveConflict('c/1', 'merge');
    expect(fetchMock).toHaveBeenCalledWith('/api/conflicts/c%2F1/resolve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ resolution: 'merge' }),
    });
  });
});

describe('reopenConflict', () => {
  it('POSTs an empty body to the reopen endpoint', async () => {
    fetchMock.mockResolvedValue(okJson({ id: 'c-2', status: 'pending' }));
    await reopenConflict('c-2');
    expect(fetchMock).toHaveBeenCalledWith('/api/conflicts/c-2/reopen', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: '{}',
    });
  });
});
