import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getConnectorLogs, listConnectors } from './connectors';

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

describe('listConnectors', () => {
  it('GETs /api/connectors', async () => {
    fetchMock.mockResolvedValue(okJson({ connectors: [] }));
    await expect(listConnectors()).resolves.toEqual({ connectors: [] });
    expect(fetchMock).toHaveBeenCalledWith('/api/connectors', {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
  });
});

describe('getConnectorLogs', () => {
  it('defaults to limit=50 offset=0 and URL-encodes the source', async () => {
    fetchMock.mockResolvedValue(okJson({ logs: [] }));
    await getConnectorLogs('ci/cd');
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/connectors/ci%2Fcd/logs?limit=50&offset=0',
      { method: 'GET', headers: { Accept: 'application/json' } },
    );
  });

  it('passes a custom limit and offset', async () => {
    fetchMock.mockResolvedValue(okJson({ logs: [] }));
    await getConnectorLogs('feishu', 100, 20);
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/connectors/feishu/logs?limit=100&offset=20',
      expect.anything(),
    );
  });
});
