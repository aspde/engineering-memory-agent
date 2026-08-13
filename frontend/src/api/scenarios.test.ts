import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { listScenarios, runScenario } from './scenarios';

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

describe('listScenarios', () => {
  it('GETs /api/scenarios', async () => {
    fetchMock.mockResolvedValue(okJson([{ key: 'postmortem', status: 'active' }]));
    await expect(listScenarios()).resolves.toEqual([{ key: 'postmortem', status: 'active' }]);
    expect(fetchMock).toHaveBeenCalledWith('/api/scenarios', {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
  });
});

describe('runScenario', () => {
  it('POSTs params and an optional thread_id', async () => {
    fetchMock.mockResolvedValue(okJson({ result: '复盘结果' }));
    await runScenario('postmortem', { scope: 'db' }, 't-1');
    expect(fetchMock).toHaveBeenCalledWith('/api/scenarios/postmortem/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ params: { scope: 'db' }, thread_id: 't-1' }),
    });
  });

  it('omits thread_id when not supplied', async () => {
    fetchMock.mockResolvedValue(okJson({ result: 'ok' }));
    await runScenario('code_review');
    expect(fetchMock).toHaveBeenCalledWith('/api/scenarios/code_review/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ params: {}, thread_id: undefined }),
    });
  });
});
