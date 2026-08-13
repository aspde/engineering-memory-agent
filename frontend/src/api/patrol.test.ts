import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  dismissFinding,
  getPatrolLog,
  listPatrolLogs,
  queuePatrolConflict,
  triggerPatrol,
} from './patrol';

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

describe('triggerPatrol', () => {
  it('POSTs patrol_type with a default scope of all', async () => {
    fetchMock.mockResolvedValue(okJson({ patrol_id: 'p-1', status: 'started' }));
    await triggerPatrol('daily');
    expect(fetchMock).toHaveBeenCalledWith('/api/patrol/trigger', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ patrol_type: 'daily', scope: 'all' }),
    });
  });

  it('passes an explicit scope through', async () => {
    fetchMock.mockResolvedValue(okJson({ patrol_id: 'p-1', status: 'started' }));
    await triggerPatrol('weekly', 'postgresql');
    expect(fetchMock).toHaveBeenCalledWith('/api/patrol/trigger', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ patrol_type: 'weekly', scope: 'postgresql' }),
    });
  });
});

describe('listPatrolLogs', () => {
  it('GETs /api/patrol/logs with no query when no params are given', async () => {
    fetchMock.mockResolvedValue(okJson({ logs: [] }));
    await expect(listPatrolLogs()).resolves.toEqual({ logs: [] });
    expect(fetchMock).toHaveBeenCalledWith('/api/patrol/logs', {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
  });

  it('serializes pagination and type filters', async () => {
    fetchMock.mockResolvedValue(okJson({ logs: [] }));
    await listPatrolLogs({ limit: 20, offset: 40, patrol_type: 'weekly' });
    expect(fetchMock).toHaveBeenCalledWith(
      '/api/patrol/logs?limit=20&offset=40&patrol_type=weekly',
      expect.anything(),
    );
  });
});

describe('getPatrolLog', () => {
  it('GETs the log detail endpoint', async () => {
    fetchMock.mockResolvedValue(okJson({ id: 'p-2', findings: [] }));
    await expect(getPatrolLog('p-2')).resolves.toEqual({ id: 'p-2', findings: [] });
    expect(fetchMock).toHaveBeenCalledWith('/api/patrol/logs/p-2', {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
  });
});

describe('dismissFinding', () => {
  it('POSTs the finding_id to the dismiss endpoint', async () => {
    fetchMock.mockResolvedValue(okJson({ log_id: 'p-3', finding_id: 'f-1' }));
    await dismissFinding('p-3', 'f-1');
    expect(fetchMock).toHaveBeenCalledWith('/api/patrol/findings/p-3/dismiss', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ finding_id: 'f-1' }),
    });
  });
});

describe('queuePatrolConflict', () => {
  it('POSTs the full finding payload to the conflict endpoint', async () => {
    const finding = { id: 'f-2', content: '矛盾内容', memory_ids: ['a', 'b'] };
    fetchMock.mockResolvedValue(okJson({ log_id: 'p-4', conflict_id: 'c-1' }));
    await queuePatrolConflict('p-4', finding);
    expect(fetchMock).toHaveBeenCalledWith('/api/patrol/findings/p-4/conflict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(finding),
    });
  });
});
