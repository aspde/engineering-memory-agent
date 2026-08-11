import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { probeCapability } from './capabilities';

const fetchMock = vi.fn();

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
  fetchMock.mockReset();
});

describe('probeCapability', () => {
  it('returns true when the endpoint responds 200', async () => {
    fetchMock.mockResolvedValue(new Response('[]', { status: 200 }));
    await expect(probeCapability('/api/connectors')).resolves.toBe(true);
  });

  it('returns false when the endpoint returns 404 (route not mounted)', async () => {
    fetchMock.mockResolvedValue(new Response('Not Found', { status: 404 }));
    await expect(probeCapability('/api/patrol/logs')).resolves.toBe(false);
  });

  it('returns true on 401 — the route exists but the key guard rejected us', async () => {
    fetchMock.mockResolvedValue(new Response('Not authenticated', { status: 401 }));
    await expect(probeCapability('/api/connectors')).resolves.toBe(true);
  });

  it('returns true on 5xx — an erroring backend must not hide an enabled feature', async () => {
    fetchMock.mockResolvedValue(new Response('boom', { status: 500 }));
    await expect(probeCapability('/api/connectors')).resolves.toBe(true);
  });

  it('returns true on network failure — a broken backend must not hide the entry', async () => {
    fetchMock.mockRejectedValue(new Error('Network error'));
    await expect(probeCapability('/api/connectors')).resolves.toBe(true);
  });
});
