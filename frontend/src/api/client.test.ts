import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { parseSSELine, normalizeSSEEvent, apiSSE, apiGet, apiPost, ApiError } from './client';

const fetchMock = vi.fn();

beforeEach(() => {
  vi.stubGlobal('fetch', fetchMock);
  // Pin the auth env so the default-request assertions don't depend on
  // whether VITE_EMA_API_KEY happens to be injected from a .env file.
  vi.stubEnv('VITE_EMA_API_KEY', '');
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  fetchMock.mockReset();
});

/** Build a ReadableStream of text chunks for SSE tests. */
function streamFromChunks(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) {
        controller.enqueue(encoder.encode(chunk));
      }
      controller.close();
    },
  });
}

describe('parseSSELine', () => {
  it('parses a valid "data: {json}" line', () => {
    expect(parseSSELine('data: {"type":"token","content":"hi"}')).toEqual({
      type: 'token',
      content: 'hi',
    });
  });

  it('returns null for "data: " with no payload', () => {
    expect(parseSSELine('data: ')).toBeNull();
  });

  it('returns null for lines not starting with "data: "', () => {
    expect(parseSSELine('event: token')).toBeNull();
    expect(parseSSELine('')).toBeNull();
  });

  it('returns null for malformed JSON', () => {
    expect(parseSSELine('data: {not json}')).toBeNull();
  });

  it('strips a trailing carriage return', () => {
    expect(parseSSELine('data: {"type":"token","content":"hi"}\r')).toEqual({
      type: 'token',
      content: 'hi',
    });
  });
});

describe('normalizeSSEEvent', () => {
  it('normalizes a token event', () => {
    expect(normalizeSSEEvent({ type: 'token', content: 'hello' })).toEqual({
      type: 'token',
      content: 'hello',
    });
  });

  it('normalizes a node event', () => {
    expect(normalizeSSEEvent({ type: 'node', node: 'reflect' })).toEqual({
      type: 'node',
      node: 'reflect',
    });
  });

  it('normalizes an interrupt event', () => {
    expect(normalizeSSEEvent({ type: 'interrupt', data: { type: 'approval' } })).toEqual({
      type: 'interrupt',
      data: { type: 'approval' },
    });
  });

  it('normalizes a meta event', () => {
    const toolCalls = [{ tool: 'search', content: 'q' }];
    const sources = [{ type: 'memory', summary: 's' }];
    const memoryWrite = { action: 'inserted', summary: '端口改为 8080' };
    expect(
      normalizeSSEEvent({ type: 'meta', tool_calls: toolCalls, sources, memory_write: memoryWrite }),
    ).toEqual({
      type: 'meta',
      tool_calls: toolCalls,
      sources,
      memory_write: memoryWrite,
    });
  });

  it('normalizes an error event', () => {
    expect(normalizeSSEEvent({ type: 'error', message: 'boom' })).toEqual({
      type: 'error',
      message: 'boom',
    });
  });

  it('normalizes a done event', () => {
    expect(normalizeSSEEvent({ type: 'done' })).toEqual({ type: 'done' });
  });

  it('returns an error event for an unknown type', () => {
    expect(normalizeSSEEvent({ type: 'weird' })).toEqual({
      type: 'error',
      message: 'Unknown SSE event type: weird',
    });
  });

  it('returns an error event when type is missing', () => {
    expect(normalizeSSEEvent({})).toEqual({
      type: 'error',
      message: 'Unknown SSE event type: undefined',
    });
  });

  it('defaults missing tool_calls and sources to empty arrays in meta', () => {
    expect(normalizeSSEEvent({ type: 'meta' })).toEqual({
      type: 'meta',
      tool_calls: [],
      sources: [],
      memory_write: null,
    });
  });
});

describe('apiGet', () => {
  it('fetches and parses JSON on success', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ threads: [] }),
    });
    const result = await apiGet<{ threads: unknown[] }>('/api/threads');
    expect(result).toEqual({ threads: [] });
    expect(fetchMock).toHaveBeenCalledWith('/api/threads', {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
  });
  it('throws ApiError on a non-2xx response', async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 404,
      statusText: 'Not Found',
      json: async () => ({ detail: 'missing' }),
    });
    await expect(apiGet('/api/missing')).rejects.toBeInstanceOf(ApiError);
    await expect(apiGet('/api/missing')).rejects.toMatchObject({ status: 404 });
  });

  it('throws Error on a network failure', async () => {
    fetchMock.mockRejectedValue(new TypeError('Failed to fetch'));
    await expect(apiGet('/api/x')).rejects.toThrow('Network error fetching /api/x');
  });
});

describe('apiPost', () => {
  it('posts a JSON body and returns the parsed response', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({ thread_id: 't-1' }),
    });
    const result = await apiPost<{ thread_id: string }>('/api/chat', { message: 'hi' });
    expect(result).toEqual({ thread_id: 't-1' });
    expect(fetchMock).toHaveBeenCalledWith('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({ message: 'hi' }),
    });
  });
});

describe('apiSSE', () => {
  it('yields events from a ReadableStream', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      body: streamFromChunks([
        'data: {"type":"token","content":"a"}\n\n',
        'data: {"type":"token","content":"b"}\n\n',
      ]),
    });
    const events: unknown[] = [];
    for await (const event of apiSSE('/api/chat', { message: 'hi' })) {
      events.push(event);
    }
    expect(events).toEqual([
      { type: 'token', content: 'a' },
      { type: 'token', content: 'b' },
    ]);
  });

  it('skips malformed and empty data lines', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      body: streamFromChunks([
        'data: \n\n',
        'not a data line\n\n',
        'data: {bad json}\n\n',
        'data: {"type":"done"}\n\n',
      ]),
    });
    const events: unknown[] = [];
    for await (const event of apiSSE('/api/chat', {})) {
      events.push(event);
    }
    expect(events).toEqual([{ type: 'done' }]);
  });

  it('flushes trailing content without a final newline', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      body: streamFromChunks(['data: {"type":"token","content":"tail"}']),
    });
    const events: unknown[] = [];
    for await (const event of apiSSE('/api/chat', {})) {
      events.push(event);
    }
    expect(events).toEqual([{ type: 'token', content: 'tail' }]);
  });

  it('throws ApiError on a non-2xx response', async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      body: streamFromChunks(['data: {"type":"done"}']),
      json: async () => ({ detail: 'boom' }),
    });
    await expect(
      (async () => {
        for await (const _event of apiSSE('/api/chat', {})) {
          // no-op
        }
      })(),
    ).rejects.toBeInstanceOf(ApiError);
  });
});

describe('auth headers', () => {
  it('omits Authorization when VITE_EMA_API_KEY is not configured', async () => {
    vi.stubEnv('VITE_EMA_API_KEY', '');
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({}),
    });
    await apiGet('/api/x');
    expect(fetchMock).toHaveBeenCalledWith('/api/x', {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
  });

  it('sends Authorization: Bearer when VITE_EMA_API_KEY is set', async () => {
    vi.stubEnv('VITE_EMA_API_KEY', 'test-key');
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      json: async () => ({}),
    });
    await apiGet('/api/x');
    expect(fetchMock).toHaveBeenCalledWith('/api/x', {
      method: 'GET',
      headers: { Accept: 'application/json', Authorization: 'Bearer test-key' },
    });
  });

  it('carries the auth header on POST and SSE requests too', async () => {
    vi.stubEnv('VITE_EMA_API_KEY', 'test-key');
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      body: streamFromChunks(['data: {"type":"done"}\n\n']),
      json: async () => ({}),
    });
    await apiPost('/api/x', { a: 1 });
    expect(fetchMock).toHaveBeenCalledWith('/api/x', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
        Authorization: 'Bearer test-key',
      },
      body: JSON.stringify({ a: 1 }),
    });

    fetchMock.mockClear();
    for await (const _event of apiSSE('/api/chat', {})) {
      // consume
    }
    expect(fetchMock).toHaveBeenCalledWith('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        Authorization: 'Bearer test-key',
      },
      body: JSON.stringify({}),
    });
  });
});
