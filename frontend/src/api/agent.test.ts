import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  listThreads,
  getThreadMessages,
  chatNonStream,
  chatStream,
  deleteThread,
} from './agent';

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

describe('listThreads', () => {
  it('GETs /api/agent/threads and returns the parsed list', async () => {
    const threads = [{ thread_id: 't-1', title: '对话一' }];
    fetchMock.mockResolvedValue(okJson(threads));
    await expect(listThreads()).resolves.toEqual(threads);
    expect(fetchMock).toHaveBeenCalledWith('/api/agent/threads', {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
  });
});

describe('getThreadMessages', () => {
  it('GETs the thread endpoint with the id URL-encoded', async () => {
    const body = { thread_id: 't/1', messages: [] };
    fetchMock.mockResolvedValue(okJson(body));
    await expect(getThreadMessages('t/1')).resolves.toEqual(body);
    expect(fetchMock).toHaveBeenCalledWith('/api/agent/thread/t%2F1', {
      method: 'GET',
      headers: { Accept: 'application/json' },
    });
  });
});

describe('chatNonStream', () => {
  it('POSTs the ChatRequest to /api/agent/chat', async () => {
    const req = { message: 'hi', thread_id: 't-1' };
    const resp = { thread_id: 't-1', status: 'completed' as const, response: 'hey', interrupt: null, tool_calls: [], sources: [] };
    fetchMock.mockResolvedValue(okJson(resp));
    await expect(chatNonStream(req)).resolves.toEqual(resp);
    expect(fetchMock).toHaveBeenCalledWith('/api/agent/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(req),
    });
  });
});

describe('chatStream', () => {
  it('POSTs to /api/agent/chat/stream and yields parsed SSE events', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      statusText: 'OK',
      body: streamFromChunks([
        'data: {"type":"token","content":"a"}\n\n',
        'data: {"type":"done"}\n\n',
      ]),
    });
    const req = { message: 'hi', thread_id: 't-1' };
    const events: unknown[] = [];
    for await (const event of chatStream(req)) {
      events.push(event);
    }
    expect(events).toEqual([{ type: 'token', content: 'a' }, { type: 'done' }]);
    expect(fetchMock).toHaveBeenCalledWith('/api/agent/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(req),
    });
  });
});

describe('deleteThread', () => {
  it('DELETEs the thread endpoint with the id URL-encoded', async () => {
    fetchMock.mockResolvedValue(okJson({ thread_id: 't-2', deleted: true }));
    await expect(deleteThread('t-2')).resolves.toEqual({ thread_id: 't-2', deleted: true });
    expect(fetchMock).toHaveBeenCalledWith('/api/agent/thread/t-2', {
      method: 'DELETE',
      headers: { Accept: 'application/json' },
    });
  });
});
