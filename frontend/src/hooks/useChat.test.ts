import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { chatNonStream, chatStream } from '../api/agent';
import { useAppDispatch, useAppState } from '../context/AppContext';
import type { SSEEvent } from '../types';
import { useChat } from './useChat';

// Mock the context
vi.mock('../context/AppContext', () => ({
  useAppState: vi.fn(),
  useAppDispatch: vi.fn(),
}));

// Mock the API
vi.mock('../api/agent', () => ({
  chatStream: vi.fn(),
  chatNonStream: vi.fn(),
}));

type Mock = ReturnType<typeof vi.fn>;

/** Build an SSE stream that emits `events` and then stays open until `resolve` is called. */
function makeStream(events: SSEEvent[]) {
  let resolve!: () => void;
  const gate = new Promise<void>((r) => (resolve = r));
  const stream = (async function* (): AsyncGenerator<SSEEvent> {
    for (const e of events) yield e;
    await gate;
  })();
  return { stream, resolve };
}

/** Build an SSE stream that emits `events` and completes immediately. */
async function* mockSSE(events: SSEEvent[]): AsyncGenerator<SSEEvent> {
  for (const e of events) yield e;
}

let dispatch: Mock;

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers();
  // Default state: threadId='test-thread', isStreaming=false
  (useAppState as unknown as Mock).mockReturnValue({
    threadId: 'test-thread',
    isStreaming: false,
  });
  dispatch = vi.fn();
  (useAppDispatch as unknown as Mock).mockReturnValue(dispatch);
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('useChat', () => {
  describe('sendMessage', () => {
    it('dispatches ADD_MESSAGE for the user message and an empty assistant placeholder immediately', async () => {
      const { stream, resolve } = makeStream([]);
      (chatStream as unknown as Mock).mockReturnValue(stream);

      const { result } = renderHook(() => useChat());

      let sendPromise: Promise<void>;
      await act(async () => {
        sendPromise = result.current.sendMessage('hello');
      });

      // Stream is still open — these dispatches have already happened.
      expect(dispatch).toHaveBeenCalledWith({ type: 'SET_STREAMING', isStreaming: true });
      expect(dispatch).toHaveBeenCalledWith({
        type: 'ADD_MESSAGE',
        message: { role: 'user', content: 'hello' },
      });
      expect(dispatch).toHaveBeenCalledWith({ type: 'SET_LOADED_THREAD', threadId: 'test-thread' });
      expect(dispatch).toHaveBeenCalledWith({
        type: 'ADD_MESSAGE',
        message: { role: 'assistant', content: '' },
      });

      act(() => resolve());
      await act(async () => {
        await sendPromise;
      });
    });

    it('guards against empty/whitespace input — does nothing', async () => {
      const { result } = renderHook(() => useChat());
      dispatch.mockClear(); // clear the mount effect's SET_STREAMING call

      await act(async () => {
        await result.current.sendMessage('   ');
      });

      expect(dispatch).not.toHaveBeenCalled();
      expect(chatStream).not.toHaveBeenCalled();
    });

    it('batches token events into a buffer and flushes on the 50ms timer', async () => {
      const { stream, resolve } = makeStream([
        { type: 'token', content: 'Hel' },
        { type: 'token', content: 'lo' },
      ]);
      (chatStream as unknown as Mock).mockReturnValue(stream);

      const { result } = renderHook(() => useChat());

      let sendPromise: Promise<void>;
      await act(async () => {
        sendPromise = result.current.sendMessage('hello');
      });

      // Tokens are buffered but not yet flushed (the interval has not fired).
      expect(dispatch).not.toHaveBeenCalledWith(
        expect.objectContaining({ type: 'UPDATE_LAST_MESSAGE', appendContent: 'Hello' }),
      );

      act(() => {
        vi.advanceTimersByTime(50);
      });

      expect(dispatch).toHaveBeenCalledWith({
        type: 'UPDATE_LAST_MESSAGE',
        appendContent: 'Hello',
      });

      act(() => resolve());
      await act(async () => {
        await sendPromise;
      });
    });

    it('appends a status label for non-generate_final nodes', async () => {
      (chatStream as unknown as Mock).mockReturnValue(mockSSE([{ type: 'node', node: 'call_llm' }]));

      const { result } = renderHook(() => useChat());
      await act(async () => {
        await result.current.sendMessage('hello');
      });

      expect(dispatch).toHaveBeenCalledWith({
        type: 'UPDATE_LAST_MESSAGE',
        appendContent: '> 思考中…\n',
      });
    });

    it('skips the generate_final node without appending a label', async () => {
      (chatStream as unknown as Mock).mockReturnValue(
        mockSSE([{ type: 'node', node: 'generate_final' }]),
      );

      const { result } = renderHook(() => useChat());
      await act(async () => {
        await result.current.sendMessage('hello');
      });

      expect(dispatch).not.toHaveBeenCalledWith(
        expect.objectContaining({ type: 'UPDATE_LAST_MESSAGE' }),
      );
    });

    it('deduplicates repeated node labels', async () => {
      (chatStream as unknown as Mock).mockReturnValue(
        mockSSE([
          { type: 'node', node: 'tools' },
          { type: 'node', node: 'tools' },
        ]),
      );

      const { result } = renderHook(() => useChat());
      await act(async () => {
        await result.current.sendMessage('hello');
      });

      const updateCalls = dispatch.mock.calls.filter((c) => c[0]?.type === 'UPDATE_LAST_MESSAGE');
      expect(updateCalls).toHaveLength(1);
      expect(dispatch).toHaveBeenCalledWith({
        type: 'UPDATE_LAST_MESSAGE',
        appendContent: '> 执行工具…\n',
      });
    });

    it('flushes tokens first, dispatches SET_INTERRUPT, adds a system message, and returns early on interrupt', async () => {
      (chatStream as unknown as Mock).mockReturnValue(
        mockSSE([
          { type: 'token', content: 'partial' },
          { type: 'interrupt', data: { type: 'conflict' } },
          { type: 'token', content: 'IGNORED' },
        ]),
      );

      const { result } = renderHook(() => useChat());
      await act(async () => {
        await result.current.sendMessage('hello');
      });

      // The token flush happens before the interrupt is surfaced.
      const calls = dispatch.mock.calls.map((c) => c[0]);
      const flushIdx = calls.findIndex(
        (a) => a?.type === 'UPDATE_LAST_MESSAGE' && a.appendContent === 'partial',
      );
      const interruptIdx = calls.findIndex((a) => a?.type === 'SET_INTERRUPT');
      expect(flushIdx).toBeGreaterThan(-1);
      expect(interruptIdx).toBeGreaterThan(flushIdx);

      expect(dispatch).toHaveBeenCalledWith({ type: 'SET_INTERRUPT', interrupt: { type: 'conflict' } });
      expect(dispatch).toHaveBeenCalledWith({
        type: 'ADD_MESSAGE',
        message: { role: 'system', content: '检测到记忆冲突，请选择如何解决。' },
      });

      // Returns early — the trailing token is never processed.
      expect(dispatch).not.toHaveBeenCalledWith(
        expect.objectContaining({
          type: 'UPDATE_LAST_MESSAGE',
          appendContent: expect.stringContaining('IGNORED'),
        }),
      );
    });

    it('flushes tokens and appends the error text on an error event', async () => {
      (chatStream as unknown as Mock).mockReturnValue(
        mockSSE([
          { type: 'token', content: 'Hi' },
          { type: 'error', message: 'boom' },
        ]),
      );

      const { result } = renderHook(() => useChat());
      await act(async () => {
        await result.current.sendMessage('hello');
      });

      expect(dispatch).toHaveBeenCalledWith({ type: 'UPDATE_LAST_MESSAGE', appendContent: 'Hi' });
      expect(dispatch).toHaveBeenCalledWith({
        type: 'UPDATE_LAST_MESSAGE',
        appendContent: '\n\n错误: boom',
      });
    });

    it('dispatches UPDATE_LAST_MESSAGE with meta on a meta event', async () => {
      const toolCalls = [{ tool: 'search', content: 'x' }];
      const sources = [{ type: 'memory' as const, id: 'm1' }];
      (chatStream as unknown as Mock).mockReturnValue(
        mockSSE([{ type: 'meta', tool_calls: toolCalls, sources }]),
      );

      const { result } = renderHook(() => useChat());
      await act(async () => {
        await result.current.sendMessage('hello');
      });

      expect(dispatch).toHaveBeenCalledWith({
        type: 'UPDATE_LAST_MESSAGE',
        meta: { toolCalls, sources },
      });
    });

    it('appends an error message when chatStream throws', async () => {
      (chatStream as unknown as Mock).mockReturnValue(
        (async function* (): AsyncGenerator<SSEEvent> {
          throw new Error('network down');
        })(),
      );

      const { result } = renderHook(() => useChat());
      await act(async () => {
        await result.current.sendMessage('hello');
      });

      expect(dispatch).toHaveBeenCalledWith({
        type: 'UPDATE_LAST_MESSAGE',
        appendContent: '\n\n错误: network down',
      });
    });

    it('aborts the previous stream when a new message is sent', async () => {
      const { stream: first, resolve: resolveFirst } = makeStream([{ type: 'token', content: 'a' }]);
      const { stream: second, resolve: resolveSecond } = makeStream([{ type: 'token', content: 'b' }]);
      const streams = [first, second];
      (chatStream as unknown as Mock).mockImplementation(() => streams.shift());

      const abortSpy = vi.spyOn(AbortController.prototype, 'abort');
      const { result, rerender } = renderHook(() => useChat());

      let p1: Promise<void>;
      await act(async () => {
        p1 = result.current.sendMessage('first');
      });
      expect(abortSpy).not.toHaveBeenCalled();

      // Switching threads resets `isStreaming` but leaves the previous
      // controller in `abortRef` (it is aborted, not nulled).
      (useAppState as unknown as Mock).mockReturnValue({
        threadId: 'test-thread-2',
        isStreaming: false,
      });
      await act(async () => {
        rerender();
      });
      abortSpy.mockClear(); // isolate sendMessage's own abort call

      let p2: Promise<void>;
      await act(async () => {
        p2 = result.current.sendMessage('second');
      });
      // sendMessage cancels the leftover controller before starting a new stream.
      expect(abortSpy).toHaveBeenCalledTimes(1);

      act(() => resolveFirst());
      act(() => resolveSecond());
      await act(async () => {
        await p1;
        await p2;
      });
    });

    it('guards against concurrent sends while a stream is already active', async () => {
      const { stream, resolve } = makeStream([{ type: 'token', content: 'a' }]);
      (chatStream as unknown as Mock).mockReturnValue(stream);

      const { result } = renderHook(() => useChat());

      let p1: Promise<void>;
      await act(async () => {
        p1 = result.current.sendMessage('first');
      });

      dispatch.mockClear();

      let p2: Promise<void>;
      await act(async () => {
        p2 = result.current.sendMessage('second');
      });

      expect(chatStream).toHaveBeenCalledTimes(1);
      expect(dispatch).not.toHaveBeenCalled();

      act(() => resolve());
      await act(async () => {
        await p1;
        await p2;
      });
    });
  });

  describe('resume', () => {
    it('calls chatNonStream with the resume data', async () => {
      (chatNonStream as unknown as Mock).mockResolvedValue({
        thread_id: 'test-thread',
        status: 'completed',
        response: 'Done!',
        interrupt: null,
        tool_calls: [],
        sources: [],
      });

      const { result } = renderHook(() => useChat());
      const resumeData = { decision: 'approve' };
      await act(async () => {
        await result.current.resume(resumeData);
      });

      expect(chatNonStream).toHaveBeenCalledWith({
        message: '',
        thread_id: 'test-thread',
        resume_data: resumeData,
      });
    });

    it('dispatches ADD_MESSAGE with the response text and _meta on completion', async () => {
      (chatNonStream as unknown as Mock).mockResolvedValue({
        thread_id: 'test-thread',
        status: 'completed',
        response: 'Done!',
        interrupt: null,
        tool_calls: [{ tool: 'search', content: 'x' }],
        sources: [],
      });

      const { result } = renderHook(() => useChat());
      await act(async () => {
        await result.current.resume({ decision: 'approve' });
      });

      expect(dispatch).toHaveBeenCalledWith({
        type: 'ADD_MESSAGE',
        message: {
          role: 'assistant',
          content: 'Done!',
          _meta: { toolCalls: [{ tool: 'search', content: 'x' }], sources: [] },
        },
      });
    });

    it('dispatches SET_INTERRUPT and a system message when the resumed call is interrupted again', async () => {
      (chatNonStream as unknown as Mock).mockResolvedValue({
        thread_id: 'test-thread',
        status: 'interrupted',
        response: '',
        interrupt: { type: 'approval' },
        tool_calls: [],
        sources: [],
      });

      const { result } = renderHook(() => useChat());
      await act(async () => {
        await result.current.resume({ decision: 'approve' });
      });

      expect(dispatch).toHaveBeenCalledWith({ type: 'SET_INTERRUPT', interrupt: { type: 'approval' } });
      expect(dispatch).toHaveBeenCalledWith({
        type: 'ADD_MESSAGE',
        message: { role: 'system', content: '另一个操作需要处理。' },
      });
    });

    it('dispatches ADD_MESSAGE with an error message when chatNonStream rejects', async () => {
      (chatNonStream as unknown as Mock).mockRejectedValue(new Error('boom'));

      const { result } = renderHook(() => useChat());
      await act(async () => {
        await result.current.resume({ decision: 'approve' });
      });

      expect(dispatch).toHaveBeenCalledWith({
        type: 'ADD_MESSAGE',
        message: { role: 'assistant', content: '错误: boom' },
      });
    });
  });

  describe('cleanup', () => {
    it('aborts the stream and clears the flush interval on unmount', async () => {
      const { stream, resolve } = makeStream([{ type: 'token', content: 'a' }]);
      (chatStream as unknown as Mock).mockReturnValue(stream);

      const abortSpy = vi.spyOn(AbortController.prototype, 'abort');
      const { result, unmount } = renderHook(() => useChat());

      let p: Promise<void>;
      await act(async () => {
        p = result.current.sendMessage('hello');
      });

      // The token-batching interval is active while streaming.
      expect(vi.getTimerCount()).toBeGreaterThan(0);

      act(() => {
        unmount();
      });

      expect(abortSpy).toHaveBeenCalled();
      expect(vi.getTimerCount()).toBe(0);

      act(() => resolve());
      await act(async () => {
        await p;
      });
    });
  });
});
