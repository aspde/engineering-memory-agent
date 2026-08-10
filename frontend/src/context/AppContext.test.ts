import { describe, it, expect, vi, afterEach } from 'vitest';
import { appReducer } from './AppContext';
import type { AppState } from '../types';

const initialState: AppState = {
  threadId: '00000000-0000-0000-0000-000000000000',
  messages: [],
  pendingInterrupt: null,
  waitingForApproval: false,
  isStreaming: false,
  threads: [],
  threadsFetchedAt: 0,
  loadedThreadId: null,
  memFilterId: null,
  activeScenario: null,
};

afterEach(() => {
  vi.restoreAllMocks();
});

describe('appReducer', () => {
  it('SET_THREAD_ID sets threadId and resets messages/interrupt/approval', () => {
    const state: AppState = {
      ...initialState,
      messages: [{ role: 'user', content: 'hi' }],
      pendingInterrupt: { type: 'approval' },
      waitingForApproval: true,
    };
    const next = appReducer(state, { type: 'SET_THREAD_ID', threadId: 't-1' });
    expect(next.threadId).toBe('t-1');
    expect(next.messages).toEqual([]);
    expect(next.pendingInterrupt).toBeNull();
    expect(next.waitingForApproval).toBe(false);
  });

  it('ADD_MESSAGE appends to the messages array', () => {
    const next = appReducer(initialState, {
      type: 'ADD_MESSAGE',
      message: { role: 'assistant', content: 'hello' },
    });
    expect(next.messages).toHaveLength(1);
    expect(next.messages[0]).toEqual({ role: 'assistant', content: 'hello' });
  });

  it('ADD_MESSAGE preserves the error kind marker on a system error message', () => {
    const next = appReducer(initialState, {
      type: 'ADD_MESSAGE',
      message: { role: 'system', kind: 'error', content: '错误: boom' },
    });
    expect(next.messages).toHaveLength(1);
    expect(next.messages[0]).toEqual({ role: 'system', kind: 'error', content: '错误: boom' });
  });

  it('UPDATE_LAST_MESSAGE appends content and updates _meta on the last message', () => {
    const state: AppState = {
      ...initialState,
      messages: [{ role: 'assistant', content: 'Hel', _meta: { toolCalls: [], sources: [] } }],
    };
    const next = appReducer(state, {
      type: 'UPDATE_LAST_MESSAGE',
      appendContent: 'lo',
      meta: { toolCalls: [{ tool: 'search', content: 'q' }], sources: [] },
    });
    expect(next.messages).toHaveLength(1);
    expect(next.messages[0].content).toBe('Hello');
    expect(next.messages[0]._meta).toEqual({
      toolCalls: [{ tool: 'search', content: 'q' }],
      sources: [],
    });
  });

  it('UPDATE_LAST_MESSAGE is a no-op when messages is empty', () => {
    const next = appReducer(initialState, { type: 'UPDATE_LAST_MESSAGE', appendContent: 'x' });
    expect(next).toBe(initialState);
  });

  it('SET_MESSAGES replaces the messages array', () => {
    const messages: AppState['messages'] = [{ role: 'user', content: 'a' }];
    const next = appReducer(initialState, { type: 'SET_MESSAGES', messages });
    expect(next.messages).toEqual(messages);
  });

  it('SET_INTERRUPT sets pendingInterrupt and waitingForApproval', () => {
    const next = appReducer(initialState, {
      type: 'SET_INTERRUPT',
      interrupt: { type: 'approval', tool_name: 'write' },
    });
    expect(next.pendingInterrupt).toEqual({ type: 'approval', tool_name: 'write' });
    expect(next.waitingForApproval).toBe(true);
  });

  it('CLEAR_INTERRUPT clears pendingInterrupt and waitingForApproval', () => {
    const state: AppState = {
      ...initialState,
      pendingInterrupt: { type: 'conflict' },
      waitingForApproval: true,
    };
    const next = appReducer(state, { type: 'CLEAR_INTERRUPT' });
    expect(next.pendingInterrupt).toBeNull();
    expect(next.waitingForApproval).toBe(false);
  });

  it('SET_STREAMING toggles isStreaming', () => {
    const next = appReducer(initialState, { type: 'SET_STREAMING', isStreaming: true });
    expect(next.isStreaming).toBe(true);
  });

  it('SET_THREADS updates threads and sets threadsFetchedAt', () => {
    vi.spyOn(Date, 'now').mockReturnValue(123456);
    const threads = [{ thread_id: 't-1', title: 'My Thread' }];
    const next = appReducer(initialState, { type: 'SET_THREADS', threads });
    expect(next.threads).toEqual(threads);
    expect(next.threadsFetchedAt).toBe(123456);
  });

  it('SET_MEM_FILTER sets memFilterId', () => {
    const next = appReducer(initialState, { type: 'SET_MEM_FILTER', memId: 'mem-1' });
    expect(next.memFilterId).toBe('mem-1');
  });

  it('CLEAR_MEM_FILTER clears memFilterId', () => {
    const state: AppState = { ...initialState, memFilterId: 'mem-1' };
    const next = appReducer(state, { type: 'CLEAR_MEM_FILTER' });
    expect(next.memFilterId).toBeNull();
  });

  it('NEW_CONVERSATION resets state with a fresh threadId', () => {
    const state: AppState = {
      ...initialState,
      threadId: 'old',
      messages: [{ role: 'user', content: 'x' }],
      pendingInterrupt: { type: 'approval' },
      waitingForApproval: true,
      isStreaming: true,
      loadedThreadId: 'old',
      memFilterId: 'mem-1',
    };
    const next = appReducer(state, { type: 'NEW_CONVERSATION', threadId: 'new-id' });
    expect(next.threadId).toBe('new-id');
    expect(next.messages).toEqual([]);
    expect(next.pendingInterrupt).toBeNull();
    expect(next.waitingForApproval).toBe(false);
    expect(next.isStreaming).toBe(false);
    expect(next.loadedThreadId).toBeNull();
  });

  it('SET_LOADED_THREAD sets loadedThreadId', () => {
    const next = appReducer(initialState, { type: 'SET_LOADED_THREAD', threadId: 't-9' });
    expect(next.loadedThreadId).toBe('t-9');
  });

  it('INVALIDATE_THREADS resets threadsFetchedAt to 0', () => {
    const state: AppState = { ...initialState, threadsFetchedAt: 12345 };
    const next = appReducer(state, { type: 'INVALIDATE_THREADS' });
    expect(next.threadsFetchedAt).toBe(0);
  });

  it('returns state unchanged for an unknown action', () => {
    const next = appReducer(initialState, { type: 'UNKNOWN_ACTION' } as never);
    expect(next).toBe(initialState);
  });
});
