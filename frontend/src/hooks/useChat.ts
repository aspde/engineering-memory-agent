import { useCallback, useEffect, useRef } from 'react';
import { useAppDispatch, useAppState } from '../context/AppContext';
import { chatStream } from '../api/agent';
import { invalidateStatsCache } from './useMemories';
import type { MemoryWriteResult } from '../types';

/** SSE `node` event → Chinese status label (mirrors frontend/app.py `_node_labels`). */
const NODE_LABELS: Record<string, string> = {
  call_llm: '思考中…',
  tools: '执行工具…',
  check_approval: '检查权限…',
  check_conflict: '检查冲突…',
};

/** How often (ms) to flush accumulated tokens into React state. */
const TOKEN_BATCH_INTERVAL_MS = 50;

/**
 * Chat orchestration hook.
 *
 * Wires the global app state to the streaming / non-streaming agent APIs.
 * Uses `AbortController` to cancel in-flight SSE streams on unmount or
 * thread switch.  Streaming state is synced to `AppState.isStreaming` so
 * the UI can show a loading indicator while the agent is busy.
 */
export function useChat() {
  const { threadId, isStreaming } = useAppState();
  const dispatch = useAppDispatch();

  const isStreamingRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const tokenBufferRef = useRef('');
  const flushTimerRef = useRef<number | null>(null);

  // Cancel any running SSE stream + timers when the hook is disposed.
  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      if (flushTimerRef.current !== null) {
        window.clearInterval(flushTimerRef.current);
        flushTimerRef.current = null;
      }
    };
  }, []);

  // Abort the current stream when the user switches to a different thread.
  useEffect(() => {
    abortRef.current?.abort();
    if (flushTimerRef.current !== null) {
      window.clearInterval(flushTimerRef.current);
      flushTimerRef.current = null;
    }
    isStreamingRef.current = false;
    dispatch({ type: 'SET_STREAMING', isStreaming: false });
  }, [threadId, dispatch]);

  /** Push accumulated tokens (if any) onto the last assistant message. */
  const flushTokens = useCallback(() => {
    if (tokenBufferRef.current) {
      const chunk = tokenBufferRef.current;
      tokenBufferRef.current = '';
      dispatch({ type: 'UPDATE_LAST_MESSAGE', appendContent: chunk });
    }
  }, [dispatch]);

  /** Stop the periodic token-flush interval, if running. */
  const stopTokenTimer = useCallback(() => {
    if (flushTimerRef.current !== null) {
      window.clearInterval(flushTimerRef.current);
      flushTimerRef.current = null;
    }
  }, []);

  /**
   * Send a user message and stream the assistant reply.
   *
   * 1. Adds the user message to state.
   * 2. Marks the current thread as loaded (so the page won't re-fetch history).
   * 3. Adds an empty assistant placeholder that tokens stream into.
   * 4. Consumes the SSE stream, handling token/node/interrupt/meta/error/done.
   *
   * When *forceWrite* is true, the request carries ``force_write`` so the
   * server injects a write_memory_tool call for this turn — the write runs
   * inside the agent flow (LLM extraction + on-the-spot conflict handling),
   * and its outcome arrives on the meta event as ``memory_write``.
   */
  const sendMessage = useCallback(
    async (rawInput: string, forceWrite = false) => {
      const text = rawInput.trim();
      if (!text || isStreamingRef.current) return;
      isStreamingRef.current = true;
      dispatch({ type: 'SET_STREAMING', isStreaming: true });

      // Cancel any previous stream before starting a new one.
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      dispatch({ type: 'ADD_MESSAGE', message: { role: 'user', content: text } });
      dispatch({ type: 'SET_LOADED_THREAD', threadId });
      dispatch({ type: 'ADD_MESSAGE', message: { role: 'assistant', content: '' } });
      // Invalidate thread list cache immediately so the new conversation
      // appears in the sidebar right away, not only after the reply arrives.
      dispatch({ type: 'INVALIDATE_THREADS' });

      tokenBufferRef.current = '';
      const yieldedNodes = new Set<string>();
      // Force-write outcome (action + summary) from the meta event — resolved
      // by sendMessage so the caller can toast what the write did.
      let memoryWrite: MemoryWriteResult | null = null;

      try {
        const stream = chatStream({
          message: text,
          thread_id: threadId,
          force_write: forceWrite,
        });
        for await (const event of stream) {
          switch (event.type) {
            case 'token':
              tokenBufferRef.current += event.content;
              if (flushTimerRef.current === null) {
                flushTimerRef.current = window.setInterval(
                  flushTokens,
                  TOKEN_BATCH_INTERVAL_MS,
                );
              }
              break;

            case 'node': {
              const node = event.node;
              if (node !== 'generate_final' && !yieldedNodes.has(node)) {
                yieldedNodes.add(node);
                const label = NODE_LABELS[node] ?? node;
                const suffix = node === 'check_conflict' ? '\n' : '';
                tokenBufferRef.current += `> ${label}\n${suffix}`;
              }
              break;
            }

            case 'interrupt': {
              // Flush before appending the system note so tokens land on the
              // assistant placeholder (which is still the last message).
              flushTokens();
              stopTokenTimer();
              dispatch({ type: 'SET_INTERRUPT', interrupt: event.data });
              dispatch({
                type: 'ADD_MESSAGE',
                message: {
                  role: 'system',
                  content:
                    event.data.type === 'conflict'
                      ? '检测到记忆冲突，请选择如何解决。'
                      : '智能体想要执行写入操作，请批准或拒绝。',
                },
              });
              return;
            }

            case 'meta':
              dispatch({
                type: 'UPDATE_LAST_MESSAGE',
                meta: { toolCalls: event.tool_calls, sources: event.sources },
              });
              if (event.memory_write) {
                memoryWrite = event.memory_write;
              }
              break;

            case 'error': {
              // Flush any partial tokens so already-generated text stays on
              // the assistant bubble, then surface the failure as a SEPARATE
              // error message — never merged into the reply body.
              flushTokens();
              stopTokenTimer();
              dispatch({
                type: 'ADD_MESSAGE',
                message: { role: 'system', kind: 'error', content: `错误: ${event.message}` },
              });
              return;
            }

            case 'done':
              break;
          }
        }
      } catch (err) {
        // Network / transport failure: keep any partial tokens on the
        // assistant bubble, then add a SEPARATE error message.
        flushTokens();
        stopTokenTimer();
        dispatch({
          type: 'ADD_MESSAGE',
          message: {
            role: 'system',
            kind: 'error',
            content: `错误: ${err instanceof Error ? err.message : String(err)}`,
          },
        });
      } finally {
        // Always flush any remaining buffered tokens and clean up.
        flushTokens();
        stopTokenTimer();
        isStreamingRef.current = false;
        dispatch({ type: 'SET_STREAMING', isStreaming: false });
        dispatch({ type: 'INVALIDATE_THREADS' });
        invalidateStatsCache();
        abortRef.current = null;
      }

      return memoryWrite;
    },
    [dispatch, threadId, flushTokens, stopTokenTimer],
  );

  /**
   * Resume an interrupted conversation with the user's decision
   * (approve/reject/conflict resolution). Streams the resumed run's tokens
   * live through SSE — identical UX to the first send, instead of waiting
   * for the whole answer and appending it at once.
   */
  const resume = useCallback(
    async (resumeData: Record<string, unknown>) => {
      if (isStreamingRef.current) return;
      isStreamingRef.current = true;
      dispatch({ type: 'SET_STREAMING', isStreaming: true });
      dispatch({ type: 'CLEAR_INTERRUPT' });

      // Fresh assistant placeholder that the resumed turn's tokens stream into.
      dispatch({ type: 'ADD_MESSAGE', message: { role: 'assistant', content: '' } });

      tokenBufferRef.current = '';
      const yieldedNodes = new Set<string>();
      // Force-write outcome from the meta event, resolved by resume just like
      // sendMessage (a resumed conflict resolution can itself persist a write).
      let memoryWrite: MemoryWriteResult | null = null;

      try {
        const stream = chatStream({
          message: '',
          thread_id: threadId,
          resume_data: resumeData,
        });
        for await (const event of stream) {
          switch (event.type) {
            case 'token':
              tokenBufferRef.current += event.content;
              if (flushTimerRef.current === null) {
                flushTimerRef.current = window.setInterval(
                  flushTokens,
                  TOKEN_BATCH_INTERVAL_MS,
                );
              }
              break;

            case 'node': {
              const node = event.node;
              if (node !== 'generate_final' && !yieldedNodes.has(node)) {
                yieldedNodes.add(node);
                const label = NODE_LABELS[node] ?? node;
                tokenBufferRef.current += `> ${label}\n`;
              }
              break;
            }

            case 'interrupt': {
              // A resumed run can pause again (e.g. a write approved, then a
              // conflict surfaced). Flush first so buffered tokens land on the
              // assistant placeholder before the interrupt card appears.
              flushTokens();
              stopTokenTimer();
              dispatch({ type: 'SET_INTERRUPT', interrupt: event.data });
              dispatch({
                type: 'ADD_MESSAGE',
                message: {
                  role: 'system',
                  content:
                    event.data.type === 'conflict'
                      ? '检测到记忆冲突，请选择如何解决。'
                      : '智能体想要执行写入操作，请批准或拒绝。',
                },
              });
              return;
            }

            case 'meta':
              dispatch({
                type: 'UPDATE_LAST_MESSAGE',
                meta: { toolCalls: event.tool_calls, sources: event.sources },
              });
              if (event.memory_write) {
                memoryWrite = event.memory_write;
              }
              break;

            case 'error': {
              // Flush any partial tokens so already-generated text stays on
              // the assistant bubble, then surface the failure as a SEPARATE
              // error message — never merged into the reply body.
              flushTokens();
              stopTokenTimer();
              dispatch({
                type: 'ADD_MESSAGE',
                message: { role: 'system', kind: 'error', content: `错误: ${event.message}` },
              });
              return;
            }

            case 'done':
              break;
          }
        }
      } catch (err) {
        // Network / transport failure: keep any partial tokens on the
        // assistant bubble, then add a SEPARATE error message.
        flushTokens();
        stopTokenTimer();
        dispatch({
          type: 'ADD_MESSAGE',
          message: {
            role: 'system',
            kind: 'error',
            content: `错误: ${err instanceof Error ? err.message : String(err)}`,
          },
        });
      } finally {
        // Always flush any remaining buffered tokens and clean up.
        flushTokens();
        stopTokenTimer();
        isStreamingRef.current = false;
        dispatch({ type: 'SET_STREAMING', isStreaming: false });
        dispatch({ type: 'INVALIDATE_THREADS' });
        invalidateStatsCache();
      }

      return memoryWrite;
    },
    [dispatch, threadId, flushTokens, stopTokenTimer],
  );

  return { sendMessage, resume, isStreaming };
}
