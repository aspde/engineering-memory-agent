import { useCallback, useEffect, useRef } from 'react';
import { useAppDispatch, useAppState } from '../context/AppContext';
import { chatNonStream, chatStream } from '../api/agent';

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
 * the sidebar can disable thread-select buttons while the agent is busy.
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
   */
  const sendMessage = useCallback(
    async (rawInput: string) => {
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

      tokenBufferRef.current = '';
      const yieldedNodes = new Set<string>();

      try {
        const stream = chatStream({ message: text, thread_id: threadId });
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
                tokenBufferRef.current += `\n\n> ${label}\n\n`;
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
              break;

            case 'error':
              flushTokens();
              stopTokenTimer();
              dispatch({
                type: 'UPDATE_LAST_MESSAGE',
                appendContent: `\n\n错误: ${event.message}`,
              });
              break;

            case 'done':
              break;
          }
        }
      } catch (err) {
        flushTokens();
        stopTokenTimer();
        dispatch({
          type: 'UPDATE_LAST_MESSAGE',
          appendContent: `\n\n错误: ${err instanceof Error ? err.message : String(err)}`,
        });
      } finally {
        // Always flush any remaining buffered tokens and clean up.
        flushTokens();
        stopTokenTimer();
        isStreamingRef.current = false;
        dispatch({ type: 'SET_STREAMING', isStreaming: false });
        abortRef.current = null;
      }
    },
    [dispatch, threadId, flushTokens, stopTokenTimer],
  );

  /**
   * Resume an interrupted conversation with the user's decision
   * (approve/reject/conflict resolution). Non-streaming response.
   */
  const resume = useCallback(
    async (resumeData: Record<string, unknown>) => {
      if (isStreamingRef.current) return;
      isStreamingRef.current = true;
      dispatch({ type: 'SET_STREAMING', isStreaming: true });
      dispatch({ type: 'CLEAR_INTERRUPT' });

      try {
        const result = await chatNonStream({
          message: '',
          thread_id: threadId,
          resume_data: resumeData,
        });

        if (result.status === 'interrupted' && result.interrupt) {
          dispatch({ type: 'SET_INTERRUPT', interrupt: result.interrupt });
          dispatch({
            type: 'ADD_MESSAGE',
            message: { role: 'system', content: '另一个操作需要处理。' },
          });
        } else {
          dispatch({
            type: 'ADD_MESSAGE',
            message: {
              role: 'assistant',
              content: result.response || '(无回复)',
              _meta: {
                toolCalls: result.tool_calls ?? [],
                sources: result.sources ?? [],
              },
            },
          });
        }
      } catch (err) {
        dispatch({
          type: 'ADD_MESSAGE',
          message: {
            role: 'assistant',
            content: `错误: ${err instanceof Error ? err.message : String(err)}`,
          },
        });
      } finally {
        isStreamingRef.current = false;
        dispatch({ type: 'SET_STREAMING', isStreaming: false });
      }
    },
    [dispatch, threadId],
  );

  return { sendMessage, resume, isStreaming };
}
