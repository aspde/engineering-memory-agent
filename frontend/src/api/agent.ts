import type {
  ChatRequest,
  ChatResponse,
  SSEEvent,
  ThreadDeleteResponse,
  ThreadInfo,
  ThreadMessagesResponse,
} from '../types';
import { apiDelete, apiGet, apiPost, apiSSE } from './client';

/** List conversation threads (most recent first). */
async function listThreads(): Promise<ThreadInfo[]> {
  return apiGet<ThreadInfo[]>('/api/agent/threads');
}

/** Fetch the message history for a given thread. */
async function getThreadMessages(threadId: string): Promise<ThreadMessagesResponse> {
  return apiGet<ThreadMessagesResponse>(
    `/api/agent/thread/${encodeURIComponent(threadId)}`,
  );
}

/** Send a message and receive a complete (non-streaming) response. */
async function chatNonStream(req: ChatRequest): Promise<ChatResponse> {
  return apiPost<ChatResponse>('/api/agent/chat', req);
}

/** Send a message and stream the response as SSE events. */
function chatStream(req: ChatRequest): AsyncGenerator<SSEEvent> {
  return apiSSE('/api/agent/chat/stream', req);
}

/** Delete a conversation thread and its checkpoint data. */
async function deleteThread(threadId: string): Promise<ThreadDeleteResponse> {
  return apiDelete<ThreadDeleteResponse>(
    `/api/agent/thread/${encodeURIComponent(threadId)}`,
  );
}

export { listThreads, getThreadMessages, chatNonStream, chatStream, deleteThread };
