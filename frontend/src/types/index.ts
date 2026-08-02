// ── SSE streaming events ──────────────────────────────────────────

export type SSEEvent =
  | { type: 'token'; content: string }
  | { type: 'node'; node: string }
  | { type: 'interrupt'; data: Interrupt }
  | { type: 'meta'; tool_calls: ToolCall[]; sources: Source[] }
  | { type: 'error'; message: string }
  | { type: 'done' };

// ── Agent API types ──────────────────────────────────────────────

export interface ChatRequest {
  message: string;
  thread_id: string;
  resume_data?: Record<string, unknown>;
}

export interface ChatResponse {
  thread_id: string;
  status: 'completed' | 'interrupted' | 'error';
  response: string;
  interrupt: Interrupt | null;
  tool_calls: ToolCall[];
  sources: Source[];
}

export interface ThreadInfo {
  thread_id: string;
  title: string;
}

export interface ThreadMessagesResponse {
  thread_id: string;
  messages: MessageFromBackend[];
}

// ── Message types ─────────────────────────────────────────────────

/** Raw message from backend (GET /api/agent/thread/{id}). */
export interface MessageFromBackend {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

/** Frontend-enriched message with optional metadata from streaming. */
export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  _meta?: {
    toolCalls: ToolCall[];
    sources: Source[];
  };
}

export interface ToolCall {
  tool: string;
  content: string;
}

export interface Source {
  type: 'memory' | 'chunk' | 'unknown';
  id?: string;
  summary?: string;
  snippet?: string;
  document_id?: string;
  relevance?: number;
}

// ── Interrupt (approval / conflict) ──────────────────────────────

export interface Interrupt {
  type: 'approval' | 'conflict';
  tool_name?: string;
  tool_args?: Record<string, unknown>;
  summary?: string;
  new_summary?: string;
  existing_summary?: string;
  [key: string]: unknown;
}

// ── Memory API types ─────────────────────────────────────────────

export interface IngestRequest {
  document_id: string;
  content: string;
  metadata?: Record<string, unknown>;
}

export interface IngestResponse {
  document_id: string;
  chunks_written: number;
}

export interface SearchRequest {
  query: string;
  top_k: number;
  use_llm_rerank?: boolean;
}

export interface SearchResult {
  content: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface SearchResponse {
  results: SearchResult[];
}

export interface MemoryWriteRequest {
  content: string;
  source_type?: string;
  metadata?: Record<string, unknown>;
}

export interface MemoryWriteResponse {
  id: string;
  action: 'inserted' | 'merged' | 'conflict';
  summary: string;
}

export interface MemorySearchRequest {
  query: string;
  top_k: number;
  use_llm_rerank?: boolean;
}

export interface MemorySearchResponse {
  results: Record<string, unknown>[];
}

export interface MemoryGetResponse {
  id: string;
  source_type: string;
  summary: string;
  entities: Record<string, unknown>[];
  relations: Record<string, unknown>[];
  decay_factor: number;
  recall_count: number;
  meta: Record<string, unknown>;
  created_at: string;
}

export interface MemoryStatsResponse {
  total_memories: number;
  total_chunks: number;
  total_conversations: number;
  by_source_type: { source_type: string; count: number }[];
  avg_decay_factor: number;
  avg_entities_per_memory: number;
  avg_relations_per_memory: number;
  recent_count_7d: number;
  top_entities: { name: string; count: number }[];
}

// ── App state ────────────────────────────────────────────────────

export interface AppState {
  threadId: string;
  messages: Message[];
  pendingInterrupt: Interrupt | null;
  waitingForApproval: boolean;
  isStreaming: boolean;
  threads: ThreadInfo[];
  threadsFetchedAt: number;
  loadedThreadId: string | null;
  memFilterId: string | null;
}

export type AppAction =
  | { type: 'SET_THREAD_ID'; threadId: string }
  | { type: 'ADD_MESSAGE'; message: Message }
  | { type: 'UPDATE_LAST_MESSAGE'; appendContent?: string; meta?: Message['_meta'] }
  | { type: 'SET_MESSAGES'; messages: Message[] }
  | { type: 'SET_INTERRUPT'; interrupt: Interrupt }
  | { type: 'CLEAR_INTERRUPT' }
  | { type: 'SET_STREAMING'; isStreaming: boolean }
  | { type: 'SET_THREADS'; threads: ThreadInfo[] }
  | { type: 'SET_MEM_FILTER'; memId: string }
  | { type: 'CLEAR_MEM_FILTER' }
  | { type: 'NEW_CONVERSATION'; threadId: string }
  | { type: 'SET_LOADED_THREAD'; threadId: string | null };
