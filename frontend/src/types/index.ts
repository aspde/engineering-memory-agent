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

export interface ThreadDeleteResponse {
  thread_id: string;
  deleted: boolean;
}

// ── Message types ─────────────────────────────────────────────────

/** Raw message from backend (GET /api/agent/thread/{id}). */
export interface MessageFromBackend {
  role: 'user' | 'assistant' | 'system';
  content: string;
  tool_calls?: ToolCall[];
  sources?: Source[];
}

/**
 * Frontend-enriched message with optional metadata from streaming.
 */
export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  /**
   * Message subtype for special rendering. `'error'` marks a failed-run
   * notice (SSE error event or network failure) so the UI can render it
   * as a distinct error bubble instead of looking like an agent reply.
   */
  kind?: 'error';
  _meta?: {
    toolCalls: ToolCall[];
    sources: Source[];
  };
}

export interface ToolCall {
  tool: string;
  content: string;
}

export interface EntityRef {
  entity_id: string;
  canonical_name: string;
  type: string;
}

export interface Source {
  type: 'memory' | 'chunk' | 'unknown';
  id?: string;
  summary?: string;
  snippet?: string;
  document_id?: string;
  chunk_index?: number;
  relevance?: number;
  entities?: EntityRef[];
}

// ── Interrupt (approval / conflict) ──────────────────────────────

/** One tool call inside a batch approval payload. */
export interface ApprovalCall {
  /** Stable tool_call id — lets the backend match a per-row decision to the
   * exact call even when the same tool is called twice in one turn. */
  id?: string;
  tool_name: string;
  tool_args?: Record<string, unknown>;
  summary?: string;
}

export interface Interrupt {
  /**
   * Backend payloads: single-tool approval has NO type field, a multi-tool
   * approval is `type: 'batch'` with a `calls` array, conflict is
   * `type: 'conflict'`.  Absent/`batch` both render the approval card.
   */
  type?: 'approval' | 'conflict' | 'batch';
  tool_name?: string;
  tool_args?: Record<string, unknown>;
  summary?: string;
  /** Present on `type: 'batch'` approvals — one entry per sensitive tool. */
  calls?: ApprovalCall[];
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

export interface MemoryDeleteResponse {
  id: string;
  deleted: boolean;
}

export interface EntityGraphStats {
  coverage_ratio: number;
  growth_rate_7d: number;
  density: number;
  total_entities: number;
}

// ── Entity API types ──────────────────────────────────────────────

export interface EntityProfile {
  id: string;
  name: string;
  canonical_name: string;
  type: string;
  memory_count: number;
  source_breakdown: { source_type: string; count: number }[];
  first_seen_at: string;
}

export interface RelatedEntity {
  entity_id: string;
  name: string;
  type: string;
  relation_type: string;
  memory_count: number;
}

export interface RecentEntityMemory {
  memory_id: string;
  summary: string;
  source_type: string;
  created_at: string;
}

export interface EntityRelationsResponse {
  entity: EntityProfile;
  related_entities: RelatedEntity[];
  recent_memories: RecentEntityMemory[];
}

export interface EntitySearchResult {
  id: string;
  name: string;
  canonical_name: string;
  type: string;
  memory_count: number;
}

export interface EntitySearchResponse {
  results: EntitySearchResult[];
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
  entity_graph: EntityGraphStats | null;
}

// ── Pending conflict types (webhook/connector HITL) ─────────────────

export interface PendingConflict {
  id: string;
  source: string;
  source_type: string | null;
  existing_id: string;
  existing_summary: string;
  new_summary: string;
  status: string;
  resolution: string | null;
  created_at: string | null;
  conflict_type?: string;
  peer_id?: string | null;
}

export interface ConflictResolveResponse {
  id: string;
  resolution: string;
  outcome: Record<string, unknown>;
}

export interface ConflictReopenResponse {
  id: string;
  status: string;
}

// ── Connector API types ──────────────────────────────────────────────

export interface ConnectorInfo {
  source_type: string;
  display_name: string;
  status: 'active' | 'pending' | 'error';
  batch_mode: 'supported' | 'pending' | 'not_applicable';
}

export interface ConnectorListResponse {
  connectors: ConnectorInfo[];
}

export interface ConnectorLogEntry {
  id: string;
  source: string;
  event_type: string | null;
  status: string;
  payload_summary: string | null;
  memory_id: string | null;
  error: string | null;
  created_at: string;
}

export interface ConnectorLogsResponse {
  logs: ConnectorLogEntry[];
}

// ── Patrol types (Phase 3: proactive agent) ──────────────────────────

export type PatrolType = 'daily' | 'weekly' | 'event_driven' | 'manual';
export type PatrolTrigger = 'cron' | 'webhook' | 'manual';
export type PatrolStatus = 'running' | 'completed' | 'failed';

export interface PatrolLogSummary {
  id: string;
  patrol_type: PatrolType;
  trigger: PatrolTrigger;
  status: PatrolStatus;
  finding_count: number;
  started_at: string;
  completed_at: string | null;
}

export interface PatrolLogList {
  items: PatrolLogSummary[];
  total: number;
}

export interface PatrolFinding {
  id?: string;
  type?: string;
  title?: string;
  description?: string;
  severity?: 'critical' | 'warning' | 'info';
  memory_a_id?: string;
  memory_a_summary?: string;
  memory_b_id?: string;
  memory_b_summary?: string;
  conflict_description?: string;
  [key: string]: unknown;
}

export interface QueuePatrolConflictResponse {
  conflict_id: string;
  status: string;
  message?: string | null;
}

export interface PatrolLogDetail {
  id: string;
  patrol_type: PatrolType;
  trigger: PatrolTrigger;
  status: PatrolStatus;
  findings: Record<string, PatrolFinding[]> | null;
  dismissed_findings: string[];
  started_at: string;
  completed_at: string | null;
}

export interface TriggerPatrolRequest {
  patrol_type: string;
  scope?: string;
}

export interface TriggerPatrolResponse {
  patrol_id: string;
  status: string;
}

export interface DismissFindingResponse {
  ok: boolean;
  log_id: string;
  dismissed_finding_id: string;
}

// ── Scenario types (Phase 4: vertical scenarios) ─────────────────────

export type ScenarioStatus = 'active' | 'beta' | 'inactive';

export interface ScenarioInfo {
  key: string;
  name: string;
  description: string;
  triggers: string[];
  status: ScenarioStatus;
}

export interface ScenarioRunRequest {
  params: Record<string, unknown>;
}

export interface ScenarioRunResponse {
  scenario: string;
  status: string;
  result: string;
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
  activeScenario: string | null;
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
  | { type: 'SET_LOADED_THREAD'; threadId: string | null }
  | { type: 'REMOVE_THREAD'; threadId: string }
  | { type: 'INVALIDATE_THREADS' }
  | { type: 'SET_ACTIVE_SCENARIO'; scenario: string }
  | { type: 'CLEAR_ACTIVE_SCENARIO' }
  | { type: 'PREPEND_THREAD'; threadId: string; title: string };
