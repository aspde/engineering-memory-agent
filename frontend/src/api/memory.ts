import type {
  IngestResponse,
  MemoryDeleteResponse,
  MemoryGetResponse,
  MemorySearchResponse,
  MemoryStatsResponse,
  MemoryWriteResponse,
} from '../types';
import { apiDelete, apiGet, apiPost } from './client';

/** Fetch aggregate statistics about the memory store. */
async function getStats(): Promise<MemoryStatsResponse> {
  return apiGet<MemoryStatsResponse>('/api/memory/stats');
}

/** Chunk, embed, and store a document's content. */
async function ingest(documentId: string, content: string): Promise<IngestResponse> {
  return apiPost<IngestResponse>('/api/memory/ingest', {
    document_id: documentId,
    content,
  });
}

/** Write structured memory from text — extraction + similarity check + persist. */
async function writeMemory(
  content: string,
  sourceType: string = 'conversation',
  metadata?: Record<string, unknown>,
): Promise<MemoryWriteResponse> {
  return apiPost<MemoryWriteResponse>('/api/memory/memories/write', {
    content,
    source_type: sourceType,
    ...(metadata ? { metadata } : {}),
  });
}

/** Semantic search over structured memories with decay-weighted ranking. */
async function searchMemories(query: string, topK: number): Promise<MemorySearchResponse> {
  return apiPost<MemorySearchResponse>('/api/memory/memories/search', {
    query,
    top_k: topK,
  });
}

/** Fetch a single memory by its id. */
async function getMemory(id: string): Promise<MemoryGetResponse> {
  return apiGet<MemoryGetResponse>(`/api/memory/memories/${encodeURIComponent(id)}`);
}

/** Delete a memory by its id. */
async function deleteMemory(id: string): Promise<MemoryDeleteResponse> {
  return apiDelete<MemoryDeleteResponse>(`/api/memory/memories/${encodeURIComponent(id)}`);
}

export { getStats, ingest, writeMemory, searchMemories, getMemory, deleteMemory };
