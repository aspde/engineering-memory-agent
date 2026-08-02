import type {
  IngestResponse,
  MemoryGetResponse,
  MemorySearchResponse,
  MemoryStatsResponse,
} from '../types';
import { apiGet, apiPost } from './client';

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

export { getStats, ingest, searchMemories, getMemory };
