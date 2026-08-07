import type { ConflictResolveResponse, PendingConflict } from '../types';
import { apiGet, apiPost } from './client';

/** List unresolved memory conflicts awaiting a human decision. */
async function getConflicts(): Promise<PendingConflict[]> {
  return apiGet<PendingConflict[]>('/api/conflicts');
}

/** Resolve a pending conflict with one of keep_existing/overwrite/merge/keep_both. */
async function resolveConflict(
  id: string,
  resolution: string,
): Promise<ConflictResolveResponse> {
  return apiPost<ConflictResolveResponse>(
    `/api/conflicts/${encodeURIComponent(id)}/resolve`,
    { resolution },
  );
}

export { getConflicts, resolveConflict };
