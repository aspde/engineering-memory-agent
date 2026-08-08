import type {
  ConflictReopenResponse,
  ConflictResolveResponse,
  PendingConflict,
} from '../types';
import { apiGet, apiPost } from './client';

interface GetConflictsParams {
  conflict_type?: string;
  status?: string;
}

/** List memory conflicts, optionally filtered by type and/or status. */
async function getConflicts(
  params?: GetConflictsParams,
): Promise<PendingConflict[]> {
  const searchParams = new URLSearchParams();
  if (params?.conflict_type) searchParams.set('conflict_type', params.conflict_type);
  if (params?.status) searchParams.set('status', params.status);
  const qs = searchParams.toString();
  return apiGet<PendingConflict[]>(`/api/conflicts${qs ? `?${qs}` : ''}`);
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

/** Reset a resolved patrol conflict to pending for re-arbitration. */
async function reopenConflict(id: string): Promise<ConflictReopenResponse> {
  return apiPost<ConflictReopenResponse>(
    `/api/conflicts/${encodeURIComponent(id)}/reopen`,
    {},
  );
}

export { getConflicts, reopenConflict, resolveConflict };
