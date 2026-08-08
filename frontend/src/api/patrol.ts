import { apiGet, apiPost } from './client';
import type {
  DismissFindingResponse,
  PatrolFinding,
  PatrolLogDetail,
  PatrolLogList,
  QueuePatrolConflictResponse,
  TriggerPatrolRequest,
  TriggerPatrolResponse,
} from '../types';

export function triggerPatrol(
  patrol_type: string,
  scope?: string,
): Promise<TriggerPatrolResponse> {
  const body: TriggerPatrolRequest = { patrol_type, scope: scope ?? 'all' };
  return apiPost<TriggerPatrolResponse>('/api/patrol/trigger', body);
}

export function listPatrolLogs(
  params?: { limit?: number; offset?: number; patrol_type?: string },
): Promise<PatrolLogList> {
  const searchParams = new URLSearchParams();
  if (params?.limit != null) searchParams.set('limit', String(params.limit));
  if (params?.offset != null) searchParams.set('offset', String(params.offset));
  if (params?.patrol_type) searchParams.set('patrol_type', params.patrol_type);
  const qs = searchParams.toString();
  return apiGet<PatrolLogList>(`/api/patrol/logs${qs ? `?${qs}` : ''}`);
}

export function getPatrolLog(id: string): Promise<PatrolLogDetail> {
  return apiGet<PatrolLogDetail>(`/api/patrol/logs/${id}`);
}

export function dismissFinding(
  logId: string,
  findingId: string,
): Promise<DismissFindingResponse> {
  return apiPost<DismissFindingResponse>(
    `/api/patrol/findings/${logId}/dismiss`,
    { finding_id: findingId },
  );
}

/** Queue a patrol contradiction finding for HITL arbitration. */
export function queuePatrolConflict(
  logId: string,
  finding: PatrolFinding,
): Promise<QueuePatrolConflictResponse> {
  return apiPost<QueuePatrolConflictResponse>(
    `/api/patrol/findings/${logId}/conflict`,
    finding,
  );
}
