import type { SaveRunMemoryResponse, ScenarioInfo, ScenarioRunResponse } from '../types';
import { apiGet, apiPost } from './client';

/** Fetch the list of visible (active + beta) scenarios. */
export async function listScenarios(): Promise<ScenarioInfo[]> {
  return apiGet<ScenarioInfo[]>('/api/scenarios');
}

/** Trigger a scenario run with optional params and thread_id for persistence. */
export async function runScenario(
  name: string,
  params: Record<string, unknown> = {},
  threadId?: string,
): Promise<ScenarioRunResponse> {
  return apiPost<ScenarioRunResponse>(`/api/scenarios/${name}/run`, { params, thread_id: threadId });
}

/** Save a completed postmortem run into the memory store. */
export async function saveRunAsMemory(runId: string): Promise<SaveRunMemoryResponse> {
  return apiPost<SaveRunMemoryResponse>(`/api/scenarios/runs/${runId}/save-as-memory`, {});
}
