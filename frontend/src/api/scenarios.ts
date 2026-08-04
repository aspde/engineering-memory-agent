import type { ScenarioInfo, ScenarioRunResponse } from '../types';
import { apiGet, apiPost } from './client';

/** Fetch the list of visible (active + beta) scenarios. */
export async function listScenarios(): Promise<ScenarioInfo[]> {
  return apiGet<ScenarioInfo[]>('/api/scenarios');
}

/** Trigger a scenario run with optional params. */
export async function runScenario(
  name: string,
  params: Record<string, unknown> = {},
): Promise<ScenarioRunResponse> {
  return apiPost<ScenarioRunResponse>(`/api/scenarios/${name}/run`, { params });
}
