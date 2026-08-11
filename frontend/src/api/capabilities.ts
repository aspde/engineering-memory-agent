import { ApiError, apiGet } from './client';

/**
 * Probe a breadth-layer endpoint to decide whether its nav entry should show.
 *
 * The backend mounts breadth-layer routes (connectors, patrol, scenarios)
 * only when the corresponding `*_ENABLED` flag is on (ADR-011); an unmounted
 * route returns 404, which FastAPI serves *before* the API-key dependency
 * runs — so 404 reliably means "feature disabled" even without a configured
 * key.  Everything else (200, 401, 5xx, network error) is treated as
 * available: a broken or unauthorized backend must never hide an enabled
 * feature.
 */
export async function probeCapability(path: string): Promise<boolean> {
  try {
    await apiGet(path);
    return true;
  } catch (err) {
    return !(err instanceof ApiError && err.status === 404);
  }
}
