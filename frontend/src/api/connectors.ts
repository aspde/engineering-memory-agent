import { apiGet } from './client';
import type {
  ConnectorListResponse,
  ConnectorLogEntry,
  ConnectorLogsResponse,
} from '../types';

/** Fetch all registered connectors with status. */
export async function listConnectors(): Promise<ConnectorListResponse> {
  return apiGet<ConnectorListResponse>('/api/connectors');
}

/** Fetch recent webhook delivery logs for a connector. */
export async function getConnectorLogs(
  source: string,
  limit = 50,
  offset = 0,
): Promise<ConnectorLogsResponse> {
  const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
  return apiGet<ConnectorLogsResponse>(
    `/api/connectors/${encodeURIComponent(source)}/logs?${params}`,
  );
}

/** Convert a raw log entry from the API into a typed object. */
export type { ConnectorLogEntry, ConnectorListResponse, ConnectorLogsResponse };
