import type { Interrupt, Source, SSEEvent, ToolCall } from '../types';

const BASE_URL = ''; // Vite proxy handles /api/* → http://localhost:8000

/**
 * Build the extra headers common to every API request.
 *
 * When `VITE_EMA_API_KEY` is configured the request carries
 * `Authorization: Bearer <key>` (the backend's API-key guard).  When it is
 * not configured no auth header is sent, preserving backward compatibility
 * with dev setups that have no key.
 */
function authHeaders(base: Record<string, string>): Record<string, string> {
  const key = import.meta.env.VITE_EMA_API_KEY as string | undefined;
  if (key) {
    return { ...base, Authorization: `Bearer ${key}` };
  }
  return base;
}

/** Error thrown for HTTP-level failures (non-2xx responses). */
export class ApiError extends Error {
  readonly status: number;
  readonly statusText: string;

  constructor(status: number, statusText: string, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.statusText = statusText;
  }
}

/** Perform a GET request and parse the JSON response. */
async function apiGet<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      method: 'GET',
      headers: authHeaders({ Accept: 'application/json' }),
    });
  } catch (err) {
    throw new Error(
      `Network error fetching ${path}: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
  return handleJsonResponse<T>(response, path);
}

/** Perform a POST request with a JSON body and parse the JSON response. */
async function apiPost<T>(path: string, body: unknown): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      method: 'POST',
      headers: authHeaders({
        'Content-Type': 'application/json',
        Accept: 'application/json',
      }),
      body: JSON.stringify(body),
    });
  } catch (err) {
    throw new Error(
      `Network error posting to ${path}: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
  return handleJsonResponse<T>(response, path);
}

/** Perform a DELETE request and parse the JSON response. */
async function apiDelete<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      method: 'DELETE',
      headers: authHeaders({ Accept: 'application/json' }),
    });
  } catch (err) {
    throw new Error(
      `Network error deleting ${path}: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
  return handleJsonResponse<T>(response, path);
}

/**
 * Stream a POST response as Server-Sent Events.
 *
 * Uses fetch() + ReadableStream (EventSource does not support POST). The
 * backend emits `data: {json}\n\n` lines; each line is parsed into an
 * {@link SSEEvent} and yielded. Malformed lines are skipped, and network
 * failures are surfaced as thrown Errors.
 */
async function* apiSSE(path: string, body: unknown): AsyncGenerator<SSEEvent> {
  let response: Response;
  try {
    response = await fetch(`${BASE_URL}${path}`, {
      method: 'POST',
      headers: authHeaders({
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      }),
      body: JSON.stringify(body),
    });
  } catch (err) {
    throw new Error(
      `Network error connecting to SSE stream ${path}: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  if (!response.ok || !response.body) {
    const detail = await readErrorDetail(response);
    throw new ApiError(
      response.status,
      response.statusText,
      `SSE stream ${path} failed (${response.status}): ${detail}`,
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      // Decode with { stream: true } so multi-byte characters split across
      // chunk boundaries are buffered internally by TextDecoder.
      buffer += decoder.decode(value, { stream: true });

      let newlineIndex: number;
      while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, newlineIndex).replace(/\r$/, '');
        buffer = buffer.slice(newlineIndex + 1);
        const event = parseSSELine(line);
        if (event) yield event;
      }
    }

    // Flush any trailing content without a final newline.
    if (buffer.length > 0) {
      const event = parseSSELine(buffer);
      if (event) yield event;
    }
  } catch (err) {
    throw new Error(
      `SSE stream ${path} read error: ${err instanceof Error ? err.message : String(err)}`,
    );
  } finally {
    reader.releaseLock();
  }
}

/** Validate a non-OK response and convert it to a typed JSON body, else throw. */
async function handleJsonResponse<T>(response: Response, path: string): Promise<T> {
  if (!response.ok) {
    const detail = await readErrorDetail(response);
    throw new ApiError(
      response.status,
      response.statusText,
      `Request to ${path} failed (${response.status}): ${detail}`,
    );
  }
  try {
    return (await response.json()) as T;
  } catch (err) {
    throw new Error(
      `Invalid JSON response from ${path}: ${err instanceof Error ? err.message : String(err)}`,
    );
  }
}

/** Extract FastAPI's `detail` field (or empty string) from an error body. */
async function readErrorDetail(response: Response): Promise<string> {
  try {
    const data: unknown = await response.json();
    if (typeof data === 'object' && data !== null && 'detail' in data) {
      const detail = (data as { detail: unknown }).detail;
      return typeof detail === 'string' ? detail : JSON.stringify(detail);
    }
  } catch {
    // Body is not JSON — fall back to statusText.
  }
  return '';
}

/** Parse a single SSE line; returns null for empty/non-`data:`/malformed lines. */
export function parseSSELine(line: string): SSEEvent | null {
  if (!line.startsWith('data: ')) return null;
  const payload = line.slice('data: '.length).trim();
  if (!payload) return null;
  try {
    const data = JSON.parse(payload) as Record<string, unknown>;
    return normalizeSSEEvent(data);
  } catch {
    return null; // skip malformed lines
  }
}

/** Convert a raw SSE JSON payload into a typed {@link SSEEvent}. */
export function normalizeSSEEvent(data: Record<string, unknown>): SSEEvent {
  switch (data.type) {
    case 'token':
      return { type: 'token', content: String(data.content ?? '') };
    case 'node':
      return { type: 'node', node: String(data.node ?? '') };
    case 'interrupt':
      return { type: 'interrupt', data: (data.data ?? {}) as Interrupt };
    case 'meta':
      return {
        type: 'meta',
        tool_calls: Array.isArray(data.tool_calls) ? (data.tool_calls as ToolCall[]) : [],
        sources: Array.isArray(data.sources) ? (data.sources as Source[]) : [],
      };
    case 'error':
      return { type: 'error', message: String(data.message ?? 'Unknown error') };
    case 'done':
      return { type: 'done' };
    default:
      return { type: 'error', message: `Unknown SSE event type: ${String(data.type)}` };
  }
}

export { apiGet, apiPost, apiDelete, apiSSE };
