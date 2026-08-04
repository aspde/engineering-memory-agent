import { useCallback, useEffect, useState } from 'react';
import type { ConnectorInfo, ConnectorLogEntry } from '../types';
import { getConnectorLogs, listConnectors } from '../api/connectors';

const STATUS_COLORS: Record<string, string> = {
  active: 'bg-green-100 text-green-700',
  pending: 'bg-yellow-100 text-yellow-700',
  error: 'bg-red-100 text-red-700',
};

const STATUS_LABELS: Record<string, string> = {
  active: '已激活',
  pending: '待配置',
  error: '异常',
};

const BATCH_LABELS: Record<string, string> = {
  supported: '批量就绪',
  pending: '逐条处理',
  not_applicable: '不适用',
};

/** Map source_type to a display icon (single-char placeholder). */
function sourceIcon(sourceType: string): string {
  if (sourceType.startsWith('pingcode')) return '📋';
  if (sourceType.startsWith('ci')) return '🔄';
  if (sourceType.startsWith('feishu')) return '💬';
  if (sourceType.startsWith('git')) return '📦';
  return '📄';
}

// ── Component ────────────────────────────────────────────────────────

export default function ConnectorsPage() {
  const [connectors, setConnectors] = useState<ConnectorInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Log viewer state
  const [selectedSource, setSelectedSource] = useState<string | null>(null);
  const [logs, setLogs] = useState<ConnectorLogEntry[]>([]);
  const [logsLoading, setLogsLoading] = useState(false);

  const fetchConnectors = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await listConnectors();
      setConnectors(data.connectors);
    } catch (err) {
      setError(err instanceof Error ? err.message : '加载连接器列表失败');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchConnectors();
  }, [fetchConnectors]);

  const openLogs = useCallback(async (source: string) => {
    setSelectedSource(source);
    setLogsLoading(true);
    try {
      const data = await getConnectorLogs(source, 50);
      setLogs(data.logs);
    } catch {
      setLogs([]);
    } finally {
      setLogsLoading(false);
    }
  }, []);

  const closeLogs = useCallback(() => {
    setSelectedSource(null);
    setLogs([]);
  }, []);

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-4xl px-6 py-8">
        <h2 className="text-2xl font-bold text-gray-900">🔌 连接器</h2>
        <p className="mt-1 text-sm text-gray-500">
          外部数据源接入状态与投递日志
        </p>

        {/* Error banner */}
        {error && (
          <div className="mt-4 rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
            <button
              type="button"
              onClick={() => void fetchConnectors()}
              className="ml-3 font-medium underline hover:no-underline"
            >
              重试
            </button>
          </div>
        )}

        {/* Loading skeleton */}
        {loading && (
          <div className="mt-6 space-y-4">
            {Array.from({ length: 3 }, (_, i) => (
              <div key={i} className="animate-pulse rounded-lg border border-gray-200 p-5">
                <div className="mb-3 h-5 w-32 rounded bg-gray-200" />
                <div className="h-4 w-48 rounded bg-gray-200" />
              </div>
            ))}
          </div>
        )}

        {/* Connector cards */}
        {!loading && connectors.length === 0 && !error && (
          <div className="mt-12 text-center text-gray-400">
            <p className="text-lg">暂无已注册的连接器</p>
            <p className="mt-1 text-sm">启动时注册的连接器将显示在这里</p>
          </div>
        )}

        {!loading && (
          <div className="mt-6 space-y-4">
            {connectors.map((c) => (
              <div
                key={c.source_type}
                className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm"
              >
                {/* Header */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="text-xl">{sourceIcon(c.source_type)}</span>
                    <div>
                      <h3 className="font-semibold text-gray-900">{c.display_name}</h3>
                      <p className="font-mono text-xs text-gray-400">{c.source_type}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span
                      className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${STATUS_COLORS[c.status] ?? 'bg-gray-100 text-gray-600'}`}
                    >
                      {STATUS_LABELS[c.status] ?? c.status}
                    </span>
                    <span className="rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-500">
                      {BATCH_LABELS[c.batch_mode] ?? c.batch_mode}
                    </span>
                  </div>
                </div>

                {/* Actions */}
                <div className="mt-3 flex items-center gap-3">
                  <button
                    type="button"
                    onClick={() => void openLogs(c.source_type)}
                    className="text-sm font-medium text-blue-600 hover:underline"
                  >
                    查看投递日志
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Log viewer modal */}
        {selectedSource && (
          <div className="mt-6 rounded-lg border border-gray-200 bg-white p-5">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="font-semibold text-gray-900">
                📋 投递日志 — {selectedSource}
              </h3>
              <button
                type="button"
                onClick={closeLogs}
                className="rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
              >
                <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            {logsLoading ? (
              <div className="animate-pulse space-y-2">
                {Array.from({ length: 3 }, (_, i) => (
                  <div key={i} className="h-8 rounded bg-gray-200" />
                ))}
              </div>
            ) : logs.length === 0 ? (
              <p className="py-4 text-center text-sm text-gray-400">暂无投递记录</p>
            ) : (
              <div className="max-h-96 space-y-1 overflow-y-auto">
                {logs.map((entry) => (
                  <div
                    key={entry.id}
                    className="flex items-center gap-3 rounded px-3 py-2 text-sm hover:bg-gray-50"
                  >
                    <span
                      className={`inline-block h-2 w-2 shrink-0 rounded-full ${
                        entry.status === 'processed' ? 'bg-green-400' : 'bg-red-400'
                      }`}
                    />
                    <span className="font-mono text-xs text-gray-400">
                      {entry.created_at?.slice(0, 19) ?? '—'}
                    </span>
                    <span className="truncate text-gray-600">
                      {entry.payload_summary || '(空)'}
                    </span>
                    {entry.error && (
                      <span className="shrink-0 text-xs text-red-500" title={entry.error}>
                        ⚠️
                      </span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
