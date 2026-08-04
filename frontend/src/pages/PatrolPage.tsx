import { useCallback, useEffect, useState } from 'react';
import { dismissFinding, getPatrolLog, listPatrolLogs, triggerPatrol } from '../api/patrol';
import type { PatrolLogDetail, PatrolLogSummary, PatrolFinding } from '../types';

const FINDING_GROUPS: Record<string, { label: string; color: string; emoji: string }> = {
  pattern_matches: { label: '模式匹配', color: 'border-red-400', emoji: '🔴' },
  knowledge_gaps: { label: '知识盲区', color: 'border-yellow-400', emoji: '🟡' },
  new_entities: { label: '新实体', color: 'border-blue-400', emoji: '🔵' },
  contradictions: { label: '矛盾发现', color: 'border-orange-400', emoji: '🟠' },
  decay_alerts: { label: '衰减预警', color: 'border-gray-400', emoji: '⚪' },
  entity_coverage: { label: '实体覆盖', color: 'border-teal-400', emoji: '🟢' },
};

const PATROL_TYPE_LABELS: Record<string, string> = {
  daily: '每日巡检',
  weekly: '每周巡检',
  event_driven: '事件驱动',
  manual: '手动触发',
};

const PAGE_SIZE = 20;

export default function PatrolPage() {
  const [logs, setLogs] = useState<PatrolLogSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [triggering, setTriggering] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<PatrolLogDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [dismissedIds, setDismissedIds] = useState<Set<string>>(new Set());

  const fetchLogs = useCallback(async (newOffset: number) => {
    setLoading(true);
    try {
      const data = await listPatrolLogs({ limit: PAGE_SIZE, offset: newOffset });
      setLogs(data.items);
      setTotal(data.total);
      setOffset(newOffset);
    } catch {
      setLogs([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchLogs(0);
  }, [fetchLogs]);

  const handleSelect = useCallback(async (id: string) => {
    setSelectedId(id);
    setDetailLoading(true);
    try {
      const data = await getPatrolLog(id);
      setDetail(data);
      setDismissedIds(new Set(data.dismissed_findings ?? []));
    } catch {
      setDetail(null);
    } finally {
      setDetailLoading(false);
    }
  }, []);

  const handleTrigger = useCallback(async (patrolType: string) => {
    setTriggering(true);
    try {
      await triggerPatrol(patrolType);
      // Refresh the log list after a short delay to let the patrol start
      setTimeout(() => {
        void fetchLogs(0);
      }, 1000);
    } finally {
      setTriggering(false);
    }
  }, [fetchLogs]);

  const handleDismiss = useCallback(async (findingKey: string) => {
    if (!selectedId) return;
    try {
      await dismissFinding(selectedId, findingKey);
      setDismissedIds((prev) => new Set(prev).add(findingKey));
    } catch {
      // silently fail — user can retry
    }
  }, [selectedId]);

  const hasPrev = offset > 0;
  const hasNext = offset + logs.length < total;

  return (
    <div className="flex h-full">
      {/* Left panel: log list */}
      <div className="w-80 shrink-0 border-r border-gray-200 flex flex-col bg-gray-50">
        <div className="px-4 py-3 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">巡检日志</h2>
          <div className="mt-2 flex gap-2">
            <button
              type="button"
              onClick={() => handleTrigger('daily')}
              disabled={triggering}
              className="rounded bg-blue-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-blue-700 disabled:opacity-50"
            >
              每日
            </button>
            <button
              type="button"
              onClick={() => handleTrigger('weekly')}
              disabled={triggering}
              className="rounded bg-violet-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-violet-700 disabled:opacity-50"
            >
              每周
            </button>
            <button
              type="button"
              onClick={() => handleTrigger('contradiction_scan')}
              disabled={triggering}
              className="rounded bg-orange-600 px-2.5 py-1 text-xs font-medium text-white hover:bg-orange-700 disabled:opacity-50"
            >
              矛盾
            </button>
          </div>
        </div>

        {/* Log rows */}
        <div className="flex-1 overflow-y-auto">
          {loading ? (
            <div className="p-4 space-y-3">
              {Array.from({ length: 5 }, (_, i) => (
                <div key={i} className="animate-pulse rounded bg-white p-3">
                  <div className="h-3 w-2/3 rounded bg-gray-200 mb-2" />
                  <div className="h-2 w-1/2 rounded bg-gray-200" />
                </div>
              ))}
            </div>
          ) : logs.length === 0 ? (
            <p className="p-4 text-sm text-gray-400">
              暂无巡检记录，点击上方按钮手动触发一次巡检
            </p>
          ) : (
            logs.map((log) => {
              const active = log.id === selectedId;
              return (
                <button
                  key={log.id}
                  type="button"
                  onClick={() => handleSelect(log.id)}
                  className={`w-full text-left px-3 py-2.5 border-b border-gray-100 transition-colors ${
                    active
                      ? 'bg-blue-50 border-l-2 border-l-blue-500'
                      : 'hover:bg-gray-100'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-medium text-gray-700">
                      {PATROL_TYPE_LABELS[log.patrol_type] ?? log.patrol_type}
                    </span>
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded-full ${
                        log.status === 'completed'
                          ? 'bg-green-100 text-green-700'
                          : log.status === 'running'
                            ? 'bg-blue-100 text-blue-700'
                            : 'bg-red-100 text-red-700'
                      }`}
                    >
                      {log.status === 'completed' ? '完成' : log.status === 'running' ? '运行中' : '失败'}
                    </span>
                  </div>
                  <div className="mt-0.5 text-[11px] text-gray-400">
                    {log.finding_count} 个发现 · {log.started_at ? new Date(log.started_at).toLocaleDateString('zh-CN') : ''}
                  </div>
                </button>
              );
            })
          )}
        </div>

        {/* Pagination */}
        {total > PAGE_SIZE && (
          <div className="px-3 py-2 border-t border-gray-200 flex justify-between text-xs text-gray-500">
            <button
              type="button"
              onClick={() => fetchLogs(offset - PAGE_SIZE)}
              disabled={!hasPrev}
              className="disabled:opacity-30 hover:text-gray-700"
            >
              ← 上一页
            </button>
            <span>{Math.floor(offset / PAGE_SIZE) + 1} / {Math.ceil(total / PAGE_SIZE)}</span>
            <button
              type="button"
              onClick={() => fetchLogs(offset + PAGE_SIZE)}
              disabled={!hasNext}
              className="disabled:opacity-30 hover:text-gray-700"
            >
              下一页 →
            </button>
          </div>
        )}
      </div>

      {/* Right panel: detail */}
      <div className="flex-1 overflow-y-auto p-5">
        {!selectedId && (
          <p className="text-gray-400 text-sm pt-10 text-center">
            ← 选择左侧的巡检记录查看详情
          </p>
        )}

        {detailLoading && (
          <div className="flex justify-center pt-10">
            <div className="animate-spin h-6 w-6 border-2 border-blue-500 border-t-transparent rounded-full" />
          </div>
        )}

        {detail && !detailLoading && (
          <div>
            <div className="mb-4">
              <h3 className="text-lg font-semibold text-gray-900">
                {PATROL_TYPE_LABELS[detail.patrol_type] ?? detail.patrol_type} 详情
              </h3>
              <p className="text-xs text-gray-400 mt-1">
                {detail.started_at ? new Date(detail.started_at).toLocaleString('zh-CN') : ''}
                {detail.completed_at ? ` → ${new Date(detail.completed_at).toLocaleString('zh-CN')}` : ''}
              </p>
            </div>

            {detail.findings && Object.keys(detail.findings).length > 0 ? (
              Object.entries(detail.findings).map(([groupKey, findings]) => {
                const group = FINDING_GROUPS[groupKey];
                if (!group || !Array.isArray(findings) || findings.length === 0) return null;
                return (
                  <div key={groupKey} className="mb-5">
                    <h4 className="text-sm font-semibold text-gray-700 mb-2">
                      {group.emoji} {group.label} ({findings.length})
                    </h4>
                    <div className="space-y-2">
                      {findings.map((f: PatrolFinding, i: number) => {
                        const fKey = f.id ? String(f.id) : `${groupKey}-${i}`;
                        const dismissed = dismissedIds.has(fKey);
                        return (
                          <div
                            key={fKey}
                            className={`rounded-lg border-l-4 bg-white p-3 shadow-sm ${group.color} ${
                              dismissed ? 'opacity-50' : ''
                            }`}
                          >
                            <div className="flex items-start justify-between gap-2">
                              <div className="min-w-0">
                                <p className="text-sm font-medium text-gray-800 truncate">
                                  {String(f.title ?? f.summary ?? `#${i + 1}`)}
                                </p>
                                {f.description && (
                                  <p className="text-xs text-gray-500 mt-0.5 line-clamp-2">
                                    {f.description}
                                  </p>
                                )}
                              </div>
                              <button
                                type="button"
                                onClick={() => handleDismiss(fKey)}
                                disabled={dismissed}
                                className={`shrink-0 text-xs px-2 py-0.5 rounded ${
                                  dismissed
                                    ? 'bg-gray-100 text-gray-400 cursor-default'
                                    : 'bg-gray-100 text-gray-500 hover:bg-red-50 hover:text-red-600'
                                }`}
                              >
                                {dismissed ? '已忽略' : '忽略'}
                              </button>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })
            ) : (
              <p className="text-sm text-gray-400">未发现需关注事项</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
