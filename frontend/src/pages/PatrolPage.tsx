import { useCallback, useEffect, useState } from 'react';
import {
  dismissFinding,
  getPatrolLog,
  listPatrolLogs,
  queuePatrolConflict,
  triggerPatrol,
} from '../api/patrol';
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
  contradiction_scan: '矛盾扫描',
  event_driven: '事件驱动',
  manual: '手动触发',
};

/** Normalized key for a contradiction finding: sorted memory pair, matching the
 *  backend's LEAST/GREATEST dedup so (A,B) and (B,A) map to the same key. */
function patrolFindingKey(f: PatrolFinding): string | null {
  const a = String(f.memory_a_id ?? '');
  const b = String(f.memory_b_id ?? '');
  if (!a || !b) return null;
  return [a, b].sort().join(':');
}

/**
 * Human-readable title + description for a finding card, derived from the
 * fields each patrol type actually emits (see backend/service/prompts.py
 * patrol templates).  None of the finding schemas use the generic
 * title/summary/description fields the card used to assume — entity_coverage
 * has entity_name/missing_domains/recommendation, contradictions have the two
 * memory summaries + conflict_description, etc. — so without this mapping a
 * whole group would render as bare "#1 #2 …" placeholders.
 */
function findingCardText(
  f: PatrolFinding,
  groupKey: string,
): { title?: string; description?: string } {
  const recommendation = f.recommendation ? String(f.recommendation) : undefined;
  switch (groupKey) {
    case 'entity_coverage': {
      const missing = Array.isArray(f.missing_domains)
        ? (f.missing_domains as string[]).join('、')
        : '';
      const description = [
        missing ? `缺失领域：${missing}` : '',
        recommendation ? `建议：${recommendation}` : '',
      ]
        .filter(Boolean)
        .join(' · ');
      return {
        title: f.entity_name ? String(f.entity_name) : undefined,
        description: description || undefined,
      };
    }
    case 'knowledge_gaps':
      return {
        title: f.entity_name ? String(f.entity_name) : undefined,
        description:
          f.missing_domain
            ? `缺失领域：${String(f.missing_domain)}`
            : recommendation,
      };
    case 'new_entities':
      return {
        title: f.entity_name ? String(f.entity_name) : undefined,
        description: recommendation,
      };
    case 'contradictions':
      return {
        title:
          f.memory_a_summary || f.memory_b_summary
            ? `${f.memory_a_summary ?? '记忆 A'} ⇄ ${f.memory_b_summary ?? '记忆 B'}`
            : undefined,
        description: f.conflict_description ? String(f.conflict_description) : undefined,
      };
    case 'pattern_matches':
      return {
        title:
          f.new_summary || f.matched_summary
            ? `${f.new_summary ?? '新记忆'} ⇄ ${f.matched_summary ?? '历史记忆'}`
            : undefined,
        description: f.reason ? String(f.reason) : undefined,
      };
    case 'decay_alerts':
      return {
        title: f.summary ? String(f.summary) : f.memory_id ? String(f.memory_id) : undefined,
        description: recommendation,
      };
    default:
      // Unknown/extension groups: keep the generic fields the card used to read.
      return {
        title: f.title || f.summary ? String(f.title ?? f.summary) : undefined,
        description: f.description ? String(f.description) : undefined,
      };
  }
}

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
  const [arbitratingKey, setArbitratingKey] = useState<string | null>(null);
  const [arbitratedKeys, setArbitratedKeys] = useState<Set<string>>(new Set());
  const [arbitrateError, setArbitrateError] = useState<string | null>(null);
  const [filterType, setFilterType] = useState<string>('');
  // Latest run *today* per patrol type (for the "已执行过" hint under the
  // trigger buttons).  Informational only — triggering again stays allowed.
  const [todayRuns, setTodayRuns] = useState<
    Record<string, { started_at: string; status: string }>
  >({});

  const fetchLogs = useCallback(async (newOffset: number) => {
    setLoading(true);
    try {
      const data = await listPatrolLogs({ limit: PAGE_SIZE, offset: newOffset, patrol_type: filterType || undefined });
      setLogs(data.items);
      setTotal(data.total);
      setOffset(newOffset);
    } catch {
      setLogs([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, [filterType]);

  // Today's runs are derived from the newest logs (the endpoint returns them
  // newest-first): the first entry per type whose started_at falls on the
  // local date is "today's run" for that type.
  const fetchTodayRuns = useCallback(async () => {
    try {
      const data = await listPatrolLogs({ limit: 100 });
      const runs: Record<string, { started_at: string; status: string }> = {};
      const todayKey = new Date().toDateString();
      for (const log of data.items) {
        if (!log.started_at) continue;
        const runDate = new Date(log.started_at);
        if (runDate.toDateString() !== todayKey) continue;
        const prev = runs[log.patrol_type];
        if (!prev || runDate > new Date(prev.started_at)) {
          runs[log.patrol_type] = { started_at: log.started_at, status: log.status };
        }
      }
      setTodayRuns(runs);
    } catch {
      setTodayRuns({});
    }
  }, []);

  useEffect(() => {
    void fetchLogs(0);
    void fetchTodayRuns();
  }, [filterType]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleFilterChange = useCallback((type: string) => {
    setFilterType(type);
  }, []);

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
      // Refresh the log list + today's-runs hint after a short delay to let
      // the patrol start
      setTimeout(() => {
        void fetchLogs(0);
        void fetchTodayRuns();
      }, 1000);
    } finally {
      setTriggering(false);
    }
  }, [fetchLogs, fetchTodayRuns]);

  const handleDismiss = useCallback(async (findingKey: string) => {
    if (!selectedId) return;
    try {
      await dismissFinding(selectedId, findingKey);
      setDismissedIds((prev) => new Set(prev).add(findingKey));
    } catch {
      // silently fail — user can retry
    }
  }, [selectedId]);

  const handleArbitrate = useCallback(
    async (finding: PatrolFinding, findingKey: string) => {
      if (!selectedId) return;
      const key = patrolFindingKey(finding);
      if (!key) return;
      setArbitratingKey(key);
      setArbitrateError(null);
      try {
        const resp = await queuePatrolConflict(selectedId, finding);
        setArbitratedKeys((prev) => new Set(prev).add(key));
        if (resp.status === 'already_pending') {
          setArbitrateError('该矛盾已在待处理列表中');
        } else if (resp.status === 'already_resolved') {
          setArbitrateError('该矛盾已仲裁过（两者都保留）');
        }
        // The finding has been handed off to the arbitration queue — dismiss it
        // from the patrol so it stops showing as an open item.  Best-effort:
        // a failed dismiss only leaves the card visible, user can re-dismiss.
        try {
          await dismissFinding(selectedId, findingKey);
          setDismissedIds((prev) => new Set(prev).add(findingKey));
        } catch {
          // ignore — the arbitration already happened
        }
      } catch (err) {
        setArbitrateError(
          err instanceof Error && /409|conflict|Conflict/.test(err.message)
            ? '该矛盾已处理或记忆已删除，可忽略'
            : err instanceof Error
              ? err.message
              : '转入仲裁失败，请重试',
        );
      } finally {
        setArbitratingKey(null);
      }
    },
    [selectedId],
  );

  const hasPrev = offset > 0;
  const hasNext = offset + logs.length < total;

  return (
    <div className="flex h-full">
      {/* Left panel: log list */}
      <div className="w-80 shrink-0 border-r border-gray-200 flex flex-col bg-gray-50">
        <div className="px-4 py-3 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">巡检日志</h2>
          {/* Filter tabs */}
          <div className="mt-2 flex gap-1 flex-wrap">
            {[
              { key: '', label: '全部' },
              { key: 'daily', label: '每日巡检' },
              { key: 'weekly', label: '每周巡检' },
              { key: 'contradiction_scan', label: '矛盾扫描' },
            ].map(({ key, label }) => (
              <button
                key={key}
                type="button"
                onClick={() => handleFilterChange(key)}
                className={`rounded px-2.5 py-1 text-xs font-medium transition-colors ${
                  filterType === key
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          {/* Trigger buttons */}
          <div className="mt-2 flex gap-1.5 text-[10px] text-gray-400">
            <span className="py-1">执行：</span>
            <button
              type="button"
              onClick={() => handleTrigger('daily')}
              disabled={triggering}
              className="rounded border border-blue-200 px-2 py-0.5 text-blue-600 hover:bg-blue-50 disabled:opacity-40"
            >
              ▶ 每日
            </button>
            <button
              type="button"
              onClick={() => handleTrigger('weekly')}
              disabled={triggering}
              className="rounded border border-violet-200 px-2 py-0.5 text-violet-600 hover:bg-violet-50 disabled:opacity-40"
            >
              ▶ 每周
            </button>
            <button
              type="button"
              onClick={() => handleTrigger('contradiction_scan')}
              disabled={triggering}
              className="rounded border border-orange-200 px-2 py-0.5 text-orange-600 hover:bg-orange-50 disabled:opacity-40"
            >
              ▶ 矛盾
            </button>
          </div>
          {/* "已执行过" hint — informational, does not block re-triggering */}
          {Object.keys(todayRuns).length > 0 && (
            <div className="mt-2 space-y-1 border-t border-gray-200 pt-1.5">
              {(['daily', 'weekly', 'contradiction_scan'] as const).map((type) => {
                const run = todayRuns[type];
                if (!run) return null;
                const ok = run.status === 'completed';
                return (
                  <p key={type} className="text-[10px] text-gray-400">
                    今日{PATROL_TYPE_LABELS[type] ?? type}已执行 ·{' '}
                    {new Date(run.started_at).toLocaleTimeString('zh-CN', {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                    {ok ? ' · 完成' : ' · 未完成（可再次执行）'}
                  </p>
                );
              })}
            </div>
          )}
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

            {arbitrateError && (
              <p className="mb-4 rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-700">
                {arbitrateError}
              </p>
            )}

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
                        const arbitrateKey = patrolFindingKey(f);
                        const arbitrated = arbitrateKey ? arbitratedKeys.has(arbitrateKey) : false;
                        const isContradiction = groupKey === 'contradictions';
                        const card = findingCardText(f, groupKey);
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
                                  {card.title ??
                                    String(f.title ?? f.summary ?? `#${i + 1}`)}
                                </p>
                                {card.description && (
                                  <p className="text-xs text-gray-500 mt-0.5 line-clamp-2">
                                    {card.description}
                                  </p>
                                )}
                              </div>
                              <div className="flex shrink-0 items-center gap-1.5">
                                {isContradiction && arbitrateKey && (
                                  <button
                                    type="button"
                                    onClick={() => void handleArbitrate(f, fKey)}
                                    disabled={arbitrated || arbitratingKey === arbitrateKey}
                                    className={`shrink-0 text-xs px-2 py-0.5 rounded ${
                                      arbitrated
                                        ? 'bg-orange-100 text-orange-400 cursor-default'
                                        : 'bg-orange-50 text-orange-600 hover:bg-orange-100 disabled:opacity-50'
                                    }`}
                                  >
                                    {arbitrated
                                      ? '已转入仲裁'
                                      : arbitratingKey === arbitrateKey
                                        ? '处理中…'
                                        : '转入仲裁'}
                                  </button>
                                )}
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
