import type { MemoryStatsResponse } from '../types';

interface StatsDashboardProps {
  stats: MemoryStatsResponse | null;
  isLoading: boolean;
  error: string | null;
  onRetry: () => void;
}

function KpiCard({
  label,
  value,
  format,
}: {
  label: string;
  value: number;
  format?: 'number' | 'percent';
}) {
  const display =
    format === 'percent'
      ? `${(value * 100).toFixed(0)}%`
      : value.toLocaleString();
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <p className="text-sm text-gray-500">{label}</p>
      <p className="mt-1 text-2xl font-bold text-gray-900">{display}</p>
    </div>
  );
}

function SkeletonCard() {
  return (
    <div className="animate-pulse rounded-lg border border-gray-200 bg-white p-4">
      <div className="h-3 w-16 rounded bg-gray-200" />
      <div className="mt-2 h-7 w-20 rounded bg-gray-200" />
    </div>
  );
}

export default function StatsDashboard({ stats, isLoading, error, onRetry }: StatsDashboardProps) {
  if (isLoading) {
    return (
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        {Array.from({ length: 4 }, (_, i) => (
          <SkeletonCard key={i} />
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-200 bg-red-50 p-4">
        <p className="text-sm text-red-700">无法加载统计数据：{error}</p>
        <button
          type="button"
          onClick={onRetry}
          className="mt-3 rounded-lg bg-red-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-red-700"
        >
          重试
        </button>
      </div>
    );
  }

  if (!stats) {
    return <p className="py-6 text-center text-sm text-gray-400">暂无数据</p>;
  }

  const bySource = stats.by_source_type ?? [];
  const topEntities = stats.top_entities ?? [];
  const maxCount = bySource.reduce((max, item) => Math.max(max, item.count), 0);

  return (
    <div>
      {/* KPI row */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <KpiCard label="总记忆数" value={stats.total_memories} />
        <KpiCard label="文档块数" value={stats.total_chunks} />
        <KpiCard label="对话数" value={stats.total_conversations} />
        <KpiCard label="近 7 天新增" value={stats.recent_count_7d} />
      </div>

      {/* Entity graph KPI row */}
      {stats.entity_graph && (
        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <KpiCard
            label="实体总数"
            value={stats.entity_graph.total_entities}
          />
          <KpiCard
            label="图谱覆盖率"
            value={stats.entity_graph.coverage_ratio}
            format="percent"
          />
          <KpiCard
            label="实体密度"
            value={stats.entity_graph.density}
          />
          <KpiCard
            label="7 日增长率"
            value={stats.entity_graph.growth_rate_7d}
            format="percent"
          />
        </div>
      )}

      <div className="mt-6 grid grid-cols-1 gap-6 md:grid-cols-2">
        {/* Source distribution */}
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h3 className="mb-3 text-sm font-semibold text-gray-900">来源分布</h3>
          {bySource.length === 0 ? (
            <p className="text-sm text-gray-400">暂无数据</p>
          ) : (
            <ul className="space-y-3">
              {bySource.map((item) => {
                const ratio = maxCount > 0 ? item.count / maxCount : 0;
                return (
                  <li key={item.source_type} className="flex items-center gap-3">
                    <span className="w-20 shrink-0 truncate font-mono text-xs text-gray-600">
                      {item.source_type}
                    </span>
                    <progress
                      value={ratio}
                      max={1}
                      className="h-2 w-28 shrink-0 appearance-none overflow-hidden rounded-full bg-gray-200 [&::-webkit-progress-bar]:rounded-full [&::-webkit-progress-bar]:bg-gray-200 [&::-webkit-progress-value]:rounded-full [&::-webkit-progress-value]:bg-blue-500 [&::-moz-progress-bar]:rounded-full [&::-moz-progress-bar]:bg-blue-500"
                    />
                    <span className="w-8 shrink-0 text-right text-sm font-semibold text-gray-900">
                      {item.count}
                    </span>
                  </li>
                );
              })}
            </ul>
          )}
        </div>

        {/* Top entities */}
        <div className="rounded-lg border border-gray-200 bg-white p-4">
          <h3 className="mb-3 text-sm font-semibold text-gray-900">高频实体</h3>
          {topEntities.length === 0 ? (
            <p className="text-sm text-gray-400">暂无数据</p>
          ) : (
            <div className="flex flex-wrap gap-2">
              {topEntities.map((entity) => (
                <span
                  key={entity.name}
                  className="rounded-full bg-gray-200 px-3 py-1 text-sm text-gray-800"
                >
                  {entity.name}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
