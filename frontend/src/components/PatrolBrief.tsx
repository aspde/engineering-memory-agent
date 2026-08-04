import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { listPatrolLogs } from '../api/patrol';
import type { PatrolLogSummary } from '../types';

export default function PatrolBrief() {
  const navigate = useNavigate();
  const [latestLog, setLatestLog] = useState<PatrolLogSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    listPatrolLogs({ limit: 1, patrol_type: 'daily' })
      .then((data) => {
        if (!cancelled) {
          setLatestLog(data.items[0] ?? null);
          setError(false);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setError(true);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (error) return null; // silent degrade — don't block the page

  if (loading) {
    return (
      <div className="animate-pulse rounded-lg border border-gray-200 bg-white p-4">
        <div className="h-3 w-20 rounded bg-gray-200 mb-2" />
        <div className="h-4 w-3/4 rounded bg-gray-200" />
      </div>
    );
  }

  if (!latestLog || latestLog.status === 'failed') {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-4">
        <p className="text-sm font-semibold text-gray-900 mb-1">🔍 今日简报</p>
        {!latestLog ? (
          <p className="text-xs text-gray-400">今日巡检尚未执行，预计 8:00 完成</p>
        ) : (
          <p className="text-xs text-red-400">最近一次巡检执行失败</p>
        )}
      </div>
    );
  }

  if (latestLog.finding_count === 0) {
    return (
      <div className="rounded-lg border border-gray-200 bg-white p-4">
        <p className="text-sm font-semibold text-gray-900 mb-1">🔍 今日简报</p>
        <p className="text-xs text-gray-400">今日巡检未发现需关注事项 ✅</p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <div className="flex items-center justify-between mb-2">
        <p className="text-sm font-semibold text-gray-900">🔍 今日简报</p>
        <button
          type="button"
          onClick={() => navigate('/patrol')}
          className="text-xs text-blue-600 hover:text-blue-700 font-medium"
        >
          查看全部 →
        </button>
      </div>
      <p className="text-xs text-gray-500">
        最近巡检 · {latestLog.finding_count} 个发现
        {latestLog.started_at
          ? ` · ${new Date(latestLog.started_at).toLocaleDateString('zh-CN')}`
          : ''}
      </p>
    </div>
  );
}
