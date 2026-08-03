import { useNavigate } from 'react-router-dom';
import { useAppDispatch } from '../context/AppContext';
import type { Source } from '../types';

interface SourcesPanelProps {
  sources: Source[];
}

const MAX_VISIBLE = 5;

/**
 * Collapsible "📚 Sources" panel.
 *
 * - Sorts memory sources by relevance (descending).
 * - Shows the top 5 by default; rest expand on click.
 * - Clicking a source navigates to the memory library page with the
 *   source pre-selected as a filter.
 */
export default function SourcesPanel({ sources }: SourcesPanelProps) {
  const dispatch = useAppDispatch();
  const navigate = useNavigate();

  const sorted = sources
    .filter((s) => s.type === 'memory')
    .sort((a, b) => (b.relevance ?? 0) - (a.relevance ?? 0));

  if (sorted.length === 0) return null;

  const visible = sorted.slice(0, MAX_VISIBLE);
  const overflow = sorted.length - MAX_VISIBLE;

  const handleClick = (id?: string) => {
    if (!id) return;
    dispatch({ type: 'SET_MEM_FILTER', memId: id });
    navigate('/memories');
  };

  const renderItem = (s: Source) => {
    const summary = (s.summary || s.snippet || '(无摘要)').slice(0, 120);
    const rel =
      s.relevance !== undefined ? ` — 相关度 ${s.relevance.toFixed(2)}` : '';
    const label = `🧠 ${summary}${rel}`;

    if (s.id) {
      return (
        <button
          type="button"
          onClick={() => handleClick(s.id)}
          className="w-full rounded-lg bg-gray-50 px-2 py-1.5 text-left text-sm text-blue-700 transition-colors hover:bg-blue-50"
        >
          {label}
        </button>
      );
    }
    return <p className="text-sm text-gray-600">{label}</p>;
  };

  return (
    <details className="group w-full rounded-xl border border-gray-200 bg-white">
      <summary className="cursor-pointer select-none rounded-xl px-3 py-2 text-xs font-semibold text-gray-600">
        📚 来源（{sorted.length} 条）
      </summary>
      <ul className="divide-y divide-gray-100 border-t border-gray-100">
        {visible.map((s, i) => (
          <li key={i} className="px-3 py-2">
            {renderItem(s)}
          </li>
        ))}
        {overflow > 0 && (
          <li className="px-3 py-2 text-center text-xs text-gray-400">
            还有 {overflow} 条未显示（按相关度排序，仅展示前 {MAX_VISIBLE} 条）
          </li>
        )}
      </ul>
    </details>
  );
}
