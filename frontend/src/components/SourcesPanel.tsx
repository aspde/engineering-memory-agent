import { useNavigate } from 'react-router-dom';
import { useAppDispatch } from '../context/AppContext';
import type { EntityRef, Source } from '../types';

interface SourcesPanelProps {
  sources: Source[];
}

const MAX_VISIBLE = 5;

/** Deduplicate entities across all sources by entity_id. */
function uniqueEntities(sources: Source[]): EntityRef[] {
  const seen = new Set<string>();
  const result: EntityRef[] = [];
  for (const s of sources) {
    for (const e of s.entities ?? []) {
      if (!seen.has(e.entity_id)) {
        seen.add(e.entity_id);
        result.push(e);
      }
    }
  }
  return result;
}

/**
 * Collapsible "📚 Sources" panel.
 *
 * - Deduplicated entity chips at top (memory sources only) → click to open
 *   entity graph.
 * - Sorts memory + chunk sources by relevance (descending), top 5 visible.
 * - Memory sources are clickable (navigate to the memory library page);
 *   chunk sources are read-only rows that show the document ID the answer's
 *   inline citation refers to.
 */
export default function SourcesPanel({ sources }: SourcesPanelProps) {
  const dispatch = useAppDispatch();
  const navigate = useNavigate();

  const sorted = sources
    .filter((s) => s.type === 'memory' || s.type === 'chunk')
    .sort((a, b) => (b.relevance ?? 0) - (a.relevance ?? 0));

  if (sorted.length === 0) return null;

  const entities = uniqueEntities(sorted.filter((s) => s.type === 'memory'));
  const visible = sorted.slice(0, MAX_VISIBLE);
  const overflow = sorted.length - MAX_VISIBLE;

  return (
    <details className="group w-full rounded-xl border border-gray-200 bg-white">
      <summary className="cursor-pointer select-none rounded-xl px-3 py-2 text-xs font-semibold text-gray-600">
        📚 来源（{sorted.length} 条）
      </summary>

      {/* Entity chips — deduplicated, shown once at top */}
      {entities.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5 border-t border-gray-100 px-3 py-2">
          <span className="text-xs text-gray-400">🔗 关联实体</span>
          {entities.map((e) => (
            <button
              key={e.entity_id}
              type="button"
              onClick={() => navigate(`/graph?entity=${encodeURIComponent(e.canonical_name)}`)}
              className="rounded-full bg-emerald-50 px-2 py-0.5 text-xs text-emerald-700 transition-colors hover:bg-emerald-100"
            >
              {e.canonical_name}
            </button>
          ))}
        </div>
      )}

      <ul className="divide-y divide-gray-100 border-t border-gray-100">
        {visible.map((s, i) => (
          <li key={i} className="px-3 py-2">
            {s.type === 'memory' ? (
              <button
                type="button"
                onClick={() => {
                  if (!s.id) return;
                  dispatch({ type: 'SET_MEM_FILTER', memId: s.id });
                  navigate('/memories');
                }}
                className="w-full rounded-lg bg-gray-50 px-2 py-1.5 text-left text-sm text-blue-700 transition-colors hover:bg-blue-50"
              >
                🧠{' '}
                {s.id ? (
                  <span className="font-mono text-xs text-gray-400">
                    [{s.id.slice(0, 8)}]{' '}
                  </span>
                ) : null}
                {(s.summary || s.snippet || '(无摘要)').slice(0, 120)}
                {s.relevance !== undefined ? ` — 相关度 ${s.relevance.toFixed(2)}` : ''}
              </button>
            ) : (
              <div className="w-full rounded-lg bg-gray-50 px-2 py-1.5 text-left text-sm text-gray-700">
                📄{' '}
                {s.document_id ? (
                  <span className="font-mono text-xs text-gray-400">
                    [{s.document_id}]{' '}
                  </span>
                ) : null}
                {(s.snippet || '(无摘要)').slice(0, 120)}
                {s.relevance !== undefined ? ` — 相关度 ${s.relevance.toFixed(2)}` : ''}
              </div>
            )}
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
