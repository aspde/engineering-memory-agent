import { useNavigate } from 'react-router-dom';
import { useAppDispatch } from '../context/AppContext';
import type { Source } from '../types';

interface SourcesPanelProps {
  sources: Source[];
}

/**
 * Renders the "📚 Sources" panel attached to an assistant message.
 *
 * - `memory` → clickable button; clicking sets `memFilterId` and navigates
 *   to the memory library (mirrors Streamlit's `_render_source`).
 * - `chunk`  → collapsible expander revealing snippet / document_id / relevance.
 * - anything else → legacy plain-text caption.
 */
export default function SourcesPanel({ sources }: SourcesPanelProps) {
  const dispatch = useAppDispatch();
  const navigate = useNavigate();

  if (sources.length === 0) return null;

  const handleMemoryClick = (id?: string) => {
    if (!id) return;
    dispatch({ type: 'SET_MEM_FILTER', memId: id });
    navigate('/memories');
  };

  const renderSource = (s: Source) => {
    const stype = s.type ?? 'unknown';

    if (stype === 'memory') {
      const memId = s.id;
      const summary = (s.summary || s.snippet || '(无摘要)').slice(0, 120);
      const relevance = s.relevance;
      const label = `🧠 ${summary}`;
      const rel = relevance !== undefined ? ` — 相关度 ${relevance.toFixed(2)}` : '';
      if (memId) {
        return (
          <button
            type="button"
            onClick={() => handleMemoryClick(memId)}
            className="w-full rounded-lg bg-gray-50 px-2 py-1.5 text-left text-sm text-blue-700 transition-colors hover:bg-blue-50"
          >
            {label}
            {rel}
          </button>
        );
      }
      return (
        <p className="text-sm text-gray-600">
          {label}
          {rel}
        </p>
      );
    }

    if (stype === 'chunk') {
      const snippet = s.snippet ?? '';
      const docId = s.document_id ?? '';
      const relevance = s.relevance;
      const title = docId ? `📄 ${docId}` : '📄 片段';
      return (
        <details className="group">
          <summary className="cursor-pointer select-none text-sm font-medium text-gray-700">
            {title}
          </summary>
          <div className="mt-1.5 space-y-1 rounded bg-gray-50 p-2 text-xs text-gray-600">
            {docId && (
              <p>
                document_id: <code className="rounded bg-gray-200 px-1">{docId}</code>
              </p>
            )}
            {relevance !== undefined && <p>相关度: {relevance.toFixed(2)}</p>}
            <pre className="whitespace-pre-wrap">{snippet}</pre>
          </div>
        </details>
      );
    }

    return (
      <p className="text-xs text-gray-500">
        {`${stype} — ${(s.snippet ?? '').slice(0, 200) || JSON.stringify(s).slice(0, 200)}`}
      </p>
    );
  };

  return (
    <div className="w-full rounded-xl border border-gray-200 bg-white">
      <div className="rounded-t-xl border-b border-gray-200 px-3 py-2 text-xs font-semibold text-gray-600">
        📚 来源
      </div>
      <ul className="divide-y divide-gray-100">
        {sources.map((s, i) => (
          <li key={i} className="px-3 py-2">
            {renderSource(s)}
          </li>
        ))}
      </ul>
    </div>
  );
}
