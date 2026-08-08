import type { ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAppDispatch } from '../context/AppContext';
import type { Source } from '../types';
import RichText from './RichText';

/**
 * Renders assistant answer text with inline source citations clickable.
 *
 * The agent.system prompt asks the model to cite source IDs inline, e.g.
 * ``（记忆 a1b2c3d4）`` or ``（文档 docs/architecture.md）``.  This component
 * splits the text on those citation spans:
 *
 * - **memory citations** that resolve to a source in ``sources`` render as a
 *   clickable chip → sets the memory filter and navigates to the memories
 *   page (the memory's full id is recovered from the source entry).
 * - unmatched memory citations and document citations (no dedicated page)
 *   render as an inert monospace chip so the reference stays visible.
 * - everything else is plain text rendered by ``RichText`` (the minimal
 *   inline-markdown renderer), so `` `code` ``, **bold**, and _italic_ keep
 *   working around citations.
 */
const CITATION_RE = /([（(]\s*(记忆|文档)[：: ]?)([0-9a-zA-Z_./-]+)(\s*[）)])/g;

export interface CitationPart {
  kind: 'text' | 'citation';
  text: string;
  refKind?: '记忆' | '文档';
  refId?: string;
}

/** Split *text* into plain-text and citation parts (exported for tests). */
export function splitCitations(text: string): CitationPart[] {
  const parts: CitationPart[] = [];
  const re = new RegExp(CITATION_RE.source, 'g');
  let lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > lastIndex) {
      parts.push({ kind: 'text', text: text.slice(lastIndex, m.index) });
    }
    parts.push({ kind: 'citation', text: m[0], refKind: m[2] as '记忆' | '文档', refId: m[3] });
    lastIndex = m.index + m[0].length;
  }
  if (lastIndex < text.length) {
    parts.push({ kind: 'text', text: text.slice(lastIndex) });
  }
  return parts;
}

/** The source whose memory id starts with *shortId* (the answer's citation). */
function resolveMemory(sources: Source[], shortId: string): Source | undefined {
  return sources.find(
    (s) =>
      s.type === 'memory' &&
      !!s.id &&
      (s.id === shortId || s.id.slice(0, 8) === shortId),
  );
}

interface CitationTextProps {
  text: string;
  sources: Source[];
}

export default function CitationText({ text, sources }: CitationTextProps) {
  const dispatch = useAppDispatch();
  const navigate = useNavigate();

  const parts = splitCitations(text);
  // Fast path: no citations — defer to RichText unchanged.
  if (parts.length === 1 && parts[0].kind === 'text') {
    return <RichText text={text} />;
  }

  const nodes: ReactNode[] = parts.map((p, i) => {
    if (p.kind === 'text') {
      return <RichText key={i} text={p.text} />;
    }
    if (p.refKind === '记忆') {
      const src = resolveMemory(sources, p.refId ?? '');
      if (src) {
        const fullId = src.id as string;
        return (
          <button
            key={i}
            type="button"
            onClick={() => {
              dispatch({ type: 'SET_MEM_FILTER', memId: fullId });
              navigate('/memories');
            }}
            className="mx-0.5 inline cursor-pointer rounded bg-emerald-50 px-1 py-0.5 font-mono text-[0.85em] text-emerald-700 transition-colors hover:bg-emerald-100"
            title="打开对应记忆"
          >
            {p.text}
          </button>
        );
      }
    }
    // Unmatched memory citation, or a document citation (no dedicated page).
    return (
      <span
        key={i}
        className="mx-0.5 rounded bg-gray-100 px-1 py-0.5 font-mono text-[0.85em] text-gray-600"
      >
        {p.text}
      </span>
    );
  });

  return <span className="whitespace-pre-wrap break-words">{nodes}</span>;
}
