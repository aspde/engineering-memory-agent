import { useRef, type ReactNode } from 'react';

let _richTextId = 0;

/**
 * Minimal inline-markdown renderer.
 *
 * Handles the small subset of markdown that the backend/agent emits:
 * `` `code` ``, `**bold**`, and `_italic_`. Everything else (including
 * newlines) is preserved as plain text. Built to avoid pulling in a full
 * markdown dependency for the MVP.
 */
function renderInline(text: string, keyPrefix: string): ReactNode[] {
  const pattern = /(`[^`]+`)|(\*\*[^*]+\*\*)|(_[^_]+_)/g;
  const nodes: ReactNode[] = [];
  let lastIndex = 0;
  let i = 0;
  let match: RegExpExecArray | null;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }
    if (match[1] !== undefined) {
      nodes.push(
        <code
          key={`${keyPrefix}-c${i}`}
          className="rounded bg-gray-100 px-1 py-0.5 font-mono text-[0.85em]"
        >
          {match[1].slice(1, -1)}
        </code>,
      );
    } else if (match[2] !== undefined) {
      nodes.push(<strong key={`${keyPrefix}-b${i}`}>{match[2].slice(2, -2)}</strong>);
    } else if (match[3] !== undefined) {
      nodes.push(<em key={`${keyPrefix}-i${i}`}>{match[3].slice(1, -1)}</em>);
    }
    lastIndex = match.index + match[0].length;
    i += 1;
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }
  return nodes;
}

interface RichTextProps {
  text: string;
}

export default function RichText({ text }: RichTextProps) {
  // Stable unique prefix per component instance — avoids React key
  // collisions when two RichText instances are siblings in the tree.
  const prefixRef = useRef<string | null>(null);
  if (prefixRef.current === null) {
    prefixRef.current = `rt${_richTextId++}`;
  }

  return (
    <span className="whitespace-pre-wrap break-words">{renderInline(text, prefixRef.current)}</span>
  );
}
