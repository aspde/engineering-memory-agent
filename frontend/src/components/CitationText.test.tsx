import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import CitationText, { splitCitations } from './CitationText';
import type { Source } from '../types';

const navigate = vi.fn();
const dispatch = vi.fn();

vi.mock('react-router-dom', () => ({
  useNavigate: () => navigate,
}));
vi.mock('../context/AppContext', () => ({
  useAppDispatch: () => dispatch,
}));

const memorySources: Source[] = [
  { type: 'memory', id: 'c4a11b2e-0000-0000-0000-000000000000', summary: '选型记忆', relevance: 0.9 },
];

describe('splitCitations', () => {
  it('extracts a memory citation', () => {
    const parts = splitCitations('我们选用了 pgvector（记忆 c4a11b2e）');
    expect(parts).toHaveLength(2);
    expect(parts[0]).toEqual({ kind: 'text', text: '我们选用了 pgvector' });
    expect(parts[1]).toEqual({ kind: 'citation', text: '（记忆 c4a11b2e）', refKind: '记忆', refId: 'c4a11b2e' });
  });

  it('extracts a document citation', () => {
    const parts = splitCitations('详见（文档 docs/architecture.md）');
    expect(parts[1]).toEqual({ kind: 'citation', text: '（文档 docs/architecture.md）', refKind: '文档', refId: 'docs/architecture.md' });
  });

  it('extracts the English prompt format (memory:/document:) and normalizes the kind', () => {
    const parts = splitCitations('选型是 pgvector（memory: c4a11b2e）详见（document: docs/architecture.md）');
    const citations = parts.filter((p) => p.kind === 'citation');
    expect(citations).toEqual([
      { kind: 'citation', text: '（memory: c4a11b2e）', refKind: '记忆', refId: 'c4a11b2e' },
      { kind: 'citation', text: '（document: docs/architecture.md）', refKind: '文档', refId: 'docs/architecture.md' },
    ]);
  });

  it('handles multiple citations and surrounding text', () => {
    const parts = splitCitations('A（记忆 11111111）B（文档 doc.md）C');
    const citations = parts.filter((p) => p.kind === 'citation');
    expect(citations).toHaveLength(2);
    expect(parts[0].text).toBe('A');
  });

  it('returns a single text part when there are no citations', () => {
    const parts = splitCitations('没有引用的答案');
    expect(parts).toHaveLength(1);
    expect(parts[0]).toEqual({ kind: 'text', text: '没有引用的答案' });
  });

  it('ignores unmatched brackets (no 记忆/文档 marker)', () => {
    const parts = splitCitations('普通括号（hello world）');
    expect(parts).toHaveLength(1);
    expect(parts[0].kind).toBe('text');
  });
});

describe('CitationText', () => {
  beforeEach(() => {
    navigate.mockClear();
    dispatch.mockClear();
  });

  it('renders plain text when no citations present', () => {
    render(<CitationText text="hello" sources={[]} />);
    expect(screen.getByText('hello')).toBeInTheDocument();
  });

  it('renders a matching memory citation as a clickable button', () => {
    render(
      <CitationText text="选型是 pgvector（记忆 c4a11b2e）" sources={memorySources} />,
    );
    const btn = screen.getByRole('button', { name: '（记忆 c4a11b2e）' });
    expect(btn).toBeInTheDocument();
  });

  it('renders a matching memory citation in English format as a clickable button', () => {
    render(
      <CitationText text="选型是 pgvector（memory: c4a11b2e）" sources={memorySources} />,
    );
    const btn = screen.getByRole('button', { name: '（memory: c4a11b2e）' });
    expect(btn).toBeInTheDocument();
  });

  it('clicking a memory citation dispatches mem filter and navigates', () => {
    render(
      <CitationText text="选型是 pgvector（记忆 c4a11b2e）" sources={memorySources} />,
    );
    fireEvent.click(screen.getByRole('button', { name: '（记忆 c4a11b2e）' }));
    expect(dispatch).toHaveBeenCalledWith({
      type: 'SET_MEM_FILTER',
      memId: 'c4a11b2e-0000-0000-0000-000000000000',
    });
    expect(navigate).toHaveBeenCalledWith('/memories');
  });

  it('renders an unmatched memory citation as an inert chip (not a button)', () => {
    render(<CitationText text="（记忆 99999999）" sources={memorySources} />);
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
    expect(screen.getByText('（记忆 99999999）')).toBeInTheDocument();
  });

  it('renders a document citation as an inert chip', () => {
    render(<CitationText text="详见（文档 docs/architecture.md）" sources={[]} />);
    expect(screen.queryByRole('button')).not.toBeInTheDocument();
    expect(screen.getByText('（文档 docs/architecture.md）')).toBeInTheDocument();
  });

  it('preserves inline markdown around citations', () => {
    render(
      <CitationText text="**重点**（记忆 c4a11b2e）`code`" sources={memorySources} />,
    );
    expect(screen.getByText('重点').tagName).toBe('STRONG');
    expect(screen.getByText('code').tagName).toBe('CODE');
  });
});
