import { describe, it, expect } from 'vitest';
import { formatToolResult } from './ToolCallPanel';

describe('formatToolResult', () => {
  it('extracts count for search_memories_tool with JSON envelope', () => {
    const json = JSON.stringify({
      display: 'Found 3 relevant memories',
      sources: [{ id: '1' }, { id: '2' }, { id: '3' }],
    });
    const result = formatToolResult('search_memories_tool', json);
    expect(result).toContain('Found **3** memories');
  });

  it('shows "No relevant memories found" for zero count', () => {
    const json = JSON.stringify({
      display: 'Found 0 relevant memories',
      sources: [],
    });
    const result = formatToolResult('search_memories_tool', json);
    expect(result).toContain('No relevant memories found');
  });

  it('falls back to raw content for non-JSON search result', () => {
    const result = formatToolResult('search_memories_tool', 'plain text result');
    expect(result).toBe('plain text result');
  });

  it('extracts count for retrieve_chunks_tool', () => {
    const json = JSON.stringify({
      display: 'Found 5 relevant chunks',
      sources: [],
    });
    const result = formatToolResult('retrieve_chunks_tool', json);
    expect(result).toContain('Found **5** chunks');
  });

  it('parses write_memory_tool JSON with action and summary', () => {
    const json = JSON.stringify({ action: 'inserted', summary: 'New memory about caching' });
    const result = formatToolResult('write_memory_tool', json);
    expect(result).toContain('New memory created');
    expect(result).toContain('New memory about caching');
  });

  it('parses extract_memory_tool JSON with entities and relations', () => {
    const json = JSON.stringify({
      summary: 'Test',
      entities: [{ name: 'A' }, { name: 'B' }, { name: 'C' }],
      relations: [{ from: 'A', to: 'B' }],
    });
    const result = formatToolResult('extract_memory_tool', json);
    expect(result).toContain('**3** entities');
    expect(result).toContain('**1** relations');
  });

  it('parses ingest_git_repo_tool with commit count', () => {
    const result = formatToolResult('ingest_git_repo_tool', 'Ingested 42 commits as memories');
    expect(result).toContain('Ingested **42** commits from Git repo');
  });

  it('parses ingest_document_tool with chunks and document_id', () => {
    const result = formatToolResult(
      'ingest_document_tool',
      "Ingested 10 chunks from document 'readme'.",
    );
    expect(result).toContain('Ingested **10** chunks');
    expect(result).toContain('readme');
  });

  it('returns truncated plain text for unknown tool', () => {
    const longText = 'x'.repeat(300);
    const result = formatToolResult('unknown_tool', longText);
    expect(result).toBe(longText.slice(0, 200));
  });

  it('handles invalid JSON for tools gracefully — falls back to raw', () => {
    const result = formatToolResult('write_memory_tool', 'not json');
    expect(result).toBe('not json');
  });
});
