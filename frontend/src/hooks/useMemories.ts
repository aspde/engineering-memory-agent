import { useCallback, useEffect, useState } from 'react';
import { deleteMemory, getMemory, getStats, ingest, searchMemories } from '../api/memory';
import type {
  IngestResponse,
  MemoryGetResponse,
  MemorySearchResponse,
  MemoryStatsResponse,
} from '../types';

/** Cache TTL in milliseconds (mirrors original Streamlit 60s `st.cache_data`). */
const STATS_CACHE_TTL_MS = 60_000;

/**
 * Module-level cache for the stats fetch so every hook instance (the page,
 * the search bar, the ingest panel) shares a single network request on first
 * render. After the TTL expires the cache is re-fetched on the next access.
 * `fetchStats()` deliberately bypasses the cache to force a refresh.
 */
let statsCache: Promise<MemoryStatsResponse> | null = null;
let statsCacheAt = 0;

/** Invalidate the stats cache so the next reader re-fetches from the API. */
export function invalidateStatsCache() {
  statsCacheAt = 0;
}

/** Read a file as text, falling back to latin-1 when it is not valid UTF-8. */
async function readFileContent(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  try {
    return new TextDecoder('utf-8', { fatal: true }).decode(buffer);
  } catch {
    return new TextDecoder('latin1').decode(buffer);
  }
}

export function useMemories() {
  // ── Stats (fetched once on mount, cached at module scope) ──
  const [stats, setStats] = useState<MemoryStatsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // ── Search ──
  const [searchResults, setSearchResults] = useState<
    MemorySearchResponse['results'] | null
  >(null);
  const [isSearching, setIsSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      if (!statsCache || Date.now() - statsCacheAt > STATS_CACHE_TTL_MS) {
        statsCache = getStats();
        statsCacheAt = Date.now();
      }
      try {
        const data = await statsCache;
        if (!cancelled) setStats(data);
      } catch (err) {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  /** Re-fetch stats from the backend (used by the retry button and after ingest). */
  const fetchStats = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getStats();
      statsCache = Promise.resolve(data);
      statsCacheAt = Date.now();
      setStats(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setIsLoading(false);
    }
  }, []);

  /** Semantic search over structured memories. */
  const search = useCallback(async (query: string, topK: number) => {
    setIsSearching(true);
    setSearchError(null);
    try {
      const data = await searchMemories(query, topK);
      setSearchResults(data.results);
    } catch (err) {
      setSearchError(err instanceof Error ? err.message : String(err));
      setSearchResults(null);
    } finally {
      setIsSearching(false);
    }
  }, []);

  /** Ingest raw text under an explicit document id. */
  const ingestText = useCallback(
    async (documentId: string, content: string): Promise<IngestResponse> =>
      ingest(documentId, content),
    [],
  );

  /** Ingest a file, decoding its contents as UTF-8 (latin-1 fallback). */
  const ingestFile = useCallback(
    async (file: File): Promise<IngestResponse> => {
      const content = await readFileContent(file);
      return ingest(file.name, content);
    },
    [],
  );

  /** Fetch a single memory by its id. */
  const getMemoryById = useCallback(
    async (id: string): Promise<MemoryGetResponse> => getMemory(id),
    [],
  );

  /** Delete a memory by its id. */
  const deleteMemoryById = useCallback(
    async (id: string): Promise<{ id: string; deleted: boolean }> => deleteMemory(id),
    [],
  );

  /** Remove a memory from the current search results by id. */
  const removeSearchResult = useCallback((id: string) => {
    setSearchResults(prev => {
      if (!prev) return null;
      const filtered = prev.filter(m => String(m.id) !== id);
      return filtered.length > 0 ? filtered : null;
    });
  }, []);

  return {
    stats,
    isLoading,
    error,
    fetchStats,
    searchResults,
    isSearching,
    searchError,
    search,
    ingestText,
    ingestFile,
    getMemoryById,
    deleteMemoryById,
    removeSearchResult,
  };
}
