import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import type {
  IngestResponse,
  MemoryGetResponse,
  MemoryStatsResponse,
} from '../types';

// Mock the API
vi.mock('../api/memory', () => ({
  getStats: vi.fn(),
  searchMemories: vi.fn(),
  ingest: vi.fn(),
  getMemory: vi.fn(),
}));

type Mock = ReturnType<typeof vi.fn>;

// Re-imported after `vi.resetModules()` in beforeEach so the module-level
// stats cache (`statsCache` / `statsCacheAt`) is fresh for every test, and the
// API mock references match the ones the freshly-imported hook uses.
let useMemories: typeof import('./useMemories').useMemories;
let getStats: Mock;
let searchMemories: Mock;
let ingest: Mock;
let getMemory: Mock;

const mockStats: MemoryStatsResponse = {
  total_memories: 10,
  total_chunks: 25,
  total_conversations: 3,
  by_source_type: [{ source_type: 'github', count: 5 }],
  avg_recall_count: 3,
  avg_entities_per_memory: 2,
  avg_relations_per_memory: 1,
  recent_count_7d: 4,
  top_entities: [{ name: 'foo', count: 3 }],
  entity_graph: null,
};

beforeEach(async () => {
  vi.clearAllMocks();
  vi.resetModules(); // Clear the module-level stats cache.
  const memoryApi = await import('../api/memory');
  getStats = memoryApi.getStats as unknown as Mock;
  searchMemories = memoryApi.searchMemories as unknown as Mock;
  ingest = memoryApi.ingest as unknown as Mock;
  getMemory = memoryApi.getMemory as unknown as Mock;
  const hooks = await import('./useMemories');
  useMemories = hooks.useMemories;
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.useRealTimers();
});

describe('useMemories', () => {
  describe('fetchStats', () => {
    it('calls getStats and sets stats on success', async () => {
      getStats.mockResolvedValue(mockStats);
      const { result } = renderHook(() => useMemories());
      await act(async () => {}); // flush the mount auto-fetch

      getStats.mockClear(); // isolate fetchStats' own call
      await act(async () => {
        await result.current.fetchStats();
      });

      expect(getStats).toHaveBeenCalledTimes(1);
      expect(result.current.stats).toEqual(mockStats);
      expect(result.current.error).toBeNull();
      expect(result.current.isLoading).toBe(false);
    });

    it('sets the error on failure', async () => {
      getStats.mockRejectedValue(new Error('boom'));
      const { result } = renderHook(() => useMemories());
      await act(async () => {});

      getStats.mockClear();
      await act(async () => {
        await result.current.fetchStats();
      });

      expect(result.current.error).toBe('boom');
      expect(result.current.stats).toBeNull();
      expect(result.current.isLoading).toBe(false);
    });
  });

  describe('stats auto-fetch and module-level cache', () => {
    it('auto-fetches stats on mount', async () => {
      getStats.mockResolvedValue(mockStats);
      const { result } = renderHook(() => useMemories());

      expect(getStats).toHaveBeenCalledTimes(1);
      await act(async () => {});

      expect(result.current.stats).toEqual(mockStats);
      expect(result.current.isLoading).toBe(false);
    });

    it('reuses the module-level cache within the TTL window', async () => {
      getStats.mockResolvedValue(mockStats);

      renderHook(() => useMemories());
      await act(async () => {});
      expect(getStats).toHaveBeenCalledTimes(1);

      const second = renderHook(() => useMemories());
      await act(async () => {});

      expect(getStats).toHaveBeenCalledTimes(1); // cached — no second network call
      expect(second.result.current.stats).toEqual(mockStats);
    });

    it('refetches stats once the cache TTL expires', async () => {
      const originalNow = Date.now.bind(Date);
      let offset = 0;
      vi.spyOn(Date, 'now').mockImplementation(() => originalNow() + offset);

      getStats.mockResolvedValue(mockStats);

      renderHook(() => useMemories());
      await act(async () => {});
      expect(getStats).toHaveBeenCalledTimes(1);

      offset = 60_001; // past the 60s TTL
      const second = renderHook(() => useMemories());
      await act(async () => {});

      expect(getStats).toHaveBeenCalledTimes(2);
      expect(second.result.current.stats).toEqual(mockStats);
    });
  });

  describe('search', () => {
    it('calls searchMemories and stores the results', async () => {
      searchMemories.mockResolvedValue({ results: [{ id: 'm1' }] });
      const { result } = renderHook(() => useMemories());

      await act(async () => {
        await result.current.search('test query', 3);
      });

      expect(searchMemories).toHaveBeenCalledWith('test query', 3);
      expect(result.current.searchResults).toEqual([{ id: 'm1' }]);
      expect(result.current.isSearching).toBe(false);
      expect(result.current.searchError).toBeNull();
    });

    it('sets searchError on failure', async () => {
      searchMemories.mockRejectedValue(new Error('search failed'));
      const { result } = renderHook(() => useMemories());

      await act(async () => {
        await result.current.search('test query', 3);
      });

      expect(result.current.searchError).toBe('search failed');
      expect(result.current.searchResults).toBeNull();
      expect(result.current.isSearching).toBe(false);
    });
  });

  describe('ingest', () => {
    it('calls ingest with the document id and content', async () => {
      ingest.mockResolvedValue({ document_id: 'doc-1', chunks_written: 3 });
      const { result } = renderHook(() => useMemories());

      let resp: IngestResponse | undefined;
      await act(async () => {
        resp = await result.current.ingestText('doc-1', 'hello world');
      });

      expect(ingest).toHaveBeenCalledWith('doc-1', 'hello world');
      expect(resp).toEqual({ document_id: 'doc-1', chunks_written: 3 });
    });
  });

  describe('getMemoryById', () => {
    it('fetches and returns a single memory', async () => {
      const memory: MemoryGetResponse = {
        id: 'm-123',
        source_type: 'github',
        summary: 'summary',
        entities: [],
        relations: [],
        recall_count: 1,
        meta: {},
        created_at: '2026-01-01T00:00:00Z',
      };
      getMemory.mockResolvedValue(memory);
      const { result } = renderHook(() => useMemories());

      let mem: MemoryGetResponse | undefined;
      await act(async () => {
        mem = await result.current.getMemoryById('m-123');
      });

      expect(getMemory).toHaveBeenCalledWith('m-123');
      expect(mem).toEqual(memory);
    });
  });

  describe('cleanup', () => {
    it('cancels an in-flight stats fetch on unmount', async () => {
      let resolveStats!: (value: MemoryStatsResponse) => void;
      const statsPromise = new Promise<MemoryStatsResponse>((r) => (resolveStats = r));
      getStats.mockReturnValue(statsPromise);

      const { result, unmount } = renderHook(() => useMemories());
      expect(result.current.isLoading).toBe(true);

      unmount();

      await act(async () => {
        resolveStats(mockStats);
        await statsPromise;
      });

      // The cancelled loader must not touch state after unmount.
      expect(result.current.stats).toBeNull();
      expect(result.current.isLoading).toBe(true);
    });
  });
});
