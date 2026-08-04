import { useState, useCallback, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import { searchEntities, getEntityRelations } from '../api/entities';
import { getMemory } from '../api/memory';
import type {
  EntitySearchResult,
  EntityRelationsResponse,
  MemoryGetResponse,
} from '../types';

// ── Color & size by entity type ──────────────────────────────────────

const TYPE_COLORS: Record<string, string> = {
  technology: '#3B82F6', // blue
  person: '#8B5CF6',     // purple
  project: '#10B981',    // green
  decision: '#F59E0B',   // amber
  event: '#EF4444',      // red
  file: '#6B7280',       // gray
  concept: '#EC4899',    // pink
};

const TYPE_SIZES: Record<string, number> = {
  technology: 28,
  person: 24,
  project: 26,
  decision: 24,
  event: 22,
  file: 18,
  concept: 20,
};

function entityColor(type: string): string {
  return TYPE_COLORS[type] ?? '#9CA3AF';
}

function entitySize(type: string): number {
  return TYPE_SIZES[type] ?? 20;
}

// ── Graph node type ──────────────────────────────────────────────────

interface GraphNode {
  id: string;
  name: string;
  type: string;
  x: number;
  y: number;
  isCenter: boolean;
  relationType?: string;
  memoryCount?: number;
}

interface GraphEdge {
  from: string;
  to: string;
  label?: string;
}

// ── Simple force-directed layout for one-degree graph ────────────────

function layoutGraph(
  center: { id: string; name: string; type: string },
  related: { entity_id: string; name: string; type: string; relation_type: string; memory_count: number }[],
  width: number,
  height: number,
): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const cx = width / 2;
  const cy = height / 2;
  const nodes: GraphNode[] = [
    { id: center.id, name: center.name, type: center.type, x: cx, y: cy, isCenter: true },
  ];
  const edges: GraphEdge[] = [];

  const count = related.length;
  const radius = Math.min(width, height) * 0.35;

  related.forEach((rel, i) => {
    const angle = count > 1
      ? (2 * Math.PI * i) / count - Math.PI / 2
      : 0;
    nodes.push({
      id: rel.entity_id,
      name: rel.name,
      type: rel.type,
      x: cx + radius * Math.cos(angle),
      y: cy + radius * Math.sin(angle),
      isCenter: false,
      relationType: rel.relation_type,
      memoryCount: rel.memory_count,
    });
    edges.push({ from: center.id, to: rel.entity_id, label: rel.relation_type });
  });

  return { nodes, edges };
}

// ── Component ────────────────────────────────────────────────────────

export default function EntityGraphPage() {
  const [searchParams] = useSearchParams();
  const [query, setQuery] = useState('');
  const [searchResults, setSearchResults] = useState<EntitySearchResult[]>([]);
  const [searching, setSearching] = useState(false);
  const [graphData, setGraphData] = useState<EntityRelationsResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedMemory, setSelectedMemory] = useState<MemoryGetResponse | null>(null);

  // Auto-load from ?entity= query param (e.g. linked from chat page)
  useEffect(() => {
    const entityParam = searchParams.get('entity');
    if (entityParam) {
      setQuery(entityParam);
      // Fire and forget — auto-search on mount
      searchEntities(entityParam)
        .then((res) => {
          if (res.results.length > 0) {
            return getEntityRelations(res.results[0].id);
          }
          return null;
        })
        .then((data) => {
          if (data) setGraphData(data);
        })
        .catch(() => {
          setError('自动加载实体失败');
        });
    }
    // Only run on mount / when the ?entity= param changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams.get('entity')]);

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;
    setSearching(true);
    setError(null);
    try {
      const res = await searchEntities(query.trim());
      setSearchResults(res.results);
    } catch {
      setError('搜索实体失败');
    } finally {
      setSearching(false);
    }
  }, [query]);

  const handleSelectEntity = useCallback(async (entity: EntitySearchResult) => {
    setLoading(true);
    setError(null);
    setSearchResults([]);
    setSelectedMemory(null);
    try {
      const res = await getEntityRelations(entity.id);
      setGraphData(res);
    } catch {
      setError('加载实体关系失败');
    } finally {
      setLoading(false);
    }
  }, []);

  const handleNodeClick = useCallback(async (node: GraphNode) => {
    if (node.isCenter) return;
    setLoading(true);
    setError(null);
    setSelectedMemory(null);
    try {
      const res = await getEntityRelations(node.id);
      setGraphData(res);
    } catch {
      setError('加载实体关系失败');
    } finally {
      setLoading(false);
    }
  }, []);

  const handleMemoryClick = useCallback(async (memoryId: string) => {
    try {
      const mem = await getMemory(memoryId);
      setSelectedMemory(mem);
    } catch {
      setError('加载记忆详情失败');
    }
  }, []);

  const svgW = 700;
  const svgH = 450;

  const { nodes, edges } = graphData
    ? layoutGraph(
        { id: graphData.entity.id, name: graphData.entity.canonical_name, type: graphData.entity.type },
        graphData.related_entities,
        svgW,
        svgH,
      )
    : { nodes: [], edges: [] };

  return (
    <div className="flex h-full flex-col overflow-auto p-6">
      <h1 className="mb-4 text-xl font-bold text-gray-900">实体图谱</h1>

      {/* Search */}
      <div className="mb-4 flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
          placeholder="搜索实体名称…"
          className="flex-1 rounded-lg border border-gray-300 px-3 py-2 text-sm focus:border-blue-500 focus:outline-none"
        />
        <button
          type="button"
          onClick={handleSearch}
          disabled={searching}
          className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
        >
          {searching ? '搜索中…' : '搜索'}
        </button>
      </div>

      {/* Search results dropdown */}
      {searchResults.length > 0 && (
        <ul className="mb-4 rounded-lg border border-gray-200 bg-white shadow-sm">
          {searchResults.map((r) => (
            <li key={r.id}>
              <button
                type="button"
                onClick={() => handleSelectEntity(r)}
                className="flex w-full items-center gap-3 px-4 py-2 text-left text-sm hover:bg-gray-50"
              >
                <span
                  className="inline-block h-3 w-3 rounded-full"
                  style={{ backgroundColor: entityColor(r.type) }}
                />
                <span className="font-medium text-gray-900">{r.canonical_name}</span>
                <span className="text-gray-400">({r.type})</span>
                <span className="ml-auto text-gray-400">{r.memory_count} 条记忆</span>
              </button>
            </li>
          ))}
        </ul>
      )}

      {/* Error */}
      {error && (
        <div className="mb-4 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div className="flex flex-1 items-center justify-center">
          <p className="text-sm text-gray-400">加载中…</p>
        </div>
      )}

      {/* Graph */}
      {!loading && graphData && (
        <div className="flex gap-6">
          {/* SVG visualization */}
          <div className="rounded-lg border border-gray-200 bg-white p-2">
            <svg width={svgW} height={svgH} viewBox={`0 0 ${svgW} ${svgH}`}>
              {/* Edges */}
              {edges.map((edge, i) => {
                const from = nodes.find((n) => n.id === edge.from);
                const to = nodes.find((n) => n.id === edge.to);
                if (!from || !to) return null;
                const mx = (from.x + to.x) / 2;
                const my = (from.y + to.y) / 2;
                return (
                  <g key={`edge-${i}`}>
                    <line
                      x1={from.x}
                      y1={from.y}
                      x2={to.x}
                      y2={to.y}
                      stroke="#D1D5DB"
                      strokeWidth={1.5}
                    />
                    {edge.label && (
                      <text
                        x={mx}
                        y={my - 6}
                        textAnchor="middle"
                        className="fill-gray-400 text-[10px]"
                      >
                        {edge.label}
                      </text>
                    )}
                  </g>
                );
              })}

              {/* Nodes */}
              {nodes.map((node) => {
                const r = entitySize(node.type);
                const color = entityColor(node.type);
                return (
                  <g
                    key={node.id}
                    onClick={() => handleNodeClick(node)}
                    style={{ cursor: node.isCenter ? 'default' : 'pointer' }}
                  >
                    <circle
                      cx={node.x}
                      cy={node.y}
                      r={node.isCenter ? r + 4 : r}
                      fill={color}
                      stroke={node.isCenter ? '#1F2937' : '#FFF'}
                      strokeWidth={node.isCenter ? 3 : 2}
                      opacity={node.isCenter ? 1 : 0.85}
                    />
                    <text
                      x={node.x}
                      y={node.y + r + 14}
                      textAnchor="middle"
                      className="fill-gray-700 text-[11px]"
                    >
                      {node.name.length > 18
                        ? node.name.slice(0, 16) + '…'
                        : node.name}
                    </text>
                    {node.memoryCount !== undefined && !node.isCenter && (
                      <text
                        x={node.x}
                        y={node.y + r + 26}
                        textAnchor="middle"
                        className="fill-gray-400 text-[10px]"
                      >
                        {node.memoryCount} 条
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Side panel: recent memories */}
          <div className="min-w-0 flex-1">
            <div className="rounded-lg border border-gray-200 bg-white p-4">
              <h2 className="mb-3 text-sm font-semibold text-gray-900">
                {graphData.entity.canonical_name}
                <span className="ml-2 font-normal text-gray-400">
                  {graphData.entity.type} · {graphData.entity.memory_count} 条记忆
                </span>
              </h2>

              {selectedMemory ? (
                <div>
                  <button
                    type="button"
                    onClick={() => setSelectedMemory(null)}
                    className="mb-2 text-xs text-blue-600 hover:underline"
                  >
                    ← 返回列表
                  </button>
                  <div className="rounded-lg bg-gray-50 p-3">
                    <p className="text-sm text-gray-900">{selectedMemory.summary}</p>
                    <div className="mt-2 flex gap-2 text-xs text-gray-400">
                      <span>{selectedMemory.source_type}</span>
                      <span>{new Date(selectedMemory.created_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                </div>
              ) : (
                <div>
                  <h3 className="mb-2 text-xs font-medium text-gray-500">最近记忆</h3>
                  {graphData.recent_memories.length === 0 ? (
                    <p className="text-sm text-gray-400">暂无关联记忆</p>
                  ) : (
                    <ul className="space-y-2">
                      {graphData.recent_memories.map((mem) => (
                        <li key={mem.memory_id}>
                          <button
                            type="button"
                            onClick={() => handleMemoryClick(mem.memory_id)}
                            className="w-full rounded-lg border border-gray-100 p-2 text-left text-sm hover:bg-gray-50"
                          >
                            <p className="line-clamp-2 text-gray-700">{mem.summary}</p>
                            <p className="mt-1 text-xs text-gray-400">
                              {mem.source_type} · {new Date(mem.created_at).toLocaleDateString()}
                            </p>
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              )}
            </div>

            {/* Legend */}
            <div className="mt-4 rounded-lg border border-gray-200 bg-white p-4">
              <h3 className="mb-2 text-xs font-medium text-gray-500">图例</h3>
              <div className="flex flex-wrap gap-3">
                {Object.entries(TYPE_COLORS).map(([type, color]) => (
                  <div key={type} className="flex items-center gap-1.5">
                    <span
                      className="inline-block h-3 w-3 rounded-full"
                      style={{ backgroundColor: color }}
                    />
                    <span className="text-xs text-gray-600">{type}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!loading && !graphData && searchResults.length === 0 && (
        <div className="flex flex-1 items-center justify-center">
          <p className="text-sm text-gray-400">搜索实体名称以开始探索知识图谱</p>
        </div>
      )}
    </div>
  );
}
