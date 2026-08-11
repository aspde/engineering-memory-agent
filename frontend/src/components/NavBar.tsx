import { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { probeCapability } from '../api/capabilities';

interface NavItem {
  path: string;
  icon: string;
  label: string;
  /** Breadth-layer probe path — the entry is hidden when it returns 404 (ADR-011). */
  probe?: string;
}

const NAV_ITEMS: NavItem[] = [
  { path: '/', icon: '💬', label: '对话' },
  { path: '/memories', icon: '📚', label: '记忆库' },
  { path: '/graph', icon: '🔗', label: '实体图谱' },
  { path: '/connectors', icon: '🔌', label: '连接器', probe: '/api/connectors' },
  { path: '/patrol', icon: '🔍', label: '巡检日志', probe: '/api/patrol/logs' },
  { path: '/conflicts', icon: '⚖️', label: '冲突' },
];

/**
 * 48 px pure-icon navigation bar on the far left.
 *
 * Each entry renders a button with a tooltip that appears on hover.
 * The active route is highlighted with a left border accent and a
 * subtle background tint.
 *
 * Breadth-layer entries (connectors, patrol) probe their backend endpoint
 * on mount and hide when it returns 404 — the route is not mounted because
 * the feature is disabled (`*_ENABLED=false`, ADR-011).  A 404 is FastAPI's
 * "route not registered" reply, which is served before the API-key check, so
 * the probe works without a configured key; any other outcome keeps the item.
 */
export default function NavBar() {
  const navigate = useNavigate();
  const { pathname } = useLocation();
  const [hidden, setHidden] = useState<Set<string>>(new Set());

  useEffect(() => {
    let cancelled = false;
    for (const item of NAV_ITEMS) {
      if (!item.probe) continue;
      probeCapability(item.probe).then((available) => {
        if (!cancelled && !available) {
          setHidden((prev) => new Set(prev).add(item.path));
        }
      });
    }
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <nav className="flex w-12 shrink-0 flex-col items-center gap-1 border-r border-gray-200 bg-gray-50 py-4">
      {NAV_ITEMS.filter((item) => !hidden.has(item.path)).map((item) => {
        const active = item.path === '/'
          ? pathname === '/'
          : pathname.startsWith(item.path);

        return (
          <div key={item.path} className="group relative">
            <button
              type="button"
              onClick={() => navigate(item.path)}
              className={`flex h-10 w-10 items-center justify-center rounded-lg text-lg transition-colors ${
                active
                  ? 'bg-blue-100 text-blue-700 ring-1 ring-blue-200'
                  : 'text-gray-500 hover:bg-gray-100 hover:text-gray-700'
              }`}
              aria-label={item.label}
              aria-current={active ? 'page' : undefined}
            >
              {item.icon}
            </button>

            {/* Tooltip — hidden by default, shown on group hover */}
            <span
              className="pointer-events-none absolute left-full top-1/2 z-50 ml-2 -translate-y-1/2
                         rounded bg-gray-800 px-2 py-1 text-xs text-white opacity-0
                         transition-opacity group-hover:opacity-100"
              style={{ whiteSpace: 'nowrap' }}
            >
              {item.label}
            </span>
          </div>
        );
      })}
    </nav>
  );
}
