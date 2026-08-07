import { useLocation, useNavigate } from 'react-router-dom';

interface NavItem {
  path: string;
  icon: string;
  label: string;
}

const NAV_ITEMS: NavItem[] = [
  { path: '/', icon: '💬', label: '对话' },
  { path: '/memories', icon: '📚', label: '记忆库' },
  { path: '/graph', icon: '🔗', label: '实体图谱' },
  { path: '/connectors', icon: '🔌', label: '连接器' },
  { path: '/patrol', icon: '🔍', label: '巡检日志' },
  { path: '/conflicts', icon: '⚖️', label: '冲突' },
];

/**
 * 48 px pure-icon navigation bar on the far left.
 *
 * Each entry renders a button with a tooltip that appears on hover.
 * The active route is highlighted with a left border accent and a
 * subtle background tint.
 */
export default function NavBar() {
  const navigate = useNavigate();
  const { pathname } = useLocation();

  return (
    <nav className="flex w-12 shrink-0 flex-col items-center gap-1 border-r border-gray-200 bg-gray-50 py-4">
      {NAV_ITEMS.map((item) => {
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
