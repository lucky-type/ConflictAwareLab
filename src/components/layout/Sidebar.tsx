import { Link, useLocation } from 'react-router-dom';
import {
  Atom,
  Beaker,
  Brain,
  Gauge,
  Layers3,
  LayoutDashboard,
  Network,
  Package,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react';
import { useSettings } from '../../context/SettingsContext';
import { twMerge } from 'tailwind-merge';

const links = [
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/environments', label: 'Environments', icon: Network },
  { to: '/agents', label: 'Agents', icon: Brain },
  { to: '/residual-connectors', label: 'Residual Connectors', icon: Layers3 },
  { to: '/reward-functions', label: 'Reward Functions', icon: Gauge },
  { to: '/models', label: 'Models', icon: Package },
  { to: '/experiments', label: 'Experiments', icon: Beaker },
];

export default function Sidebar() {
  const { pathname } = useLocation();
  const {
    settings: { sidebarCollapsed },
    toggleSetting,
  } = useSettings();

  return (
    <aside
      className={twMerge(
        'flex h-screen w-60 flex-shrink-0 flex-col bg-notion-sidebar border-r border-notion-border px-3 py-4 transition-all duration-200 sticky top-0',
        sidebarCollapsed && 'w-16'
      )}
    >
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-notion-blue/10 text-notion-blue">
            <Atom size={18} />
          </div>
          {!sidebarCollapsed && (
            <div>
              <p className="text-sm font-semibold text-notion-text">ConflictAwareLab</p>
            </div>
          )}
        </div>
        <button
          onClick={() => toggleSetting('sidebarCollapsed')}
          className="rounded-md p-1 text-notion-text-secondary hover:bg-notion-hover transition-colors"
          aria-label="Toggle sidebar"
        >
          {sidebarCollapsed ? <ChevronRight size={16} /> : <ChevronLeft size={16} />}
        </button>
      </div>
      <nav className="flex flex-1 flex-col gap-0.5">
        {links.map((link) => {
          const Icon = link.icon;
          const active = pathname.startsWith(link.to);
          return (
            <Link
              key={link.to}
              to={link.to}
              className={twMerge(
                'flex items-center gap-2.5 rounded-md px-2.5 py-1.5 text-sm text-notion-text-secondary transition-colors hover:bg-notion-hover',
                active && 'bg-notion-blue/10 text-notion-blue font-medium',
                sidebarCollapsed && 'justify-center px-2'
              )}
            >
              <Icon size={18} className={active ? 'text-notion-blue' : 'text-notion-text-secondary'} />
              {!sidebarCollapsed && <span>{link.label}</span>}
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
