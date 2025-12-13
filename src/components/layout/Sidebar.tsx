import { Link, useLocation } from 'react-router-dom';
import {
  Activity,
  Atom,
  Beaker,
  Brain,
  Gauge,
  LayoutDashboard,
  Network,
  Package,
  Settings,
} from 'lucide-react';
import { useSettings } from '../../context/SettingsContext';
import { twMerge } from 'tailwind-merge';

const links = [
  { to: '/dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/environments', label: 'Environments', icon: Network },
  { to: '/algorithms', label: 'Algorithms', icon: Brain },
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
        'glass-panel sticky top-0 flex h-screen w-72 flex-col border-r border-slate-800/80 px-4 py-6 transition-all duration-300',
        sidebarCollapsed && 'w-20'
      )}
    >
      <div className="mb-8 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-cyan-500/10 text-cyan-400">
            <Atom size={22} />
          </div>
          {!sidebarCollapsed && (
            <div>
              <p className="text-sm text-slate-400">DroneSim</p>
              <p className="text-lg font-semibold text-white">Control Center</p>
            </div>
          )}
        </div>
        <button
          onClick={() => toggleSetting('sidebarCollapsed')}
          className="rounded-lg border border-slate-800/80 bg-slate-900/70 px-2 py-1 text-slate-300 hover:border-cyan-500/50"
          aria-label="Toggle sidebar"
        >
          {sidebarCollapsed ? '›' : '‹'}
        </button>
      </div>
      <nav className="flex flex-1 flex-col gap-1">
        {links.map((link) => {
          const Icon = link.icon;
          const active = pathname.startsWith(link.to);
          return (
            <Link
              key={link.to}
              to={link.to}
              className={twMerge(
                'flex items-center gap-3 rounded-xl px-3 py-3 text-sm font-medium text-slate-200 transition hover:bg-slate-800/60',
                active && 'bg-cyan-500/15 text-white shadow-inner shadow-cyan-500/30',
                sidebarCollapsed && 'justify-center'
              )}
            >
              <Icon size={20} className={active ? 'text-cyan-400' : 'text-slate-400'} />
              {!sidebarCollapsed && <span>{link.label}</span>}
            </Link>
          );
        })}
      </nav>
      <div className={twMerge('mt-auto space-y-3', sidebarCollapsed && 'items-center text-center')}>
        <button className="flex w-full items-center gap-3 rounded-xl border border-slate-800/80 bg-slate-900/70 px-3 py-3 text-sm text-slate-200 hover:border-cyan-500/50">
          <Activity size={18} className="text-cyan-400" />
          {!sidebarCollapsed && <span>Live monitors</span>}
        </button>
        <button className="flex w-full items-center gap-3 rounded-xl border border-slate-800/80 bg-slate-900/70 px-3 py-3 text-sm text-slate-200 hover:border-cyan-500/50">
          <Settings size={18} className="text-cyan-400" />
          {!sidebarCollapsed && <span>Settings</span>}
        </button>
      </div>
    </aside>
  );
}
