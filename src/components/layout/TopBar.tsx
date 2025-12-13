import { Bell, Moon, Settings2, SlidersHorizontal, SunMedium } from 'lucide-react';
import { useSettings } from '../../context/SettingsContext';

export default function TopBar() {
  const {
    settings: { compactMode, showMiniMap },
    toggleSetting,
  } = useSettings();

  return (
    <header className="glass-panel sticky top-0 z-10 mb-6 flex items-center justify-between rounded-2xl px-6 py-4">
      <div>
        <p className="text-xs uppercase tracking-[0.2em] text-cyan-300">Simulation Control</p>
        <h1 className="text-xl font-semibold text-white">Fleet mission overview</h1>
      </div>
      <div className="flex items-center gap-3">
        <button
          onClick={() => toggleSetting('compactMode')}
          className="flex items-center gap-2 rounded-xl border border-slate-800/80 bg-slate-900/80 px-3 py-2 text-sm text-slate-200 hover:border-cyan-500/40"
        >
          {compactMode ? <SunMedium size={16} /> : <Moon size={16} />}
          <span className="hidden md:inline">{compactMode ? 'Comfort' : 'Compact'}</span>
        </button>
        <button
          onClick={() => toggleSetting('showMiniMap')}
          className="hidden items-center gap-2 rounded-xl border border-slate-800/80 bg-slate-900/80 px-3 py-2 text-sm text-slate-200 hover:border-cyan-500/40 md:flex"
        >
          <SlidersHorizontal size={16} />
          <span>Mini-map {showMiniMap ? 'on' : 'off'}</span>
        </button>
        <button className="flex h-10 w-10 items-center justify-center rounded-xl border border-slate-800/80 bg-slate-900/80 text-slate-200 hover:border-cyan-500/40">
          <Bell size={18} />
        </button>
        <button className="flex h-10 w-10 items-center justify-center rounded-xl border border-slate-800/80 bg-slate-900/80 text-slate-200 hover:border-cyan-500/40">
          <Settings2 size={18} />
        </button>
        <div className="flex items-center gap-2 rounded-full border border-slate-800/80 bg-slate-900/80 px-3 py-2">
          <div className="h-2 w-2 rounded-full bg-emerald-400" />
          <span className="text-sm text-slate-200">Systems nominal</span>
        </div>
      </div>
    </header>
  );
}
