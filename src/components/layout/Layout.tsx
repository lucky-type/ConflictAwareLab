import { ReactNode } from 'react';
import Sidebar from './Sidebar';
import TopBar from './TopBar';
import { useSettings } from '../../context/SettingsContext';
import { twMerge } from 'tailwind-merge';

export default function Layout({ children }: { children: ReactNode }) {
  const {
    settings: { compactMode },
  } = useSettings();

  return (
    <div className={twMerge('min-h-screen bg-slate-950/95 text-slate-100', compactMode && 'text-sm')}>
      <div className="mx-auto grid max-w-7xl grid-cols-[auto,1fr] gap-4 px-4 pb-10 pt-6">
        <Sidebar />
        <main className="space-y-6">
          <TopBar />
          {children}
        </main>
      </div>
    </div>
  );
}
