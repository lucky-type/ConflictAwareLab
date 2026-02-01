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
    <div className={twMerge('flex min-h-screen bg-white text-notion-text', compactMode && 'text-sm')}>
      <Sidebar />
      <main className="flex-1 overflow-x-hidden">
        <div className="mx-auto max-w-7xl space-y-6 px-6 pb-10 pt-6">
          <TopBar />
          {children}
        </div>
      </main>
    </div>
  );
}
