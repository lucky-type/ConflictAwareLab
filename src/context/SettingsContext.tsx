import { createContext, ReactNode, useContext, useMemo } from 'react';
import { useLocalStorage } from '../hooks/useLocalStorage';

type Settings = {
  compactMode: boolean;
  sidebarCollapsed: boolean;
  showMiniMap: boolean;
};

const SettingsContext = createContext<{
  settings: Settings;
  toggleSetting: (key: keyof Settings) => void;
} | null>(null);

export function SettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useLocalStorage<Settings>('dronesim-settings', {
    compactMode: false,
    sidebarCollapsed: false,
    showMiniMap: true,
  });

  const value = useMemo(
    () => ({
      settings,
      toggleSetting: (key: keyof Settings) =>
        setSettings((prev) => ({ ...prev, [key]: !prev[key] })),
    }),
    [settings, setSettings]
  );

  return <SettingsContext.Provider value={value}>{children}</SettingsContext.Provider>;
}

export function useSettings() {
  const ctx = useContext(SettingsContext);
  if (!ctx) throw new Error('SettingsContext missing');
  return ctx;
}
