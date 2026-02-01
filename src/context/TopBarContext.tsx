import { createContext, useContext, useState, ReactNode } from 'react';

interface TopBarContextType {
  actions: ReactNode;
  setActions: (actions: ReactNode) => void;
}

const TopBarContext = createContext<TopBarContextType | undefined>(undefined);

export function TopBarProvider({ children }: { children: ReactNode }) {
  const [actions, setActions] = useState<ReactNode>(null);

  return (
    <TopBarContext.Provider value={{ actions, setActions }}>
      {children}
    </TopBarContext.Provider>
  );
}

export function useTopBar() {
  const context = useContext(TopBarContext);
  if (!context) {
    throw new Error('useTopBar must be used within TopBarProvider');
  }
  return context;
}
