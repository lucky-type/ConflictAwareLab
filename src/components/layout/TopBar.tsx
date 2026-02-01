import { useLocation } from 'react-router-dom';
import { useTopBar } from '../../context/TopBarContext';

const pageTitles: Record<string, string> = {
  '/dashboard': 'Dashboard',
  '/environments': 'Environments',
  '/agents': 'Agents',
  '/reward-functions': 'Reward Functions',
  '/models': 'Models',
  '/experiments': 'Experiments',
};

export default function TopBar() {
  const { pathname } = useLocation();
  const { actions } = useTopBar();
  const pageTitle = pageTitles[pathname] || 'Dashboard';

  return (
    <header className="flex items-center justify-between border-b border-notion-border pb-4 mb-2">
      <div>
        <h1 className="text-xl font-semibold text-notion-text">{pageTitle}</h1>
      </div>
      {actions && (
        <div className="flex items-center gap-3">
          {actions}
        </div>
      )}
    </header>
  );
}
