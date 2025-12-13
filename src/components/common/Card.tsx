import { ReactNode } from 'react';
import { twMerge } from 'tailwind-merge';

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={twMerge('glass-panel rounded-xl p-4 shadow-lg shadow-cyan-500/5', className)}>
      {children}
    </div>
  );
}

export function CardHeader({ title, actions }: { title: string; actions?: ReactNode }) {
  return (
    <div className="mb-4 flex items-center justify-between">
      <h3 className="text-base font-semibold text-slate-100">{title}</h3>
      {actions}
    </div>
  );
}

export function StatValue({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-slate-400">{label}</p>
      <div className="flex items-center gap-2">
        <p className="text-2xl font-semibold text-white">{value}</p>
        {hint && <span className="text-xs text-slate-400">{hint}</span>}
      </div>
    </div>
  );
}
