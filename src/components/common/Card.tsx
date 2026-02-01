import { ReactNode } from 'react';
import { twMerge } from 'tailwind-merge';

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={twMerge('rounded-lg border border-notion-border bg-white p-4', className)}>
      {children}
    </div>
  );
}

export function CardHeader({
  title,
  subtitle,
  actions,
}: {
  title: ReactNode;
  subtitle?: string;
  actions?: ReactNode;
}) {
  return (
    <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <h3 className="text-base font-semibold text-notion-text">{title}</h3>
        {subtitle && <p className="text-xs text-notion-text-secondary">{subtitle}</p>}
      </div>
      {actions && <div className="sm:self-start">{actions}</div>}
    </div>
  );
}

export function StatValue({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div>
      <p className="text-xs uppercase tracking-wide text-notion-text-secondary">{label}</p>
      <div className="flex items-center gap-2">
        <p className="text-2xl font-semibold text-notion-text">{value}</p>
        {hint && <span className="text-xs text-notion-text-secondary">{hint}</span>}
      </div>
    </div>
  );
}
