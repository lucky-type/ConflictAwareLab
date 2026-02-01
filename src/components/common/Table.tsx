import { ReactNode } from 'react';

export function Table({ children }: { children: ReactNode }) {
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm text-notion-text">{children}</table>
    </div>
  );
}

export function TableHead({ children }: { children: ReactNode }) {
  return <thead className="bg-notion-light-gray text-left text-xs uppercase text-notion-text-secondary border-b border-notion-border">{children}</thead>;
}

export function TableBody({ children }: { children: ReactNode }) {
  return <tbody className="divide-y divide-notion-border">{children}</tbody>;
}

export function TableRow({ children }: { children: ReactNode }) {
  return <tr className="hover:bg-notion-hover transition-colors">{children}</tr>;
}

export function TableCell({ children, className }: { children?: ReactNode; className?: string }) {
  return <td className={className ? `px-4 py-3 ${className}` : 'px-4 py-3'}>{children}</td>;
}
