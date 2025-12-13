import { twMerge } from 'tailwind-merge';

export function Badge({ label, tone }: { label: string; tone?: string }) {
  return (
    <span
      className={twMerge(
        'rounded-full border px-3 py-1 text-xs font-medium',
        tone ?? 'border-slate-700/60 bg-slate-800 text-slate-200'
      )}
    >
      {label}
    </span>
  );
}
