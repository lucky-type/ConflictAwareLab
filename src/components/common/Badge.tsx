import { twMerge } from 'tailwind-merge';

const toneStyles: Record<string, string> = {
  default: 'bg-notion-light-gray text-notion-text',
  blue: 'bg-blue-50 text-notion-blue',
  green: 'bg-green-50 text-notion-green',
  red: 'bg-red-50 text-notion-red',
  orange: 'bg-orange-50 text-notion-orange',
  purple: 'bg-purple-50 text-notion-purple',
  yellow: 'bg-yellow-50 text-notion-yellow',
  pink: 'bg-pink-50 text-notion-pink',
};

export function Badge({ label, tone = 'default' }: { label: string; tone?: keyof typeof toneStyles | string }) {
  const toneClass = toneStyles[tone] || tone;
  return (
    <span
      className={twMerge(
        'rounded-md px-2 py-0.5 text-xs font-medium',
        toneClass
      )}
    >
      {label}
    </span>
  );
}
