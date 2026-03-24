import { clsx } from 'clsx';

interface ConfidenceBarProps {
  value: number; // 0-1
  label?: string;
  size?: 'sm' | 'md' | 'lg';
  showPercentage?: boolean;
  className?: string;
}

function getBarColor(value: number): string {
  if (value >= 0.9) return 'bg-green-500';
  if (value >= 0.7) return 'bg-blue-500';
  if (value >= 0.5) return 'bg-amber-500';
  return 'bg-red-500';
}

export function ConfidenceBar({
  value,
  label,
  size = 'md',
  showPercentage = true,
  className,
}: ConfidenceBarProps) {
  const percentage = Math.round(value * 100);

  return (
    <div className={clsx('flex items-center gap-3', className)}>
      {label && (
        <span className="text-sm text-text-secondary min-w-0 truncate">
          {label}
        </span>
      )}
      <div className="flex-1 flex items-center gap-2">
        <div
          className={clsx(
            'flex-1 rounded-full bg-gray-100 dark:bg-gray-800 overflow-hidden',
            size === 'sm' && 'h-1.5',
            size === 'md' && 'h-2.5',
            size === 'lg' && 'h-3.5'
          )}
        >
          <div
            className={clsx(
              'h-full rounded-full transition-all duration-500 ease-out',
              getBarColor(value)
            )}
            style={{ width: `${percentage}%` }}
          />
        </div>
        {showPercentage && (
          <span
            className={clsx(
              'font-mono font-medium tabular-nums text-text-secondary',
              size === 'sm' ? 'text-xs w-8' : 'text-sm w-10',
              'text-right'
            )}
          >
            {percentage}%
          </span>
        )}
      </div>
    </div>
  );
}
