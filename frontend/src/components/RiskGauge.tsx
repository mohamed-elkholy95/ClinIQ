import { clsx } from 'clsx';
import type { RiskLevel } from '../types';

interface RiskGaugeProps {
  score: number; // 0-100
  level: RiskLevel;
  size?: number;
  className?: string;
}

const riskColors: Record<RiskLevel, { stroke: string; text: string; bg: string; label: string }> = {
  low: {
    stroke: '#22c55e',
    text: 'text-green-600 dark:text-green-400',
    bg: 'bg-green-50 dark:bg-green-950/30',
    label: 'Low Risk',
  },
  moderate: {
    stroke: '#eab308',
    text: 'text-yellow-600 dark:text-yellow-400',
    bg: 'bg-yellow-50 dark:bg-yellow-950/30',
    label: 'Moderate Risk',
  },
  high: {
    stroke: '#f97316',
    text: 'text-orange-600 dark:text-orange-400',
    bg: 'bg-orange-50 dark:bg-orange-950/30',
    label: 'High Risk',
  },
  critical: {
    stroke: '#ef4444',
    text: 'text-red-600 dark:text-red-400',
    bg: 'bg-red-50 dark:bg-red-950/30',
    label: 'Critical Risk',
  },
};

export function RiskGauge({ score, level, size = 160, className }: RiskGaugeProps) {
  const colors = riskColors[level];
  const radius = 45;
  const circumference = 2 * Math.PI * radius;
  const progress = (score / 100) * circumference;
  const offset = circumference - progress;

  return (
    <div className={clsx('flex flex-col items-center gap-2', className)}>
      <div className="relative" style={{ width: size, height: size }}>
        <svg
          viewBox="0 0 100 100"
          width={size}
          height={size}
          className="-rotate-90"
        >
          {/* Background circle */}
          <circle
            cx="50"
            cy="50"
            r={radius}
            fill="none"
            stroke="currentColor"
            strokeWidth="8"
            className="text-gray-100 dark:text-gray-800"
          />
          {/* Progress circle */}
          <circle
            cx="50"
            cy="50"
            r={radius}
            fill="none"
            stroke={colors.stroke}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            className="gauge-animate"
          />
        </svg>
        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span
            className={clsx('font-bold tabular-nums', colors.text)}
            style={{ fontSize: size * 0.2 }}
          >
            {score}
          </span>
          <span
            className="text-text-muted font-medium"
            style={{ fontSize: size * 0.08 }}
          >
            / 100
          </span>
        </div>
      </div>
      <span
        className={clsx(
          'inline-flex items-center rounded-full px-3 py-1 text-sm font-semibold',
          colors.bg,
          colors.text
        )}
      >
        {colors.label}
      </span>
    </div>
  );
}

export { riskColors };
