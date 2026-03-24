import { clsx } from 'clsx';
import type { EntityType } from '../types';

interface EntityTagProps {
  type: EntityType;
  text: string;
  confidence?: number;
  size?: 'sm' | 'md';
  onClick?: () => void;
}

const entityStyles: Record<EntityType, { bg: string; text: string; border: string; dot: string }> = {
  disease: {
    bg: 'bg-red-50 dark:bg-red-950/40',
    text: 'text-red-700 dark:text-red-300',
    border: 'border-red-200 dark:border-red-800',
    dot: 'bg-red-500',
  },
  medication: {
    bg: 'bg-blue-50 dark:bg-blue-950/40',
    text: 'text-blue-700 dark:text-blue-300',
    border: 'border-blue-200 dark:border-blue-800',
    dot: 'bg-blue-500',
  },
  procedure: {
    bg: 'bg-green-50 dark:bg-green-950/40',
    text: 'text-green-700 dark:text-green-300',
    border: 'border-green-200 dark:border-green-800',
    dot: 'bg-green-500',
  },
  anatomy: {
    bg: 'bg-purple-50 dark:bg-purple-950/40',
    text: 'text-purple-700 dark:text-purple-300',
    border: 'border-purple-200 dark:border-purple-800',
    dot: 'bg-purple-500',
  },
  symptom: {
    bg: 'bg-amber-50 dark:bg-amber-950/40',
    text: 'text-amber-700 dark:text-amber-300',
    border: 'border-amber-200 dark:border-amber-800',
    dot: 'bg-amber-500',
  },
  lab_value: {
    bg: 'bg-cyan-50 dark:bg-cyan-950/40',
    text: 'text-cyan-700 dark:text-cyan-300',
    border: 'border-cyan-200 dark:border-cyan-800',
    dot: 'bg-cyan-500',
  },
};

const entityLabels: Record<EntityType, string> = {
  disease: 'Disease',
  medication: 'Medication',
  procedure: 'Procedure',
  anatomy: 'Anatomy',
  symptom: 'Symptom',
  lab_value: 'Lab Value',
};

export function EntityTag({ type, text, confidence, size = 'md', onClick }: EntityTagProps) {
  const style = entityStyles[type];

  return (
    <span
      onClick={onClick}
      className={clsx(
        'inline-flex items-center gap-1.5 rounded-full border font-medium',
        style.bg,
        style.text,
        style.border,
        size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm',
        onClick && 'cursor-pointer hover:opacity-80 transition-opacity'
      )}
    >
      <span className={clsx('inline-block rounded-full', style.dot, size === 'sm' ? 'h-1.5 w-1.5' : 'h-2 w-2')} />
      <span>{text}</span>
      {confidence !== undefined && (
        <span className="opacity-60 text-xs">
          {Math.round(confidence * 100)}%
        </span>
      )}
    </span>
  );
}

export function EntityTypeBadge({ type }: { type: EntityType }) {
  const style = entityStyles[type];
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 rounded px-2 py-0.5 text-xs font-medium',
        style.bg,
        style.text,
        style.border,
        'border'
      )}
    >
      <span className={clsx('h-1.5 w-1.5 rounded-full', style.dot)} />
      {entityLabels[type]}
    </span>
  );
}

export { entityStyles, entityLabels };
