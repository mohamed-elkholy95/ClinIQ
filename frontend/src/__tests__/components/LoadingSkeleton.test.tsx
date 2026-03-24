/**
 * Tests for skeleton loading components.
 *
 * Verifies that each skeleton variant renders the expected structure
 * (number of animated placeholder bars, rows, chart bars).
 */
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import {
  Skeleton,
  CardSkeleton,
  TableSkeleton,
  ChartSkeleton,
  TextBlockSkeleton,
} from '../../components/LoadingSkeleton';

describe('Skeleton', () => {
  it('renders a div with animate-pulse class', () => {
    const { container } = render(<Skeleton />);
    const el = container.firstElementChild;
    expect(el?.tagName).toBe('DIV');
    expect(el?.className).toContain('animate-pulse');
  });

  it('applies custom className', () => {
    const { container } = render(<Skeleton className="h-8 w-32" />);
    const el = container.firstElementChild;
    expect(el?.className).toContain('h-8');
    expect(el?.className).toContain('w-32');
  });

  it('applies custom style', () => {
    const { container } = render(<Skeleton style={{ height: '50%' }} />);
    const el = container.firstElementChild as HTMLElement;
    expect(el.style.height).toBe('50%');
  });
});

describe('CardSkeleton', () => {
  it('renders a card container', () => {
    const { container } = render(<CardSkeleton />);
    expect(container.querySelector('.rounded-xl')).toBeTruthy();
  });

  it('contains multiple skeleton bars', () => {
    const { container } = render(<CardSkeleton />);
    const pulses = container.querySelectorAll('.animate-pulse');
    expect(pulses.length).toBeGreaterThanOrEqual(3);
  });
});

describe('TableSkeleton', () => {
  it('renders default 5 rows', () => {
    const { container } = render(<TableSkeleton />);
    const rows = container.querySelectorAll('.border-b');
    // header row + 5 data rows
    expect(rows.length).toBeGreaterThanOrEqual(5);
  });

  it('renders custom number of rows', () => {
    const { container } = render(<TableSkeleton rows={3} />);
    // Filter for data rows (excluding header border)
    const pulseGroups = container.querySelectorAll('.animate-pulse');
    // 4 header skeletons + 4 per row × 3 rows = 16
    expect(pulseGroups.length).toBe(4 + 4 * 3);
  });

  it('renders table container with border', () => {
    const { container } = render(<TableSkeleton />);
    expect(container.querySelector('.border-border')).toBeTruthy();
  });
});

describe('ChartSkeleton', () => {
  it('renders 12 bar placeholders', () => {
    const { container } = render(<ChartSkeleton />);
    // 12 bars + 1 title skeleton = 13 total
    const pulses = container.querySelectorAll('.animate-pulse');
    expect(pulses.length).toBe(13);
  });

  it('applies random heights to chart bars', () => {
    const { container } = render(<ChartSkeleton />);
    const bars = container.querySelectorAll('.flex-1.animate-pulse') as NodeListOf<HTMLElement>;
    // At least some bars should have different heights
    const heights = Array.from(bars).map((b) => b.style.height);
    const unique = new Set(heights);
    // With 12 random heights, extremely unlikely all identical
    expect(unique.size).toBeGreaterThan(1);
  });
});

describe('TextBlockSkeleton', () => {
  it('renders default 4 lines', () => {
    const { container } = render(<TextBlockSkeleton />);
    const pulses = container.querySelectorAll('.animate-pulse');
    expect(pulses.length).toBe(4);
  });

  it('renders custom number of lines', () => {
    const { container } = render(<TextBlockSkeleton lines={7} />);
    const pulses = container.querySelectorAll('.animate-pulse');
    expect(pulses.length).toBe(7);
  });

  it('makes last line shorter (60% width)', () => {
    const { container } = render(<TextBlockSkeleton lines={3} />);
    const bars = container.querySelectorAll('.animate-pulse') as NodeListOf<HTMLElement>;
    const lastBar = bars[bars.length - 1];
    expect(lastBar.style.width).toBe('60%');
  });

  it('makes non-last lines full width', () => {
    const { container } = render(<TextBlockSkeleton lines={3} />);
    const bars = container.querySelectorAll('.animate-pulse') as NodeListOf<HTMLElement>;
    expect(bars[0].style.width).toBe('100%');
    expect(bars[1].style.width).toBe('100%');
  });
});
