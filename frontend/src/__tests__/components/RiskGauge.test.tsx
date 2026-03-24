/**
 * Tests for RiskGauge component.
 *
 * Verifies score display, risk level labels, colour mapping,
 * SVG gauge rendering, and size customisation.
 */
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { RiskGauge, riskColors } from '../../components/RiskGauge';
import type { RiskLevel } from '../../types';

const allLevels: RiskLevel[] = ['low', 'moderate', 'high', 'critical'];

describe('RiskGauge', () => {
  it('displays the numeric score', () => {
    render(<RiskGauge score={72} level="high" />);
    expect(screen.getByText('72')).toBeInTheDocument();
  });

  it('displays "/ 100" denominator', () => {
    render(<RiskGauge score={50} level="moderate" />);
    expect(screen.getByText('/ 100')).toBeInTheDocument();
  });

  it.each(allLevels)('displays correct label for level: %s', (level) => {
    render(<RiskGauge score={50} level={level} />);
    expect(screen.getByText(riskColors[level].label)).toBeInTheDocument();
  });

  it('renders an SVG element for the gauge', () => {
    const { container } = render(<RiskGauge score={40} level="low" />);
    expect(container.querySelector('svg')).toBeTruthy();
  });

  it('renders two circle elements (background + progress)', () => {
    const { container } = render(<RiskGauge score={60} level="moderate" />);
    const circles = container.querySelectorAll('circle');
    expect(circles.length).toBe(2);
  });

  it('sets SVG size based on size prop', () => {
    const { container } = render(<RiskGauge score={50} level="low" size={200} />);
    const svg = container.querySelector('svg');
    expect(svg?.getAttribute('width')).toBe('200');
    expect(svg?.getAttribute('height')).toBe('200');
  });

  it('defaults to size 160', () => {
    const { container } = render(<RiskGauge score={50} level="low" />);
    const svg = container.querySelector('svg');
    expect(svg?.getAttribute('width')).toBe('160');
  });

  it('applies progress colour from riskColors mapping', () => {
    const { container } = render(<RiskGauge score={80} level="critical" />);
    const progressCircle = container.querySelectorAll('circle')[1];
    expect(progressCircle?.getAttribute('stroke')).toBe(riskColors.critical.stroke);
  });

  it('computes correct strokeDashoffset for 0 score', () => {
    const { container } = render(<RiskGauge score={0} level="low" />);
    const progressCircle = container.querySelectorAll('circle')[1];
    const radius = 45;
    const circumference = 2 * Math.PI * radius;
    // At score 0, offset should equal circumference (no visible arc)
    const offset = parseFloat(progressCircle?.getAttribute('stroke-dashoffset') ?? '0');
    expect(offset).toBeCloseTo(circumference, 0);
  });

  it('computes correct strokeDashoffset for 100 score', () => {
    const { container } = render(<RiskGauge score={100} level="critical" />);
    const progressCircle = container.querySelectorAll('circle')[1];
    // At score 100, offset should be 0 (full circle)
    const offset = parseFloat(progressCircle?.getAttribute('stroke-dashoffset') ?? '0');
    expect(offset).toBeCloseTo(0, 0);
  });

  it('computes correct strokeDashoffset for 50 score', () => {
    const { container } = render(<RiskGauge score={50} level="moderate" />);
    const progressCircle = container.querySelectorAll('circle')[1];
    const radius = 45;
    const circumference = 2 * Math.PI * radius;
    const expectedOffset = circumference - (50 / 100) * circumference;
    const actual = parseFloat(progressCircle?.getAttribute('stroke-dashoffset') ?? '0');
    expect(actual).toBeCloseTo(expectedOffset, 0);
  });

  it('applies custom className', () => {
    const { container } = render(
      <RiskGauge score={50} level="low" className="my-custom-class" />
    );
    expect(container.firstElementChild?.className).toContain('my-custom-class');
  });

  it.each(allLevels)('renders risk badge with correct colours for: %s', (level) => {
    const { container } = render(<RiskGauge score={50} level={level} />);
    const badge = screen.getByText(riskColors[level].label);
    const bgClass = riskColors[level].bg.split(' ')[0];
    expect(badge.className).toContain(bgClass);
  });
});
