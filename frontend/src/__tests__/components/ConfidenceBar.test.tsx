/**
 * Tests for ConfidenceBar component.
 *
 * Verifies percentage display, colour thresholds, size variants,
 * label rendering, and percentage visibility toggle.
 */
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ConfidenceBar } from '../../components/ConfidenceBar';

describe('ConfidenceBar', () => {
  it('displays percentage by default', () => {
    render(<ConfidenceBar value={0.85} />);
    expect(screen.getByText('85%')).toBeInTheDocument();
  });

  it('rounds percentage to nearest integer', () => {
    render(<ConfidenceBar value={0.876} />);
    expect(screen.getByText('88%')).toBeInTheDocument();
  });

  it('hides percentage when showPercentage is false', () => {
    const { container } = render(<ConfidenceBar value={0.5} showPercentage={false} />);
    expect(container.textContent).not.toContain('%');
  });

  it('renders label when provided', () => {
    render(<ConfidenceBar value={0.7} label="NER Confidence" />);
    expect(screen.getByText('NER Confidence')).toBeInTheDocument();
  });

  it('does not render label text when not provided', () => {
    const { container } = render(<ConfidenceBar value={0.7} />);
    // When no label prop, the only text should be the percentage
    const spans = container.querySelectorAll('span');
    const labelSpans = Array.from(spans).filter(
      (s) => s.className.includes('truncate')
    );
    expect(labelSpans.length).toBe(0);
  });

  it('sets bar width based on percentage', () => {
    const { container } = render(<ConfidenceBar value={0.65} />);
    const bar = container.querySelector('[style*="width"]');
    expect(bar).toBeTruthy();
    expect(bar?.getAttribute('style')).toContain('65%');
  });

  // Colour threshold tests
  it('uses green for values >= 0.9', () => {
    const { container } = render(<ConfidenceBar value={0.95} />);
    const bar = container.querySelector('.bg-green-500');
    expect(bar).toBeTruthy();
  });

  it('uses blue for values >= 0.7 and < 0.9', () => {
    const { container } = render(<ConfidenceBar value={0.75} />);
    const bar = container.querySelector('.bg-blue-500');
    expect(bar).toBeTruthy();
  });

  it('uses amber for values >= 0.5 and < 0.7', () => {
    const { container } = render(<ConfidenceBar value={0.55} />);
    const bar = container.querySelector('.bg-amber-500');
    expect(bar).toBeTruthy();
  });

  it('uses red for values < 0.5', () => {
    const { container } = render(<ConfidenceBar value={0.3} />);
    const bar = container.querySelector('.bg-red-500');
    expect(bar).toBeTruthy();
  });

  // Boundary values
  it('uses green at exactly 0.9', () => {
    const { container } = render(<ConfidenceBar value={0.9} />);
    expect(container.querySelector('.bg-green-500')).toBeTruthy();
  });

  it('uses blue at exactly 0.7', () => {
    const { container } = render(<ConfidenceBar value={0.7} />);
    expect(container.querySelector('.bg-blue-500')).toBeTruthy();
  });

  it('uses amber at exactly 0.5', () => {
    const { container } = render(<ConfidenceBar value={0.5} />);
    expect(container.querySelector('.bg-amber-500')).toBeTruthy();
  });

  // Size variants
  it('renders sm size', () => {
    const { container } = render(<ConfidenceBar value={0.5} size="sm" />);
    expect(container.querySelector('.h-1\\.5')).toBeTruthy();
  });

  it('renders md size (default)', () => {
    const { container } = render(<ConfidenceBar value={0.5} />);
    expect(container.querySelector('.h-2\\.5')).toBeTruthy();
  });

  it('renders lg size', () => {
    const { container } = render(<ConfidenceBar value={0.5} size="lg" />);
    expect(container.querySelector('.h-3\\.5')).toBeTruthy();
  });

  it('handles 0% value', () => {
    render(<ConfidenceBar value={0} />);
    expect(screen.getByText('0%')).toBeInTheDocument();
  });

  it('handles 100% value', () => {
    render(<ConfidenceBar value={1} />);
    expect(screen.getByText('100%')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(<ConfidenceBar value={0.5} className="custom-test" />);
    expect(container.firstElementChild?.className).toContain('custom-test');
  });
});
