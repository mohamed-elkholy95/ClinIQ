/**
 * Tests for the QualityAnalyzer page component.
 *
 * Validates page structure, sample loading, API integration, grade display,
 * dimension scores, recommendations, and error handling.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { QualityAnalyzer } from '../../pages/QualityAnalyzer';

const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

beforeEach(() => {
  vi.clearAllMocks();
});

// ─── Page Structure ──────────────────────────────────────────

describe('QualityAnalyzer — page structure', () => {
  it('renders the page heading', () => {
    render(<QualityAnalyzer />);
    expect(screen.getByText('Note Quality Analysis')).toBeTruthy();
  });

  it('renders three sample buttons', () => {
    render(<QualityAnalyzer />);
    expect(screen.getByText('Good H&P Note')).toBeTruthy();
    expect(screen.getByText('Poor Note')).toBeTruthy();
    expect(screen.getByText('Moderate Note')).toBeTruthy();
  });

  it('renders Analyze Quality button disabled initially', () => {
    render(<QualityAnalyzer />);
    const btn = screen.getByText('Analyze Quality');
    expect((btn as HTMLButtonElement).disabled).toBe(true);
  });
});

// ─── Sample Loading ──────────────────────────────────────────

describe('QualityAnalyzer — sample loading', () => {
  it('loads Good H&P Note sample', () => {
    render(<QualityAnalyzer />);
    fireEvent.click(screen.getByText('Good H&P Note'));
    const textarea = screen.getByPlaceholderText(/evaluate quality/i) as HTMLTextAreaElement;
    expect(textarea.value).toContain('CHIEF COMPLAINT');
  });

  it('loads Poor Note sample', () => {
    render(<QualityAnalyzer />);
    fireEvent.click(screen.getByText('Poor Note'));
    const textarea = screen.getByPlaceholderText(/evaluate quality/i) as HTMLTextAreaElement;
    expect(textarea.value).toContain('pt c/o pain');
  });

  it('enables button after loading sample', () => {
    render(<QualityAnalyzer />);
    fireEvent.click(screen.getByText('Good H&P Note'));
    const btn = screen.getByText('Analyze Quality');
    expect((btn as HTMLButtonElement).disabled).toBe(false);
  });
});

// ─── API Integration ─────────────────────────────────────────

describe('QualityAnalyzer — API integration', () => {
  const mockReport = {
    overall_score: 87,
    grade: 'B',
    dimensions: [
      {
        dimension: 'completeness',
        score: 0.92,
        weight: 0.25,
        findings: [
          { message: 'All expected sections present', severity: 'info', dimension: 'completeness' },
        ],
      },
      {
        dimension: 'readability',
        score: 0.78,
        weight: 0.20,
        findings: [
          { message: 'High abbreviation density', severity: 'warning', dimension: 'readability' },
        ],
      },
      {
        dimension: 'structure',
        score: 0.85,
        weight: 0.20,
        findings: [],
      },
      {
        dimension: 'information_density',
        score: 0.90,
        weight: 0.20,
        findings: [],
      },
      {
        dimension: 'consistency',
        score: 0.88,
        weight: 0.15,
        findings: [],
      },
    ],
    recommendations: [
      'Reduce abbreviation density for improved readability',
      'Consider adding a Review of Systems section',
    ],
    analysis_time_ms: 3.2,
  };

  it('renders grade after analysis', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockReport,
    });
    render(<QualityAnalyzer />);
    fireEvent.click(screen.getByText('Good H&P Note'));
    fireEvent.click(screen.getByText('Analyze Quality'));

    await waitFor(() => {
      expect(screen.getByText('B')).toBeTruthy();
      expect(screen.getByText('Overall Grade')).toBeTruthy();
    });
  });

  it('renders overall score', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockReport,
    });
    render(<QualityAnalyzer />);
    fireEvent.click(screen.getByText('Good H&P Note'));
    fireEvent.click(screen.getByText('Analyze Quality'));

    await waitFor(() => {
      expect(screen.getByText('87')).toBeTruthy();
      expect(screen.getByText('Overall Score')).toBeTruthy();
    });
  });

  it('renders dimension scores', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockReport,
    });
    render(<QualityAnalyzer />);
    fireEvent.click(screen.getByText('Good H&P Note'));
    fireEvent.click(screen.getByText('Analyze Quality'));

    await waitFor(() => {
      expect(screen.getByText('Dimension Scores')).toBeTruthy();
      expect(screen.getByText(/Completeness/)).toBeTruthy();
      expect(screen.getByText(/Readability/)).toBeTruthy();
      expect(screen.getByText(/Structure/)).toBeTruthy();
      expect(screen.getByText(/Information Density/)).toBeTruthy();
      expect(screen.getByText(/Consistency/)).toBeTruthy();
    });
  });

  it('renders dimension findings', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockReport,
    });
    render(<QualityAnalyzer />);
    fireEvent.click(screen.getByText('Good H&P Note'));
    fireEvent.click(screen.getByText('Analyze Quality'));

    await waitFor(() => {
      expect(screen.getByText('All expected sections present')).toBeTruthy();
      expect(screen.getByText('High abbreviation density')).toBeTruthy();
    });
  });

  it('renders recommendations', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockReport,
    });
    render(<QualityAnalyzer />);
    fireEvent.click(screen.getByText('Good H&P Note'));
    fireEvent.click(screen.getByText('Analyze Quality'));

    await waitFor(() => {
      expect(screen.getByText(/Reduce abbreviation density/)).toBeTruthy();
      expect(screen.getByText(/Review of Systems/)).toBeTruthy();
    });
  });

  it('renders quick stats', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockReport,
    });
    render(<QualityAnalyzer />);
    fireEvent.click(screen.getByText('Good H&P Note'));
    fireEvent.click(screen.getByText('Analyze Quality'));

    await waitFor(() => {
      expect(screen.getByText('QUICK STATS')).toBeTruthy();
      expect(screen.getByText('Dimensions scored')).toBeTruthy();
    });
  });

  it('handles API errors', async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    render(<QualityAnalyzer />);
    fireEvent.click(screen.getByText('Good H&P Note'));
    fireEvent.click(screen.getByText('Analyze Quality'));

    await waitFor(() => {
      expect(screen.getByText(/server error: 500/i)).toBeTruthy();
    });
  });

  it('shows processing time', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockReport,
    });
    render(<QualityAnalyzer />);
    fireEvent.click(screen.getByText('Good H&P Note'));
    fireEvent.click(screen.getByText('Analyze Quality'));

    await waitFor(() => {
      expect(screen.getByText(/3\.2ms/)).toBeTruthy();
    });
  });
});

