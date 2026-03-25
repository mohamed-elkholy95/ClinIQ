/**
 * Tests for the ComorbidityCalculator page component.
 *
 * Validates page structure, sample loading, API integration, CCI score display,
 * risk group styling, category breakdown, mortality estimation, and error handling.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ComorbidityCalculator } from '../../pages/ComorbidityCalculator';

const mockFetch = vi.fn();
global.fetch = mockFetch;

beforeEach(() => {
  vi.clearAllMocks();
});

// ─── Page Structure ──────────────────────────────────────────

describe('ComorbidityCalculator — page structure', () => {
  it('renders the page heading', () => {
    render(<ComorbidityCalculator />);
    expect(screen.getByText('Charlson Comorbidity Index')).toBeTruthy();
  });

  it('renders description text', () => {
    render(<ComorbidityCalculator />);
    expect(screen.getByText(/disease burden/i)).toBeTruthy();
  });

  it('renders three sample buttons', () => {
    render(<ComorbidityCalculator />);
    expect(screen.getByText('Complex Patient')).toBeTruthy();
    expect(screen.getByText('Healthy Adult')).toBeTruthy();
    expect(screen.getByText('Oncology Patient')).toBeTruthy();
  });

  it('renders Calculate CCI button disabled initially', () => {
    render(<ComorbidityCalculator />);
    const btn = screen.getByText('Calculate CCI');
    expect((btn as HTMLButtonElement).disabled).toBe(true);
  });

  it('renders ICD codes input', () => {
    render(<ComorbidityCalculator />);
    expect(screen.getByPlaceholderText(/E11.42/)).toBeTruthy();
  });

  it('renders age input', () => {
    render(<ComorbidityCalculator />);
    expect(screen.getByPlaceholderText(/e\.g\. 72/)).toBeTruthy();
  });
});

// ─── Sample Loading ──────────────────────────────────────────

describe('ComorbidityCalculator — sample loading', () => {
  it('loads Complex Patient sample', () => {
    render(<ComorbidityCalculator />);
    fireEvent.click(screen.getByText('Complex Patient'));
    const textarea = screen.getByPlaceholderText(/medical history/i) as HTMLTextAreaElement;
    expect(textarea.value).toContain('Congestive Heart Failure');
  });

  it('loads ICD codes for sample', () => {
    render(<ComorbidityCalculator />);
    fireEvent.click(screen.getByText('Complex Patient'));
    const codesInput = screen.getByPlaceholderText(/E11.42/) as HTMLInputElement;
    expect(codesInput.value).toContain('E11.42');
    expect(codesInput.value).toContain('I50.23');
  });

  it('loads age for sample', () => {
    render(<ComorbidityCalculator />);
    fireEvent.click(screen.getByText('Complex Patient'));
    const ageInput = screen.getByPlaceholderText(/e\.g\. 72/) as HTMLInputElement;
    expect(ageInput.value).toBe('72');
  });

  it('enables button after loading sample', () => {
    render(<ComorbidityCalculator />);
    fireEvent.click(screen.getByText('Complex Patient'));
    const btn = screen.getByText('Calculate CCI');
    expect((btn as HTMLButtonElement).disabled).toBe(false);
  });
});

// ─── API Integration ─────────────────────────────────────────

describe('ComorbidityCalculator — API integration', () => {
  const mockResult = {
    total_score: 8,
    age_adjusted_score: 10,
    risk_group: 'severe',
    estimated_mortality: 0.78,
    categories: [
      {
        name: 'congestive_heart_failure',
        weight: 1,
        detected: true,
        source: 'icd_code',
        description: 'Congestive heart failure',
        matched_codes: ['I50.23'],
        confidence: 1.0,
      },
      {
        name: 'diabetes_with_complications',
        weight: 2,
        detected: true,
        source: 'icd_code',
        description: 'Diabetes with end-organ damage',
        matched_codes: ['E11.42'],
        confidence: 1.0,
      },
      {
        name: 'moderate_severe_renal',
        weight: 2,
        detected: true,
        source: 'both',
        description: 'Moderate or severe renal disease',
        matched_codes: ['N18.4'],
        confidence: 0.95,
      },
      {
        name: 'chronic_pulmonary',
        weight: 1,
        detected: true,
        source: 'icd_code',
        description: 'Chronic pulmonary disease',
        matched_codes: ['J44.1'],
        confidence: 1.0,
      },
      {
        name: 'cerebrovascular',
        weight: 1,
        detected: true,
        source: 'icd_code',
        description: 'Cerebrovascular disease',
        matched_codes: ['I63.9'],
        confidence: 1.0,
      },
    ],
    processing_time_ms: 5.3,
  };

  it('renders CCI score after calculation', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResult,
    });
    render(<ComorbidityCalculator />);
    fireEvent.click(screen.getByText('Complex Patient'));
    fireEvent.click(screen.getByText('Calculate CCI'));

    await waitFor(() => {
      expect(screen.getByText('8')).toBeTruthy();
      expect(screen.getByText('CCI Score')).toBeTruthy();
    });
  });

  it('renders risk group label', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResult,
    });
    render(<ComorbidityCalculator />);
    fireEvent.click(screen.getByText('Complex Patient'));
    fireEvent.click(screen.getByText('Calculate CCI'));

    await waitFor(() => {
      // "Severe Risk" appears in both the score card and the reference guide
      const severeLabels = screen.getAllByText('Severe Risk');
      expect(severeLabels.length).toBeGreaterThan(0);
    });
  });

  it('renders age-adjusted score', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResult,
    });
    render(<ComorbidityCalculator />);
    fireEvent.click(screen.getByText('Complex Patient'));
    fireEvent.click(screen.getByText('Calculate CCI'));

    await waitFor(() => {
      expect(screen.getByText('10')).toBeTruthy();
      expect(screen.getByText('Age-Adjusted Score')).toBeTruthy();
    });
  });

  it('renders 10-year mortality estimate', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResult,
    });
    render(<ComorbidityCalculator />);
    fireEvent.click(screen.getByText('Complex Patient'));
    fireEvent.click(screen.getByText('Calculate CCI'));

    await waitFor(() => {
      expect(screen.getByText('78%')).toBeTruthy();
      expect(screen.getByText('10-Year Mortality')).toBeTruthy();
    });
  });

  it('renders disease categories', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResult,
    });
    render(<ComorbidityCalculator />);
    fireEvent.click(screen.getByText('Complex Patient'));
    fireEvent.click(screen.getByText('Calculate CCI'));

    await waitFor(() => {
      expect(screen.getByText('Disease Categories Identified')).toBeTruthy();
      // Category names formatted from snake_case — may appear in textarea too
      expect(screen.getAllByText(/Congestive/i).length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText(/Diabetes/i).length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText(/Pulmonary/i).length).toBeGreaterThanOrEqual(1);
    });
  });

  it('renders matched ICD codes for categories', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResult,
    });
    render(<ComorbidityCalculator />);
    fireEvent.click(screen.getByText('Complex Patient'));
    fireEvent.click(screen.getByText('Calculate CCI'));

    await waitFor(() => {
      expect(screen.getByText('I50.23')).toBeTruthy();
      expect(screen.getByText('E11.42')).toBeTruthy();
      expect(screen.getByText('N18.4')).toBeTruthy();
    });
  });

  it('renders categories count', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResult,
    });
    render(<ComorbidityCalculator />);
    fireEvent.click(screen.getByText('Complex Patient'));
    fireEvent.click(screen.getByText('Calculate CCI'));

    await waitFor(() => {
      expect(screen.getByText('5')).toBeTruthy();
      expect(screen.getByText('Categories Identified')).toBeTruthy();
    });
  });

  it('renders risk group reference guide', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResult,
    });
    render(<ComorbidityCalculator />);
    fireEvent.click(screen.getByText('Complex Patient'));
    fireEvent.click(screen.getByText('Calculate CCI'));

    await waitFor(() => {
      expect(screen.getByText('RISK GROUP REFERENCE')).toBeTruthy();
      expect(screen.getByText('Low Risk')).toBeTruthy();
      expect(screen.getByText('Mild Risk')).toBeTruthy();
      expect(screen.getByText('Moderate Risk')).toBeTruthy();
    });
  });

  it('handles API errors', async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    render(<ComorbidityCalculator />);
    fireEvent.click(screen.getByText('Complex Patient'));
    fireEvent.click(screen.getByText('Calculate CCI'));

    await waitFor(() => {
      expect(screen.getByText(/server error: 500/i)).toBeTruthy();
    });
  });

  it('shows processing time', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResult,
    });
    render(<ComorbidityCalculator />);
    fireEvent.click(screen.getByText('Complex Patient'));
    fireEvent.click(screen.getByText('Calculate CCI'));

    await waitFor(() => {
      expect(screen.getByText(/5\.3ms/)).toBeTruthy();
    });
  });
});
