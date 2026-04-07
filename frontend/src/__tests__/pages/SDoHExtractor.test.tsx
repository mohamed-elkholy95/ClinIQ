/**
 * Tests for the SDoHExtractor page component.
 *
 * Validates page structure, sample loading, API integration, domain filters,
 * sentiment display, Z-code badges, and error handling.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { SDoHExtractor } from '../../pages/SDoHExtractor';

const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

beforeEach(() => {
  vi.clearAllMocks();
});

// ─── Page Structure ──────────────────────────────────────────

describe('SDoHExtractor — page structure', () => {
  it('renders the page heading', () => {
    render(<SDoHExtractor />);
    expect(screen.getByText('Social Determinants of Health')).toBeTruthy();
  });

  it('renders three sample buttons', () => {
    render(<SDoHExtractor />);
    expect(screen.getByText('Social History')).toBeTruthy();
    expect(screen.getByText('Protective Factors')).toBeTruthy();
    expect(screen.getByText('Mixed Factors')).toBeTruthy();
  });

  it('renders Extract SDoH button disabled initially', () => {
    render(<SDoHExtractor />);
    const btn = screen.getByText('Extract SDoH');
    expect((btn as HTMLButtonElement).disabled).toBe(true);
  });
});

// ─── Sample Loading ──────────────────────────────────────────

describe('SDoHExtractor — sample loading', () => {
  it('loads Social History sample', () => {
    render(<SDoHExtractor />);
    fireEvent.click(screen.getByText('Social History'));
    const textarea = screen.getByPlaceholderText(/social history/i) as HTMLTextAreaElement;
    expect(textarea.value).toContain('homeless');
  });

  it('enables button after loading sample', () => {
    render(<SDoHExtractor />);
    fireEvent.click(screen.getByText('Social History'));
    const btn = screen.getByText('Extract SDoH');
    expect((btn as HTMLButtonElement).disabled).toBe(false);
  });
});

// ─── API Integration ─────────────────────────────────────────

describe('SDoHExtractor — API integration', () => {
  const mockResponse = {
    findings: [
      {
        domain: 'housing',
        trigger_text: 'homeless',
        sentiment: 'adverse',
        z_code: 'Z59.0',
        confidence: 0.90,
        start_char: 0,
        end_char: 8,
      },
      {
        domain: 'substance_use',
        trigger_text: 'smoker, 1 pack per day',
        sentiment: 'adverse',
        z_code: 'Z72.0',
        confidence: 0.85,
        start_char: 100,
        end_char: 122,
      },
      {
        domain: 'food_security',
        trigger_text: 'food insecurity',
        sentiment: 'adverse',
        z_code: 'Z59.4',
        confidence: 0.82,
        start_char: 200,
        end_char: 215,
      },
      {
        domain: 'social_support',
        trigger_text: 'feeling isolated',
        sentiment: 'adverse',
        z_code: 'Z60.2',
        confidence: 0.78,
        start_char: 300,
        end_char: 316,
      },
    ],
    domains_detected: ['housing', 'substance_use', 'food_security', 'social_support'],
    adverse_count: 4,
    protective_count: 0,
    processing_time_ms: 4.5,
  };

  it('renders findings after API call', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<SDoHExtractor />);
    fireEvent.click(screen.getByText('Social History'));
    fireEvent.click(screen.getByText('Extract SDoH'));

    await waitFor(() => {
      expect(screen.getByText('homeless')).toBeTruthy();
      expect(screen.getByText('food insecurity')).toBeTruthy();
    });
  });

  it('renders summary cards', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<SDoHExtractor />);
    fireEvent.click(screen.getByText('Social History'));
    fireEvent.click(screen.getByText('Extract SDoH'));

    await waitFor(() => {
      expect(screen.getByText('Total Findings')).toBeTruthy();
      expect(screen.getByText('⚠️ Risk Factors')).toBeTruthy();
      expect(screen.getByText('🛡️ Protective')).toBeTruthy();
      expect(screen.getByText('Domains Active')).toBeTruthy();
    });
  });

  it('renders Z-code badges', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<SDoHExtractor />);
    fireEvent.click(screen.getByText('Social History'));
    fireEvent.click(screen.getByText('Extract SDoH'));

    await waitFor(() => {
      expect(screen.getByText('Z59.0')).toBeTruthy();
      expect(screen.getByText('Z72.0')).toBeTruthy();
      expect(screen.getByText('Z59.4')).toBeTruthy();
    });
  });

  it('renders sentiment labels', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<SDoHExtractor />);
    fireEvent.click(screen.getByText('Social History'));
    fireEvent.click(screen.getByText('Extract SDoH'));

    await waitFor(() => {
      // Each finding has "⚠️ Risk Factor" label
      const riskFactors = screen.getAllByText(/Risk Factor/);
      expect(riskFactors.length).toBeGreaterThanOrEqual(4);
    });
  });

  it('handles API errors', async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    render(<SDoHExtractor />);
    fireEvent.click(screen.getByText('Social History'));
    fireEvent.click(screen.getByText('Extract SDoH'));

    await waitFor(() => {
      expect(screen.getByText(/server error: 500/i)).toBeTruthy();
    });
  });

  it('shows processing time', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<SDoHExtractor />);
    fireEvent.click(screen.getByText('Social History'));
    fireEvent.click(screen.getByText('Extract SDoH'));

    await waitFor(() => {
      expect(screen.getByText(/4\.5ms/)).toBeTruthy();
    });
  });
});

// ─── Domain Filter ───────────────────────────────────────────

describe('SDoHExtractor — domain filter', () => {
  const mockResponse = {
    findings: [
      {
        domain: 'housing',
        trigger_text: 'homeless',
        sentiment: 'adverse',
        z_code: 'Z59.0',
        confidence: 0.90,
        start_char: 0,
        end_char: 8,
      },
      {
        domain: 'substance_use',
        trigger_text: 'smoker',
        sentiment: 'adverse',
        z_code: 'Z72.0',
        confidence: 0.85,
        start_char: 100,
        end_char: 106,
      },
    ],
    domains_detected: ['housing', 'substance_use'],
    adverse_count: 2,
    protective_count: 0,
    processing_time_ms: 2.0,
  };

  it('filters findings by domain', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<SDoHExtractor />);
    fireEvent.click(screen.getByText('Social History'));
    fireEvent.click(screen.getByText('Extract SDoH'));

    await waitFor(() => {
      expect(screen.getByText('homeless')).toBeTruthy();
      expect(screen.getByText('smoker')).toBeTruthy();
    });

    // Click housing filter button — it has icon prefix and count suffix
    const housingBtn = screen.getByText(/housing \(1\)/i);
    fireEvent.click(housingBtn);

    expect(screen.getByText('homeless')).toBeTruthy();
    expect(screen.queryByText('smoker')).toBeNull();
  });

  it('shows All filter by default', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<SDoHExtractor />);
    fireEvent.click(screen.getByText('Social History'));
    fireEvent.click(screen.getByText('Extract SDoH'));

    await waitFor(() => {
      expect(screen.getByText(/All \(2\)/)).toBeTruthy();
    });
  });
});

