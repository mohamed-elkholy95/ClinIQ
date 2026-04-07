/**
 * Tests for the VitalSigns page component.
 *
 * Validates page structure, sample loading, API integration, vital cards
 * with interpretation styling, critical/abnormal banners, and error handling.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { VitalSigns } from '../../pages/VitalSigns';

const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

beforeEach(() => {
  vi.clearAllMocks();
});

// ─── Page Structure ──────────────────────────────────────────

describe('VitalSigns — page structure', () => {
  it('renders the page heading', () => {
    render(<VitalSigns />);
    expect(screen.getByText('Vital Signs Extraction')).toBeTruthy();
  });

  it('renders three sample buttons', () => {
    render(<VitalSigns />);
    expect(screen.getByText('Emergency Triage')).toBeTruthy();
    expect(screen.getByText('Routine Physical')).toBeTruthy();
    expect(screen.getByText('ICU Progress Note')).toBeTruthy();
  });

  it('renders Extract Vitals button disabled initially', () => {
    render(<VitalSigns />);
    const btn = screen.getByText('Extract Vitals');
    expect((btn as HTMLButtonElement).disabled).toBe(true);
  });
});

// ─── Sample Loading ──────────────────────────────────────────

describe('VitalSigns — sample loading', () => {
  it('loads Emergency Triage sample', () => {
    render(<VitalSigns />);
    fireEvent.click(screen.getByText('Emergency Triage'));
    const textarea = screen.getByPlaceholderText(/vital signs/i) as HTMLTextAreaElement;
    expect(textarea.value).toContain('158/94');
  });

  it('enables button after loading sample', () => {
    render(<VitalSigns />);
    fireEvent.click(screen.getByText('Emergency Triage'));
    const btn = screen.getByText('Extract Vitals');
    expect((btn as HTMLButtonElement).disabled).toBe(false);
  });
});

// ─── API Integration ─────────────────────────────────────────

describe('VitalSigns — API integration', () => {
  const mockResponse = {
    vitals: [
      {
        type: 'blood_pressure',
        value: 158,
        unit: 'mmHg',
        interpretation: 'high',
        trend: null,
        confidence: 0.95,
        start_char: 0,
        end_char: 15,
        diastolic: 94,
        map: 115,
      },
      {
        type: 'heart_rate',
        value: 112,
        unit: 'bpm',
        interpretation: 'high',
        trend: 'improving',
        confidence: 0.90,
        start_char: 20,
        end_char: 30,
      },
      {
        type: 'temperature',
        value: 101.8,
        unit: '°F',
        interpretation: 'high',
        trend: null,
        confidence: 0.85,
        start_char: 35,
        end_char: 45,
      },
      {
        type: 'oxygen_saturation',
        value: 93,
        unit: '%',
        interpretation: 'low',
        trend: null,
        confidence: 0.90,
        start_char: 50,
        end_char: 60,
      },
    ],
    count: 4,
    processing_time_ms: 1.8,
  };

  it('renders vital cards after API call', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<VitalSigns />);
    fireEvent.click(screen.getByText('Emergency Triage'));
    fireEvent.click(screen.getByText('Extract Vitals'));

    await waitFor(() => {
      expect(screen.getByText('Blood Pressure')).toBeTruthy();
      expect(screen.getByText('Heart Rate')).toBeTruthy();
      expect(screen.getByText('Temperature')).toBeTruthy();
      expect(screen.getByText('Oxygen Saturation')).toBeTruthy();
    });
  });

  it('shows vital values and units', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<VitalSigns />);
    fireEvent.click(screen.getByText('Emergency Triage'));
    fireEvent.click(screen.getByText('Extract Vitals'));

    await waitFor(() => {
      expect(screen.getByText('158')).toBeTruthy();
      expect(screen.getByText('mmHg')).toBeTruthy();
      expect(screen.getByText('112')).toBeTruthy();
      expect(screen.getByText('bpm')).toBeTruthy();
    });
  });

  it('shows diastolic and MAP for blood pressure', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<VitalSigns />);
    fireEvent.click(screen.getByText('Emergency Triage'));
    fireEvent.click(screen.getByText('Extract Vitals'));

    await waitFor(() => {
      // Check for diastolic display in a non-textarea element
      expect(screen.getByText(/Systolic\/Diastolic/)).toBeTruthy();
      expect(screen.getByText(/MAP: 115/)).toBeTruthy();
    });
  });

  it('shows abnormal count in banner', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<VitalSigns />);
    fireEvent.click(screen.getByText('Emergency Triage'));
    fireEvent.click(screen.getByText('Extract Vitals'));

    await waitFor(() => {
      // 3 high + 1 low = 4 abnormal total
      expect(screen.getByText(/abnormal/i)).toBeTruthy();
    });
  });

  it('shows trend information', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<VitalSigns />);
    fireEvent.click(screen.getByText('Emergency Triage'));
    fireEvent.click(screen.getByText('Extract Vitals'));

    await waitFor(() => {
      expect(screen.getByText(/improving/)).toBeTruthy();
    });
  });

  it('shows vitals detected count', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<VitalSigns />);
    fireEvent.click(screen.getByText('Emergency Triage'));
    fireEvent.click(screen.getByText('Extract Vitals'));

    await waitFor(() => {
      expect(screen.getByText('4')).toBeTruthy();
      expect(screen.getByText(/vitals detected/)).toBeTruthy();
    });
  });

  it('shows processing time', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<VitalSigns />);
    fireEvent.click(screen.getByText('Emergency Triage'));
    fireEvent.click(screen.getByText('Extract Vitals'));

    await waitFor(() => {
      expect(screen.getByText('1.8ms')).toBeTruthy();
    });
  });

  it('handles API errors', async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    render(<VitalSigns />);
    fireEvent.click(screen.getByText('Emergency Triage'));
    fireEvent.click(screen.getByText('Extract Vitals'));

    await waitFor(() => {
      expect(screen.getByText(/server error: 500/i)).toBeTruthy();
    });
  });

  it('shows all-normal banner when no abnormals', async () => {
    const normalResponse = {
      vitals: [
        {
          type: 'heart_rate',
          value: 72,
          unit: 'bpm',
          interpretation: 'normal',
          trend: null,
          confidence: 0.90,
          start_char: 0,
          end_char: 10,
        },
      ],
      count: 1,
      processing_time_ms: 0.5,
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => normalResponse,
    });
    render(<VitalSigns />);
    fireEvent.click(screen.getByText('Emergency Triage'));
    fireEvent.click(screen.getByText('Extract Vitals'));

    await waitFor(() => {
      expect(screen.getByText(/all within normal range/i)).toBeTruthy();
    });
  });
});

// ─── Interpretation Legend ───────────────────────────────────

describe('VitalSigns — interpretation legend', () => {
  it('shows legend after results load', async () => {
    const mockResponse = {
      vitals: [
        {
          type: 'heart_rate',
          value: 72,
          unit: 'bpm',
          interpretation: 'normal',
          trend: null,
          confidence: 0.90,
          start_char: 0,
          end_char: 10,
        },
      ],
      count: 1,
      processing_time_ms: 0.5,
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<VitalSigns />);
    fireEvent.click(screen.getByText('Emergency Triage'));
    fireEvent.click(screen.getByText('Extract Vitals'));

    await waitFor(() => {
      expect(screen.getByText('INTERPRETATION LEGEND')).toBeTruthy();
      expect(screen.getByText('normal')).toBeTruthy();
    });
  });
});

