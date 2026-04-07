/**
 * Tests for the SectionParser page component.
 *
 * Covers page structure, sample note loading, section parsing with
 * mock API responses, section card rendering, category summary,
 * stats display, and error handling.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { SectionParser } from '../../pages/SectionParser';

// ─── Mocks ───────────────────────────────────────────────────

const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

beforeEach(() => {
  vi.clearAllMocks();
});

const mockResponse = {
  sections: [
    {
      category: 'chief_complaint',
      header_text: 'CHIEF COMPLAINT',
      body_text: 'Chest pain and shortness of breath for 3 days.',
      header_start: 0,
      header_end: 16,
      body_end: 62,
      confidence: 0.95,
    },
    {
      category: 'history',
      header_text: 'HISTORY OF PRESENT ILLNESS',
      body_text: 'Mr. Johnson is a 68-year-old male with a history of hypertension.',
      header_start: 63,
      header_end: 90,
      body_end: 154,
      confidence: 0.92,
    },
    {
      category: 'medications',
      header_text: 'MEDICATIONS',
      body_text: '1. Metoprolol succinate 50mg daily\n2. Lisinopril 20mg daily',
      header_start: 155,
      header_end: 167,
      body_end: 226,
      confidence: 0.98,
    },
    {
      category: 'physical_exam',
      header_text: 'PHYSICAL EXAMINATION',
      body_text: 'Vital Signs: BP 158/92, HR 88, RR 20',
      header_start: 227,
      header_end: 247,
      body_end: 284,
      confidence: 0.90,
    },
    {
      category: 'assessment_and_plan',
      header_text: 'ASSESSMENT AND PLAN',
      body_text: '1. Acute coronary syndrome — rule out NSTEMI',
      header_start: 285,
      header_end: 304,
      body_end: 348,
      confidence: 0.96,
    },
  ],
  count: 5,
  processing_time_ms: 1.8,
};

function renderPage() {
  return render(<SectionParser />);
}

// ─── Page Structure ──────────────────────────────────────────

describe('SectionParser — Page Structure', () => {
  it('renders page title and description', () => {
    renderPage();
    expect(screen.getByText(/Section Parser/)).toBeInTheDocument();
    expect(screen.getByText(/Parse clinical documents/)).toBeInTheDocument();
  });

  it('renders textarea input', () => {
    renderPage();
    expect(screen.getByPlaceholderText(/Paste a clinical document/)).toBeInTheDocument();
  });

  it('renders parse button (disabled when empty)', () => {
    renderPage();
    const btn = screen.getByRole('button', { name: /Parse Sections/ });
    expect(btn).toBeDisabled();
  });

  it('renders all sample note buttons', () => {
    renderPage();
    expect(screen.getByRole('button', { name: /H&P Note/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Discharge Summary/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Dental Treatment Note/ })).toBeInTheDocument();
  });
});

// ─── Sample Note Loading ─────────────────────────────────────

describe('SectionParser — Sample Notes', () => {
  it('loads H&P Note sample into textarea', () => {
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /H&P Note/ }));
    const textarea = screen.getByPlaceholderText(/Paste a clinical document/) as HTMLTextAreaElement;
    expect(textarea.value).toContain('CHIEF COMPLAINT');
    expect(textarea.value).toContain('ASSESSMENT AND PLAN');
  });

  it('loads Discharge Summary sample', () => {
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /Discharge Summary/ }));
    const textarea = screen.getByPlaceholderText(/Paste a clinical document/) as HTMLTextAreaElement;
    expect(textarea.value).toContain('DISCHARGE SUMMARY');
  });

  it('enables parse button after loading sample', () => {
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /H&P Note/ }));
    const btn = screen.getByRole('button', { name: /Parse Sections/ });
    expect(btn).not.toBeDisabled();
  });

  it('shows word count', () => {
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /H&P Note/ }));
    expect(screen.getByText(/\d+ words/)).toBeInTheDocument();
  });
});

// ─── API Integration ─────────────────────────────────────────

describe('SectionParser — API Integration', () => {
  it('sends POST request to /sections', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /H&P Note/ }));
    fireEvent.click(screen.getByRole('button', { name: /Parse Sections/ }));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/v1/sections',
        expect.objectContaining({ method: 'POST' })
      );
    });
  });

  it('displays section count in stats', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /H&P Note/ }));
    fireEvent.click(screen.getByRole('button', { name: /Parse Sections/ }));

    await waitFor(() => {
      expect(screen.getByText('Sections Found')).toBeInTheDocument();
    });
  });

  it('displays section headers', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /H&P Note/ }));
    fireEvent.click(screen.getByRole('button', { name: /Parse Sections/ }));

    await waitFor(() => {
      expect(screen.getByText('CHIEF COMPLAINT')).toBeInTheDocument();
      expect(screen.getByText('MEDICATIONS')).toBeInTheDocument();
      expect(screen.getByText('PHYSICAL EXAMINATION')).toBeInTheDocument();
    });
  });

  it('displays body text for auto-expanded sections', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /H&P Note/ }));
    fireEvent.click(screen.getByRole('button', { name: /Parse Sections/ }));

    await waitFor(() => {
      expect(screen.getByText(/Chest pain and shortness of breath/)).toBeInTheDocument();
      expect(screen.getByText(/Metoprolol succinate/)).toBeInTheDocument();
    });
  });

  it('shows processing time', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /H&P Note/ }));
    fireEvent.click(screen.getByRole('button', { name: /Parse Sections/ }));

    await waitFor(() => {
      expect(screen.getByText('1.8ms')).toBeInTheDocument();
    });
  });

  it('shows category summary', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /H&P Note/ }));
    fireEvent.click(screen.getByRole('button', { name: /Parse Sections/ }));

    await waitFor(() => {
      expect(screen.getByText(/Category Summary/)).toBeInTheDocument();
    });
  });

  it('shows character position metadata', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /H&P Note/ }));
    fireEvent.click(screen.getByRole('button', { name: /Parse Sections/ }));

    await waitFor(() => {
      expect(screen.getByText(/Start: char 0/)).toBeInTheDocument();
    });
  });
});

// ─── Error Handling ──────────────────────────────────────────

describe('SectionParser — Error Handling', () => {
  it('shows error on API failure', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /H&P Note/ }));
    fireEvent.click(screen.getByRole('button', { name: /Parse Sections/ }));

    await waitFor(() => {
      expect(screen.getByText(/HTTP 500/)).toBeInTheDocument();
    });
  });

  it('shows error on network failure', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Connection refused'));

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /H&P Note/ }));
    fireEvent.click(screen.getByRole('button', { name: /Parse Sections/ }));

    await waitFor(() => {
      expect(screen.getByText(/Connection refused/)).toBeInTheDocument();
    });
  });
});

