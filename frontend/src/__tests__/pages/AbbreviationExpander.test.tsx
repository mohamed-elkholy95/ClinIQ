/**
 * Tests for the AbbreviationExpander page component.
 *
 * Covers page structure, sample note loading, abbreviation analysis
 * with mock API responses, results display (table, domain distribution,
 * expanded text), confidence slider, and error handling.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { AbbreviationExpander } from '../../pages/AbbreviationExpander';

// ─── Mocks ───────────────────────────────────────────────────

const mockFetch = vi.fn();
global.fetch = mockFetch;

beforeEach(() => {
  vi.clearAllMocks();
});

const mockResponse = {
  matches: [
    {
      abbreviation: 'HTN',
      expansion: 'hypertension',
      start: 10,
      end: 13,
      confidence: 0.95,
      domain: 'cardiology',
      is_ambiguous: false,
      resolution: 'unambiguous',
      alternative_expansions: [],
    },
    {
      abbreviation: 'DM',
      expansion: 'diabetes mellitus',
      start: 20,
      end: 22,
      confidence: 0.85,
      domain: 'endocrine',
      is_ambiguous: true,
      resolution: 'context',
      alternative_expansions: ['dermatomyositis'],
    },
    {
      abbreviation: 'SOB',
      expansion: 'shortness of breath',
      start: 30,
      end: 33,
      confidence: 0.90,
      domain: 'pulmonology',
      is_ambiguous: false,
      resolution: 'unambiguous',
      alternative_expansions: [],
    },
  ],
  expanded_text: 'Patient with hypertension and diabetes mellitus presents with shortness of breath.',
  total_found: 3,
  ambiguous_count: 1,
  processing_time_ms: 2.5,
};

function renderPage() {
  return render(<AbbreviationExpander />);
}

// ─── Page Structure ──────────────────────────────────────────

describe('AbbreviationExpander — Page Structure', () => {
  it('renders page title and description', () => {
    renderPage();
    expect(screen.getByText(/Abbreviation Expander/)).toBeInTheDocument();
    expect(screen.getByText(/context-aware disambiguation/)).toBeInTheDocument();
  });

  it('renders textarea input', () => {
    renderPage();
    expect(screen.getByPlaceholderText(/Paste clinical text/)).toBeInTheDocument();
  });

  it('renders analyze button (disabled when empty)', () => {
    renderPage();
    const btn = screen.getByRole('button', { name: /Expand Abbreviations/ });
    expect(btn).toBeDisabled();
  });

  it('renders all sample note buttons', () => {
    renderPage();
    expect(screen.getByRole('button', { name: /ED Assessment/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Dental Progress Note/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Discharge Summary/ })).toBeInTheDocument();
  });

  it('renders confidence slider', () => {
    renderPage();
    expect(screen.getByText(/Min Confidence/)).toBeInTheDocument();
  });
});

// ─── Sample Note Loading ─────────────────────────────────────

describe('AbbreviationExpander — Sample Notes', () => {
  it('loads ED Assessment sample into textarea', () => {
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /ED Assessment/ }));
    const textarea = screen.getByPlaceholderText(/Paste clinical text/) as HTMLTextAreaElement;
    expect(textarea.value).toContain('SOB');
    expect(textarea.value).toContain('HTN');
  });

  it('loads Dental sample into textarea', () => {
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /Dental Progress Note/ }));
    const textarea = screen.getByPlaceholderText(/Paste clinical text/) as HTMLTextAreaElement;
    expect(textarea.value).toContain('SRP');
  });

  it('shows word count after loading sample', () => {
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /ED Assessment/ }));
    expect(screen.getByText(/\d+ words/)).toBeInTheDocument();
  });

  it('enables analyze button after loading sample', () => {
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /ED Assessment/ }));
    const btn = screen.getByRole('button', { name: /Expand Abbreviations/ });
    expect(btn).not.toBeDisabled();
  });
});

// ─── API Integration ─────────────────────────────────────────

describe('AbbreviationExpander — API Integration', () => {
  it('sends POST request on analyze', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /ED Assessment/ }));
    fireEvent.click(screen.getByRole('button', { name: /Expand Abbreviations/ }));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/v1/abbreviations',
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        })
      );
    });
  });

  it('displays abbreviation count in results', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /ED Assessment/ }));
    fireEvent.click(screen.getByRole('button', { name: /Expand Abbreviations/ }));

    await waitFor(() => {
      expect(screen.getByText('Abbreviations Found')).toBeInTheDocument();
    });
  });

  it('displays detected abbreviations in table', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /ED Assessment/ }));
    fireEvent.click(screen.getByRole('button', { name: /Expand Abbreviations/ }));

    await waitFor(() => {
      expect(screen.getByText('HTN')).toBeInTheDocument();
      expect(screen.getByText('hypertension')).toBeInTheDocument();
      expect(screen.getByText('DM')).toBeInTheDocument();
      expect(screen.getByText('diabetes mellitus')).toBeInTheDocument();
    });
  });

  it('shows expanded text section', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /ED Assessment/ }));
    fireEvent.click(screen.getByRole('button', { name: /Expand Abbreviations/ }));

    await waitFor(() => {
      expect(screen.getByText(/Expanded Text/)).toBeInTheDocument();
    });
  });

  it('displays ambiguous and clear badges', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /ED Assessment/ }));
    fireEvent.click(screen.getByRole('button', { name: /Expand Abbreviations/ }));

    await waitFor(() => {
      // Ambiguous badge in table row + ambiguous count stat
      const ambiguousBadges = screen.getAllByText(/Ambiguous/);
      expect(ambiguousBadges.length).toBeGreaterThanOrEqual(1);
      const clearBadges = screen.getAllByText(/Clear/);
      expect(clearBadges.length).toBe(2);
    });
  });

  it('shows domain distribution', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /ED Assessment/ }));
    fireEvent.click(screen.getByRole('button', { name: /Expand Abbreviations/ }));

    await waitFor(() => {
      expect(screen.getByText(/Domain Distribution/)).toBeInTheDocument();
      // Domains appear in both table and distribution, so use getAllByText
      const cardioElems = screen.getAllByText('cardiology');
      expect(cardioElems.length).toBeGreaterThanOrEqual(2);
    });
  });

  it('shows processing time', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /ED Assessment/ }));
    fireEvent.click(screen.getByRole('button', { name: /Expand Abbreviations/ }));

    await waitFor(() => {
      expect(screen.getByText('2.5ms')).toBeInTheDocument();
    });
  });
});

// ─── Error Handling ──────────────────────────────────────────

describe('AbbreviationExpander — Error Handling', () => {
  it('shows error on API failure', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /ED Assessment/ }));
    fireEvent.click(screen.getByRole('button', { name: /Expand Abbreviations/ }));

    await waitFor(() => {
      expect(screen.getByText(/HTTP 500/)).toBeInTheDocument();
    });
  });

  it('shows error on network failure', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /ED Assessment/ }));
    fireEvent.click(screen.getByRole('button', { name: /Expand Abbreviations/ }));

    await waitFor(() => {
      expect(screen.getByText(/Network error/)).toBeInTheDocument();
    });
  });
});
