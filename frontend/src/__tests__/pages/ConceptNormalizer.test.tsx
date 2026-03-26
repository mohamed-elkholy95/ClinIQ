/**
 * Tests for the ConceptNormalizer page component.
 *
 * Covers page structure, single/batch mode toggle, sample batch loading,
 * normalization with mock API responses, result card rendering
 * (match types, CUI, codes, confidence), stats display, and error handling.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ConceptNormalizer } from '../../pages/ConceptNormalizer';

// ─── Mocks ───────────────────────────────────────────────────

const mockFetch = vi.fn();
global.fetch = mockFetch;

beforeEach(() => {
  vi.clearAllMocks();
});

const singleResult = {
  input_text: 'heart attack',
  matched: true,
  match_type: 'alias' as const,
  preferred_term: 'Myocardial Infarction',
  cui: 'C0027051',
  codes: { 'ICD-10': 'I21.9', SNOMED: '22298006' },
  confidence: 0.92,
  processing_time_ms: 1.5,
};

const batchResponse = {
  results: [
    {
      input_text: 'heart attack',
      matched: true,
      match_type: 'alias' as const,
      preferred_term: 'Myocardial Infarction',
      cui: 'C0027051',
      codes: { 'ICD-10': 'I21.9' },
      confidence: 0.92,
    },
    {
      input_text: 'high blood pressure',
      matched: true,
      match_type: 'exact' as const,
      preferred_term: 'Hypertension',
      cui: 'C0020538',
      codes: { 'ICD-10': 'I10', SNOMED: '38341003' },
      confidence: 1.0,
    },
    {
      input_text: 'xyznoterm',
      matched: false,
      match_type: 'none' as const,
      preferred_term: null,
      cui: null,
      codes: {},
      confidence: 0.0,
    },
  ],
  processing_time_ms: 3.2,
};

function renderPage() {
  return render(<ConceptNormalizer />);
}

// ─── Page Structure ──────────────────────────────────────────

describe('ConceptNormalizer — Page Structure', () => {
  it('renders page title and description', () => {
    renderPage();
    expect(screen.getByText(/Concept Normalizer/)).toBeInTheDocument();
    expect(screen.getByText(/standardized concepts/)).toBeInTheDocument();
  });

  it('renders single term input by default', () => {
    renderPage();
    expect(screen.getByPlaceholderText(/Enter a clinical term/)).toBeInTheDocument();
  });

  it('renders normalize button (disabled when empty)', () => {
    renderPage();
    const btn = screen.getByRole('button', { name: /Normalize$/ });
    expect(btn).toBeDisabled();
  });

  it('renders mode toggle buttons', () => {
    renderPage();
    expect(screen.getByRole('button', { name: /Single Term/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Batch Input/ })).toBeInTheDocument();
  });
});

// ─── Mode Toggle ─────────────────────────────────────────────

describe('ConceptNormalizer — Mode Toggle', () => {
  it('switches to batch mode and shows textarea', () => {
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /Batch Input/ }));
    expect(screen.getByPlaceholderText(/one term per line/)).toBeInTheDocument();
  });

  it('shows sample batch buttons in batch mode', () => {
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /Batch Input/ }));
    expect(screen.getByRole('button', { name: /Common Conditions/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Drug Names/ })).toBeInTheDocument();
  });

  it('loads sample batch into textarea', () => {
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /Batch Input/ }));
    fireEvent.click(screen.getByRole('button', { name: /Common Conditions/ }));
    const textarea = screen.getByPlaceholderText(/one term per line/) as HTMLTextAreaElement;
    expect(textarea.value).toContain('heart attack');
    expect(textarea.value).toContain('diabetes');
  });

  it('shows term count in batch mode', () => {
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /Batch Input/ }));
    fireEvent.click(screen.getByRole('button', { name: /Common Conditions/ }));
    expect(screen.getByText(/10 terms/)).toBeInTheDocument();
  });

  it('switches back to single mode', () => {
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /Batch Input/ }));
    fireEvent.click(screen.getByRole('button', { name: /Single Term/ }));
    expect(screen.getByPlaceholderText(/Enter a clinical term/)).toBeInTheDocument();
  });
});

// ─── Single Term API ─────────────────────────────────────────

describe('ConceptNormalizer — Single Term', () => {
  it('sends POST to /normalize for single term', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => singleResult,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Enter a clinical term/), {
      target: { value: 'heart attack' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Normalize$/ }));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/v1/normalize',
        expect.objectContaining({ method: 'POST' })
      );
    });
  });

  it('displays match result with preferred term', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => singleResult,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Enter a clinical term/), {
      target: { value: 'heart attack' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Normalize$/ }));

    await waitFor(() => {
      expect(screen.getByText('Myocardial Infarction')).toBeInTheDocument();
      expect(screen.getByText(/Alias Match/)).toBeInTheDocument();
    });
  });

  it('displays CUI code', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => singleResult,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Enter a clinical term/), {
      target: { value: 'heart attack' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Normalize$/ }));

    await waitFor(() => {
      expect(screen.getByText('C0027051')).toBeInTheDocument();
    });
  });

  it('displays code system mappings', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => singleResult,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Enter a clinical term/), {
      target: { value: 'heart attack' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Normalize$/ }));

    await waitFor(() => {
      expect(screen.getByText('I21.9')).toBeInTheDocument();
      expect(screen.getByText('22298006')).toBeInTheDocument();
    });
  });

  it('displays confidence bar', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => singleResult,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Enter a clinical term/), {
      target: { value: 'heart attack' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Normalize$/ }));

    await waitFor(() => {
      expect(screen.getByText('92%')).toBeInTheDocument();
    });
  });
});

// ─── Batch API ───────────────────────────────────────────────

describe('ConceptNormalizer — Batch', () => {
  it('sends POST to /normalize/batch', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => batchResponse,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /Batch Input/ }));
    fireEvent.click(screen.getByRole('button', { name: /Common Conditions/ }));
    fireEvent.click(screen.getByRole('button', { name: /Normalize$/ }));

    await waitFor(() => {
      expect(mockFetch).toHaveBeenCalledWith(
        '/api/v1/normalize/batch',
        expect.objectContaining({ method: 'POST' })
      );
    });
  });

  it('displays batch stats', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => batchResponse,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /Batch Input/ }));
    fireEvent.click(screen.getByRole('button', { name: /Common Conditions/ }));
    fireEvent.click(screen.getByRole('button', { name: /Normalize$/ }));

    await waitFor(() => {
      expect(screen.getByText('Total Terms')).toBeInTheDocument();
      expect(screen.getByText('Match Rate')).toBeInTheDocument();
      expect(screen.getByText('67%')).toBeInTheDocument();
    });
  });

  it('displays No Match card for unmatched terms', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => batchResponse,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /Batch Input/ }));
    fireEvent.click(screen.getByRole('button', { name: /Common Conditions/ }));
    fireEvent.click(screen.getByRole('button', { name: /Normalize$/ }));

    await waitFor(() => {
      expect(screen.getByText('xyznoterm')).toBeInTheDocument();
      expect(screen.getByText(/No Match/)).toBeInTheDocument();
    });
  });

  it('displays multiple result cards', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => batchResponse,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /Batch Input/ }));
    fireEvent.click(screen.getByRole('button', { name: /Common Conditions/ }));
    fireEvent.click(screen.getByRole('button', { name: /Normalize$/ }));

    await waitFor(() => {
      expect(screen.getByText('heart attack')).toBeInTheDocument();
      expect(screen.getByText('high blood pressure')).toBeInTheDocument();
      expect(screen.getByText('Hypertension')).toBeInTheDocument();
    });
  });

  it('shows exact match count in stats', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => batchResponse,
    });

    renderPage();
    fireEvent.click(screen.getByRole('button', { name: /Batch Input/ }));
    fireEvent.click(screen.getByRole('button', { name: /Common Conditions/ }));
    fireEvent.click(screen.getByRole('button', { name: /Normalize$/ }));

    await waitFor(() => {
      expect(screen.getByText('Exact')).toBeInTheDocument();
      expect(screen.getByText('Alias')).toBeInTheDocument();
    });
  });
});

// ─── Error Handling ──────────────────────────────────────────

describe('ConceptNormalizer — Error Handling', () => {
  it('shows error on API failure', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 422,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Enter a clinical term/), {
      target: { value: 'test' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Normalize$/ }));

    await waitFor(() => {
      expect(screen.getByText(/HTTP 422/)).toBeInTheDocument();
    });
  });

  it('shows error on network failure', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Failed to fetch'));

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Enter a clinical term/), {
      target: { value: 'test' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Normalize$/ }));

    await waitFor(() => {
      expect(screen.getByText(/Failed to fetch/)).toBeInTheDocument();
    });
  });
});
