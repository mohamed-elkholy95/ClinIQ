/**
 * Tests for the Deidentification page component.
 *
 * Verifies page structure, strategy selection, PHI type filtering,
 * sample loading, and result rendering including split-pane view
 * and detection badges.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Deidentification } from '../../pages/Deidentification';

// ─── Mock Response ───────────────────────────────────────────

const mockResponse = {
  deidentified_text: 'PATIENT: [NAME]\nDOB: [DATE]\nSSN: [SSN]',
  detections: [
    {
      type: 'NAME' as const,
      text: 'John Smith',
      replacement: '[NAME]',
      start_char: 9,
      end_char: 19,
      confidence: 0.95,
    },
    {
      type: 'DATE' as const,
      text: '03/15/1960',
      replacement: '[DATE]',
      start_char: 25,
      end_char: 35,
      confidence: 0.98,
    },
    {
      type: 'SSN' as const,
      text: '123-45-6789',
      replacement: '[SSN]',
      start_char: 41,
      end_char: 52,
      confidence: 0.99,
    },
    {
      type: 'PHONE' as const,
      text: '(555) 867-5309',
      replacement: '[PHONE]',
      start_char: 100,
      end_char: 114,
      confidence: 0.92,
    },
  ],
  count: 4,
  strategy: 'redact' as const,
  processing_time_ms: 1.8,
};

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <Deidentification />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

describe('Deidentification', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  // ── Page Structure ───────────────────────────────────────

  it('renders the page heading', () => {
    renderPage();
    expect(screen.getByText(/PHI De-identification/)).toBeInTheDocument();
  });

  it('renders HIPAA description', () => {
    renderPage();
    expect(screen.getByText(/HIPAA Safe Harbor/)).toBeInTheDocument();
  });

  // ── Strategy Cards ───────────────────────────────────────

  it('renders all three strategy cards', () => {
    renderPage();
    expect(screen.getByText('Redact')).toBeInTheDocument();
    expect(screen.getByText('Mask')).toBeInTheDocument();
    expect(screen.getByText('Surrogate')).toBeInTheDocument();
  });

  it('shows strategy descriptions', () => {
    renderPage();
    expect(screen.getByText(/Replace PHI with bracketed type labels/)).toBeInTheDocument();
    expect(screen.getByText(/Replace characters with asterisks/)).toBeInTheDocument();
    expect(screen.getByText(/Replace with realistic synthetic values/)).toBeInTheDocument();
  });

  it('shows strategy examples', () => {
    renderPage();
    expect(screen.getByText(/John Smith → \[NAME\]/)).toBeInTheDocument();
  });

  // ── PHI Type Filter ──────────────────────────────────────

  it('renders PHI type filter buttons', () => {
    renderPage();
    // Use getAllByText since labels appear in multiple contexts
    expect(screen.getAllByText(/Name/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/Date/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText(/Phone/).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Filter PHI types (empty = detect all):')).toBeInTheDocument();
  });

  // ── Confidence Slider ────────────────────────────────────

  it('renders confidence slider with default value', () => {
    renderPage();
    expect(screen.getByText(/Min confidence: 70%/)).toBeInTheDocument();
  });

  // ── Sample Note ──────────────────────────────────────────

  it('loads sample note on button click', () => {
    renderPage();
    fireEvent.click(screen.getByText('Load sample note'));
    const textarea = screen.getByPlaceholderText(/Paste clinical text/);
    expect((textarea as HTMLTextAreaElement).value).toContain('John Smith');
    expect((textarea as HTMLTextAreaElement).value).toContain('SSN');
  });

  // ── De-identify Button ───────────────────────────────────

  it('renders the de-identify button', () => {
    renderPage();
    expect(screen.getByText(/De-identify/)).toBeInTheDocument();
  });

  it('disables button when textarea is empty', () => {
    renderPage();
    const button = screen.getByText(/De-identify/);
    expect(button).toBeDisabled();
  });

  it('enables button after entering text', () => {
    renderPage();
    const textarea = screen.getByPlaceholderText(/Paste clinical text/);
    fireEvent.change(textarea, { target: { value: 'Patient: John Smith' } });
    const button = screen.getByText(/De-identify/);
    expect(button).not.toBeDisabled();
  });

  // ── API Integration ──────────────────────────────────────

  it('calls API and renders results', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Paste clinical text/), {
      target: { value: 'Patient: John Smith\nSSN: 123-45-6789' },
    });
    fireEvent.click(screen.getByText(/De-identify/));

    await waitFor(() => {
      expect(screen.getByText('PHI Detected')).toBeInTheDocument();
    });

    // Stats — use getAllByText since "4" may appear multiple times
    expect(screen.getAllByText('4').length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('1.8ms')).toBeInTheDocument();
  });

  it('shows de-identified text in split pane', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Paste clinical text/), {
      target: { value: 'PATIENT: John Smith' },
    });
    fireEvent.click(screen.getByText(/De-identify/));

    await waitFor(() => {
      expect(screen.getByText(/Original Text/)).toBeInTheDocument();
      expect(screen.getByText(/De-identified Text/)).toBeInTheDocument();
    });

    // De-identified output should contain replacement tags
    expect(screen.getAllByText(/\[NAME\]/).length).toBeGreaterThanOrEqual(1);
  });

  it('shows detection badges with type labels', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Paste clinical text/), {
      target: { value: 'test' },
    });
    fireEvent.click(screen.getByText(/De-identify/));

    await waitFor(() => {
      expect(screen.getByText('PHI Detections (4)')).toBeInTheDocument();
    });
  });

  it('shows error on API failure', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Paste clinical text/), {
      target: { value: 'test' },
    });
    fireEvent.click(screen.getByText(/De-identify/));

    await waitFor(() => {
      expect(screen.getByText(/API error: 500/)).toBeInTheDocument();
    });
  });

  it('renders type distribution after analysis', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Paste clinical text/), {
      target: { value: 'test' },
    });
    fireEvent.click(screen.getByText(/De-identify/));

    await waitFor(() => {
      expect(screen.getByText('Detection Distribution')).toBeInTheDocument();
    });
  });
});

