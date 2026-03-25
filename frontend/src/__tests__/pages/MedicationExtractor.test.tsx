/**
 * Tests for the MedicationExtractor page component.
 *
 * Verifies page structure, sample note loading, input controls,
 * and result table rendering with sort/filter capabilities.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MedicationExtractor } from '../../pages/MedicationExtractor';

// ─── Mock API Response ───────────────────────────────────────

const mockResponse = {
  medications: [
    {
      drug_name: 'Metformin',
      generic_name: 'metformin',
      brand_names: ['Glucophage'],
      dosage: '500mg',
      route: 'PO',
      frequency: 'BID',
      duration: null,
      indication: 'diabetes',
      prn: false,
      status: 'active',
      confidence: 0.92,
      start_char: 0,
      end_char: 10,
    },
    {
      drug_name: 'Albuterol',
      generic_name: 'albuterol',
      brand_names: ['ProAir', 'Ventolin'],
      dosage: '2 puffs',
      route: 'inhaled',
      frequency: 'q4-6h',
      duration: null,
      indication: 'shortness of breath',
      prn: true,
      status: 'active',
      confidence: 0.85,
      start_char: 50,
      end_char: 60,
    },
    {
      drug_name: 'Hydrochlorothiazide',
      generic_name: 'hydrochlorothiazide',
      brand_names: [],
      dosage: '25mg',
      route: 'PO',
      frequency: null,
      duration: null,
      indication: null,
      prn: false,
      status: 'discontinued',
      confidence: 0.78,
      start_char: 100,
      end_char: 120,
    },
  ],
  count: 3,
  processing_time_ms: 2.4,
};

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <MedicationExtractor />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

describe('MedicationExtractor', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  // ── Page Structure ───────────────────────────────────────

  it('renders the page heading', () => {
    renderPage();
    expect(screen.getByText('Medication Extractor')).toBeInTheDocument();
  });

  it('renders the page description', () => {
    renderPage();
    expect(
      screen.getByText(/Extract structured medication data from clinical text/)
    ).toBeInTheDocument();
  });

  it('renders all sample note buttons', () => {
    renderPage();
    expect(screen.getByText('Discharge Medications')).toBeInTheDocument();
    expect(screen.getByText('Progress Note')).toBeInTheDocument();
    expect(screen.getByText('Dental Note')).toBeInTheDocument();
  });

  it('renders the textarea placeholder', () => {
    renderPage();
    expect(
      screen.getByPlaceholderText(/Paste clinical text containing medication information/)
    ).toBeInTheDocument();
  });

  it('renders the extract button', () => {
    renderPage();
    expect(screen.getByText('Extract Medications')).toBeInTheDocument();
  });

  it('disables the extract button when textarea is empty', () => {
    renderPage();
    const button = screen.getByText('Extract Medications');
    expect(button).toBeDisabled();
  });

  // ── Sample Loading ───────────────────────────────────────

  it('loads Discharge Medications sample into textarea', () => {
    renderPage();
    fireEvent.click(screen.getByText('Discharge Medications'));
    const textarea = screen.getByPlaceholderText(/Paste clinical text/);
    expect((textarea as HTMLTextAreaElement).value).toContain('Metformin');
    expect((textarea as HTMLTextAreaElement).value).toContain('DISCHARGE MEDICATIONS');
  });

  it('loads Progress Note sample', () => {
    renderPage();
    fireEvent.click(screen.getByText('Progress Note'));
    const textarea = screen.getByPlaceholderText(/Paste clinical text/);
    expect((textarea as HTMLTextAreaElement).value).toContain('Lipitor');
  });

  it('enables the extract button after loading a sample', () => {
    renderPage();
    fireEvent.click(screen.getByText('Discharge Medications'));
    const button = screen.getByText('Extract Medications');
    expect(button).not.toBeDisabled();
  });

  // ── Word Count ───────────────────────────────────────────

  it('shows word count after typing', () => {
    renderPage();
    const textarea = screen.getByPlaceholderText(/Paste clinical text/);
    fireEvent.change(textarea, { target: { value: 'Metformin 500mg PO BID' } });
    expect(screen.getByText('4 words')).toBeInTheDocument();
  });

  // ── API Integration ──────────────────────────────────────

  it('calls API and renders results on analyze', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    const textarea = screen.getByPlaceholderText(/Paste clinical text/);
    fireEvent.change(textarea, { target: { value: 'Metformin 500mg PO BID' } });
    fireEvent.click(screen.getByText('Extract Medications'));

    await waitFor(() => {
      expect(screen.getByText('Total Medications')).toBeInTheDocument();
    });

    // Stats
    expect(screen.getByText('3')).toBeInTheDocument();
    expect(screen.getByText('2.4ms')).toBeInTheDocument();

    // Medication rows
    expect(screen.getByText('Metformin')).toBeInTheDocument();
    expect(screen.getByText('Albuterol')).toBeInTheDocument();
    expect(screen.getByText('Hydrochlorothiazide')).toBeInTheDocument();
  });

  it('shows error message on API failure', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
    });

    renderPage();
    const textarea = screen.getByPlaceholderText(/Paste clinical text/);
    fireEvent.change(textarea, { target: { value: 'some text' } });
    fireEvent.click(screen.getByText('Extract Medications'));

    await waitFor(() => {
      expect(screen.getByText(/API error: 500/)).toBeInTheDocument();
    });
  });

  it('displays PRN badge for PRN medications', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    const textarea = screen.getByPlaceholderText(/Paste clinical text/);
    fireEvent.change(textarea, { target: { value: 'test medication note' } });
    fireEvent.click(screen.getByText('Extract Medications'));

    await waitFor(() => {
      // PRN appears both as badge and in stat cards; use getAllByText
      expect(screen.getAllByText('PRN').length).toBeGreaterThanOrEqual(1);
    });
  });

  it('shows medication indication', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Paste clinical text/), {
      target: { value: 'test' },
    });
    fireEvent.click(screen.getByText('Extract Medications'));

    await waitFor(() => {
      expect(screen.getByText('for diabetes')).toBeInTheDocument();
    });
  });

  it('displays status badges with correct text', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Paste clinical text/), {
      target: { value: 'test' },
    });
    fireEvent.click(screen.getByText('Extract Medications'));

    await waitFor(() => {
      const activeBadges = screen.getAllByText('active');
      expect(activeBadges.length).toBeGreaterThanOrEqual(2);
      expect(screen.getByText('discontinued')).toBeInTheDocument();
    });
  });

  // ── Status Filter ────────────────────────────────────────

  it('renders status filter dropdown after analysis', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Paste clinical text/), {
      target: { value: 'test' },
    });
    fireEvent.click(screen.getByText('Extract Medications'));

    await waitFor(() => {
      expect(screen.getByText('Filter by status:')).toBeInTheDocument();
    });
  });

  // ── Table Headers ────────────────────────────────────────

  it('renders sortable table column headers', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Paste clinical text/), {
      target: { value: 'test' },
    });
    fireEvent.click(screen.getByText('Extract Medications'));

    await waitFor(() => {
      expect(screen.getByText('Medication')).toBeInTheDocument();
      expect(screen.getByText('Dosage')).toBeInTheDocument();
      expect(screen.getByText('Route')).toBeInTheDocument();
      expect(screen.getByText('Frequency')).toBeInTheDocument();
      expect(screen.getByText('Status')).toBeInTheDocument();
      expect(screen.getByText('Confidence')).toBeInTheDocument();
    });
  });
});
