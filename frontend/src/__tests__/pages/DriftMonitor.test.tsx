/**
 * Tests for the DriftMonitor page component.
 *
 * Verifies page structure, overall status rendering, PSI gauge display,
 * per-model status cards, auto-refresh toggle, and error handling.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DriftMonitor } from '../../pages/DriftMonitor';

// ─── Mock API ────────────────────────────────────────────────

const stableResponse = {
  overall_status: 'stable' as const,
  text_distribution_psi: 0.045,
  model_drift: {
    'ner-biobert': 'stable' as const,
    'icd-classifier': 'stable' as const,
    'risk-scorer': 'warning' as const,
  },
  last_updated: '2026-03-25T20:30:00Z',
};

const driftedResponse = {
  overall_status: 'drifted' as const,
  text_distribution_psi: 0.38,
  model_drift: {
    'ner-biobert': 'drifted' as const,
    'icd-classifier': 'warning' as const,
  },
  last_updated: '2026-03-25T21:00:00Z',
};

vi.mock('../../services/clinical', () => ({
  getDriftStatus: vi.fn(),
}));

import { getDriftStatus } from '../../services/clinical';
const mockGetDrift = vi.mocked(getDriftStatus);

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <DriftMonitor />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  mockGetDrift.mockResolvedValue(stableResponse);
});

// ─── Page Structure ──────────────────────────────────────────

describe('DriftMonitor — page structure', () => {
  it('renders the page heading', async () => {
    renderPage();
    expect(screen.getByText('Drift Monitor')).toBeInTheDocument();
  });

  it('renders the description', () => {
    renderPage();
    expect(screen.getByText(/Track data distribution shifts/)).toBeInTheDocument();
  });

  it('renders the refresh button', () => {
    renderPage();
    expect(screen.getByLabelText('Refresh status')).toBeInTheDocument();
  });

  it('renders auto-refresh checkbox', () => {
    renderPage();
    expect(screen.getByLabelText(/Auto-refresh/)).toBeInTheDocument();
  });
});

// ─── Overall Status ──────────────────────────────────────────

describe('DriftMonitor — overall status', () => {
  it('shows overall system status heading', async () => {
    renderPage();
    await waitFor(() => expect(screen.getByText('Overall System Status')).toBeInTheDocument());
  });

  it('shows stable label for stable response', async () => {
    renderPage();
    await waitFor(() => expect(screen.getByTestId('overall-status')).toBeInTheDocument());
    expect(screen.getAllByText('Stable').length).toBeGreaterThanOrEqual(1);
  });

  it('shows drifted label for drifted response', async () => {
    mockGetDrift.mockResolvedValueOnce(driftedResponse);
    renderPage();
    await waitFor(() => expect(screen.getByTestId('overall-status')).toBeInTheDocument());
    expect(screen.getAllByText('Drifted').length).toBeGreaterThanOrEqual(1);
  });
});

// ─── PSI Gauge ───────────────────────────────────────────────

describe('DriftMonitor — PSI gauge', () => {
  it('renders PSI gauge section', async () => {
    renderPage();
    await waitFor(() => expect(screen.getByTestId('psi-gauge')).toBeInTheDocument());
  });

  it('shows PSI value', async () => {
    renderPage();
    await waitFor(() => expect(screen.getByText('0.045')).toBeInTheDocument());
  });

  it('shows stable interpretation for low PSI', async () => {
    renderPage();
    await waitFor(() => expect(screen.getByText('No significant shift')).toBeInTheDocument());
  });

  it('shows significant shift interpretation for high PSI', async () => {
    mockGetDrift.mockResolvedValueOnce(driftedResponse);
    renderPage();
    await waitFor(() => expect(screen.getByText('Significant distribution shift')).toBeInTheDocument());
  });

  it('renders the PSI bar', async () => {
    renderPage();
    await waitFor(() => expect(screen.getByTestId('psi-bar')).toBeInTheDocument());
  });

  it('shows threshold legend', async () => {
    renderPage();
    await waitFor(() => expect(screen.getByTestId('psi-gauge')).toBeInTheDocument());
    expect(screen.getByText(/0.10–0.25 Warning/)).toBeInTheDocument();
  });
});

// ─── Per-Model Status ────────────────────────────────────────

describe('DriftMonitor — per-model status', () => {
  it('renders model status cards', async () => {
    renderPage();
    await waitFor(() => {
      const cards = screen.getAllByTestId('model-status-card');
      expect(cards).toHaveLength(3);
    });
  });

  it('shows model names', async () => {
    renderPage();
    await waitFor(() => expect(screen.getByText('ner-biobert')).toBeInTheDocument());
    expect(screen.getByText('icd-classifier')).toBeInTheDocument();
    expect(screen.getByText('risk-scorer')).toBeInTheDocument();
  });

  it('shows warning status for risk-scorer', async () => {
    renderPage();
    await waitFor(() => expect(screen.getByText('risk-scorer')).toBeInTheDocument());
    // The warning label should appear for risk-scorer
    expect(screen.getAllByText('Warning').length).toBeGreaterThanOrEqual(1);
  });

  it('shows model count in heading', async () => {
    renderPage();
    await waitFor(() => expect(screen.getByText(/Per-Model Status \(3\)/)).toBeInTheDocument());
  });
});

// ─── Refresh ─────────────────────────────────────────────────

describe('DriftMonitor — refresh', () => {
  it('fetches status on initial load', async () => {
    renderPage();
    await waitFor(() => expect(mockGetDrift).toHaveBeenCalledTimes(1));
  });

  it('fetches status on refresh button click', async () => {
    renderPage();
    await waitFor(() => expect(mockGetDrift).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByLabelText('Refresh status'));
    await waitFor(() => expect(mockGetDrift).toHaveBeenCalledTimes(2));
  });
});

// ─── Error Handling ──────────────────────────────────────────

describe('DriftMonitor — error handling', () => {
  it('displays error on API failure', async () => {
    mockGetDrift.mockRejectedValueOnce(new Error('Service unavailable'));
    renderPage();
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Service unavailable'));
  });
});
