/**
 * Tests for the SearchExplorer page component.
 *
 * Verifies page structure, advanced options panel, search execution,
 * result rendering with query expansion, score colouring, and empty states.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { SearchExplorer } from '../../pages/SearchExplorer';

// ─── Mock API ────────────────────────────────────────────────

const mockResponse = {
  hits: [
    { document_id: 'doc-001', score: 0.92, snippet: 'Patient presents with chest pain and elevated troponin levels.', title: 'Emergency Visit Note' },
    { document_id: 'doc-002', score: 0.67, snippet: 'History of hypertension and type 2 diabetes mellitus.', title: null },
    { document_id: 'doc-003', score: 0.23, snippet: 'Follow-up visit for chronic kidney disease stage 3.', title: 'Nephrology Consult' },
  ],
  total: 3,
  query_expansion: {
    original_query: 'chest pain troponin',
    expanded_terms: ['myocardial infarction', 'acute coronary syndrome', 'cardiac biomarker'],
  },
  reranked: true,
  processing_time_ms: 42.5,
};

vi.mock('../../services/clinical', () => ({
  searchDocuments: vi.fn(),
}));

import { searchDocuments } from '../../services/clinical';
const mockSearch = vi.mocked(searchDocuments);

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <SearchExplorer />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  mockSearch.mockResolvedValue(mockResponse);
});

// ─── Page Structure ──────────────────────────────────────────

describe('SearchExplorer — page structure', () => {
  it('renders the page heading', () => {
    renderPage();
    expect(screen.getByText('Clinical Document Search')).toBeInTheDocument();
  });

  it('renders the page description', () => {
    renderPage();
    expect(screen.getByText(/Hybrid retrieval/)).toBeInTheDocument();
  });

  it('renders the search input', () => {
    renderPage();
    expect(screen.getByLabelText('Search query')).toBeInTheDocument();
  });

  it('renders the search button', () => {
    renderPage();
    expect(screen.getByRole('button', { name: /Search/i })).toBeInTheDocument();
  });

  it('shows empty state initially', () => {
    renderPage();
    expect(screen.getByText('Search Clinical Documents')).toBeInTheDocument();
  });
});

// ─── Advanced Options ────────────────────────────────────────

describe('SearchExplorer — advanced options', () => {
  it('advanced panel is hidden by default', () => {
    renderPage();
    expect(screen.queryByTestId('advanced-options')).not.toBeInTheDocument();
  });

  it('shows advanced panel when toggled', () => {
    renderPage();
    fireEvent.click(screen.getByText('Advanced Options'));
    expect(screen.getByTestId('advanced-options')).toBeInTheDocument();
  });

  it('renders top-k input in advanced panel', () => {
    renderPage();
    fireEvent.click(screen.getByText('Advanced Options'));
    expect(screen.getByLabelText('Top K')).toBeInTheDocument();
  });

  it('renders query expansion checkbox', () => {
    renderPage();
    fireEvent.click(screen.getByText('Advanced Options'));
    expect(screen.getByLabelText(/Query Expansion/)).toBeInTheDocument();
  });

  it('renders neural reranking checkbox', () => {
    renderPage();
    fireEvent.click(screen.getByText('Advanced Options'));
    expect(screen.getByLabelText(/Neural Reranking/)).toBeInTheDocument();
  });
});

// ─── Search Execution ────────────────────────────────────────

describe('SearchExplorer — search', () => {
  it('disables search button when query is empty', () => {
    renderPage();
    expect(screen.getByRole('button', { name: /Search/i })).toBeDisabled();
  });

  it('enables search button when query has text', () => {
    renderPage();
    fireEvent.change(screen.getByLabelText('Search query'), { target: { value: 'chest pain' } });
    expect(screen.getByRole('button', { name: /Search/i })).not.toBeDisabled();
  });

  it('calls searchDocuments on button click', async () => {
    renderPage();
    fireEvent.change(screen.getByLabelText('Search query'), { target: { value: 'chest pain' } });
    fireEvent.click(screen.getByRole('button', { name: /Search/i }));
    await waitFor(() => expect(mockSearch).toHaveBeenCalledTimes(1));
  });

  it('calls searchDocuments on Enter key', async () => {
    renderPage();
    const input = screen.getByLabelText('Search query');
    fireEvent.change(input, { target: { value: 'chest pain' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(() => expect(mockSearch).toHaveBeenCalledTimes(1));
  });
});

// ─── Results Rendering ───────────────────────────────────────

describe('SearchExplorer — results', () => {
  async function searchAndWait() {
    renderPage();
    fireEvent.change(screen.getByLabelText('Search query'), { target: { value: 'chest pain' } });
    fireEvent.click(screen.getByRole('button', { name: /Search/i }));
    await waitFor(() => expect(screen.getByText('3 results')).toBeInTheDocument());
  }

  it('displays result count', async () => {
    await searchAndWait();
    expect(screen.getByText('3 results')).toBeInTheDocument();
  });

  it('displays processing time', async () => {
    await searchAndWait();
    expect(screen.getByText(/43ms/)).toBeInTheDocument();
  });

  it('shows reranked badge', async () => {
    await searchAndWait();
    expect(screen.getByText('Reranked')).toBeInTheDocument();
  });

  it('renders all result cards', async () => {
    await searchAndWait();
    const cards = screen.getAllByTestId('search-result');
    expect(cards).toHaveLength(3);
  });

  it('displays document title when present', async () => {
    await searchAndWait();
    expect(screen.getByText('Emergency Visit Note')).toBeInTheDocument();
  });

  it('falls back to document_id when title is null', async () => {
    await searchAndWait();
    // doc-002 has no title, so it should show the ID as the heading
    expect(screen.getByText('doc-002')).toBeInTheDocument();
  });

  it('displays snippet text', async () => {
    await searchAndWait();
    expect(screen.getByText(/elevated troponin levels/)).toBeInTheDocument();
  });

  it('displays score percentages', async () => {
    await searchAndWait();
    expect(screen.getByText('92.0%')).toBeInTheDocument();
    expect(screen.getByText('67.0%')).toBeInTheDocument();
  });
});

// ─── Query Expansion ─────────────────────────────────────────

describe('SearchExplorer — query expansion', () => {
  async function searchAndWait() {
    renderPage();
    fireEvent.change(screen.getByLabelText('Search query'), { target: { value: 'chest pain' } });
    fireEvent.click(screen.getByRole('button', { name: /Search/i }));
    await waitFor(() => expect(screen.getByTestId('query-expansion')).toBeInTheDocument());
  }

  it('shows query expansion panel', async () => {
    await searchAndWait();
    expect(screen.getByTestId('query-expansion')).toBeInTheDocument();
  });

  it('shows original query', async () => {
    await searchAndWait();
    expect(screen.getByText(/"chest pain troponin"/)).toBeInTheDocument();
  });

  it('shows expanded terms', async () => {
    await searchAndWait();
    expect(screen.getByText('myocardial infarction')).toBeInTheDocument();
    expect(screen.getByText('acute coronary syndrome')).toBeInTheDocument();
    expect(screen.getByText('cardiac biomarker')).toBeInTheDocument();
  });
});

// ─── Error Handling ──────────────────────────────────────────

describe('SearchExplorer — error handling', () => {
  it('displays error message on API failure', async () => {
    mockSearch.mockRejectedValueOnce(new Error('Network error'));
    renderPage();
    fireEvent.change(screen.getByLabelText('Search query'), { target: { value: 'test' } });
    fireEvent.click(screen.getByRole('button', { name: /Search/i }));
    await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Network error'));
  });

  it('shows no results message for empty hits', async () => {
    mockSearch.mockResolvedValueOnce({ ...mockResponse, hits: [], total: 0, query_expansion: undefined });
    renderPage();
    fireEvent.change(screen.getByLabelText('Search query'), { target: { value: 'obscure query' } });
    fireEvent.click(screen.getByRole('button', { name: /Search/i }));
    await waitFor(() => expect(screen.getByText(/No documents matched/)).toBeInTheDocument());
  });
});
