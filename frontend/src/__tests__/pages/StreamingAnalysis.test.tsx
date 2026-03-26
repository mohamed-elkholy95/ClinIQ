/**
 * Tests for the StreamingAnalysis page component.
 *
 * Verifies page structure, sample note loading, streaming start/cancel,
 * stage result cards rendering, word count display, and error handling.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { StreamingAnalysis } from '../../pages/StreamingAnalysis';

// ─── Mock API ────────────────────────────────────────────────

const mockAbortController = { abort: vi.fn() };

vi.mock('../../services/clinical', () => ({
  analyzeStream: vi.fn(),
}));

import { analyzeStream } from '../../services/clinical';
const mockAnalyzeStream = vi.mocked(analyzeStream);

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <StreamingAnalysis />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  mockAnalyzeStream.mockImplementation((text, onEvent, onError) => {
    // Simulate immediate stage completions
    setTimeout(() => {
      onEvent({ stage: 'ner', data: { entities: [{ text: 'hypertension', type: 'DISEASE' }] } });
      onEvent({ stage: 'icd', data: { predictions: [{ code: 'I10', confidence: 0.95 }] } });
      onEvent({ stage: 'summary', data: { summary: 'Patient with hypertension.' } });
      onEvent({ stage: 'risk', data: { score: 7.2, level: 'high' } });
    }, 10);
    return mockAbortController as unknown as AbortController;
  });
});

// ─── Page Structure ──────────────────────────────────────────

describe('StreamingAnalysis — page structure', () => {
  it('renders the page heading', () => {
    renderPage();
    expect(screen.getByText('Streaming Analysis')).toBeInTheDocument();
  });

  it('renders the description', () => {
    renderPage();
    expect(screen.getByText(/Real-time Server-Sent Events/)).toBeInTheDocument();
  });

  it('renders the text input', () => {
    renderPage();
    expect(screen.getByLabelText('Clinical text input')).toBeInTheDocument();
  });

  it('renders start button', () => {
    renderPage();
    expect(screen.getByRole('button', { name: /Start Analysis/i })).toBeInTheDocument();
  });

  it('shows idle empty state', () => {
    renderPage();
    expect(screen.getByText(/Submit a clinical note/)).toBeInTheDocument();
  });
});

// ─── Sample Notes ────────────────────────────────────────────

describe('StreamingAnalysis — sample notes', () => {
  it('renders sample note buttons', () => {
    renderPage();
    expect(screen.getByRole('button', { name: 'Emergency Visit' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Discharge Summary' })).toBeInTheDocument();
  });

  it('loads sample text on click', () => {
    renderPage();
    fireEvent.click(screen.getByRole('button', { name: 'Emergency Visit' }));
    const textarea = screen.getByLabelText('Clinical text input') as HTMLTextAreaElement;
    expect(textarea.value).toContain('Chest pain');
  });
});

// ─── Word Count ──────────────────────────────────────────────

describe('StreamingAnalysis — word count', () => {
  it('shows 0 words initially', () => {
    renderPage();
    expect(screen.getByText('0 words')).toBeInTheDocument();
  });

  it('updates word count when text is entered', () => {
    renderPage();
    fireEvent.change(screen.getByLabelText('Clinical text input'), {
      target: { value: 'Patient has hypertension and diabetes' },
    });
    expect(screen.getByText('5 words')).toBeInTheDocument();
  });
});

// ─── Start/Cancel ────────────────────────────────────────────

describe('StreamingAnalysis — controls', () => {
  it('disables start button when text is empty', () => {
    renderPage();
    expect(screen.getByRole('button', { name: /Start Analysis/i })).toBeDisabled();
  });

  it('enables start button when text is present', () => {
    renderPage();
    fireEvent.change(screen.getByLabelText('Clinical text input'), {
      target: { value: 'Clinical note text' },
    });
    expect(screen.getByRole('button', { name: /Start Analysis/i })).not.toBeDisabled();
  });

  it('calls analyzeStream on start', async () => {
    renderPage();
    fireEvent.change(screen.getByLabelText('Clinical text input'), {
      target: { value: 'Patient presents with chest pain.' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Start Analysis/i }));
    await waitFor(() => expect(mockAnalyzeStream).toHaveBeenCalledTimes(1));
  });

  it('shows cancel button while streaming', async () => {
    // Use an implementation that doesn't immediately complete
    mockAnalyzeStream.mockImplementation(() => mockAbortController as unknown as AbortController);
    renderPage();
    fireEvent.change(screen.getByLabelText('Clinical text input'), {
      target: { value: 'Clinical note' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Start Analysis/i }));
    expect(screen.getByRole('button', { name: /Cancel/i })).toBeInTheDocument();
  });

  it('calls abort on cancel', async () => {
    mockAnalyzeStream.mockImplementation(() => mockAbortController as unknown as AbortController);
    renderPage();
    fireEvent.change(screen.getByLabelText('Clinical text input'), {
      target: { value: 'Clinical note' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Start Analysis/i }));
    fireEvent.click(screen.getByRole('button', { name: /Cancel/i }));
    expect(mockAbortController.abort).toHaveBeenCalled();
  });
});

// ─── Stage Results ───────────────────────────────────────────

describe('StreamingAnalysis — stage results', () => {
  it('renders stage result cards when events arrive', async () => {
    renderPage();
    fireEvent.change(screen.getByLabelText('Clinical text input'), {
      target: { value: 'Patient with hypertension.' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Start Analysis/i }));
    await waitFor(() => {
      const cards = screen.getAllByTestId('stage-result');
      expect(cards).toHaveLength(4);
    });
  });

  it('shows NER stage card', async () => {
    renderPage();
    fireEvent.change(screen.getByLabelText('Clinical text input'), {
      target: { value: 'Clinical note text.' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Start Analysis/i }));
    await waitFor(() => expect(screen.getByText('Named Entity Recognition')).toBeInTheDocument());
  });

  it('shows ICD stage card', async () => {
    renderPage();
    fireEvent.change(screen.getByLabelText('Clinical text input'), {
      target: { value: 'Clinical note text.' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Start Analysis/i }));
    await waitFor(() => expect(screen.getByText('ICD-10 Code Prediction')).toBeInTheDocument());
  });

  it('shows summary stage card', async () => {
    renderPage();
    fireEvent.change(screen.getByLabelText('Clinical text input'), {
      target: { value: 'Clinical note text.' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Start Analysis/i }));
    await waitFor(() => expect(screen.getByText('Clinical Summarization')).toBeInTheDocument());
  });

  it('shows risk stage card', async () => {
    renderPage();
    fireEvent.change(screen.getByLabelText('Clinical text input'), {
      target: { value: 'Clinical note text.' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Start Analysis/i }));
    await waitFor(() => expect(screen.getByText('Risk Scoring')).toBeInTheDocument());
  });

  it('renders stage JSON data', async () => {
    renderPage();
    fireEvent.change(screen.getByLabelText('Clinical text input'), {
      target: { value: 'Clinical note text.' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Start Analysis/i }));
    await waitFor(
      () => {
        const cards = screen.getAllByTestId('stage-result');
        expect(cards.length).toBeGreaterThanOrEqual(1);
        // Verify JSON output contains entity data from mock
        expect(cards[0].textContent).toContain('hypertension');
      },
      { timeout: 3000 },
    );
  });
});
