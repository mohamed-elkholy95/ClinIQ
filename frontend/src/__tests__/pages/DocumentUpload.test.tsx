/**
 * Tests for the DocumentUpload page component.
 *
 * Verifies text input, file upload zone, "Load sample" prefill,
 * analyze button state, analysis result rendering (annotated text,
 * entity tags, ICD predictions, risk gauge, and clinical summary).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { DocumentUpload } from '../../pages/DocumentUpload';

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <DocumentUpload />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

describe('DocumentUpload', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  // --- Page structure ---
  it('renders the page heading', () => {
    renderPage();
    expect(screen.getByText('Document Upload')).toBeInTheDocument();
  });

  it('renders the page description', () => {
    renderPage();
    expect(
      screen.getByText(/Upload or paste clinical text for NLP analysis/)
    ).toBeInTheDocument();
  });

  it('renders "Clinical Text Input" section heading', () => {
    renderPage();
    expect(screen.getByText('Clinical Text Input')).toBeInTheDocument();
  });

  it('renders "Load sample" link', () => {
    renderPage();
    expect(screen.getByText('Load sample')).toBeInTheDocument();
  });

  // --- Text area ---
  it('renders textarea with placeholder', () => {
    renderPage();
    const textarea = screen.getByPlaceholderText(/Paste clinical text here/);
    expect(textarea).toBeInTheDocument();
  });

  it('updates textarea on typing', () => {
    renderPage();
    const textarea = screen.getByPlaceholderText(
      /Paste clinical text here/
    ) as HTMLTextAreaElement;
    fireEvent.change(textarea, { target: { value: 'Test note' } });
    expect(textarea.value).toBe('Test note');
  });

  it('shows word count when text is entered', () => {
    renderPage();
    const textarea = screen.getByPlaceholderText(/Paste clinical text here/);
    fireEvent.change(textarea, { target: { value: 'one two three' } });
    expect(screen.getByText('3 words')).toBeInTheDocument();
  });

  // --- Load sample ---
  it('fills textarea with sample text when "Load sample" is clicked', () => {
    renderPage();
    fireEvent.click(screen.getByText('Load sample'));
    const textarea = screen.getByPlaceholderText(
      /Paste clinical text here/
    ) as HTMLTextAreaElement;
    expect(textarea.value).toContain('type 2 diabetes');
  });

  // --- Analyze button ---
  it('renders Analyze button', () => {
    renderPage();
    expect(screen.getByText('Analyze')).toBeInTheDocument();
  });

  it('disables Analyze button when textarea is empty', () => {
    renderPage();
    const btn = screen.getByText('Analyze').closest('button')!;
    expect(btn).toBeDisabled();
  });

  it('enables Analyze button when text is present', () => {
    renderPage();
    const textarea = screen.getByPlaceholderText(/Paste clinical text here/);
    fireEvent.change(textarea, { target: { value: 'clinical note' } });
    const btn = screen.getByText('Analyze').closest('button')!;
    expect(btn).not.toBeDisabled();
  });

  // --- File upload zone ---
  it('renders file upload drop zone', () => {
    renderPage();
    expect(screen.getByText('Click to upload')).toBeInTheDocument();
    expect(screen.getByText(/TXT, DOC, DOCX up to 10MB/)).toBeInTheDocument();
  });

  it('renders hidden file input accepting text files', () => {
    renderPage();
    const input = document.querySelector('input[type="file"]') as HTMLInputElement;
    expect(input).toBeInTheDocument();
    expect(input.accept).toContain('.txt');
  });

  // --- Clear button ---
  it('shows clear button when text is entered', () => {
    renderPage();
    const textarea = screen.getByPlaceholderText(/Paste clinical text here/);
    fireEvent.change(textarea, { target: { value: 'some text' } });
    // The X clear button should be present (within the header area)
    const buttons = screen.getAllByRole('button');
    // There should be at least one additional button (the X) compared to empty state
    expect(buttons.length).toBeGreaterThanOrEqual(3);
  });

  // --- Analysis results (after simulate) ---
  it('shows "Analyzing..." state while processing', () => {
    renderPage();
    const textarea = screen.getByPlaceholderText(/Paste clinical text here/);
    fireEvent.change(textarea, { target: { value: 'Patient has diabetes' } });

    fireEvent.click(screen.getByText('Analyze'));
    expect(screen.getByText('Analyzing...')).toBeInTheDocument();
  });

  it('renders analysis results after simulated delay', () => {
    renderPage();
    const textarea = screen.getByPlaceholderText(/Paste clinical text here/);
    fireEvent.change(textarea, { target: { value: 'Patient has diabetes' } });

    fireEvent.click(screen.getByText('Analyze'));

    act(() => {
      vi.advanceTimersByTime(1500);
    });

    expect(screen.getByText(/Analysis completed in/)).toBeInTheDocument();
  });

  it('renders Annotated Text section after analysis', () => {
    renderPage();
    fireEvent.click(screen.getByText('Load sample'));
    fireEvent.click(screen.getByText('Analyze'));
    act(() => {
      vi.advanceTimersByTime(1500);
    });

    expect(screen.getByText('Annotated Text')).toBeInTheDocument();
  });

  it('renders ICD-10 Predictions section after analysis', () => {
    renderPage();
    fireEvent.click(screen.getByText('Load sample'));
    fireEvent.click(screen.getByText('Analyze'));
    act(() => {
      vi.advanceTimersByTime(1500);
    });

    expect(screen.getByText('ICD-10 Predictions')).toBeInTheDocument();
  });

  it('renders individual ICD codes in results', () => {
    renderPage();
    fireEvent.click(screen.getByText('Load sample'));
    fireEvent.click(screen.getByText('Analyze'));
    act(() => {
      vi.advanceTimersByTime(1500);
    });

    expect(screen.getByText('E11.65')).toBeInTheDocument();
    expect(screen.getByText('I10')).toBeInTheDocument();
  });

  it('renders Risk Assessment section after analysis', () => {
    renderPage();
    fireEvent.click(screen.getByText('Load sample'));
    fireEvent.click(screen.getByText('Analyze'));
    act(() => {
      vi.advanceTimersByTime(1500);
    });

    expect(screen.getByText('Risk Assessment')).toBeInTheDocument();
  });

  it('renders Clinical Summary section after analysis', () => {
    renderPage();
    fireEvent.click(screen.getByText('Load sample'));
    fireEvent.click(screen.getByText('Analyze'));
    act(() => {
      vi.advanceTimersByTime(1500);
    });

    expect(screen.getByText('Clinical Summary')).toBeInTheDocument();
  });

  it('renders Key Findings list after analysis', () => {
    renderPage();
    fireEvent.click(screen.getByText('Load sample'));
    fireEvent.click(screen.getByText('Analyze'));
    act(() => {
      vi.advanceTimersByTime(1500);
    });

    expect(screen.getByText('Key Findings')).toBeInTheDocument();
    expect(
      screen.getByText(/Uncontrolled type 2 diabetes/)
    ).toBeInTheDocument();
  });

  it('renders entity tags in the annotated text section', () => {
    renderPage();
    fireEvent.click(screen.getByText('Load sample'));
    fireEvent.click(screen.getByText('Analyze'));
    act(() => {
      vi.advanceTimersByTime(1500);
    });

    // Entity tags show the entity text — check for medications
    expect(screen.getAllByText('metformin').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('lisinopril').length).toBeGreaterThanOrEqual(1);
  });

  it('displays processing time in results', () => {
    renderPage();
    fireEvent.click(screen.getByText('Load sample'));
    fireEvent.click(screen.getByText('Analyze'));
    act(() => {
      vi.advanceTimersByTime(1500);
    });

    expect(screen.getByText(/342ms/)).toBeInTheDocument();
  });
});
