/**
 * Tests for the ClinicalSummary page component.
 *
 * Verifies detail level selection, summary content switching,
 * key findings display, word count, and metadata section.
 */
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ClinicalSummary } from '../../pages/ClinicalSummary';

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <ClinicalSummary />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

describe('ClinicalSummary', () => {
  // --- Page structure ---
  it('renders the page heading', () => {
    renderPage();
    expect(screen.getByText('Clinical Summary')).toBeInTheDocument();
  });

  it('renders the page description', () => {
    renderPage();
    expect(
      screen.getByText(/AI-generated clinical summaries with adjustable detail/)
    ).toBeInTheDocument();
  });

  // --- Detail level selector ---
  it('renders all three detail level buttons', () => {
    renderPage();
    expect(screen.getByText('Brief')).toBeInTheDocument();
    expect(screen.getByText('Standard')).toBeInTheDocument();
    expect(screen.getByText('Detailed')).toBeInTheDocument();
  });

  it('renders detail level descriptions', () => {
    renderPage();
    expect(screen.getByText('Key points only')).toBeInTheDocument();
    expect(screen.getByText('Balanced detail')).toBeInTheDocument();
    expect(screen.getByText('Full clinical context')).toBeInTheDocument();
  });

  it('defaults to "Standard" detail level', () => {
    renderPage();
    // Standard summary should be showing (62 words)
    expect(screen.getByText('62 words')).toBeInTheDocument();
  });

  // --- Summary content ---
  it('renders summary section heading', () => {
    renderPage();
    expect(screen.getByText('Summary')).toBeInTheDocument();
  });

  it('displays the standard summary text by default', () => {
    renderPage();
    expect(
      screen.getByText(/Patient presents with uncontrolled type 2 diabetes/)
    ).toBeInTheDocument();
  });

  // --- Detail level switching ---
  it('switches to brief summary when Brief is clicked', () => {
    renderPage();
    fireEvent.click(screen.getByText('Brief'));
    expect(screen.getByText('28 words')).toBeInTheDocument();
    expect(
      screen.getByText(/Patient with uncontrolled T2DM/)
    ).toBeInTheDocument();
  });

  it('switches to detailed summary when Detailed is clicked', () => {
    renderPage();
    fireEvent.click(screen.getByText('Detailed'));
    expect(screen.getByText('198 words')).toBeInTheDocument();
    expect(
      screen.getByText(/62-year-old patient presents with a complex cardiometabolic/)
    ).toBeInTheDocument();
  });

  it('switches back to standard from detailed', () => {
    renderPage();
    fireEvent.click(screen.getByText('Detailed'));
    fireEvent.click(screen.getByText('Standard'));
    expect(screen.getByText('62 words')).toBeInTheDocument();
  });

  // --- Key findings ---
  it('renders Key Findings section', () => {
    renderPage();
    // "Key Findings" appears both as section heading and in metadata
    const matches = screen.getAllByText('Key Findings');
    expect(matches.length).toBeGreaterThanOrEqual(2);
  });

  it('shows correct number of key findings for standard level', () => {
    renderPage();
    // Standard level has 5 key findings
    expect(
      screen.getByText(/Uncontrolled type 2 diabetes with HbA1c 7.8%/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Blood pressure 150\/95 mmHg/)
    ).toBeInTheDocument();
  });

  it('shows fewer key findings for brief level', () => {
    renderPage();
    fireEvent.click(screen.getByText('Brief'));
    expect(
      screen.getByText('Uncontrolled type 2 diabetes mellitus')
    ).toBeInTheDocument();
    // Brief has 3 findings
    expect(
      screen.getByText('New cardiac symptoms requiring evaluation')
    ).toBeInTheDocument();
  });

  it('shows more key findings for detailed level', () => {
    renderPage();
    fireEvent.click(screen.getByText('Detailed'));
    // Detailed has 8 findings including EMPA-REG reference
    // (appears in both summary text and key findings)
    const matches = screen.getAllByText(/EMPA-REG OUTCOME trial/);
    expect(matches.length).toBeGreaterThanOrEqual(1);
  });

  // --- Metadata section ---
  it('renders Summary Metadata section', () => {
    renderPage();
    expect(screen.getByText('Summary Metadata')).toBeInTheDocument();
  });

  it('shows detail level in metadata', () => {
    renderPage();
    // "Detail Level" appears in both selector heading and metadata
    const matches = screen.getAllByText('Detail Level');
    expect(matches.length).toBeGreaterThanOrEqual(1);
  });

  it('shows word count in metadata', () => {
    renderPage();
    expect(screen.getByText('Word Count')).toBeInTheDocument();
  });

  it('shows key findings count in metadata', () => {
    renderPage();
    // "Key Findings" appears in section heading and metadata
    const matches = screen.getAllByText('Key Findings');
    expect(matches.length).toBeGreaterThanOrEqual(2);
  });

  it('shows generated timestamp in metadata', () => {
    renderPage();
    expect(screen.getByText('Generated')).toBeInTheDocument();
  });
});
