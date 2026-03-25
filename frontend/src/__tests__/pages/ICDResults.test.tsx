/**
 * Tests for the ICDResults page component.
 *
 * Verifies chapter grouping, expand/collapse, search filtering,
 * confidence slider, ICD code display, evidence tags, and external links.
 */
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ICDResults } from '../../pages/ICDResults';

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <ICDResults />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

describe('ICDResults', () => {
  // --- Page structure ---
  it('renders the page heading', () => {
    renderPage();
    expect(screen.getByText('ICD-10 Predictions')).toBeInTheDocument();
  });

  it('renders the page description', () => {
    renderPage();
    expect(
      screen.getByText(/Predicted ICD-10 codes with confidence scores/)
    ).toBeInTheDocument();
  });

  // --- Search ---
  it('renders search input', () => {
    renderPage();
    expect(
      screen.getByPlaceholderText('Search by code or description...')
    ).toBeInTheDocument();
  });

  it('filters predictions by code search', () => {
    renderPage();
    const input = screen.getByPlaceholderText('Search by code or description...');
    fireEvent.change(input, { target: { value: 'E11' } });
    // Should show only E11.* codes (2 in demo data)
    expect(screen.getByText(/2 predictions/)).toBeInTheDocument();
  });

  it('filters predictions by description search', () => {
    renderPage();
    const input = screen.getByPlaceholderText('Search by code or description...');
    fireEvent.change(input, { target: { value: 'hypertension' } });
    expect(screen.getByText(/1 prediction/)).toBeInTheDocument();
  });

  // --- Confidence slider ---
  it('renders confidence slider', () => {
    renderPage();
    expect(screen.getByText('Min confidence:')).toBeInTheDocument();
    const slider = screen.getByRole('slider');
    expect(slider).toBeInTheDocument();
  });

  it('filters out low-confidence predictions', () => {
    renderPage();
    const slider = screen.getByRole('slider');
    // Set to 80%: only predictions ≥ 0.80 should remain
    fireEvent.change(slider, { target: { value: '0.80' } });
    expect(screen.getByText('80%')).toBeInTheDocument();
    // Several low-conf predictions filtered out
    expect(screen.getByText(/predictions in/)).toBeInTheDocument();
  });

  // --- Chapter grouping ---
  it('shows chapter headers for grouped predictions', () => {
    renderPage();
    expect(
      screen.getByText('Endocrine, nutritional and metabolic diseases')
    ).toBeInTheDocument();
    expect(
      screen.getByText('Diseases of the circulatory system')
    ).toBeInTheDocument();
    expect(
      screen.getByText('Symptoms, signs and abnormal findings')
    ).toBeInTheDocument();
    expect(
      screen.getByText('Factors influencing health status')
    ).toBeInTheDocument();
  });

  it('shows prediction counts per chapter', () => {
    renderPage();
    // Endocrine has 2 codes, circulatory has 3, symptoms has 3-4, factors has 2
    const codeTexts = screen.getAllByText(/code/);
    expect(codeTexts.length).toBeGreaterThanOrEqual(4);
  });

  // --- Expand/collapse ---
  it('chapters are collapsed by default', () => {
    renderPage();
    // E11.65 should not be visible because chapter is collapsed
    expect(screen.queryByText('E11.65')).not.toBeInTheDocument();
  });

  it('expands chapter on click to reveal codes', () => {
    renderPage();
    const chapterBtn = screen
      .getByText('Endocrine, nutritional and metabolic diseases')
      .closest('button')!;
    fireEvent.click(chapterBtn);

    expect(screen.getByText('E11.65')).toBeInTheDocument();
    expect(
      screen.getByText('Type 2 diabetes mellitus with hyperglycemia')
    ).toBeInTheDocument();
  });

  it('"Expand all" reveals all codes', () => {
    renderPage();
    fireEvent.click(screen.getByText('Expand all'));

    expect(screen.getByText('E11.65')).toBeInTheDocument();
    expect(screen.getByText('I10')).toBeInTheDocument();
    expect(screen.getByText('R53.83')).toBeInTheDocument();
    expect(screen.getByText('Z79.84')).toBeInTheDocument();
  });

  it('"Collapse all" hides all codes', () => {
    renderPage();
    fireEvent.click(screen.getByText('Expand all'));
    expect(screen.getByText('E11.65')).toBeInTheDocument();

    fireEvent.click(screen.getByText('Collapse all'));
    expect(screen.queryByText('E11.65')).not.toBeInTheDocument();
  });

  // --- Code details (expanded) ---
  it('shows evidence tags for predictions', () => {
    renderPage();
    fireEvent.click(screen.getByText('Expand all'));

    expect(screen.getByText('type 2 diabetes mellitus')).toBeInTheDocument();
    expect(screen.getByText('HbA1c 7.8%')).toBeInTheDocument();
    expect(screen.getByText('metformin therapy')).toBeInTheDocument();
  });

  it('shows category for predictions', () => {
    renderPage();
    fireEvent.click(screen.getByText('Expand all'));

    // "Category: Diabetes mellitus" appears for both E11.65 and E11.22
    const diabetesCategories = screen.getAllByText('Category: Diabetes mellitus');
    expect(diabetesCategories.length).toBe(2);
    expect(screen.getByText('Category: Hypertensive diseases')).toBeInTheDocument();
  });

  it('renders external WHO ICD links', () => {
    renderPage();
    fireEvent.click(screen.getByText('Expand all'));

    const links = document.querySelectorAll('a[href*="icd.who.int"]');
    expect(links.length).toBeGreaterThanOrEqual(1);
    expect(links[0].getAttribute('target')).toBe('_blank');
  });

  // --- Total count ---
  it('shows total prediction count', () => {
    renderPage();
    expect(screen.getByText(/11 predictions in 4 chapters/)).toBeInTheDocument();
  });
});
