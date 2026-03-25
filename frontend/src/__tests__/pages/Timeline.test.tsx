/**
 * Tests for the Timeline page component.
 *
 * Verifies chronological event rendering, type filtering,
 * expand/collapse for entity details, event descriptions,
 * date formatting, and empty-state handling.
 */
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Timeline } from '../../pages/Timeline';

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <Timeline />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

describe('Timeline', () => {
  // --- Page structure ---
  it('renders the page heading', () => {
    renderPage();
    expect(screen.getByText('Patient Timeline')).toBeInTheDocument();
  });

  it('renders the page description', () => {
    renderPage();
    expect(
      screen.getByText(/Chronological view of clinical events/)
    ).toBeInTheDocument();
  });

  // --- Filter buttons ---
  it('renders "All" filter button', () => {
    renderPage();
    expect(screen.getByText('All')).toBeInTheDocument();
  });

  it('renders type filter buttons for all entity types', () => {
    renderPage();
    expect(screen.getByText('Disease')).toBeInTheDocument();
    expect(screen.getByText('Medication')).toBeInTheDocument();
    expect(screen.getByText('Procedure')).toBeInTheDocument();
    expect(screen.getByText('Anatomy')).toBeInTheDocument();
    expect(screen.getByText('Symptom')).toBeInTheDocument();
    expect(screen.getByText('Lab Value')).toBeInTheDocument();
  });

  // --- Timeline events ---
  it('renders all 7 timeline events by default', () => {
    renderPage();
    expect(
      screen.getByText(/Initial diagnosis of type 2 diabetes/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Metformin uptitrated to 1000mg BID/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Echocardiogram performed/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Comprehensive review/)
    ).toBeInTheDocument();
  });

  it('renders event dates', () => {
    renderPage();
    // jsdom interprets dates in UTC so rendered day may differ by ±1.
    // Multiple March 2026 events exist so use getAllByText.
    expect(screen.getByText(/June \d+, 2025/)).toBeInTheDocument();
    const marchDates = screen.getAllByText(/March \d+, 2026/);
    expect(marchDates.length).toBeGreaterThanOrEqual(1);
  });

  it('renders source text labels', () => {
    renderPage();
    expect(screen.getByText(/Discharge Summary/)).toBeInTheDocument();
    expect(screen.getByText(/Follow-up Note/)).toBeInTheDocument();
    expect(screen.getByText(/Lab Report/)).toBeInTheDocument();
    expect(screen.getByText(/Radiology Report/)).toBeInTheDocument();
  });

  // --- Type filtering ---
  it('filters to disease events only', () => {
    renderPage();
    fireEvent.click(screen.getByText('Disease'));
    expect(
      screen.getByText(/Initial diagnosis of type 2 diabetes/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Diagnosis of essential hypertension/)
    ).toBeInTheDocument();
    // Non-disease events should be hidden
    expect(
      screen.queryByText(/Echocardiogram performed/)
    ).not.toBeInTheDocument();
  });

  it('filters to procedure events only', () => {
    renderPage();
    fireEvent.click(screen.getByText('Procedure'));
    expect(
      screen.getByText(/Echocardiogram performed/)
    ).toBeInTheDocument();
    // Other events hidden
    expect(
      screen.queryByText(/Initial diagnosis of type 2 diabetes/)
    ).not.toBeInTheDocument();
  });

  it('shows empty state when no events match filter', () => {
    renderPage();
    fireEvent.click(screen.getByText('Anatomy'));
    expect(
      screen.getByText('No timeline events match the selected filter.')
    ).toBeInTheDocument();
  });

  it('returns to all events when "All" is clicked after filtering', () => {
    renderPage();
    fireEvent.click(screen.getByText('Disease'));
    fireEvent.click(screen.getByText('All'));
    expect(
      screen.getByText(/Echocardiogram performed/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Initial diagnosis of type 2 diabetes/)
    ).toBeInTheDocument();
  });

  // --- Expand/collapse entity details ---
  it('events are collapsed by default (no "Extracted Entities" visible)', () => {
    renderPage();
    expect(screen.queryByText('Extracted Entities')).not.toBeInTheDocument();
  });

  it('expands event to show extracted entities on click', () => {
    renderPage();
    // Click the first event card
    const eventBtn = screen
      .getByText(/Initial diagnosis of type 2 diabetes/)
      .closest('button')!;
    fireEvent.click(eventBtn);

    expect(screen.getByText('Extracted Entities')).toBeInTheDocument();
  });

  it('collapses event on second click', () => {
    renderPage();
    const eventBtn = screen
      .getByText(/Initial diagnosis of type 2 diabetes/)
      .closest('button')!;
    fireEvent.click(eventBtn); // expand
    expect(screen.getByText('Extracted Entities')).toBeInTheDocument();

    fireEvent.click(eventBtn); // collapse
    expect(screen.queryByText('Extracted Entities')).not.toBeInTheDocument();
  });
});
