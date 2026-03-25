/**
 * Tests for the EntityViewer page component.
 *
 * Verifies entity list rendering, search filtering, type filtering,
 * frequency chart section, entity detail panel, and result counts.
 */
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { EntityViewer } from '../../pages/EntityViewer';

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <EntityViewer />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

describe('EntityViewer', () => {
  // --- Page structure ---
  it('renders the page heading', () => {
    renderPage();
    expect(screen.getByText('Entity Viewer')).toBeInTheDocument();
  });

  it('renders the page description', () => {
    renderPage();
    expect(
      screen.getByText(/Browse and filter extracted clinical entities/)
    ).toBeInTheDocument();
  });

  // --- Frequency chart ---
  it('renders entity frequency chart heading', () => {
    renderPage();
    expect(screen.getByText('Entity Frequency by Type')).toBeInTheDocument();
  });

  // --- Search ---
  it('renders search input', () => {
    renderPage();
    expect(
      screen.getByPlaceholderText('Search entities...')
    ).toBeInTheDocument();
  });

  it('filters entities by search text', () => {
    renderPage();
    const input = screen.getByPlaceholderText('Search entities...');
    fireEvent.change(input, { target: { value: 'metformin' } });
    expect(screen.getByText(/Showing 1 of 22 entities/)).toBeInTheDocument();
  });

  it('shows 0 results for non-matching search', () => {
    renderPage();
    const input = screen.getByPlaceholderText('Search entities...');
    fireEvent.change(input, { target: { value: 'xyznonexistent' } });
    expect(screen.getByText(/Showing 0 of 22 entities/)).toBeInTheDocument();
    expect(
      screen.getByText('No entities match your search criteria.')
    ).toBeInTheDocument();
  });

  // --- Type filter ---
  it('renders type filter dropdown', () => {
    renderPage();
    const select = screen.getByRole('combobox');
    expect(select).toBeInTheDocument();
  });

  it('filters entities by type selection', () => {
    renderPage();
    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'disease' } });
    // Demo data has 4 disease entities
    expect(screen.getByText(/Showing 4 of 22 entities/)).toBeInTheDocument();
  });

  it('shows all entities when "All Types" is selected', () => {
    renderPage();
    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'disease' } });
    fireEvent.change(select, { target: { value: 'all' } });
    expect(screen.getByText(/Showing 22 of 22 entities/)).toBeInTheDocument();
  });

  // --- Entity count ---
  it('shows total entity count', () => {
    renderPage();
    expect(screen.getByText(/Showing 22 of 22 entities/)).toBeInTheDocument();
  });

  // --- Entity list items ---
  it('renders entity names in the list', () => {
    renderPage();
    expect(screen.getByText('type 2 diabetes mellitus')).toBeInTheDocument();
    expect(screen.getByText('metformin')).toBeInTheDocument();
    expect(screen.getByText('echocardiogram')).toBeInTheDocument();
  });

  it('renders CUI identifiers for entities that have them', () => {
    renderPage();
    expect(screen.getByText('C0011860')).toBeInTheDocument();
    expect(screen.getByText('C0020538')).toBeInTheDocument();
    expect(screen.getByText('C0025598')).toBeInTheDocument();
  });

  // --- Detail panel ---
  it('shows default prompt in detail panel', () => {
    renderPage();
    expect(screen.getByText('Entity Details')).toBeInTheDocument();
    expect(
      screen.getByText('Select an entity from the list to view details.')
    ).toBeInTheDocument();
  });

  it('shows entity details when an entity is clicked', () => {
    renderPage();
    // Click on the first entity button
    const entityButton = screen.getByText('type 2 diabetes mellitus').closest('button')!;
    fireEvent.click(entityButton);

    // Detail panel should show type and confidence info
    expect(screen.getByText('Type')).toBeInTheDocument();
    expect(screen.getByText('Confidence')).toBeInTheDocument();
    expect(screen.getByText('Text Span')).toBeInTheDocument();
  });

  it('shows UMLS CUI in detail panel when entity has one', () => {
    renderPage();
    const entityButton = screen.getByText('type 2 diabetes mellitus').closest('button')!;
    fireEvent.click(entityButton);

    expect(screen.getByText('UMLS CUI')).toBeInTheDocument();
  });

  // --- Combined filters ---
  it('applies search and type filter together', () => {
    renderPage();
    const select = screen.getByRole('combobox');
    fireEvent.change(select, { target: { value: 'medication' } });
    const input = screen.getByPlaceholderText('Search entities...');
    fireEvent.change(input, { target: { value: 'aspirin' } });
    expect(screen.getByText(/Showing 1 of 22 entities/)).toBeInTheDocument();
  });
});
