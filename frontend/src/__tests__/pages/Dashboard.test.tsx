/**
 * Tests for the Dashboard page component.
 *
 * Verifies stat cards rendering, entity distribution display,
 * recent activity list, and page structure.  We use demo data baked
 * into the component (no API mocking needed yet since Dashboard
 * uses local state with demoStats).
 */
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Dashboard } from '../../pages/Dashboard';

function renderDashboard() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <Dashboard />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

describe('Dashboard', () => {
  it('renders the page heading', () => {
    renderDashboard();
    expect(screen.getByText('Dashboard')).toBeInTheDocument();
  });

  it('renders the page description', () => {
    renderDashboard();
    expect(
      screen.getByText(/Overview of clinical document processing/)
    ).toBeInTheDocument();
  });

  // Stat cards
  it('shows "Documents Processed" card', () => {
    renderDashboard();
    expect(screen.getByText('Documents Processed')).toBeInTheDocument();
    expect(screen.getByText('1,247')).toBeInTheDocument();
  });

  it('shows "Entities Found" card', () => {
    renderDashboard();
    expect(screen.getByText('Entities Found')).toBeInTheDocument();
    expect(screen.getByText('18,432')).toBeInTheDocument();
  });

  it('shows "Avg Risk Score" card', () => {
    renderDashboard();
    expect(screen.getByText('Avg Risk Score')).toBeInTheDocument();
    expect(screen.getByText('42')).toBeInTheDocument();
  });

  it("shows Today's Documents card", () => {
    renderDashboard();
    expect(screen.getByText("Today's Documents")).toBeInTheDocument();
    expect(screen.getByText('23')).toBeInTheDocument();
  });

  // Trend indicators
  it('shows percentage change indicators', () => {
    renderDashboard();
    expect(screen.getByText('+12%')).toBeInTheDocument();
    expect(screen.getByText('+8%')).toBeInTheDocument();
    expect(screen.getByText('-3%')).toBeInTheDocument();
    expect(screen.getByText('+18%')).toBeInTheDocument();
  });

  // Section headings
  it('shows "Processing Volume" chart section', () => {
    renderDashboard();
    expect(screen.getByText('Processing Volume')).toBeInTheDocument();
  });

  it('shows "Recent Activity" section', () => {
    renderDashboard();
    expect(screen.getByText('Recent Activity')).toBeInTheDocument();
  });

  it('shows "Entity Distribution" section', () => {
    renderDashboard();
    expect(screen.getByText('Entity Distribution')).toBeInTheDocument();
  });

  // Entity distribution counts
  it('renders entity type counts', () => {
    renderDashboard();
    expect(screen.getByText('4,521')).toBeInTheDocument(); // disease
    expect(screen.getByText('5,234')).toBeInTheDocument(); // medication
    expect(screen.getByText('2,187')).toBeInTheDocument(); // procedure
  });

  it('renders entity type labels', () => {
    renderDashboard();
    expect(screen.getByText('disease')).toBeInTheDocument();
    expect(screen.getByText('medication')).toBeInTheDocument();
    expect(screen.getByText('lab value')).toBeInTheDocument();
  });

  // Recent activity items
  it('renders recent activity items', () => {
    renderDashboard();
    expect(screen.getByText('Discharge Summary - Patient 4821')).toBeInTheDocument();
    expect(screen.getByText('Radiology Reports Batch #42')).toBeInTheDocument();
  });

  it('renders activity actions', () => {
    renderDashboard();
    const actions = screen.getAllByText(/Document analyzed/);
    expect(actions.length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Batch completed')).toBeInTheDocument();
    expect(screen.getByText('Risk alert')).toBeInTheDocument();
  });
});
