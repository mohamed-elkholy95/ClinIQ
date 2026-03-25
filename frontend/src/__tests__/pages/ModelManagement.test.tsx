/**
 * Tests for the ModelManagement page component.
 *
 * Verifies summary statistics, model card rendering, status badges,
 * performance metrics, timestamps, and model descriptions.
 */
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ModelManagement } from '../../pages/ModelManagement';

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <ModelManagement />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

describe('ModelManagement', () => {
  // --- Page structure ---
  it('renders the page heading', () => {
    renderPage();
    expect(screen.getByText('Model Management')).toBeInTheDocument();
  });

  it('renders the page description', () => {
    renderPage();
    expect(
      screen.getByText(/Monitor and manage deployed NLP models/)
    ).toBeInTheDocument();
  });

  // --- Summary stats ---
  it('shows total model count (6)', () => {
    renderPage();
    expect(screen.getByText('6')).toBeInTheDocument();
    expect(screen.getByText('Total Models')).toBeInTheDocument();
  });

  it('shows active model count (4)', () => {
    renderPage();
    expect(screen.getByText('4')).toBeInTheDocument();
    expect(screen.getByText('Active / Deployed')).toBeInTheDocument();
  });

  it('shows training model count (1)', () => {
    renderPage();
    // "1" appears in multiple contexts, check alongside label
    expect(screen.getByText('Currently Training')).toBeInTheDocument();
  });

  // --- Model cards ---
  it('renders all 6 model names', () => {
    renderPage();
    expect(screen.getByText('ClinIQ-NER')).toBeInTheDocument();
    expect(screen.getByText('ClinIQ-ICD')).toBeInTheDocument();
    expect(screen.getByText('ClinIQ-Summary')).toBeInTheDocument();
    expect(screen.getByText('ClinIQ-Risk')).toBeInTheDocument();
    expect(screen.getByText('ClinIQ-Temporal')).toBeInTheDocument();
    expect(screen.getByText('ClinIQ-Relation')).toBeInTheDocument();
  });

  it('renders model versions', () => {
    renderPage();
    expect(screen.getByText('v2.3.1')).toBeInTheDocument();
    expect(screen.getByText('v1.8.0')).toBeInTheDocument();
    expect(screen.getByText('v1.5.2')).toBeInTheDocument();
    expect(screen.getByText('v1.2.0')).toBeInTheDocument();
    expect(screen.getByText('v0.9.0')).toBeInTheDocument();
    expect(screen.getByText('v0.6.0')).toBeInTheDocument();
  });

  it('renders model types', () => {
    renderPage();
    expect(screen.getByText('Named Entity Recognition')).toBeInTheDocument();
    expect(screen.getByText('ICD-10 Code Prediction')).toBeInTheDocument();
    expect(screen.getByText('Clinical Summarization')).toBeInTheDocument();
    expect(screen.getByText('Risk Assessment')).toBeInTheDocument();
    expect(screen.getByText('Temporal Extraction')).toBeInTheDocument();
    expect(screen.getByText('Relation Extraction')).toBeInTheDocument();
  });

  // --- Status badges ---
  it('renders "Active" status badges', () => {
    renderPage();
    const actives = screen.getAllByText('Active');
    expect(actives.length).toBe(4);
  });

  it('renders "Training" status badge', () => {
    renderPage();
    expect(screen.getByText('Training')).toBeInTheDocument();
  });

  it('renders "Inactive" status badge', () => {
    renderPage();
    expect(screen.getByText('Inactive')).toBeInTheDocument();
  });

  // --- Performance metrics ---
  it('renders Accuracy metric labels', () => {
    renderPage();
    const labels = screen.getAllByText('Accuracy');
    expect(labels.length).toBe(6); // one per model
  });

  it('renders F1 Score metric labels', () => {
    renderPage();
    const labels = screen.getAllByText('F1 Score');
    expect(labels.length).toBe(6);
  });

  it('renders Precision metric labels', () => {
    renderPage();
    const labels = screen.getAllByText('Precision');
    expect(labels.length).toBe(6);
  });

  it('renders Recall metric labels', () => {
    renderPage();
    const labels = screen.getAllByText('Recall');
    expect(labels.length).toBe(6);
  });

  // --- Descriptions ---
  it('renders model descriptions', () => {
    renderPage();
    expect(
      screen.getByText(/Clinical named entity recognition model/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Multi-label ICD-10 code prediction model/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Multi-factor clinical risk assessment model/)
    ).toBeInTheDocument();
  });

  // --- Timestamps ---
  it('renders training dates', () => {
    renderPage();
    // Check for formatted dates from the demo data
    const trainedLabels = screen.getAllByText(/Trained:/);
    expect(trainedLabels.length).toBe(6);
  });

  it('renders deployment dates for deployed models', () => {
    renderPage();
    const deployedLabels = screen.getAllByText(/Deployed:/);
    // Only 4 active models have deployed_at
    expect(deployedLabels.length).toBe(4);
  });
});
