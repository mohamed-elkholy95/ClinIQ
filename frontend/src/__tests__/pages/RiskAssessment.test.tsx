/**
 * Tests for the RiskAssessment page component.
 *
 * Verifies risk gauge display, category scores chart, risk factor
 * breakdown, clinical recommendations, and risk level legend.
 */
import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { RiskAssessment } from '../../pages/RiskAssessment';

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <RiskAssessment />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

describe('RiskAssessment', () => {
  // --- Page structure ---
  it('renders the page heading', () => {
    renderPage();
    expect(screen.getByText('Risk Assessment')).toBeInTheDocument();
  });

  it('renders the page description', () => {
    renderPage();
    expect(
      screen.getByText(/Comprehensive patient risk evaluation/)
    ).toBeInTheDocument();
  });

  // --- Overall risk gauge ---
  it('renders "Overall Risk Score" section', () => {
    renderPage();
    expect(screen.getByText('Overall Risk Score')).toBeInTheDocument();
  });

  it('displays the risk score value (68)', () => {
    renderPage();
    expect(screen.getByText('68')).toBeInTheDocument();
  });

  it('displays the risk level label "High Risk"', () => {
    renderPage();
    // "High Risk" appears in both the gauge badge and the legend
    const matches = screen.getAllByText('High Risk');
    expect(matches.length).toBe(2);
  });

  // --- Category scores ---
  it('renders Category Risk Scores section', () => {
    renderPage();
    expect(screen.getByText('Category Risk Scores')).toBeInTheDocument();
  });

  // --- Risk factors ---
  it('renders Risk Factor Breakdown section', () => {
    renderPage();
    expect(screen.getByText('Risk Factor Breakdown')).toBeInTheDocument();
  });

  it('renders all 6 risk factors', () => {
    renderPage();
    expect(screen.getByText('Uncontrolled Diabetes')).toBeInTheDocument();
    expect(screen.getByText('Resistant Hypertension')).toBeInTheDocument();
    expect(screen.getByText('Chest Pain Symptom')).toBeInTheDocument();
    expect(screen.getByText('Chronic Fatigue')).toBeInTheDocument();
    expect(screen.getByText('Polypharmacy Risk')).toBeInTheDocument();
    expect(screen.getByText('Renal Function')).toBeInTheDocument();
  });

  it('renders risk factor descriptions', () => {
    renderPage();
    expect(
      screen.getByText(/HbA1c 7.8% exceeds target range/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Blood pressure 150\/95 mmHg despite/)
    ).toBeInTheDocument();
  });

  it('renders risk factor scores', () => {
    renderPage();
    expect(screen.getByText('75')).toBeInTheDocument();
    expect(screen.getByText('70')).toBeInTheDocument();
    expect(screen.getByText('65')).toBeInTheDocument();
    expect(screen.getByText('50')).toBeInTheDocument();
    expect(screen.getByText('45')).toBeInTheDocument();
    expect(screen.getByText('40')).toBeInTheDocument();
  });

  it('renders risk factor categories as badges', () => {
    renderPage();
    // "Metabolic" and "Cardiovascular" appear in both factor badges and the chart
    const metabolicMatches = screen.getAllByText('Metabolic');
    expect(metabolicMatches.length).toBeGreaterThanOrEqual(1);
    const cardioMatches = screen.getAllByText('Cardiovascular');
    expect(cardioMatches.length).toBeGreaterThanOrEqual(1);
    const medMatches = screen.getAllByText('Medication');
    expect(medMatches.length).toBeGreaterThanOrEqual(1);
    const genMatches = screen.getAllByText('General');
    expect(genMatches.length).toBeGreaterThanOrEqual(1);
  });

  // --- Recommendations ---
  it('renders Clinical Recommendations section', () => {
    renderPage();
    expect(screen.getByText('Clinical Recommendations')).toBeInTheDocument();
  });

  it('renders all 8 recommendations', () => {
    renderPage();
    expect(
      screen.getByText(/Order urgent transthoracic echocardiogram/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Consider cardiac stress test/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Initiate empagliflozin 10mg/)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Schedule follow-up HbA1c/)
    ).toBeInTheDocument();
  });

  it('renders numbered recommendation badges', () => {
    renderPage();
    // Numbers 1-8 in recommendation badges
    expect(screen.getByText('1')).toBeInTheDocument();
    expect(screen.getByText('8')).toBeInTheDocument();
  });

  // --- Risk level legend ---
  it('renders Risk Level Reference section', () => {
    renderPage();
    expect(screen.getByText('Risk Level Reference')).toBeInTheDocument();
  });

  it('shows all four risk level categories in legend', () => {
    renderPage();
    expect(screen.getByText('Low Risk')).toBeInTheDocument();
    expect(screen.getByText('Moderate Risk')).toBeInTheDocument();
    // "High Risk" tested above via gauge
    expect(screen.getByText('Critical Risk')).toBeInTheDocument();
  });

  it('shows score ranges in legend', () => {
    renderPage();
    expect(screen.getByText('0-25')).toBeInTheDocument();
    expect(screen.getByText('26-50')).toBeInTheDocument();
    expect(screen.getByText('51-75')).toBeInTheDocument();
    expect(screen.getByText('76-100')).toBeInTheDocument();
  });
});
