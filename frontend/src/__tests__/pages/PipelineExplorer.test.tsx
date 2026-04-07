/**
 * Tests for the PipelineExplorer page component.
 *
 * Verifies module toggle grid, enable/disable all, sample loading,
 * analysis submission, and result section rendering for the enhanced
 * 14-module clinical NLP pipeline.
 */
import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PipelineExplorer } from '../../pages/PipelineExplorer';

// ─── Minimal Mock Response ───────────────────────────────────

const mockResponse = {
  classification: {
    predicted_type: 'discharge_summary',
    scores: [
      { document_type: 'discharge_summary', score: 0.92, evidence: ['DISCHARGE SUMMARY'] },
      { document_type: 'progress_note', score: 0.05, evidence: [] },
    ],
    confidence: 0.92,
    processing_time_ms: 0.8,
  },
  sections: {
    sections: [
      {
        category: 'chief_complaint',
        header_text: 'CHIEF COMPLAINT:',
        body_text: 'Chest pain and shortness of breath for 3 days.',
        header_start: 50,
        header_end: 66,
        body_end: 112,
        confidence: 1.0,
      },
    ],
    count: 1,
    processing_time_ms: 0.3,
  },
  quality: {
    overall_score: 78,
    grade: 'C',
    dimensions: [
      { dimension: 'completeness', score: 85, weight: 0.3, findings: [] },
    ],
    recommendations: ['Consider adding more detail to the Assessment section'],
    processing_time_ms: 1.2,
  },
  deidentification: null,
  abbreviations: {
    abbreviations: [
      {
        abbreviation: 'HTN',
        expansion: 'hypertension',
        domain: 'cardiology',
        is_ambiguous: false,
        confidence: 0.95,
        start_char: 200,
        end_char: 203,
      },
    ],
    expanded_text: null,
    count: 1,
    processing_time_ms: 0.5,
  },
  medications: {
    medications: [
      {
        drug_name: 'Metformin',
        generic_name: 'metformin',
        brand_names: [],
        dosage: '1000mg',
        route: 'PO',
        frequency: 'BID',
        duration: null,
        indication: null,
        prn: false,
        status: 'active',
        confidence: 0.88,
        start_char: 300,
        end_char: 310,
      },
    ],
    count: 1,
    processing_time_ms: 1.5,
  },
  allergies: {
    allergies: [
      {
        allergen: 'penicillin',
        canonical_name: 'Penicillin',
        category: 'drug',
        reactions: ['rash', 'hives'],
        severity: 'moderate',
        assertion_status: 'present',
        confidence: 0.9,
        start_char: 400,
        end_char: 410,
      },
    ],
    nkda_detected: false,
    count: 1,
    processing_time_ms: 0.8,
  },
  vitals: {
    vitals: [
      {
        type: 'blood_pressure',
        value: 158,
        unit: 'mmHg',
        interpretation: 'high',
        trend: null,
        confidence: 0.95,
        start_char: 500,
        end_char: 516,
      },
    ],
    count: 1,
    processing_time_ms: 0.4,
  },
  temporal: {
    expressions: [],
    count: 0,
    processing_time_ms: 0.2,
  },
  sdoh: {
    findings: [
      {
        domain: 'substance_use',
        trigger_text: 'Former smoker',
        sentiment: 'protective',
        z_code: 'Z87.891',
        confidence: 0.85,
        start_char: 600,
        end_char: 613,
      },
    ],
    domains_detected: ['substance_use'],
    adverse_count: 0,
    protective_count: 1,
    processing_time_ms: 0.6,
  },
  comorbidity: {
    score: 4,
    age_adjusted_score: 6,
    risk_group: 'moderate',
    ten_year_mortality: 0.52,
    categories: [
      { name: 'Diabetes', weight: 1, detected: true, source: 'text_extraction' },
      { name: 'COPD', weight: 1, detected: true, source: 'text_extraction' },
      { name: 'Myocardial Infarction', weight: 1, detected: true, source: 'text_extraction' },
    ],
    processing_time_ms: 0.9,
  },
  relations: {
    relations: [
      {
        subject: 'Metformin',
        object: 'diabetes',
        relation_type: 'treats',
        confidence: 0.82,
        evidence: 'Metformin 1000mg PO BID for diabetes',
      },
    ],
    count: 1,
    processing_time_ms: 0.7,
  },
  component_errors: {},
  processing_time_ms: 8.5,
};

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <PipelineExplorer />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

describe('PipelineExplorer', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  // ── Page Structure ───────────────────────────────────────

  it('renders the page heading', () => {
    renderPage();
    expect(screen.getByText(/Pipeline Explorer/)).toBeInTheDocument();
  });

  it('renders the description', () => {
    renderPage();
    expect(screen.getByText(/14-module clinical NLP pipeline/)).toBeInTheDocument();
  });

  // ── Phase Headings ───────────────────────────────────────

  it('renders Phase 1 heading', () => {
    renderPage();
    expect(screen.getByText(/Phase 1 — Pre-processing/)).toBeInTheDocument();
  });

  it('renders Phase 2 heading', () => {
    renderPage();
    expect(screen.getByText(/Phase 2 — Extraction & Scoring/)).toBeInTheDocument();
  });

  // ── Module Toggles ───────────────────────────────────────

  it('renders all 14 module labels', () => {
    renderPage();
    expect(screen.getByText('Document Classification')).toBeInTheDocument();
    expect(screen.getByText('Section Parsing')).toBeInTheDocument();
    expect(screen.getByText('Quality Analysis')).toBeInTheDocument();
    expect(screen.getByText('De-identification')).toBeInTheDocument();
    expect(screen.getByText('Abbreviation Expansion')).toBeInTheDocument();
    expect(screen.getByText('Medication Extraction')).toBeInTheDocument();
    expect(screen.getByText('Allergy Extraction')).toBeInTheDocument();
    expect(screen.getByText('Vital Signs')).toBeInTheDocument();
    expect(screen.getByText('Temporal Extraction')).toBeInTheDocument();
    expect(screen.getByText('Assertion Detection')).toBeInTheDocument();
    expect(screen.getByText('Concept Normalization')).toBeInTheDocument();
    expect(screen.getByText('SDoH Extraction')).toBeInTheDocument();
    expect(screen.getByText('Relation Extraction')).toBeInTheDocument();
    expect(screen.getByText('Comorbidity Scoring')).toBeInTheDocument();
  });

  it('shows module count (13/14 default — deidentification off)', () => {
    renderPage();
    expect(screen.getByText('13/14 modules enabled')).toBeInTheDocument();
  });

  // ── Quick Actions ────────────────────────────────────────

  it('renders Enable all / Disable all / Reset defaults links', () => {
    renderPage();
    expect(screen.getByText('Enable all')).toBeInTheDocument();
    expect(screen.getByText('Disable all')).toBeInTheDocument();
    expect(screen.getByText('Reset defaults')).toBeInTheDocument();
  });

  it('Enable all sets count to 14/14', () => {
    renderPage();
    fireEvent.click(screen.getByText('Enable all'));
    expect(screen.getByText('14/14 modules enabled')).toBeInTheDocument();
  });

  it('Disable all sets count to 0/14', () => {
    renderPage();
    fireEvent.click(screen.getByText('Disable all'));
    expect(screen.getByText('0/14 modules enabled')).toBeInTheDocument();
  });

  it('Reset defaults restores to 13/14', () => {
    renderPage();
    fireEvent.click(screen.getByText('Disable all'));
    fireEvent.click(screen.getByText('Reset defaults'));
    expect(screen.getByText('13/14 modules enabled')).toBeInTheDocument();
  });

  // ── Input ────────────────────────────────────────────────

  it('renders the textarea', () => {
    renderPage();
    expect(
      screen.getByPlaceholderText(/Paste a clinical document/)
    ).toBeInTheDocument();
  });

  it('loads sample note', () => {
    renderPage();
    fireEvent.click(screen.getByText('Load sample discharge summary'));
    const textarea = screen.getByPlaceholderText(/Paste a clinical document/);
    expect((textarea as HTMLTextAreaElement).value).toContain('DISCHARGE SUMMARY');
  });

  it('shows character and word count after text input', () => {
    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Paste a clinical document/), {
      target: { value: 'Patient presents with chest pain' },
    });
    expect(screen.getByText(/32 chars/)).toBeInTheDocument();
    expect(screen.getByText(/5 words/)).toBeInTheDocument();
  });

  // ── Analyze Button ───────────────────────────────────────

  it('disables analyze button when no modules enabled', () => {
    renderPage();
    fireEvent.click(screen.getByText('Disable all'));
    fireEvent.change(screen.getByPlaceholderText(/Paste a clinical document/), {
      target: { value: 'test text' },
    });
    const button = screen.getByText(/Analyze \(0 modules\)/);
    expect(button).toBeDisabled();
  });

  // ── API Results ──────────────────────────────────────────

  it('renders analysis results after API call', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Paste a clinical document/), {
      target: { value: 'Discharge summary text...' },
    });
    fireEvent.click(screen.getByText(/Analyze/));

    await waitFor(() => {
      expect(screen.getByText(/Analysis complete/)).toBeInTheDocument();
      expect(screen.getByText(/8\.5ms/)).toBeInTheDocument();
    });
  });

  it('renders classification result section', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Paste a clinical document/), {
      target: { value: 'Patient presents with chest pain and SOB' },
    });
    fireEvent.click(screen.getByText(/Analyze/));

    await waitFor(() => {
      // The predicted type appears both as main label and in score list
      expect(screen.getAllByText('discharge summary').length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText(/92% confidence/)).toBeInTheDocument();
    });
  });

  it('renders quality grade and score', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Paste a clinical document/), {
      target: { value: 'Patient presents with chest pain and SOB' },
    });
    fireEvent.click(screen.getByText(/Analyze/));

    await waitFor(() => {
      // Quality score is unique to results section
      expect(screen.getByText(/Overall Score: 78\/100/)).toBeInTheDocument();
    });
  });

  it('renders medication results', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Paste a clinical document/), {
      target: { value: 'Patient presents with chest pain and SOB' },
    });
    fireEvent.click(screen.getByText(/Analyze/));

    await waitFor(() => {
      // Drug name may appear in both medications and relations sections
      expect(screen.getAllByText('Metformin').length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText('1000mg')).toBeInTheDocument();
    });
  });

  it('renders comorbidity score and risk group', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Paste a clinical document/), {
      target: { value: 'Patient presents with chest pain and SOB' },
    });
    fireEvent.click(screen.getByText(/Analyze/));

    await waitFor(() => {
      expect(screen.getByText('CCI Score')).toBeInTheDocument();
      expect(screen.getByText('52.0%')).toBeInTheDocument();
    });
  });

  it('renders relation extraction results', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockResponse,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Paste a clinical document/), {
      target: { value: 'Patient presents with chest pain and SOB' },
    });
    fireEvent.click(screen.getByText(/Analyze/));

    await waitFor(() => {
      expect(screen.getByText(/treats/)).toBeInTheDocument();
    });
  });

  it('shows error on API failure', async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 422,
    });

    renderPage();
    fireEvent.change(screen.getByPlaceholderText(/Paste a clinical document/), {
      target: { value: 'test' },
    });
    fireEvent.click(screen.getByText(/Analyze/));

    await waitFor(() => {
      expect(screen.getByText(/API error: 422/)).toBeInTheDocument();
    });
  });
});

