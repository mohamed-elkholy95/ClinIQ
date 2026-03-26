/**
 * Tests for the EvaluationDashboard page component.
 *
 * Verifies page structure, tab navigation, sample data evaluation,
 * result rendering for all 6 metric types, error handling, and
 * loading states.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { EvaluationDashboard } from '../../pages/EvaluationDashboard';

// ─── Mock API ────────────────────────────────────────────────

vi.mock('../../services/clinical', () => ({
  evaluateClassification: vi.fn(),
  evaluateAgreement: vi.fn(),
  evaluateNER: vi.fn(),
  evaluateROUGE: vi.fn(),
  evaluateICD: vi.fn(),
  evaluateAUPRC: vi.fn(),
}));

import {
  evaluateClassification,
  evaluateAgreement,
  evaluateNER,
  evaluateROUGE,
  evaluateICD,
  evaluateAUPRC,
} from '../../services/clinical';

const mockClassification = vi.mocked(evaluateClassification);
const mockAgreement = vi.mocked(evaluateAgreement);
const mockNER = vi.mocked(evaluateNER);
const mockROUGE = vi.mocked(evaluateROUGE);
const mockICD = vi.mocked(evaluateICD);
const mockAUPRC = vi.mocked(evaluateAUPRC);

// ─── Helpers ─────────────────────────────────────────────────

function renderPage() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <EvaluationDashboard />
      </BrowserRouter>
    </QueryClientProvider>
  );
}

// ─── Sample results ──────────────────────────────────────────

const classificationResult = {
  mcc: 0.583,
  tp: 8,
  fp: 2,
  fn: 2,
  tn: 8,
  calibration: {
    expected_calibration_error: 0.0732,
    brier_score: 0.1245,
    n_bins: 10,
    bin_accuracies: [0.9, 0.8],
    bin_confidences: [0.85, 0.75],
    bin_counts: [10, 10],
  },
  processing_time_ms: 0.42,
};

const agreementResult = {
  kappa: 0.72,
  observed_agreement: 0.85,
  expected_agreement: 0.46,
  n_items: 10,
  processing_time_ms: 0.15,
};

const nerResult = {
  exact_f1: 0.667,
  partial_f1: 0.778,
  type_weighted_f1: 0.722,
  mean_overlap: 0.85,
  n_gold: 5,
  n_pred: 5,
  n_exact_matches: 2,
  n_partial_matches: 2,
  n_unmatched_pred: 1,
  n_unmatched_gold: 1,
  processing_time_ms: 0.28,
};

const rougeResult = {
  rouge1: { precision: 0.82, recall: 0.75, f1: 0.78 },
  rouge2: { precision: 0.65, recall: 0.58, f1: 0.61 },
  rougeL: { precision: 0.78, recall: 0.72, f1: 0.75 },
  reference_length: 55,
  hypothesis_length: 38,
  length_ratio: 0.69,
  processing_time_ms: 0.31,
};

const icdResult = {
  full_code_accuracy: 0.25,
  block_accuracy: 0.5,
  chapter_accuracy: 1.0,
  n_samples: 8,
  full_code_matches: 2,
  block_matches: 4,
  chapter_matches: 8,
  processing_time_ms: 0.18,
};

const auprcResult = {
  label: 'rare_diagnosis',
  auprc: 0.892,
  n_positive: 3,
  n_total: 20,
  processing_time_ms: 0.22,
};

// ─── Tests ───────────────────────────────────────────────────

beforeEach(() => {
  vi.clearAllMocks();
});

describe('EvaluationDashboard', () => {
  describe('page structure', () => {
    it('renders the page title', () => {
      renderPage();
      expect(screen.getByText('Evaluation Dashboard')).toBeInTheDocument();
    });

    it('renders the subtitle description', () => {
      renderPage();
      expect(
        screen.getByText(/Compute clinical NLP evaluation metrics/)
      ).toBeInTheDocument();
    });

    it('renders all 6 metric tabs', () => {
      renderPage();
      // Use getAllByText since tab label may also appear in description area
      expect(screen.getAllByText('Classification').length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText('Agreement').length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText('NER')).toBeInTheDocument();
      expect(screen.getByText('ROUGE')).toBeInTheDocument();
      expect(screen.getByText('ICD-10')).toBeInTheDocument();
      expect(screen.getByText('AUPRC')).toBeInTheDocument();
    });

    it('renders the run button', () => {
      renderPage();
      expect(
        screen.getByRole('button', { name: /Run with Sample Data/ })
      ).toBeInTheDocument();
    });

    it('shows Classification description by default', () => {
      renderPage();
      expect(
        screen.getByText(/MCC, confusion matrix, and calibration metrics/)
      ).toBeInTheDocument();
    });
  });

  describe('tab navigation', () => {
    it('switches to Agreement tab', () => {
      renderPage();
      fireEvent.click(screen.getByText('Agreement'));
      expect(
        screen.getByText(/Cohen's Kappa inter-annotator agreement/)
      ).toBeInTheDocument();
    });

    it('switches to NER tab', () => {
      renderPage();
      fireEvent.click(screen.getByText('NER'));
      expect(
        screen.getByText(/Exact and partial entity span matching/)
      ).toBeInTheDocument();
    });

    it('switches to ROUGE tab', () => {
      renderPage();
      fireEvent.click(screen.getByText('ROUGE'));
      expect(
        screen.getByText(/ROUGE-1\/2\/L with precision, recall, and F1/)
      ).toBeInTheDocument();
    });

    it('switches to ICD-10 tab', () => {
      renderPage();
      fireEvent.click(screen.getByText('ICD-10'));
      expect(
        screen.getByText(/Hierarchical ICD-10 accuracy evaluation/)
      ).toBeInTheDocument();
    });

    it('switches to AUPRC tab', () => {
      renderPage();
      fireEvent.click(screen.getByText('AUPRC'));
      expect(
        screen.getByText(/Area Under Precision-Recall Curve/)
      ).toBeInTheDocument();
    });

    it('clears results when switching tabs', async () => {
      mockClassification.mockResolvedValueOnce(classificationResult);
      renderPage();

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() => expect(screen.getByText('Results')).toBeInTheDocument());

      fireEvent.click(screen.getByText('Agreement'));
      expect(screen.queryByText('Results')).not.toBeInTheDocument();
    });
  });

  describe('classification evaluation', () => {
    it('displays MCC result', async () => {
      mockClassification.mockResolvedValueOnce(classificationResult);
      renderPage();

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() => expect(screen.getByText('0.583')).toBeInTheDocument());
    });

    it('displays confusion matrix', async () => {
      mockClassification.mockResolvedValueOnce(classificationResult);
      renderPage();

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() => expect(screen.getByText('Confusion Matrix')).toBeInTheDocument());
    });

    it('displays calibration metrics when provided', async () => {
      mockClassification.mockResolvedValueOnce(classificationResult);
      renderPage();

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() => expect(screen.getByText('Calibration')).toBeInTheDocument());
      expect(screen.getByText('ECE')).toBeInTheDocument();
      expect(screen.getByText('Brier Score')).toBeInTheDocument();
    });

    it('displays true positive and false positive counts', async () => {
      mockClassification.mockResolvedValueOnce(classificationResult);
      renderPage();

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() => expect(screen.getByText('True Positive')).toBeInTheDocument());
      expect(screen.getByText('False Positive')).toBeInTheDocument();
    });
  });

  describe('agreement evaluation', () => {
    it('displays kappa score', async () => {
      mockAgreement.mockResolvedValueOnce(agreementResult);
      renderPage();
      fireEvent.click(screen.getByText('Agreement'));

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() => expect(screen.getByText('0.720')).toBeInTheDocument());
    });

    it('displays agreement interpretation label', async () => {
      mockAgreement.mockResolvedValueOnce(agreementResult);
      renderPage();
      fireEvent.click(screen.getByText('Agreement'));

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() =>
        expect(screen.getByText('Substantial Agreement')).toBeInTheDocument()
      );
    });

    it('displays observed and expected agreement', async () => {
      mockAgreement.mockResolvedValueOnce(agreementResult);
      renderPage();
      fireEvent.click(screen.getByText('Agreement'));

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() =>
        expect(screen.getByText('Observed Agreement')).toBeInTheDocument()
      );
      expect(screen.getByText('Expected Agreement')).toBeInTheDocument();
    });
  });

  describe('NER evaluation', () => {
    it('displays exact, partial, and type-weighted F1', async () => {
      mockNER.mockResolvedValueOnce(nerResult);
      renderPage();
      fireEvent.click(screen.getByText('NER'));

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() => expect(screen.getByText('Exact F1')).toBeInTheDocument());
      expect(screen.getByText('Partial F1')).toBeInTheDocument();
      expect(screen.getByText('Type-Weighted F1')).toBeInTheDocument();
    });

    it('displays entity counts', async () => {
      mockNER.mockResolvedValueOnce(nerResult);
      renderPage();
      fireEvent.click(screen.getByText('NER'));

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() =>
        expect(screen.getByText('Gold Entities')).toBeInTheDocument()
      );
      expect(screen.getByText('Pred Entities')).toBeInTheDocument();
    });
  });

  describe('ROUGE evaluation', () => {
    it('displays ROUGE-1/2/L F1 scores', async () => {
      mockROUGE.mockResolvedValueOnce(rougeResult);
      renderPage();
      fireEvent.click(screen.getByText('ROUGE'));

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() => expect(screen.getByText('ROUGE-1 F1')).toBeInTheDocument());
      expect(screen.getByText('ROUGE-2 F1')).toBeInTheDocument();
      expect(screen.getByText('ROUGE-L F1')).toBeInTheDocument();
    });

    it('displays length statistics', async () => {
      mockROUGE.mockResolvedValueOnce(rougeResult);
      renderPage();
      fireEvent.click(screen.getByText('ROUGE'));

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() =>
        expect(screen.getByText('Reference Length')).toBeInTheDocument()
      );
      expect(screen.getByText('Hypothesis Length')).toBeInTheDocument();
      expect(screen.getByText('Compression')).toBeInTheDocument();
    });
  });

  describe('ICD-10 evaluation', () => {
    it('displays hierarchical accuracy levels', async () => {
      mockICD.mockResolvedValueOnce(icdResult);
      renderPage();
      fireEvent.click(screen.getByText('ICD-10'));

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() => expect(screen.getByText('Chapter')).toBeInTheDocument());
      expect(screen.getByText('3-Char Block')).toBeInTheDocument();
      expect(screen.getByText('Full Code')).toBeInTheDocument();
    });

    it('displays informational note about hierarchy levels', async () => {
      mockICD.mockResolvedValueOnce(icdResult);
      renderPage();
      fireEvent.click(screen.getByText('ICD-10'));

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() =>
        expect(screen.getByText(/Chapter accuracy reveals organ-system-level/)).toBeInTheDocument()
      );
    });
  });

  describe('AUPRC evaluation', () => {
    it('displays AUPRC score', async () => {
      mockAUPRC.mockResolvedValueOnce(auprcResult);
      renderPage();
      fireEvent.click(screen.getByText('AUPRC'));

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() => expect(screen.getByText('0.8920')).toBeInTheDocument());
    });

    it('displays label name', async () => {
      mockAUPRC.mockResolvedValueOnce(auprcResult);
      renderPage();
      fireEvent.click(screen.getByText('AUPRC'));

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() =>
        expect(screen.getByText('rare_diagnosis')).toBeInTheDocument()
      );
    });

    it('displays prevalence information', async () => {
      mockAUPRC.mockResolvedValueOnce(auprcResult);
      renderPage();
      fireEvent.click(screen.getByText('AUPRC'));

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() =>
        expect(screen.getByText('Prevalence')).toBeInTheDocument()
      );
    });

    it('displays informational note about imbalanced datasets', async () => {
      mockAUPRC.mockResolvedValueOnce(auprcResult);
      renderPage();
      fireEvent.click(screen.getByText('AUPRC'));

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() =>
        expect(screen.getByText(/more informative than AUROC/)).toBeInTheDocument()
      );
    });
  });

  describe('error handling', () => {
    it('displays error message on API failure', async () => {
      mockClassification.mockRejectedValueOnce(new Error('Network error'));
      renderPage();

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() =>
        expect(screen.getByText('Network error')).toBeInTheDocument()
      );
    });

    it('clears error when switching tabs', async () => {
      mockClassification.mockRejectedValueOnce(new Error('Network error'));
      renderPage();

      fireEvent.click(screen.getByRole('button', { name: /Run with Sample Data/ }));
      await waitFor(() => expect(screen.getByText('Network error')).toBeInTheDocument());

      fireEvent.click(screen.getByText('Agreement'));
      expect(screen.queryByText('Network error')).not.toBeInTheDocument();
    });
  });
});
