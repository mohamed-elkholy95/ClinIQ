/**
 * Tests for the DocumentClassifier page.
 *
 * Covers page structure, sample note loading, word count,
 * API integration (classification results, score bars, evidence, error).
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { DocumentClassifier } from '../../pages/DocumentClassifier';

// ─── Mocks ───────────────────────────────────────────────────

const mockClassifyDocument = vi.fn();

vi.mock('../../services/clinical', () => ({
  classifyDocument: (...args: unknown[]) => mockClassifyDocument(...args),
}));

// ─── Helpers ─────────────────────────────────────────────────

const MOCK_RESPONSE = {
  predicted_type: 'discharge_summary',
  confidence: 0.94,
  processing_time_ms: 3.2,
  scores: [
    {
      document_type: 'discharge_summary',
      score: 0.94,
      evidence: ['discharge', 'admission', 'hospital course', 'medications'],
    },
    {
      document_type: 'progress_note',
      score: 0.35,
      evidence: ['assessment', 'plan'],
    },
    {
      document_type: 'operative_note',
      score: 0.08,
      evidence: [],
    },
    {
      document_type: 'radiology_report',
      score: 0.03,
      evidence: [],
    },
    {
      document_type: 'history_physical',
      score: 0.22,
      evidence: ['history', 'medications'],
    },
  ],
};

// ─── Tests ───────────────────────────────────────────────────

describe('DocumentClassifier', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // Page structure
  describe('page structure', () => {
    it('renders the page title', () => {
      render(<DocumentClassifier />);
      expect(screen.getByText('Document Classifier')).toBeInTheDocument();
    });

    it('renders the textarea', () => {
      render(<DocumentClassifier />);
      expect(
        screen.getByPlaceholderText(/Paste a clinical document to classify/i)
      ).toBeInTheDocument();
    });

    it('renders the classify button', () => {
      render(<DocumentClassifier />);
      expect(
        screen.getByRole('button', { name: /Classify Document/i })
      ).toBeInTheDocument();
    });

    it('renders all 3 sample note buttons', () => {
      render(<DocumentClassifier />);
      expect(screen.getByText('Discharge Summary')).toBeInTheDocument();
      expect(screen.getByText('Operative Note')).toBeInTheDocument();
      expect(screen.getByText('Radiology Report')).toBeInTheDocument();
    });

    it('disables classify button when textarea is empty', () => {
      render(<DocumentClassifier />);
      const btn = screen.getByRole('button', { name: /Classify Document/i });
      expect(btn).toBeDisabled();
    });
  });

  // Sample loading
  describe('sample loading', () => {
    it('loads Discharge Summary sample text', () => {
      render(<DocumentClassifier />);
      fireEvent.click(screen.getByText('Discharge Summary'));
      const textarea = screen.getByPlaceholderText(/Paste a clinical document/i) as HTMLTextAreaElement;
      expect(textarea.value).toContain('DISCHARGE SUMMARY');
    });

    it('loads Operative Note sample text', () => {
      render(<DocumentClassifier />);
      fireEvent.click(screen.getByText('Operative Note'));
      const textarea = screen.getByPlaceholderText(/Paste a clinical document/i) as HTMLTextAreaElement;
      expect(textarea.value).toContain('OPERATIVE NOTE');
    });

    it('enables classify button after loading sample', () => {
      render(<DocumentClassifier />);
      fireEvent.click(screen.getByText('Radiology Report'));
      const btn = screen.getByRole('button', { name: /Classify Document/i });
      expect(btn).not.toBeDisabled();
    });
  });

  // Word count
  describe('word count', () => {
    it('displays word count', () => {
      render(<DocumentClassifier />);
      const textarea = screen.getByPlaceholderText(/Paste a clinical document/i);
      fireEvent.change(textarea, { target: { value: 'discharge summary text here' } });
      expect(screen.getByText('4 words')).toBeInTheDocument();
    });
  });

  // API integration
  describe('API integration', () => {
    it('calls classifyDocument on analyze', async () => {
      mockClassifyDocument.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<DocumentClassifier />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical document/i);
      fireEvent.change(textarea, { target: { value: 'DISCHARGE SUMMARY Patient was admitted...' } });
      fireEvent.click(screen.getByRole('button', { name: /Classify Document/i }));

      await waitFor(() => {
        expect(mockClassifyDocument).toHaveBeenCalledWith(
          'DISCHARGE SUMMARY Patient was admitted...'
        );
      });
    });

    it('displays predicted document type', async () => {
      mockClassifyDocument.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<DocumentClassifier />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical document/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Classify Document/i }));

      await waitFor(() => {
        // The heading renders the type with spaces and capitalize
        const heading = screen.getByRole('heading', { level: 2 });
        expect(heading.textContent).toContain('discharge summary');
      });
    });

    it('displays confidence percentage', async () => {
      mockClassifyDocument.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<DocumentClassifier />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical document/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Classify Document/i }));

      await waitFor(() => {
        expect(screen.getByText('94.0% confidence')).toBeInTheDocument();
      });
    });

    it('displays predicted type card after classification', async () => {
      mockClassifyDocument.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<DocumentClassifier />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical document/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Classify Document/i }));

      await waitFor(() => {
        expect(screen.getByText('94.0% confidence')).toBeInTheDocument();
        expect(screen.getByText(/3.2ms/)).toBeInTheDocument();
      });
    });

    it('renders classification score bars', async () => {
      mockClassifyDocument.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<DocumentClassifier />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical document/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Classify Document/i }));

      await waitFor(() => {
        expect(screen.getByText('Classification Scores')).toBeInTheDocument();
      });
    });

    it('renders evidence keywords for top predictions', async () => {
      mockClassifyDocument.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<DocumentClassifier />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical document/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Classify Document/i }));

      await waitFor(() => {
        // Evidence keywords for discharge_summary
        expect(screen.getByText('discharge')).toBeInTheDocument();
        expect(screen.getByText('admission')).toBeInTheDocument();
        expect(screen.getByText('hospital course')).toBeInTheDocument();
      });
    });

    it('displays processing time', async () => {
      mockClassifyDocument.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<DocumentClassifier />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical document/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Classify Document/i }));

      await waitFor(() => {
        expect(screen.getByText(/3.2ms/)).toBeInTheDocument();
      });
    });

    it('displays error on API failure', async () => {
      mockClassifyDocument.mockRejectedValueOnce(new Error('Service unavailable'));
      render(<DocumentClassifier />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical document/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Classify Document/i }));

      await waitFor(() => {
        expect(screen.getByText('Service unavailable')).toBeInTheDocument();
      });
    });
  });
});
