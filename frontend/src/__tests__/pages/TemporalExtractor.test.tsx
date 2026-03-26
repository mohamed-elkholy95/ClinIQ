/**
 * Tests for the TemporalExtractor page.
 *
 * Covers page structure, sample note loading, word count display,
 * API integration (success, filtering, empty, error), and summary stats.
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { TemporalExtractor } from '../../pages/TemporalExtractor';

// ─── Mocks ───────────────────────────────────────────────────

const mockExtractTemporal = vi.fn();

vi.mock('../../services/clinical', () => ({
  extractTemporal: (...args: unknown[]) => mockExtractTemporal(...args),
}));

// ─── Helpers ─────────────────────────────────────────────────

const MOCK_RESPONSE = {
  expressions: [
    {
      type: 'date',
      text: 'March 15, 2025',
      normalized_value: '2025-03-15',
      confidence: 0.95,
      start_char: 32,
      end_char: 46,
    },
    {
      type: 'duration',
      text: '48 hours',
      normalized_value: 'PT48H',
      confidence: 0.88,
      start_char: 120,
      end_char: 128,
    },
    {
      type: 'relative_time',
      text: 'next Monday',
      normalized_value: null,
      confidence: 0.75,
      start_char: 200,
      end_char: 211,
    },
    {
      type: 'frequency',
      text: 'twice daily',
      normalized_value: 'BID',
      confidence: 0.92,
      start_char: 300,
      end_char: 311,
    },
    {
      type: 'postoperative_day',
      text: 'postoperative day 2',
      normalized_value: 'POD2',
      confidence: 0.9,
      start_char: 400,
      end_char: 419,
    },
  ],
  count: 5,
  processing_time_ms: 2.3,
};

// ─── Tests ───────────────────────────────────────────────────

describe('TemporalExtractor', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // Page structure
  describe('page structure', () => {
    it('renders the page title', () => {
      render(<TemporalExtractor />);
      expect(screen.getByText('Temporal Expression Extractor')).toBeInTheDocument();
    });

    it('renders the textarea with placeholder', () => {
      render(<TemporalExtractor />);
      expect(
        screen.getByPlaceholderText(/Paste a clinical note to extract temporal/i)
      ).toBeInTheDocument();
    });

    it('renders the analyze button', () => {
      render(<TemporalExtractor />);
      expect(
        screen.getByRole('button', { name: /Extract Temporal Expressions/i })
      ).toBeInTheDocument();
    });

    it('renders all 3 sample note buttons', () => {
      render(<TemporalExtractor />);
      expect(screen.getByText('Discharge Summary')).toBeInTheDocument();
      expect(screen.getByText('Progress Note')).toBeInTheDocument();
      expect(screen.getByText('Dental Treatment Plan')).toBeInTheDocument();
    });

    it('disables analyze button when textarea is empty', () => {
      render(<TemporalExtractor />);
      const btn = screen.getByRole('button', { name: /Extract Temporal/i });
      expect(btn).toBeDisabled();
    });
  });

  // Sample loading
  describe('sample loading', () => {
    it('loads sample text when sample button is clicked', () => {
      render(<TemporalExtractor />);
      fireEvent.click(screen.getByText('Discharge Summary'));
      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i) as HTMLTextAreaElement;
      expect(textarea.value).toContain('DISCHARGE SUMMARY');
    });

    it('enables analyze button after loading sample', () => {
      render(<TemporalExtractor />);
      fireEvent.click(screen.getByText('Progress Note'));
      const btn = screen.getByRole('button', { name: /Extract Temporal/i });
      expect(btn).not.toBeDisabled();
    });
  });

  // Word count
  describe('word count', () => {
    it('displays word count for entered text', () => {
      render(<TemporalExtractor />);
      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i);
      fireEvent.change(textarea, { target: { value: 'hello world test' } });
      expect(screen.getByText('3 words')).toBeInTheDocument();
    });
  });

  // API integration
  describe('API integration', () => {
    it('calls extractTemporal on analyze', async () => {
      mockExtractTemporal.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<TemporalExtractor />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i);
      fireEvent.change(textarea, { target: { value: 'Patient admitted on March 15.' } });
      fireEvent.click(screen.getByRole('button', { name: /Extract Temporal/i }));

      await waitFor(() => {
        expect(mockExtractTemporal).toHaveBeenCalledWith('Patient admitted on March 15.');
      });
    });

    it('displays results after successful extraction', async () => {
      mockExtractTemporal.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<TemporalExtractor />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Extract Temporal/i }));

      await waitFor(() => {
        expect(screen.getByText('Total Expressions')).toBeInTheDocument();
      });
      expect(screen.getByText('Types Found')).toBeInTheDocument();
    });

    it('renders temporal expression text in results table', async () => {
      mockExtractTemporal.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<TemporalExtractor />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Extract Temporal/i }));

      await waitFor(() => {
        // Check for date expression
        expect(screen.getByText(/March 15, 2025/)).toBeInTheDocument();
        // Check for normalized value
        expect(screen.getByText('2025-03-15')).toBeInTheDocument();
      });
    });

    it('displays type filter buttons', async () => {
      mockExtractTemporal.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<TemporalExtractor />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Extract Temporal/i }));

      await waitFor(() => {
        // Filter buttons contain type label + count in parens
        const filterBtns = screen.getAllByRole('button').filter(
          (btn) => btn.textContent?.includes('(1)') || btn.textContent?.includes('All (5)')
        );
        expect(filterBtns.length).toBeGreaterThanOrEqual(2);
      });
    });

    it('filters results by type when filter button clicked', async () => {
      mockExtractTemporal.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<TemporalExtractor />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Extract Temporal/i }));

      await waitFor(() => {
        expect(screen.getByText(/March 15, 2025/)).toBeInTheDocument();
      });

      // Click the date filter button (contains "Date" and "(1)")
      const dateFilterBtn = screen.getAllByRole('button').find(
        (btn) => btn.textContent?.includes('Date') && btn.textContent?.includes('(1)')
      );
      expect(dateFilterBtn).toBeDefined();
      fireEvent.click(dateFilterBtn!);

      // Duration expression should not be visible now
      expect(screen.queryByText(/48 hours/)).not.toBeInTheDocument();
      // Date expression should still be visible
      expect(screen.getByText(/March 15, 2025/)).toBeInTheDocument();
    });

    it('displays error on API failure', async () => {
      mockExtractTemporal.mockRejectedValueOnce(new Error('Network error'));
      render(<TemporalExtractor />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Extract Temporal/i }));

      await waitFor(() => {
        expect(screen.getByText('Network error')).toBeInTheDocument();
      });
    });

    it('displays processing time', async () => {
      mockExtractTemporal.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<TemporalExtractor />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Extract Temporal/i }));

      await waitFor(() => {
        expect(screen.getByText('2.3ms')).toBeInTheDocument();
      });
    });
  });
});
