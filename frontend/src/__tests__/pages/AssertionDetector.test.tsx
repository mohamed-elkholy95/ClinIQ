/**
 * Tests for the AssertionDetector page.
 *
 * Covers page structure, sample note loading, entity count display,
 * API integration (batch entity analysis, status rendering, error).
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { AssertionDetector } from '../../pages/AssertionDetector';

// ─── Mocks ───────────────────────────────────────────────────

const mockDetectAssertion = vi.fn();

vi.mock('../../services/clinical', () => ({
  detectAssertion: (...args: unknown[]) => mockDetectAssertion(...args),
}));

// ─── Helpers ─────────────────────────────────────────────────

function mockAssertionResponse(status: string, trigger: string | null = null) {
  return {
    status,
    trigger_text: trigger,
    confidence: 0.9,
    scope_start: 0,
    scope_end: 50,
  };
}

// ─── Tests ───────────────────────────────────────────────────

describe('AssertionDetector', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // Page structure
  describe('page structure', () => {
    it('renders the page title', () => {
      render(<AssertionDetector />);
      expect(screen.getByText('Assertion Detector')).toBeInTheDocument();
    });

    it('renders the textarea', () => {
      render(<AssertionDetector />);
      expect(
        screen.getByPlaceholderText(/Paste a clinical note/i)
      ).toBeInTheDocument();
    });

    it('renders the analyze button', () => {
      render(<AssertionDetector />);
      expect(
        screen.getByRole('button', { name: /Detect Assertions/i })
      ).toBeInTheDocument();
    });

    it('renders all 3 sample note buttons', () => {
      render(<AssertionDetector />);
      expect(screen.getByText('H&P Note')).toBeInTheDocument();
      expect(screen.getByText('Discharge Summary')).toBeInTheDocument();
      expect(screen.getByText('Dental Progress Note')).toBeInTheDocument();
    });

    it('disables analyze button when no text or entities', () => {
      render(<AssertionDetector />);
      const btn = screen.getByRole('button', { name: /Detect Assertions/i });
      expect(btn).toBeDisabled();
    });

    it('renders assertion status legend', () => {
      render(<AssertionDetector />);
      expect(screen.getByText('Assertion Status Legend')).toBeInTheDocument();
      expect(screen.getByText('Present')).toBeInTheDocument();
      expect(screen.getByText('Absent (Negated)')).toBeInTheDocument();
      expect(screen.getByText('Possible')).toBeInTheDocument();
      expect(screen.getByText('Family History')).toBeInTheDocument();
    });
  });

  // Sample loading
  describe('sample loading', () => {
    it('loads H&P Note sample text', () => {
      render(<AssertionDetector />);
      fireEvent.click(screen.getByText('H&P Note'));
      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i) as HTMLTextAreaElement;
      expect(textarea.value).toContain('HISTORY OF PRESENT ILLNESS');
    });

    it('shows entity count after loading sample', () => {
      render(<AssertionDetector />);
      fireEvent.click(screen.getByText('H&P Note'));
      expect(screen.getByText(/11 entities will be analysed/i)).toBeInTheDocument();
    });

    it('enables analyze button after loading sample', () => {
      render(<AssertionDetector />);
      fireEvent.click(screen.getByText('Dental Progress Note'));
      const btn = screen.getByRole('button', { name: /Detect Assertions/i });
      expect(btn).not.toBeDisabled();
    });

    it('highlights active sample button', () => {
      render(<AssertionDetector />);
      const btn = screen.getByText('H&P Note');
      fireEvent.click(btn);
      expect(btn.className).toContain('bg-blue-600');
    });
  });

  // Word count
  describe('word count', () => {
    it('displays word count for entered text', () => {
      render(<AssertionDetector />);
      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i);
      fireEvent.change(textarea, { target: { value: 'Patient denies pain' } });
      expect(screen.getByText('3 words')).toBeInTheDocument();
    });
  });

  // API integration
  describe('API integration', () => {
    it('calls detectAssertion for each entity in sample', async () => {
      // Dental sample has 6 entities
      mockDetectAssertion.mockResolvedValue(mockAssertionResponse('present'));
      render(<AssertionDetector />);

      fireEvent.click(screen.getByText('Dental Progress Note'));
      fireEvent.click(screen.getByRole('button', { name: /Detect Assertions/i }));

      await waitFor(() => {
        expect(mockDetectAssertion).toHaveBeenCalledTimes(6);
      });
    });

    it('renders entity text in results table', async () => {
      mockDetectAssertion
        .mockResolvedValueOnce(mockAssertionResponse('absent', 'No'))
        .mockResolvedValueOnce(mockAssertionResponse('absent', 'denies'))
        .mockResolvedValueOnce(mockAssertionResponse('possible', 'Possible'))
        .mockResolvedValueOnce(mockAssertionResponse('family', 'Family history'))
        .mockResolvedValueOnce(mockAssertionResponse('conditional', 'If'))
        .mockResolvedValueOnce(mockAssertionResponse('absent', 'No'));

      render(<AssertionDetector />);
      fireEvent.click(screen.getByText('Dental Progress Note'));
      fireEvent.click(screen.getByRole('button', { name: /Detect Assertions/i }));

      await waitFor(() => {
        expect(screen.getByText('pain')).toBeInTheDocument();
        expect(screen.getByText('sensitivity')).toBeInTheDocument();
        expect(screen.getByText('crack')).toBeInTheDocument();
      });
    });

    it('renders status badges for detected assertions', async () => {
      mockDetectAssertion
        .mockResolvedValueOnce(mockAssertionResponse('absent', 'No'))
        .mockResolvedValueOnce(mockAssertionResponse('present'))
        .mockResolvedValueOnce(mockAssertionResponse('possible', 'Possible'))
        .mockResolvedValueOnce(mockAssertionResponse('family', 'Family history'))
        .mockResolvedValueOnce(mockAssertionResponse('hypothetical', 'If'))
        .mockResolvedValueOnce(mockAssertionResponse('absent', 'No'));

      render(<AssertionDetector />);
      fireEvent.click(screen.getByText('Dental Progress Note'));
      fireEvent.click(screen.getByRole('button', { name: /Detect Assertions/i }));

      await waitFor(() => {
        // Check for status badges in the results table
        const absentBadges = screen.getAllByText(/Absent \(Negated\)/i);
        expect(absentBadges.length).toBeGreaterThanOrEqual(1);
      });
    });

    it('renders trigger text when present', async () => {
      mockDetectAssertion.mockResolvedValue(mockAssertionResponse('absent', 'denies'));

      render(<AssertionDetector />);
      fireEvent.click(screen.getByText('Dental Progress Note'));
      fireEvent.click(screen.getByRole('button', { name: /Detect Assertions/i }));

      await waitFor(() => {
        const triggers = screen.getAllByText('denies');
        expect(triggers.length).toBeGreaterThanOrEqual(1);
      });
    });

    it('displays error on API failure', async () => {
      mockDetectAssertion.mockRejectedValueOnce(new Error('Server error'));

      render(<AssertionDetector />);
      fireEvent.click(screen.getByText('Dental Progress Note'));
      fireEvent.click(screen.getByRole('button', { name: /Detect Assertions/i }));

      await waitFor(() => {
        expect(screen.getByText('Server error')).toBeInTheDocument();
      });
    });

    it('displays results count header', async () => {
      mockDetectAssertion.mockResolvedValue(mockAssertionResponse('present'));

      render(<AssertionDetector />);
      fireEvent.click(screen.getByText('Dental Progress Note'));
      fireEvent.click(screen.getByRole('button', { name: /Detect Assertions/i }));

      await waitFor(() => {
        expect(screen.getByText('Assertion Results (6 entities)')).toBeInTheDocument();
      });
    });
  });
});
