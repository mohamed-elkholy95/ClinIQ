/**
 * Tests for the RelationExplorer page.
 *
 * Covers page structure, sample note loading, word count,
 * API integration (success, filtering, evidence, error), and stats.
 */

import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { vi, describe, it, expect, beforeEach } from 'vitest';
import { RelationExplorer } from '../../pages/RelationExplorer';

// ─── Mocks ───────────────────────────────────────────────────

const mockExtractRelations = vi.fn();

vi.mock('../../services/clinical', () => ({
  extractRelations: (...args: unknown[]) => mockExtractRelations(...args),
}));

// ─── Helpers ─────────────────────────────────────────────────

const MOCK_RESPONSE = {
  relations: [
    {
      subject: 'lisinopril',
      object: 'hypertension',
      relation_type: 'treats',
      confidence: 0.92,
      evidence: 'Hypertension — treated with lisinopril 20 mg daily.',
    },
    {
      subject: 'amlodipine',
      object: 'ankle edema',
      relation_type: 'causes',
      confidence: 0.85,
      evidence: 'Amlodipine 5 mg causes ankle edema.',
    },
    {
      subject: 'atorvastatin',
      object: 'cardiovascular events',
      relation_type: 'prevents',
      confidence: 0.88,
      evidence: 'atorvastatin 40 mg prevents cardiovascular events.',
    },
    {
      subject: 'NSAIDs',
      object: 'kidney function',
      relation_type: 'worsens',
      confidence: 0.9,
      evidence: 'NSAIDs worsen kidney function.',
    },
    {
      subject: 'HbA1c',
      object: 'glycemic control',
      relation_type: 'monitors',
      confidence: 0.87,
      evidence: 'HbA1c monitors glycemic control.',
    },
  ],
  count: 5,
  processing_time_ms: 4.1,
};

// ─── Tests ───────────────────────────────────────────────────

describe('RelationExplorer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // Page structure
  describe('page structure', () => {
    it('renders the page title', () => {
      render(<RelationExplorer />);
      expect(screen.getByText('Relation Explorer')).toBeInTheDocument();
    });

    it('renders the textarea', () => {
      render(<RelationExplorer />);
      expect(
        screen.getByPlaceholderText(/Paste a clinical note to extract entity relations/i)
      ).toBeInTheDocument();
    });

    it('renders the analyze button', () => {
      render(<RelationExplorer />);
      expect(
        screen.getByRole('button', { name: /Extract Relations/i })
      ).toBeInTheDocument();
    });

    it('renders all 3 sample note buttons', () => {
      render(<RelationExplorer />);
      expect(screen.getByText('Treatment Plan')).toBeInTheDocument();
      expect(screen.getByText('Admission Note')).toBeInTheDocument();
      expect(screen.getByText('Dental Note')).toBeInTheDocument();
    });

    it('disables analyze button when textarea is empty', () => {
      render(<RelationExplorer />);
      const btn = screen.getByRole('button', { name: /Extract Relations/i });
      expect(btn).toBeDisabled();
    });
  });

  // Sample loading
  describe('sample loading', () => {
    it('loads Treatment Plan sample text', () => {
      render(<RelationExplorer />);
      fireEvent.click(screen.getByText('Treatment Plan'));
      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i) as HTMLTextAreaElement;
      expect(textarea.value).toContain('ASSESSMENT AND PLAN');
    });

    it('enables analyze button after loading sample', () => {
      render(<RelationExplorer />);
      fireEvent.click(screen.getByText('Admission Note'));
      const btn = screen.getByRole('button', { name: /Extract Relations/i });
      expect(btn).not.toBeDisabled();
    });
  });

  // Word count
  describe('word count', () => {
    it('displays word count', () => {
      render(<RelationExplorer />);
      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i);
      fireEvent.change(textarea, { target: { value: 'metformin treats diabetes' } });
      expect(screen.getByText('3 words')).toBeInTheDocument();
    });
  });

  // API integration
  describe('API integration', () => {
    it('calls extractRelations on analyze', async () => {
      mockExtractRelations.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<RelationExplorer />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i);
      fireEvent.change(textarea, { target: { value: 'test text' } });
      fireEvent.click(screen.getByRole('button', { name: /Extract Relations/i }));

      await waitFor(() => {
        expect(mockExtractRelations).toHaveBeenCalledWith('test text');
      });
    });

    it('displays summary stats after extraction', async () => {
      mockExtractRelations.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<RelationExplorer />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Extract Relations/i }));

      await waitFor(() => {
        expect(screen.getByText('Total Relations')).toBeInTheDocument();
        expect(screen.getByText('Relation Types')).toBeInTheDocument();
      });
    });

    it('renders relation cards with subject and object', async () => {
      mockExtractRelations.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<RelationExplorer />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Extract Relations/i }));

      await waitFor(() => {
        expect(screen.getByText('lisinopril')).toBeInTheDocument();
        expect(screen.getByText('hypertension')).toBeInTheDocument();
        expect(screen.getByText('amlodipine')).toBeInTheDocument();
        expect(screen.getByText('ankle edema')).toBeInTheDocument();
      });
    });

    it('renders relation type badges', async () => {
      mockExtractRelations.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<RelationExplorer />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Extract Relations/i }));

      await waitFor(() => {
        expect(screen.getByText('→ treats →')).toBeInTheDocument();
        expect(screen.getByText('→ causes →')).toBeInTheDocument();
        expect(screen.getByText('→ prevents →')).toBeInTheDocument();
      });
    });

    it('renders evidence text', async () => {
      mockExtractRelations.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<RelationExplorer />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Extract Relations/i }));

      await waitFor(() => {
        expect(
          screen.getByText(/Hypertension — treated with lisinopril/i)
        ).toBeInTheDocument();
      });
    });

    it('filters relations by type', async () => {
      mockExtractRelations.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<RelationExplorer />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Extract Relations/i }));

      await waitFor(() => {
        expect(screen.getByText(/Treats/)).toBeInTheDocument();
      });

      // Click "Treats" filter
      fireEvent.click(screen.getByText(/Treats \(1\)/));

      // Only treats relation should be visible
      expect(screen.getByText('lisinopril')).toBeInTheDocument();
      expect(screen.queryByText('amlodipine')).not.toBeInTheDocument();
    });

    it('displays error on API failure', async () => {
      mockExtractRelations.mockRejectedValueOnce(new Error('Connection refused'));
      render(<RelationExplorer />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Extract Relations/i }));

      await waitFor(() => {
        expect(screen.getByText('Connection refused')).toBeInTheDocument();
      });
    });

    it('displays processing time', async () => {
      mockExtractRelations.mockResolvedValueOnce(MOCK_RESPONSE);
      render(<RelationExplorer />);

      const textarea = screen.getByPlaceholderText(/Paste a clinical note/i);
      fireEvent.change(textarea, { target: { value: 'test' } });
      fireEvent.click(screen.getByRole('button', { name: /Extract Relations/i }));

      await waitFor(() => {
        expect(screen.getByText('4.1ms')).toBeInTheDocument();
      });
    });
  });
});
