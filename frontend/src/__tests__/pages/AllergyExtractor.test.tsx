/**
 * Tests for the AllergyExtractor page component.
 *
 * Validates page structure, sample loading, category filters, API integration,
 * severity badges, assertion status display, NKDA detection, and error handling.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { AllergyExtractor } from '../../pages/AllergyExtractor';

// ─── Mocks ───────────────────────────────────────────────────

const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

beforeEach(() => {
  vi.clearAllMocks();
});

// ─── Page Structure ──────────────────────────────────────────

describe('AllergyExtractor — page structure', () => {
  it('renders the page heading', () => {
    render(<AllergyExtractor />);
    expect(screen.getByText('Allergy Extraction')).toBeTruthy();
  });

  it('renders the description text', () => {
    render(<AllergyExtractor />);
    expect(
      screen.getByText(/drug, food, and environmental allergies/i)
    ).toBeTruthy();
  });

  it('renders three sample note buttons', () => {
    render(<AllergyExtractor />);
    expect(screen.getByText('Allergy List')).toBeTruthy();
    expect(screen.getByText('H&P Note')).toBeTruthy();
    expect(screen.getByText('Dental Pre-Op')).toBeTruthy();
  });

  it('renders the textarea placeholder', () => {
    render(<AllergyExtractor />);
    expect(screen.getByPlaceholderText(/paste clinical note/i)).toBeTruthy();
  });

  it('renders Extract Allergies button disabled initially', () => {
    render(<AllergyExtractor />);
    const btn = screen.getByText('Extract Allergies');
    expect(btn).toBeTruthy();
    expect((btn as HTMLButtonElement).disabled).toBe(true);
  });

  it('renders confidence slider', () => {
    render(<AllergyExtractor />);
    expect(screen.getByText(/min confidence/i)).toBeTruthy();
  });
});

// ─── Sample Loading ──────────────────────────────────────────

describe('AllergyExtractor — sample loading', () => {
  it('loads first sample into textarea', () => {
    render(<AllergyExtractor />);
    fireEvent.click(screen.getByText('Allergy List'));
    const textarea = screen.getByPlaceholderText(/paste clinical note/i) as HTMLTextAreaElement;
    expect(textarea.value).toContain('Penicillin');
    expect(textarea.value).toContain('anaphylaxis');
  });

  it('loads second sample into textarea', () => {
    render(<AllergyExtractor />);
    fireEvent.click(screen.getByText('H&P Note'));
    const textarea = screen.getByPlaceholderText(/paste clinical note/i) as HTMLTextAreaElement;
    expect(textarea.value).toContain('amoxicillin');
  });

  it('loads third sample into textarea', () => {
    render(<AllergyExtractor />);
    fireEvent.click(screen.getByText('Dental Pre-Op'));
    const textarea = screen.getByPlaceholderText(/paste clinical note/i) as HTMLTextAreaElement;
    expect(textarea.value).toContain('Lidocaine');
  });

  it('enables button after loading sample', () => {
    render(<AllergyExtractor />);
    fireEvent.click(screen.getByText('Allergy List'));
    const btn = screen.getByText('Extract Allergies');
    expect((btn as HTMLButtonElement).disabled).toBe(false);
  });
});

// ─── Word Count ──────────────────────────────────────────────

describe('AllergyExtractor — word count', () => {
  it('shows word count for loaded sample', () => {
    render(<AllergyExtractor />);
    fireEvent.click(screen.getByText('Allergy List'));
    // Should show some word count > 0
    const wordCounts = screen.getAllByText(/\d+ words/);
    expect(wordCounts.length).toBeGreaterThan(0);
  });
});

// ─── API Integration ─────────────────────────────────────────

describe('AllergyExtractor — API integration', () => {
  const mockResponse = {
    allergies: [
      {
        allergen: 'penicillin',
        canonical_name: 'penicillin',
        category: 'drug',
        reactions: ['anaphylaxis'],
        severity: 'life_threatening',
        assertion_status: 'present',
        confidence: 0.95,
        start_char: 0,
        end_char: 10,
      },
      {
        allergen: 'shellfish',
        canonical_name: 'shellfish',
        category: 'food',
        reactions: ['urticaria'],
        severity: 'moderate',
        assertion_status: 'present',
        confidence: 0.80,
        start_char: 50,
        end_char: 59,
      },
      {
        allergen: 'latex',
        canonical_name: 'latex',
        category: 'environmental',
        reactions: ['contact dermatitis'],
        severity: 'mild',
        assertion_status: 'historical',
        confidence: 0.70,
        start_char: 100,
        end_char: 105,
      },
    ],
    nkda_detected: false,
    count: 3,
    processing_time_ms: 2.1,
  };

  it('calls API and renders results', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<AllergyExtractor />);
    fireEvent.click(screen.getByText('Allergy List'));
    fireEvent.click(screen.getByText('Extract Allergies'));

    await waitFor(() => {
      expect(screen.getByText('penicillin')).toBeTruthy();
      expect(screen.getByText('shellfish')).toBeTruthy();
      expect(screen.getByText('latex')).toBeTruthy();
    });
  });

  it('renders summary cards with counts', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<AllergyExtractor />);
    fireEvent.click(screen.getByText('Allergy List'));
    fireEvent.click(screen.getByText('Extract Allergies'));

    await waitFor(() => {
      // Total: 3
      expect(screen.getByText('3')).toBeTruthy();
      // Should show category counts
      expect(screen.getByText('💊 Drug')).toBeTruthy();
      expect(screen.getByText('🍽️ Food')).toBeTruthy();
      expect(screen.getByText('🌿 Environmental')).toBeTruthy();
    });
  });

  it('renders severity badges', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<AllergyExtractor />);
    fireEvent.click(screen.getByText('Allergy List'));
    fireEvent.click(screen.getByText('Extract Allergies'));

    await waitFor(() => {
      // Severity badges — text may be split by emoji prefix
      expect(screen.getAllByText(/life-threatening/i).length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText(/moderate/i).length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText(/mild/i).length).toBeGreaterThanOrEqual(1);
    });
  });

  it('renders assertion status labels', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<AllergyExtractor />);
    fireEvent.click(screen.getByText('Allergy List'));
    fireEvent.click(screen.getByText('Extract Allergies'));

    await waitFor(() => {
      expect(screen.getByText('Historical')).toBeTruthy();
      // 2 × Confirmed for the two 'present' allergies
      const confirmed = screen.getAllByText('Confirmed');
      expect(confirmed.length).toBe(2);
    });
  });

  it('renders NKDA indicator when detected', async () => {
    const nkdaResponse = { ...mockResponse, nkda_detected: true };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => nkdaResponse,
    });
    render(<AllergyExtractor />);
    fireEvent.click(screen.getByText('Allergy List'));
    fireEvent.click(screen.getByText('Extract Allergies'));

    await waitFor(() => {
      expect(screen.getByText(/NKDA/)).toBeTruthy();
    });
  });

  it('handles API errors', async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });
    render(<AllergyExtractor />);
    fireEvent.click(screen.getByText('Allergy List'));
    fireEvent.click(screen.getByText('Extract Allergies'));

    await waitFor(() => {
      expect(screen.getByText(/server error: 500/i)).toBeTruthy();
    });
  });

  it('shows processing time', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<AllergyExtractor />);
    fireEvent.click(screen.getByText('Allergy List'));
    fireEvent.click(screen.getByText('Extract Allergies'));

    await waitFor(() => {
      expect(screen.getByText(/2\.1ms/)).toBeTruthy();
    });
  });
});

// ─── Category Filter ─────────────────────────────────────────

describe('AllergyExtractor — category filter', () => {
  const mockResponse = {
    allergies: [
      {
        allergen: 'penicillin',
        canonical_name: 'penicillin',
        category: 'drug',
        reactions: ['rash'],
        severity: 'moderate',
        assertion_status: 'present',
        confidence: 0.90,
        start_char: 0,
        end_char: 10,
      },
      {
        allergen: 'peanuts',
        canonical_name: 'peanuts',
        category: 'food',
        reactions: ['anaphylaxis'],
        severity: 'life_threatening',
        assertion_status: 'present',
        confidence: 0.85,
        start_char: 50,
        end_char: 57,
      },
    ],
    nkda_detected: false,
    count: 2,
    processing_time_ms: 1.5,
  };

  it('shows all results by default', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<AllergyExtractor />);
    fireEvent.click(screen.getByText('Allergy List'));
    fireEvent.click(screen.getByText('Extract Allergies'));

    await waitFor(() => {
      expect(screen.getByText('penicillin')).toBeTruthy();
      expect(screen.getByText('peanuts')).toBeTruthy();
    });
  });

  it('filters by drug category', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<AllergyExtractor />);
    fireEvent.click(screen.getByText('Allergy List'));
    fireEvent.click(screen.getByText('Extract Allergies'));

    await waitFor(() => {
      expect(screen.getByText('penicillin')).toBeTruthy();
    });

    fireEvent.click(screen.getByText('Drug'));

    expect(screen.getByText('penicillin')).toBeTruthy();
    expect(screen.queryByText('peanuts')).toBeNull();
  });
});

// ─── Table Headers ───────────────────────────────────────────

describe('AllergyExtractor — table structure', () => {
  it('renders table column headers', async () => {
    const mockResponse = {
      allergies: [
        {
          allergen: 'test',
          canonical_name: 'test',
          category: 'drug',
          reactions: [],
          severity: 'mild',
          assertion_status: 'present',
          confidence: 0.80,
          start_char: 0,
          end_char: 4,
        },
      ],
      nkda_detected: false,
      count: 1,
      processing_time_ms: 1,
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockResponse,
    });
    render(<AllergyExtractor />);
    fireEvent.click(screen.getByText('Allergy List'));
    fireEvent.click(screen.getByText('Extract Allergies'));

    await waitFor(() => {
      expect(screen.getByText('Allergen')).toBeTruthy();
      expect(screen.getByText('Category')).toBeTruthy();
      expect(screen.getByText('Reactions')).toBeTruthy();
      expect(screen.getByText('Severity')).toBeTruthy();
      expect(screen.getByText('Status')).toBeTruthy();
      expect(screen.getByText('Confidence')).toBeTruthy();
    });
  });
});


