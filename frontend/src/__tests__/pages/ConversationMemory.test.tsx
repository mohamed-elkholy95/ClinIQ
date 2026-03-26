/**
 * Tests for the ConversationMemory page component.
 *
 * Verifies page structure, stats display, sample turn buttons, context
 * rendering, session list, and session clearing.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ConversationMemory } from '../../pages/ConversationMemory';

// ─── Mock API ────────────────────────────────────────────────

vi.mock('../../services/clinical', () => ({
  addConversationTurn: vi.fn(),
  getConversationContext: vi.fn(),
  clearConversationSession: vi.fn(),
  getConversationStats: vi.fn(),
  listConversationSessions: vi.fn(),
}));

import {
  addConversationTurn,
  getConversationContext,
  clearConversationSession,
  getConversationStats,
  listConversationSessions,
} from '../../services/clinical';

const mockAddTurn = vi.mocked(addConversationTurn);
const mockGetContext = vi.mocked(getConversationContext);
const mockClear = vi.mocked(clearConversationSession);
const mockStats = vi.mocked(getConversationStats);
const mockSessions = vi.mocked(listConversationSessions);

const emptyContext = {
  session_id: 'demo-session-1',
  turn_count: 0,
  turns: [],
  unique_entities: [],
  unique_icd_codes: [],
  overall_risk_trend: [],
};

const populatedContext = {
  session_id: 'demo-session-1',
  turn_count: 2,
  turns: [
    {
      turn: 1,
      timestamp: 1711425600.0,
      text_length: 200,
      entities_by_type: { SYMPTOM: ['chest pain'] },
      entity_count: 1,
      risk: { score: 0.72, level: 'high' },
    },
    {
      turn: 2,
      timestamp: 1711425700.0,
      text_length: 150,
      entity_count: 3,
    },
  ],
  unique_entities: ['chest pain', 'hypertension'],
  unique_icd_codes: ['R07.9', 'I10'],
  overall_risk_trend: [0.72, 0.35],
};

const defaultStats = {
  active_sessions: 3,
  total_turns: 12,
  max_sessions: 5000,
  max_turns_per_session: 50,
  session_ttl_seconds: 7200.0,
};

const defaultSessions = {
  sessions: [
    {
      session_id: 'demo-session-1',
      turn_count: 2,
      last_access: Date.now() / 1000,
      oldest_turn_id: 1,
      newest_turn_id: 2,
    },
  ],
  total: 1,
};

function renderPage() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={qc}>
      <BrowserRouter>
        <ConversationMemory />
      </BrowserRouter>
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  mockStats.mockResolvedValue(defaultStats);
  mockSessions.mockResolvedValue(defaultSessions);
  mockGetContext.mockResolvedValue(emptyContext);
  mockAddTurn.mockResolvedValue({
    session_id: 'demo-session-1',
    turn_id: 1,
    turn_count: 1,
  });
  mockClear.mockResolvedValue({
    session_id: 'demo-session-1',
    status: 'cleared',
  });
});

// ─── Page Structure ──────────────────────────────────────────

describe('ConversationMemory — page structure', () => {
  it('renders the page heading', async () => {
    renderPage();
    expect(screen.getByText('Conversation Memory')).toBeInTheDocument();
  });

  it('renders the subtitle', () => {
    renderPage();
    expect(
      screen.getByText(/Session-scoped context tracking/),
    ).toBeInTheDocument();
  });

  it('renders the refresh button', () => {
    renderPage();
    expect(screen.getByText('Refresh')).toBeInTheDocument();
  });

  it('renders the add turn section heading', () => {
    renderPage();
    expect(screen.getByText('Add Sample Analysis Turn')).toBeInTheDocument();
  });

  it('renders three sample note buttons', () => {
    renderPage();
    expect(screen.getByText('ER Chest Pain')).toBeInTheDocument();
    expect(screen.getByText('Follow-up Visit')).toBeInTheDocument();
    expect(screen.getByText('Dental Progress Note')).toBeInTheDocument();
  });
});

// ─── Stats Display ───────────────────────────────────────────

describe('ConversationMemory — stats display', () => {
  it('displays active sessions count', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('3')).toBeInTheDocument();
    });
  });

  it('displays total turns count', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('12')).toBeInTheDocument();
    });
  });

  it('displays session TTL in minutes', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('120m')).toBeInTheDocument();
    });
  });

  it('displays the session ID input', async () => {
    renderPage();
    await waitFor(() => {
      const input = screen.getByDisplayValue('demo-session-1');
      expect(input).toBeInTheDocument();
    });
  });
});

// ─── Adding Turns ────────────────────────────────────────────

describe('ConversationMemory — adding turns', () => {
  it('calls addConversationTurn when sample is clicked', async () => {
    renderPage();
    await waitFor(() => expect(mockStats).toHaveBeenCalled());

    fireEvent.click(screen.getByText('ER Chest Pain'));

    await waitFor(() => {
      expect(mockAddTurn).toHaveBeenCalledTimes(1);
      expect(mockAddTurn).toHaveBeenCalledWith(
        expect.objectContaining({
          session_id: 'demo-session-1',
          document_id: 'sample-1',
        }),
      );
    });
  });

  it('displays success message after adding turn', async () => {
    renderPage();
    await waitFor(() => expect(mockStats).toHaveBeenCalled());

    // After adding turn, context should reload with populated data
    mockGetContext.mockResolvedValueOnce(populatedContext);
    fireEvent.click(screen.getByText('ER Chest Pain'));

    await waitFor(() => {
      expect(screen.getByText(/Turn #1 recorded/)).toBeInTheDocument();
    });
  });

  it('includes entities in the turn payload', async () => {
    renderPage();
    await waitFor(() => expect(mockStats).toHaveBeenCalled());

    fireEvent.click(screen.getByText('ER Chest Pain'));

    await waitFor(() => {
      const call = mockAddTurn.mock.calls[0][0];
      expect(call.entities).toBeDefined();
      expect(call.entities!.length).toBeGreaterThan(0);
      expect(call.entities![0].text).toBe('chest pain');
    });
  });

  it('includes risk score in the turn payload', async () => {
    renderPage();
    await waitFor(() => expect(mockStats).toHaveBeenCalled());

    fireEvent.click(screen.getByText('Follow-up Visit'));

    await waitFor(() => {
      const call = mockAddTurn.mock.calls[0][0];
      expect(call.risk_score).toBe(0.35);
      expect(call.risk_level).toBe('low');
    });
  });
});

// ─── Context Display ─────────────────────────────────────────

describe('ConversationMemory — context display', () => {
  beforeEach(() => {
    mockGetContext.mockResolvedValue(populatedContext);
  });

  it('displays turn count', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText(/2 turns/)).toBeInTheDocument();
    });
  });

  it('displays unique entities', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('chest pain')).toBeInTheDocument();
      expect(screen.getByText('hypertension')).toBeInTheDocument();
    });
  });

  it('displays unique ICD codes', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('R07.9')).toBeInTheDocument();
      expect(screen.getByText('I10')).toBeInTheDocument();
    });
  });

  it('displays risk trend bars', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('72%')).toBeInTheDocument();
      expect(screen.getByText('35%')).toBeInTheDocument();
    });
  });

  it('displays turn detail cards', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Turn #1')).toBeInTheDocument();
      expect(screen.getByText('Turn #2')).toBeInTheDocument();
    });
  });

  it('expands turn details on click', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Turn #1')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Turn #1'));

    await waitFor(() => {
      // Should show JSON content
      expect(screen.getByText(/"turn": 1/)).toBeInTheDocument();
    });
  });
});

// ─── Empty State ─────────────────────────────────────────────

describe('ConversationMemory — empty state', () => {
  it('shows empty state message when no turns', async () => {
    renderPage();
    await waitFor(() => {
      expect(
        screen.getByText(/No conversation history for this session/),
      ).toBeInTheDocument();
    });
  });
});

// ─── Session Management ──────────────────────────────────────

describe('ConversationMemory — session management', () => {
  beforeEach(() => {
    mockGetContext.mockResolvedValue(populatedContext);
  });

  it('displays active sessions table', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Active Sessions (1)')).toBeInTheDocument();
      expect(screen.getByText('demo-session-1')).toBeInTheDocument();
    });
  });

  it('shows clear session button when context has turns', async () => {
    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Clear Session')).toBeInTheDocument();
    });
  });

  it('calls clearConversationSession when clear is clicked', async () => {
    mockGetContext
      .mockResolvedValueOnce(populatedContext) // initial load
      .mockResolvedValueOnce(emptyContext); // after clear

    renderPage();
    await waitFor(() => {
      expect(screen.getByText('Clear Session')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByText('Clear Session'));

    await waitFor(() => {
      expect(mockClear).toHaveBeenCalledWith('demo-session-1');
    });
  });
});

// ─── Error Handling ──────────────────────────────────────────

describe('ConversationMemory — error handling', () => {
  it('displays error when add turn fails', async () => {
    mockAddTurn.mockRejectedValueOnce(new Error('Network error'));

    renderPage();
    await waitFor(() => expect(mockStats).toHaveBeenCalled());

    fireEvent.click(screen.getByText('ER Chest Pain'));

    await waitFor(() => {
      expect(screen.getByText('Network error')).toBeInTheDocument();
    });
  });
});
