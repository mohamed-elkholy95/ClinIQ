# Model Card: Conversation Memory

## Overview

| Field | Value |
|-------|-------|
| **Component** | `app.ml.search.conversation_memory` |
| **Type** | Session-scoped in-memory context manager |
| **ML Dependencies** | None (pure Python) |
| **Latency** | <0.1ms per operation |
| **Thread Safety** | Yes (global lock) |

## Purpose

Provides session-scoped conversation memory for context-aware clinical analysis. Tracks analysis history within a session so subsequent queries can reference previous results — for example, after analysing a discharge summary a user can ask "what medications were mentioned?" and the conversation memory supplies the prior document context.

## Architecture

```
                    ┌──────────────────────────────────┐
                    │       ConversationMemory          │
                    │  (singleton, thread-safe)         │
                    ├──────────────────────────────────┤
                    │  _sessions: dict[str, _Session]  │
                    │  _lock: threading.Lock            │
                    │  TTL-based eviction               │
                    │  Max-session cap eviction          │
                    └──────────┬───────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        ┌──────────┐   ┌──────────┐     ┌──────────┐
        │ _Session │   │ _Session │     │ _Session │
        │  (deque) │   │  (deque) │     │  (deque) │
        └──────────┘   └──────────┘     └──────────┘
              │                │                │
    ┌─────┬──┴──┐    ┌────┬──┴──┐     ┌────┬──┴──┐
    │Turn │Turn │    │Turn│Turn │     │Turn│Turn │
    └─────┴─────┘    └────┴─────┘     └────┴─────┘
```

## Data Model

### ConversationTurn

| Field | Type | Description |
|-------|------|-------------|
| `turn_id` | int | Monotonically increasing per-session |
| `timestamp` | float | Unix epoch |
| `text_snippet` | str | First 500 chars of input (memory-bounded) |
| `text_length` | int | Full document character count |
| `entities` | list[dict] | Extracted entities (type, text, confidence) |
| `icd_codes` | list[dict] | Predicted ICD-10 codes |
| `risk_score` | float \| None | Overall risk (0–1) |
| `risk_level` | str \| None | Risk category |
| `summary` | str \| None | Extractive summary |
| `document_id` | str \| None | External document ID |
| `metadata` | dict[str, str] | Arbitrary key-value pairs |

### SessionContext (aggregated output)

| Field | Type | Description |
|-------|------|-------------|
| `session_id` | str | Session identifier |
| `turns` | list[dict] | Context dicts from recent N turns |
| `turn_count` | int | Total turns in session |
| `unique_entities` | list[str] | Deduplicated entity texts |
| `unique_icd_codes` | list[str] | Deduplicated ICD-10 codes |
| `overall_risk_trend` | list[float] | Risk scores over time |

## REST API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/conversation/turns` | Record an analysis turn |
| POST | `/conversation/context` | Get aggregated session context |
| DELETE | `/conversation/{session_id}` | Clear session history |
| GET | `/conversation/stats` | Memory usage statistics |
| GET | `/conversation/sessions` | List active sessions |

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_turns_per_session` | 50 | FIFO eviction after limit |
| `session_ttl_seconds` | 7200 (2h) | Idle session expiry |
| `max_sessions` | 5000 | Global session cap |
| `eviction_interval` | 120s | Amortised eviction sweep frequency |

## Design Decisions

1. **In-memory only** — No persistence layer. For multi-replica deployments, swap to Redis; the interface is designed for transparent backend swap.
2. **Bounded per session** — Deque with maxlen prevents unbounded memory growth. FIFO drops oldest turns.
3. **Structured context format** — Returns dicts not prose, so downstream consumers (LLMs, rule engines) parse reliably.
4. **Snippet not full text** — Stores only first 500 chars to bound memory; full text length is tracked.
5. **Amortised eviction** — TTL expiry check runs at most once per `eviction_interval` to avoid scanning on every `add_turn`.
6. **Global lock** — Simple threading.Lock guards dict mutations. Per-session locks are unnecessary since the critical section is brief.

## Limitations

- **Single-process only** — Memory is not shared across workers/replicas. Production deployments with multiple API workers need a Redis or database backing store.
- **No persistence** — All conversation history is lost on process restart.
- **No authentication** — Any caller with a session_id can read/write that session's history. Integrate with auth middleware for production use.
- **Memory-bounded** — Large deployments (5000 sessions × 50 turns) use approximately 50–100 MB RAM depending on entity/code density.

## Performance

| Operation | Complexity | Typical Latency |
|-----------|-----------|-----------------|
| `add_turn` | O(1) amortised | <0.05ms |
| `get_context` | O(N) where N = turns | <0.1ms |
| `clear_session` | O(1) | <0.01ms |
| `stats` | O(S) where S = sessions | <0.1ms |
| Eviction sweep | O(S) | <1ms for 5000 sessions |

## Test Coverage

- **Unit tests**: `tests/test_conversation_memory.py` (430 lines)
- **Route tests**: `tests/test_conversation_route.py` (26 tests)
  - AddTurn: 10 tests (success, multiple, minimal, validation, bounds, different sessions, metadata)
  - GetContext: 6 tests (empty, after turn, last_n, aggregation, validation)
  - ClearSession: 3 tests (existing, nonexistent, verify empty after clear)
  - Stats: 2 tests (empty, after turns)
  - ListSessions: 4 tests (empty, after turns, sort order, after clear)
  - Workflow: 1 end-to-end lifecycle test

## Ethical Considerations

- Conversation memory may contain PHI from clinical notes. The `text_snippet` field stores raw clinical text.
- Session data should be treated as PHI under HIPAA. Ensure proper access controls in production.
- TTL-based eviction provides automatic data minimisation but is not sufficient for compliance — implement explicit purge policies.
