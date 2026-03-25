"""Conversation memory for context-aware clinical analysis.

Tracks analysis history within a session so subsequent queries can
reference previous results.  For example, after analysing a discharge
summary a user might ask "what medications were mentioned?" — the
conversation memory provides the prior document context needed to
answer without re-uploading the text.

Architecture
------------
* **In-memory session store** — Each session is a bounded deque of
  ``ConversationTurn`` objects containing the input text, extracted
  entities, ICD predictions, and risk assessment.  Sessions are
  identified by an opaque string key (e.g. a JWT sub or session cookie).
* **TTL-based eviction** — Idle sessions are evicted after a
  configurable TTL to prevent unbounded memory growth.  The eviction
  check is amortised across ``add_turn`` calls.
* **Context window** — When building context for a new analysis, the
  memory returns the last N turns as a structured summary, not raw
  text, to keep prompt length bounded.

Design decisions
----------------
* **No persistence** — This is a stateless-server pattern (memory lives
  in process RAM).  For multi-replica deployments, session state would
  move to Redis; the ``ConversationMemory`` interface is designed to
  make that swap transparent.
* **Bounded per session** — Each session stores at most ``max_turns``
  turns.  Older turns are dropped FIFO to bound memory usage.
* **Summary format** — Context is returned as structured dicts, not
  prose, so downstream consumers (LLMs, rule engines) can parse it
  reliably.

Thread safety
-------------
Operations are guarded by a ``threading.Lock`` per ``ConversationMemory``
instance.  Fine-grained per-session locks are not needed because the
global lock is held only for the brief dict-mutation operations.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ConversationTurn:
    """A single turn in the conversation history.

    Attributes
    ----------
    turn_id:
        Monotonically increasing turn identifier within the session.
    timestamp:
        Unix epoch timestamp of when the turn was recorded.
    text_snippet:
        First N characters of the input document (for context, not
        the full text to save memory).
    text_length:
        Full character length of the original document.
    entities:
        List of extracted entity dicts (type, text, confidence).
    icd_codes:
        List of predicted ICD-10 code dicts (code, description, confidence).
    risk_score:
        Overall risk score (0-1) if risk assessment was run.
    risk_level:
        Risk category string (low/moderate/high/critical).
    summary:
        Brief extractive summary if summarisation was run.
    document_id:
        Optional application-level document ID.
    metadata:
        Arbitrary key-value metadata.
    """

    turn_id: int = 0
    timestamp: float = field(default_factory=time.time)
    text_snippet: str = ""
    text_length: int = 0
    entities: list[dict] = field(default_factory=list)
    icd_codes: list[dict] = field(default_factory=list)
    risk_score: float | None = None
    risk_level: str | None = None
    summary: str | None = None
    document_id: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)

    def to_context_dict(self) -> dict:
        """Serialise this turn to a context-friendly dict.

        Returns only the fields relevant for downstream context
        injection, omitting raw text to save tokens.

        Returns
        -------
        dict
            Structured context summary.
        """
        ctx: dict = {
            "turn": self.turn_id,
            "timestamp": self.timestamp,
            "text_length": self.text_length,
        }

        if self.entities:
            # Summarise entities by type
            by_type: dict[str, list[str]] = {}
            for ent in self.entities:
                etype = ent.get("entity_type", "unknown")
                by_type.setdefault(etype, []).append(ent.get("text", ""))
            ctx["entities_by_type"] = {
                k: v[:5] for k, v in by_type.items()  # Cap per-type
            }
            ctx["entity_count"] = len(self.entities)

        if self.icd_codes:
            ctx["icd_codes"] = [
                {"code": c.get("code"), "description": c.get("description")}
                for c in self.icd_codes[:5]
            ]

        if self.risk_score is not None:
            ctx["risk"] = {
                "score": self.risk_score,
                "level": self.risk_level,
            }

        if self.summary:
            ctx["summary"] = self.summary[:300]  # Truncate long summaries

        if self.document_id:
            ctx["document_id"] = self.document_id

        return ctx


@dataclass
class SessionContext:
    """Aggregated context from conversation history.

    Attributes
    ----------
    session_id:
        The session identifier.
    turns:
        Context dicts from recent turns.
    turn_count:
        Total number of turns in session history.
    unique_entities:
        Deduplicated set of entity texts across all turns.
    unique_icd_codes:
        Deduplicated set of ICD-10 codes across all turns.
    overall_risk_trend:
        List of risk scores over time (for trend detection).
    """

    session_id: str
    turns: list[dict] = field(default_factory=list)
    turn_count: int = 0
    unique_entities: list[str] = field(default_factory=list)
    unique_icd_codes: list[str] = field(default_factory=list)
    overall_risk_trend: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Session wrapper
# ---------------------------------------------------------------------------


class _Session:
    """Internal session container with bounded turn history."""

    __slots__ = ("session_id", "turns", "last_access", "_next_turn_id", "max_turns")

    def __init__(self, session_id: str, max_turns: int = 20) -> None:
        self.session_id = session_id
        self.turns: deque[ConversationTurn] = deque(maxlen=max_turns)
        self.last_access = time.time()
        self._next_turn_id = 1
        self.max_turns = max_turns

    def add_turn(self, turn: ConversationTurn) -> int:
        """Add a turn and return its assigned ID.

        Parameters
        ----------
        turn:
            The conversation turn to record.

        Returns
        -------
        int
            The assigned turn ID.
        """
        turn.turn_id = self._next_turn_id
        self._next_turn_id += 1
        self.turns.append(turn)
        self.last_access = time.time()
        return turn.turn_id

    def get_context(self, last_n: int = 5) -> list[dict]:
        """Get context dicts from the most recent N turns.

        Parameters
        ----------
        last_n:
            Number of recent turns to include.

        Returns
        -------
        list[dict]
            Context dicts ordered oldest-first.
        """
        self.last_access = time.time()
        recent = list(self.turns)[-last_n:]
        return [t.to_context_dict() for t in recent]


# ---------------------------------------------------------------------------
# Conversation memory manager
# ---------------------------------------------------------------------------


class ConversationMemory:
    """Session-scoped conversation memory for context-aware analysis.

    Parameters
    ----------
    max_turns_per_session:
        Maximum number of turns stored per session (FIFO eviction).
    session_ttl_seconds:
        Time-to-live for idle sessions (seconds).  Default 1 hour.
    max_sessions:
        Maximum number of concurrent sessions.  Oldest idle session
        is evicted when the limit is reached.
    eviction_interval:
        Minimum seconds between eviction sweeps to amortise cost.

    Examples
    --------
    >>> mem = ConversationMemory(max_turns_per_session=10)
    >>> mem.add_turn("user-1", ConversationTurn(
    ...     text_snippet="Patient presents with chest pain...",
    ...     text_length=1500,
    ...     entities=[{"text": "chest pain", "entity_type": "SYMPTOM"}],
    ... ))
    1
    >>> ctx = mem.get_session_context("user-1")
    >>> ctx.turn_count
    1
    """

    def __init__(
        self,
        max_turns_per_session: int = 20,
        session_ttl_seconds: float = 3600.0,
        max_sessions: int = 1000,
        eviction_interval: float = 60.0,
    ) -> None:
        self._max_turns = max_turns_per_session
        self._ttl = session_ttl_seconds
        self._max_sessions = max_sessions
        self._eviction_interval = eviction_interval
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()
        self._last_eviction = time.time()

    def _maybe_evict(self) -> None:
        """Evict expired sessions if enough time has passed.

        Called inside the lock.  Amortised to avoid scanning on every
        ``add_turn`` call.
        """
        now = time.time()
        if now - self._last_eviction < self._eviction_interval:
            return

        self._last_eviction = now
        expired = [
            sid
            for sid, sess in self._sessions.items()
            if now - sess.last_access > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]

        if expired:
            logger.info("Evicted %d expired conversation sessions", len(expired))

        # If still over limit, evict oldest
        while len(self._sessions) > self._max_sessions:
            oldest = min(self._sessions, key=lambda s: self._sessions[s].last_access)
            del self._sessions[oldest]

    def add_turn(
        self,
        session_id: str,
        turn: ConversationTurn,
    ) -> int:
        """Record a conversation turn for a session.

        Parameters
        ----------
        session_id:
            Unique session identifier (e.g. user ID, session cookie).
        turn:
            The conversation turn to record.

        Returns
        -------
        int
            The assigned turn ID within the session.
        """
        with self._lock:
            self._maybe_evict()

            if session_id not in self._sessions:
                self._sessions[session_id] = _Session(
                    session_id, max_turns=self._max_turns,
                )

            return self._sessions[session_id].add_turn(turn)

    def get_session_context(
        self,
        session_id: str,
        last_n: int = 5,
    ) -> SessionContext:
        """Get aggregated context from a session's history.

        Parameters
        ----------
        session_id:
            Session to retrieve context for.
        last_n:
            Number of recent turns to include in detail.

        Returns
        -------
        SessionContext
            Aggregated context with deduplicated entities, codes,
            and risk trend.
        """
        with self._lock:
            session = self._sessions.get(session_id)

        if session is None:
            return SessionContext(session_id=session_id)

        turns_ctx = session.get_context(last_n)

        # Aggregate unique entities and ICD codes across all turns
        entity_set: set[str] = set()
        code_set: set[str] = set()
        risk_trend: list[float] = []

        for turn in session.turns:
            for ent in turn.entities:
                entity_set.add(ent.get("text", "").lower())
            for code in turn.icd_codes:
                code_set.add(code.get("code", ""))
            if turn.risk_score is not None:
                risk_trend.append(turn.risk_score)

        return SessionContext(
            session_id=session_id,
            turns=turns_ctx,
            turn_count=len(session.turns),
            unique_entities=sorted(entity_set),
            unique_icd_codes=sorted(code_set),
            overall_risk_trend=risk_trend,
        )

    def clear_session(self, session_id: str) -> bool:
        """Remove a session's conversation history.

        Parameters
        ----------
        session_id:
            Session to clear.

        Returns
        -------
        bool
            True if the session existed and was cleared.
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                return True
            return False

    def active_sessions(self) -> int:
        """Return the number of active sessions.

        Returns
        -------
        int
            Count of sessions currently in memory.
        """
        with self._lock:
            return len(self._sessions)

    def stats(self) -> dict:
        """Return memory usage statistics.

        Returns
        -------
        dict
            Stats including active sessions, total turns, and memory
            configuration.
        """
        with self._lock:
            total_turns = sum(len(s.turns) for s in self._sessions.values())
            return {
                "active_sessions": len(self._sessions),
                "total_turns": total_turns,
                "max_sessions": self._max_sessions,
                "max_turns_per_session": self._max_turns,
                "session_ttl_seconds": self._ttl,
            }
