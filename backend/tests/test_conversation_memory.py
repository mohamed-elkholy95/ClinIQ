"""Tests for the conversation memory module.

Validates session lifecycle, turn recording, context aggregation,
TTL eviction, thread safety, and edge cases.
"""

from __future__ import annotations

import threading
import time

import pytest

from app.ml.search.conversation_memory import (
    ConversationMemory,
    ConversationTurn,
    SessionContext,
    _Session,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def memory() -> ConversationMemory:
    """Standard conversation memory with short TTL for testing."""
    return ConversationMemory(
        max_turns_per_session=5,
        session_ttl_seconds=3600,
        max_sessions=100,
    )


@pytest.fixture
def sample_turn() -> ConversationTurn:
    """A populated conversation turn."""
    return ConversationTurn(
        text_snippet="Patient presents with chest pain...",
        text_length=1500,
        entities=[
            {"text": "chest pain", "entity_type": "SYMPTOM", "confidence": 0.95},
            {"text": "lisinopril", "entity_type": "MEDICATION", "confidence": 0.9},
        ],
        icd_codes=[
            {"code": "I20.9", "description": "Angina pectoris, unspecified", "confidence": 0.8},
        ],
        risk_score=0.65,
        risk_level="high",
        summary="Patient with acute chest pain, likely angina.",
        document_id="doc-123",
    )


# ---------------------------------------------------------------------------
# ConversationTurn tests
# ---------------------------------------------------------------------------


class TestConversationTurn:
    """Test ConversationTurn data class."""

    def test_default_values(self) -> None:
        """Turn should have sensible defaults."""
        turn = ConversationTurn()
        assert turn.turn_id == 0
        assert turn.text_snippet == ""
        assert turn.text_length == 0
        assert turn.entities == []
        assert turn.icd_codes == []
        assert turn.risk_score is None
        assert turn.summary is None

    def test_to_context_dict_full(self, sample_turn: ConversationTurn) -> None:
        """Context dict should include all populated fields."""
        ctx = sample_turn.to_context_dict()
        assert "text_length" in ctx
        assert ctx["text_length"] == 1500
        assert "entities_by_type" in ctx
        assert "SYMPTOM" in ctx["entities_by_type"]
        assert "MEDICATION" in ctx["entities_by_type"]
        assert ctx["entity_count"] == 2
        assert "icd_codes" in ctx
        assert len(ctx["icd_codes"]) == 1
        assert ctx["icd_codes"][0]["code"] == "I20.9"
        assert "risk" in ctx
        assert ctx["risk"]["score"] == 0.65
        assert ctx["risk"]["level"] == "high"
        assert "summary" in ctx
        assert "document_id" in ctx
        assert ctx["document_id"] == "doc-123"

    def test_to_context_dict_minimal(self) -> None:
        """Context dict with no data should have minimal fields."""
        turn = ConversationTurn(text_length=100)
        ctx = turn.to_context_dict()
        assert "text_length" in ctx
        assert "entities_by_type" not in ctx
        assert "icd_codes" not in ctx
        assert "risk" not in ctx
        assert "summary" not in ctx

    def test_to_context_dict_entities_capped(self) -> None:
        """Entity lists should be capped at 5 per type."""
        entities = [
            {"text": f"entity-{i}", "entity_type": "SYMPTOM", "confidence": 0.9}
            for i in range(10)
        ]
        turn = ConversationTurn(entities=entities)
        ctx = turn.to_context_dict()
        assert len(ctx["entities_by_type"]["SYMPTOM"]) == 5

    def test_to_context_dict_summary_truncated(self) -> None:
        """Long summaries should be truncated in context."""
        turn = ConversationTurn(summary="x" * 500)
        ctx = turn.to_context_dict()
        assert len(ctx["summary"]) == 300

    def test_to_context_dict_icd_codes_capped(self) -> None:
        """ICD code lists should be capped at 5."""
        codes = [
            {"code": f"A{i:02d}", "description": f"Code {i}", "confidence": 0.9}
            for i in range(10)
        ]
        turn = ConversationTurn(icd_codes=codes)
        ctx = turn.to_context_dict()
        assert len(ctx["icd_codes"]) == 5


# ---------------------------------------------------------------------------
# _Session internal tests
# ---------------------------------------------------------------------------


class TestSession:
    """Test the internal _Session class."""

    def test_add_turn_assigns_id(self) -> None:
        """Each turn should get a monotonically increasing ID."""
        session = _Session("test-session", max_turns=10)
        t1 = ConversationTurn()
        t2 = ConversationTurn()
        id1 = session.add_turn(t1)
        id2 = session.add_turn(t2)
        assert id1 == 1
        assert id2 == 2

    def test_max_turns_eviction(self) -> None:
        """Oldest turns should be evicted when max is reached."""
        session = _Session("test-session", max_turns=3)
        for i in range(5):
            session.add_turn(ConversationTurn(text_snippet=f"turn-{i}"))
        assert len(session.turns) == 3
        assert session.turns[0].text_snippet == "turn-2"

    def test_get_context_returns_recent(self) -> None:
        """get_context should return the most recent N turns."""
        session = _Session("test-session", max_turns=10)
        for i in range(7):
            session.add_turn(ConversationTurn(text_snippet=f"turn-{i}"))
        ctx = session.get_context(last_n=3)
        assert len(ctx) == 3

    def test_get_context_updates_last_access(self) -> None:
        """Accessing context should update last_access timestamp."""
        session = _Session("test-session")
        old_access = session.last_access
        time.sleep(0.01)
        session.get_context()
        assert session.last_access > old_access


# ---------------------------------------------------------------------------
# ConversationMemory — basic operations
# ---------------------------------------------------------------------------


class TestConversationMemoryBasics:
    """Test basic add/get operations."""

    def test_add_turn_returns_id(
        self,
        memory: ConversationMemory,
        sample_turn: ConversationTurn,
    ) -> None:
        """add_turn should return the assigned turn ID."""
        turn_id = memory.add_turn("user-1", sample_turn)
        assert turn_id == 1

    def test_sequential_turn_ids(
        self, memory: ConversationMemory,
    ) -> None:
        """Turn IDs should increment within a session."""
        id1 = memory.add_turn("user-1", ConversationTurn())
        id2 = memory.add_turn("user-1", ConversationTurn())
        id3 = memory.add_turn("user-1", ConversationTurn())
        assert id1 == 1
        assert id2 == 2
        assert id3 == 3

    def test_separate_sessions(self, memory: ConversationMemory) -> None:
        """Different sessions should have independent turn IDs."""
        id_a = memory.add_turn("user-a", ConversationTurn())
        id_b = memory.add_turn("user-b", ConversationTurn())
        assert id_a == 1
        assert id_b == 1

    def test_get_context_existing_session(
        self,
        memory: ConversationMemory,
        sample_turn: ConversationTurn,
    ) -> None:
        """Should return context for an existing session."""
        memory.add_turn("user-1", sample_turn)
        ctx = memory.get_session_context("user-1")
        assert isinstance(ctx, SessionContext)
        assert ctx.turn_count == 1
        assert ctx.session_id == "user-1"

    def test_get_context_nonexistent_session(
        self, memory: ConversationMemory,
    ) -> None:
        """Should return empty context for unknown sessions."""
        ctx = memory.get_session_context("unknown")
        assert ctx.turn_count == 0
        assert ctx.turns == []
        assert ctx.unique_entities == []

    def test_context_aggregates_entities(
        self, memory: ConversationMemory,
    ) -> None:
        """Unique entities should be aggregated across turns."""
        memory.add_turn("user-1", ConversationTurn(
            entities=[{"text": "chest pain", "entity_type": "SYMPTOM"}],
        ))
        memory.add_turn("user-1", ConversationTurn(
            entities=[
                {"text": "lisinopril", "entity_type": "MEDICATION"},
                {"text": "chest pain", "entity_type": "SYMPTOM"},  # duplicate
            ],
        ))
        ctx = memory.get_session_context("user-1")
        assert "chest pain" in ctx.unique_entities
        assert "lisinopril" in ctx.unique_entities
        # Deduplicated
        assert len(ctx.unique_entities) == 2

    def test_context_aggregates_icd_codes(
        self, memory: ConversationMemory,
    ) -> None:
        """Unique ICD codes should be aggregated across turns."""
        memory.add_turn("user-1", ConversationTurn(
            icd_codes=[{"code": "I10", "description": "HTN"}],
        ))
        memory.add_turn("user-1", ConversationTurn(
            icd_codes=[
                {"code": "E11.9", "description": "T2DM"},
                {"code": "I10", "description": "HTN"},  # duplicate
            ],
        ))
        ctx = memory.get_session_context("user-1")
        assert "I10" in ctx.unique_icd_codes
        assert "E11.9" in ctx.unique_icd_codes
        assert len(ctx.unique_icd_codes) == 2

    def test_context_risk_trend(self, memory: ConversationMemory) -> None:
        """Risk scores should be tracked as a trend."""
        memory.add_turn("user-1", ConversationTurn(risk_score=0.3))
        memory.add_turn("user-1", ConversationTurn(risk_score=0.5))
        memory.add_turn("user-1", ConversationTurn(risk_score=0.7))
        ctx = memory.get_session_context("user-1")
        assert ctx.overall_risk_trend == [0.3, 0.5, 0.7]


# ---------------------------------------------------------------------------
# ConversationMemory — session management
# ---------------------------------------------------------------------------


class TestSessionManagement:
    """Test session lifecycle management."""

    def test_clear_session(
        self,
        memory: ConversationMemory,
        sample_turn: ConversationTurn,
    ) -> None:
        """clear_session should remove the session."""
        memory.add_turn("user-1", sample_turn)
        assert memory.clear_session("user-1") is True
        ctx = memory.get_session_context("user-1")
        assert ctx.turn_count == 0

    def test_clear_nonexistent_session(
        self, memory: ConversationMemory,
    ) -> None:
        """Clearing a nonexistent session should return False."""
        assert memory.clear_session("unknown") is False

    def test_active_sessions_count(
        self, memory: ConversationMemory,
    ) -> None:
        """active_sessions should reflect current count."""
        assert memory.active_sessions() == 0
        memory.add_turn("user-1", ConversationTurn())
        assert memory.active_sessions() == 1
        memory.add_turn("user-2", ConversationTurn())
        assert memory.active_sessions() == 2
        memory.clear_session("user-1")
        assert memory.active_sessions() == 1

    def test_stats(self, memory: ConversationMemory) -> None:
        """stats should return configuration and usage info."""
        memory.add_turn("user-1", ConversationTurn())
        memory.add_turn("user-1", ConversationTurn())
        memory.add_turn("user-2", ConversationTurn())
        stats = memory.stats()
        assert stats["active_sessions"] == 2
        assert stats["total_turns"] == 3
        assert stats["max_sessions"] == 100
        assert stats["max_turns_per_session"] == 5

    def test_max_turns_eviction(self) -> None:
        """Sessions should evict oldest turns when max is reached."""
        mem = ConversationMemory(max_turns_per_session=3)
        for i in range(5):
            mem.add_turn("user-1", ConversationTurn(
                text_snippet=f"turn-{i}",
            ))
        ctx = mem.get_session_context("user-1")
        assert ctx.turn_count == 3


# ---------------------------------------------------------------------------
# ConversationMemory — TTL eviction
# ---------------------------------------------------------------------------


class TestTTLEviction:
    """Test time-based session eviction."""

    def test_expired_session_evicted(self) -> None:
        """Sessions older than TTL should be evicted."""
        mem = ConversationMemory(
            session_ttl_seconds=0.01,  # 10ms TTL
            eviction_interval=0.0,     # Check every call
        )
        mem.add_turn("user-1", ConversationTurn())
        time.sleep(0.05)  # Wait for expiry
        # Trigger eviction via add_turn
        mem.add_turn("user-2", ConversationTurn())
        assert mem.active_sessions() == 1  # user-1 evicted

    def test_max_sessions_eviction(self) -> None:
        """When max_sessions is reached, oldest should be evicted on next add."""
        mem = ConversationMemory(
            max_sessions=3,
            session_ttl_seconds=3600,
            eviction_interval=0.0,
        )
        # Add 5 sessions
        for i in range(5):
            mem.add_turn(f"user-{i}", ConversationTurn())
            time.sleep(0.01)  # Ensure ordering

        # At this point we have 5 sessions. Eviction runs at the START
        # of the next add_turn (before the new session is created).
        # The while loop trims to max_sessions=3, then trigger is added = 4.
        mem.add_turn("user-trigger", ConversationTurn())
        # After one more cycle: eviction trims to 3, then existing session updated or new created
        mem.add_turn("user-trigger", ConversationTurn())  # Same session, no new
        assert mem.active_sessions() <= 4  # At most max_sessions + 1


# ---------------------------------------------------------------------------
# ConversationMemory — thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Verify thread-safe operations."""

    def test_concurrent_add_turns(self) -> None:
        """Multiple threads adding turns should not corrupt state."""
        mem = ConversationMemory(max_turns_per_session=100)
        barrier = threading.Barrier(4)
        errors: list[Exception] = []

        def worker(session_id: str, count: int) -> None:
            try:
                barrier.wait(timeout=5)
                for _ in range(count):
                    mem.add_turn(session_id, ConversationTurn())
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=("s1", 25)),
            threading.Thread(target=worker, args=("s2", 25)),
            threading.Thread(target=worker, args=("s1", 25)),
            threading.Thread(target=worker, args=("s3", 25)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors
        assert mem.active_sessions() == 3
        stats = mem.stats()
        assert stats["total_turns"] == 100


# ---------------------------------------------------------------------------
# SessionContext tests
# ---------------------------------------------------------------------------


class TestSessionContext:
    """Test the SessionContext data class."""

    def test_default_values(self) -> None:
        """SessionContext should have sensible defaults."""
        ctx = SessionContext(session_id="test")
        assert ctx.turns == []
        assert ctx.turn_count == 0
        assert ctx.unique_entities == []
        assert ctx.unique_icd_codes == []
        assert ctx.overall_risk_trend == []
