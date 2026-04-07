"""Tests for conversation memory REST API endpoints.

Covers all five endpoints: add turn, get context, clear session,
stats, and list sessions.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

# Reset the singleton between tests
import app.api.v1.routes.conversation as conv_module
from app.main import app


@pytest.fixture(autouse=True)
def _reset_memory():
    """Reset the conversation memory singleton before each test."""
    conv_module._memory = None
    yield
    conv_module._memory = None


@pytest.fixture
async def client():
    """Async HTTP client wired to the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _turn_payload(
    session_id: str = "sess-1",
    text: str = "Patient presents with chest pain and shortness of breath.",
    **overrides,
) -> dict:
    """Build a valid AddTurnRequest payload with optional overrides."""
    base = {
        "session_id": session_id,
        "text": text,
        "entities": [
            {"text": "chest pain", "entity_type": "SYMPTOM", "confidence": 0.95},
            {"text": "shortness of breath", "entity_type": "SYMPTOM", "confidence": 0.88},
        ],
        "icd_codes": [
            {"code": "R07.9", "description": "Chest pain, unspecified", "confidence": 0.82},
        ],
        "risk_score": 0.65,
        "risk_level": "moderate",
        "summary": "Patient with acute chest pain and dyspnea.",
        "document_id": "doc-001",
        "metadata": {"source": "ER"},
    }
    base.update(overrides)
    return base


# ===================================================================
# POST /conversation/turns
# ===================================================================


class TestAddTurn:
    """Tests for the add-turn endpoint."""

    @pytest.mark.anyio
    async def test_add_first_turn(self, client: AsyncClient) -> None:
        """First turn in a new session should return turn_id=1."""
        resp = await client.post(
            "/api/v1/conversation/turns", json=_turn_payload()
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "sess-1"
        assert body["turn_id"] == 1
        assert body["turn_count"] == 1

    @pytest.mark.anyio
    async def test_add_multiple_turns(self, client: AsyncClient) -> None:
        """Turn IDs should increment monotonically."""
        for i in range(1, 4):
            resp = await client.post(
                "/api/v1/conversation/turns",
                json=_turn_payload(text=f"Note {i}: Patient stable."),
            )
            assert resp.status_code == 200
            body = resp.json()
            assert body["turn_id"] == i
            assert body["turn_count"] == i

    @pytest.mark.anyio
    async def test_add_turn_minimal(self, client: AsyncClient) -> None:
        """A turn with only session_id and text should succeed."""
        resp = await client.post(
            "/api/v1/conversation/turns",
            json={"session_id": "sess-min", "text": "Minimal note."},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["turn_id"] == 1

    @pytest.mark.anyio
    async def test_add_turn_missing_session_id(self, client: AsyncClient) -> None:
        """Missing session_id should return 422."""
        resp = await client.post(
            "/api/v1/conversation/turns", json={"text": "No session."}
        )
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_add_turn_missing_text(self, client: AsyncClient) -> None:
        """Missing text should return 422."""
        resp = await client.post(
            "/api/v1/conversation/turns", json={"session_id": "s1"}
        )
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_add_turn_empty_session_id(self, client: AsyncClient) -> None:
        """Empty session_id string should return 422."""
        resp = await client.post(
            "/api/v1/conversation/turns",
            json={"session_id": "", "text": "Test."},
        )
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_add_turn_empty_text(self, client: AsyncClient) -> None:
        """Empty text string should return 422."""
        resp = await client.post(
            "/api/v1/conversation/turns",
            json={"session_id": "s1", "text": ""},
        )
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_add_turn_risk_score_bounds(self, client: AsyncClient) -> None:
        """Risk score outside 0-1 should return 422."""
        resp = await client.post(
            "/api/v1/conversation/turns",
            json=_turn_payload(risk_score=1.5),
        )
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_add_turn_different_sessions(self, client: AsyncClient) -> None:
        """Turns in different sessions should have independent IDs."""
        r1 = await client.post(
            "/api/v1/conversation/turns",
            json=_turn_payload(session_id="sess-A"),
        )
        r2 = await client.post(
            "/api/v1/conversation/turns",
            json=_turn_payload(session_id="sess-B"),
        )
        assert r1.json()["turn_id"] == 1
        assert r2.json()["turn_id"] == 1

    @pytest.mark.anyio
    async def test_add_turn_with_metadata(self, client: AsyncClient) -> None:
        """Metadata dict should be accepted."""
        resp = await client.post(
            "/api/v1/conversation/turns",
            json=_turn_payload(metadata={"department": "cardiology", "provider": "Dr. Smith"}),
        )
        assert resp.status_code == 200


# ===================================================================
# POST /conversation/context
# ===================================================================


class TestGetContext:
    """Tests for the get-context endpoint."""

    @pytest.mark.anyio
    async def test_context_empty_session(self, client: AsyncClient) -> None:
        """Unknown session should return empty context, not 404."""
        resp = await client.post(
            "/api/v1/conversation/context",
            json={"session_id": "unknown"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["turn_count"] == 0
        assert body["turns"] == []
        assert body["unique_entities"] == []
        assert body["unique_icd_codes"] == []

    @pytest.mark.anyio
    async def test_context_after_one_turn(self, client: AsyncClient) -> None:
        """Context after a single turn should reflect its entities/codes."""
        await client.post(
            "/api/v1/conversation/turns", json=_turn_payload()
        )
        resp = await client.post(
            "/api/v1/conversation/context",
            json={"session_id": "sess-1"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["turn_count"] == 1
        assert len(body["turns"]) == 1
        assert "chest pain" in body["unique_entities"]
        assert "R07.9" in body["unique_icd_codes"]
        assert body["overall_risk_trend"] == [0.65]

    @pytest.mark.anyio
    async def test_context_last_n_limits(self, client: AsyncClient) -> None:
        """The last_n parameter should control how many turns are returned."""
        for i in range(5):
            await client.post(
                "/api/v1/conversation/turns",
                json=_turn_payload(text=f"Note {i}"),
            )

        resp = await client.post(
            "/api/v1/conversation/context",
            json={"session_id": "sess-1", "last_n": 2},
        )
        body = resp.json()
        assert body["turn_count"] == 5
        assert len(body["turns"]) == 2  # Only last 2

    @pytest.mark.anyio
    async def test_context_aggregates_across_turns(self, client: AsyncClient) -> None:
        """Unique entities/codes should aggregate across all turns."""
        await client.post(
            "/api/v1/conversation/turns",
            json=_turn_payload(
                entities=[{"text": "diabetes", "entity_type": "DISEASE", "confidence": 0.9}],
                icd_codes=[{"code": "E11.9", "description": "Type 2 DM", "confidence": 0.8}],
                risk_score=0.3,
            ),
        )
        await client.post(
            "/api/v1/conversation/turns",
            json=_turn_payload(
                text="Follow-up note.",
                entities=[{"text": "hypertension", "entity_type": "DISEASE", "confidence": 0.85}],
                icd_codes=[{"code": "I10", "description": "Essential HTN", "confidence": 0.7}],
                risk_score=0.5,
            ),
        )

        resp = await client.post(
            "/api/v1/conversation/context",
            json={"session_id": "sess-1"},
        )
        body = resp.json()
        assert "diabetes" in body["unique_entities"]
        assert "hypertension" in body["unique_entities"]
        assert "E11.9" in body["unique_icd_codes"]
        assert "I10" in body["unique_icd_codes"]
        assert body["overall_risk_trend"] == [0.3, 0.5]

    @pytest.mark.anyio
    async def test_context_missing_session_id(self, client: AsyncClient) -> None:
        """Missing session_id should return 422."""
        resp = await client.post(
            "/api/v1/conversation/context", json={}
        )
        assert resp.status_code == 422

    @pytest.mark.anyio
    async def test_context_last_n_too_low(self, client: AsyncClient) -> None:
        """last_n < 1 should return 422."""
        resp = await client.post(
            "/api/v1/conversation/context",
            json={"session_id": "s1", "last_n": 0},
        )
        assert resp.status_code == 422


# ===================================================================
# DELETE /conversation/{session_id}
# ===================================================================


class TestClearSession:
    """Tests for the clear-session endpoint."""

    @pytest.mark.anyio
    async def test_clear_existing_session(self, client: AsyncClient) -> None:
        """Clearing an existing session should succeed."""
        await client.post(
            "/api/v1/conversation/turns", json=_turn_payload()
        )
        resp = await client.delete("/api/v1/conversation/sess-1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == "sess-1"
        assert body["status"] == "cleared"

    @pytest.mark.anyio
    async def test_clear_nonexistent_session(self, client: AsyncClient) -> None:
        """Clearing an unknown session should return 404."""
        resp = await client.delete("/api/v1/conversation/nonexistent")
        assert resp.status_code == 404

    @pytest.mark.anyio
    async def test_clear_then_context_empty(self, client: AsyncClient) -> None:
        """After clearing, context should be empty."""
        await client.post(
            "/api/v1/conversation/turns", json=_turn_payload()
        )
        await client.delete("/api/v1/conversation/sess-1")

        resp = await client.post(
            "/api/v1/conversation/context",
            json={"session_id": "sess-1"},
        )
        assert resp.json()["turn_count"] == 0


# ===================================================================
# GET /conversation/stats
# ===================================================================


class TestStats:
    """Tests for the stats endpoint."""

    @pytest.mark.anyio
    async def test_stats_empty(self, client: AsyncClient) -> None:
        """Stats on fresh memory should show zero sessions and turns."""
        resp = await client.get("/api/v1/conversation/stats")
        assert resp.status_code == 200
        body = resp.json()
        assert body["active_sessions"] == 0
        assert body["total_turns"] == 0
        assert body["max_sessions"] == 5000
        assert body["max_turns_per_session"] == 50
        assert body["session_ttl_seconds"] == 7200.0

    @pytest.mark.anyio
    async def test_stats_after_turns(self, client: AsyncClient) -> None:
        """Stats should reflect added turns."""
        await client.post(
            "/api/v1/conversation/turns",
            json=_turn_payload(session_id="sess-A"),
        )
        await client.post(
            "/api/v1/conversation/turns",
            json=_turn_payload(session_id="sess-A", text="Second note."),
        )
        await client.post(
            "/api/v1/conversation/turns",
            json=_turn_payload(session_id="sess-B"),
        )

        resp = await client.get("/api/v1/conversation/stats")
        body = resp.json()
        assert body["active_sessions"] == 2
        assert body["total_turns"] == 3


# ===================================================================
# GET /conversation/sessions
# ===================================================================


class TestListSessions:
    """Tests for the list-sessions endpoint."""

    @pytest.mark.anyio
    async def test_list_empty(self, client: AsyncClient) -> None:
        """Empty memory should return no sessions."""
        resp = await client.get("/api/v1/conversation/sessions")
        assert resp.status_code == 200
        body = resp.json()
        assert body["sessions"] == []
        assert body["total"] == 0

    @pytest.mark.anyio
    async def test_list_after_turns(self, client: AsyncClient) -> None:
        """Listed sessions should include turn counts and IDs."""
        await client.post(
            "/api/v1/conversation/turns",
            json=_turn_payload(session_id="sess-X"),
        )
        await client.post(
            "/api/v1/conversation/turns",
            json=_turn_payload(session_id="sess-X", text="Second note."),
        )
        await client.post(
            "/api/v1/conversation/turns",
            json=_turn_payload(session_id="sess-Y"),
        )

        resp = await client.get("/api/v1/conversation/sessions")
        body = resp.json()
        assert body["total"] == 2

        sessions_by_id = {s["session_id"]: s for s in body["sessions"]}
        assert sessions_by_id["sess-X"]["turn_count"] == 2
        assert sessions_by_id["sess-X"]["oldest_turn_id"] == 1
        assert sessions_by_id["sess-X"]["newest_turn_id"] == 2
        assert sessions_by_id["sess-Y"]["turn_count"] == 1

    @pytest.mark.anyio
    async def test_list_sorted_by_recency(self, client: AsyncClient) -> None:
        """Sessions should be sorted most-recent-first."""
        await client.post(
            "/api/v1/conversation/turns",
            json=_turn_payload(session_id="old-sess"),
        )
        # Add a newer session
        await client.post(
            "/api/v1/conversation/turns",
            json=_turn_payload(session_id="new-sess"),
        )

        resp = await client.get("/api/v1/conversation/sessions")
        sessions = resp.json()["sessions"]
        assert len(sessions) >= 2
        # Most recent should be first
        assert sessions[0]["session_id"] == "new-sess"

    @pytest.mark.anyio
    async def test_list_after_clear(self, client: AsyncClient) -> None:
        """Cleared session should disappear from the list."""
        await client.post(
            "/api/v1/conversation/turns",
            json=_turn_payload(session_id="to-clear"),
        )
        await client.delete("/api/v1/conversation/to-clear")

        resp = await client.get("/api/v1/conversation/sessions")
        ids = [s["session_id"] for s in resp.json()["sessions"]]
        assert "to-clear" not in ids


# ===================================================================
# End-to-end workflow
# ===================================================================


class TestWorkflow:
    """End-to-end conversation memory workflow tests."""

    @pytest.mark.anyio
    async def test_full_lifecycle(self, client: AsyncClient) -> None:
        """Test the full lifecycle: add turns → get context → clear."""
        # 1. Add two turns
        r1 = await client.post(
            "/api/v1/conversation/turns",
            json=_turn_payload(
                session_id="lifecycle",
                text="Initial presentation with chest pain.",
                entities=[{"text": "chest pain", "entity_type": "SYMPTOM", "confidence": 0.9}],
                risk_score=0.7,
                risk_level="high",
            ),
        )
        assert r1.json()["turn_id"] == 1

        r2 = await client.post(
            "/api/v1/conversation/turns",
            json=_turn_payload(
                session_id="lifecycle",
                text="Follow-up: chest pain resolved. New finding: atrial fibrillation.",
                entities=[
                    {"text": "chest pain", "entity_type": "SYMPTOM", "confidence": 0.85},
                    {"text": "atrial fibrillation", "entity_type": "DISEASE", "confidence": 0.92},
                ],
                icd_codes=[
                    {"code": "I48.91", "description": "Unspecified atrial fibrillation", "confidence": 0.78},
                ],
                risk_score=0.5,
                risk_level="moderate",
            ),
        )
        assert r2.json()["turn_id"] == 2
        assert r2.json()["turn_count"] == 2

        # 2. Get context
        ctx = await client.post(
            "/api/v1/conversation/context",
            json={"session_id": "lifecycle"},
        )
        body = ctx.json()
        assert body["turn_count"] == 2
        assert "atrial fibrillation" in body["unique_entities"]
        assert "I48.91" in body["unique_icd_codes"]
        assert body["overall_risk_trend"] == [0.7, 0.5]

        # 3. Verify stats
        stats = await client.get("/api/v1/conversation/stats")
        assert stats.json()["active_sessions"] >= 1

        # 4. List sessions
        sessions = await client.get("/api/v1/conversation/sessions")
        ids = [s["session_id"] for s in sessions.json()["sessions"]]
        assert "lifecycle" in ids

        # 5. Clear
        clear = await client.delete("/api/v1/conversation/lifecycle")
        assert clear.json()["status"] == "cleared"

        # 6. Verify empty
        ctx2 = await client.post(
            "/api/v1/conversation/context",
            json={"session_id": "lifecycle"},
        )
        assert ctx2.json()["turn_count"] == 0
