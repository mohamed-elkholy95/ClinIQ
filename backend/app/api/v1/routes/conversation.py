"""Conversation memory REST API endpoints.

Provides session-scoped conversation memory for context-aware clinical
analysis.  Users can record analysis turns, retrieve aggregated context,
clear sessions, and inspect memory statistics.

Endpoints
---------
POST /conversation/turns
    Record a new analysis turn in a session's conversation history.
POST /conversation/context
    Retrieve aggregated context from a session's recent turns.
DELETE /conversation/{session_id}
    Clear a session's entire conversation history.
GET /conversation/stats
    Return memory usage statistics (active sessions, total turns, config).
GET /conversation/sessions
    List active session IDs with turn counts and last-access timestamps.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.search.conversation_memory import (
    ConversationMemory,
    ConversationTurn,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["conversation"])

# ---------------------------------------------------------------------------
# Singleton memory instance
# ---------------------------------------------------------------------------

_memory: ConversationMemory | None = None


def _get_memory() -> ConversationMemory:
    """Return or lazily initialise the global ConversationMemory.

    Returns
    -------
    ConversationMemory
        Singleton instance configured with sensible defaults.
    """
    global _memory
    if _memory is None:
        _memory = ConversationMemory(
            max_turns_per_session=50,
            session_ttl_seconds=7200.0,  # 2 hours
            max_sessions=5000,
            eviction_interval=120.0,
        )
        logger.info("Conversation memory initialised")
    return _memory


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class EntityItem(BaseModel):
    """A single extracted entity."""

    text: str = Field(..., description="Surface form of the entity.")
    entity_type: str = Field(
        ..., description="Entity category (e.g. DISEASE, MEDICATION)."
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Model confidence score."
    )


class ICDItem(BaseModel):
    """A single ICD-10 prediction."""

    code: str = Field(..., description="ICD-10 code (e.g. E11.9).")
    description: str = Field("", description="Code description.")
    confidence: float = Field(
        0.0, ge=0.0, le=1.0, description="Model confidence score."
    )


class AddTurnRequest(BaseModel):
    """Request body for recording a conversation turn.

    The ``session_id`` groups turns into logical conversations so that
    subsequent analyses can reference prior context.
    """

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Unique session identifier (e.g. user ID, session cookie).",
    )
    text: str = Field(
        ...,
        min_length=1,
        max_length=500_000,
        description="Clinical text that was analysed.",
    )
    entities: list[EntityItem] = Field(
        default_factory=list,
        description="Entities extracted from the text.",
    )
    icd_codes: list[ICDItem] = Field(
        default_factory=list,
        description="ICD-10 codes predicted for the text.",
    )
    risk_score: float | None = Field(
        None, ge=0.0, le=1.0, description="Overall risk score."
    )
    risk_level: str | None = Field(
        None, description="Risk category (low/moderate/high/critical)."
    )
    summary: str | None = Field(
        None, max_length=5000, description="Generated clinical summary."
    )
    document_id: str | None = Field(
        None, max_length=256, description="Optional external document identifier."
    )
    metadata: dict[str, str] = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata.",
    )


class AddTurnResponse(BaseModel):
    """Response after recording a conversation turn."""

    session_id: str
    turn_id: int
    turn_count: int = Field(description="Total turns now in the session.")


class ContextRequest(BaseModel):
    """Request body for retrieving session context."""

    session_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Session to retrieve context for.",
    )
    last_n: int = Field(
        5,
        ge=1,
        le=50,
        description="Number of recent turns to include.",
    )


class ContextResponse(BaseModel):
    """Aggregated context from a session's conversation history."""

    session_id: str
    turn_count: int
    turns: list[dict]
    unique_entities: list[str]
    unique_icd_codes: list[str]
    overall_risk_trend: list[float]


class StatsResponse(BaseModel):
    """Memory usage statistics."""

    active_sessions: int
    total_turns: int
    max_sessions: int
    max_turns_per_session: int
    session_ttl_seconds: float


class SessionInfo(BaseModel):
    """Summary of a single active session."""

    session_id: str
    turn_count: int
    last_access: float = Field(description="Unix epoch of last activity.")
    oldest_turn_id: int | None = Field(
        None, description="ID of the oldest turn in the session."
    )
    newest_turn_id: int | None = Field(
        None, description="ID of the newest turn in the session."
    )


class SessionsListResponse(BaseModel):
    """List of active conversation sessions."""

    sessions: list[SessionInfo]
    total: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/conversation/turns",
    response_model=AddTurnResponse,
    summary="Record an analysis turn",
    description=(
        "Add a completed analysis result to a session's conversation history.  "
        "Subsequent ``/conversation/context`` requests for the same session "
        "will include this turn's entities, ICD codes, and risk information."
    ),
    responses={
        200: {"description": "Turn successfully recorded."},
        422: {"description": "Validation error in request body."},
    },
)
async def add_turn(request: AddTurnRequest) -> AddTurnResponse:
    """Record a new analysis turn in the conversation memory.

    Parameters
    ----------
    request:
        Turn data including session ID, analysed text, and extracted
        clinical intelligence.

    Returns
    -------
    AddTurnResponse
        Confirmation with the assigned turn ID and updated turn count.
    """
    memory = _get_memory()

    # Build the internal ConversationTurn
    snippet_length = 500
    turn = ConversationTurn(
        timestamp=time.time(),
        text_snippet=request.text[:snippet_length],
        text_length=len(request.text),
        entities=[e.model_dump() for e in request.entities],
        icd_codes=[c.model_dump() for c in request.icd_codes],
        risk_score=request.risk_score,
        risk_level=request.risk_level,
        summary=request.summary,
        document_id=request.document_id,
        metadata=request.metadata,
    )

    turn_id = memory.add_turn(request.session_id, turn)

    # Get updated count
    ctx = memory.get_session_context(request.session_id, last_n=0)

    logger.info(
        "Recorded conversation turn %d for session %s",
        turn_id,
        request.session_id,
    )

    return AddTurnResponse(
        session_id=request.session_id,
        turn_id=turn_id,
        turn_count=ctx.turn_count,
    )


@router.post(
    "/conversation/context",
    response_model=ContextResponse,
    summary="Get session context",
    description=(
        "Retrieve aggregated context from a session's recent conversation "
        "history.  Returns deduplicated entities, ICD codes, and risk "
        "trends across all recorded turns."
    ),
    responses={
        200: {"description": "Context retrieved (may be empty for unknown sessions)."},
    },
)
async def get_context(request: ContextRequest) -> ContextResponse:
    """Retrieve aggregated context for a conversation session.

    Parameters
    ----------
    request:
        Session ID and number of recent turns to include.

    Returns
    -------
    ContextResponse
        Aggregated context including deduplicated entities, ICD codes,
        and risk score trends.
    """
    memory = _get_memory()
    ctx = memory.get_session_context(request.session_id, last_n=request.last_n)

    return ContextResponse(
        session_id=ctx.session_id,
        turn_count=ctx.turn_count,
        turns=ctx.turns,
        unique_entities=ctx.unique_entities,
        unique_icd_codes=ctx.unique_icd_codes,
        overall_risk_trend=ctx.overall_risk_trend,
    )


@router.delete(
    "/conversation/{session_id}",
    summary="Clear session history",
    description=(
        "Remove all conversation history for a session.  Returns 404 if "
        "the session does not exist."
    ),
    responses={
        200: {"description": "Session cleared successfully."},
        404: {"description": "Session not found."},
    },
)
async def clear_session(session_id: str) -> dict:
    """Clear a session's conversation history.

    Parameters
    ----------
    session_id:
        The session to clear.

    Returns
    -------
    dict
        Confirmation with the cleared session ID.

    Raises
    ------
    HTTPException
        404 if the session does not exist.
    """
    memory = _get_memory()
    cleared = memory.clear_session(session_id)

    if not cleared:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found.",
        )

    logger.info("Cleared conversation session %s", session_id)
    return {"session_id": session_id, "status": "cleared"}


@router.get(
    "/conversation/stats",
    response_model=StatsResponse,
    summary="Memory statistics",
    description=(
        "Return conversation memory usage statistics including active "
        "session count, total turns stored, and configuration limits."
    ),
)
async def get_stats() -> StatsResponse:
    """Return conversation memory statistics.

    Returns
    -------
    StatsResponse
        Active sessions, total turns, and memory configuration.
    """
    memory = _get_memory()
    stats = memory.stats()
    return StatsResponse(**stats)


@router.get(
    "/conversation/sessions",
    response_model=SessionsListResponse,
    summary="List active sessions",
    description=(
        "List all active conversation sessions with turn counts and "
        "last-access timestamps.  Useful for monitoring and debugging."
    ),
)
async def list_sessions() -> SessionsListResponse:
    """List all active conversation sessions.

    Returns
    -------
    SessionsListResponse
        List of active sessions with metadata.
    """
    memory = _get_memory()

    sessions: list[SessionInfo] = []
    with memory._lock:
        for sid, sess in memory._sessions.items():
            oldest_id = sess.turns[0].turn_id if sess.turns else None
            newest_id = sess.turns[-1].turn_id if sess.turns else None
            sessions.append(
                SessionInfo(
                    session_id=sid,
                    turn_count=len(sess.turns),
                    last_access=sess.last_access,
                    oldest_turn_id=oldest_id,
                    newest_turn_id=newest_id,
                )
            )

    # Sort by most recently active first
    sessions.sort(key=lambda s: s.last_access, reverse=True)

    return SessionsListResponse(
        sessions=sessions,
        total=len(sessions),
    )
