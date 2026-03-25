"""Tests for the document search API endpoint.

Covers the search route handler, reindex endpoint, lazy index
initialisation, and request validation.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.v1.routes.search import (
    ReindexResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    _ensure_index,
)
from app.ml.search.hybrid import HybridSearchEngine, SearchResult


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestSearchRequest:
    """Tests for SearchRequest Pydantic model."""

    def test_valid_request(self) -> None:
        """Valid request with all fields."""
        req = SearchRequest(query="diabetes", top_k=5, min_score=0.1, alpha=0.6)
        assert req.query == "diabetes"
        assert req.top_k == 5

    def test_defaults(self) -> None:
        """Default values are applied."""
        req = SearchRequest(query="test")
        assert req.top_k == 10
        assert req.min_score == 0.01
        assert req.alpha == 0.5

    def test_empty_query_rejected(self) -> None:
        """Empty query string fails validation."""
        with pytest.raises(Exception):
            SearchRequest(query="")

    def test_top_k_bounds(self) -> None:
        """top_k must be between 1 and 100."""
        with pytest.raises(Exception):
            SearchRequest(query="test", top_k=0)
        with pytest.raises(Exception):
            SearchRequest(query="test", top_k=101)

    def test_alpha_bounds(self) -> None:
        """Alpha must be between 0 and 1."""
        with pytest.raises(Exception):
            SearchRequest(query="test", alpha=1.5)
        with pytest.raises(Exception):
            SearchRequest(query="test", alpha=-0.1)


class TestSearchHit:
    """Tests for SearchHit response model."""

    def test_all_fields(self) -> None:
        """All fields serialise correctly."""
        hit = SearchHit(
            doc_id="doc-1",
            score=0.85,
            bm25_score=0.9,
            tfidf_score=0.8,
            snippet="Patient has diabetes...",
        )
        assert hit.doc_id == "doc-1"
        assert hit.score == 0.85


class TestSearchResponse:
    """Tests for SearchResponse model."""

    def test_response_structure(self) -> None:
        """Response contains all expected fields."""
        resp = SearchResponse(
            query="test",
            results=[],
            total_hits=0,
            corpus_size=100,
            processing_time_ms=5.2,
            alpha=0.5,
        )
        assert resp.total_hits == 0
        assert resp.corpus_size == 100


# ---------------------------------------------------------------------------
# Index initialisation tests
# ---------------------------------------------------------------------------


class TestEnsureIndex:
    """Tests for the _ensure_index helper."""

    @pytest.mark.asyncio
    async def test_builds_from_empty_db(self) -> None:
        """Builds empty index when no documents in database."""
        db = AsyncMock()
        # Simulate empty result set
        result_mock = MagicMock()
        result_mock.fetchall.return_value = []
        db.execute.return_value = result_mock

        # Reset module-level singleton
        import app.api.v1.routes.search as search_mod
        search_mod._engine = None

        engine = await _ensure_index(db, force=True)
        assert engine.corpus_size == 0

    @pytest.mark.asyncio
    async def test_builds_from_documents(self) -> None:
        """Indexes documents from the database."""
        db = AsyncMock()
        result_mock = MagicMock()
        result_mock.fetchall.return_value = [
            ("id-1", "Patient has diabetes mellitus"),
            ("id-2", "Routine dental examination"),
            ("id-3", "Chest pain evaluation"),
        ]
        db.execute.return_value = result_mock

        import app.api.v1.routes.search as search_mod
        search_mod._engine = None

        engine = await _ensure_index(db, force=True)
        assert engine.corpus_size == 3

    @pytest.mark.asyncio
    async def test_returns_cached_engine(self) -> None:
        """Returns existing engine without rebuilding when not forced."""
        import app.api.v1.routes.search as search_mod

        mock_engine = MagicMock(spec=HybridSearchEngine)
        mock_engine.alpha = 0.5
        search_mod._engine = mock_engine

        db = AsyncMock()
        engine = await _ensure_index(db, alpha=0.6)
        assert engine is mock_engine
        assert mock_engine.alpha == 0.6  # Alpha updated
        db.execute.assert_not_called()

        # Cleanup
        search_mod._engine = None

    @pytest.mark.asyncio
    async def test_handles_db_error(self) -> None:
        """Gracefully handles database errors."""
        db = AsyncMock()
        db.execute.side_effect = Exception("DB connection failed")

        import app.api.v1.routes.search as search_mod
        search_mod._engine = None

        engine = await _ensure_index(db, force=True)
        assert engine.corpus_size == 0

        # Cleanup
        search_mod._engine = None

    @pytest.mark.asyncio
    async def test_skips_null_content(self) -> None:
        """Documents with None content are excluded."""
        db = AsyncMock()
        result_mock = MagicMock()
        result_mock.fetchall.return_value = [
            ("id-1", "Valid document text"),
            ("id-2", None),  # Null content
            ("id-3", "Another valid document"),
        ]
        db.execute.return_value = result_mock

        import app.api.v1.routes.search as search_mod
        search_mod._engine = None

        engine = await _ensure_index(db, force=True)
        assert engine.corpus_size == 2

        # Cleanup
        search_mod._engine = None


# ---------------------------------------------------------------------------
# ReindexResponse tests
# ---------------------------------------------------------------------------


class TestReindexResponse:
    """Tests for ReindexResponse model."""

    def test_fields(self) -> None:
        """Response has required fields."""
        resp = ReindexResponse(documents_indexed=42, processing_time_ms=123.4)
        assert resp.documents_indexed == 42
        assert resp.processing_time_ms == 123.4
