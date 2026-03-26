"""Extended tests for search route — targeting uncovered handler code paths.

Covers:
- ``search_documents()`` handler with query expansion
- ``search_documents()`` handler without expansion (no terms added)
- ``search_documents()`` handler with re-ranking enabled
- ``search_documents()`` handler without re-ranking
- ``reindex()`` handler
- End-to-end FastAPI test client calls
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def _reset_search_engine():
    """Reset the module-level search engine before each test."""
    import app.api.v1.routes.search as search_mod
    original = search_mod._engine
    search_mod._engine = None
    yield
    search_mod._engine = original


class TestSearchEndpointIntegration:
    """Full integration tests through the FastAPI test client."""

    @pytest.mark.asyncio
    async def test_search_basic_query(self, _reset_search_engine) -> None:
        """POST /api/v1/search should return search results."""
        import app.api.v1.routes.search as search_mod

        # Set up a pre-built engine with documents
        engine = MagicMock()
        engine.corpus_size = 3
        engine.alpha = 0.5

        mock_result = MagicMock()
        mock_result.doc_id = "doc-1"
        mock_result.score = 0.85
        mock_result.bm25_score = 0.9
        mock_result.tfidf_score = 0.8
        mock_result.snippet = "Patient has diabetes mellitus type 2"
        engine.search.return_value = [mock_result]

        search_mod._engine = engine

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/v1/search",
                json={
                    "query": "diabetes",
                    "top_k": 5,
                    "expand_query": False,
                    "rerank": False,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["total_hits"] >= 0
        assert "results" in data

    @pytest.mark.asyncio
    async def test_search_with_query_expansion(self, _reset_search_engine) -> None:
        """Search with query expansion should include expansion info."""
        import app.api.v1.routes.search as search_mod

        engine = MagicMock()
        engine.corpus_size = 2
        engine.alpha = 0.5
        engine.search.return_value = []
        search_mod._engine = engine

        # Mock the expander to return expanded terms
        mock_expansion = MagicMock()
        mock_expansion.expansion_count = 2
        mock_expansion.expanded_query = "hypertension high blood pressure HTN"
        mock_expansion.original = "hypertension"
        mock_expansion.expanded_terms = ["high blood pressure", "HTN"]
        mock_expansion.expansion_sources = {
            "high blood pressure": "synonym",
            "HTN": "abbreviation",
        }

        with patch.object(search_mod._expander, "expand", return_value=mock_expansion):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    "/api/v1/search",
                    json={
                        "query": "hypertension",
                        "expand_query": True,
                        "rerank": False,
                    },
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data.get("query_expansion") is not None
        assert data["query_expansion"]["original_query"] == "hypertension"

    @pytest.mark.asyncio
    async def test_search_expansion_no_terms_added(
        self, _reset_search_engine,
    ) -> None:
        """When expansion produces 0 new terms, expansion_info should be None."""
        import app.api.v1.routes.search as search_mod

        engine = MagicMock()
        engine.corpus_size = 1
        engine.alpha = 0.5
        engine.search.return_value = []
        search_mod._engine = engine

        mock_expansion = MagicMock()
        mock_expansion.expansion_count = 0
        mock_expansion.expanded_query = "xyz_unknown_query"
        mock_expansion.original = "xyz_unknown_query"
        mock_expansion.expanded_terms = []
        mock_expansion.expansion_sources = {}

        with patch.object(search_mod._expander, "expand", return_value=mock_expansion):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    "/api/v1/search",
                    json={
                        "query": "xyz_unknown_query",
                        "expand_query": True,
                        "rerank": False,
                    },
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data.get("query_expansion") is None

    @pytest.mark.asyncio
    async def test_search_with_reranking(self, _reset_search_engine) -> None:
        """Search with rerank=True should apply re-ranking."""
        import app.api.v1.routes.search as search_mod

        engine = MagicMock()
        engine.corpus_size = 5
        engine.alpha = 0.5

        # Return multiple results to trigger reranking
        results = []
        for i in range(3):
            r = MagicMock()
            r.doc_id = f"doc-{i}"
            r.score = 0.9 - i * 0.1
            r.bm25_score = 0.8 - i * 0.1
            r.tfidf_score = 0.7 - i * 0.1
            r.snippet = f"Document {i} content about diabetes"
            results.append(r)
        engine.search.return_value = results
        search_mod._engine = engine

        # Mock the reranker
        reranked_results = []
        for i in range(2):
            rr = MagicMock()
            rr.doc_id = f"doc-{i}"
            rr.score = 0.95 - i * 0.05
            rr.text = f"Document {i} content about diabetes"
            rr.score_components = {"initial": 0.8, "reranker": 0.9}
            reranked_results.append(rr)

        with patch.object(
            search_mod._reranker, "rerank", return_value=reranked_results,
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post(
                    "/api/v1/search",
                    json={
                        "query": "diabetes",
                        "top_k": 5,
                        "expand_query": False,
                        "rerank": True,
                    },
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["reranked"] is True

    @pytest.mark.asyncio
    async def test_search_rerank_single_result_skips_reranking(
        self, _reset_search_engine,
    ) -> None:
        """Reranking should be skipped when only 1 result returned."""
        import app.api.v1.routes.search as search_mod

        engine = MagicMock()
        engine.corpus_size = 1
        engine.alpha = 0.5

        single_result = MagicMock()
        single_result.doc_id = "doc-0"
        single_result.score = 0.9
        single_result.bm25_score = 0.85
        single_result.tfidf_score = 0.8
        single_result.snippet = "Only result"
        engine.search.return_value = [single_result]
        search_mod._engine = engine

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/v1/search",
                json={
                    "query": "test",
                    "rerank": True,
                    "expand_query": False,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["reranked"] is False

    @pytest.mark.asyncio
    async def test_reindex_endpoint(self, _reset_search_engine) -> None:
        """POST /api/v1/search/reindex should rebuild the index."""
        import app.api.v1.routes.search as search_mod

        # Mock the DB to return some documents
        with patch.object(
            search_mod, "_ensure_index",
        ) as mock_ensure:
            mock_engine = MagicMock()
            mock_engine.corpus_size = 10
            mock_ensure.return_value = mock_engine

            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                resp = await client.post("/api/v1/search/reindex")

        assert resp.status_code == 200
        data = resp.json()
        assert data["documents_indexed"] == 10

    @pytest.mark.asyncio
    async def test_search_empty_results(self, _reset_search_engine) -> None:
        """Search returning no results should return empty list."""
        import app.api.v1.routes.search as search_mod

        engine = MagicMock()
        engine.corpus_size = 100
        engine.alpha = 0.5
        engine.search.return_value = []
        search_mod._engine = engine

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            resp = await client.post(
                "/api/v1/search",
                json={
                    "query": "nonexistent query xyz",
                    "expand_query": False,
                    "rerank": False,
                },
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []
        assert data["total_hits"] == 0
