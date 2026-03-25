"""Document search endpoint with hybrid BM25 + TF-IDF ranking.

Provides a natural-language search interface over previously ingested
clinical documents.  Results are ranked using a hybrid score that
interpolates BM25 lexical matching with TF-IDF cosine similarity,
giving clinicians the ability to find relevant notes using both exact
medical terms and related concepts.

Design decisions
----------------
* **In-memory index** — The search index lives in process memory and is
  rebuilt on first query or when ``POST /search/reindex`` is called.
  This avoids an external search-engine dependency (Elasticsearch) for
  the MVP while still supporting corpora up to ~50k documents on a
  single node.
* **Lazy initialisation** — The index is not built at startup because
  the database may not yet contain documents.  The first search triggers
  a synchronous index build (typically <1 s for a few thousand docs).
* **Alpha tuning** — The interpolation weight is exposed as a query
  parameter so domain experts can experiment without redeploying.
"""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.ml.search.hybrid import HybridSearchEngine, SearchResult

router = APIRouter(tags=["search"])
logger = logging.getLogger(__name__)

# Module-level singleton — rebuilt via /search/reindex or on first query
_engine: HybridSearchEngine | None = None
_indexed_count: int = 0


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    """Search query payload.

    Attributes
    ----------
    query:
        Natural-language search string.
    top_k:
        Maximum number of results to return.
    min_score:
        Minimum hybrid score threshold.
    alpha:
        BM25 vs TF-IDF interpolation weight (0 = pure TF-IDF, 1 = pure BM25).
    """

    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: int = Field(default=10, ge=1, le=100, description="Max results")
    min_score: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Minimum score threshold",
    )
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="BM25 weight (0=pure TF-IDF, 1=pure BM25)",
    )


class SearchHit(BaseModel):
    """A single search result.

    Attributes
    ----------
    doc_id:
        Database document identifier.
    score:
        Combined hybrid relevance score.
    bm25_score:
        Normalised BM25 component.
    tfidf_score:
        TF-IDF cosine similarity component.
    snippet:
        Contextual excerpt around the best-matching query term.
    """

    doc_id: str
    score: float
    bm25_score: float
    tfidf_score: float
    snippet: str


class SearchResponse(BaseModel):
    """Search endpoint response.

    Attributes
    ----------
    query:
        Echo of the search query for logging.
    results:
        Ranked list of matching documents.
    total_hits:
        Number of results returned.
    corpus_size:
        Total documents in the search index.
    processing_time_ms:
        Wall-clock search time in milliseconds.
    alpha:
        The BM25/TF-IDF interpolation weight used for this query.
    """

    query: str
    results: list[SearchHit]
    total_hits: int
    corpus_size: int
    processing_time_ms: float
    alpha: float


class ReindexResponse(BaseModel):
    """Response from the reindex endpoint.

    Attributes
    ----------
    documents_indexed:
        Number of documents in the new index.
    processing_time_ms:
        Time taken to build the index.
    """

    documents_indexed: int
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------


async def _ensure_index(
    db: AsyncSession,
    alpha: float = 0.5,
    force: bool = False,
) -> HybridSearchEngine:
    """Build or return the search index.

    Parameters
    ----------
    db:
        Database session for loading document texts.
    alpha:
        Interpolation weight for the hybrid engine.
    force:
        If True, rebuild even if an index already exists.

    Returns
    -------
    HybridSearchEngine
        Ready-to-query search engine.
    """
    global _engine, _indexed_count

    if _engine is not None and not force:
        # Update alpha if changed
        _engine.alpha = alpha
        return _engine

    t0 = time.monotonic()

    # Load document texts from the database
    # Using raw SQL to stay ORM-agnostic (works with any table shape)
    try:
        result = await db.execute(
            text("SELECT id, content FROM documents ORDER BY created_at")
        )
        rows = result.fetchall()
    except Exception:
        # Table may not exist yet — return empty engine
        logger.warning("Could not load documents for search index")
        rows = []

    engine = HybridSearchEngine(alpha=alpha)
    if rows:
        texts = [row[1] for row in rows if row[1]]
        doc_ids = [str(row[0]) for row in rows if row[1]]
        engine.index(texts, doc_ids)
    else:
        engine.index([], [])

    elapsed = (time.monotonic() - t0) * 1000
    _indexed_count = engine.corpus_size
    _engine = engine

    logger.info(
        "Search index built: %d documents in %.1f ms",
        _indexed_count,
        elapsed,
    )
    return engine


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search clinical documents",
    description=(
        "Search over previously ingested clinical documents using hybrid "
        "BM25 + TF-IDF ranking.  Returns ranked results with contextual "
        "snippets and per-component scores for transparency."
    ),
)
async def search_documents(
    payload: SearchRequest,
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> SearchResponse:
    """Execute a hybrid search query over the document corpus.

    Parameters
    ----------
    payload:
        Search query with ranking parameters.
    db:
        Database session for lazy index initialisation.

    Returns
    -------
    SearchResponse
        Ranked results with scoring details.
    """
    t0 = time.monotonic()
    engine = await _ensure_index(db, alpha=payload.alpha)

    results = engine.search(
        query=payload.query,
        top_k=payload.top_k,
        min_score=payload.min_score,
    )

    elapsed = (time.monotonic() - t0) * 1000

    return SearchResponse(
        query=payload.query,
        results=[
            SearchHit(
                doc_id=r.doc_id,
                score=r.score,
                bm25_score=r.bm25_score,
                tfidf_score=r.tfidf_score,
                snippet=r.snippet,
            )
            for r in results
        ],
        total_hits=len(results),
        corpus_size=engine.corpus_size,
        processing_time_ms=round(elapsed, 2),
        alpha=payload.alpha,
    )


@router.post(
    "/search/reindex",
    response_model=ReindexResponse,
    status_code=status.HTTP_200_OK,
    summary="Rebuild the search index",
    description=(
        "Force a full rebuild of the in-memory search index from the "
        "document database.  Use after bulk document ingestion."
    ),
)
async def reindex(
    db: Annotated[AsyncSession, Depends(get_db_session)],
) -> ReindexResponse:
    """Rebuild the search index from the document database.

    Parameters
    ----------
    db:
        Database session for loading documents.

    Returns
    -------
    ReindexResponse
        Number of documents indexed and build time.
    """
    t0 = time.monotonic()
    engine = await _ensure_index(db, force=True)
    elapsed = (time.monotonic() - t0) * 1000

    return ReindexResponse(
        documents_indexed=engine.corpus_size,
        processing_time_ms=round(elapsed, 2),
    )
