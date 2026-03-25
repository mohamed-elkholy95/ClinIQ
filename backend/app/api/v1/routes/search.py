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
from app.ml.search.query_expansion import MedicalQueryExpander
from app.ml.search.reranker import ClinicalRuleReRanker, ReRankCandidate

router = APIRouter(tags=["search"])
logger = logging.getLogger(__name__)

# Module-level singletons
_engine: HybridSearchEngine | None = None
_indexed_count: int = 0
_expander = MedicalQueryExpander(max_expansions=6)
_reranker = ClinicalRuleReRanker()


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
    expand_query:
        Whether to apply medical query expansion (synonyms, abbreviations).
    rerank:
        Whether to apply re-ranking on initial retrieval results.
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
    expand_query: bool = Field(
        default=True,
        description="Apply medical query expansion (synonyms, abbreviations)",
    )
    rerank: bool = Field(
        default=True,
        description="Apply re-ranking to refine initial retrieval results",
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


class QueryExpansionInfo(BaseModel):
    """Details of query expansion applied.

    Attributes
    ----------
    original_query:
        The user's original query before expansion.
    expanded_terms:
        Terms added via synonym/abbreviation expansion.
    expansion_sources:
        Mapping of each expanded term to the reason it was added.
    """

    original_query: str
    expanded_terms: list[str] = Field(default_factory=list)
    expansion_sources: dict[str, str] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    """Search endpoint response.

    Attributes
    ----------
    query:
        The effective query used for search (may be expanded).
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
    query_expansion:
        Details of medical query expansion (if applied).
    reranked:
        Whether results were re-ranked.
    """

    query: str
    results: list[SearchHit]
    total_hits: int
    corpus_size: int
    processing_time_ms: float
    alpha: float
    query_expansion: QueryExpansionInfo | None = None
    reranked: bool = False


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

    # Step 1: Query expansion (optional)
    expansion_info: QueryExpansionInfo | None = None
    search_query = payload.query

    if payload.expand_query:
        expanded = _expander.expand(payload.query)
        if expanded.expansion_count > 0:
            search_query = expanded.expanded_query
            expansion_info = QueryExpansionInfo(
                original_query=expanded.original,
                expanded_terms=expanded.expanded_terms,
                expansion_sources=expanded.expansion_sources,
            )
            logger.info(
                "Query expanded: '%s' → +%d terms",
                payload.query,
                expanded.expansion_count,
            )

    # Step 2: Initial retrieval (over-fetch for re-ranking)
    retrieval_k = payload.top_k * 3 if payload.rerank else payload.top_k
    initial_results = engine.search(
        query=search_query,
        top_k=retrieval_k,
        min_score=payload.min_score,
    )

    # Step 3: Re-ranking (optional)
    reranked = False
    if payload.rerank and len(initial_results) > 1:
        candidates = [
            ReRankCandidate(
                doc_id=r.doc_id,
                text=r.snippet,
                initial_score=r.score,
            )
            for r in initial_results
        ]
        reranked_results = _reranker.rerank(
            query=payload.query,  # Use original query for re-ranking
            candidates=candidates,
            top_k=payload.top_k,
            initial_weight=0.4,
        )
        final_results = [
            SearchHit(
                doc_id=rr.doc_id,
                score=rr.score,
                bm25_score=rr.score_components.get("initial", 0.0),
                tfidf_score=rr.score_components.get("reranker", 0.0),
                snippet=rr.text,
            )
            for rr in reranked_results
        ]
        reranked = True
    else:
        final_results = [
            SearchHit(
                doc_id=r.doc_id,
                score=r.score,
                bm25_score=r.bm25_score,
                tfidf_score=r.tfidf_score,
                snippet=r.snippet,
            )
            for r in initial_results[:payload.top_k]
        ]

    elapsed = (time.monotonic() - t0) * 1000

    return SearchResponse(
        query=search_query,
        results=final_results,
        total_hits=len(final_results),
        corpus_size=engine.corpus_size,
        processing_time_ms=round(elapsed, 2),
        alpha=payload.alpha,
        query_expansion=expansion_info,
        reranked=reranked,
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
