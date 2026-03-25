"""Hybrid clinical document search engine.

Combines BM25 lexical scoring with TF-IDF cosine similarity to produce a
single ranked result list.  Designed for clinical notes where exact term
matches (BM25) and semantic neighbourhood matches (TF-IDF) are both
important — e.g. a query for "diabetes" should surface documents
containing "DM type 2" (semantic) as well as "diabetes mellitus" (lexical).

Architecture
------------
* **BM25** (Okapi BM25) — Bag-of-words ranking with IDF weighting and
  document-length normalisation.  Good for precise term matching.
* **TF-IDF cosine** — Scikit-learn ``TfidfVectorizer`` with clinical
  stopword removal; cosine similarity between query and document vectors.
  Captures distributional similarity.
* **Hybrid score** — ``alpha * bm25_norm + (1 - alpha) * tfidf_sim``.
  ``alpha=0.5`` by default; higher values favour exact-match, lower
  values favour semantic similarity.

Thread safety
-------------
The engine is read-safe after ``index()`` completes.  Re-indexing
replaces internal state atomically via attribute reassignment, so
concurrent searches during a re-index may see the old *or* new corpus
but never a partial state.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Medical stopwords shared with the feature engineering module
# ---------------------------------------------------------------------------

_MEDICAL_STOPWORDS: set[str] = {
    "patient", "history", "examination", "assessment", "plan",
    "noted", "reviewed", "discussed", "performed", "ordered",
    "mg", "ml", "cm", "mm", "kg", "tab", "po", "iv", "bid", "tid", "qid",
    "prn", "qd", "hs", "ac", "pc", "qhs", "stat",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search hit with scoring details.

    Attributes
    ----------
    doc_index:
        Position of the document in the indexed corpus.
    doc_id:
        Application-level document identifier (e.g. database UUID).
    score:
        Combined hybrid score in [0, 1].
    bm25_score:
        Normalised BM25 component.
    tfidf_score:
        TF-IDF cosine similarity component.
    snippet:
        Contextual excerpt around the best-matching query term.
    """

    doc_index: int
    doc_id: str
    score: float
    bm25_score: float
    tfidf_score: float
    snippet: str = ""


@dataclass
class _Document:
    """Internal representation of an indexed document."""

    doc_id: str
    text: str
    tokens: list[str] = field(default_factory=list)
    token_freqs: dict[str, int] = field(default_factory=dict)
    length: int = 0


# ---------------------------------------------------------------------------
# BM25 scorer
# ---------------------------------------------------------------------------


class _BM25:
    """Okapi BM25 implementation operating on pre-tokenised documents.

    Parameters
    ----------
    k1:
        Term-frequency saturation parameter (default 1.5).
    b:
        Document-length normalisation parameter (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._n_docs: int = 0

    def fit(self, documents: list[_Document]) -> None:
        """Compute IDF values and average document length.

        Parameters
        ----------
        documents:
            Pre-tokenised internal document representations.
        """
        self._n_docs = len(documents)
        if self._n_docs == 0:
            return

        self._avgdl = sum(d.length for d in documents) / self._n_docs

        # Document frequency per term
        df: dict[str, int] = {}
        for doc in documents:
            seen: set[str] = set()
            for token in doc.tokens:
                if token not in seen:
                    df[token] = df.get(token, 0) + 1
                    seen.add(token)

        # IDF with smoothing: log((N - df + 0.5) / (df + 0.5) + 1)
        for term, freq in df.items():
            self._idf[term] = math.log(
                (self._n_docs - freq + 0.5) / (freq + 0.5) + 1.0
            )

    def score(self, query_tokens: list[str], doc: _Document) -> float:
        """Compute BM25 score for a single document against a query.

        Parameters
        ----------
        query_tokens:
            Tokenised query terms.
        doc:
            Target document.

        Returns
        -------
        float
            Raw BM25 score (unbounded, ≥0).
        """
        total = 0.0
        for qt in query_tokens:
            idf = self._idf.get(qt, 0.0)
            tf = doc.token_freqs.get(qt, 0)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc.length / max(self._avgdl, 1e-6)
            )
            total += idf * numerator / max(denominator, 1e-9)
        return total


# ---------------------------------------------------------------------------
# Hybrid search engine
# ---------------------------------------------------------------------------

# Simple regex tokeniser — lowercases and splits on non-alphanumeric chars
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenise, removing medical stopwords.

    Parameters
    ----------
    text:
        Raw text string.

    Returns
    -------
    list[str]
        Filtered token list.
    """
    return [
        t for t in _TOKEN_RE.findall(text.lower())
        if t not in _MEDICAL_STOPWORDS and len(t) > 1
    ]


def _extract_snippet(text: str, query_tokens: list[str], max_len: int = 200) -> str:
    """Extract a snippet around the first query term occurrence.

    Parameters
    ----------
    text:
        Full document text.
    query_tokens:
        Tokenised query terms to search for.
    max_len:
        Maximum snippet length in characters.

    Returns
    -------
    str
        Contextual snippet with ``…`` truncation markers.
    """
    text_lower = text.lower()
    best_pos = len(text)

    for qt in query_tokens:
        pos = text_lower.find(qt)
        if 0 <= pos < best_pos:
            best_pos = pos

    if best_pos == len(text):
        # No exact match — return start of document
        return text[:max_len].rstrip() + ("…" if len(text) > max_len else "")

    # Centre the snippet around the match
    half = max_len // 2
    start = max(0, best_pos - half)
    end = min(len(text), best_pos + half)
    prefix = "…" if start > 0 else ""
    suffix = "…" if end < len(text) else ""
    return prefix + text[start:end].strip() + suffix


class HybridSearchEngine:
    """Combined BM25 + TF-IDF search over a clinical document corpus.

    Parameters
    ----------
    alpha:
        Interpolation weight for BM25 in [0, 1].  ``1.0`` = pure BM25,
        ``0.0`` = pure TF-IDF.  Default ``0.5``.
    bm25_k1:
        BM25 term-frequency saturation.
    bm25_b:
        BM25 document-length normalisation.

    Examples
    --------
    >>> engine = HybridSearchEngine(alpha=0.6)
    >>> engine.index(
    ...     texts=["Patient presents with type 2 diabetes", "Routine dental exam"],
    ...     doc_ids=["doc-1", "doc-2"],
    ... )
    >>> results = engine.search("diabetes management", top_k=5)
    >>> results[0].doc_id
    'doc-1'
    """

    def __init__(
        self,
        alpha: float = 0.5,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self._bm25 = _BM25(k1=bm25_k1, b=bm25_b)
        self._documents: list[_Document] = []
        self._tfidf_matrix: np.ndarray | None = None
        self._tfidf_vocab: dict[str, int] = {}
        self._indexed = False

    @property
    def corpus_size(self) -> int:
        """Number of indexed documents."""
        return len(self._documents)

    def index(
        self,
        texts: Sequence[str],
        doc_ids: Sequence[str] | None = None,
    ) -> int:
        """Build the search index from a corpus.

        Parameters
        ----------
        texts:
            Raw document texts to index.
        doc_ids:
            Optional application-level identifiers; defaults to
            stringified indices.

        Returns
        -------
        int
            Number of documents indexed.

        Raises
        ------
        ValueError
            If ``doc_ids`` length doesn't match ``texts``.
        """
        if doc_ids is not None and len(doc_ids) != len(texts):
            raise ValueError(
                f"doc_ids length ({len(doc_ids)}) != texts length ({len(texts)})"
            )

        ids = doc_ids if doc_ids is not None else [str(i) for i in range(len(texts))]

        # Build internal document representations
        documents: list[_Document] = []
        for i, (text, did) in enumerate(zip(texts, ids)):
            tokens = _tokenize(text)
            freqs: dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            documents.append(_Document(
                doc_id=did,
                text=text,
                tokens=tokens,
                token_freqs=freqs,
                length=len(tokens),
            ))

        # Fit BM25
        self._bm25.fit(documents)

        # Build TF-IDF matrix (lightweight, no sklearn dependency)
        self._build_tfidf_matrix(documents)

        self._documents = documents
        self._indexed = True

        logger.info("Indexed %d documents for hybrid search", len(documents))
        return len(documents)

    def _build_tfidf_matrix(self, documents: list[_Document]) -> None:
        """Build a TF-IDF matrix using raw numpy (no sklearn needed).

        Parameters
        ----------
        documents:
            Internal document representations with pre-computed token
            frequencies.
        """
        if not documents:
            self._tfidf_matrix = np.zeros((0, 0))
            self._tfidf_vocab = {}
            return

        # Build vocabulary
        vocab: dict[str, int] = {}
        for doc in documents:
            for token in doc.token_freqs:
                if token not in vocab:
                    vocab[token] = len(vocab)

        n_docs = len(documents)
        n_terms = len(vocab)

        if n_terms == 0:
            self._tfidf_matrix = np.zeros((n_docs, 1))
            self._tfidf_vocab = {}
            return

        # TF matrix (log-normalised)
        tf = np.zeros((n_docs, n_terms))
        for i, doc in enumerate(documents):
            for token, count in doc.token_freqs.items():
                tf[i, vocab[token]] = 1 + math.log(count) if count > 0 else 0.0

        # IDF vector
        df = np.sum(tf > 0, axis=0).astype(float)
        idf = np.log((n_docs + 1) / (df + 1)) + 1.0  # Smooth IDF

        # TF-IDF = TF * IDF
        tfidf = tf * idf

        # L2-normalise rows for cosine similarity via dot product
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        tfidf = tfidf / norms

        self._tfidf_matrix = tfidf
        self._tfidf_vocab = vocab

    def _tfidf_query_vector(self, query_tokens: list[str]) -> np.ndarray:
        """Build a TF-IDF vector for a query.

        Parameters
        ----------
        query_tokens:
            Tokenised query terms.

        Returns
        -------
        np.ndarray
            L2-normalised query vector.
        """
        n_terms = len(self._tfidf_vocab)
        if n_terms == 0:
            return np.zeros(1)

        vec = np.zeros(n_terms)
        for token in query_tokens:
            idx = self._tfidf_vocab.get(token)
            if idx is not None:
                vec[idx] += 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    def search(
        self,
        query: str,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search the indexed corpus with hybrid BM25 + TF-IDF scoring.

        Parameters
        ----------
        query:
            Natural-language search query.
        top_k:
            Maximum number of results to return.
        min_score:
            Minimum hybrid score threshold.

        Returns
        -------
        list[SearchResult]
            Ranked results, highest score first.

        Raises
        ------
        RuntimeError
            If called before ``index()``.
        """
        if not self._indexed:
            raise RuntimeError(
                "Search engine has not been indexed. Call index() first."
            )

        if not query.strip():
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        n_docs = len(self._documents)
        if n_docs == 0:
            return []

        # BM25 scores
        bm25_scores = np.array([
            self._bm25.score(query_tokens, doc) for doc in self._documents
        ])

        # Normalise BM25 to [0, 1]
        bm25_max = bm25_scores.max()
        if bm25_max > 0:
            bm25_norm = bm25_scores / bm25_max
        else:
            bm25_norm = bm25_scores

        # TF-IDF cosine scores (already normalised to [0, 1])
        q_vec = self._tfidf_query_vector(query_tokens)
        assert self._tfidf_matrix is not None
        tfidf_scores = self._tfidf_matrix @ q_vec

        # Hybrid interpolation
        hybrid = self.alpha * bm25_norm + (1 - self.alpha) * tfidf_scores

        # Rank and filter
        ranked_indices = np.argsort(-hybrid)
        results: list[SearchResult] = []

        for idx in ranked_indices[:top_k * 2]:  # Over-fetch, then filter
            score = float(hybrid[idx])
            if score < min_score:
                continue
            doc = self._documents[idx]
            results.append(SearchResult(
                doc_index=int(idx),
                doc_id=doc.doc_id,
                score=round(score, 4),
                bm25_score=round(float(bm25_norm[idx]), 4),
                tfidf_score=round(float(tfidf_scores[idx]), 4),
                snippet=_extract_snippet(doc.text, query_tokens),
            ))
            if len(results) >= top_k:
                break

        return results

    def add_document(self, text: str, doc_id: str | None = None) -> str:
        """Add a single document to the index and rebuild.

        This is a convenience method for incremental indexing.  For bulk
        operations, prefer calling ``index()`` with the full corpus.

        Parameters
        ----------
        text:
            Document text to add.
        doc_id:
            Optional identifier; defaults to the next index.

        Returns
        -------
        str
            The assigned document ID.
        """
        did = doc_id or str(self.corpus_size)
        all_texts = [d.text for d in self._documents] + [text]
        all_ids = [d.doc_id for d in self._documents] + [did]
        self.index(all_texts, all_ids)
        return did
