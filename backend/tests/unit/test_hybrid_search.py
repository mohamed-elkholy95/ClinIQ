"""Tests for the hybrid BM25 + TF-IDF search engine.

Covers indexing, BM25 scoring, TF-IDF cosine similarity, hybrid
interpolation, snippet extraction, edge cases, and incremental indexing.
"""

from __future__ import annotations

import pytest

from app.ml.search.hybrid import (
    HybridSearchEngine,
    SearchResult,
    _BM25,
    _Document,
    _extract_snippet,
    _tokenize,
)


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------


class TestTokenize:
    """Tests for the _tokenize helper."""

    def test_basic_tokenization(self) -> None:
        """Splits text into lowercase tokens."""
        tokens = _tokenize("Patient has Type 2 Diabetes")
        assert "type" in tokens
        assert "diabetes" in tokens

    def test_removes_medical_stopwords(self) -> None:
        """Filters out medical stopwords."""
        tokens = _tokenize("patient history examination")
        assert "patient" not in tokens
        assert "history" not in tokens

    def test_removes_single_char_tokens(self) -> None:
        """Filters tokens with length <= 1."""
        tokens = _tokenize("a b c diabetes")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "diabetes" in tokens

    def test_empty_text(self) -> None:
        """Returns empty list for empty input."""
        assert _tokenize("") == []
        assert _tokenize("   ") == []

    def test_only_stopwords(self) -> None:
        """Returns empty list when all tokens are stopwords."""
        assert _tokenize("patient history mg ml") == []

    def test_preserves_numbers(self) -> None:
        """Keeps numeric tokens longer than 1 char."""
        tokens = _tokenize("ICD-10 code E11.9")
        assert "10" in tokens
        assert "icd" in tokens


# ---------------------------------------------------------------------------
# Snippet extraction tests
# ---------------------------------------------------------------------------


class TestExtractSnippet:
    """Tests for the _extract_snippet helper."""

    def test_match_found(self) -> None:
        """Centres snippet around the first query term match."""
        text = "A" * 100 + " diabetes " + "B" * 100
        snippet = _extract_snippet(text, ["diabetes"], max_len=40)
        assert "diabetes" in snippet.lower()

    def test_no_match(self) -> None:
        """Returns start of document when no match found."""
        text = "This is a clinical note about hypertension."
        snippet = _extract_snippet(text, ["xyz123"], max_len=20)
        assert snippet.startswith("This")

    def test_short_text(self) -> None:
        """Returns full text without truncation markers when short."""
        text = "Short note."
        snippet = _extract_snippet(text, ["short"], max_len=200)
        assert "…" not in snippet or snippet == text

    def test_truncation_markers(self) -> None:
        """Adds ellipsis markers when truncated."""
        text = "A" * 100 + " diabetes " + "B" * 100
        snippet = _extract_snippet(text, ["diabetes"], max_len=30)
        assert "…" in snippet

    def test_empty_query_tokens(self) -> None:
        """Handles empty query token list gracefully."""
        snippet = _extract_snippet("Some text here", [], max_len=50)
        assert len(snippet) > 0


# ---------------------------------------------------------------------------
# BM25 scorer tests
# ---------------------------------------------------------------------------


class TestBM25:
    """Tests for the internal BM25 implementation."""

    def _make_doc(self, text: str, doc_id: str = "d") -> _Document:
        """Create an internal document from text."""
        tokens = _tokenize(text)
        freqs: dict[str, int] = {}
        for t in tokens:
            freqs[t] = freqs.get(t, 0) + 1
        return _Document(doc_id=doc_id, text=text, tokens=tokens,
                         token_freqs=freqs, length=len(tokens))

    def test_single_doc_score(self) -> None:
        """A matching query scores higher than zero."""
        bm25 = _BM25()
        doc = self._make_doc("Type 2 diabetes mellitus with complications")
        bm25.fit([doc])
        score = bm25.score(["diabetes"], doc)
        assert score > 0

    def test_no_match_score_zero(self) -> None:
        """Non-matching query scores zero."""
        bm25 = _BM25()
        doc = self._make_doc("Routine dental examination")
        bm25.fit([doc])
        score = bm25.score(["cardiac"], doc)
        assert score == 0.0

    def test_idf_higher_for_rare_terms(self) -> None:
        """Rare terms get higher IDF than common ones."""
        bm25 = _BM25()
        docs = [
            self._make_doc("diabetes mellitus complications"),
            self._make_doc("diabetes treatment insulin"),
            self._make_doc("cardiac arrest emergency"),
        ]
        bm25.fit(docs)
        # "cardiac" appears in 1 doc, "diabetes" in 2 — cardiac has higher IDF
        assert bm25._idf.get("cardiac", 0) > bm25._idf.get("diabetes", 0)

    def test_empty_corpus(self) -> None:
        """Fitting on empty corpus doesn't crash."""
        bm25 = _BM25()
        bm25.fit([])
        assert bm25._n_docs == 0


# ---------------------------------------------------------------------------
# HybridSearchEngine tests
# ---------------------------------------------------------------------------


class TestHybridSearchEngine:
    """Tests for the combined hybrid search engine."""

    @pytest.fixture()
    def clinical_corpus(self) -> list[tuple[str, str]]:
        """A small clinical document corpus for testing."""
        return [
            ("doc-1", "Patient presents with uncontrolled type 2 diabetes mellitus. "
                       "HbA1c of 9.2 percent. Started metformin 1000mg twice daily."),
            ("doc-2", "Routine dental examination reveals mild gingivitis. "
                       "Periodontal probing depths within normal limits."),
            ("doc-3", "Chest X-ray shows bilateral infiltrates consistent with "
                       "community-acquired pneumonia. Started on azithromycin."),
            ("doc-4", "Follow-up visit for hypertension. Blood pressure 142/88. "
                       "Adjusted lisinopril dose from 10mg to 20mg daily."),
            ("doc-5", "Type 1 diabetes with diabetic retinopathy. Referred to "
                       "ophthalmology for laser photocoagulation treatment."),
        ]

    @pytest.fixture()
    def engine(self, clinical_corpus: list[tuple[str, str]]) -> HybridSearchEngine:
        """Pre-indexed search engine."""
        eng = HybridSearchEngine(alpha=0.5)
        doc_ids = [d[0] for d in clinical_corpus]
        texts = [d[1] for d in clinical_corpus]
        eng.index(texts, doc_ids)
        return eng

    def test_index_corpus_size(self, engine: HybridSearchEngine) -> None:
        """Index stores all documents."""
        assert engine.corpus_size == 5

    def test_search_before_index_raises(self) -> None:
        """Searching without indexing raises RuntimeError."""
        eng = HybridSearchEngine()
        with pytest.raises(RuntimeError, match="not been indexed"):
            eng.search("test")

    def test_diabetes_query_returns_relevant_docs(
        self, engine: HybridSearchEngine,
    ) -> None:
        """Diabetes query should rank diabetes documents first."""
        results = engine.search("diabetes mellitus treatment")
        assert len(results) > 0
        # Top results should be doc-1 or doc-5 (diabetes docs)
        top_ids = {r.doc_id for r in results[:2]}
        assert top_ids & {"doc-1", "doc-5"}

    def test_dental_query(self, engine: HybridSearchEngine) -> None:
        """Dental query should prioritise the dental document."""
        results = engine.search("dental gingivitis periodontal")
        assert results[0].doc_id == "doc-2"

    def test_pneumonia_query(self, engine: HybridSearchEngine) -> None:
        """Pneumonia query matches the chest X-ray document."""
        results = engine.search("pneumonia chest infiltrates")
        assert results[0].doc_id == "doc-3"

    def test_top_k_limits_results(self, engine: HybridSearchEngine) -> None:
        """top_k parameter caps the number of results."""
        results = engine.search("diabetes", top_k=2)
        assert len(results) <= 2

    def test_min_score_filters(self, engine: HybridSearchEngine) -> None:
        """min_score filters out low-relevance results."""
        results = engine.search("diabetes", min_score=0.99)
        # Very high threshold should return few or no results
        assert len(results) <= 1

    def test_result_has_all_fields(self, engine: HybridSearchEngine) -> None:
        """Search results contain all expected fields."""
        results = engine.search("diabetes")
        assert len(results) > 0
        r = results[0]
        assert isinstance(r.doc_id, str)
        assert isinstance(r.score, float)
        assert isinstance(r.bm25_score, float)
        assert isinstance(r.tfidf_score, float)
        assert isinstance(r.snippet, str)
        assert 0.0 <= r.score <= 1.0
        assert 0.0 <= r.bm25_score <= 1.0
        assert 0.0 <= r.tfidf_score <= 1.0

    def test_results_sorted_by_score(self, engine: HybridSearchEngine) -> None:
        """Results are in descending score order."""
        results = engine.search("diabetes treatment")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query_returns_empty(self, engine: HybridSearchEngine) -> None:
        """Empty query string returns no results."""
        assert engine.search("") == []
        assert engine.search("   ") == []

    def test_stopword_only_query(self, engine: HybridSearchEngine) -> None:
        """Query with only stopwords returns empty."""
        assert engine.search("patient history mg") == []

    def test_alpha_pure_bm25(
        self, clinical_corpus: list[tuple[str, str]],
    ) -> None:
        """Alpha=1.0 uses only BM25 scoring."""
        eng = HybridSearchEngine(alpha=1.0)
        eng.index([t for _, t in clinical_corpus], [d for d, _ in clinical_corpus])
        results = eng.search("diabetes")
        # With pure BM25, tfidf_score should not affect ranking
        assert len(results) > 0

    def test_alpha_pure_tfidf(
        self, clinical_corpus: list[tuple[str, str]],
    ) -> None:
        """Alpha=0.0 uses only TF-IDF scoring."""
        eng = HybridSearchEngine(alpha=0.0)
        eng.index([t for _, t in clinical_corpus], [d for d, _ in clinical_corpus])
        results = eng.search("diabetes")
        assert len(results) > 0

    def test_invalid_alpha_raises(self) -> None:
        """Alpha outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="alpha"):
            HybridSearchEngine(alpha=1.5)
        with pytest.raises(ValueError, match="alpha"):
            HybridSearchEngine(alpha=-0.1)

    def test_doc_ids_length_mismatch_raises(self) -> None:
        """Mismatched doc_ids and texts lengths raises ValueError."""
        eng = HybridSearchEngine()
        with pytest.raises(ValueError, match="doc_ids length"):
            eng.index(["text1", "text2"], ["id1"])

    def test_empty_corpus_index(self) -> None:
        """Indexing empty corpus works and returns 0."""
        eng = HybridSearchEngine()
        count = eng.index([], [])
        assert count == 0
        assert eng.corpus_size == 0

    def test_add_document_incremental(self, engine: HybridSearchEngine) -> None:
        """add_document extends the corpus."""
        original_size = engine.corpus_size
        new_id = engine.add_document(
            "Acute myocardial infarction with ST elevation",
            doc_id="doc-6",
        )
        assert new_id == "doc-6"
        assert engine.corpus_size == original_size + 1
        results = engine.search("myocardial infarction")
        assert any(r.doc_id == "doc-6" for r in results)

    def test_add_document_auto_id(self, engine: HybridSearchEngine) -> None:
        """add_document assigns auto-generated ID when none provided."""
        new_id = engine.add_document("Some new clinical note")
        assert new_id == str(engine.corpus_size - 1)

    def test_reindex_replaces_corpus(self) -> None:
        """Calling index() again replaces the entire corpus."""
        eng = HybridSearchEngine()
        eng.index(["old text about asthma"], ["old-1"])
        assert eng.corpus_size == 1
        eng.index(["new text about fracture", "another note"], ["new-1", "new-2"])
        assert eng.corpus_size == 2
        results = eng.search("asthma")
        assert len(results) == 0 or all(r.score < 0.01 for r in results)


# ---------------------------------------------------------------------------
# SearchResult dataclass tests
# ---------------------------------------------------------------------------


class TestSearchResult:
    """Tests for the SearchResult data class."""

    def test_frozen(self) -> None:
        """SearchResult instances are immutable."""
        r = SearchResult(
            doc_index=0, doc_id="d1", score=0.9,
            bm25_score=0.8, tfidf_score=0.7, snippet="test",
        )
        with pytest.raises(AttributeError):
            r.score = 0.5  # type: ignore[misc]

    def test_default_snippet(self) -> None:
        """Default snippet is empty string."""
        r = SearchResult(
            doc_index=0, doc_id="d1", score=0.5,
            bm25_score=0.3, tfidf_score=0.2,
        )
        assert r.snippet == ""
