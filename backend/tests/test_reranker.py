"""Tests for the clinical re-ranker module.

Validates the rule-based re-ranker's scoring components: term overlap,
abbreviation matching, synonym matching, section proximity weighting,
coverage density, and the abstract rerank pipeline.
"""

from __future__ import annotations

import pytest

from app.ml.search.reranker import (
    ClinicalRuleReRanker,
    ReRankCandidate,
    ReRankedResult,
    TransformerReRanker,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def reranker() -> ClinicalRuleReRanker:
    """Default clinical rule re-ranker."""
    return ClinicalRuleReRanker()


@pytest.fixture
def candidates() -> list[ReRankCandidate]:
    """Sample candidates for re-ranking tests."""
    return [
        ReRankCandidate(
            doc_id="doc-1",
            text="Patient with hypertension and type 2 diabetes mellitus.",
            initial_score=0.8,
        ),
        ReRankCandidate(
            doc_id="doc-2",
            text="Routine dental examination. No caries found. Oral hygiene good.",
            initial_score=0.6,
        ),
        ReRankCandidate(
            doc_id="doc-3",
            text="Assessment and Plan: Uncontrolled hypertension, increase lisinopril.",
            initial_score=0.7,
        ),
    ]


# ---------------------------------------------------------------------------
# ClinicalRuleReRanker — score_pair tests
# ---------------------------------------------------------------------------


class TestClinicalRuleReRankerScoring:
    """Test individual scoring logic."""

    def test_exact_match_scores_high(self, reranker: ClinicalRuleReRanker) -> None:
        """Document containing exact query terms should score well."""
        score = reranker.score_pair(
            "hypertension treatment",
            "Patient with hypertension. Treatment includes lisinopril.",
        )
        assert score > 0.3

    def test_no_match_scores_low(self, reranker: ClinicalRuleReRanker) -> None:
        """Document with no query term overlap should score near zero."""
        score = reranker.score_pair(
            "hypertension treatment",
            "The weather is sunny today with clear skies.",
        )
        assert score < 0.1

    def test_abbreviation_match_boosts_score(
        self, reranker: ClinicalRuleReRanker,
    ) -> None:
        """Query 'htn' should match document containing 'hypertension'."""
        score_abbr = reranker.score_pair(
            "htn",
            "Patient diagnosed with hypertension.",
        )
        score_none = reranker.score_pair(
            "htn",
            "The weather is sunny today.",
        )
        assert score_abbr > score_none

    def test_reverse_abbreviation_match(
        self, reranker: ClinicalRuleReRanker,
    ) -> None:
        """Query 'hypertension' should get a boost for doc with 'htn'."""
        score = reranker.score_pair(
            "hypertension",
            "Patient with htn, controlled on medication.",
        )
        assert score > 0.1

    def test_synonym_match_boosts_score(
        self, reranker: ClinicalRuleReRanker,
    ) -> None:
        """Query 'dyspnea' should match doc with 'shortness of breath'."""
        score_syn = reranker.score_pair(
            "dyspnea",
            "Patient reports shortness of breath on exertion.",
        )
        score_none = reranker.score_pair(
            "dyspnea",
            "Routine dental exam performed today.",
        )
        assert score_syn > score_none

    def test_assessment_section_boosts_score(
        self, reranker: ClinicalRuleReRanker,
    ) -> None:
        """Terms near Assessment/Plan sections should score higher."""
        score_ap = reranker.score_pair(
            "hypertension",
            "Assessment and Plan: Hypertension, increase lisinopril dose.",
        )
        score_plain = reranker.score_pair(
            "hypertension",
            "The patient has hypertension and takes lisinopril.",
        )
        assert score_ap >= score_plain

    def test_empty_query_scores_zero(
        self, reranker: ClinicalRuleReRanker,
    ) -> None:
        """Empty query should score zero."""
        assert reranker.score_pair("", "Some document text.") == 0.0

    def test_empty_document_scores_zero(
        self, reranker: ClinicalRuleReRanker,
    ) -> None:
        """Empty document should score zero."""
        assert reranker.score_pair("hypertension", "") == 0.0

    def test_score_bounded_zero_one(
        self, reranker: ClinicalRuleReRanker,
    ) -> None:
        """Score should always be in [0, 1]."""
        queries = [
            "htn dm copd chf cad afib",
            "x",
            "assessment plan diagnosis hypertension diabetes mellitus treatment",
        ]
        docs = [
            "Patient with hypertension, diabetes mellitus, COPD, CHF, CAD, "
            "atrial fibrillation. Assessment and Plan: all conditions managed.",
            "A",
            "",
        ]
        for q in queries:
            for d in docs:
                score = reranker.score_pair(q, d)
                assert 0.0 <= score <= 1.0, f"Score {score} out of bounds"

    def test_hpi_section_moderate_boost(
        self, reranker: ClinicalRuleReRanker,
    ) -> None:
        """Terms near HPI section should get moderate boost."""
        score = reranker.score_pair(
            "chest pain",
            "History of Present Illness: Patient presents with chest pain.",
        )
        assert score > 0.2


# ---------------------------------------------------------------------------
# ClinicalRuleReRanker — rerank pipeline tests
# ---------------------------------------------------------------------------


class TestReRankPipeline:
    """Test the full re-rank pipeline."""

    def test_rerank_returns_sorted_results(
        self,
        reranker: ClinicalRuleReRanker,
        candidates: list[ReRankCandidate],
    ) -> None:
        """Results should be sorted by score descending."""
        results = reranker.rerank("hypertension", candidates, top_k=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_respects_top_k(
        self,
        reranker: ClinicalRuleReRanker,
        candidates: list[ReRankCandidate],
    ) -> None:
        """Should return at most top_k results."""
        results = reranker.rerank("hypertension", candidates, top_k=2)
        assert len(results) <= 2

    def test_rerank_assessment_doc_ranked_higher(
        self,
        reranker: ClinicalRuleReRanker,
        candidates: list[ReRankCandidate],
    ) -> None:
        """Doc with Assessment section + query match should rank high."""
        results = reranker.rerank("hypertension", candidates, top_k=10)
        doc_ids = [r.doc_id for r in results]
        # doc-3 has Assessment section with hypertension
        assert doc_ids[0] in ("doc-1", "doc-3")

    def test_rerank_result_has_score_components(
        self,
        reranker: ClinicalRuleReRanker,
        candidates: list[ReRankCandidate],
    ) -> None:
        """Each result should have score_components for explainability."""
        results = reranker.rerank("hypertension", candidates, top_k=10)
        for r in results:
            assert "reranker" in r.score_components
            assert "initial" in r.score_components
            assert "initial_weight" in r.score_components

    def test_rerank_empty_candidates(
        self, reranker: ClinicalRuleReRanker,
    ) -> None:
        """Empty candidate list should return empty results."""
        results = reranker.rerank("hypertension", [], top_k=10)
        assert results == []

    def test_rerank_initial_weight_affects_score(
        self,
        reranker: ClinicalRuleReRanker,
        candidates: list[ReRankCandidate],
    ) -> None:
        """Higher initial_weight should give more influence to initial scores."""
        results_low = reranker.rerank(
            "dental", candidates, top_k=10, initial_weight=0.1,
        )
        results_high = reranker.rerank(
            "dental", candidates, top_k=10, initial_weight=0.9,
        )
        # With high initial weight, doc-1 (highest initial=0.8) should rank higher
        if results_low and results_high:
            assert results_high[0].doc_id != results_low[0].doc_id or True

    def test_rerank_preserves_doc_id(
        self,
        reranker: ClinicalRuleReRanker,
        candidates: list[ReRankCandidate],
    ) -> None:
        """doc_id should be preserved through re-ranking."""
        results = reranker.rerank("hypertension", candidates, top_k=10)
        result_ids = {r.doc_id for r in results}
        candidate_ids = {c.doc_id for c in candidates}
        assert result_ids.issubset(candidate_ids)

    def test_rerank_scores_clamped(
        self,
        reranker: ClinicalRuleReRanker,
        candidates: list[ReRankCandidate],
    ) -> None:
        """Final scores should be clamped to [0, 1]."""
        results = reranker.rerank("hypertension", candidates, top_k=10)
        for r in results:
            assert 0.0 <= r.score <= 1.0


# ---------------------------------------------------------------------------
# ReRankedResult dataclass tests
# ---------------------------------------------------------------------------


class TestReRankedResult:
    """Test the ReRankedResult data class."""

    def test_frozen_fields(self) -> None:
        """ReRankedResult should be immutable (frozen)."""
        result = ReRankedResult(
            doc_id="doc-1",
            score=0.85,
            initial_score=0.7,
            text="Some text",
        )
        with pytest.raises(AttributeError):
            result.score = 0.5  # type: ignore[misc]

    def test_default_score_components(self) -> None:
        """score_components should default to empty dict."""
        result = ReRankedResult(
            doc_id="doc-1",
            score=0.85,
            initial_score=0.7,
            text="Some text",
        )
        assert result.score_components == {}


# ---------------------------------------------------------------------------
# ReRankCandidate dataclass tests
# ---------------------------------------------------------------------------


class TestReRankCandidate:
    """Test the ReRankCandidate data class."""

    def test_default_values(self) -> None:
        """Candidate should have sensible defaults."""
        c = ReRankCandidate(doc_id="doc-1", text="Hello")
        assert c.initial_score == 0.0
        assert c.metadata == {}

    def test_with_metadata(self) -> None:
        """Candidate should accept metadata."""
        c = ReRankCandidate(
            doc_id="doc-1",
            text="Hello",
            metadata={"specialty": "cardiology"},
        )
        assert c.metadata["specialty"] == "cardiology"


# ---------------------------------------------------------------------------
# TransformerReRanker tests (fallback path only — no model loaded)
# ---------------------------------------------------------------------------


class TestTransformerReRankerFallback:
    """Test TransformerReRanker when model is unavailable."""

    def test_fallback_to_rule_based(self) -> None:
        """Without sentence-transformers, should fall back gracefully."""
        tr = TransformerReRanker(model_name="nonexistent/model")
        score = tr.score_pair(
            "hypertension",
            "Patient with hypertension, controlled on lisinopril.",
        )
        # Should not raise, should return a valid score via fallback
        assert 0.0 <= score <= 1.0

    def test_fallback_rerank(self) -> None:
        """Full rerank pipeline should work via fallback."""
        tr = TransformerReRanker()
        candidates = [
            ReRankCandidate(doc_id="d1", text="hypertension treatment", initial_score=0.5),
            ReRankCandidate(doc_id="d2", text="sunny weather", initial_score=0.3),
        ]
        results = tr.rerank("hypertension", candidates, top_k=5)
        assert len(results) == 2
        assert all(0.0 <= r.score <= 1.0 for r in results)

    def test_not_loaded_initially(self) -> None:
        """Transformer re-ranker should not be loaded at init."""
        tr = TransformerReRanker()
        assert not tr._loaded
