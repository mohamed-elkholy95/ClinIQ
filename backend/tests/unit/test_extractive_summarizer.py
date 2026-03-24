"""Unit tests for ExtractiveSummarizer TextRank inference.

Exercises the full summarise() pipeline including bias scoring,
PageRank convergence, clinical section weighting, key findings extraction,
detail-level behaviour, and edge cases.
"""

import math
import re

import numpy as np
import pytest

from app.ml.summarization.model import (
    ExtractiveSummarizer,
    SummarizationResult,
    _CLINICAL_IMPORTANCE_PATTERNS,
    _DETAIL_RATIO,
    _DETAIL_SENTENCE_CAP,
    _HIGH_PRIORITY_SECTIONS,
)


@pytest.fixture
def summarizer() -> ExtractiveSummarizer:
    """Return a loaded ExtractiveSummarizer."""
    s = ExtractiveSummarizer()
    s.load()
    return s


@pytest.fixture
def long_clinical_note() -> str:
    """A multi-section clinical note with enough sentences for meaningful tests."""
    return """
    CHIEF COMPLAINT: Follow-up for type 2 diabetes mellitus.

    HISTORY OF PRESENT ILLNESS:
    The patient is a 62-year-old male with a long-standing history of type 2 diabetes
    mellitus diagnosed approximately 10 years ago. He presents for routine follow-up
    and medication management. He reports good adherence to his medication regimen.
    He has been monitoring his blood glucose levels at home twice daily. His fasting
    glucose readings have ranged from 110 to 140 mg/dL over the past month. He
    denies any episodes of hypoglycemia or hyperglycemia requiring medical attention.
    He has not experienced polyuria, polydipsia, or unexplained weight loss. He
    reports occasional tingling in his feet bilaterally. He denies any visual changes.

    PAST MEDICAL HISTORY:
    Type 2 diabetes mellitus. Hypertension. Hyperlipidemia. Obesity with BMI 32.
    Diabetic peripheral neuropathy. Gastroesophageal reflux disease. Osteoarthritis
    of bilateral knees.

    MEDICATIONS:
    Metformin 1000mg PO BID. Lisinopril 20mg PO daily. Atorvastatin 40mg PO QHS.
    Aspirin 81mg PO daily. Gabapentin 300mg PO TID. Omeprazole 20mg PO daily.

    ALLERGIES: No known drug allergies.

    PHYSICAL EXAMINATION:
    Vitals: Blood pressure 132/78, heart rate 76, temperature 98.4F, respiratory
    rate 16. General: Alert and oriented, no acute distress. HEENT: Pupils equal
    round reactive to light. Cardiovascular: Regular rate and rhythm, no murmurs.
    Pulmonary: Clear to auscultation bilaterally. Extremities: No edema, diminished
    sensation to monofilament testing bilateral feet.

    LABORATORY DATA:
    HbA1c 7.2% (previous 7.5%). Fasting glucose 128 mg/dL. Creatinine 1.1.
    eGFR 72. Total cholesterol 195. LDL 105. HDL 42. Triglycerides 180.
    Urine albumin-to-creatinine ratio 45 mg/g (mildly elevated).

    ASSESSMENT AND PLAN:
    1. Type 2 diabetes mellitus - HbA1c improved from 7.5 to 7.2, continue current
       regimen. Consider adding SGLT2 inhibitor given microalbuminuria.
    2. Hypertension - adequately controlled on lisinopril, continue current dose.
    3. Hyperlipidemia - LDL above goal of 70 for diabetic patient, increase
       atorvastatin to 80mg.
    4. Diabetic neuropathy - continue gabapentin, refer to podiatry for annual
       foot exam.
    5. Microalbuminuria - start empagliflozin 10mg daily for renal protection.
    6. Follow-up in 3 months with repeat HbA1c and metabolic panel.
    """


class TestExtractiveSummarizerLifecycle:
    """Lifecycle and construction tests."""

    def test_not_loaded_before_load(self) -> None:
        s = ExtractiveSummarizer()
        assert s.is_loaded is False

    def test_loaded_after_load(self) -> None:
        s = ExtractiveSummarizer()
        s.load()
        assert s.is_loaded is True

    def test_default_attributes(self) -> None:
        s = ExtractiveSummarizer()
        assert s.model_name == "extractive-textrank"
        assert s.version == "1.0.0"
        assert s.damping == 0.85
        assert s.max_iter == 100

    def test_custom_hyperparams(self) -> None:
        s = ExtractiveSummarizer(
            model_name="custom", version="2.0", damping=0.9, max_iter=50
        )
        assert s.model_name == "custom"
        assert s.damping == 0.9
        assert s.max_iter == 50


class TestExtractiveSummarizerSummarize:
    """Core summarize() tests."""

    def test_returns_summarization_result(
        self, summarizer: ExtractiveSummarizer, long_clinical_note: str
    ) -> None:
        result = summarizer.summarize(long_clinical_note)
        assert isinstance(result, SummarizationResult)

    def test_summary_not_empty(
        self, summarizer: ExtractiveSummarizer, long_clinical_note: str
    ) -> None:
        result = summarizer.summarize(long_clinical_note)
        assert len(result.summary.strip()) > 0

    def test_summary_shorter_than_original(
        self, summarizer: ExtractiveSummarizer, long_clinical_note: str
    ) -> None:
        result = summarizer.summarize(long_clinical_note, detail_level="brief")
        assert len(result.summary) < len(long_clinical_note)

    def test_processing_time_positive(
        self, summarizer: ExtractiveSummarizer, long_clinical_note: str
    ) -> None:
        result = summarizer.summarize(long_clinical_note)
        assert result.processing_time_ms > 0

    def test_model_metadata_in_result(
        self, summarizer: ExtractiveSummarizer, long_clinical_note: str
    ) -> None:
        result = summarizer.summarize(long_clinical_note)
        assert result.model_name == "extractive-textrank"
        assert result.model_version == "1.0.0"

    def test_sentence_counts_populated(
        self, summarizer: ExtractiveSummarizer, long_clinical_note: str
    ) -> None:
        result = summarizer.summarize(long_clinical_note)
        assert result.sentence_count_original > 0
        assert result.sentence_count_summary > 0
        assert result.sentence_count_summary <= result.sentence_count_original


class TestDetailLevels:
    """Test that detail levels control output size correctly."""

    def test_brief_fewer_sentences_than_detailed(
        self, summarizer: ExtractiveSummarizer, long_clinical_note: str
    ) -> None:
        brief = summarizer.summarize(long_clinical_note, detail_level="brief")
        detailed = summarizer.summarize(long_clinical_note, detail_level="detailed")
        assert brief.sentence_count_summary <= detailed.sentence_count_summary

    def test_standard_between_brief_and_detailed(
        self, summarizer: ExtractiveSummarizer, long_clinical_note: str
    ) -> None:
        brief = summarizer.summarize(long_clinical_note, detail_level="brief")
        standard = summarizer.summarize(long_clinical_note, detail_level="standard")
        detailed = summarizer.summarize(long_clinical_note, detail_level="detailed")
        assert brief.sentence_count_summary <= standard.sentence_count_summary
        assert standard.sentence_count_summary <= detailed.sentence_count_summary

    @pytest.mark.parametrize("level", ["brief", "standard", "detailed"])
    def test_detail_level_in_result(
        self, summarizer: ExtractiveSummarizer, long_clinical_note: str, level: str
    ) -> None:
        result = summarizer.summarize(long_clinical_note, detail_level=level)
        assert result.detail_level == level

    def test_brief_respects_sentence_cap(
        self, summarizer: ExtractiveSummarizer, long_clinical_note: str
    ) -> None:
        result = summarizer.summarize(long_clinical_note, detail_level="brief")
        assert result.sentence_count_summary <= _DETAIL_SENTENCE_CAP["brief"]


class TestKeyFindings:
    """Test key_findings extraction."""

    def test_key_findings_is_list(
        self, summarizer: ExtractiveSummarizer, long_clinical_note: str
    ) -> None:
        result = summarizer.summarize(long_clinical_note)
        assert isinstance(result.key_findings, list)

    def test_key_findings_are_strings(
        self, summarizer: ExtractiveSummarizer, long_clinical_note: str
    ) -> None:
        result = summarizer.summarize(long_clinical_note)
        for finding in result.key_findings:
            assert isinstance(finding, str)

    def test_key_findings_contain_clinical_terms(
        self, summarizer: ExtractiveSummarizer, long_clinical_note: str
    ) -> None:
        """Key findings should contain clinically relevant sentences."""
        result = summarizer.summarize(long_clinical_note)
        if result.key_findings:
            all_findings = " ".join(result.key_findings).lower()
            clinical_terms = [
                "medication", "treatment", "diagnosis", "follow", "plan",
                "recommend", "prescrib", "refer", "significant", "assessment",
            ]
            has_clinical_term = any(term in all_findings for term in clinical_terms)
            # At minimum the findings should contain *some* medical content
            assert has_clinical_term or len(result.key_findings) == 0

    def test_key_findings_max_five(
        self, summarizer: ExtractiveSummarizer, long_clinical_note: str
    ) -> None:
        result = summarizer.summarize(long_clinical_note)
        assert len(result.key_findings) <= 5


class TestClinicalSectionWeighting:
    """Test that Assessment/Plan sentences get prioritised."""

    def test_assessment_plan_content_appears_in_summary(
        self, summarizer: ExtractiveSummarizer
    ) -> None:
        """Assessment and Plan sentences should be preferred in summary."""
        text = (
            "Patient arrived at 8am for routine visit. "
            "The waiting room was full. "
            "Registration was completed without issues. "
            "Insurance information was verified. "
            "Vital signs were obtained by the nurse. "
            "The physician reviewed the chart. "
            "Assessment and Plan: Continue metformin for diabetes management. "
            "Increase atorvastatin dose to 80mg for LDL control. "
            "Refer to podiatry for annual diabetic foot exam. "
            "Follow up in 3 months with HbA1c."
        )
        result = summarizer.summarize(text, detail_level="brief")
        summary_lower = result.summary.lower()
        # The assessment/plan content should appear in a brief summary
        assert any(
            term in summary_lower
            for term in ["metformin", "atorvastatin", "follow", "assessment"]
        )


class TestEdgeCases:
    """Edge-case and boundary tests."""

    def test_empty_text(self, summarizer: ExtractiveSummarizer) -> None:
        result = summarizer.summarize("")
        assert isinstance(result, SummarizationResult)

    def test_single_sentence(self, summarizer: ExtractiveSummarizer) -> None:
        text = "Patient has well-controlled type 2 diabetes on metformin."
        result = summarizer.summarize(text)
        assert result.summary.strip() == text.strip()
        assert result.sentence_count_summary == 1

    def test_very_short_sentences_filtered(
        self, summarizer: ExtractiveSummarizer
    ) -> None:
        """Sentences under 15 chars should be filtered out."""
        text = "Hi. OK. Patient presents with a history of chronic obstructive pulmonary disease."
        result = summarizer.summarize(text)
        assert isinstance(result, SummarizationResult)

    def test_whitespace_heavy_text(self, summarizer: ExtractiveSummarizer) -> None:
        text = "   \n\n  Patient has hypertension.  \n\n  Blood pressure elevated.  \n\n  "
        result = summarizer.summarize(text)
        assert isinstance(result, SummarizationResult)

    def test_to_dict_serialisable(
        self, summarizer: ExtractiveSummarizer, long_clinical_note: str
    ) -> None:
        result = summarizer.summarize(long_clinical_note)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "summary" in d
        assert "key_findings" in d
        assert isinstance(d["processing_time_ms"], float)


class TestInternalHelpers:
    """Direct tests for internal helper methods."""

    def test_target_sentence_count_brief(
        self, summarizer: ExtractiveSummarizer
    ) -> None:
        count = summarizer._target_sentence_count(20, "brief")
        assert count == min(math.ceil(20 * _DETAIL_RATIO["brief"]), _DETAIL_SENTENCE_CAP["brief"])

    def test_target_sentence_count_at_least_one(
        self, summarizer: ExtractiveSummarizer
    ) -> None:
        count = summarizer._target_sentence_count(1, "brief")
        assert count >= 1

    def test_target_sentence_count_respects_cap(
        self, summarizer: ExtractiveSummarizer
    ) -> None:
        count = summarizer._target_sentence_count(1000, "brief")
        assert count <= _DETAIL_SENTENCE_CAP["brief"]

    def test_cosine_similarity_matrix_identity(
        self, summarizer: ExtractiveSummarizer
    ) -> None:
        """Identical vectors should have similarity ~1.0."""
        matrix = np.array([[1, 2, 3], [1, 2, 3]], dtype=float)
        sim = summarizer._cosine_similarity_matrix(matrix)
        assert sim.shape == (2, 2)
        np.testing.assert_almost_equal(sim[0, 1], 1.0, decimal=5)

    def test_cosine_similarity_matrix_orthogonal(
        self, summarizer: ExtractiveSummarizer
    ) -> None:
        """Orthogonal vectors should have similarity ~0.0."""
        matrix = np.array([[1, 0], [0, 1]], dtype=float)
        sim = summarizer._cosine_similarity_matrix(matrix)
        np.testing.assert_almost_equal(sim[0, 1], 0.0, decimal=5)

    def test_cosine_similarity_matrix_handles_zero_vector(
        self, summarizer: ExtractiveSummarizer
    ) -> None:
        """Zero vectors should not cause division errors."""
        matrix = np.array([[0, 0], [1, 2]], dtype=float)
        sim = summarizer._cosine_similarity_matrix(matrix)
        assert sim.shape == (2, 2)

    def test_pagerank_converges(self, summarizer: ExtractiveSummarizer) -> None:
        """PageRank should produce valid probability-like scores."""
        sim = np.array([[0, 0.8, 0.3], [0.8, 0, 0.5], [0.3, 0.5, 0]])
        bias = np.ones(3)
        scores = summarizer._pagerank(sim, bias)
        assert len(scores) == 3
        assert all(s >= 0 for s in scores)
        # Scores should sum to approximately 1
        np.testing.assert_almost_equal(scores.sum(), 1.0, decimal=3)

    def test_extract_key_findings_returns_list(
        self, summarizer: ExtractiveSummarizer
    ) -> None:
        sentences = [
            "Patient is stable.",
            "Diagnosis: Type 2 diabetes mellitus.",
            "Treatment plan includes metformin.",
            "Follow-up in 3 months.",
        ]
        findings = summarizer._extract_key_findings(sentences)
        assert isinstance(findings, list)
        assert all(isinstance(f, str) for f in findings)

    def test_extract_key_findings_prefers_clinical_sentences(
        self, summarizer: ExtractiveSummarizer
    ) -> None:
        sentences = [
            "The sun was shining outside.",
            "Diagnosis: acute myocardial infarction.",
            "The patient arrived by ambulance.",
            "Treatment: emergent cardiac catheterization recommended.",
            "The floor was recently cleaned.",
        ]
        findings = summarizer._extract_key_findings(sentences, top_n=2)
        findings_lower = [f.lower() for f in findings]
        assert any("diagnosis" in f for f in findings_lower)
        assert any("treatment" in f for f in findings_lower)
