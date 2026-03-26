"""Tests for ClinIQ SDK data models.

Validates dataclass construction, defaults, and the ``from_dict``
factories that transform raw API JSON into typed model instances.
Covers all 20+ model types across core analysis and specialized modules.
"""

import pytest

from cliniq_client.models import (
    AbbreviationMatch,
    AbbreviationResult,
    Allergy,
    AllergyResult,
    AnalysisResult,
    AssertionResult,
    AUPRCResult,
    BatchJob,
    ClassificationEvalResult,
    ClassificationResult,
    ClassificationScore,
    ComorbidityResult,
    ConversationContext,
    ConversationSessionInfo,
    ConversationStats,
    ConversationTurnResult,
    EnhancedAnalysisResult,
    Entity,
    ICDEvalResult,
    ICDPrediction,
    KappaResult,
    MatchedCategory,
    Medication,
    MedicationResult,
    NEREvalResult,
    NormalizationResult,
    QualityDimension,
    QualityReport,
    Relation,
    RelationResult,
    RiskAssessment,
    RiskFactor,
    ROUGEEvalResult,
    ROUGEScores,
    SDoHExtraction,
    SDoHResult,
    SearchHit,
    SearchResult,
    Section,
    SectionResult,
    Summary,
    VitalSign,
    VitalSignResult,
)


# ===================================================================
# Core analysis models
# ===================================================================


class TestEntity:
    """Tests for the Entity dataclass."""

    def test_required_fields(self) -> None:
        entity = Entity(
            text="metformin", entity_type="MEDICATION",
            start_char=10, end_char=19, confidence=0.95,
        )
        assert entity.text == "metformin"
        assert entity.entity_type == "MEDICATION"
        assert entity.confidence == 0.95

    def test_defaults(self) -> None:
        entity = Entity("x", "T", 0, 1, 0.5)
        assert entity.normalized_text is None
        assert entity.umls_cui is None
        assert entity.is_negated is False
        assert entity.is_uncertain is False


class TestICDPrediction:
    def test_minimal(self) -> None:
        pred = ICDPrediction(code="I21.9", description="AMI", confidence=0.87)
        assert pred.code == "I21.9"
        assert pred.chapter is None

    def test_full(self) -> None:
        pred = ICDPrediction(
            code="E11.9", description="T2DM", confidence=0.74,
            chapter="Endocrine", contributing_text=["diabetes", "metformin"],
        )
        assert len(pred.contributing_text) == 2


class TestSummary:
    def test_defaults(self) -> None:
        s = Summary(summary="Stable.")
        assert s.key_findings == []
        assert s.word_count == 0

    def test_full(self) -> None:
        s = Summary(
            summary="Stable post-op.",
            key_findings=["stable", "no infection"],
            detail_level="detailed", word_count=3,
        )
        assert len(s.key_findings) == 2


class TestRiskAssessment:
    def test_risk_factor(self) -> None:
        f = RiskFactor(name="poly", score=0.8, weight=0.6, category="med")
        assert f.description == ""

    def test_defaults(self) -> None:
        r = RiskAssessment(overall_score=0.5, risk_level="moderate")
        assert r.factors == []
        assert r.recommendations == []


class TestAnalysisResultFromDict:
    def test_empty(self) -> None:
        r = AnalysisResult.from_dict({})
        assert r.entities == []
        assert r.summary is None

    def test_entities(self) -> None:
        r = AnalysisResult.from_dict({
            "entities": [{"text": "aspirin", "entity_type": "MEDICATION",
                          "start_char": 0, "end_char": 7, "confidence": 0.95}],
        })
        assert len(r.entities) == 1

    def test_full_response(self) -> None:
        data = {
            "entities": [{"text": "x", "entity_type": "T", "start_char": 0,
                          "end_char": 1, "confidence": 0.9}],
            "icd_predictions": [{"code": "E11", "description": "T2DM", "confidence": 0.7}],
            "summary": {"summary": "OK.", "key_findings": ["ok"]},
            "risk_assessment": {
                "overall_score": 0.4, "risk_level": "low",
                "factors": [{"name": "a", "score": 0.1, "weight": 0.1, "category": "c"}],
                "recommendations": ["monitor"],
            },
            "processing_time_ms": 42.0,
            "model_versions": {"ner": "1.0"},
        }
        r = AnalysisResult.from_dict(data)
        assert r.processing_time_ms == 42.0
        assert r.risk_assessment.risk_level == "low"

    def test_null_fields(self) -> None:
        r = AnalysisResult.from_dict({"entities": None, "icd_predictions": None})
        assert r.entities == []
        assert r.icd_predictions == []


class TestBatchJob:
    def test_minimal(self) -> None:
        j = BatchJob(job_id="abc", status="pending", total_documents=10)
        assert j.progress == 0.0
        assert j.result_file is None

    def test_completed(self) -> None:
        j = BatchJob(
            job_id="x", status="completed", total_documents=50,
            processed_documents=50, progress=1.0, result_file="/r.json",
        )
        assert j.status == "completed"


# ===================================================================
# Document classification
# ===================================================================


class TestClassificationResult:
    def test_from_dict(self) -> None:
        r = ClassificationResult.from_dict({
            "predicted_type": "discharge_summary",
            "scores": [{"document_type": "discharge_summary", "confidence": 0.92}],
            "processing_time_ms": 1.5,
        })
        assert r.predicted_type == "discharge_summary"
        assert len(r.scores) == 1
        assert r.scores[0].confidence == 0.92

    def test_from_dict_empty(self) -> None:
        r = ClassificationResult.from_dict({})
        assert r.predicted_type == "unknown"
        assert r.scores == []

    def test_score_evidence(self) -> None:
        s = ClassificationScore(
            document_type="progress_note", confidence=0.85,
            evidence=["HPI header", "A/P section"],
        )
        assert len(s.evidence) == 2


# ===================================================================
# Medication extraction
# ===================================================================


class TestMedicationResult:
    def test_from_dict(self) -> None:
        r = MedicationResult.from_dict({
            "medication_count": 2,
            "medications": [
                {"drug_name": "metformin", "dosage": "1000 mg", "route": "PO",
                 "frequency": "BID", "confidence": 0.9},
                {"drug_name": "lisinopril", "confidence": 0.8},
            ],
            "processing_time_ms": 3.2,
        })
        assert r.medication_count == 2
        assert r.medications[0].drug_name == "metformin"
        assert r.medications[1].generic_name is None

    def test_medication_defaults(self) -> None:
        m = Medication(drug_name="aspirin")
        assert m.prn is False
        assert m.status == "active"
        assert m.confidence == 0.0


# ===================================================================
# Allergy extraction
# ===================================================================


class TestAllergyResult:
    def test_from_dict(self) -> None:
        r = AllergyResult.from_dict({
            "allergy_count": 1,
            "no_known_allergies": False,
            "allergies": [
                {"allergen": "penicillin", "category": "drug",
                 "severity": "severe", "confidence": 0.95},
            ],
        })
        assert r.allergy_count == 1
        assert r.allergies[0].allergen == "penicillin"

    def test_nkda(self) -> None:
        r = AllergyResult.from_dict({
            "allergy_count": 0, "no_known_allergies": True, "allergies": [],
        })
        assert r.no_known_allergies is True

    def test_allergy_defaults(self) -> None:
        a = Allergy(allergen="peanuts")
        assert a.category == "unknown"
        assert a.reactions == []


# ===================================================================
# Vital signs extraction
# ===================================================================


class TestVitalSignResult:
    def test_from_dict(self) -> None:
        r = VitalSignResult.from_dict({
            "vital_count": 2,
            "vitals": [
                {"vital_type": "heart_rate", "value": 92.0, "unit": "bpm",
                 "interpretation": "normal", "confidence": 0.9},
                {"vital_type": "blood_pressure_systolic", "value": 165.0,
                 "unit": "mmHg", "interpretation": "high", "confidence": 0.95},
            ],
        })
        assert r.vital_count == 2
        assert r.vitals[1].interpretation == "high"

    def test_vital_sign_defaults(self) -> None:
        v = VitalSign(vital_type="temp", value=98.6, unit="°F")
        assert v.interpretation == "normal"
        assert v.confidence == 0.0


# ===================================================================
# Section parsing
# ===================================================================


class TestSectionResult:
    def test_from_dict(self) -> None:
        r = SectionResult.from_dict({
            "section_count": 3,
            "sections": [
                {"category": "chief_complaint", "header": "CHIEF COMPLAINT:",
                 "confidence": 1.0},
            ],
            "categories_found": ["chief_complaint", "hpi", "assessment"],
        })
        assert r.section_count == 3
        assert len(r.sections) == 1
        assert len(r.categories_found) == 3

    def test_section_defaults(self) -> None:
        s = Section(category="unknown", header="")
        assert s.header_start == 0
        assert s.confidence == 0.0


# ===================================================================
# Abbreviation expansion
# ===================================================================


class TestAbbreviationResult:
    def test_from_dict(self) -> None:
        r = AbbreviationResult.from_dict({
            "total_found": 3,
            "expanded_text": "Patient has hypertension (HTN)...",
            "matches": [
                {"abbreviation": "HTN", "expansion": "hypertension",
                 "start": 20, "end": 23, "confidence": 0.9, "domain": "cardiology"},
            ],
        })
        assert r.total_found == 3
        assert r.matches[0].abbreviation == "HTN"

    def test_match_defaults(self) -> None:
        m = AbbreviationMatch(abbreviation="DM", expansion="diabetes mellitus")
        assert m.is_ambiguous is False
        assert m.domain == "general"


# ===================================================================
# Quality analysis
# ===================================================================


class TestQualityReport:
    def test_from_dict(self) -> None:
        r = QualityReport.from_dict({
            "overall_score": 82.5,
            "grade": "B",
            "dimensions": [
                {"dimension": "completeness", "score": 90.0, "weight": 0.3,
                 "finding_count": 1},
            ],
            "recommendation_count": 2,
            "top_recommendations": ["Add vitals section", "Expand HPI"],
        })
        assert r.grade == "B"
        assert len(r.dimensions) == 1
        assert r.recommendation_count == 2

    def test_dimension_defaults(self) -> None:
        d = QualityDimension(dimension="readability", score=75.0, weight=0.2)
        assert d.finding_count == 0


# ===================================================================
# Assertion detection
# ===================================================================


class TestAssertionResult:
    def test_defaults(self) -> None:
        a = AssertionResult(entity_text="diabetes")
        assert a.status == "present"
        assert a.trigger_text is None

    def test_negated(self) -> None:
        a = AssertionResult(
            entity_text="fever", entity_type="SYMPTOM",
            status="absent", confidence=0.92, trigger_text="denies",
        )
        assert a.status == "absent"


# ===================================================================
# Concept normalization
# ===================================================================


class TestNormalizationResult:
    def test_defaults(self) -> None:
        n = NormalizationResult(entity_text="HTN")
        assert n.match_type == "exact"
        assert n.snomed_code is None

    def test_full(self) -> None:
        n = NormalizationResult(
            entity_text="hypertension", entity_type="DISEASE",
            cui="C0020538", preferred_term="Hypertensive disease",
            match_type="exact", confidence=1.0,
            snomed_code="38341003", icd10_code="I10",
        )
        assert n.cui == "C0020538"
        assert n.icd10_code == "I10"


# ===================================================================
# SDoH extraction
# ===================================================================


class TestSDoHResult:
    def test_from_dict(self) -> None:
        r = SDoHResult.from_dict({
            "extraction_count": 2,
            "adverse_count": 1,
            "protective_count": 1,
            "domain_summary": {"substance_use": 1, "social_support": 1},
            "extractions": [
                {"domain": "substance_use", "text": "former smoker",
                 "sentiment": "protective", "confidence": 0.85},
            ],
        })
        assert r.extraction_count == 2
        assert len(r.extractions) == 1

    def test_extraction_defaults(self) -> None:
        e = SDoHExtraction(domain="housing", text="homeless")
        assert e.sentiment == "adverse"
        assert e.z_codes == []


# ===================================================================
# Comorbidity scoring
# ===================================================================


class TestComorbidityResult:
    def test_from_dict(self) -> None:
        r = ComorbidityResult.from_dict({
            "raw_score": 3,
            "age_adjusted_score": 5,
            "risk_group": "moderate",
            "ten_year_mortality": 0.52,
            "category_count": 2,
            "matched_categories": [
                {"category": "diabetes_uncomplicated", "weight": 1,
                 "source": "icd", "evidence": "E11.9", "confidence": 1.0},
            ],
        })
        assert r.raw_score == 3
        assert r.risk_group == "moderate"
        assert len(r.matched_categories) == 1

    def test_defaults(self) -> None:
        r = ComorbidityResult(raw_score=0)
        assert r.risk_group == "low"
        assert r.matched_categories == []


# ===================================================================
# Relation extraction
# ===================================================================


class TestRelationResult:
    def test_from_dict(self) -> None:
        r = RelationResult.from_dict({
            "relation_count": 1,
            "pair_count": 3,
            "relations": [
                {"subject": "metoprolol", "subject_type": "MEDICATION",
                 "object": "hypertension", "object_type": "DISEASE",
                 "relation_type": "treats", "confidence": 0.87,
                 "evidence": "treats"},
            ],
        })
        assert r.relation_count == 1
        assert r.relations[0].relation_type == "treats"

    def test_relation_defaults(self) -> None:
        r = Relation(
            subject="aspirin", subject_type="MEDICATION",
            object="pain", object_type="SYMPTOM",
            relation_type="treats",
        )
        assert r.confidence == 0.0
        assert r.evidence == ""


# ===================================================================
# Enhanced analysis result
# ===================================================================


class TestEnhancedAnalysisResult:
    def test_from_dict_empty(self) -> None:
        r = EnhancedAnalysisResult.from_dict({})
        assert r.base_result is None
        assert r.classification is None
        assert r.component_errors == {}

    def test_from_dict_with_base(self) -> None:
        r = EnhancedAnalysisResult.from_dict({
            "base_result": {
                "entities": [{"text": "x", "entity_type": "T",
                              "start_char": 0, "end_char": 1, "confidence": 0.9}],
            },
            "classification": {"predicted_type": "progress_note"},
            "processing_time_ms": 150.0,
        })
        assert r.base_result is not None
        assert len(r.base_result.entities) == 1
        assert r.classification["predicted_type"] == "progress_note"
        assert r.processing_time_ms == 150.0

    def test_from_dict_all_modules(self) -> None:
        data = {
            "sections": {"section_count": 5},
            "quality": {"overall_score": 85},
            "medications": {"medication_count": 3},
            "allergies": {"allergy_count": 1},
            "vitals": {"vital_count": 4},
            "temporal": {"expressions": []},
            "assertions": [{"entity_text": "fever", "status": "absent"}],
            "normalization": [{"entity_text": "HTN", "cui": "C0020538"}],
            "sdoh": {"extraction_count": 2},
            "relations": {"relation_count": 1},
            "comorbidity": {"raw_score": 3},
            "component_errors": {"base_pipeline": "No NER model loaded"},
        }
        r = EnhancedAnalysisResult.from_dict(data)
        assert r.sections["section_count"] == 5
        assert r.vitals["vital_count"] == 4
        assert len(r.component_errors) == 1


# ===================================================================
# Search
# ===================================================================


class TestSearchResult:
    def test_from_dict(self) -> None:
        r = SearchResult.from_dict({
            "hits": [
                {"document_id": "doc-1", "score": 0.92, "snippet": "diabetes..."},
            ],
            "total": 15,
            "reranked": True,
        })
        assert r.total == 15
        assert r.reranked is True
        assert len(r.hits) == 1

    def test_from_dict_empty(self) -> None:
        r = SearchResult.from_dict({})
        assert r.hits == []
        assert r.total == 0
        assert r.reranked is False

    def test_search_hit_defaults(self) -> None:
        h = SearchHit(document_id="x", score=0.5)
        assert h.snippet == ""
        assert h.title == ""


# ===================================================================
# Evaluation models
# ===================================================================


class TestClassificationEvalResult:
    """Tests for ClassificationEvalResult dataclass."""

    def test_from_dict(self) -> None:
        r = ClassificationEvalResult.from_dict({
            "mcc": 0.85, "tp": 40, "fp": 5, "fn": 3, "tn": 52,
            "calibration": None, "processing_time_ms": 1.2,
        })
        assert r.mcc == 0.85
        assert r.tp == 40
        assert r.calibration is None

    def test_from_dict_with_calibration(self) -> None:
        r = ClassificationEvalResult.from_dict({
            "mcc": 0.72, "tp": 30, "fp": 8, "fn": 5, "tn": 57,
            "calibration": {"expected_calibration_error": 0.03, "brier_score": 0.12},
        })
        assert r.calibration is not None
        assert r.calibration["brier_score"] == 0.12

    def test_from_dict_empty(self) -> None:
        r = ClassificationEvalResult.from_dict({})
        assert r.mcc == 0.0
        assert r.tp == 0


class TestKappaResult:
    """Tests for KappaResult dataclass."""

    def test_from_dict(self) -> None:
        r = KappaResult.from_dict({
            "kappa": 0.82, "observed_agreement": 0.91,
            "expected_agreement": 0.50, "n_items": 100,
        })
        assert r.kappa == 0.82
        assert r.n_items == 100

    def test_from_dict_empty(self) -> None:
        r = KappaResult.from_dict({})
        assert r.kappa == 0.0
        assert r.n_items == 0


class TestNEREvalResult:
    """Tests for NEREvalResult dataclass."""

    def test_from_dict(self) -> None:
        r = NEREvalResult.from_dict({
            "exact_f1": 0.75, "partial_f1": 0.88, "type_weighted_f1": 0.82,
            "mean_overlap": 0.91, "n_gold": 10, "n_pred": 12,
            "n_exact_matches": 7, "n_partial_matches": 2,
            "n_unmatched_pred": 3, "n_unmatched_gold": 1,
        })
        assert r.exact_f1 == 0.75
        assert r.n_gold == 10
        assert r.n_unmatched_pred == 3

    def test_from_dict_empty(self) -> None:
        r = NEREvalResult.from_dict({})
        assert r.exact_f1 == 0.0
        assert r.n_gold == 0


class TestROUGEEvalResult:
    """Tests for ROUGEEvalResult dataclass."""

    def test_from_dict(self) -> None:
        r = ROUGEEvalResult.from_dict({
            "rouge1": {"precision": 0.8, "recall": 0.75, "f1": 0.77},
            "rouge2": {"precision": 0.6, "recall": 0.55, "f1": 0.57},
            "rougeL": {"precision": 0.7, "recall": 0.65, "f1": 0.67},
            "reference_length": 50, "hypothesis_length": 45, "length_ratio": 0.9,
        })
        assert r.rouge1.f1 == 0.77
        assert r.rouge2.precision == 0.6
        assert r.rougeL.recall == 0.65
        assert r.length_ratio == 0.9

    def test_from_dict_empty(self) -> None:
        r = ROUGEEvalResult.from_dict({})
        assert r.rouge1.f1 == 0.0
        assert r.reference_length == 0

    def test_rouge_scores_dataclass(self) -> None:
        s = ROUGEScores(precision=0.9, recall=0.85, f1=0.87)
        assert s.precision == 0.9


class TestICDEvalResult:
    """Tests for ICDEvalResult dataclass."""

    def test_from_dict(self) -> None:
        r = ICDEvalResult.from_dict({
            "full_code_accuracy": 0.65, "block_accuracy": 0.80,
            "chapter_accuracy": 0.95, "n_samples": 100,
            "full_code_matches": 65, "block_matches": 80, "chapter_matches": 95,
        })
        assert r.full_code_accuracy == 0.65
        assert r.chapter_matches == 95

    def test_from_dict_empty(self) -> None:
        r = ICDEvalResult.from_dict({})
        assert r.n_samples == 0


class TestAUPRCResult:
    """Tests for AUPRCResult dataclass."""

    def test_from_dict(self) -> None:
        r = AUPRCResult.from_dict({
            "label": "disease", "auprc": 0.92,
            "n_positive": 30, "n_total": 100,
        })
        assert r.label == "disease"
        assert r.auprc == 0.92

    def test_from_dict_empty(self) -> None:
        r = AUPRCResult.from_dict({})
        assert r.label == "positive"
        assert r.auprc == 0.0


# ===================================================================
# Conversation memory models
# ===================================================================


class TestConversationTurnResult:
    """Tests for ConversationTurnResult dataclass."""

    def test_construction(self) -> None:
        t = ConversationTurnResult(
            session_id="sess-001", turn_id=3, turn_count=3,
        )
        assert t.session_id == "sess-001"
        assert t.turn_id == 3


class TestConversationContext:
    """Tests for ConversationContext dataclass."""

    def test_from_dict(self) -> None:
        r = ConversationContext.from_dict({
            "session_id": "sess-001", "turn_count": 3,
            "unique_entities": ["diabetes", "metformin"],
            "unique_icd_codes": ["E11.9"],
            "overall_risk_trend": [0.3, 0.5, 0.7],
            "context": [{"turn_id": 1, "text": "note 1"}],
        })
        assert r.turn_count == 3
        assert "diabetes" in r.unique_entities
        assert len(r.overall_risk_trend) == 3

    def test_from_dict_empty(self) -> None:
        r = ConversationContext.from_dict({})
        assert r.session_id == ""
        assert r.unique_entities == []


class TestConversationStats:
    """Tests for ConversationStats dataclass."""

    def test_from_dict(self) -> None:
        r = ConversationStats.from_dict({
            "active_sessions": 12, "total_turns": 87,
            "max_turns_per_session": 50, "session_ttl_seconds": 7200.0,
            "max_sessions": 5000,
        })
        assert r.active_sessions == 12
        assert r.total_turns == 87

    def test_from_dict_defaults(self) -> None:
        r = ConversationStats.from_dict({})
        assert r.max_turns_per_session == 50
        assert r.max_sessions == 5000


class TestConversationSessionInfo:
    """Tests for ConversationSessionInfo dataclass."""

    def test_construction(self) -> None:
        s = ConversationSessionInfo(
            session_id="sess-001", turn_count=5,
            oldest_turn_id=1, newest_turn_id=5,
            last_access="2026-03-26T10:00:00Z",
        )
        assert s.session_id == "sess-001"
        assert s.turn_count == 5
        assert s.last_access == "2026-03-26T10:00:00Z"
