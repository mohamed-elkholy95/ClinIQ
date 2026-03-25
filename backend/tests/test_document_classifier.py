"""Tests for the clinical document type classifier module.

Covers the RuleBasedDocumentClassifier comprehensively: enum completeness,
dataclass serialisation, all 13 document types via representative clinical
texts, section/keyword/structural scoring components, batch classification,
minimum confidence filtering, and TransformerDocumentClassifier fallback.
"""

from __future__ import annotations

import pytest

from app.ml.classifier.document_classifier import (
    DOCUMENT_KEYWORDS,
    DOCUMENT_SECTION_PATTERNS,
    STRUCTURAL_PROFILES,
    BaseDocumentClassifier,
    ClassificationResult,
    ClassificationScore,
    DocumentType,
    RuleBasedDocumentClassifier,
    TransformerDocumentClassifier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classifier() -> RuleBasedDocumentClassifier:
    """Create a fresh rule-based classifier."""
    return RuleBasedDocumentClassifier()


@pytest.fixture
def discharge_summary_text() -> str:
    """Representative discharge summary."""
    return """
    DISCHARGE SUMMARY

    ADMISSION DATE: 03/20/2026
    DISCHARGE DATE: 03/25/2026

    DISCHARGE DIAGNOSIS:
    1. Community-acquired pneumonia
    2. Type 2 diabetes mellitus

    HOSPITAL COURSE:
    Patient was admitted with fever, cough, and shortness of breath.
    Chest X-ray confirmed right lower lobe consolidation. Started on
    IV ceftriaxone and azithromycin. Blood cultures negative. Improved
    over 5-day hospital course. Transitioned to oral antibiotics.
    Blood glucose managed with sliding scale insulin.

    DISCHARGE MEDICATIONS:
    1. Amoxicillin-clavulanate 875mg PO BID x 7 days
    2. Metformin 500mg PO BID
    3. Lisinopril 10mg PO daily

    CONDITION AT DISCHARGE: Stable, improved

    FOLLOW-UP INSTRUCTIONS:
    - PCP follow-up in 1 week
    - Repeat chest X-ray in 6 weeks
    - Continue home medications
    """


@pytest.fixture
def progress_note_text() -> str:
    """Representative SOAP progress note."""
    return """
    PROGRESS NOTE — Day 3

    Subjective:
    Patient reports feeling better today. Cough improving, less sputum.
    Slept well overnight. No chest pain. Appetite improving.

    Objective:
    Vitals: T 98.6 F, HR 78, BP 128/76, RR 16, SpO2 97% on RA
    Lungs: Decreased crackles at right base, improved from yesterday.
    Heart: RRR, no murmurs.
    Abdomen: Soft, non-tender.

    Assessment:
    1. Community-acquired pneumonia — improving on IV antibiotics, day 3
    2. Type 2 DM — glucose 132, well controlled on sliding scale

    Plan:
    1. Continue IV antibiotics, transition to PO if afebrile x 24h
    2. Continue sliding scale insulin
    3. PT eval for ambulation
    4. Anticipate discharge tomorrow if continues to improve
    """


@pytest.fixture
def operative_note_text() -> str:
    """Representative operative note."""
    return """
    OPERATIVE NOTE

    DATE OF PROCEDURE: 03/25/2026

    PRE-OPERATIVE DIAGNOSIS: Right inguinal hernia

    POST-OPERATIVE DIAGNOSIS: Right inguinal hernia, direct type

    PROCEDURE PERFORMED: Open right inguinal hernia repair with mesh

    SURGEON: Dr. Smith
    ASSISTANT: Dr. Jones
    ANESTHESIA: General

    FINDINGS:
    Direct inguinal hernia through Hesselbach's triangle, approximately
    3 cm defect. No incarcerated bowel.

    SURGICAL TECHNIQUE:
    Standard groin incision made. External oblique aponeurosis opened.
    Cord structures identified and preserved. Direct hernia sac reduced.
    Polypropylene mesh placed in tension-free fashion. Hemostasis achieved.
    Closure performed in layers with absorbable sutures.

    ESTIMATED BLOOD LOSS: 25 mL
    SPECIMENS SENT: None
    COMPLICATIONS: None
    CONDITION: Stable, to recovery
    """


@pytest.fixture
def radiology_report_text() -> str:
    """Representative radiology report."""
    return """
    RADIOLOGY REPORT

    EXAMINATION: CT Chest with contrast

    CLINICAL INDICATION: Persistent cough, rule out pulmonary embolism

    TECHNIQUE: Helical CT of the chest performed with IV contrast
    administration. Axial images reviewed with multiplanar reconstructions.

    COMPARISON: Chest X-ray from 03/20/2026

    FINDINGS:
    No pulmonary embolism identified.
    Right lower lobe consolidation with air bronchograms, consistent with
    pneumonia. Small right pleural effusion.
    Heart size normal. Mediastinal structures unremarkable.
    No suspicious pulmonary nodule or mass.
    Visualised portions of the upper abdomen unremarkable.

    IMPRESSION:
    1. Right lower lobe pneumonia with small pleural effusion.
    2. No evidence of pulmonary embolism.
    """


@pytest.fixture
def dental_note_text() -> str:
    """Representative dental note."""
    return """
    DENTAL EXAMINATION NOTE

    Patient presents for comprehensive dental exam and prophylaxis.

    Tooth #3: MOD amalgam restoration present, recurrent caries at
    distal margin. Recommend replacement with composite restoration.

    Tooth #14: Existing PFM crown, marginal integrity intact.

    Tooth #19: Class II mesial caries detected on bitewing radiograph.
    Probing depths 2-3mm throughout, no attachment loss.

    Periodontal assessment:
    Probing depths within normal limits (2-3mm).
    No bleeding on probing. Mobility grade 0 throughout.
    Mild calculus noted in lower anterior region.

    Panoramic radiograph reviewed: No pathology identified.
    Bitewing radiographs: Caries #3 and #19 as noted above.

    Treatment plan:
    1. Prophylaxis completed today (D1110)
    2. Schedule composite restoration #3 (D2392)
    3. Schedule composite restoration #19 (D2391)
    """


# ---------------------------------------------------------------------------
# Enum completeness
# ---------------------------------------------------------------------------


class TestDocumentTypeEnum:
    """Tests for the DocumentType enum."""

    def test_all_types_defined(self) -> None:
        """All expected document types exist in the enum."""
        expected = {
            "discharge_summary", "progress_note", "history_physical",
            "operative_note", "consultation_note", "radiology_report",
            "pathology_report", "laboratory_report", "nursing_note",
            "emergency_note", "dental_note", "prescription", "referral",
            "unknown",
        }
        actual = {dt.value for dt in DocumentType}
        assert actual == expected

    def test_enum_count(self) -> None:
        """14 document types total (13 classifiable + UNKNOWN)."""
        assert len(DocumentType) == 14

    def test_string_enum(self) -> None:
        """DocumentType is a string enum for JSON serialisation."""
        assert isinstance(DocumentType.DISCHARGE_SUMMARY, str)
        assert DocumentType.DISCHARGE_SUMMARY == "discharge_summary"


# ---------------------------------------------------------------------------
# Dataclass serialisation
# ---------------------------------------------------------------------------


class TestClassificationScore:
    """Tests for ClassificationScore dataclass."""

    def test_to_dict(self) -> None:
        """to_dict produces expected structure."""
        score = ClassificationScore(
            document_type=DocumentType.DISCHARGE_SUMMARY,
            confidence=0.8765,
            evidence=["Section headers: DISCHARGE SUMMARY"],
        )
        d = score.to_dict()
        assert d["document_type"] == "discharge_summary"
        assert d["confidence"] == 0.8765
        assert len(d["evidence"]) == 1

    def test_default_evidence(self) -> None:
        """Evidence defaults to empty list."""
        score = ClassificationScore(
            document_type=DocumentType.UNKNOWN,
            confidence=0.0,
        )
        assert score.evidence == []


class TestClassificationResult:
    """Tests for ClassificationResult dataclass."""

    def test_to_dict(self) -> None:
        """to_dict produces complete serialisation."""
        result = ClassificationResult(
            predicted_type=DocumentType.PROGRESS_NOTE,
            scores=[
                ClassificationScore(
                    document_type=DocumentType.PROGRESS_NOTE,
                    confidence=0.85,
                ),
            ],
            processing_time_ms=0.42,
            classifier_version="1.0.0",
        )
        d = result.to_dict()
        assert d["predicted_type"] == "progress_note"
        assert len(d["scores"]) == 1
        assert d["processing_time_ms"] == 0.42
        assert d["classifier_version"] == "1.0.0"


# ---------------------------------------------------------------------------
# Data completeness
# ---------------------------------------------------------------------------


class TestDataCompleteness:
    """Verify all classifiable types have patterns, keywords, and profiles."""

    def test_all_types_have_section_patterns(self) -> None:
        """Every non-UNKNOWN type has section header patterns."""
        for dt in DocumentType:
            if dt == DocumentType.UNKNOWN:
                continue
            assert dt in DOCUMENT_SECTION_PATTERNS, f"Missing patterns for {dt}"
            assert len(DOCUMENT_SECTION_PATTERNS[dt]) > 0

    def test_all_types_have_keywords(self) -> None:
        """Every non-UNKNOWN type has keyword lists."""
        for dt in DocumentType:
            if dt == DocumentType.UNKNOWN:
                continue
            assert dt in DOCUMENT_KEYWORDS, f"Missing keywords for {dt}"
            assert len(DOCUMENT_KEYWORDS[dt]) > 0

    def test_all_types_have_structural_profiles(self) -> None:
        """Every non-UNKNOWN type has a structural profile."""
        for dt in DocumentType:
            if dt == DocumentType.UNKNOWN:
                continue
            assert dt in STRUCTURAL_PROFILES, f"Missing profile for {dt}"


# ---------------------------------------------------------------------------
# Scoring components
# ---------------------------------------------------------------------------


class TestScoringComponents:
    """Tests for individual scoring methods."""

    def test_section_score_no_match(self, classifier: RuleBasedDocumentClassifier) -> None:
        """Section score is 0 when no patterns match."""
        score, evidence = classifier._section_score(
            "The quick brown fox jumps over the lazy dog.",
            DocumentType.OPERATIVE_NOTE,
        )
        assert score == 0.0
        assert evidence == []

    def test_section_score_with_match(self, classifier: RuleBasedDocumentClassifier) -> None:
        """Section score is positive when patterns match."""
        text = "OPERATIVE NOTE\nPre-operative diagnosis: hernia\nPost-operative diagnosis: hernia"
        score, evidence = classifier._section_score(text, DocumentType.OPERATIVE_NOTE)
        assert score > 0.0
        assert len(evidence) > 0

    def test_section_score_position_bonus(self, classifier: RuleBasedDocumentClassifier) -> None:
        """First match in opening 500 chars gets position bonus."""
        early_text = "DISCHARGE SUMMARY\n" + "x" * 1000
        late_text = ("x" * 600) + "\nDISCHARGE SUMMARY\n"
        early_score, _ = classifier._section_score(early_text, DocumentType.DISCHARGE_SUMMARY)
        late_score, _ = classifier._section_score(late_text, DocumentType.DISCHARGE_SUMMARY)
        assert early_score > late_score

    def test_keyword_score_no_match(self, classifier: RuleBasedDocumentClassifier) -> None:
        """Keyword score is 0 when no keywords found."""
        score, found = classifier._keyword_score(
            "completely unrelated text about cooking recipes",
            DocumentType.PATHOLOGY_REPORT,
        )
        assert score == 0.0
        assert found == []

    def test_keyword_score_with_matches(self, classifier: RuleBasedDocumentClassifier) -> None:
        """Keyword score reflects fraction of keywords found."""
        text = "specimen gross microscopic diagnosis margins tumor"
        score, found = classifier._keyword_score(text, DocumentType.PATHOLOGY_REPORT)
        assert score > 0.0
        assert len(found) >= 5

    def test_structural_score_in_range(self, classifier: RuleBasedDocumentClassifier) -> None:
        """Structural score is positive when lines and sections match profile."""
        score = classifier._structural_score(
            line_count=50, section_count=6, doc_type=DocumentType.DISCHARGE_SUMMARY,
        )
        assert score > 0.0

    def test_structural_score_out_of_range(self, classifier: RuleBasedDocumentClassifier) -> None:
        """Structural score is lower when line count is outside typical range."""
        score = classifier._structural_score(
            line_count=2, section_count=0, doc_type=DocumentType.DISCHARGE_SUMMARY,
        )
        # 2 lines is below typical_min_lines=30 for discharge summaries
        assert score < 1.0

    def test_count_sections_caps_headers(self, classifier: RuleBasedDocumentClassifier) -> None:
        """All-caps lines are counted as section headers."""
        text = "ASSESSMENT\nSome content here\nPLAN\nMore content\n"
        assert classifier._count_sections(text) >= 2

    def test_count_sections_colon_headers(self, classifier: RuleBasedDocumentClassifier) -> None:
        """Lines ending with colon are counted as section headers."""
        text = "Assessment:\nSome content\nPlan:\nMore content\n"
        assert classifier._count_sections(text) >= 2


# ---------------------------------------------------------------------------
# Document type classification (integration)
# ---------------------------------------------------------------------------


class TestDischargeClassification:
    """Discharge summary classification."""

    def test_classifies_discharge_summary(
        self, classifier: RuleBasedDocumentClassifier, discharge_summary_text: str,
    ) -> None:
        """Discharge summary text is classified correctly."""
        result = classifier.classify(discharge_summary_text)
        assert result.predicted_type == DocumentType.DISCHARGE_SUMMARY

    def test_discharge_has_evidence(
        self, classifier: RuleBasedDocumentClassifier, discharge_summary_text: str,
    ) -> None:
        """Classification includes evidence strings."""
        result = classifier.classify(discharge_summary_text)
        top_score = result.scores[0]
        assert len(top_score.evidence) > 0


class TestProgressNoteClassification:
    """Progress note classification."""

    def test_classifies_progress_note(
        self, classifier: RuleBasedDocumentClassifier, progress_note_text: str,
    ) -> None:
        """SOAP progress note is classified correctly."""
        result = classifier.classify(progress_note_text)
        assert result.predicted_type == DocumentType.PROGRESS_NOTE


class TestOperativeNoteClassification:
    """Operative note classification."""

    def test_classifies_operative_note(
        self, classifier: RuleBasedDocumentClassifier, operative_note_text: str,
    ) -> None:
        """Operative note is classified correctly."""
        result = classifier.classify(operative_note_text)
        assert result.predicted_type == DocumentType.OPERATIVE_NOTE

    def test_operative_confidence_high(
        self, classifier: RuleBasedDocumentClassifier, operative_note_text: str,
    ) -> None:
        """Operative note classification has high confidence."""
        result = classifier.classify(operative_note_text)
        top = result.scores[0]
        assert top.confidence >= 0.3


class TestRadiologyClassification:
    """Radiology report classification."""

    def test_classifies_radiology_report(
        self, classifier: RuleBasedDocumentClassifier, radiology_report_text: str,
    ) -> None:
        """Radiology report is classified correctly."""
        result = classifier.classify(radiology_report_text)
        assert result.predicted_type == DocumentType.RADIOLOGY_REPORT


class TestDentalClassification:
    """Dental note classification."""

    def test_classifies_dental_note(
        self, classifier: RuleBasedDocumentClassifier, dental_note_text: str,
    ) -> None:
        """Dental note is classified correctly."""
        result = classifier.classify(dental_note_text)
        assert result.predicted_type == DocumentType.DENTAL_NOTE


class TestOtherDocumentTypes:
    """Test remaining document types via minimal representative texts."""

    def test_history_physical(self, classifier: RuleBasedDocumentClassifier) -> None:
        """H&P note is classified correctly."""
        text = """
        HISTORY AND PHYSICAL

        Chief Complaint: Chest pain

        History of Present Illness:
        67-year-old male presents with 2 hours of substernal chest pain.

        Past Medical History: HTN, DM2, Hyperlipidemia
        Social History: Former smoker, quit 5 years ago
        Family History: Father with MI at age 55

        Review of Systems:
        Positive for chest pain, shortness of breath. Negative for fever.

        Physical Examination:
        General: Alert, in mild distress
        Heart: Regular rate and rhythm, no murmurs
        Lungs: Clear bilaterally
        Abdomen: Soft, non-tender
        """
        result = classifier.classify(text)
        assert result.predicted_type == DocumentType.HISTORY_PHYSICAL

    def test_consultation_note(self, classifier: RuleBasedDocumentClassifier) -> None:
        """Consultation note is classified correctly."""
        text = """
        CONSULTATION NOTE

        Requesting Physician: Dr. Adams
        Reason for Consultation: Evaluation of persistent anemia

        I was consulted regarding this patient's unexplained anemia.

        Impression and Recommendations:
        1. Iron deficiency anemia, likely due to chronic blood loss
        2. Recommend upper and lower endoscopy
        3. Start iron supplementation 325mg PO TID
        """
        result = classifier.classify(text)
        assert result.predicted_type == DocumentType.CONSULTATION_NOTE

    def test_pathology_report(self, classifier: RuleBasedDocumentClassifier) -> None:
        """Pathology report is classified correctly."""
        text = """
        SURGICAL PATHOLOGY REPORT

        Specimen Type: Skin punch biopsy, left forearm

        Gross Description:
        Received in formalin, a 4mm punch biopsy of skin.

        Microscopic Examination:
        Sections show atypical melanocytic proliferation with pagetoid spread.
        Tumor thickness 0.8mm (Breslow).

        Final Pathologic Diagnosis:
        Malignant melanoma, superficial spreading type.
        Margins: Clear at 2mm.
        """
        result = classifier.classify(text)
        assert result.predicted_type == DocumentType.PATHOLOGY_REPORT

    def test_laboratory_report(self, classifier: RuleBasedDocumentClassifier) -> None:
        """Laboratory report is classified correctly."""
        text = """
        LABORATORY REPORT

        Specimen collected: 03/25/2026

        CBC Results:
        WBC: 12.5 (H) Reference Range: 4.5-11.0
        RBC: 4.8 Reference Range: 4.5-5.5
        Hemoglobin: 13.2 Reference Range: 13.5-17.5
        Platelet: 225 Reference Range: 150-400

        Flag: Abnormal - WBC elevated
        Critical Value: None
        """
        result = classifier.classify(text)
        assert result.predicted_type == DocumentType.LABORATORY_REPORT

    def test_nursing_note(self, classifier: RuleBasedDocumentClassifier) -> None:
        """Nursing note is classified correctly."""
        text = """
        NURSING NOTE

        Patient Assessment:
        Alert and oriented x4. Skin warm and dry.
        Pain assessment: 3/10, controlled with current medications.

        Vital Signs:
        T 98.4, HR 76, BP 122/78, RR 14, SpO2 98% RA

        Intake and Output:
        PO intake 1200mL, Urine output 800mL

        Patient ambulated with PT. IV site right forearm, no redness.
        Wound dressing changed, clean and dry.
        Fall risk: Low (Morse score 25).
        """
        result = classifier.classify(text)
        assert result.predicted_type == DocumentType.NURSING_NOTE

    def test_emergency_note(self, classifier: RuleBasedDocumentClassifier) -> None:
        """Emergency note is classified correctly."""
        text = """
        EMERGENCY DEPARTMENT NOTE

        Mode of Arrival: Ambulance
        Triage Assessment: ESI Level 2
        Chief Complaint: Fall with hip pain

        Time of Arrival: 14:32

        Medical Decision Making:
        High complexity given age and mechanism.
        X-ray right hip obtained - intertrochanteric fracture.
        Orthopedics consulted for surgical repair.

        Disposition: Admitted to orthopedic surgery
        """
        result = classifier.classify(text)
        assert result.predicted_type == DocumentType.EMERGENCY_NOTE

    def test_prescription(self, classifier: RuleBasedDocumentClassifier) -> None:
        """Prescription is classified correctly."""
        text = """
        PRESCRIPTION

        Rx: Amoxicillin 500mg capsule
        Sig: Take 1 capsule by mouth three times daily
        Dispense: 30 capsules
        Refills: 0
        Days Supply: 10

        DEA #: AB1234567
        NPI #: 1234567890
        """
        result = classifier.classify(text)
        assert result.predicted_type == DocumentType.PRESCRIPTION

    def test_referral(self, classifier: RuleBasedDocumentClassifier) -> None:
        """Referral letter is classified correctly."""
        text = """
        REFERRAL NOTE

        Referring Physician: Dr. Martinez, Family Medicine
        Referred to: Dr. Kim, Gastroenterology

        Reason for Referral:
        Chronic GERD unresponsive to PPI therapy. Patient has had
        persistent symptoms despite 8 weeks of omeprazole 40mg daily.

        Urgency: Routine
        Authorization: Approved
        """
        result = classifier.classify(text)
        assert result.predicted_type == DocumentType.REFERRAL


# ---------------------------------------------------------------------------
# Batch & config
# ---------------------------------------------------------------------------


class TestBatchClassification:
    """Tests for batch classification."""

    def test_batch_returns_correct_count(
        self,
        classifier: RuleBasedDocumentClassifier,
        discharge_summary_text: str,
        progress_note_text: str,
    ) -> None:
        """Batch classification returns one result per input."""
        results = classifier.classify_batch([discharge_summary_text, progress_note_text])
        assert len(results) == 2

    def test_batch_preserves_order(
        self,
        classifier: RuleBasedDocumentClassifier,
        discharge_summary_text: str,
        progress_note_text: str,
    ) -> None:
        """Results are in the same order as inputs."""
        results = classifier.classify_batch([discharge_summary_text, progress_note_text])
        assert results[0].predicted_type == DocumentType.DISCHARGE_SUMMARY
        assert results[1].predicted_type == DocumentType.PROGRESS_NOTE

    def test_empty_batch(self, classifier: RuleBasedDocumentClassifier) -> None:
        """Empty input list returns empty results."""
        results = classifier.classify_batch([])
        assert results == []


class TestMinConfidenceFiltering:
    """Tests for minimum confidence threshold."""

    def test_high_threshold_reduces_scores(self) -> None:
        """High min_confidence removes low-scoring types."""
        clf = RuleBasedDocumentClassifier(min_confidence=0.5)
        result = clf.classify("Just a short note with no clear structure.")
        # With high threshold, most types should be filtered out
        for score in result.scores:
            assert score.confidence >= 0.5

    def test_low_threshold_includes_more(self) -> None:
        """Low min_confidence includes more candidate types."""
        clf_low = RuleBasedDocumentClassifier(min_confidence=0.01)
        clf_high = RuleBasedDocumentClassifier(min_confidence=0.3)
        text = "Patient seen today. Assessment: stable. Plan: continue."
        result_low = clf_low.classify(text)
        result_high = clf_high.classify(text)
        assert len(result_low.scores) >= len(result_high.scores)


# ---------------------------------------------------------------------------
# Result metadata
# ---------------------------------------------------------------------------


class TestResultMetadata:
    """Tests for result metadata fields."""

    def test_processing_time_populated(
        self, classifier: RuleBasedDocumentClassifier, discharge_summary_text: str,
    ) -> None:
        """Processing time is a positive number."""
        result = classifier.classify(discharge_summary_text)
        assert result.processing_time_ms > 0

    def test_classifier_version(
        self, classifier: RuleBasedDocumentClassifier, discharge_summary_text: str,
    ) -> None:
        """Classifier version is included in result."""
        result = classifier.classify(discharge_summary_text)
        assert result.classifier_version == "1.0.0"

    def test_scores_sorted_descending(
        self, classifier: RuleBasedDocumentClassifier, discharge_summary_text: str,
    ) -> None:
        """Scores are sorted in descending confidence order."""
        result = classifier.classify(discharge_summary_text)
        confidences = [s.confidence for s in result.scores]
        assert confidences == sorted(confidences, reverse=True)


# ---------------------------------------------------------------------------
# Transformer fallback
# ---------------------------------------------------------------------------


class TestTransformerFallback:
    """Tests for TransformerDocumentClassifier fallback behaviour."""

    def test_fallback_when_not_loaded(self, discharge_summary_text: str) -> None:
        """Uses rule-based fallback when transformer is not loaded."""
        clf = TransformerDocumentClassifier(model_name="nonexistent-model")
        # Don't call load() — should fall back automatically
        result = clf.classify(discharge_summary_text)
        assert result.predicted_type == DocumentType.DISCHARGE_SUMMARY

    def test_fallback_on_load_failure(self, progress_note_text: str) -> None:
        """Falls back gracefully when load fails."""
        clf = TransformerDocumentClassifier(model_name="nonexistent-model")
        clf.load()  # Will fail and set _is_loaded = False
        assert not clf._is_loaded
        result = clf.classify(progress_note_text)
        assert result.predicted_type == DocumentType.PROGRESS_NOTE

    def test_batch_with_fallback(
        self, discharge_summary_text: str, progress_note_text: str,
    ) -> None:
        """Batch classification works via fallback."""
        clf = TransformerDocumentClassifier(model_name="nonexistent-model")
        results = clf.classify_batch([discharge_summary_text, progress_note_text])
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_empty_text(self, classifier: RuleBasedDocumentClassifier) -> None:
        """Empty text still produces a result (may be UNKNOWN or low confidence)."""
        result = classifier.classify("")
        assert isinstance(result, ClassificationResult)

    def test_very_short_text(self, classifier: RuleBasedDocumentClassifier) -> None:
        """Very short text produces a valid result."""
        result = classifier.classify("Patient seen.")
        assert isinstance(result, ClassificationResult)

    def test_unrelated_text(self, classifier: RuleBasedDocumentClassifier) -> None:
        """Non-clinical text produces low confidence scores."""
        result = classifier.classify(
            "The weather forecast calls for rain tomorrow. "
            "We should bring umbrellas to the picnic."
        )
        # If any scores exist, they should be low confidence
        if result.scores:
            assert result.scores[0].confidence < 0.5

    def test_mixed_type_text(self, classifier: RuleBasedDocumentClassifier) -> None:
        """Text with signals from multiple types produces multiple scores."""
        text = """
        DISCHARGE SUMMARY

        Hospital Course: Patient admitted for surgery.

        OPERATIVE NOTE:
        Procedure performed: Appendectomy
        Pre-operative diagnosis: Appendicitis
        Post-operative diagnosis: Appendicitis
        Anesthesia: General
        Blood loss: 50mL
        """
        result = classifier.classify(text)
        # Should have scores for both discharge summary and operative note
        types_in_results = {s.document_type for s in result.scores}
        assert DocumentType.DISCHARGE_SUMMARY in types_in_results
        assert DocumentType.OPERATIVE_NOTE in types_in_results

    def test_repeated_classification_deterministic(
        self, classifier: RuleBasedDocumentClassifier, discharge_summary_text: str,
    ) -> None:
        """Same input always produces the same predicted type."""
        results = [classifier.classify(discharge_summary_text) for _ in range(5)]
        types = {r.predicted_type for r in results}
        assert len(types) == 1
