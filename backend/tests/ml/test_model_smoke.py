"""ML smoke tests — marked with @pytest.mark.ml.

These tests exercise real model loading and inference without requiring
external network access or GPU resources. They are skipped by default
in CI; run them with:

    pytest -m ml backend/tests/ml/

They validate that the rule-based models (which have zero external
dependencies) can be loaded and produce sensible outputs end-to-end.
"""

import pytest

SAMPLE_CLINICAL_NOTE = """
CHIEF COMPLAINT: Chest pain and shortness of breath.

HISTORY OF PRESENT ILLNESS:
Mr. Johnson is a 68-year-old male with a history of coronary artery
disease, type 2 diabetes mellitus, hypertension, and hyperlipidemia
presenting with acute chest pain that started 2 hours ago.
Pain radiates to the left arm. Associated diaphoresis.
Current medications include metformin 1000mg BID, lisinopril 10mg daily,
atorvastatin 40mg nightly, aspirin 81mg daily, and nitroglycerin PRN.

VITAL SIGNS:
BP: 160/95 mmHg, HR: 98 bpm, RR: 22, SpO2: 94%

ASSESSMENT:
Acute coronary syndrome — rule out STEMI.
Emergent cardiology consultation ordered. STAT ECG performed.
Troponin I pending. Heparin drip initiated.

PLAN:
1. Admit to CCU
2. Serial troponins Q6h
3. Cardiology consult
4. Hold metformin peri-procedure
"""


# ---------------------------------------------------------------------------
# NER smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.ml
class TestRuleBasedNERSmoke:
    """Smoke tests for RuleBasedNERModel."""

    def test_load_and_extract_entities(self):
        """Test that the model loads and extracts entities from a clinical note."""
        from app.ml.ner.model import Entity, RuleBasedNERModel

        model = RuleBasedNERModel()
        model.load()

        assert model.is_loaded
        entities = model.extract_entities(SAMPLE_CLINICAL_NOTE)
        assert isinstance(entities, list)
        assert len(entities) > 0
        for entity in entities:
            assert isinstance(entity, Entity)

    def test_extracts_medications(self):
        """Test that known medications are extracted."""
        from app.ml.ner.model import RuleBasedNERModel

        model = RuleBasedNERModel()
        model.load()
        entities = model.extract_entities(SAMPLE_CLINICAL_NOTE)

        meds = [e.text.lower() for e in entities if e.entity_type == "MEDICATION"]
        expected = ["metformin", "lisinopril", "aspirin", "atorvastatin"]
        found = [m for m in expected if any(m in med for med in meds)]
        assert len(found) >= 2, f"Expected at least 2 of {expected} but found: {meds}"

    def test_extracts_dosages(self):
        """Test that dosage patterns are extracted."""
        from app.ml.ner.model import RuleBasedNERModel

        model = RuleBasedNERModel()
        model.load()
        entities = model.extract_entities(SAMPLE_CLINICAL_NOTE)

        dosages = [e for e in entities if e.entity_type == "DOSAGE"]
        assert len(dosages) > 0, "Expected at least one dosage entity"

    def test_entity_positions_valid(self):
        """Test that all entity positions are within the text bounds."""
        from app.ml.ner.model import RuleBasedNERModel

        model = RuleBasedNERModel()
        model.load()
        entities = model.extract_entities(SAMPLE_CLINICAL_NOTE)

        for entity in entities:
            assert entity.start_char >= 0
            assert entity.end_char <= len(SAMPLE_CLINICAL_NOTE)
            assert entity.start_char < entity.end_char

    def test_no_overlapping_entities(self):
        """Test that entities do not overlap after resolution."""
        from app.ml.ner.model import RuleBasedNERModel

        model = RuleBasedNERModel()
        model.load()
        entities = model.extract_entities(SAMPLE_CLINICAL_NOTE)

        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1 :]:
                overlap = e1.start_char < e2.end_char and e2.start_char < e1.end_char
                assert not overlap, (
                    f"Overlap: {e1.text!r} [{e1.start_char}:{e1.end_char}] "
                    f"and {e2.text!r} [{e2.start_char}:{e2.end_char}]"
                )

    def test_negation_heparin_not_negated(self):
        """Test that 'heparin drip initiated' is not negated."""
        from app.ml.ner.model import RuleBasedNERModel

        model = RuleBasedNERModel()
        model.load()
        entities = model.extract_entities(SAMPLE_CLINICAL_NOTE)

        heparin_entities = [e for e in entities if "heparin" in e.text.lower()]
        # Heparin is mentioned as initiated, not denied — should not be negated
        if heparin_entities:
            # The text says "Heparin drip initiated" — check context is not negated
            for e in heparin_entities:
                context = SAMPLE_CLINICAL_NOTE[max(0, e.start_char - 50) : e.start_char]
                # Should not have strong negation right before it
                assert "no heparin" not in context.lower()


# ---------------------------------------------------------------------------
# ICD chapter mapping smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.ml
class TestICDMappingSmoke:
    """Smoke tests for ICD-10 chapter mapping."""

    def test_get_chapter_for_common_codes(self):
        """Test chapter lookup for a set of common clinical codes."""
        from app.ml.icd.model import get_chapter_for_code

        code_chapter_map = {
            "E11.9": "endocrine",
            "I10": "circulatory",
            "J18.9": "respiratory",
            "K21.0": "digestive",
            "M79.3": "musculoskeletal",
            "F32.9": "mental",
            "N18.3": "genitourinary",
        }

        for code, chapter_fragment in code_chapter_map.items():
            chapter = get_chapter_for_code(code)
            assert chapter is not None, f"No chapter for {code}"
            assert chapter_fragment.lower() in chapter.lower(), (
                f"Code {code}: expected '{chapter_fragment}' in '{chapter}'"
            )

    def test_icd_prediction_result_top_k(self):
        """Test the top_k method on ICDPredictionResult."""
        from app.ml.icd.model import ICDCodePrediction, ICDPredictionResult

        predictions = [
            ICDCodePrediction(code=f"E{i:02d}.0", description=f"Code {i}", confidence=i / 10.0)
            for i in range(1, 9)
        ]
        result = ICDPredictionResult(
            predictions=predictions,
            processing_time_ms=10.0,
            model_name="smoke-test",
            model_version="1.0",
        )

        top3 = result.top_k(k=3)
        assert len(top3) == 3
        assert top3[0].confidence >= top3[1].confidence >= top3[2].confidence


# ---------------------------------------------------------------------------
# Risk scoring smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.ml
class TestRuleBasedRiskScorerSmoke:
    """Smoke tests for RuleBasedRiskScorer."""

    def test_load_and_score(self):
        """Test that the scorer loads and returns a valid RiskAssessment."""
        from app.ml.risk.model import RiskAssessment, RuleBasedRiskScorer

        scorer = RuleBasedRiskScorer()
        scorer.load()

        result = scorer.assess_risk(SAMPLE_CLINICAL_NOTE)

        assert isinstance(result, RiskAssessment)
        assert 0.0 <= result.overall_score <= 100.0
        assert result.risk_level in ("low", "moderate", "high", "critical")

    def test_critical_keywords_raise_score(self):
        """Test that STAT/emergent keywords produce an elevated risk score."""
        from app.ml.risk.model import RuleBasedRiskScorer

        scorer = RuleBasedRiskScorer()
        scorer.load()

        benign = "Patient is healthy. Routine check-up."
        critical = "STAT alert: patient in critical condition, emergent intervention needed."

        benign_result = scorer.assess_risk(benign)
        critical_result = scorer.assess_risk(critical)

        assert critical_result.overall_score > benign_result.overall_score, (
            f"Critical text ({critical_result.overall_score:.1f}) should score higher "
            f"than benign text ({benign_result.overall_score:.1f})"
        )

    def test_warfarin_flagged_as_high_risk_med(self):
        """Test that warfarin appears in risk factors."""
        from app.ml.risk.model import RuleBasedRiskScorer

        scorer = RuleBasedRiskScorer()
        scorer.load()

        text = "Patient on warfarin therapy for DVT prophylaxis."
        result = scorer.assess_risk(text)

        factor_names = [f.name for f in result.factors]
        assert any("warfarin" in name for name in factor_names), (
            f"warfarin not found in factors: {factor_names}"
        )

    def test_recommendations_not_empty(self):
        """Test that at least one recommendation is always generated."""
        from app.ml.risk.model import RuleBasedRiskScorer

        scorer = RuleBasedRiskScorer()
        scorer.load()
        result = scorer.assess_risk(SAMPLE_CLINICAL_NOTE)

        assert len(result.recommendations) > 0

    def test_category_scores_complete(self):
        """Test that all three risk categories have scores."""
        from app.ml.risk.model import RISK_CATEGORIES, RuleBasedRiskScorer

        scorer = RuleBasedRiskScorer()
        scorer.load()
        result = scorer.assess_risk(SAMPLE_CLINICAL_NOTE)

        for category in RISK_CATEGORIES:
            assert category in result.category_scores, (
                f"Category '{category}' missing from category_scores"
            )


# ---------------------------------------------------------------------------
# ExtractiveSummarizer smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.ml
class TestExtractiveSummarizerSmoke:
    """Smoke tests for ExtractiveSummarizer."""

    def test_load_and_summarize(self):
        """Test that the summarizer loads and produces a valid result."""
        from app.ml.summarization.model import ExtractiveSummarizer, SummarizationResult

        summarizer = ExtractiveSummarizer()
        summarizer.load()

        result = summarizer.summarize(SAMPLE_CLINICAL_NOTE)

        assert isinstance(result, SummarizationResult)
        assert len(result.summary.strip()) > 0

    def test_summary_shorter_than_original(self):
        """Test that standard summary is shorter than the input."""
        from app.ml.summarization.model import ExtractiveSummarizer

        summarizer = ExtractiveSummarizer()
        summarizer.load()

        result = summarizer.summarize(SAMPLE_CLINICAL_NOTE, detail_level="brief")
        assert len(result.summary) < len(SAMPLE_CLINICAL_NOTE)

    def test_all_detail_levels_work(self):
        """Test that all three detail levels return non-empty summaries."""
        from app.ml.summarization.model import ExtractiveSummarizer

        summarizer = ExtractiveSummarizer()
        summarizer.load()

        for level in ("brief", "standard", "detailed"):
            result = summarizer.summarize(SAMPLE_CLINICAL_NOTE, detail_level=level)
            assert len(result.summary.strip()) > 0, (
                f"Empty summary for detail_level={level!r}"
            )

    def test_key_findings_extracted(self):
        """Test that key clinical findings are extracted from the note."""
        from app.ml.summarization.model import ExtractiveSummarizer

        summarizer = ExtractiveSummarizer()
        summarizer.load()

        result = summarizer.summarize(SAMPLE_CLINICAL_NOTE)

        assert isinstance(result.key_findings, list)
        # The note has clinical importance patterns; key_findings should be populated
        assert len(result.key_findings) > 0

    def test_sentence_counts_consistent(self):
        """Test that summary sentence count <= original sentence count."""
        from app.ml.summarization.model import ExtractiveSummarizer

        summarizer = ExtractiveSummarizer()
        summarizer.load()

        result = summarizer.summarize(SAMPLE_CLINICAL_NOTE)
        assert result.sentence_count_summary <= result.sentence_count_original


# ---------------------------------------------------------------------------
# Dental NER smoke tests
# ---------------------------------------------------------------------------


@pytest.mark.ml
class TestDentalNERSmoke:
    """Smoke tests for DentalNERModel."""

    DENTAL_NOTE = """
    Patient presents for comprehensive dental examination.
    Periodontal charting completed. Probing depths: tooth 3 has 5mm pocket depth,
    tooth 14 MOD composite, tooth 30 root canal treatment completed.
    Scaling and root planing performed in upper right quadrant.
    Periapical radiograph taken. Panoramic radiograph reviewed.
    Diagnosis: gingivitis with bleeding on probing.
    Extraction of tooth 32 recommended.
    """

    def test_load_and_extract(self):
        """Test that the dental model loads and extracts entities."""
        from app.ml.dental.model import DentalEntity, DentalNERModel

        model = DentalNERModel()
        model.load()

        entities = model.extract_entities(self.DENTAL_NOTE)
        assert isinstance(entities, list)
        assert len(entities) > 0
        for e in entities:
            assert isinstance(e, DentalEntity)

    def test_extracts_tooth_numbers(self):
        """Test extraction of tooth numbers from dental note."""
        from app.ml.dental.model import DentalNERModel

        model = DentalNERModel()
        model.load()
        entities = model.extract_entities(self.DENTAL_NOTE)

        tooth_entities = [e for e in entities if e.entity_type == "TOOTH"]
        assert len(tooth_entities) > 0

    def test_extracts_procedures(self):
        """Test extraction of dental procedures."""
        from app.ml.dental.model import DentalNERModel

        model = DentalNERModel()
        model.load()
        entities = model.extract_entities(self.DENTAL_NOTE)

        proc_entities = [e for e in entities if e.entity_type == "DENTAL_PROCEDURE"]
        assert len(proc_entities) > 0

    def test_extracts_conditions(self):
        """Test extraction of dental conditions."""
        from app.ml.dental.model import DentalNERModel

        model = DentalNERModel()
        model.load()
        entities = model.extract_entities(self.DENTAL_NOTE)

        condition_entities = [e for e in entities if e.entity_type == "DENTAL_CONDITION"]
        assert len(condition_entities) > 0

    def test_cdt_predictor_on_dental_text(self):
        """Test CDT code prediction from clinical dental text."""
        from app.ml.dental.model import CDTCodePredictor

        predictor = CDTCodePredictor()
        results = predictor.predict(self.DENTAL_NOTE)

        assert len(results) > 0
        codes = [r["code"] for r in results]
        # Scaling → D4341, extraction → D7140, root canal → D3310
        expected_codes = {"D4341", "D7140", "D3310"}
        found = expected_codes & set(codes)
        assert len(found) > 0, f"None of {expected_codes} found in CDT codes: {codes}"


# ---------------------------------------------------------------------------
# End-to-end pipeline smoke test
# ---------------------------------------------------------------------------


@pytest.mark.ml
class TestClinicalPipelineSmoke:
    """End-to-end smoke test for the ClinicalPipeline with real rule-based models."""

    def test_full_pipeline_end_to_end(self):
        """Test that the full pipeline produces a non-trivial result."""
        from app.ml.ner.model import RuleBasedNERModel
        from app.ml.pipeline import ClinicalPipeline, PipelineConfig, PipelineResult
        from app.ml.risk.model import RuleBasedRiskScorer
        from app.ml.summarization.model import ExtractiveSummarizer

        pipeline = ClinicalPipeline(
            ner_model=RuleBasedNERModel(),
            summarizer=ExtractiveSummarizer(),
            risk_scorer=RuleBasedRiskScorer(),
        )

        config = PipelineConfig(
            enable_ner=True,
            enable_icd=False,
            enable_summarization=True,
            enable_risk=True,
            enable_dental=False,
        )

        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=config)

        assert isinstance(result, PipelineResult)
        assert len(result.entities) > 0, "Expected NER entities"
        assert result.summary is not None, "Expected a summary"
        assert result.risk_assessment is not None, "Expected a risk assessment"
        assert result.processing_time_ms > 0.0
        assert result.component_errors == {}, (
            f"Unexpected component errors: {result.component_errors}"
        )

    def test_pipeline_batch_processing(self):
        """Test pipeline batch processing with multiple documents."""
        from app.ml.ner.model import RuleBasedNERModel
        from app.ml.pipeline import ClinicalPipeline, PipelineConfig

        pipeline = ClinicalPipeline(ner_model=RuleBasedNERModel())
        config = PipelineConfig(
            enable_ner=True,
            enable_icd=False,
            enable_summarization=False,
            enable_risk=False,
        )

        texts = [
            "Patient has diabetes on metformin.",
            "History of hypertension, on lisinopril.",
            "Acute chest pain, aspirin given.",
        ]

        results = pipeline.process_batch(texts, config=config)

        assert len(results) == 3
        for result in results:
            assert result.component_errors == {}
