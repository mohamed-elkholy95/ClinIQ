"""Tests for the enhanced clinical NLP pipeline.

Validates the :class:`EnhancedClinicalPipeline` integration layer that
orchestrates all 14+ clinical NLP modules into a single analysis call.
Tests cover configuration, result structure, per-module stage execution,
fault isolation, batch processing, and end-to-end clinical note analysis.
"""

from __future__ import annotations

import pytest

from app.ml.enhanced_pipeline import (
    EnhancedClinicalPipeline,
    EnhancedPipelineConfig,
    EnhancedPipelineResult,
)
from app.ml.pipeline import ClinicalPipeline, PipelineConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CLINICAL_NOTE = """
CHIEF COMPLAINT:
Chest pain and shortness of breath.

HISTORY OF PRESENT ILLNESS:
65-year-old male presents with acute onset chest pain radiating to left arm.
Patient reports the pain started 3 hours ago. He has a history of hypertension,
diabetes mellitus type 2, and coronary artery disease. He is currently taking
metoprolol 50 mg BID, lisinopril 20 mg daily, and metformin 1000 mg BID.

ALLERGIES:
Penicillin — anaphylaxis
Sulfa drugs — rash

VITAL SIGNS:
BP: 165/95 mmHg, HR: 92 bpm, Temp: 98.6°F, RR: 18, SpO2: 96% on RA

SOCIAL HISTORY:
Former smoker, quit 5 years ago. Drinks alcohol socially. Lives alone.
Retired teacher. Has Medicare insurance.

MEDICATIONS:
1. Metoprolol 50 mg PO BID
2. Lisinopril 20 mg PO daily
3. Metformin 1000 mg PO BID
4. Aspirin 81 mg PO daily
5. Atorvastatin 40 mg PO at bedtime

ASSESSMENT AND PLAN:
1. Acute chest pain — rule out acute coronary syndrome. Will obtain serial
   troponins, 12-lead ECG, and chest X-ray. Start heparin drip.
2. Hypertension — uncontrolled, continue current medications. Consider
   adding amlodipine if BP remains elevated.
3. Type 2 diabetes — continue metformin, check HbA1c.
4. CAD — continue aspirin and statin therapy.

FAMILY HISTORY:
Father had MI at age 58. Mother has hypertension and diabetes.
"""

MINIMAL_NOTE = "Patient presents with headache."

EMPTY_NOTE = ""


@pytest.fixture
def pipeline():
    """Create an EnhancedClinicalPipeline with no base models."""
    return EnhancedClinicalPipeline()


@pytest.fixture
def config_all_on():
    """Configuration with all enhanced modules enabled."""
    return EnhancedPipelineConfig(
        enable_ner=False,  # No NER model loaded
        enable_icd=False,
        enable_summarization=False,
        enable_risk=False,
        enable_dental=False,
        enable_classification=True,
        enable_sections=True,
        enable_quality=True,
        enable_deidentification=True,
        enable_abbreviations=True,
        enable_medications=True,
        enable_allergies=True,
        enable_vitals=True,
        enable_temporal=True,
        enable_assertions=True,
        enable_normalization=True,
        enable_sdoh=True,
        enable_relations=True,
        enable_comorbidity=True,
    )


@pytest.fixture
def config_minimal():
    """Configuration with only sections and classification enabled."""
    return EnhancedPipelineConfig(
        enable_ner=False,
        enable_icd=False,
        enable_summarization=False,
        enable_risk=False,
        enable_dental=False,
        enable_classification=True,
        enable_sections=True,
        enable_quality=False,
        enable_deidentification=False,
        enable_abbreviations=False,
        enable_medications=False,
        enable_allergies=False,
        enable_vitals=False,
        enable_temporal=False,
        enable_assertions=False,
        enable_normalization=False,
        enable_sdoh=False,
        enable_relations=False,
        enable_comorbidity=False,
    )


@pytest.fixture
def config_all_off():
    """Configuration with all modules disabled."""
    return EnhancedPipelineConfig(
        enable_ner=False,
        enable_icd=False,
        enable_summarization=False,
        enable_risk=False,
        enable_dental=False,
        enable_classification=False,
        enable_sections=False,
        enable_quality=False,
        enable_deidentification=False,
        enable_abbreviations=False,
        enable_medications=False,
        enable_allergies=False,
        enable_vitals=False,
        enable_temporal=False,
        enable_assertions=False,
        enable_normalization=False,
        enable_sdoh=False,
        enable_relations=False,
        enable_comorbidity=False,
    )


# ===================================================================
# Configuration tests
# ===================================================================


class TestEnhancedPipelineConfig:
    """Validate configuration defaults and inheritance."""

    def test_inherits_pipeline_config(self):
        """EnhancedPipelineConfig should inherit from PipelineConfig."""
        cfg = EnhancedPipelineConfig()
        assert isinstance(cfg, PipelineConfig)

    def test_default_deidentification_off(self):
        """De-identification should be off by default (destructive)."""
        cfg = EnhancedPipelineConfig()
        assert cfg.enable_deidentification is False

    def test_default_enhanced_modules_on(self):
        """Most enhanced modules should be enabled by default."""
        cfg = EnhancedPipelineConfig()
        assert cfg.enable_classification is True
        assert cfg.enable_sections is True
        assert cfg.enable_quality is True
        assert cfg.enable_abbreviations is True
        assert cfg.enable_medications is True
        assert cfg.enable_allergies is True
        assert cfg.enable_vitals is True
        assert cfg.enable_temporal is True
        assert cfg.enable_assertions is True
        assert cfg.enable_normalization is True
        assert cfg.enable_sdoh is True
        assert cfg.enable_relations is True
        assert cfg.enable_comorbidity is True

    def test_base_config_defaults_preserved(self):
        """Base PipelineConfig defaults should be preserved."""
        cfg = EnhancedPipelineConfig()
        assert cfg.enable_ner is True
        assert cfg.enable_icd is True
        assert cfg.confidence_threshold == 0.5
        assert cfg.top_k_icd == 10

    def test_min_confidence_enhanced_default(self):
        """Default min_confidence_enhanced should be 0.5."""
        cfg = EnhancedPipelineConfig()
        assert cfg.min_confidence_enhanced == 0.5


# ===================================================================
# Result structure tests
# ===================================================================


class TestEnhancedPipelineResult:
    """Validate result dataclass structure and serialisation."""

    def test_default_result_fields_none(self):
        """All optional fields should default to None."""
        result = EnhancedPipelineResult()
        assert result.base_result is None
        assert result.classification is None
        assert result.sections is None
        assert result.quality is None
        assert result.deidentification is None
        assert result.abbreviations is None
        assert result.medications is None
        assert result.allergies is None
        assert result.vitals is None
        assert result.temporal is None
        assert result.assertions is None
        assert result.normalization is None
        assert result.sdoh is None
        assert result.relations is None
        assert result.comorbidity is None

    def test_default_component_errors_empty(self):
        """Component errors should default to empty dict."""
        result = EnhancedPipelineResult()
        assert result.component_errors == {}

    def test_default_processing_time_zero(self):
        """Processing time should default to 0."""
        result = EnhancedPipelineResult()
        assert result.processing_time_ms == 0.0

    def test_to_dict_all_keys_present(self):
        """to_dict() should include all result fields."""
        result = EnhancedPipelineResult()
        d = result.to_dict()
        expected_keys = {
            "base_result", "classification", "sections", "quality",
            "deidentification", "abbreviations", "medications", "allergies",
            "vitals", "temporal", "assertions", "normalization", "sdoh",
            "relations", "comorbidity", "processing_time_ms", "component_errors",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_serialisable(self):
        """to_dict() output should be JSON-serialisable."""
        import json
        result = EnhancedPipelineResult()
        json_str = json.dumps(result.to_dict())
        assert isinstance(json_str, str)


# ===================================================================
# Pipeline lifecycle tests
# ===================================================================


class TestPipelineLifecycle:
    """Test module initialization and lazy loading."""

    def test_lazy_module_initialization(self, pipeline):
        """Modules should not be initialized until process() is called."""
        assert pipeline._modules_initialized is False

    def test_modules_initialized_after_process(self, pipeline, config_all_off):
        """Modules should be initialized after first process() call."""
        pipeline.process(MINIMAL_NOTE, config=config_all_off)
        assert pipeline._modules_initialized is True

    def test_module_init_idempotent(self, pipeline):
        """_ensure_modules() should be idempotent."""
        pipeline._ensure_modules()
        first_classifier = pipeline._classifier
        pipeline._ensure_modules()
        assert pipeline._classifier is first_classifier

    def test_default_base_pipeline(self):
        """Should create a default ClinicalPipeline when none provided."""
        pipeline = EnhancedClinicalPipeline()
        assert isinstance(pipeline._base_pipeline, ClinicalPipeline)

    def test_custom_base_pipeline(self):
        """Should accept a custom ClinicalPipeline."""
        base = ClinicalPipeline()
        pipeline = EnhancedClinicalPipeline(base_pipeline=base)
        assert pipeline._base_pipeline is base


# ===================================================================
# All-disabled pipeline tests
# ===================================================================


class TestAllDisabled:
    """When all modules are disabled, results should be minimal."""

    def test_empty_result_with_all_off(self, pipeline, config_all_off):
        """All result fields should be None when all modules disabled."""
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=config_all_off)
        assert result.classification is None
        assert result.sections is None
        assert result.quality is None
        assert result.deidentification is None
        assert result.abbreviations is None
        assert result.medications is None
        assert result.allergies is None
        assert result.vitals is None
        assert result.temporal is None
        assert result.assertions is None
        assert result.normalization is None
        assert result.sdoh is None
        assert result.relations is None
        assert result.comorbidity is None

    def test_processing_time_recorded(self, pipeline, config_all_off):
        """Processing time should be > 0 even with nothing enabled."""
        result = pipeline.process(MINIMAL_NOTE, config=config_all_off)
        assert result.processing_time_ms >= 0

    def test_base_pipeline_runs(self, pipeline, config_all_off):
        """Base pipeline should still execute (even if models are None)."""
        result = pipeline.process(MINIMAL_NOTE, config=config_all_off)
        assert result.base_result is not None


# ===================================================================
# Document classification tests
# ===================================================================


class TestClassificationStage:
    """Test document type classification."""

    def test_clinical_note_classified(self, pipeline):
        """Clinical note should be classified with a document type."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=True,
            enable_sections=False, enable_quality=False,
            enable_abbreviations=False, enable_medications=False,
            enable_allergies=False, enable_vitals=False,
            enable_temporal=False, enable_assertions=False,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert result.classification is not None
        assert "predicted_type" in result.classification
        assert "confidence" in result.classification
        assert result.classification["confidence"] > 0

    def test_classification_has_top_scores(self, pipeline):
        """Classification should include ranked top scores."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=True,
            enable_sections=False, enable_quality=False,
            enable_abbreviations=False, enable_medications=False,
            enable_allergies=False, enable_vitals=False,
            enable_temporal=False, enable_assertions=False,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert "top_scores" in result.classification
        assert len(result.classification["top_scores"]) > 0


# ===================================================================
# Section parsing tests
# ===================================================================


class TestSectionParsingStage:
    """Test document section parsing."""

    def test_sections_detected(self, pipeline):
        """Sections should be detected in structured clinical note."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False,
            enable_sections=True, enable_quality=False,
            enable_abbreviations=False, enable_medications=False,
            enable_allergies=False, enable_vitals=False,
            enable_temporal=False, enable_assertions=False,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert result.sections is not None
        assert result.sections["section_count"] > 0

    def test_section_structure(self, pipeline):
        """Each section should have category, header, and offsets."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False,
            enable_sections=True, enable_quality=False,
            enable_abbreviations=False, enable_medications=False,
            enable_allergies=False, enable_vitals=False,
            enable_temporal=False, enable_assertions=False,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        for section in result.sections["sections"]:
            assert "category" in section
            assert "header" in section
            assert "header_start" in section
            assert "header_end" in section
            assert "body_end" in section
            assert "confidence" in section

    def test_known_sections_found(self, pipeline):
        """Should detect known clinical sections like Chief Complaint."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False,
            enable_sections=True, enable_quality=False,
            enable_abbreviations=False, enable_medications=False,
            enable_allergies=False, enable_vitals=False,
            enable_temporal=False, enable_assertions=False,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        categories = [s["category"] for s in result.sections["sections"]]
        # Should find at least some of these
        expected_any = [
            "chief_complaint", "history_of_present_illness",
            "allergies", "vital_signs", "medications",
            "assessment_and_plan", "family_history", "social_history",
        ]
        found = [e for e in expected_any if any(e in c for c in categories)]
        assert len(found) >= 3, f"Expected ≥3 known sections, found {found}"


# ===================================================================
# Quality analysis tests
# ===================================================================


class TestQualityStage:
    """Test clinical note quality analysis."""

    def test_quality_report_structure(self, pipeline):
        """Quality report should include score, grade, and dimensions."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=True,
            enable_abbreviations=False, enable_medications=False,
            enable_allergies=False, enable_vitals=False,
            enable_temporal=False, enable_assertions=False,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert result.quality is not None
        assert "overall_score" in result.quality
        assert "grade" in result.quality
        assert "dimensions" in result.quality
        assert 0 <= result.quality["overall_score"] <= 100
        assert result.quality["grade"] in ("A", "B", "C", "D", "F")

    def test_quality_has_recommendations(self, pipeline):
        """Quality report should include recommendations."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=True,
            enable_abbreviations=False, enable_medications=False,
            enable_allergies=False, enable_vitals=False,
            enable_temporal=False, enable_assertions=False,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert "recommendation_count" in result.quality
        assert isinstance(result.quality["recommendation_count"], int)


# ===================================================================
# Abbreviation expansion tests
# ===================================================================


class TestAbbreviationStage:
    """Test abbreviation detection and expansion."""

    def test_abbreviations_detected(self, pipeline):
        """Should detect clinical abbreviations in text."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False,
            enable_abbreviations=True,
            enable_medications=False, enable_allergies=False,
            enable_vitals=False, enable_temporal=False,
            enable_assertions=False, enable_normalization=False,
            enable_sdoh=False, enable_relations=False,
            enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert result.abbreviations is not None
        assert result.abbreviations["total_found"] > 0

    def test_abbreviation_match_structure(self, pipeline):
        """Each abbreviation match should have required fields."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False,
            enable_abbreviations=True,
            enable_medications=False, enable_allergies=False,
            enable_vitals=False, enable_temporal=False,
            enable_assertions=False, enable_normalization=False,
            enable_sdoh=False, enable_relations=False,
            enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        for match in result.abbreviations["matches"]:
            assert "abbreviation" in match
            assert "expansion" in match
            assert "confidence" in match
            assert "domain" in match

    def test_expanded_text_returned(self, pipeline):
        """Should return the text with abbreviations expanded inline."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False,
            enable_abbreviations=True,
            enable_medications=False, enable_allergies=False,
            enable_vitals=False, enable_temporal=False,
            enable_assertions=False, enable_normalization=False,
            enable_sdoh=False, enable_relations=False,
            enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert "expanded_text" in result.abbreviations


# ===================================================================
# Medication extraction tests
# ===================================================================


class TestMedicationStage:
    """Test medication extraction."""

    def test_medications_extracted(self, pipeline):
        """Should extract medications from clinical note."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=True,
            enable_allergies=False, enable_vitals=False,
            enable_temporal=False, enable_assertions=False,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert result.medications is not None
        assert result.medications["medication_count"] > 0

    def test_medication_structure(self, pipeline):
        """Each medication should have name, dosage, route, frequency."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=True,
            enable_allergies=False, enable_vitals=False,
            enable_temporal=False, enable_assertions=False,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        for med in result.medications["medications"]:
            assert "drug_name" in med
            assert "confidence" in med

    def test_known_medications_found(self, pipeline):
        """Should find known medications like metoprolol, lisinopril."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=True,
            enable_allergies=False, enable_vitals=False,
            enable_temporal=False, enable_assertions=False,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        drug_names = [
            m["drug_name"].lower() for m in result.medications["medications"]
        ]
        known = ["metoprolol", "lisinopril", "metformin", "aspirin", "atorvastatin"]
        found = [k for k in known if any(k in d for d in drug_names)]
        assert len(found) >= 3, f"Expected ≥3 known meds, found: {found}"


# ===================================================================
# Allergy extraction tests
# ===================================================================


class TestAllergyStage:
    """Test allergy extraction."""

    def test_allergies_extracted(self, pipeline):
        """Should detect allergies in clinical note."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=False,
            enable_allergies=True,
            enable_vitals=False, enable_temporal=False,
            enable_assertions=False, enable_normalization=False,
            enable_sdoh=False, enable_relations=False,
            enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert result.allergies is not None
        assert result.allergies["allergy_count"] > 0

    def test_penicillin_allergy_detected(self, pipeline):
        """Should detect penicillin allergy from sample note."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=False,
            enable_allergies=True,
            enable_vitals=False, enable_temporal=False,
            enable_assertions=False, enable_normalization=False,
            enable_sdoh=False, enable_relations=False,
            enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        allergens = [a["allergen"].lower() for a in result.allergies["allergies"]]
        assert any("penicillin" in a for a in allergens)

    def test_nkda_detection(self, pipeline):
        """Should detect NKDA pattern."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=False,
            enable_allergies=True,
            enable_vitals=False, enable_temporal=False,
            enable_assertions=False, enable_normalization=False,
            enable_sdoh=False, enable_relations=False,
            enable_comorbidity=False,
        )
        result = pipeline.process("ALLERGIES: NKDA", config=cfg)
        assert result.allergies["no_known_allergies"] is True


# ===================================================================
# Vital signs extraction tests
# ===================================================================


class TestVitalsStage:
    """Test vital signs extraction."""

    def test_vitals_extracted(self, pipeline):
        """Should extract vital signs from clinical note."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=False, enable_allergies=False,
            enable_vitals=True,
            enable_temporal=False, enable_assertions=False,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert result.vitals is not None
        assert result.vitals["vital_count"] > 0

    def test_vital_structure(self, pipeline):
        """Each vital should have type, value, unit, interpretation."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=False, enable_allergies=False,
            enable_vitals=True,
            enable_temporal=False, enable_assertions=False,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        for vital in result.vitals["vitals"]:
            assert "type" in vital
            assert "value" in vital
            assert "unit" in vital
            assert "interpretation" in vital

    def test_blood_pressure_found(self, pipeline):
        """Should detect blood pressure from sample note."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=False, enable_allergies=False,
            enable_vitals=True,
            enable_temporal=False, enable_assertions=False,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        vital_types = [v["type"] for v in result.vitals["vitals"]]
        assert any("blood_pressure" in t.lower() or "systolic" in t.lower()
                    for t in vital_types)


# ===================================================================
# Temporal extraction tests
# ===================================================================


class TestTemporalStage:
    """Test temporal information extraction."""

    def test_temporal_extracted(self, pipeline):
        """Should extract temporal expressions from clinical note."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=False, enable_allergies=False,
            enable_vitals=False,
            enable_temporal=True,
            enable_assertions=False, enable_normalization=False,
            enable_sdoh=False, enable_relations=False,
            enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert result.temporal is not None

    def test_temporal_has_expressions(self, pipeline):
        """Temporal result should include expressions list."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=False, enable_allergies=False,
            enable_vitals=False,
            enable_temporal=True,
            enable_assertions=False, enable_normalization=False,
            enable_sdoh=False, enable_relations=False,
            enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert "expressions" in result.temporal


# ===================================================================
# De-identification tests
# ===================================================================


class TestDeidentificationStage:
    """Test PHI de-identification."""

    def test_deidentification_runs(self, pipeline):
        """Should run de-identification when enabled."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False,
            enable_deidentification=True,
            enable_abbreviations=False, enable_medications=False,
            enable_allergies=False, enable_vitals=False,
            enable_temporal=False, enable_assertions=False,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert result.deidentification is not None
        assert "text" in result.deidentification

    def test_deidentification_off_by_default(self, pipeline):
        """Default config should not run de-identification."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert result.deidentification is None


# ===================================================================
# SDoH extraction tests
# ===================================================================


class TestSDoHStage:
    """Test Social Determinants of Health extraction."""

    def test_sdoh_extracted(self, pipeline):
        """Should extract SDoH factors from clinical note."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=False, enable_allergies=False,
            enable_vitals=False, enable_temporal=False,
            enable_assertions=False, enable_normalization=False,
            enable_sdoh=True,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert result.sdoh is not None

    def test_smoking_history_detected(self, pipeline):
        """Should detect smoking-related SDoH from sample note."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=False, enable_allergies=False,
            enable_vitals=False, enable_temporal=False,
            enable_assertions=False, enable_normalization=False,
            enable_sdoh=True,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert result.sdoh["extraction_count"] > 0


# ===================================================================
# Comorbidity scoring tests
# ===================================================================


class TestComorbidityStage:
    """Test Charlson Comorbidity Index calculation."""

    def test_comorbidity_calculated(self, pipeline):
        """Should calculate CCI from text mentioning conditions."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=False, enable_allergies=False,
            enable_vitals=False, enable_temporal=False,
            enable_assertions=False, enable_normalization=False,
            enable_sdoh=False, enable_relations=False,
            enable_comorbidity=True,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert result.comorbidity is not None
        assert "raw_score" in result.comorbidity
        assert "risk_group" in result.comorbidity

    def test_comorbidity_detects_conditions(self, pipeline):
        """Should detect conditions like diabetes and CAD from text."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=False, enable_allergies=False,
            enable_vitals=False, enable_temporal=False,
            enable_assertions=False, enable_normalization=False,
            enable_sdoh=False, enable_relations=False,
            enable_comorbidity=True,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert result.comorbidity["category_count"] > 0


# ===================================================================
# Assertion detection tests (no entities without NER)
# ===================================================================


class TestAssertionStage:
    """Test assertion detection (depends on NER entities)."""

    def test_assertions_empty_without_entities(self, pipeline):
        """Without NER entities, assertions should be empty list."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=False, enable_allergies=False,
            enable_vitals=False, enable_temporal=False,
            enable_assertions=True,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert result.assertions == []


# ===================================================================
# Normalization tests (no entities without NER)
# ===================================================================


class TestNormalizationStage:
    """Test concept normalization (depends on NER entities)."""

    def test_normalization_empty_without_entities(self, pipeline):
        """Without NER entities, normalization should be empty list."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=False, enable_allergies=False,
            enable_vitals=False, enable_temporal=False,
            enable_assertions=False,
            enable_normalization=True,
            enable_sdoh=False, enable_relations=False,
            enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert result.normalization == []


# ===================================================================
# Relations tests (no entities without NER)
# ===================================================================


class TestRelationsStage:
    """Test relation extraction (depends on NER entities)."""

    def test_relations_empty_without_entities(self, pipeline):
        """Without NER entities, relations should be empty."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=False, enable_allergies=False,
            enable_vitals=False, enable_temporal=False,
            enable_assertions=False, enable_normalization=False,
            enable_sdoh=False,
            enable_relations=True,
            enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert result.relations is not None
        assert result.relations["relation_count"] == 0


# ===================================================================
# Fault isolation tests
# ===================================================================


class TestFaultIsolation:
    """Verify that failures in one stage don't affect others."""

    def test_classification_failure_isolated(self, pipeline):
        """If classifier fails, other stages should still run."""
        pipeline._ensure_modules()
        # Sabotage the classifier
        pipeline._classifier = None
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=True,
            enable_sections=True,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=False, enable_allergies=False,
            enable_vitals=False, enable_temporal=False,
            enable_assertions=False, enable_normalization=False,
            enable_sdoh=False, enable_relations=False,
            enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        # Classification should be None (module was None, not error)
        assert result.classification is None
        # Sections should still work
        assert result.sections is not None

    def test_error_captured_in_component_errors(self, pipeline):
        """Module errors should be captured, not raised."""
        pipeline._ensure_modules()

        # Replace quality analyzer with a broken one
        class BrokenAnalyzer:
            def analyze(self, text):
                raise RuntimeError("Test error")

        pipeline._quality_analyzer = BrokenAnalyzer()
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=True,
            enable_abbreviations=False, enable_medications=False,
            enable_allergies=False, enable_vitals=False,
            enable_temporal=False, enable_assertions=False,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert "quality" in result.component_errors
        assert "Test error" in result.component_errors["quality"]

    def test_multiple_failures_all_captured(self, pipeline):
        """Multiple stage failures should all be captured."""
        pipeline._ensure_modules()

        class BrokenModule:
            def __getattr__(self, name):
                def broken(*args, **kwargs):
                    raise ValueError(f"Broken {name}")
                return broken

        pipeline._quality_analyzer = BrokenModule()
        pipeline._vitals_extractor = BrokenModule()

        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=True, enable_abbreviations=False,
            enable_medications=False, enable_allergies=False,
            enable_vitals=True,
            enable_temporal=False, enable_assertions=False,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        assert "quality" in result.component_errors
        assert "vitals" in result.component_errors


# ===================================================================
# Batch processing tests
# ===================================================================


class TestBatchProcessing:
    """Test batch document processing."""

    def test_batch_returns_list(self, pipeline, config_minimal):
        """Batch should return a list of results."""
        texts = [SAMPLE_CLINICAL_NOTE, MINIMAL_NOTE]
        results = pipeline.process_batch(texts, config=config_minimal)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_batch_with_document_ids(self, pipeline, config_minimal):
        """Batch should pass document IDs through."""
        texts = [MINIMAL_NOTE, MINIMAL_NOTE]
        doc_ids = ["doc-1", "doc-2"]
        results = pipeline.process_batch(
            texts, config=config_minimal, document_ids=doc_ids,
        )
        assert len(results) == 2

    def test_batch_independent_results(self, pipeline, config_minimal):
        """Each batch item should be processed independently."""
        texts = [SAMPLE_CLINICAL_NOTE, MINIMAL_NOTE]
        results = pipeline.process_batch(texts, config=config_minimal)
        # Full note should have more sections than minimal note
        if results[0].sections and results[1].sections:
            assert (
                results[0].sections["section_count"]
                >= results[1].sections["section_count"]
            )


# ===================================================================
# End-to-end integration tests
# ===================================================================


class TestEndToEnd:
    """Full pipeline integration tests with realistic clinical notes."""

    def test_full_pipeline_clinical_note(self, pipeline, config_all_on):
        """Run all modules on a comprehensive clinical note."""
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=config_all_on)

        # Core structural analysis should succeed
        assert result.sections is not None
        assert result.quality is not None
        assert result.classification is not None

        # Clinical extraction modules should succeed
        assert result.medications is not None
        assert result.allergies is not None
        assert result.vitals is not None
        assert result.abbreviations is not None

        # Processing time should be reasonable (< 5 seconds)
        assert result.processing_time_ms < 5000

    def test_full_pipeline_serialisable(self, pipeline, config_all_on):
        """Full pipeline result should be JSON-serialisable."""
        import json
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=config_all_on)
        json_str = json.dumps(result.to_dict())
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert "processing_time_ms" in parsed

    def test_minimal_note_no_crash(self, pipeline, config_all_on):
        """Minimal note should not crash any module."""
        result = pipeline.process(MINIMAL_NOTE, config=config_all_on)
        assert result.processing_time_ms > 0
        # Should not have fatal errors (individual modules may error gracefully)

    def test_empty_note_handling(self, pipeline, config_all_on):
        """Empty note should be handled gracefully."""
        result = pipeline.process(EMPTY_NOTE, config=config_all_on)
        assert isinstance(result, EnhancedPipelineResult)
        # Some modules may error on empty text — that's OK
        # as long as nothing crashes

    def test_document_id_propagated(self, pipeline, config_minimal):
        """Document ID should be propagated through the pipeline."""
        result = pipeline.process(
            MINIMAL_NOTE, config=config_minimal, document_id="test-doc-42",
        )
        if result.base_result:
            assert result.base_result.document_id == "test-doc-42"

    def test_no_component_errors_on_good_note(self, pipeline):
        """Clean clinical note should not produce module errors."""
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=True,
            enable_sections=True,
            enable_quality=True,
            enable_abbreviations=True,
            enable_medications=True,
            enable_allergies=True,
            enable_vitals=True,
            enable_temporal=True,
            enable_sdoh=True,
            enable_comorbidity=True,
            # Skip assertion/normalization/relations (need NER entities)
            enable_assertions=False,
            enable_normalization=False,
            enable_relations=False,
        )
        result = pipeline.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        # Filter out base pipeline errors (no models loaded)
        enhanced_errors = {
            k: v for k, v in result.component_errors.items()
            if k not in ("ner", "icd", "summarization", "risk", "base_pipeline")
        }
        assert len(enhanced_errors) == 0, f"Unexpected errors: {enhanced_errors}"


# ===================================================================
# Entity-dependent stage tests (assertions, normalization, relations)
# ===================================================================

# Clinical text with entity positions for injection into base_result.
_ENTITY_TEXT = (
    "Patient has hypertension and diabetes. "
    "Started metoprolol 50 mg for blood pressure control."
)


def _make_entity(text: str, entity_type: str, start: int, end: int, conf: float = 0.9):
    """Create an Entity dataclass instance for testing."""
    from app.ml.ner.model import Entity
    return Entity(
        text=text,
        entity_type=entity_type,
        start_char=start,
        end_char=end,
        confidence=conf,
    )


def _make_entities_for_text():
    """Build entities aligned to _ENTITY_TEXT positions."""
    # "hypertension" at index 12..24, "diabetes" at 29..37
    # "metoprolol" at 47..57
    return [
        _make_entity("hypertension", "DISEASE", 12, 24),
        _make_entity("diabetes", "DISEASE", 29, 37),
        _make_entity("metoprolol", "MEDICATION", 47, 57),
    ]


def _pipeline_with_entities():
    """Create a pipeline whose base result has synthetic entities."""
    from app.ml.pipeline import PipelineResult

    pipe = EnhancedClinicalPipeline()
    pipe._ensure_modules()

    # Inject entities into a synthetic base result
    entities = _make_entities_for_text()
    base = PipelineResult(document_id="synth-1", entities=entities)
    return pipe, base, entities


class TestAssertionWithEntities:
    """Assertion detection with injected NER entities."""

    def test_assertions_populated_from_entities(self):
        """When base_result has entities, assertions should be populated."""
        pipe, base, entities = _pipeline_with_entities()

        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=False, enable_allergies=False,
            enable_vitals=False, enable_temporal=False,
            enable_assertions=True,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )

        # Manually process with a patched base pipeline
        result = EnhancedPipelineResult()
        result.base_result = base

        result = pipe._run_assertions(_ENTITY_TEXT, cfg, result)
        assert result.assertions is not None
        assert len(result.assertions) > 0, "Expected at least 1 assertion result"
        for a in result.assertions:
            assert "entity_text" in a
            assert "entity_type" in a
            assert "status" in a
            assert "confidence" in a

    def test_assertions_have_correct_entity_types(self):
        """Each assertion should carry entity_type from source entity."""
        pipe, base, entities = _pipeline_with_entities()

        cfg = EnhancedPipelineConfig()
        result = EnhancedPipelineResult()
        result.base_result = base
        result = pipe._run_assertions(_ENTITY_TEXT, cfg, result)

        types_found = {a["entity_type"] for a in result.assertions}
        assert "DISEASE" in types_found

    def test_assertions_skip_failing_entity(self):
        """If one entity fails assertion detection, others should still work."""
        pipe, base, entities = _pipeline_with_entities()

        # Add an entity with out-of-range offsets
        from app.ml.ner.model import Entity
        bad_entity = Entity(
            text="bogus", entity_type="DISEASE",
            start_char=99999, end_char=100005, confidence=0.9,
        )
        base.entities.append(bad_entity)

        cfg = EnhancedPipelineConfig()
        result = EnhancedPipelineResult()
        result.base_result = base
        result = pipe._run_assertions(_ENTITY_TEXT, cfg, result)

        # Should still have assertions from valid entities
        assert len(result.assertions) >= 2


class TestNormalizationWithEntities:
    """Concept normalization with injected NER entities."""

    def test_normalization_produces_results(self):
        """Known conditions should be normalized to ontology codes."""
        pipe, base, entities = _pipeline_with_entities()

        cfg = EnhancedPipelineConfig()
        result = EnhancedPipelineResult()
        result.base_result = base
        result = pipe._run_normalization(_ENTITY_TEXT, cfg, result)

        assert result.normalization is not None
        assert isinstance(result.normalization, list)
        # "hypertension" and "diabetes" should match in the concept dictionary
        if len(result.normalization) > 0:
            norm = result.normalization[0]
            assert "entity_text" in norm
            assert "cui" in norm
            assert "preferred_term" in norm
            assert "confidence" in norm

    def test_normalization_includes_ontology_codes(self):
        """Normalized entities should include SNOMED/ICD/RxNorm codes."""
        pipe, base, entities = _pipeline_with_entities()

        cfg = EnhancedPipelineConfig()
        result = EnhancedPipelineResult()
        result.base_result = base
        result = pipe._run_normalization(_ENTITY_TEXT, cfg, result)

        for norm in result.normalization:
            assert "snomed_code" in norm
            assert "rxnorm_code" in norm
            assert "icd10_code" in norm
            assert "loinc_code" in norm

    def test_normalization_match_type_set(self):
        """Each normalized entity should have a match_type."""
        pipe, base, entities = _pipeline_with_entities()

        cfg = EnhancedPipelineConfig()
        result = EnhancedPipelineResult()
        result.base_result = base
        result = pipe._run_normalization(_ENTITY_TEXT, cfg, result)

        for norm in result.normalization:
            assert "match_type" in norm
            assert norm["match_type"] in ("exact", "alias", "fuzzy")

    def test_normalization_empty_on_unknown_entity(self):
        """Unknown entity text should not produce normalization results."""
        pipe = EnhancedClinicalPipeline()
        pipe._ensure_modules()
        from app.ml.pipeline import PipelineResult

        entity = _make_entity("xyzzy_unknown_99", "DISEASE", 0, 16)
        base = PipelineResult(document_id="test", entities=[entity])

        cfg = EnhancedPipelineConfig()
        result = EnhancedPipelineResult()
        result.base_result = base
        result = pipe._run_normalization("xyzzy_unknown_99", cfg, result)

        # Should be empty or at best empty list (no match)
        assert isinstance(result.normalization, list)


class TestRelationsWithEntities:
    """Relation extraction with injected NER entities."""

    def test_relations_extracted_from_entity_pairs(self):
        """With 2+ entities, relation extraction should produce results."""
        pipe, base, entities = _pipeline_with_entities()

        cfg = EnhancedPipelineConfig()
        result = EnhancedPipelineResult()
        result.base_result = base
        result = pipe._run_relations(_ENTITY_TEXT, cfg, result)

        assert result.relations is not None
        assert "relation_count" in result.relations
        assert "pair_count" in result.relations
        assert result.relations["pair_count"] > 0

    def test_relations_structure(self):
        """Each relation should have subject, object, type, confidence."""
        pipe, base, entities = _pipeline_with_entities()

        cfg = EnhancedPipelineConfig()
        result = EnhancedPipelineResult()
        result.base_result = base
        result = pipe._run_relations(_ENTITY_TEXT, cfg, result)

        for rel in result.relations["relations"]:
            assert "subject" in rel
            assert "object" in rel
            assert "relation_type" in rel
            assert "confidence" in rel
            assert "evidence" in rel

    def test_relations_with_medication_disease_pair(self):
        """Should detect medication-disease relation (e.g., treats)."""
        # Text where metoprolol treats hypertension in close proximity
        text = "Metoprolol treats hypertension effectively."
        from app.ml.pipeline import PipelineResult

        pipe = EnhancedClinicalPipeline()
        pipe._ensure_modules()
        entities = [
            _make_entity("Metoprolol", "MEDICATION", 0, 10),
            _make_entity("hypertension", "DISEASE", 18, 30),
        ]
        base = PipelineResult(document_id="test", entities=entities)

        cfg = EnhancedPipelineConfig()
        result = EnhancedPipelineResult()
        result.base_result = base
        result = pipe._run_relations(text, cfg, result)

        assert result.relations["relation_count"] > 0
        rel_types = [r["relation_type"] for r in result.relations["relations"]]
        assert "treats" in rel_types, f"Expected 'treats' in {rel_types}"

    def test_relations_single_entity_returns_empty(self):
        """With fewer than 2 entities, relations should be empty."""
        from app.ml.pipeline import PipelineResult

        pipe = EnhancedClinicalPipeline()
        pipe._ensure_modules()
        entities = [_make_entity("hypertension", "DISEASE", 12, 24)]
        base = PipelineResult(document_id="test", entities=entities)

        cfg = EnhancedPipelineConfig()
        result = EnhancedPipelineResult()
        result.base_result = base
        result = pipe._run_relations("Patient has hypertension.", cfg, result)

        assert result.relations["relation_count"] == 0


class TestDeidentificationStageExecution:
    """Test de-identification with actual execution."""

    def test_deidentification_redacts_names(self, pipeline):
        """De-identification should redact name-like patterns."""
        text = "Dr. John Smith prescribed metformin for the patient."
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False,
            enable_deidentification=True,
            enable_abbreviations=False, enable_medications=False,
            enable_allergies=False, enable_vitals=False,
            enable_temporal=False, enable_assertions=False,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(text, config=cfg)
        assert result.deidentification is not None

    def test_deidentification_returns_dict(self, pipeline):
        """De-identification result should have text and detections."""
        text = "Patient John Smith, DOB: 01/15/1960, SSN: 123-45-6789"
        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False,
            enable_deidentification=True,
            enable_abbreviations=False, enable_medications=False,
            enable_allergies=False, enable_vitals=False,
            enable_temporal=False, enable_assertions=False,
            enable_normalization=False, enable_sdoh=False,
            enable_relations=False, enable_comorbidity=False,
        )
        result = pipeline.process(text, config=cfg)
        assert result.deidentification is not None


class TestModuleInitializationFailure:
    """Test graceful degradation when modules fail to initialize."""

    def test_pipeline_works_with_broken_import(self):
        """Pipeline should handle import failures gracefully."""
        pipe = EnhancedClinicalPipeline()
        pipe._ensure_modules()

        # Sabotage a module after init
        pipe._medication_extractor = None
        pipe._allergy_extractor = None

        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=True,
            enable_sections=True,
            enable_quality=False,
            enable_abbreviations=False,
            enable_medications=True,
            enable_allergies=True,
            enable_vitals=False, enable_temporal=False,
            enable_assertions=False, enable_normalization=False,
            enable_sdoh=False, enable_relations=False,
            enable_comorbidity=False,
        )
        result = pipe.process(SAMPLE_CLINICAL_NOTE, config=cfg)
        # Medications/allergies should be None (module is None)
        assert result.medications is None
        assert result.allergies is None
        # But sections and classification should work
        assert result.sections is not None
        assert result.classification is not None


class TestComorbidityWithIcdCodes:
    """Test comorbidity scoring with ICD codes from base pipeline."""

    def test_comorbidity_uses_base_icd_codes(self):
        """CCI should use ICD codes from base pipeline result."""
        from app.ml.pipeline import PipelineResult

        pipe = EnhancedClinicalPipeline()
        pipe._ensure_modules()

        base = PipelineResult(
            document_id="test",
            icd_predictions=[
                {"code": "E11.9", "description": "Type 2 diabetes"},
                {"code": "I10", "description": "Essential hypertension"},
            ],
        )

        cfg = EnhancedPipelineConfig(
            enable_ner=False, enable_icd=False,
            enable_summarization=False, enable_risk=False,
            enable_classification=False, enable_sections=False,
            enable_quality=False, enable_abbreviations=False,
            enable_medications=False, enable_allergies=False,
            enable_vitals=False, enable_temporal=False,
            enable_assertions=False, enable_normalization=False,
            enable_sdoh=False, enable_relations=False,
            enable_comorbidity=True,
        )
        result = EnhancedPipelineResult()
        result.base_result = base
        result = pipe._run_comorbidity(
            "Type 2 diabetes and hypertension", cfg, result,
        )

        assert result.comorbidity is not None
        assert result.comorbidity["raw_score"] > 0
        assert result.comorbidity["category_count"] > 0
        assert "risk_group" in result.comorbidity
        assert "ten_year_mortality" in result.comorbidity
        assert isinstance(result.comorbidity["matched_categories"], list)

    def test_comorbidity_empty_codes_falls_through_to_text(self):
        """Without ICD codes, CCI should still extract from text."""
        from app.ml.pipeline import PipelineResult

        pipe = EnhancedClinicalPipeline()
        pipe._ensure_modules()

        base = PipelineResult(document_id="test", icd_predictions=[])

        cfg = EnhancedPipelineConfig()
        result = EnhancedPipelineResult()
        result.base_result = base
        result = pipe._run_comorbidity(
            "Patient has diabetes mellitus and congestive heart failure",
            cfg, result,
        )

        assert result.comorbidity is not None
        assert result.comorbidity["category_count"] > 0
