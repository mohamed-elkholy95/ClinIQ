"""Unit tests for the ClinicalPipeline orchestrator."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.ml.icd.model import ICDCodePrediction, ICDPredictionResult
from app.ml.ner.model import Entity
from app.ml.pipeline import ClinicalPipeline, PipelineConfig, PipelineResult
from app.ml.risk.model import RiskAssessment, RiskFactor
from app.ml.summarization.model import SummarizationResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_ner_model() -> MagicMock:
    """Provide a mock NER model that returns two entities."""
    model = MagicMock()
    model.is_loaded = True
    model.version = "1.0.0"
    model.ensure_loaded = MagicMock()
    model.extract_entities.return_value = [
        Entity(
            text="metformin",
            entity_type="MEDICATION",
            start_char=10,
            end_char=19,
            confidence=0.95,
        ),
        Entity(
            text="diabetes",
            entity_type="DISEASE",
            start_char=30,
            end_char=38,
            confidence=0.90,
        ),
    ]
    return model


@pytest.fixture
def mock_icd_classifier() -> MagicMock:
    """Provide a mock ICD classifier."""
    model = MagicMock()
    model.is_loaded = True
    model.version = "1.0.0"
    model.ensure_loaded = MagicMock()
    model.predict.return_value = ICDPredictionResult(
        predictions=[
            ICDCodePrediction(
                code="E11.9",
                description="Type 2 diabetes mellitus without complications",
                confidence=0.85,
                chapter="Endocrine, nutritional and metabolic diseases",
            ),
        ],
        processing_time_ms=50.0,
        model_name="test-icd",
        model_version="1.0.0",
    )
    return model


@pytest.fixture
def mock_summarizer() -> MagicMock:
    """Provide a mock summarizer."""
    model = MagicMock()
    model.is_loaded = True
    model.version = "1.0.0"
    model.ensure_loaded = MagicMock()
    model.summarize.return_value = SummarizationResult(
        summary="Patient has diabetes managed with metformin.",
        key_findings=["Diabetes well controlled."],
        detail_level="standard",
        processing_time_ms=100.0,
        model_name="test-summarizer",
        model_version="1.0.0",
        sentence_count_original=10,
        sentence_count_summary=2,
    )
    return model


@pytest.fixture
def mock_risk_scorer() -> MagicMock:
    """Provide a mock risk scorer."""
    model = MagicMock()
    model.is_loaded = True
    model.version = "1.0.0"
    model.ensure_loaded = MagicMock()
    model.assess_risk.return_value = RiskAssessment(
        overall_score=30.0,
        risk_level="low",
        factors=[],
        recommendations=["Routine follow-up in 3 months."],
        processing_time_ms=10.0,
        category_scores={
            "medication_risk": 20.0,
            "diagnostic_complexity": 35.0,
            "follow_up_urgency": 10.0,
        },
        model_name="test-risk",
        model_version="1.0.0",
    )
    return model


@pytest.fixture
def full_pipeline(
    mock_ner_model: MagicMock,
    mock_icd_classifier: MagicMock,
    mock_summarizer: MagicMock,
    mock_risk_scorer: MagicMock,
) -> ClinicalPipeline:
    """Provide a ClinicalPipeline with all mock components injected."""
    pipeline = ClinicalPipeline(
        ner_model=mock_ner_model,
        icd_classifier=mock_icd_classifier,
        summarizer=mock_summarizer,
        risk_scorer=mock_risk_scorer,
    )
    pipeline._is_loaded = True
    return pipeline


SAMPLE_TEXT = (
    "Patient is a 55-year-old male with type 2 diabetes mellitus "
    "on metformin 1000mg twice daily. Blood glucose is well controlled."
)


# ---------------------------------------------------------------------------
# PipelineConfig tests
# ---------------------------------------------------------------------------


class TestPipelineConfig:
    """Tests for the PipelineConfig dataclass."""

    def test_default_enable_ner(self):
        """Test that enable_ner defaults to True."""
        cfg = PipelineConfig()
        assert cfg.enable_ner is True

    def test_default_enable_icd(self):
        """Test that enable_icd defaults to True."""
        cfg = PipelineConfig()
        assert cfg.enable_icd is True

    def test_default_enable_summarization(self):
        """Test that enable_summarization defaults to True."""
        cfg = PipelineConfig()
        assert cfg.enable_summarization is True

    def test_default_enable_risk(self):
        """Test that enable_risk defaults to True."""
        cfg = PipelineConfig()
        assert cfg.enable_risk is True

    def test_default_enable_dental_is_false(self):
        """Test that enable_dental defaults to False."""
        cfg = PipelineConfig()
        assert cfg.enable_dental is False

    def test_default_confidence_threshold(self):
        """Test default confidence threshold."""
        cfg = PipelineConfig()
        assert cfg.confidence_threshold == 0.5

    def test_default_top_k_icd(self):
        """Test default top_k_icd."""
        cfg = PipelineConfig()
        assert cfg.top_k_icd == 10

    def test_default_detail_level(self):
        """Test that default detail_level is 'standard'."""
        cfg = PipelineConfig()
        assert cfg.detail_level == "standard"

    def test_custom_config(self):
        """Test customising pipeline configuration."""
        cfg = PipelineConfig(
            enable_ner=True,
            enable_icd=False,
            enable_summarization=False,
            enable_risk=False,
            confidence_threshold=0.7,
            top_k_icd=5,
            detail_level="brief",
        )

        assert cfg.enable_ner is True
        assert cfg.enable_icd is False
        assert cfg.enable_summarization is False
        assert cfg.confidence_threshold == 0.7
        assert cfg.top_k_icd == 5
        assert cfg.detail_level == "brief"


# ---------------------------------------------------------------------------
# PipelineResult tests
# ---------------------------------------------------------------------------


class TestPipelineResult:
    """Tests for the PipelineResult dataclass."""

    def test_creation_minimal(self):
        """Test creating a PipelineResult with just document_id."""
        result = PipelineResult(document_id="doc-001")
        assert result.document_id == "doc-001"

    def test_default_empty_lists(self):
        """Test that list fields default to empty lists."""
        result = PipelineResult(document_id=None)
        assert result.entities == []
        assert result.icd_predictions == []

    def test_default_none_fields(self):
        """Test that optional component results default to None."""
        result = PipelineResult(document_id=None)
        assert result.summary is None
        assert result.risk_assessment is None
        assert result.dental_assessment is None

    def test_default_processing_time(self):
        """Test that processing_time_ms defaults to 0.0."""
        result = PipelineResult(document_id=None)
        assert result.processing_time_ms == 0.0

    def test_to_dict_structure(self):
        """Test the structure of to_dict() output."""
        result = PipelineResult(document_id="doc-001")
        d = result.to_dict()

        expected_keys = {
            "document_id",
            "entities",
            "icd_predictions",
            "summary",
            "risk_assessment",
            "dental_assessment",
            "processing_time_ms",
            "model_versions",
            "component_errors",
        }
        assert expected_keys.issubset(set(d.keys()))

    def test_to_dict_entities_serialised(self):
        """Test that entities are serialised as dicts in to_dict()."""
        result = PipelineResult(
            document_id="doc-001",
            entities=[
                Entity("metformin", "MEDICATION", 0, 9, 0.9),
            ],
        )
        d = result.to_dict()
        assert isinstance(d["entities"], list)
        assert isinstance(d["entities"][0], dict)


# ---------------------------------------------------------------------------
# ClinicalPipeline tests
# ---------------------------------------------------------------------------


class TestClinicalPipeline:
    """Tests for ClinicalPipeline."""

    def test_creation_with_no_components(self):
        """Test that ClinicalPipeline can be created with no components."""
        pipeline = ClinicalPipeline()
        assert pipeline._ner_model is None
        assert pipeline._icd_classifier is None
        assert pipeline._summarizer is None
        assert pipeline._risk_scorer is None
        assert pipeline._dental_model is None

    def test_is_loaded_false_initially(self):
        """Test that is_loaded is False before load()."""
        pipeline = ClinicalPipeline()
        assert pipeline.is_loaded is False

    def test_load_sets_is_loaded(self, mock_ner_model: MagicMock):
        """Test that load() sets is_loaded to True."""
        pipeline = ClinicalPipeline(ner_model=mock_ner_model)
        pipeline.load()
        assert pipeline.is_loaded is True

    def test_ensure_loaded_triggers_load(self, mock_ner_model: MagicMock):
        """Test that ensure_loaded() calls load() when not yet loaded."""
        pipeline = ClinicalPipeline(ner_model=mock_ner_model)
        pipeline.ensure_loaded()
        assert pipeline.is_loaded is True

    def test_process_returns_pipeline_result(self, full_pipeline: ClinicalPipeline):
        """Test that process() returns a PipelineResult."""
        result = full_pipeline.process(SAMPLE_TEXT)
        assert isinstance(result, PipelineResult)

    def test_process_extracts_entities(self, full_pipeline: ClinicalPipeline):
        """Test that process() populates entities when NER is enabled."""
        config = PipelineConfig(enable_ner=True)
        result = full_pipeline.process(SAMPLE_TEXT, config=config)

        assert len(result.entities) > 0

    def test_process_confidence_threshold_filters(
        self, full_pipeline: ClinicalPipeline, mock_ner_model: MagicMock
    ):
        """Test that entities below confidence_threshold are filtered out."""
        mock_ner_model.extract_entities.return_value = [
            Entity("metformin", "MEDICATION", 0, 9, 0.95),
            Entity("low-conf-entity", "DISEASE", 10, 25, 0.3),  # Below threshold
        ]
        config = PipelineConfig(confidence_threshold=0.5)
        result = full_pipeline.process(SAMPLE_TEXT, config=config)

        # Only the high-confidence entity should pass the threshold
        assert all(e.confidence >= 0.5 for e in result.entities)

    def test_process_icd_predictions(self, full_pipeline: ClinicalPipeline):
        """Test that process() populates icd_predictions when ICD is enabled."""
        config = PipelineConfig(enable_icd=True)
        result = full_pipeline.process(SAMPLE_TEXT, config=config)

        assert len(result.icd_predictions) > 0
        assert result.icd_predictions[0]["code"] == "E11.9"

    def test_process_summarization(self, full_pipeline: ClinicalPipeline):
        """Test that process() populates summary when summarization is enabled."""
        config = PipelineConfig(enable_summarization=True)
        result = full_pipeline.process(SAMPLE_TEXT, config=config)

        assert result.summary is not None
        assert isinstance(result.summary, SummarizationResult)

    def test_process_risk_assessment(self, full_pipeline: ClinicalPipeline):
        """Test that process() populates risk_assessment when risk is enabled."""
        config = PipelineConfig(enable_risk=True)
        result = full_pipeline.process(SAMPLE_TEXT, config=config)

        assert result.risk_assessment is not None
        assert isinstance(result.risk_assessment, RiskAssessment)

    def test_process_ner_disabled(self, full_pipeline: ClinicalPipeline):
        """Test that NER is skipped when enable_ner=False."""
        config = PipelineConfig(enable_ner=False)
        result = full_pipeline.process(SAMPLE_TEXT, config=config)

        assert result.entities == []
        full_pipeline._ner_model.extract_entities.assert_not_called()

    def test_process_icd_disabled(self, full_pipeline: ClinicalPipeline):
        """Test that ICD is skipped when enable_icd=False."""
        config = PipelineConfig(enable_icd=False)
        result = full_pipeline.process(SAMPLE_TEXT, config=config)

        assert result.icd_predictions == []
        full_pipeline._icd_classifier.predict.assert_not_called()

    def test_process_summarization_disabled(self, full_pipeline: ClinicalPipeline):
        """Test that summarization is skipped when enable_summarization=False."""
        config = PipelineConfig(enable_summarization=False)
        result = full_pipeline.process(SAMPLE_TEXT, config=config)

        assert result.summary is None
        full_pipeline._summarizer.summarize.assert_not_called()

    def test_process_risk_disabled(self, full_pipeline: ClinicalPipeline):
        """Test that risk scoring is skipped when enable_risk=False."""
        config = PipelineConfig(enable_risk=False)
        result = full_pipeline.process(SAMPLE_TEXT, config=config)

        assert result.risk_assessment is None
        full_pipeline._risk_scorer.assess_risk.assert_not_called()

    def test_process_processing_time_set(self, full_pipeline: ClinicalPipeline):
        """Test that processing_time_ms is set after processing."""
        result = full_pipeline.process(SAMPLE_TEXT)
        assert result.processing_time_ms > 0.0

    def test_process_document_id_preserved(self, full_pipeline: ClinicalPipeline):
        """Test that the document_id is preserved in the result."""
        result = full_pipeline.process(SAMPLE_TEXT, document_id="note-42")
        assert result.document_id == "note-42"

    def test_process_model_versions_populated(self, full_pipeline: ClinicalPipeline):
        """Test that model_versions is populated after processing."""
        result = full_pipeline.process(SAMPLE_TEXT)
        assert isinstance(result.model_versions, dict)

    def test_partial_result_on_ner_failure(self, full_pipeline: ClinicalPipeline):
        """Test that a NER failure produces a partial result, not an exception."""
        full_pipeline._ner_model.extract_entities.side_effect = RuntimeError("NER crash")

        # Should not raise
        result = full_pipeline.process(SAMPLE_TEXT)

        assert "ner" in result.component_errors
        assert "NER crash" in result.component_errors["ner"]
        # Other components should still have run
        assert len(result.icd_predictions) > 0 or "icd" in result.component_errors

    def test_partial_result_on_icd_failure(self, full_pipeline: ClinicalPipeline):
        """Test that an ICD failure produces a partial result."""
        full_pipeline._icd_classifier.predict.side_effect = RuntimeError("ICD crash")

        result = full_pipeline.process(SAMPLE_TEXT)

        assert "icd" in result.component_errors
        # Entities from NER should still be present
        assert len(result.entities) > 0

    def test_partial_result_on_summarization_failure(
        self, full_pipeline: ClinicalPipeline
    ):
        """Test that a summarization failure produces a partial result."""
        full_pipeline._summarizer.summarize.side_effect = RuntimeError("Summarizer crash")

        result = full_pipeline.process(SAMPLE_TEXT)

        assert "summarization" in result.component_errors
        assert result.summary is None

    def test_partial_result_on_risk_failure(self, full_pipeline: ClinicalPipeline):
        """Test that a risk scoring failure produces a partial result."""
        full_pipeline._risk_scorer.assess_risk.side_effect = RuntimeError("Risk crash")

        result = full_pipeline.process(SAMPLE_TEXT)

        assert "risk" in result.component_errors
        assert result.risk_assessment is None

    def test_process_batch_returns_list(self, full_pipeline: ClinicalPipeline):
        """Test that process_batch() returns a list of PipelineResults."""
        texts = [
            "Patient has diabetes.",
            "History of hypertension.",
            "Acute appendicitis.",
        ]
        results = full_pipeline.process_batch(texts)

        assert isinstance(results, list)
        assert len(results) == 3
        for result in results:
            assert isinstance(result, PipelineResult)

    def test_process_batch_with_document_ids(self, full_pipeline: ClinicalPipeline):
        """Test that process_batch() preserves document IDs."""
        texts = ["Text 1.", "Text 2."]
        ids = ["doc-001", "doc-002"]
        results = full_pipeline.process_batch(texts, document_ids=ids)

        assert results[0].document_id == "doc-001"
        assert results[1].document_id == "doc-002"

    def test_process_with_none_config_uses_defaults(
        self, full_pipeline: ClinicalPipeline
    ):
        """Test that None config uses PipelineConfig defaults."""
        result = full_pipeline.process(SAMPLE_TEXT, config=None)
        assert isinstance(result, PipelineResult)

    def test_pipeline_with_no_components_returns_empty_result(self):
        """Test that a pipeline with no components returns an empty result."""
        pipeline = ClinicalPipeline()
        pipeline._is_loaded = True

        result = pipeline.process(SAMPLE_TEXT)

        assert result.entities == []
        assert result.icd_predictions == []
        assert result.summary is None
        assert result.risk_assessment is None
        assert result.component_errors == {}

    def test_icd_predictions_serialised_as_dicts(
        self, full_pipeline: ClinicalPipeline
    ):
        """Test that ICD predictions in the result are serialised as dicts."""
        config = PipelineConfig(enable_icd=True)
        result = full_pipeline.process(SAMPLE_TEXT, config=config)

        for pred in result.icd_predictions:
            assert isinstance(pred, dict)
            assert "code" in pred
