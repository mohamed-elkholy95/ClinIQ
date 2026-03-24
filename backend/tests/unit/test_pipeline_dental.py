"""Unit tests for ClinicalPipeline dental analysis and batch processing.

Targets the uncovered _run_dental method (lines 386–429), load error
handling (lines 194–196, 202), batch processing (line 267),
_components helper, and _collect_model_versions helper in pipeline.py.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.ml.dental.model import DentalAssessment
from app.ml.icd.model import ICDCodePrediction, ICDPredictionResult
from app.ml.ner.model import Entity
from app.ml.pipeline import ClinicalPipeline, PipelineConfig, PipelineResult
from app.ml.risk.model import RiskAssessment
from app.ml.summarization.model import SummarizationResult


SAMPLE_TEXT = (
    "Patient presents with moderate periodontitis affecting teeth #18 and #19. "
    "Bleeding on probing noted. Recommend D4341 scaling and root planing."
)




# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_dental_model() -> MagicMock:
    """Provide a mock dental NER model."""
    model = MagicMock()
    model.is_loaded = True
    model.version = "dental-1.0.0"
    model.model_name = "dental-ner"
    model.ensure_loaded = MagicMock()
    model.extract_entities.return_value = [
        Entity(
            text="tooth #18",
            entity_type="TOOTH",
            start_char=55,
            end_char=64,
            confidence=0.92,
            metadata={"cdt_code": "D4341"},
        ),
        Entity(
            text="periodontitis",
            entity_type="CONDITION",
            start_char=28,
            end_char=41,
            confidence=0.88,
            metadata={},
        ),
    ]
    return model


@pytest.fixture
def mock_perio_assessor() -> MagicMock:
    """Provide a mock periodontal risk assessor."""
    assessor = MagicMock()
    assessor.ensure_loaded = MagicMock()
    assessor.version = "perio-1.0.0"
    assessor.assess.return_value = {
        "risk_score": 65.0,
        "classification": "Moderate",
        "recommendations": [
            "Schedule follow-up in 3 months",
            "Consider adjunctive antibiotic therapy",
        ],
        "processing_time_ms": 15.0,
    }
    return assessor


@pytest.fixture
def dental_pipeline(
    mock_dental_model: MagicMock,
    mock_perio_assessor: MagicMock,
) -> ClinicalPipeline:
    """Pipeline with dental components only."""
    pipeline = ClinicalPipeline(
        dental_model=mock_dental_model,
        perio_assessor=mock_perio_assessor,
    )
    pipeline._is_loaded = True
    return pipeline


# ---------------------------------------------------------------------------
# _run_dental tests
# ---------------------------------------------------------------------------


class TestRunDental:
    """Tests for the _run_dental component runner."""

    def test_dental_assessment_populated(
        self, dental_pipeline: ClinicalPipeline
    ) -> None:
        """Dental processing should populate dental_assessment."""
        config = PipelineConfig(
            enable_ner=False,
            enable_icd=False,
            enable_summarization=False,
            enable_risk=False,
            enable_dental=True,
        )
        result = dental_pipeline.process(SAMPLE_TEXT, config=config)

        assert result.dental_assessment is not None
        assert isinstance(result.dental_assessment, DentalAssessment)

    def test_dental_entities_extracted(
        self, dental_pipeline: ClinicalPipeline
    ) -> None:
        """Dental entities should be present in the assessment."""
        config = PipelineConfig(
            enable_ner=False,
            enable_icd=False,
            enable_summarization=False,
            enable_risk=False,
            enable_dental=True,
        )
        result = dental_pipeline.process(SAMPLE_TEXT, config=config)

        assert len(result.dental_assessment.entities) == 2

    def test_periodontal_risk_score(
        self, dental_pipeline: ClinicalPipeline
    ) -> None:
        """Perio risk score should come from the assessor."""
        config = PipelineConfig(
            enable_ner=False,
            enable_icd=False,
            enable_summarization=False,
            enable_risk=False,
            enable_dental=True,
        )
        result = dental_pipeline.process(SAMPLE_TEXT, config=config)

        assert result.dental_assessment.periodontal_risk_score == 65.0
        assert result.dental_assessment.periodontal_classification == "Moderate"

    def test_cdt_codes_collected(
        self, dental_pipeline: ClinicalPipeline
    ) -> None:
        """CDT codes from entity metadata should be collected via class attr."""
        # Set CDT_CODES class attribute on the mock dental model so the
        # pipeline's reverse-lookup can find the code.
        dental_pipeline._dental_model.CDT_CODES = {
            "scaling": {"code": "D4341", "description": "Periodontal scaling - quadrant"},
        }
        config = PipelineConfig(
            enable_ner=False,
            enable_icd=False,
            enable_summarization=False,
            enable_risk=False,
            enable_dental=True,
        )
        result = dental_pipeline.process(SAMPLE_TEXT, config=config)

        assert "D4341" in result.dental_assessment.cdt_codes

    def test_dental_recommendations(
        self, dental_pipeline: ClinicalPipeline
    ) -> None:
        """Recommendations from perio assessor should be included."""
        config = PipelineConfig(
            enable_ner=False,
            enable_icd=False,
            enable_summarization=False,
            enable_risk=False,
            enable_dental=True,
        )
        result = dental_pipeline.process(SAMPLE_TEXT, config=config)

        assert len(result.dental_assessment.recommendations) == 2

    def test_dental_model_metadata(
        self, dental_pipeline: ClinicalPipeline
    ) -> None:
        """Model name and version should be set from the dental model."""
        config = PipelineConfig(
            enable_ner=False,
            enable_icd=False,
            enable_summarization=False,
            enable_risk=False,
            enable_dental=True,
        )
        result = dental_pipeline.process(SAMPLE_TEXT, config=config)

        assert result.dental_assessment.model_name == "dental-ner"
        assert result.dental_assessment.model_version == "dental-1.0.0"

    def test_dental_without_perio_assessor(
        self, mock_dental_model: MagicMock
    ) -> None:
        """Pipeline with dental model but no perio assessor should work."""
        pipeline = ClinicalPipeline(dental_model=mock_dental_model)
        pipeline._is_loaded = True
        config = PipelineConfig(
            enable_ner=False,
            enable_icd=False,
            enable_summarization=False,
            enable_risk=False,
            enable_dental=True,
        )
        result = pipeline.process(SAMPLE_TEXT, config=config)

        assert result.dental_assessment is not None
        assert result.dental_assessment.periodontal_risk_score == 0.0
        assert result.dental_assessment.periodontal_classification == "Unknown"
        assert result.dental_assessment.recommendations == []

    def test_dental_failure_captured_in_errors(
        self, dental_pipeline: ClinicalPipeline
    ) -> None:
        """Dental component failure → error in component_errors, not exception."""
        dental_pipeline._dental_model.extract_entities.side_effect = RuntimeError(
            "Dental crash"
        )
        config = PipelineConfig(
            enable_ner=False,
            enable_icd=False,
            enable_summarization=False,
            enable_risk=False,
            enable_dental=True,
        )
        result = dental_pipeline.process(SAMPLE_TEXT, config=config)

        assert "dental" in result.component_errors
        assert "Dental crash" in result.component_errors["dental"]
        assert result.dental_assessment is None

    def test_dental_disabled_skips_processing(
        self, dental_pipeline: ClinicalPipeline
    ) -> None:
        """Dental disabled → no dental processing."""
        config = PipelineConfig(enable_dental=False)
        result = dental_pipeline.process(SAMPLE_TEXT, config=config)
        dental_pipeline._dental_model.extract_entities.assert_not_called()


# ---------------------------------------------------------------------------
# load() error handling
# ---------------------------------------------------------------------------


class TestPipelineLoad:
    """Tests for ClinicalPipeline.load() including component failures."""

    def test_load_with_failing_component(self) -> None:
        """A component that fails to load should be logged, not crash."""
        bad_model = MagicMock()
        bad_model.ensure_loaded.side_effect = RuntimeError("Model file not found")
        bad_model.version = "0.0.0"

        pipeline = ClinicalPipeline(ner_model=bad_model)
        # Should not raise
        pipeline.load()

        assert pipeline.is_loaded is True

    def test_load_multiple_failures_still_completes(self) -> None:
        """Multiple component failures during load should all be captured."""
        bad_ner = MagicMock()
        bad_ner.ensure_loaded.side_effect = RuntimeError("NER fail")
        bad_ner.version = "0.0.0"

        bad_icd = MagicMock()
        bad_icd.ensure_loaded.side_effect = RuntimeError("ICD fail")
        bad_icd.version = "0.0.0"

        pipeline = ClinicalPipeline(ner_model=bad_ner, icd_classifier=bad_icd)
        pipeline.load()

        assert pipeline.is_loaded is True

    def test_ensure_loaded_only_calls_load_once(self) -> None:
        """ensure_loaded should not call load if already loaded."""
        mock_ner = MagicMock()
        mock_ner.version = "1.0.0"
        pipeline = ClinicalPipeline(ner_model=mock_ner)
        pipeline.load()
        # Reset the ensure_loaded call count
        mock_ner.ensure_loaded.reset_mock()

        pipeline.ensure_loaded()
        # Should not have called ensure_loaded again on the component
        mock_ner.ensure_loaded.assert_not_called()


# ---------------------------------------------------------------------------
# _components and _collect_model_versions helpers
# ---------------------------------------------------------------------------


class TestPipelineHelpers:
    """Tests for _components and _collect_model_versions."""

    def test_components_lists_all_non_none(self) -> None:
        """_components should return all injected components."""
        mock_ner = MagicMock()
        mock_ner.ensure_loaded = MagicMock()
        mock_icd = MagicMock()
        mock_icd.ensure_loaded = MagicMock()

        pipeline = ClinicalPipeline(ner_model=mock_ner, icd_classifier=mock_icd)
        components = pipeline._components()

        names = [name for name, _ in components]
        assert "ner" in names
        assert "icd" in names
        assert len(components) == 2

    def test_components_empty_pipeline(self) -> None:
        """Empty pipeline → empty components list."""
        pipeline = ClinicalPipeline()
        assert pipeline._components() == []

    def test_collect_model_versions_populated(self) -> None:
        """_collect_model_versions should gather version strings."""
        mock_ner = MagicMock()
        mock_ner.version = "ner-2.1.0"
        mock_summarizer = MagicMock()
        mock_summarizer.version = "sum-1.0.0"

        pipeline = ClinicalPipeline(
            ner_model=mock_ner, summarizer=mock_summarizer
        )
        versions = pipeline._collect_model_versions()

        assert versions["ner"] == "ner-2.1.0"
        assert versions["summarizer"] == "sum-1.0.0"
        assert "icd" not in versions

    def test_collect_model_versions_no_version_attr(self) -> None:
        """Components without .version should be skipped."""
        mock = MagicMock(spec=[])  # No attributes
        pipeline = ClinicalPipeline(ner_model=mock)
        versions = pipeline._collect_model_versions()
        assert "ner" not in versions

    def test_components_includes_dental_and_perio(self) -> None:
        """Dental and perio components should appear in _components."""
        mock_dental = MagicMock()
        mock_dental.ensure_loaded = MagicMock()
        mock_perio = MagicMock()
        mock_perio.ensure_loaded = MagicMock()

        pipeline = ClinicalPipeline(
            dental_model=mock_dental, perio_assessor=mock_perio
        )
        names = [name for name, _ in pipeline._components()]
        assert "dental" in names
        assert "perio" in names


# ---------------------------------------------------------------------------
# Batch processing edge cases
# ---------------------------------------------------------------------------


class TestPipelineBatch:
    """Edge cases for process_batch."""

    def test_batch_empty_list(self) -> None:
        """Empty input list → empty results."""
        pipeline = ClinicalPipeline()
        pipeline._is_loaded = True
        results = pipeline.process_batch([])
        assert results == []

    def test_batch_without_document_ids(self) -> None:
        """Batch without IDs → document_id is None for each result."""
        pipeline = ClinicalPipeline()
        pipeline._is_loaded = True
        results = pipeline.process_batch(["Text A.", "Text B."])
        assert len(results) == 2
        assert all(r.document_id is None for r in results)

    def test_batch_with_config(self) -> None:
        """Shared config should be passed to each process call."""
        mock_ner = MagicMock()
        mock_ner.version = "1.0"
        mock_ner.ensure_loaded = MagicMock()
        mock_ner.extract_entities.return_value = []

        pipeline = ClinicalPipeline(ner_model=mock_ner)
        pipeline._is_loaded = True

        config = PipelineConfig(enable_icd=False, enable_summarization=False, enable_risk=False)
        results = pipeline.process_batch(
            ["Text 1.", "Text 2."], config=config
        )
        assert len(results) == 2
        assert mock_ner.extract_entities.call_count == 2
