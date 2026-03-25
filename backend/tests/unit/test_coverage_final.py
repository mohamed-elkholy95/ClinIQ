"""Final coverage push — targeted tests for remaining uncovered lines.

Covers edge cases in NER composite model, ICD batch predict, dental extraction,
main.py exception handlers, summarization error propagation, drift detector
branches, metrics collector gauge methods, and document service.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.exceptions import InferenceError
from app.ml.dental.model import DentalNERModel
from app.ml.icd.model import SklearnICDClassifier
from app.ml.monitoring.drift_detector import PredictionMonitor, TextDistributionMonitor
from app.ml.monitoring.metrics_collector import ModelMetrics
from app.ml.ner.model import CompositeNERModel, Entity, TransformerNERModel
from app.ml.summarization.model import ExtractiveSummarizer

# ---------------------------------------------------------------------------
# 1. NER — TransformerNER BIO tag continuation, CompositeNER intersection/overlap
# ---------------------------------------------------------------------------

class TestNERModelEdgeCases:
    """Cover ner/model.py lines 405-415, 439, 526, 585-586."""

    def test_transformer_ner_bio_beginning_tag_starts_new_entity(self):
        """When a B- tag is found while a current entity exists, the current
        entity is finalized and a new one starts (line ~439)."""
        model = TransformerNERModel.__new__(TransformerNERModel)
        model.model_name = "test-bio"
        model.version = "1.0"
        model.label_map = {0: "O", 1: "B-DISEASE", 2: "I-DISEASE", 3: "B-DRUG", 4: "I-DRUG"}

        # B-DISEASE at chars 0-5, B-DRUG at chars 6-13
        predictions = np.array([1, 3])
        offsets = np.array([[0, 5], [6, 13]])
        text = "fever aspirin"

        entities = model._extract_from_bio_tags(predictions, offsets, text)
        assert len(entities) == 2
        assert entities[0].entity_type == "DISEASE"
        assert entities[1].entity_type == "DRUG"

    def test_transformer_ner_bio_continuation_tag(self):
        """I- tag continues the current entity."""
        model = TransformerNERModel.__new__(TransformerNERModel)
        model.model_name = "test-bio"
        model.version = "1.0"
        model.label_map = {0: "O", 1: "B-DISEASE", 2: "I-DISEASE"}

        # "heart" chars 0-5, " failure" chars 5-13
        predictions = np.array([1, 2])
        offsets = np.array([[0, 5], [5, 13]])
        text = "heart failure"

        entities = model._extract_from_bio_tags(predictions, offsets, text)
        assert len(entities) == 1
        assert entities[0].text == "heart failure"

    def test_transformer_ner_bio_outside_after_entity(self):
        """O tag after an active entity finalizes it."""
        model = TransformerNERModel.__new__(TransformerNERModel)
        model.model_name = "test-bio"
        model.version = "1.0"
        model.label_map = {0: "O", 1: "B-DISEASE", 2: "I-DISEASE"}

        predictions = np.array([1, 0])
        offsets = np.array([[0, 5], [6, 10]])
        text = "fever with"

        entities = model._extract_from_bio_tags(predictions, offsets, text)
        assert len(entities) == 1
        assert entities[0].text == "fever"

    def test_transformer_ner_bio_special_token_skip(self):
        """Tokens where start == end (special tokens) are skipped."""
        model = TransformerNERModel.__new__(TransformerNERModel)
        model.model_name = "test-bio"
        model.version = "1.0"
        model.label_map = {0: "O", 1: "B-DISEASE"}

        # Special token at position 0, real token at 1
        predictions = np.array([0, 1])
        offsets = np.array([[0, 0], [0, 5]])
        text = "fever"

        entities = model._extract_from_bio_tags(predictions, offsets, text)
        assert len(entities) == 1

    def test_composite_intersection_vote_returns_common_entities(self):
        """_intersection_vote returns entities found by all models (line ~526)."""
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        model = CompositeNERModel(
            models=[mock_model1, mock_model2],
            voting="intersection",
        )

        e1 = Entity(text="aspirin", entity_type="DRUG", start_char=0, end_char=7, confidence=0.9)
        e2 = Entity(text="fever", entity_type="SYMPTOM", start_char=10, end_char=15, confidence=0.8)

        # Model 1 finds both, Model 2 only finds aspirin
        mock_model1.extract_entities.return_value = [e1, e2]
        mock_model2.extract_entities.return_value = [
            Entity(text="aspirin", entity_type="DRUG", start_char=0, end_char=7, confidence=0.85),
        ]

        result = model.extract_entities("aspirin for fever")
        assert len(result) == 1
        assert result[0].text == "aspirin"

    def test_composite_intersection_vote_empty(self):
        """_intersection_vote returns empty for no input."""
        model = CompositeNERModel.__new__(CompositeNERModel)
        model.model_name = "composite"
        result = model._intersection_vote([])
        assert result == []

    def test_composite_overlap_resolution_keeps_non_overlapping(self):
        """_resolve_overlaps keeps non-overlapping, removes overlapping (lines 585-586)."""
        model = CompositeNERModel.__new__(CompositeNERModel)

        e1 = Entity(text="heart", entity_type="ANATOMY", start_char=0, end_char=5, confidence=0.95)
        e2 = Entity(text="heart failure", entity_type="DISEASE", start_char=0, end_char=13, confidence=0.8)
        e3 = Entity(text="aspirin", entity_type="DRUG", start_char=20, end_char=27, confidence=0.9)

        # Sorted by start_char, then -confidence: e1(0), e2(0), e3(20)
        resolved = model._resolve_overlaps([e1, e2, e3])
        # e1 first (start=0, higher conf), e2 overlaps with e1 → dropped, e3 no overlap → kept
        assert len(resolved) == 2
        texts = {e.text for e in resolved}
        assert "heart" in texts
        assert "aspirin" in texts

    def test_composite_majority_vote(self):
        """_majority_vote returns entities found by > half the models."""
        mock1 = MagicMock()
        mock2 = MagicMock()
        mock3 = MagicMock()
        model = CompositeNERModel(models=[mock1, mock2, mock3], voting="majority")

        e_common = Entity(text="aspirin", entity_type="DRUG", start_char=0, end_char=7, confidence=0.9)
        e_rare = Entity(text="fever", entity_type="SYMPTOM", start_char=10, end_char=15, confidence=0.8)

        mock1.extract_entities.return_value = [e_common, e_rare]
        mock2.extract_entities.return_value = [
            Entity(text="aspirin", entity_type="DRUG", start_char=0, end_char=7, confidence=0.88),
        ]
        mock3.extract_entities.return_value = [
            Entity(text="aspirin", entity_type="DRUG", start_char=0, end_char=7, confidence=0.85),
        ]

        result = model.extract_entities("aspirin for fever")
        assert any(e.text == "aspirin" for e in result)


# ---------------------------------------------------------------------------
# 2. ICD — SklearnICDClassifier 1D proba reshape and batch predict
# ---------------------------------------------------------------------------

class TestICDModelEdgeCases:
    """Cover icd/model.py lines 226, 231, 246-247, 266-269, 291-292."""

    def _make_sklearn_icd(self):
        """Helper to create a properly initialized SklearnICDClassifier."""
        model = SklearnICDClassifier.__new__(SklearnICDClassifier)
        model.model_name = "test-icd"
        model.version = "1.0"
        model._is_loaded = True
        model.code_descriptions = {}
        # label_binarizer with classes_ attribute for label name lookup
        mock_lb = MagicMock()
        mock_lb.classes_ = np.array(["E11.9", "I10"])
        model.label_binarizer = mock_lb
        return model

    def test_predict_1d_proba_reshape(self):
        """When predict_proba returns 1D array, it's reshaped to 2D (line ~226)."""
        model = self._make_sklearn_icd()

        mock_classifier = MagicMock()
        mock_classifier.predict_proba.return_value = np.array([0.8, 0.2])
        model.classifier = mock_classifier

        mock_extractor = MagicMock()
        mock_extractor.transform.return_value = np.array([[1.0, 0.0]])
        model.feature_extractor = mock_extractor

        result = model.predict("diabetes", top_k=2)
        assert result.predictions[0].code == "E11.9"
        assert result.predictions[0].confidence == pytest.approx(0.8)

    def test_predict_decision_function_sigmoid(self):
        """decision_function fallback with sigmoid transform (lines 230-233)."""
        model = self._make_sklearn_icd()

        mock_classifier = MagicMock(spec=["decision_function"])
        mock_classifier.decision_function.return_value = np.array([[2.0, -2.0]])
        model.classifier = mock_classifier

        mock_extractor = MagicMock()
        mock_extractor.transform.return_value = np.array([[1.0, 0.0]])
        model.feature_extractor = mock_extractor

        result = model.predict("diabetes", top_k=2)
        assert result.predictions[0].code == "E11.9"
        assert result.predictions[0].confidence > 0.8

    def test_batch_predict(self):
        """Batch predict processes multiple texts and distributes avg time (lines 266-292)."""
        model = self._make_sklearn_icd()

        mock_classifier = MagicMock()
        mock_classifier.predict_proba.return_value = np.array([
            [0.9, 0.1],
            [0.3, 0.7],
        ])
        model.classifier = mock_classifier

        mock_extractor = MagicMock()
        mock_extractor.transform.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
        model.feature_extractor = mock_extractor

        results = model.predict_batch(["diabetes", "hypertension"], top_k=2)
        assert len(results) == 2
        assert results[0].predictions[0].code == "E11.9"
        assert results[1].predictions[0].code == "I10"
        assert results[0].processing_time_ms == results[1].processing_time_ms

    def test_batch_predict_decision_function_with_list(self):
        """Batch predict with decision_function returning a plain list (lines 270-273)."""
        model = self._make_sklearn_icd()

        mock_classifier = MagicMock(spec=["decision_function"])
        mock_classifier.decision_function.return_value = [[2.0, -2.0], [-1.0, 1.5]]
        model.classifier = mock_classifier

        mock_extractor = MagicMock()
        mock_extractor.transform.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
        model.feature_extractor = mock_extractor

        results = model.predict_batch(["diabetes", "hypertension"], top_k=2)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# 3. Dental model — quadrant mapping and surface validation
# ---------------------------------------------------------------------------

class TestDentalModelEdgeCases:
    """Cover dental/model.py lines 376, 503-504."""

    def test_quadrant_lower_left(self):
        """Tooth numbers 17-24 map to quadrant 3 (line 376)."""
        model = DentalNERModel.__new__(DentalNERModel)
        assert model._get_quadrant_universal(20) == 3

    def test_quadrant_lower_right(self):
        """Tooth numbers 25-32 map to quadrant 4."""
        model = DentalNERModel.__new__(DentalNERModel)
        assert model._get_quadrant_universal(28) == 4

    def test_quadrant_invalid(self):
        """Out-of-range tooth returns 0."""
        model = DentalNERModel.__new__(DentalNERModel)
        assert model._get_quadrant_universal(0) == 0
        assert model._get_quadrant_universal(33) == 0

    def test_quadrant_upper_left(self):
        """Tooth numbers 1-8 map to quadrant 1."""
        model = DentalNERModel.__new__(DentalNERModel)
        assert model._get_quadrant_universal(5) == 1

    def test_quadrant_upper_right(self):
        """Tooth numbers 9-16 map to quadrant 2."""
        model = DentalNERModel.__new__(DentalNERModel)
        assert model._get_quadrant_universal(12) == 2

    def test_surface_validation_valid(self):
        """Valid surface codes are accepted."""
        model = DentalNERModel.__new__(DentalNERModel)
        assert model._is_valid_surface("M") is True
        assert model._is_valid_surface("MB") is True
        assert model._is_valid_surface("DL") is True

    def test_surface_validation_invalid(self):
        """Invalid surface codes are rejected."""
        model = DentalNERModel.__new__(DentalNERModel)
        assert model._is_valid_surface("XYZ") is False
        assert model._is_valid_surface("MOD") is False


# ---------------------------------------------------------------------------
# 4. Drift detector — empty current records and label/confidence branches
# ---------------------------------------------------------------------------

class TestDriftDetectorEdgeCases:
    """Cover drift_detector.py lines 275, 339, 498, 548."""

    def test_text_monitor_no_current_records(self):
        """When no current records, compute_drift returns no-drift (line ~275)."""
        monitor = TextDistributionMonitor(reference_size=5)
        # Track enough reference data to freeze
        for _ in range(5):
            monitor.track("Some clinical text with medical terms and abbreviations for reference.")
        # Don't add current records after reference is frozen
        report = monitor.compute_drift(window_size=10)
        # Should not crash — returns a valid report
        assert report.drift_score >= 0

    def test_prediction_monitor_no_recent_confidence(self):
        """When no recent confidence records, returns empty report (line ~498)."""
        monitor = PredictionMonitor()
        # Track some reference data
        for i in range(20):
            monitor.track_prediction(
                "model-a",
                predictions=["A"],
                confidence=0.8 + (i % 5) * 0.02,
            )
        # detect_confidence_drift — all data is reference
        report = monitor.detect_confidence_drift(window_size=5, model_name="model-a")
        assert report.drift_score >= 0

    def test_prediction_monitor_no_recent_labels(self):
        """When no recent label records, returns empty report (line ~548)."""
        monitor = PredictionMonitor()
        for i in range(20):
            monitor.track_prediction(
                "model-a",
                predictions=[["A", "B"][i % 2]],
                confidence=0.8,
            )
        report = monitor.detect_prediction_drift(window_size=5, model_name="model-a")
        assert report.drift_score >= 0

    def test_text_monitor_extract_stats_vocab_diversity(self):
        """_extract_stats computes vocab_diversity and abbrev_density (line ~339)."""
        monitor = TextDistributionMonitor()
        stats = monitor._extract_stats(
            "Patient with DM HTN CHF presented with SOB and CP to the ED today."
        )
        assert "vocab_diversity" in stats
        assert stats["vocab_diversity"] > 0
        assert "abbrev_density" in stats


# ---------------------------------------------------------------------------
# 5. Metrics collector — set_model_load_time, set_active_models, get_metrics
# ---------------------------------------------------------------------------

class TestMetricsCollectorEdgeCases:
    """Cover metrics_collector.py lines 236, 249, 266."""

    def test_set_model_load_time_fallback(self):
        """set_model_load_time stores in fallback gauge (line ~236)."""
        metrics = ModelMetrics(namespace="test", use_prometheus=False)
        metrics.set_model_load_time("ner-model", 1.5)
        collected = metrics.get_metrics()
        assert isinstance(collected, dict)

    def test_set_active_models_fallback(self):
        """set_active_models stores in fallback gauge (line ~249)."""
        metrics = ModelMetrics(namespace="test", use_prometheus=False)
        metrics.set_active_models(3)
        collected = metrics.get_metrics()
        assert isinstance(collected, dict)

    def test_get_metrics_after_inference(self):
        """get_metrics returns populated dict from fallback store (line ~266)."""
        metrics = ModelMetrics(namespace="test", use_prometheus=False)
        metrics.record_inference("test-model", 50.0)
        metrics.record_batch("test-model", 10)
        result = metrics.get_metrics()
        assert isinstance(result, dict)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# 6. Main app exception handlers — TestClient ASGI integration
# ---------------------------------------------------------------------------

class TestMainExceptionHandlers:
    """Cover main.py lines 113-114, 127-128."""

    def test_cliniq_error_returns_structured_json(self):
        """ClinIQError handler returns error_code and details (lines 113-114)."""
        from fastapi import APIRouter
        from fastapi.testclient import TestClient

        from app.core.exceptions import ClinIQError
        from app.main import app

        test_router = APIRouter(prefix="/test-err")

        @test_router.get("/cliniq")
        async def raise_cliniq_error():
            raise ClinIQError(
                message="Test error",
                error_code="TEST_ERROR",
                details={"field": "value"},
            )

        app.include_router(test_router)
        try:
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/test-err/cliniq")
            data = response.json()
            assert data["error"] == "Test error"
            assert data["error_code"] == "TEST_ERROR"
            assert data["details"] == {"field": "value"}
        finally:
            # Clean up: remove the test router
            app.routes[:] = [r for r in app.routes if not (hasattr(r, 'path') and r.path.startswith('/test-err'))]

    def test_general_exception_returns_500(self):
        """General exception handler returns 500 with INTERNAL_ERROR (lines 127-128)."""
        from fastapi import APIRouter
        from fastapi.testclient import TestClient

        from app.main import app

        test_router = APIRouter(prefix="/test-err2")

        @test_router.get("/general")
        async def raise_general_error():
            raise RuntimeError("Something unexpected")

        app.include_router(test_router)
        try:
            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/test-err2/general")
            assert response.status_code == 500
            data = response.json()
            assert data["error_code"] == "INTERNAL_ERROR"
        finally:
            app.routes[:] = [r for r in app.routes if not (hasattr(r, 'path') and r.path.startswith('/test-err2'))]


# ---------------------------------------------------------------------------
# 7. Summarization — InferenceError propagation
# ---------------------------------------------------------------------------

class TestSummarizationEdgeCases:
    """Cover summarization/model.py lines 356-357."""

    def test_extractive_summarizer_inference_error(self):
        """When internal processing fails, InferenceError is raised (lines 356-357)."""
        model = ExtractiveSummarizer.__new__(ExtractiveSummarizer)
        model.model_name = "extractive"
        model.version = "1.0"
        model._is_loaded = True
        model._vectorizer = None  # Will cause AttributeError
        model._preprocessor = MagicMock()
        model._preprocessor.preprocess.side_effect = RuntimeError("bad input")

        with pytest.raises(InferenceError):
            model.summarize("Some clinical text.", detail_level="brief")


# ---------------------------------------------------------------------------
# 8. Document service — InferenceError re-raise (line 120)
# ---------------------------------------------------------------------------

class TestDocumentServiceEdgeCases:
    """Cover document_service.py line 120."""

    @pytest.mark.asyncio
    async def test_analyze_reraises_inference_error(self):
        """InferenceError during analysis is re-raised (line 120)."""
        from app.services.document_service import AnalysisService

        mock_pipeline = MagicMock()
        mock_pipeline.process.side_effect = InferenceError("pipeline", "model failed")

        service = AnalysisService(pipeline=mock_pipeline)
        with pytest.raises(InferenceError):
            await service.analyze("Some clinical text")

    @pytest.mark.asyncio
    async def test_analyze_wraps_generic_exception_as_inference_error(self):
        """Non-InferenceError exceptions are wrapped (line ~122)."""
        from app.services.document_service import AnalysisService

        mock_pipeline = MagicMock()
        mock_pipeline.process.side_effect = ValueError("unexpected")

        service = AnalysisService(pipeline=mock_pipeline)
        with pytest.raises(InferenceError):
            await service.analyze("Some clinical text")


# ---------------------------------------------------------------------------
# 9. Analysis schema — empty text validation (line 172)
# ---------------------------------------------------------------------------

class TestAnalysisSchemaEdge:
    """Cover analysis.py line 172."""

    def test_analysis_request_empty_text_rejected(self):
        """Empty text in AnalysisRequest raises ValidationError."""
        from pydantic import ValidationError

        from app.api.schemas.analysis import AnalysisRequest

        with pytest.raises(ValidationError):
            AnalysisRequest(text="")

    def test_analysis_request_missing_text_rejected(self):
        """Missing text field raises ValidationError."""
        from pydantic import ValidationError

        from app.api.schemas.analysis import AnalysisRequest

        with pytest.raises(ValidationError):
            AnalysisRequest()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# 10. Rate limit — constructor and fallback mode
# ---------------------------------------------------------------------------

class TestRateLimitEdge:
    """Cover rate_limit.py edge case."""

    def test_rate_limit_middleware_init(self):
        """RateLimitMiddleware can be initialized."""
        from starlette.applications import Starlette

        from app.middleware.rate_limit import RateLimitMiddleware

        app = Starlette()
        middleware = RateLimitMiddleware(app)
        assert middleware is not None


# ---------------------------------------------------------------------------
# 11. SHAP format_explanation — verify highlighted segments
# ---------------------------------------------------------------------------

class TestSHAPFormatExplanation:
    """Cover shap_explainer.py format_explanation path."""

    def test_format_explanation_positive_and_negative(self):
        """format_explanation produces segments with correct directions."""
        from app.ml.explainability.shap_explainer import SHAPExplanation, format_explanation

        explanation = SHAPExplanation(
            feature_attributions={"diabetes": 0.5, "normal": -0.3},
            base_value=0.5,
            predicted_value=0.7,
            top_positive_features=[("diabetes", 0.5)],
            top_negative_features=[("normal", -0.3)],
        )

        result = format_explanation(explanation, "Patient has diabetes with normal vitals")
        assert "highlighted_segments" in result
        segments = result["highlighted_segments"]
        assert len(segments) >= 2
        directions = {s["direction"] for s in segments}
        assert "positive" in directions
        assert "negative" in directions
        assert result["base_value"] == 0.5
        assert result["predicted_value"] == 0.7

    def test_format_explanation_skips_tiny_attributions(self):
        """Attributions with abs < 0.001 are skipped."""
        from app.ml.explainability.shap_explainer import SHAPExplanation, format_explanation

        explanation = SHAPExplanation(
            feature_attributions={"diabetes": 0.0001, "fever": 0.5},
            base_value=0.5,
            predicted_value=0.7,
            top_positive_features=[("fever", 0.5)],
            top_negative_features=[],
        )

        result = format_explanation(explanation, "diabetes and fever")
        segments = result["highlighted_segments"]
        # Only fever should be included (diabetes has tiny attribution)
        texts = [s["text"] for s in segments]
        assert "fever" in texts
        assert "diabetes" not in texts
