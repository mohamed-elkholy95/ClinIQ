"""Unit tests for MLRiskScorer.

Tests the ML-based risk scorer placeholder including loading (with and without
a model file), null assessment fallback, feature extraction, and error handling.
Also covers the RuleBasedRiskScorer recommendation generation paths.
"""

from __future__ import annotations

from unittest.mock import MagicMock, mock_open, patch

import pytest

from app.core.exceptions import InferenceError, ModelLoadError
from app.ml.ner.model import Entity
from app.ml.risk.model import MLRiskScorer, RiskAssessment, RiskFactor, RuleBasedRiskScorer

# ---------------------------------------------------------------------------
# MLRiskScorer
# ---------------------------------------------------------------------------

class TestMLRiskScorerLoad:
    """Tests for MLRiskScorer loading."""

    def test_load_without_path(self) -> None:
        scorer = MLRiskScorer()
        scorer.load()
        assert scorer.is_loaded
        assert scorer._classifier is None

    def test_load_from_file(self) -> None:
        mock_clf = MagicMock()
        with patch("builtins.open", mock_open()), \
             patch("pickle.load", return_value=mock_clf):
            scorer = MLRiskScorer(model_path="/model.pkl")
            scorer.load()
        assert scorer.is_loaded
        assert scorer._classifier is mock_clf

    def test_load_failure_raises(self) -> None:
        with patch("builtins.open", side_effect=FileNotFoundError):
            scorer = MLRiskScorer(model_path="/bad.pkl")
            with pytest.raises(ModelLoadError):
                scorer.load()


class TestMLRiskScorerAssessment:
    """Tests for MLRiskScorer.assess_risk."""

    def test_null_assessment_without_classifier(self) -> None:
        scorer = MLRiskScorer()
        scorer._is_loaded = True
        result = scorer.assess_risk("Patient with hypertension")
        assert isinstance(result, RiskAssessment)
        assert result.overall_score == 0.0
        assert result.risk_level == "low"
        assert "not yet trained" in result.recommendations[0]

    def test_prediction_with_classifier(self) -> None:
        mock_clf = MagicMock()
        import numpy as np
        mock_clf.predict_proba.return_value = np.array([[0.3, 0.7]])

        scorer = MLRiskScorer()
        scorer._is_loaded = True
        scorer._classifier = mock_clf

        result = scorer.assess_risk("Critical unstable patient")
        assert result.overall_score == pytest.approx(70.0, abs=0.1)
        assert result.risk_level in ("high", "moderate")

    def test_feature_extraction(self) -> None:
        scorer = MLRiskScorer()
        scorer._is_loaded = True

        entities = [
            Entity(text="diabetes", entity_type="DISEASE", start_char=0,
                   end_char=8, confidence=0.9, is_negated=True),
            Entity(text="metformin", entity_type="MEDICATION", start_char=10,
                   end_char=19, confidence=0.9, is_uncertain=True),
        ]

        features = scorer._extract_features(
            "Patient has diabetes on metformin",
            entities=entities,
            icd_codes=["E11.9", "I10"],
        )

        assert isinstance(features, list)
        assert len(features) > 30  # text_len + entity_types + neg/unc + ICD + keywords
        # Text length feature
        assert 0 < features[0] <= 1.0
        # DISEASE count = 1/10 = 0.1
        assert features[1] == pytest.approx(0.1)
        # MEDICATION count = 1/10 = 0.1
        assert features[2] == pytest.approx(0.1)
        # Negation ratio = 1/2 = 0.5
        assert features[6] == pytest.approx(0.5)
        # Uncertainty ratio = 1/2 = 0.5
        assert features[7] == pytest.approx(0.5)

    def test_feature_extraction_no_entities(self) -> None:
        scorer = MLRiskScorer()
        scorer._is_loaded = True

        features = scorer._extract_features("text", entities=None, icd_codes=None)
        assert isinstance(features, list)
        # Entity type counts should all be 0
        assert features[1:6] == [0.0] * 5
        # Negation/uncertainty = 0
        assert features[6] == 0.0
        assert features[7] == 0.0

    def test_feature_extraction_icd_chapters(self) -> None:
        scorer = MLRiskScorer()
        scorer._is_loaded = True

        features = scorer._extract_features(
            "text", entities=None, icd_codes=["E11.9", "I10"]
        )
        # ICD chapters start at index 8, for A-Z (26 chars)
        # E is index 4 (A=0), I is index 8
        assert features[8 + 4] == 1.0   # E
        assert features[8 + 8] == 1.0   # I
        assert features[8 + 0] == 0.0   # A (not present)


class TestMLRiskScorerErrors:
    """Test error handling."""

    def test_inference_error_on_exception(self) -> None:
        mock_clf = MagicMock()
        mock_clf.predict_proba.side_effect = RuntimeError("model broken")

        scorer = MLRiskScorer()
        scorer._is_loaded = True
        scorer._classifier = mock_clf

        with pytest.raises(InferenceError):
            scorer.assess_risk("text")


# ---------------------------------------------------------------------------
# RuleBasedRiskScorer — recommendation paths
# ---------------------------------------------------------------------------

class TestRuleBasedRiskScorerRecommendations:
    """Tests for recommendation generation at various risk levels."""

    def _make_scorer(self) -> RuleBasedRiskScorer:
        scorer = RuleBasedRiskScorer()
        scorer._is_loaded = True
        return scorer

    def test_critical_risk_recommendations(self) -> None:
        scorer = self._make_scorer()
        recs = scorer._generate_recommendations(
            overall_score=85,
            category_scores={"medication_risk": 70, "diagnostic_complexity": 65, "follow_up_urgency": 55},
            factors=[
                RiskFactor(name="critical_indicator", score=0.9, weight=1.0,
                           category="diagnostic_complexity", description="Critical finding"),
            ],
        )
        assert any("URGENT" in r for r in recs)
        assert any("Medication reconciliation" in r for r in recs)
        assert any("Multi-disciplinary" in r for r in recs)
        assert any("Proactive care" in r for r in recs)
        assert any("High-severity" in r for r in recs)

    def test_high_risk_recommendations(self) -> None:
        scorer = self._make_scorer()
        recs = scorer._generate_recommendations(
            overall_score=65,
            category_scores={"medication_risk": 40, "diagnostic_complexity": 30, "follow_up_urgency": 20},
            factors=[],
        )
        assert any("HIGH PRIORITY" in r for r in recs)
        assert any("polypharmacy" in r for r in recs)

    def test_moderate_risk_recommendations(self) -> None:
        scorer = self._make_scorer()
        recs = scorer._generate_recommendations(
            overall_score=40,
            category_scores={"medication_risk": 20, "diagnostic_complexity": 20, "follow_up_urgency": 20},
            factors=[],
        )
        assert any("Routine" in r for r in recs)

    def test_low_risk_recommendations(self) -> None:
        scorer = self._make_scorer()
        recs = scorer._generate_recommendations(
            overall_score=10,
            category_scores={"medication_risk": 5, "diagnostic_complexity": 5, "follow_up_urgency": 5},
            factors=[],
        )
        assert any("Standard monitoring" in r for r in recs)

    def test_max_recommendations_capped(self) -> None:
        scorer = self._make_scorer()
        recs = scorer._generate_recommendations(
            overall_score=90,
            category_scores={"medication_risk": 80, "diagnostic_complexity": 80, "follow_up_urgency": 80},
            factors=[
                RiskFactor(name=f"f{i}", score=0.9, weight=1.0,
                           category="diagnostic_complexity", description="") for i in range(5)
            ],
        )
        assert len(recs) <= 6
