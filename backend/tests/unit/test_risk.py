"""Unit tests for the risk scoring model."""

import pytest

from app.ml.ner.model import Entity
from app.ml.risk.model import (
    RISK_CATEGORIES,
    RiskAssessment,
    RiskFactor,
    RuleBasedRiskScorer,
)


class TestRiskFactor:
    """Tests for the RiskFactor dataclass."""

    def test_creation(self):
        """Test creating a RiskFactor."""
        factor = RiskFactor(
            name="polypharmacy",
            score=0.7,
            weight=0.6,
            category="medication_risk",
            description="Multiple medications detected",
        )

        assert factor.name == "polypharmacy"
        assert factor.score == 0.7
        assert factor.weight == 0.6
        assert factor.category == "medication_risk"
        assert factor.description == "Multiple medications detected"

    def test_to_dict_keys(self):
        """Test that to_dict() returns all expected keys."""
        factor = RiskFactor(
            name="test_factor",
            score=0.5,
            weight=0.4,
            category="diagnostic_complexity",
            description="Test factor",
        )
        result = factor.to_dict()

        expected_keys = {"name", "score", "weight", "category", "description"}
        assert set(result.keys()) == expected_keys

    def test_to_dict_values(self):
        """Test that to_dict() returns correct values."""
        factor = RiskFactor(
            name="urgency_stat",
            score=1.0,
            weight=1.0,
            category="diagnostic_complexity",
            description="STAT order present",
        )
        result = factor.to_dict()

        assert result["name"] == "urgency_stat"
        assert result["score"] == 1.0
        assert result["weight"] == 1.0
        assert result["category"] == "diagnostic_complexity"


class TestRiskAssessment:
    """Tests for the RiskAssessment dataclass."""

    @pytest.fixture
    def sample_assessment(self) -> RiskAssessment:
        """Provide a sample RiskAssessment."""
        return RiskAssessment(
            overall_score=45.0,
            risk_level="moderate",
            factors=[
                RiskFactor(
                    name="polypharmacy",
                    score=0.5,
                    weight=0.6,
                    category="medication_risk",
                    description="5+ medications",
                )
            ],
            recommendations=["Follow-up in 1-2 weeks"],
            processing_time_ms=12.5,
            category_scores={
                "medication_risk": 30.0,
                "diagnostic_complexity": 50.0,
                "follow_up_urgency": 10.0,
            },
            model_name="rule-based-risk",
            model_version="1.0.0",
        )

    def test_to_dict_keys(self, sample_assessment: RiskAssessment):
        """Test that to_dict() includes all required keys."""
        result = sample_assessment.to_dict()
        expected_keys = {
            "overall_score",
            "risk_level",
            "factors",
            "recommendations",
            "processing_time_ms",
            "category_scores",
            "model_name",
            "model_version",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_to_dict_factors_are_dicts(self, sample_assessment: RiskAssessment):
        """Test that factors in to_dict() output are serialised as dicts."""
        result = sample_assessment.to_dict()
        for factor in result["factors"]:
            assert isinstance(factor, dict)

    def test_to_dict_overall_score(self, sample_assessment: RiskAssessment):
        """Test that overall_score is preserved in to_dict()."""
        result = sample_assessment.to_dict()
        assert result["overall_score"] == 45.0

    def test_risk_level_values(self):
        """Test that risk_level can be any of the four expected values."""
        for level in ("low", "moderate", "high", "critical"):
            assessment = RiskAssessment(
                overall_score=0.0,
                risk_level=level,
                factors=[],
                recommendations=[],
                processing_time_ms=0.0,
            )
            assert assessment.risk_level == level


class TestRiskCategories:
    """Tests for the RISK_CATEGORIES constant."""

    def test_risk_categories_is_tuple(self):
        """Test that RISK_CATEGORIES is a tuple."""
        assert isinstance(RISK_CATEGORIES, tuple)

    def test_contains_three_categories(self):
        """Test that there are exactly three risk categories."""
        assert len(RISK_CATEGORIES) == 3

    def test_expected_categories_present(self):
        """Test that the three expected category names are present."""
        expected = {"medication_risk", "diagnostic_complexity", "follow_up_urgency"}
        assert expected == set(RISK_CATEGORIES)


class TestRuleBasedRiskScorer:
    """Tests for RuleBasedRiskScorer."""

    @pytest.fixture
    def scorer(self) -> RuleBasedRiskScorer:
        """Create a loaded RuleBasedRiskScorer."""
        s = RuleBasedRiskScorer()
        s.load()
        return s

    def test_load_sets_is_loaded(self):
        """Test that load() sets _is_loaded to True."""
        scorer = RuleBasedRiskScorer()
        assert scorer.is_loaded is False
        scorer.load()
        assert scorer.is_loaded is True

    def test_ensure_loaded_triggers_load(self):
        """Test that ensure_loaded() calls load() when not loaded."""
        scorer = RuleBasedRiskScorer()
        scorer.ensure_loaded()
        assert scorer.is_loaded is True

    def test_assess_risk_returns_risk_assessment(self, scorer: RuleBasedRiskScorer):
        """Test that assess_risk() returns a RiskAssessment."""
        result = scorer.assess_risk("Patient is healthy.")
        assert isinstance(result, RiskAssessment)

    def test_low_risk_for_benign_text(self, scorer: RuleBasedRiskScorer):
        """Test that a benign clinical note scores as low risk."""
        text = "Patient is healthy. Routine follow-up. No concerns."
        result = scorer.assess_risk(text)

        assert result.risk_level == "low"
        assert result.overall_score < 35.0

    def test_high_risk_for_critical_text(self, scorer: RuleBasedRiskScorer):
        """Test that critical keywords elevate the risk score."""
        text = (
            "STAT alert: Patient in critical condition. "
            "Emergent intervention required. Cardiac arrest suspected."
        )
        result = scorer.assess_risk(text)

        assert result.risk_level in ("moderate", "high", "critical")
        assert result.overall_score >= 35.0

    def test_overall_score_range(self, scorer: RuleBasedRiskScorer):
        """Test that overall_score is always in [0, 100]."""
        texts = [
            "Healthy patient, no concerns.",
            "Emergent critical condition, stat intervention required.",
            "Patient with diabetes, hypertension, on warfarin, insulin.",
        ]
        for text in texts:
            result = scorer.assess_risk(text)
            assert 0.0 <= result.overall_score <= 100.0, (
                f"overall_score {result.overall_score} out of [0,100] for: {text!r}"
            )

    def test_risk_level_is_valid_string(self, scorer: RuleBasedRiskScorer):
        """Test that risk_level is one of the four valid values."""
        valid_levels = {"low", "moderate", "high", "critical"}
        text = "Patient has diabetes mellitus and hypertension."
        result = scorer.assess_risk(text)
        assert result.risk_level in valid_levels

    def test_category_scores_contain_all_categories(self, scorer: RuleBasedRiskScorer):
        """Test that category_scores has an entry for each category."""
        result = scorer.assess_risk("Test text.")
        for category in RISK_CATEGORIES:
            assert category in result.category_scores

    def test_category_scores_in_range(self, scorer: RuleBasedRiskScorer):
        """Test that category scores are in [0, 100]."""
        result = scorer.assess_risk("Patient on warfarin with urgent follow-up needed.")
        for cat, score in result.category_scores.items():
            assert 0.0 <= score <= 100.0, (
                f"Category score {score} for '{cat}' is out of [0, 100]"
            )

    def test_factors_is_list(self, scorer: RuleBasedRiskScorer):
        """Test that factors is a list."""
        result = scorer.assess_risk("Test text.")
        assert isinstance(result.factors, list)

    def test_recommendations_is_list(self, scorer: RuleBasedRiskScorer):
        """Test that recommendations is a list of strings."""
        result = scorer.assess_risk("Test text.")
        assert isinstance(result.recommendations, list)
        for rec in result.recommendations:
            assert isinstance(rec, str)

    def test_processing_time_is_positive(self, scorer: RuleBasedRiskScorer):
        """Test that processing_time_ms is a positive number."""
        result = scorer.assess_risk("Test text.")
        assert result.processing_time_ms > 0.0

    def test_high_risk_medication_detected(self, scorer: RuleBasedRiskScorer):
        """Test that warfarin is flagged as a high-risk medication."""
        text = "Patient is currently on warfarin therapy for atrial fibrillation."
        result = scorer.assess_risk(text)

        factor_names = [f.name for f in result.factors]
        assert any("warfarin" in name for name in factor_names), (
            f"Expected warfarin factor, got: {factor_names}"
        )

    def test_polypharmacy_detected_from_entities(self, scorer: RuleBasedRiskScorer):
        """Test that polypharmacy is detected when 5+ medication entities provided."""
        entities = [
            Entity("warfarin", "MEDICATION", 0, 8, 0.9),
            Entity("metformin", "MEDICATION", 10, 19, 0.9),
            Entity("lisinopril", "MEDICATION", 21, 31, 0.9),
            Entity("atorvastatin", "MEDICATION", 33, 45, 0.9),
            Entity("aspirin", "MEDICATION", 47, 54, 0.9),
            Entity("amlodipine", "MEDICATION", 56, 66, 0.9),
        ]
        result = scorer.assess_risk(
            "Patient on multiple medications.", entities=entities
        )

        factor_names = [f.name for f in result.factors]
        assert "polypharmacy" in factor_names

    def test_no_polypharmacy_for_few_medications(self, scorer: RuleBasedRiskScorer):
        """Test that fewer than 5 medications do not trigger polypharmacy."""
        entities = [
            Entity("metformin", "MEDICATION", 0, 9, 0.9),
            Entity("lisinopril", "MEDICATION", 11, 21, 0.9),
        ]
        result = scorer.assess_risk(
            "Patient on metformin and lisinopril.", entities=entities
        )

        factor_names = [f.name for f in result.factors]
        assert "polypharmacy" not in factor_names

    def test_icd_codes_influence_score(self, scorer: RuleBasedRiskScorer):
        """Test that providing ICD codes influences the assessment."""
        text = "Assessment: Type 2 diabetes mellitus, hypertension."

        without_icd = scorer.assess_risk(text)
        with_icd = scorer.assess_risk(text, icd_codes=["E11.9", "I10", "C50.9"])

        # With oncology code (C) the diagnostic complexity should be higher
        assert with_icd.category_scores["diagnostic_complexity"] >= without_icd.category_scores["diagnostic_complexity"]

    def test_follow_up_risk_non_compliant(self, scorer: RuleBasedRiskScorer):
        """Test that non-compliance text raises follow_up_urgency score."""
        text = "Patient is non-compliant with medications and missed last appointment."
        result = scorer.assess_risk(text)

        assert result.category_scores["follow_up_urgency"] > 0.0

    def test_model_name_in_assessment(self, scorer: RuleBasedRiskScorer):
        """Test that model_name is present in the RiskAssessment."""
        result = scorer.assess_risk("Test.")
        assert result.model_name == "rule-based-risk"

    def test_model_version_in_assessment(self, scorer: RuleBasedRiskScorer):
        """Test that model_version is present in the RiskAssessment."""
        result = scorer.assess_risk("Test.")
        assert result.model_version == "1.0.0"

    def test_factors_count_capped_at_15(self, scorer: RuleBasedRiskScorer):
        """Test that at most 15 factors are returned."""
        # Use a complex text that triggers many factors
        text = (
            "Patient on warfarin, heparin, insulin, fentanyl, morphine, oxycodone, "
            "digoxin, lithium, methotrexate. STAT critical emergent urgent. "
            "Missed appointment, non-compliant, lost to follow-up."
        )
        result = scorer.assess_risk(text)
        assert len(result.factors) <= 15

    def test_custom_category_weights(self):
        """Test that custom category weights are applied."""
        weights = {
            "medication_risk": 0.5,
            "diagnostic_complexity": 0.3,
            "follow_up_urgency": 0.2,
        }
        scorer = RuleBasedRiskScorer(category_weights=weights)
        scorer.load()

        assert scorer.category_weights["medication_risk"] == 0.5
        assert scorer.category_weights["diagnostic_complexity"] == 0.3

    @pytest.mark.parametrize(
        "score,expected_level",
        [
            (0.0, "low"),
            (20.0, "low"),
            (34.9, "low"),
            (35.0, "moderate"),
            (59.9, "moderate"),
            (60.0, "high"),
            (79.9, "high"),
            (80.0, "critical"),
            (100.0, "critical"),
        ],
    )
    def test_risk_level_thresholds(self, score: float, expected_level: str):
        """Test the static _risk_level_from_score() thresholds."""
        level = RuleBasedRiskScorer._risk_level_from_score(score)
        assert level == expected_level, (
            f"Expected {expected_level!r} for score {score}, got {level!r}"
        )
