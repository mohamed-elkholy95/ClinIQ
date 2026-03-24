"""Unit tests for the RiskScorer clinical risk assessment module.

Tests the rule-based risk scoring system that evaluates clinical documents
across five risk categories: medication, cardiovascular, infection, surgical,
and follow-up.  Validates factor extraction from text, entities, and ICD
predictions, overall score calculation, risk level determination, and
recommendation generation.
"""

from __future__ import annotations

import pytest

from app.ml.ner.model import Entity
from app.ml.risk.scorer import (
    HIGH_RISK_CONDITIONS,
    HIGH_RISK_MEDICATIONS,
    RISK_CATEGORIES,
    URGENCY_KEYWORDS,
    RiskFactor,
    RiskScore,
    RiskScorer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def scorer() -> RiskScorer:
    """Default RiskScorer instance."""
    return RiskScorer()


@pytest.fixture
def high_risk_text() -> str:
    """Clinical note with multiple high-risk indicators."""
    return (
        "Patient admitted with acute myocardial infarction. History of "
        "coronary artery disease and diabetes mellitus. Currently on warfarin "
        "and insulin. Stat cardiology consult requested. Evidence of "
        "decompensated heart failure with sepsis."
    )


@pytest.fixture
def low_risk_text() -> str:
    """Clinical note with minimal risk indicators."""
    return (
        "Patient presents for routine annual physical examination. "
        "No acute complaints. Blood pressure 120/80. All vitals stable."
    )


@pytest.fixture
def medication_entities() -> list[Entity]:
    """Entities representing a polypharmacy scenario with high-risk meds."""
    meds = ["warfarin", "insulin", "methotrexate", "lisinopril", "metformin", "aspirin"]
    return [
        Entity(
            text=name,
            entity_type="MEDICATION",
            start_char=i * 20,
            end_char=i * 20 + len(name),
            confidence=0.9,
        )
        for i, name in enumerate(meds)
    ]


@pytest.fixture
def disease_entities() -> list[Entity]:
    """Entities representing high-risk diseases (non-negated)."""
    diseases = [("heart failure", False), ("pneumonia", False), ("cancer", False)]
    entities = []
    for i, (name, negated) in enumerate(diseases):
        entities.append(
            Entity(
                text=name,
                entity_type="DISEASE",
                start_char=i * 30,
                end_char=i * 30 + len(name),
                confidence=0.85,
                is_negated=negated,
            )
        )
    return entities


@pytest.fixture
def icd_predictions() -> list[dict]:
    """Sample ICD-10 predictions spanning multiple chapters."""
    return [
        {"code": "I21.0", "description": "Acute ST elevation MI", "confidence": 0.9},
        {"code": "E11.9", "description": "Type 2 diabetes mellitus", "confidence": 0.8},
        {"code": "C34.1", "description": "Malignant neoplasm of lung", "confidence": 0.7},
        {"code": "J18.9", "description": "Pneumonia, unspecified", "confidence": 0.6},
        {"code": "K21.0", "description": "GERD with esophagitis", "confidence": 0.5},
    ]


# ---------------------------------------------------------------------------
# RiskFactor dataclass
# ---------------------------------------------------------------------------


class TestRiskFactor:
    """Tests for the RiskFactor dataclass and serialisation."""

    def test_to_dict_includes_all_fields(self):
        factor = RiskFactor(
            name="test_factor",
            description="A test risk factor",
            weight=0.7,
            value=1.0,
            source="text",
            evidence="keyword found",
        )
        d = factor.to_dict()
        assert d["name"] == "test_factor"
        assert d["weight"] == 0.7
        assert d["value"] == 1.0
        assert d["source"] == "text"
        assert d["evidence"] == "keyword found"

    def test_to_dict_with_none_evidence(self):
        factor = RiskFactor(
            name="x", description="x", weight=0.5, value=0.5, source="derived"
        )
        assert factor.evidence is None
        assert factor.to_dict()["evidence"] is None


# ---------------------------------------------------------------------------
# RiskScore dataclass
# ---------------------------------------------------------------------------


class TestRiskScore:
    """Tests for the RiskScore dataclass and serialisation."""

    def test_to_dict_structure(self):
        factor = RiskFactor(
            name="f1", description="d", weight=0.5, value=1.0, source="text"
        )
        score = RiskScore(
            overall_score=0.65,
            risk_level="high",
            category_scores={"medication": 0.4, "cardiovascular": 0.8},
            risk_factors=[factor],
            protective_factors=[],
            recommendations=["Follow up"],
            processing_time_ms=12.3,
            model_name="rule-based-risk",
            model_version="1.0.0",
        )
        d = score.to_dict()
        assert d["overall_score"] == 0.65
        assert d["risk_level"] == "high"
        assert len(d["risk_factors"]) == 1
        assert d["risk_factors"][0]["name"] == "f1"
        assert d["recommendations"] == ["Follow up"]


# ---------------------------------------------------------------------------
# Constants / lookup tables
# ---------------------------------------------------------------------------


class TestConstants:
    """Smoke tests for module-level lookup tables."""

    def test_risk_categories_has_expected_keys(self):
        expected = {"medication", "cardiovascular", "infection", "surgical", "follow_up"}
        assert set(RISK_CATEGORIES.keys()) == expected

    def test_high_risk_conditions_non_empty(self):
        assert len(HIGH_RISK_CONDITIONS) > 10
        for condition, weight in HIGH_RISK_CONDITIONS.items():
            assert 0.0 < weight <= 1.0, f"{condition} weight out of range"

    def test_high_risk_medications_non_empty(self):
        assert len(HIGH_RISK_MEDICATIONS) > 5
        for med, weight in HIGH_RISK_MEDICATIONS.items():
            assert 0.0 < weight <= 1.0, f"{med} weight out of range"

    def test_urgency_keywords_levels(self):
        assert set(URGENCY_KEYWORDS.keys()) == {"critical", "high", "moderate"}
        for level, keywords in URGENCY_KEYWORDS.items():
            assert len(keywords) > 0


# ---------------------------------------------------------------------------
# Text-based risk factor extraction
# ---------------------------------------------------------------------------


class TestTextRiskFactors:
    """Tests for _extract_text_risk_factors."""

    def test_urgency_keywords_detected(self, scorer: RiskScorer):
        text = "Stat cardiology consult for emergent chest pain"
        factors = scorer._extract_text_risk_factors(text)
        names = {f.name for f in factors}
        assert "urgency_stat" in names
        assert "urgency_emergent" in names

    def test_high_risk_conditions_detected(self, scorer: RiskScorer):
        text = "Patient with acute myocardial infarction and sepsis"
        factors = scorer._extract_text_risk_factors(text)
        names = {f.name for f in factors}
        assert "acute_myocardial_infarction" in names
        assert "sepsis" in names

    def test_no_factors_for_benign_text(self, scorer: RiskScorer):
        text = "Patient feeling well. No complaints today."
        factors = scorer._extract_text_risk_factors(text)
        # May pick up zero or very few factors
        high_weight = [f for f in factors if f.weight > 0.5]
        assert len(high_weight) == 0

    def test_multiple_issues_factor(self, scorer: RiskScorer):
        """Multiple issue-indicator keywords trigger a derived factor."""
        text = (
            "diagnosis of pneumonia, symptom of fever, abnormal WBC, "
            "positive for influenza, evidence of consolidation, "
            "diagnosis unclear, symptom worsening"
        )
        factors = scorer._extract_text_risk_factors(text)
        issue_factors = [f for f in factors if f.name == "multiple_issues"]
        # 'diagnosis' appears twice, etc. — need ≥5 unique indicators hit
        # The code counts each indicator once, so we need ≥5 distinct matches
        # "diagnosis", "symptom", "abnormal", "positive for", "evidence of" = 5
        assert len(issue_factors) == 1
        assert issue_factors[0].source == "derived"


# ---------------------------------------------------------------------------
# Entity-based risk factor extraction
# ---------------------------------------------------------------------------


class TestEntityRiskFactors:
    """Tests for _extract_entity_risk_factors."""

    def test_polypharmacy_detected(
        self, scorer: RiskScorer, medication_entities: list[Entity]
    ):
        """Six medications should trigger the polypharmacy factor."""
        factors = scorer._extract_entity_risk_factors(medication_entities)
        poly = [f for f in factors if f.name == "polypharmacy"]
        assert len(poly) == 1
        assert "6 medications" in poly[0].evidence

    def test_high_risk_medications_detected(
        self, scorer: RiskScorer, medication_entities: list[Entity]
    ):
        factors = scorer._extract_entity_risk_factors(medication_entities)
        high_risk_names = {f.name for f in factors if f.name.startswith("high_risk_med_")}
        assert "high_risk_med_warfarin" in high_risk_names
        assert "high_risk_med_insulin" in high_risk_names
        assert "high_risk_med_methotrexate" in high_risk_names

    def test_disease_entities_matched(
        self, scorer: RiskScorer, disease_entities: list[Entity]
    ):
        factors = scorer._extract_entity_risk_factors(disease_entities)
        names = {f.name for f in factors}
        # "heart failure" and "cancer" are in HIGH_RISK_CONDITIONS
        assert any("heart" in n for n in names)
        assert any("cancer" in n for n in names)

    def test_negated_diseases_excluded(self, scorer: RiskScorer):
        """Negated diseases should NOT produce risk factors."""
        entities = [
            Entity(
                text="heart failure",
                entity_type="DISEASE",
                start_char=0,
                end_char=13,
                confidence=0.9,
                is_negated=True,
            )
        ]
        factors = scorer._extract_entity_risk_factors(entities)
        disease_factors = [f for f in factors if f.source == "entity" and "heart" in f.name]
        assert len(disease_factors) == 0

    def test_no_polypharmacy_with_few_meds(self, scorer: RiskScorer):
        """Fewer than 5 medications should not trigger polypharmacy."""
        entities = [
            Entity(
                text="aspirin",
                entity_type="MEDICATION",
                start_char=0,
                end_char=7,
                confidence=0.9,
            )
        ]
        factors = scorer._extract_entity_risk_factors(entities)
        poly = [f for f in factors if f.name == "polypharmacy"]
        assert len(poly) == 0


# ---------------------------------------------------------------------------
# ICD-based risk factor extraction
# ---------------------------------------------------------------------------


class TestICDRiskFactors:
    """Tests for _extract_icd_risk_factors."""

    def test_icd_factors_created(
        self, scorer: RiskScorer, icd_predictions: list[dict]
    ):
        factors = scorer._extract_icd_risk_factors(icd_predictions)
        assert len(factors) == 5
        codes = {f.evidence for f in factors}
        assert "I21.0" in codes
        assert "E11.9" in codes

    def test_icd_chapter_weighting(self, scorer: RiskScorer):
        """Circulatory codes should get higher weight than digestive."""
        predictions = [
            {"code": "I50.9", "description": "Heart failure", "confidence": 1.0},
            {"code": "K21.0", "description": "GERD", "confidence": 1.0},
        ]
        factors = scorer._extract_icd_risk_factors(predictions)
        circulatory = [f for f in factors if f.evidence == "I50.9"][0]
        digestive = [f for f in factors if f.evidence == "K21.0"][0]
        assert circulatory.weight > digestive.weight

    def test_neoplasm_codes_high_weight(self, scorer: RiskScorer):
        predictions = [
            {"code": "C34.1", "description": "Lung cancer", "confidence": 1.0},
        ]
        factors = scorer._extract_icd_risk_factors(predictions)
        assert factors[0].weight == pytest.approx(0.7, abs=0.01)

    def test_empty_predictions(self, scorer: RiskScorer):
        assert scorer._extract_icd_risk_factors([]) == []


# ---------------------------------------------------------------------------
# Category score calculation
# ---------------------------------------------------------------------------


class TestCategoryScores:
    """Tests for _calculate_category_scores."""

    def test_cardiovascular_keywords_boost_score(self, scorer: RiskScorer):
        text = "hypertension, cardiac arrhythmia, coronary artery disease, heart murmur"
        factors: list[RiskFactor] = []
        scores = scorer._calculate_category_scores(factors, text)
        assert scores["cardiovascular"] > 0.5

    def test_infection_keywords(self, scorer: RiskScorer):
        text = "Patient with sepsis and bacterial infection, immunocompromised"
        scores = scorer._calculate_category_scores([], text)
        assert scores["infection"] > 0.5

    def test_surgical_keywords(self, scorer: RiskScorer):
        text = "Post-op day 2 after surgery. Incision site clean. Procedure uneventful."
        scores = scorer._calculate_category_scores([], text)
        assert scores["surgical"] > 0.3

    def test_medication_score_from_factors(self, scorer: RiskScorer):
        """Medication score should derive from med-related risk factors."""
        factors = [
            RiskFactor(
                name="high_risk_med_warfarin",
                description="d",
                weight=0.7,
                value=1.0,
                source="entity",
            ),
            RiskFactor(
                name="polypharmacy",
                description="d",
                weight=0.5,
                value=0.8,
                source="entity",
            ),
        ]
        scores = scorer._calculate_category_scores(factors, "some text")
        assert scores["medication"] > 0

    def test_benign_text_low_scores(self, scorer: RiskScorer, low_risk_text: str):
        scores = scorer._calculate_category_scores([], low_risk_text)
        assert all(v < 0.3 for v in scores.values())


# ---------------------------------------------------------------------------
# Overall score & risk level
# ---------------------------------------------------------------------------


class TestOverallScoring:
    """Tests for _calculate_overall_score and _determine_risk_level."""

    def test_score_bounded_zero_one(self, scorer: RiskScorer):
        """Overall score must be in [0, 1]."""
        # Extreme high
        category_scores = {k: 1.0 for k in RISK_CATEGORIES}
        factors = [
            RiskFactor(name="x", description="d", weight=0.9, value=1.0, source="text")
            for _ in range(20)
        ]
        score = scorer._calculate_overall_score(category_scores, factors)
        assert 0.0 <= score <= 1.0

        # All zeros
        zero_scores = {k: 0.0 for k in RISK_CATEGORIES}
        score_low = scorer._calculate_overall_score(zero_scores, [])
        assert score_low == 0.0

    def test_risk_level_critical(self, scorer: RiskScorer):
        assert scorer._determine_risk_level(0.85) == "critical"

    def test_risk_level_high(self, scorer: RiskScorer):
        assert scorer._determine_risk_level(0.65) == "high"

    def test_risk_level_moderate(self, scorer: RiskScorer):
        assert scorer._determine_risk_level(0.45) == "moderate"

    def test_risk_level_low(self, scorer: RiskScorer):
        assert scorer._determine_risk_level(0.2) == "low"

    def test_risk_level_boundaries(self, scorer: RiskScorer):
        assert scorer._determine_risk_level(0.8) == "critical"
        assert scorer._determine_risk_level(0.6) == "high"
        assert scorer._determine_risk_level(0.4) == "moderate"
        assert scorer._determine_risk_level(0.39) == "low"


# ---------------------------------------------------------------------------
# Recommendation generation
# ---------------------------------------------------------------------------


class TestRecommendations:
    """Tests for _generate_recommendations."""

    def test_critical_recommendations(self, scorer: RiskScorer):
        recs = scorer._generate_recommendations(0.85, {}, [])
        assert any("immediate" in r.lower() for r in recs)
        assert any("specialist" in r.lower() or "escalation" in r.lower() for r in recs)

    def test_high_risk_recommendations(self, scorer: RiskScorer):
        recs = scorer._generate_recommendations(0.65, {}, [])
        assert any("48-72" in r or "urgent" in r.lower() for r in recs)

    def test_medication_category_recommendation(self, scorer: RiskScorer):
        recs = scorer._generate_recommendations(
            0.5, {"medication": 0.7, "cardiovascular": 0.3}, []
        )
        assert any("medication review" in r.lower() for r in recs)

    def test_cardiovascular_recommendation(self, scorer: RiskScorer):
        recs = scorer._generate_recommendations(
            0.5, {"cardiovascular": 0.7}, []
        )
        assert any("cardiology" in r.lower() for r in recs)

    def test_max_five_recommendations(self, scorer: RiskScorer):
        """Should never return more than 5 recommendations."""
        all_high = {k: 0.8 for k in RISK_CATEGORIES}
        many_factors = [
            RiskFactor(name=f"f{i}", description="d", weight=0.9, value=1.0, source="text")
            for i in range(10)
        ]
        recs = scorer._generate_recommendations(0.9, all_high, many_factors)
        assert len(recs) <= 5


# ---------------------------------------------------------------------------
# End-to-end calculate_risk
# ---------------------------------------------------------------------------


class TestCalculateRisk:
    """Integration-style tests for the full calculate_risk method."""

    def test_high_risk_document(
        self,
        scorer: RiskScorer,
        high_risk_text: str,
        medication_entities: list[Entity],
        icd_predictions: list[dict],
    ):
        result = scorer.calculate_risk(
            high_risk_text,
            entities=medication_entities,
            icd_predictions=icd_predictions,
        )
        assert isinstance(result, RiskScore)
        assert result.overall_score > 0.3
        assert result.risk_level in {"moderate", "high", "critical"}
        assert len(result.risk_factors) > 0
        assert result.processing_time_ms > 0
        assert len(result.recommendations) > 0

    def test_low_risk_document(self, scorer: RiskScorer, low_risk_text: str):
        result = scorer.calculate_risk(low_risk_text)
        assert result.overall_score < 0.4
        assert result.risk_level == "low"

    def test_text_only_no_entities(self, scorer: RiskScorer, high_risk_text: str):
        """Works without entity or ICD inputs."""
        result = scorer.calculate_risk(high_risk_text)
        assert isinstance(result, RiskScore)
        assert result.overall_score > 0

    def test_risk_factors_capped_at_ten(self, scorer: RiskScorer):
        """Result should contain at most 10 risk factors."""
        text = " ".join(HIGH_RISK_CONDITIONS.keys()) + " " + " ".join(
            kw for kwlist in URGENCY_KEYWORDS.values() for kw in kwlist
        )
        result = scorer.calculate_risk(text)
        assert len(result.risk_factors) <= 10

    def test_result_serialisable(self, scorer: RiskScorer, high_risk_text: str):
        """to_dict() should produce a JSON-safe dictionary."""
        result = scorer.calculate_risk(high_risk_text)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert isinstance(d["risk_factors"], list)
        assert isinstance(d["overall_score"], float)

    def test_custom_category_weights(self):
        """Custom weights change the scoring behaviour."""
        # All weight on medication category
        scorer = RiskScorer(
            category_weights={
                "medication": 1.0,
                "cardiovascular": 0.0,
                "infection": 0.0,
                "surgical": 0.0,
                "follow_up": 0.0,
            }
        )
        # Text with only cardiovascular keywords — medication category = 0
        text = "Patient with hypertension and cardiac history"
        result = scorer.calculate_risk(text)
        # Category contribution is 0 for cardio since weight is 0
        # (but factor_boost may still contribute)
        assert result.overall_score < 0.5

    def test_empty_text(self, scorer: RiskScorer):
        """Empty string should produce a valid low-risk result."""
        result = scorer.calculate_risk("")
        assert result.overall_score == 0.0
        assert result.risk_level == "low"
