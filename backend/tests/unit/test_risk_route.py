"""Unit tests for the clinical risk scoring route handler.

Tests cover:
- ``_score_to_category``: threshold-based category mapping
- ``_run_risk_scoring``: score normalisation, category mapping, factor
  conversion, domain filtering, protective factors, recommendations
- ``calculate_risk_score``: endpoint handler — happy path, InferenceError,
  unexpected error, timing
"""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.schemas.risk import RiskScoreRequest, RiskScoreResponse
from app.api.v1.routes.risk import _run_risk_scoring, _score_to_category, calculate_risk_score

# ---------------------------------------------------------------------------
# Fake stand-ins
# ---------------------------------------------------------------------------


@dataclass
class FakeRiskFactor:
    name: str = "polypharmacy"
    description: str = "Multiple medications detected"
    weight: float = 0.5
    score: float = 0.6
    category: str = "medication_risk"


@dataclass
class FakeRiskAssessment:
    overall_score: float = 45.0
    factors: list[FakeRiskFactor] = field(default_factory=lambda: [FakeRiskFactor()])
    category_scores: dict[str, float] = field(
        default_factory=lambda: {
            "medication_risk": 50.0,
            "cardiovascular": 30.0,
        }
    )
    recommendations: list[str] = field(
        default_factory=lambda: ["Medication review recommended"]
    )


TEXT = "Patient on warfarin, digoxin, and insulin. History of heart failure."


def _mock_scorer(assessment: FakeRiskAssessment | None = None) -> MagicMock:
    model = MagicMock()
    model.assess_risk.return_value = assessment or FakeRiskAssessment()
    model.model_name = "rule-based-risk"
    model.version = "1.0.0"
    return model


# ---------------------------------------------------------------------------
# _score_to_category
# ---------------------------------------------------------------------------


class TestScoreToCategory:
    """Tests for the _score_to_category helper."""

    def test_low(self) -> None:
        assert _score_to_category(0.0) == "low"
        assert _score_to_category(39.9) == "low"

    def test_moderate(self) -> None:
        assert _score_to_category(40.0) == "moderate"
        assert _score_to_category(59.9) == "moderate"

    def test_high(self) -> None:
        assert _score_to_category(60.0) == "high"
        assert _score_to_category(79.9) == "high"

    def test_critical(self) -> None:
        assert _score_to_category(80.0) == "critical"
        assert _score_to_category(100.0) == "critical"


# ---------------------------------------------------------------------------
# _run_risk_scoring
# ---------------------------------------------------------------------------


class TestRunRiskScoring:
    """Tests for the _run_risk_scoring helper."""

    @patch("app.api.v1.routes.risk.get_risk_scorer")
    def test_score_normalised(self, mock_get: MagicMock) -> None:
        """Overall score is normalised from 0-100 to 0-1."""
        mock_get.return_value = _mock_scorer(
            FakeRiskAssessment(overall_score=72.0)
        )
        req = RiskScoreRequest(text=TEXT)
        resp = _run_risk_scoring(req)
        assert resp.score == round(72.0 / 100.0, 4)

    @patch("app.api.v1.routes.risk.get_risk_scorer")
    def test_category_scores_normalised(self, mock_get: MagicMock) -> None:
        """Per-category scores are normalised from 0-100 to 0-1."""
        mock_get.return_value = _mock_scorer(
            FakeRiskAssessment(
                overall_score=50.0,
                category_scores={"medication_risk": 80.0, "cardiovascular": 40.0},
            )
        )
        req = RiskScoreRequest(text=TEXT)
        resp = _run_risk_scoring(req)
        assert resp.category_scores["medication_risk"] == round(80.0 / 100.0, 4)
        assert resp.category_scores["cardiovascular"] == round(40.0 / 100.0, 4)

    @patch("app.api.v1.routes.risk.get_risk_scorer")
    def test_risk_domains_filter(self, mock_get: MagicMock) -> None:
        """When risk_domains is set, only those domains appear in category_scores."""
        mock_get.return_value = _mock_scorer(
            FakeRiskAssessment(
                overall_score=50.0,
                category_scores={"medication_risk": 80.0, "cardiovascular": 40.0, "infection": 10.0},
            )
        )
        req = RiskScoreRequest(text=TEXT, risk_domains=["medication_risk"])
        resp = _run_risk_scoring(req)
        assert "medication_risk" in resp.category_scores
        assert "cardiovascular" not in resp.category_scores
        assert "infection" not in resp.category_scores

    @patch("app.api.v1.routes.risk.get_risk_scorer")
    def test_no_domain_filter_returns_all(self, mock_get: MagicMock) -> None:
        """When risk_domains is None, all categories are returned."""
        mock_get.return_value = _mock_scorer()
        req = RiskScoreRequest(text=TEXT, risk_domains=None)
        resp = _run_risk_scoring(req)
        assert len(resp.category_scores) == 2  # medication_risk + cardiovascular

    @patch("app.api.v1.routes.risk.get_risk_scorer")
    def test_zero_score_factors_excluded(self, mock_get: MagicMock) -> None:
        """Factors with score == 0 are filtered out."""
        factors = [
            FakeRiskFactor(name="active", score=0.7),
            FakeRiskFactor(name="inactive", score=0.0),
        ]
        mock_get.return_value = _mock_scorer(
            FakeRiskAssessment(factors=factors, overall_score=40.0)
        )
        req = RiskScoreRequest(text=TEXT)
        resp = _run_risk_scoring(req)
        names = [f.name for f in resp.risk_factors]
        assert "active" in names
        assert "inactive" not in names

    @patch("app.api.v1.routes.risk.get_risk_scorer")
    def test_protective_factors_empty(self, mock_get: MagicMock) -> None:
        """Protective factors are typically empty for rule-based scorer."""
        mock_get.return_value = _mock_scorer()
        req = RiskScoreRequest(text=TEXT)
        resp = _run_risk_scoring(req)
        assert resp.protective_factors == []

    @patch("app.api.v1.routes.risk.get_risk_scorer")
    def test_recommendations_forwarded(self, mock_get: MagicMock) -> None:
        """Recommendations from the model are forwarded to the response."""
        recs = ["Immediate review", "Monitor INR"]
        mock_get.return_value = _mock_scorer(
            FakeRiskAssessment(recommendations=recs, overall_score=60.0)
        )
        req = RiskScoreRequest(text=TEXT)
        resp = _run_risk_scoring(req)
        assert resp.recommendations == recs

    @patch("app.api.v1.routes.risk.get_risk_scorer")
    def test_model_metadata(self, mock_get: MagicMock) -> None:
        """Model name and version come from the scorer."""
        mock_get.return_value = _mock_scorer()
        req = RiskScoreRequest(text=TEXT)
        resp = _run_risk_scoring(req)
        assert resp.model_name == "rule-based-risk"
        assert resp.model_version == "1.0.0"

    @patch("app.api.v1.routes.risk.get_risk_scorer")
    def test_factor_source_is_derived(self, mock_get: MagicMock) -> None:
        """All factors from the rule-based scorer have source='derived'."""
        mock_get.return_value = _mock_scorer()
        req = RiskScoreRequest(text=TEXT)
        resp = _run_risk_scoring(req)
        for f in resp.risk_factors:
            assert f.source == "derived"


# ---------------------------------------------------------------------------
# calculate_risk_score
# ---------------------------------------------------------------------------


class TestCalculateRiskScore:
    """Tests for the calculate_risk_score endpoint handler."""

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.risk.get_risk_scorer")
    async def test_happy_path(self, mock_get: MagicMock) -> None:
        """Successful scoring returns RiskScoreResponse."""
        mock_get.return_value = _mock_scorer()
        payload = RiskScoreRequest(text=TEXT)
        resp = await calculate_risk_score(payload, AsyncMock(), MagicMock())

        assert isinstance(resp, RiskScoreResponse)
        assert resp.processing_time_ms >= 0.0
        assert 0.0 <= resp.score <= 1.0

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.risk.get_risk_scorer")
    async def test_timing_overwritten(self, mock_get: MagicMock) -> None:
        """Handler overwrites processing_time_ms from _run_risk_scoring."""
        mock_get.return_value = _mock_scorer()
        payload = RiskScoreRequest(text=TEXT)
        resp = await calculate_risk_score(payload, AsyncMock(), MagicMock())
        assert resp.processing_time_ms >= 0.0

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.risk.get_risk_scorer")
    async def test_inference_error_raises_500(self, mock_get: MagicMock) -> None:
        """InferenceError → HTTP 500."""
        from fastapi import HTTPException

        from app.core.exceptions import InferenceError

        mock_get.return_value = MagicMock(
            assess_risk=MagicMock(side_effect=InferenceError("risk failed"))
        )
        payload = RiskScoreRequest(text=TEXT)

        with pytest.raises(HTTPException) as exc_info:
            await calculate_risk_score(payload, AsyncMock(), MagicMock())
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    @patch("app.api.v1.routes.risk.get_risk_scorer")
    async def test_unexpected_error_raises_500(self, mock_get: MagicMock) -> None:
        """Unexpected exception → HTTP 500."""
        from fastapi import HTTPException

        mock_get.return_value = MagicMock(
            assess_risk=MagicMock(side_effect=MemoryError("OOM"))
        )
        payload = RiskScoreRequest(text=TEXT)

        with pytest.raises(HTTPException) as exc_info:
            await calculate_risk_score(payload, AsyncMock(), MagicMock())
        assert exc_info.value.status_code == 500
        assert "unexpectedly" in exc_info.value.detail
