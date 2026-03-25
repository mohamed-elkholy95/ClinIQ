"""Tests for comorbidity scoring API endpoints.

Covers:
- POST /comorbidity (single CCI calculation)
- POST /comorbidity/batch (batch CCI calculation)
- GET /comorbidity/categories (list all categories)
- GET /comorbidity/categories/{name} (category detail)
- Validation and error handling
"""

import pytest

from app.api.v1.routes.comorbidity import (
    CategoryInfoResponse,
    CCIBatchRequest,
    CCIRequest,
    CCIResponse,
    _result_to_response,
)
from app.ml.comorbidity import CharlsonCalculator, CharlsonConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def calculator() -> CharlsonCalculator:
    return CharlsonCalculator()


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


class TestSchemas:
    """Test Pydantic request/response schema validation."""

    def test_cci_request_with_codes(self) -> None:
        req = CCIRequest(icd_codes=["I21.0", "E11.9"])
        assert len(req.icd_codes) == 2

    def test_cci_request_with_text(self) -> None:
        req = CCIRequest(text="Patient has COPD")
        assert req.text == "Patient has COPD"

    def test_cci_request_defaults(self) -> None:
        req = CCIRequest(icd_codes=["I21.0"])
        assert req.age_adjust is False
        assert req.include_text_extraction is True
        assert req.hierarchical_exclusion is True
        assert req.patient_age is None


# ---------------------------------------------------------------------------
# Result conversion
# ---------------------------------------------------------------------------


class TestResultConversion:
    """Test internal → API response conversion."""

    def test_result_to_response(self, calculator: CharlsonCalculator) -> None:
        result = calculator.calculate(
            icd_codes=["I21.0", "E11.9"],
            config=CharlsonConfig(include_text_extraction=False),
        )
        response = _result_to_response(result)
        assert isinstance(response, CCIResponse)
        assert response.raw_score == 2
        assert response.category_count == 2
        assert len(response.matched_categories) == 2

    def test_result_with_exclusion(self, calculator: CharlsonCalculator) -> None:
        result = calculator.calculate(
            icd_codes=["E11.9", "E11.2"],
            config=CharlsonConfig(include_text_extraction=False),
        )
        response = _result_to_response(result)
        assert len(response.excluded_categories) == 1
        assert response.excluded_categories[0].category == "diabetes_uncomplicated"

    def test_result_mortality_fields(self, calculator: CharlsonCalculator) -> None:
        result = calculator.calculate(icd_codes=["I21.0"])
        response = _result_to_response(result)
        assert 0.0 <= response.mortality_estimate.ten_year_mortality <= 1.0
        assert response.mortality_estimate.risk_group in ("low", "mild", "moderate", "severe")


# ---------------------------------------------------------------------------
# Route handler logic (direct invocation, no HTTP)
# ---------------------------------------------------------------------------


class TestRouteHandlers:
    """Test route handler functions directly."""

    @pytest.mark.asyncio
    async def test_calculate_cci_with_codes(self) -> None:
        from app.api.v1.routes.comorbidity import calculate_cci

        request = CCIRequest(icd_codes=["I21.0", "J44.1"])
        response = await calculate_cci(request)
        assert response.raw_score == 2
        assert response.category_count == 2

    @pytest.mark.asyncio
    async def test_calculate_cci_with_text(self) -> None:
        from app.api.v1.routes.comorbidity import calculate_cci

        request = CCIRequest(text="Patient has congestive heart failure and COPD")
        response = await calculate_cci(request)
        assert response.raw_score >= 2

    @pytest.mark.asyncio
    async def test_calculate_cci_with_age(self) -> None:
        from app.api.v1.routes.comorbidity import calculate_cci

        request = CCIRequest(
            icd_codes=["I21.0"],
            age_adjust=True,
            patient_age=72,
        )
        response = await calculate_cci(request)
        assert response.age_adjusted_score == 4  # 1 (MI) + 3 (age)

    @pytest.mark.asyncio
    async def test_calculate_cci_no_input_raises(self) -> None:
        from fastapi import HTTPException

        from app.api.v1.routes.comorbidity import calculate_cci

        request = CCIRequest()
        with pytest.raises(HTTPException) as exc_info:
            await calculate_cci(request)
        assert exc_info.value.status_code == 422

    @pytest.mark.asyncio
    async def test_batch_cci(self) -> None:
        from app.api.v1.routes.comorbidity import calculate_cci_batch

        request = CCIBatchRequest(
            patients=[
                CCIRequest(icd_codes=["I21.0"]),
                CCIRequest(icd_codes=["I21.0", "E11.9", "N18.3"]),
                CCIRequest(text="Patient has COPD"),
            ]
        )
        response = await calculate_cci_batch(request)
        assert response.summary.total == 3
        assert response.summary.min_score <= response.summary.max_score
        assert len(response.results) == 3

    @pytest.mark.asyncio
    async def test_batch_cci_summary_stats(self) -> None:
        from app.api.v1.routes.comorbidity import calculate_cci_batch

        request = CCIBatchRequest(
            patients=[
                CCIRequest(icd_codes=["I21.0"]),  # score 1
                CCIRequest(icd_codes=["C78.0"]),  # score 6
            ]
        )
        response = await calculate_cci_batch(request)
        assert response.summary.min_score == 1
        assert response.summary.max_score == 6
        assert response.summary.avg_score == 3.5

    @pytest.mark.asyncio
    async def test_batch_cci_risk_distribution(self) -> None:
        from app.api.v1.routes.comorbidity import calculate_cci_batch

        request = CCIBatchRequest(
            patients=[
                CCIRequest(icd_codes=["I21.0"]),  # score 1 → mild
                CCIRequest(icd_codes=["C78.0"]),  # score 6 → severe
            ]
        )
        response = await calculate_cci_batch(request)
        assert "mild" in response.summary.risk_distribution
        assert "severe" in response.summary.risk_distribution

    @pytest.mark.asyncio
    async def test_list_categories(self) -> None:
        from app.api.v1.routes.comorbidity import list_cci_categories

        categories = await list_cci_categories()
        assert len(categories) == 17
        assert all(isinstance(c, CategoryInfoResponse) for c in categories)

    @pytest.mark.asyncio
    async def test_get_category_detail(self) -> None:
        from app.api.v1.routes.comorbidity import get_category_detail

        detail = await get_category_detail("myocardial_infarction")
        assert detail["category"] == "myocardial_infarction"
        assert detail["weight"] == 1
        assert len(detail["icd10_prefixes"]) > 0
        assert detail["prefix_count"] > 0

    @pytest.mark.asyncio
    async def test_get_category_detail_not_found(self) -> None:
        from fastapi import HTTPException

        from app.api.v1.routes.comorbidity import get_category_detail

        with pytest.raises(HTTPException) as exc_info:
            await get_category_detail("nonexistent")
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_batch_empty_patient_raises(self) -> None:
        from fastapi import HTTPException

        from app.api.v1.routes.comorbidity import calculate_cci_batch

        request = CCIBatchRequest(
            patients=[
                CCIRequest(icd_codes=["I21.0"]),
                CCIRequest(),  # No codes or text.
            ]
        )
        with pytest.raises(HTTPException) as exc_info:
            await calculate_cci_batch(request)
        assert exc_info.value.status_code == 422
