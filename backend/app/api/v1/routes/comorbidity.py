"""Comorbidity scoring API endpoints.

Provides REST endpoints for Charlson Comorbidity Index (CCI) calculation
from ICD-10-CM codes and/or free-text clinical narratives, with optional
age adjustment and 10-year mortality estimates.
"""

import logging
import time
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.ml.comorbidity import (
    CCICategory,
    CCIResult,
    CharlsonCalculator,
    CharlsonConfig,
)

logger = logging.getLogger(__name__)

# Module-level calculator instance (stateless, safe to share).
_calculator: CharlsonCalculator | None = None


def _get_calculator() -> CharlsonCalculator:
    """Get or create the Charlson calculator singleton.

    Returns
    -------
    CharlsonCalculator
        Shared calculator instance.
    """
    global _calculator
    if _calculator is None:
        _calculator = CharlsonCalculator()
    return _calculator


# -- Request / Response schemas --


class CCIRequest(BaseModel):
    """Request for Charlson Comorbidity Index calculation.

    Parameters
    ----------
    icd_codes : list[str] | None
        ICD-10-CM codes (e.g., ["I21.0", "E11.9"]).
    text : str | None
        Free-text clinical narrative to scan for comorbidities.
    patient_age : int | None
        Patient age for age-adjusted scoring.
    age_adjust : bool
        Whether to include age-based adjustment points.
    include_text_extraction : bool
        Whether to also scan free text for conditions.
    hierarchical_exclusion : bool
        When True, mild categories are excluded when severe variants present.
    """

    icd_codes: list[str] | None = Field(None, max_length=200, description="ICD-10-CM codes")
    text: str | None = Field(None, max_length=100_000, description="Clinical text")
    patient_age: int | None = Field(None, ge=0, le=130, description="Patient age")
    age_adjust: bool = Field(False, description="Enable age adjustment")
    include_text_extraction: bool = Field(True, description="Extract from text")
    hierarchical_exclusion: bool = Field(True, description="Enable hierarchical exclusion")

    @field_validator("icd_codes", "text")
    @classmethod
    def at_least_one_input(cls, v: Any, _info: Any) -> Any:
        """Validate on model level; actual check in endpoint."""
        return v


class ComorbidityMatchResponse(BaseModel):
    """A single matched comorbidity condition.

    Parameters
    ----------
    category : str
        CCI disease category name.
    weight : int
        Charlson weight (1, 2, 3, or 6).
    source : str
        Match source: "icd_code" or "text".
    evidence : str
        Code or phrase that triggered the match.
    confidence : float
        Match confidence (0.0–1.0).
    description : str
        Human-readable category description.
    """

    category: str
    weight: int
    source: str
    evidence: str
    confidence: float
    description: str


class MortalityEstimateResponse(BaseModel):
    """10-year mortality estimate.

    Parameters
    ----------
    ten_year_mortality : float
        Estimated 10-year mortality probability.
    ten_year_survival : float
        Estimated 10-year survival probability.
    risk_group : str
        Risk group: low, mild, moderate, or severe.
    """

    ten_year_mortality: float
    ten_year_survival: float
    risk_group: str


class CCIResponse(BaseModel):
    """Complete CCI calculation response.

    Parameters
    ----------
    raw_score : int
        Sum of Charlson weights.
    age_adjusted_score : int | None
        Score with age points, if applicable.
    matched_categories : list[ComorbidityMatchResponse]
        All matched comorbidity conditions.
    excluded_categories : list[ComorbidityMatchResponse]
        Categories removed by hierarchical exclusion.
    mortality_estimate : MortalityEstimateResponse
        10-year mortality/survival estimates.
    category_count : int
        Number of distinct CCI categories detected.
    processing_time_ms : float
        Calculation time.
    """

    raw_score: int
    age_adjusted_score: int | None
    matched_categories: list[ComorbidityMatchResponse]
    excluded_categories: list[ComorbidityMatchResponse]
    mortality_estimate: MortalityEstimateResponse
    category_count: int
    processing_time_ms: float


class CCIBatchRequest(BaseModel):
    """Batch CCI calculation request.

    Parameters
    ----------
    patients : list[CCIRequest]
        Up to 50 patient records.
    """

    patients: list[CCIRequest] = Field(
        ..., min_length=1, max_length=50, description="Patient records"
    )


class CCIBatchSummary(BaseModel):
    """Aggregate statistics for a batch calculation.

    Parameters
    ----------
    total : int
        Number of patients processed.
    avg_score : float
        Average raw CCI score.
    min_score : int
        Minimum raw CCI score.
    max_score : int
        Maximum raw CCI score.
    risk_distribution : dict[str, int]
        Count per risk group.
    processing_time_ms : float
        Total batch processing time.
    """

    total: int
    avg_score: float
    min_score: int
    max_score: int
    risk_distribution: dict[str, int]
    processing_time_ms: float


class CCIBatchResponse(BaseModel):
    """Batch CCI calculation response.

    Parameters
    ----------
    results : list[CCIResponse]
        Individual calculation results.
    summary : CCIBatchSummary
        Aggregate statistics.
    """

    results: list[CCIResponse]
    summary: CCIBatchSummary


class CategoryInfoResponse(BaseModel):
    """Information about a CCI disease category.

    Parameters
    ----------
    category : str
        Category identifier.
    weight : int
        Charlson weight.
    description : str
        Human-readable description.
    """

    category: str
    weight: int
    description: str


# -- Conversion helpers --


def _result_to_response(result: CCIResult) -> CCIResponse:
    """Convert internal CCIResult to API response.

    Parameters
    ----------
    result : CCIResult
        Internal calculation result.

    Returns
    -------
    CCIResponse
        API response model.
    """
    return CCIResponse(
        raw_score=result.raw_score,
        age_adjusted_score=result.age_adjusted_score,
        matched_categories=[
            ComorbidityMatchResponse(
                category=m.category.value,
                weight=m.weight,
                source=m.source,
                evidence=m.evidence,
                confidence=round(m.confidence, 4),
                description=m.description,
            )
            for m in result.matched_categories
        ],
        excluded_categories=[
            ComorbidityMatchResponse(
                category=m.category.value,
                weight=m.weight,
                source=m.source,
                evidence=m.evidence,
                confidence=round(m.confidence, 4),
                description=m.description,
            )
            for m in result.excluded_categories
        ],
        mortality_estimate=MortalityEstimateResponse(
            ten_year_mortality=round(result.mortality_estimate.ten_year_mortality, 4),
            ten_year_survival=round(result.mortality_estimate.ten_year_survival, 4),
            risk_group=result.mortality_estimate.risk_group,
        ),
        category_count=result.category_count,
        processing_time_ms=round(result.processing_time_ms, 2),
    )


# -- Route handlers --

try:
    from fastapi import APIRouter, HTTPException

    router = APIRouter(prefix="/comorbidity", tags=["comorbidity"])

    @router.post("", response_model=CCIResponse)
    async def calculate_cci(request: CCIRequest) -> CCIResponse:
        """Calculate Charlson Comorbidity Index for a single patient.

        Accepts ICD-10-CM codes and/or free-text clinical narrative and
        returns the CCI score with matched comorbidities, hierarchical
        exclusions, and 10-year mortality estimates.

        Parameters
        ----------
        request : CCIRequest
            Patient data with codes and/or text.

        Returns
        -------
        CCIResponse
            Complete CCI calculation result.

        Raises
        ------
        HTTPException
            422 if neither codes nor text provided.
        """
        if not request.icd_codes and not request.text:
            raise HTTPException(
                status_code=422,
                detail="At least one of icd_codes or text must be provided",
            )

        calculator = _get_calculator()
        config = CharlsonConfig(
            age_adjust=request.age_adjust,
            patient_age=request.patient_age,
            include_text_extraction=request.include_text_extraction,
            hierarchical_exclusion=request.hierarchical_exclusion,
        )

        result = calculator.calculate(
            icd_codes=request.icd_codes,
            text=request.text,
            config=config,
        )
        return _result_to_response(result)

    @router.post("/batch", response_model=CCIBatchResponse)
    async def calculate_cci_batch(request: CCIBatchRequest) -> CCIBatchResponse:
        """Calculate CCI for multiple patients in a single request.

        Parameters
        ----------
        request : CCIBatchRequest
            Batch of patient records.

        Returns
        -------
        CCIBatchResponse
            Individual results with aggregate summary.

        Raises
        ------
        HTTPException
            422 if any patient record lacks both codes and text.
        """
        calculator = _get_calculator()
        start_time = time.perf_counter()

        results: list[CCIResponse] = []
        scores: list[int] = []
        risk_dist: dict[str, int] = {}

        for i, patient in enumerate(request.patients):
            if not patient.icd_codes and not patient.text:
                raise HTTPException(
                    status_code=422,
                    detail=f"Patient {i}: at least one of icd_codes or text must be provided",
                )

            config = CharlsonConfig(
                age_adjust=patient.age_adjust,
                patient_age=patient.patient_age,
                include_text_extraction=patient.include_text_extraction,
                hierarchical_exclusion=patient.hierarchical_exclusion,
            )

            result = calculator.calculate(
                icd_codes=patient.icd_codes,
                text=patient.text,
                config=config,
            )

            response = _result_to_response(result)
            results.append(response)
            scores.append(result.raw_score)

            rg = result.mortality_estimate.risk_group
            risk_dist[rg] = risk_dist.get(rg, 0) + 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        summary = CCIBatchSummary(
            total=len(results),
            avg_score=round(sum(scores) / len(scores), 2) if scores else 0.0,
            min_score=min(scores) if scores else 0,
            max_score=max(scores) if scores else 0,
            risk_distribution=risk_dist,
            processing_time_ms=round(elapsed_ms, 2),
        )

        return CCIBatchResponse(results=results, summary=summary)

    @router.get("/categories", response_model=list[CategoryInfoResponse])
    async def list_cci_categories() -> list[CategoryInfoResponse]:
        """List all 17 Charlson Comorbidity Index categories.

        Returns
        -------
        list[CategoryInfoResponse]
            Category names, weights, and descriptions.
        """
        calculator = _get_calculator()
        return [
            CategoryInfoResponse(**info)
            for info in calculator.get_category_info()
        ]

    @router.get("/categories/{category_name}")
    async def get_category_detail(category_name: str) -> dict[str, Any]:
        """Get detailed information about a specific CCI category.

        Includes the ICD-10-CM code prefixes that map to this category
        and the text patterns used for free-text extraction.

        Parameters
        ----------
        category_name : str
            Category identifier (e.g., "myocardial_infarction").

        Returns
        -------
        dict[str, Any]
            Category details with code mappings.

        Raises
        ------
        HTTPException
            404 if category name not found.
        """
        try:
            category = CCICategory(category_name)
        except ValueError:
            valid = [c.value for c in CCICategory]
            raise HTTPException(
                status_code=404,
                detail=f"Category '{category_name}' not found. Valid: {valid}",
            )

        from app.ml.comorbidity.charlson import (
            CATEGORY_DESCRIPTIONS,
            CATEGORY_WEIGHTS,
            ICD10_PREFIXES,
        )

        return {
            "category": category.value,
            "weight": CATEGORY_WEIGHTS[category],
            "description": CATEGORY_DESCRIPTIONS[category],
            "icd10_prefixes": list(ICD10_PREFIXES.get(category, ())),
            "prefix_count": len(ICD10_PREFIXES.get(category, ())),
        }

except ImportError:
    router = None  # type: ignore[assignment]
    logger.warning("FastAPI not available — comorbidity routes not registered")
