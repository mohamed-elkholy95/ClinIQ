"""REST API endpoints for clinical vital signs extraction.

Provides endpoints for extracting structured vital sign measurements
from clinical free text, including blood pressure, heart rate,
temperature, respiratory rate, oxygen saturation, weight, height,
BMI, and pain scale readings.

Endpoints
---------
- ``POST /vitals`` — extract vital signs from a single document
- ``POST /vitals/batch`` — batch extraction for up to 50 documents
- ``GET /vitals/types`` — catalogue of 9 extractable vital sign types
- ``GET /vitals/ranges`` — adult reference ranges for clinical interpretation
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.ml.vitals.extractor import (
    _ADULT_RANGES,
    _DIASTOLIC_RANGES,
    ClinicalVitalSignsExtractor,
    VitalSignType,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["vitals"])

# ---------------------------------------------------------------------------
# Singleton extractor
# ---------------------------------------------------------------------------

_extractor = ClinicalVitalSignsExtractor()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class VitalSignReadingResponse(BaseModel):
    """A single vital sign reading."""

    vital_type: str = Field(..., description="Vital sign category")
    value: float = Field(..., description="Numeric value in standard units")
    unit: str = Field(..., description="Standard unit")
    raw_text: str = Field(..., description="Matched text span")
    start: int = Field(..., description="Character offset start")
    end: int = Field(..., description="Character offset end")
    confidence: float = Field(..., description="Extraction confidence [0–1]")
    interpretation: str = Field(..., description="Clinical interpretation")
    secondary_value: float | None = Field(None, description="Diastolic BP value")
    trend: str = Field("unknown", description="Detected trend")
    metadata: dict[str, Any] = Field(default_factory=dict)


class VitalSignsSummary(BaseModel):
    """Aggregate summary of extracted vital signs."""

    total: int
    by_type: dict[str, int]
    critical_findings: list[dict[str, Any]]


class VitalsRequest(BaseModel):
    """Request body for single-document vital signs extraction."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=100_000,
        description="Clinical note text",
    )
    min_confidence: float = Field(
        0.50,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold",
    )


class VitalsResponse(BaseModel):
    """Response body for vital signs extraction."""

    readings: list[VitalSignReadingResponse]
    text_hash: str
    extraction_time_ms: float
    summary: VitalSignsSummary


class VitalsBatchRequest(BaseModel):
    """Request body for batch vital signs extraction."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description="List of clinical note texts (max 50)",
    )
    min_confidence: float = Field(
        0.50,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold",
    )


class VitalsBatchResponse(BaseModel):
    """Response for batch extraction."""

    results: list[VitalsResponse]
    total_readings: int
    total_time_ms: float
    aggregate: dict[str, Any] = Field(
        default_factory=dict,
        description="Aggregate statistics across all documents",
    )


class VitalTypeInfo(BaseModel):
    """Information about a vital sign type."""

    name: str
    description: str
    standard_unit: str


class RangeInfo(BaseModel):
    """Reference range for a vital sign type."""

    vital_type: str
    critical_low: float
    low: float
    high: float
    critical_high: float
    unit: str


# ---------------------------------------------------------------------------
# Type catalogue
# ---------------------------------------------------------------------------

_VITAL_TYPE_INFO: dict[str, VitalTypeInfo] = {
    VitalSignType.BLOOD_PRESSURE: VitalTypeInfo(
        name="Blood Pressure",
        description="Systolic/diastolic arterial pressure measurement",
        standard_unit="mmHg",
    ),
    VitalSignType.HEART_RATE: VitalTypeInfo(
        name="Heart Rate",
        description="Cardiac rhythm frequency (pulse rate)",
        standard_unit="bpm",
    ),
    VitalSignType.TEMPERATURE: VitalTypeInfo(
        name="Temperature",
        description="Core body temperature",
        standard_unit="°F",
    ),
    VitalSignType.RESPIRATORY_RATE: VitalTypeInfo(
        name="Respiratory Rate",
        description="Breathing frequency per minute",
        standard_unit="breaths/min",
    ),
    VitalSignType.OXYGEN_SATURATION: VitalTypeInfo(
        name="Oxygen Saturation",
        description="Peripheral oxygen saturation (SpO2)",
        standard_unit="%",
    ),
    VitalSignType.WEIGHT: VitalTypeInfo(
        name="Weight",
        description="Body weight measurement",
        standard_unit="kg",
    ),
    VitalSignType.HEIGHT: VitalTypeInfo(
        name="Height",
        description="Body height / stature",
        standard_unit="cm",
    ),
    VitalSignType.BMI: VitalTypeInfo(
        name="BMI",
        description="Body Mass Index (weight/height²)",
        standard_unit="kg/m²",
    ),
    VitalSignType.PAIN_SCALE: VitalTypeInfo(
        name="Pain Scale",
        description="Numeric pain rating (0–10)",
        standard_unit="/10",
    ),
}

_UNIT_MAP: dict[str, str] = {
    VitalSignType.BLOOD_PRESSURE: "mmHg",
    VitalSignType.HEART_RATE: "bpm",
    VitalSignType.TEMPERATURE: "°F",
    VitalSignType.RESPIRATORY_RATE: "breaths/min",
    VitalSignType.OXYGEN_SATURATION: "%",
    VitalSignType.WEIGHT: "kg",
    VitalSignType.HEIGHT: "cm",
    VitalSignType.BMI: "kg/m²",
    VitalSignType.PAIN_SCALE: "/10",
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/vitals", response_model=VitalsResponse)
async def extract_vitals(request: VitalsRequest) -> VitalsResponse:
    """Extract vital signs from a single clinical note.

    Parameters
    ----------
    request : VitalsRequest
        Clinical text and optional confidence threshold.

    Returns
    -------
    VitalsResponse
        Extracted vital signs with clinical interpretations.
    """
    try:
        ext = ClinicalVitalSignsExtractor(min_confidence=request.min_confidence)
        result = ext.extract(request.text)
    except Exception as exc:
        logger.exception("Vital signs extraction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    readings = [
        VitalSignReadingResponse(
            vital_type=r.vital_type.value,
            value=r.value,
            unit=r.unit,
            raw_text=r.raw_text,
            start=r.start,
            end=r.end,
            confidence=round(r.confidence, 4),
            interpretation=r.interpretation.value,
            secondary_value=r.secondary_value,
            trend=r.trend.value,
            metadata=r.metadata,
        )
        for r in result.readings
    ]

    return VitalsResponse(
        readings=readings,
        text_hash=result.text_hash,
        extraction_time_ms=round(result.extraction_time_ms, 2),
        summary=VitalSignsSummary(**result.summary),
    )


@router.post("/vitals/batch", response_model=VitalsBatchResponse)
async def extract_vitals_batch(request: VitalsBatchRequest) -> VitalsBatchResponse:
    """Batch extract vital signs from up to 50 clinical notes.

    Parameters
    ----------
    request : VitalsBatchRequest
        List of clinical texts and optional confidence threshold.

    Returns
    -------
    VitalsBatchResponse
        Per-document results with aggregate statistics.
    """
    t0 = time.perf_counter()
    ext = ClinicalVitalSignsExtractor(min_confidence=request.min_confidence)

    results: list[VitalsResponse] = []
    total_readings = 0
    type_counts: dict[str, int] = {}
    all_critical: list[dict[str, Any]] = []

    for text in request.texts:
        try:
            r = ext.extract(text)
        except Exception:
            logger.exception("Batch item extraction failed")
            results.append(VitalsResponse(
                readings=[],
                text_hash="",
                extraction_time_ms=0.0,
                summary=VitalSignsSummary(total=0, by_type={}, critical_findings=[]),
            ))
            continue

        readings_resp = [
            VitalSignReadingResponse(
                vital_type=rd.vital_type.value,
                value=rd.value,
                unit=rd.unit,
                raw_text=rd.raw_text,
                start=rd.start,
                end=rd.end,
                confidence=round(rd.confidence, 4),
                interpretation=rd.interpretation.value,
                secondary_value=rd.secondary_value,
                trend=rd.trend.value,
                metadata=rd.metadata,
            )
            for rd in r.readings
        ]

        total_readings += len(readings_resp)
        for rd in r.readings:
            k = rd.vital_type.value
            type_counts[k] = type_counts.get(k, 0) + 1

        all_critical.extend(r.summary.get("critical_findings", []))

        results.append(VitalsResponse(
            readings=readings_resp,
            text_hash=r.text_hash,
            extraction_time_ms=round(r.extraction_time_ms, 2),
            summary=VitalSignsSummary(**r.summary),
        ))

    elapsed = (time.perf_counter() - t0) * 1000.0

    return VitalsBatchResponse(
        results=results,
        total_readings=total_readings,
        total_time_ms=round(elapsed, 2),
        aggregate={
            "documents_processed": len(results),
            "total_readings": total_readings,
            "by_type": type_counts,
            "critical_findings_count": len(all_critical),
        },
    )


@router.get("/vitals/types", response_model=list[VitalTypeInfo])
async def list_vital_types() -> list[VitalTypeInfo]:
    """Return catalogue of 9 extractable vital sign types.

    Returns
    -------
    list[VitalTypeInfo]
        Type name, description, and standard unit for each vital sign.
    """
    return list(_VITAL_TYPE_INFO.values())


@router.get("/vitals/ranges", response_model=list[RangeInfo])
async def list_vital_ranges() -> list[RangeInfo]:
    """Return adult reference ranges for clinical interpretation.

    Returns
    -------
    list[RangeInfo]
        Reference ranges with critical/normal boundaries.
    """
    result: list[RangeInfo] = []
    for vtype, (crit_low, low, high, crit_high) in _ADULT_RANGES.items():
        result.append(RangeInfo(
            vital_type=vtype.value,
            critical_low=crit_low,
            low=low,
            high=high,
            critical_high=crit_high,
            unit=_UNIT_MAP.get(vtype, ""),
        ))
    # Add diastolic as separate entry
    result.append(RangeInfo(
        vital_type="diastolic_bp",
        critical_low=_DIASTOLIC_RANGES[0],
        low=_DIASTOLIC_RANGES[1],
        high=_DIASTOLIC_RANGES[2],
        critical_high=_DIASTOLIC_RANGES[3],
        unit="mmHg",
    ))
    return result
