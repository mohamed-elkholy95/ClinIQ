"""Risk scoring request and response schemas."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

MAX_TEXT_LENGTH = 100_000

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class RiskScoreRequest(BaseModel):
    """Request body for clinical risk scoring."""

    text: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_TEXT_LENGTH,
            description="Clinical text to assess for risk (plain text, up to 100 000 characters)",
        ),
    ]
    run_ner: bool = Field(
        default=True,
        description=(
            "When True, run NER on the input text first and use extracted entities "
            "to enrich the risk calculation. Set to False to skip NER for speed."
        ),
    )
    run_icd: bool = Field(
        default=False,
        description=(
            "When True, run ICD-10 prediction on the input text and incorporate the "
            "predicted codes into the risk calculation. Adds latency."
        ),
    )
    category_weights: dict[str, float] | None = Field(
        default=None,
        description=(
            "Override the default per-category weights used when aggregating the overall score. "
            "Keys must be valid risk categories: medication, cardiovascular, infection, surgical, follow_up. "
            "Values must sum to approximately 1.0. Omit to use server defaults."
        ),
    )
    min_risk_factor_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum entity confidence required for entity-derived risk factors to be included",
    )
    risk_domains: list[str] | None = Field(
        default=None,
        description=(
            "Restrict risk scoring to specific clinical domains (e.g. 'cardiovascular', "
            "'medication_risk'). Omit to score all domains."
        ),
    )
    patient_context: dict[str, str | int | float | bool] | None = Field(
        default=None,
        description=(
            "Optional patient demographics and context (age, gender, comorbidities) "
            "to augment text-based risk prediction."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": (
                    "Patient is a 75-year-old male admitted emergently with acute myocardial infarction. "
                    "History includes heart failure, chronic kidney disease stage 3, and type 2 diabetes. "
                    "Currently on warfarin, digoxin, and insulin. Non-compliant with follow-up appointments. "
                    "Lab: creatinine 2.4, INR 3.8 (supratherapeutic)."
                ),
                "run_ner": True,
                "run_icd": False,
                "category_weights": None,
                "min_risk_factor_confidence": 0.5,
            }
        }
    }


# ---------------------------------------------------------------------------
# Response atoms
# ---------------------------------------------------------------------------


class RiskFactorResponse(BaseModel):
    """A single contributing risk or protective factor."""

    name: str = Field(description="Machine-readable factor identifier (e.g. 'polypharmacy', 'high_risk_med_warfarin')")
    description: str = Field(description="Human-readable description of the factor")
    weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Relative importance of this factor in the overall score (0 = minimal, 1 = critical)",
    )
    value: float = Field(
        ge=0.0,
        le=1.0,
        description="Activation strength of this factor (0 = not present, 1 = fully present)",
    )
    source: Literal["entity", "text", "derived", "icd"] = Field(
        description=(
            "How this factor was identified: "
            "'entity' from NER output, 'text' from pattern matching, "
            "'derived' from aggregate signals, 'icd' from ICD-10 predictions"
        )
    )
    evidence: str | None = Field(
        default=None,
        description="Textual evidence that triggered this factor (e.g. 'warfarin', 'INR 3.8')",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "name": "high_risk_med_warfarin",
                "description": "High-risk medication: warfarin",
                "weight": 0.7,
                "value": 1.0,
                "source": "entity",
                "evidence": "warfarin",
            }
        }
    }


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------


class RiskScoreResponse(BaseModel):
    """Top-level risk scoring response."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Overall risk score in [0, 1] where 0 = minimal risk and 1 = critical risk",
    )
    category: Literal["low", "moderate", "high", "critical"] = Field(
        description=(
            "Qualitative risk category derived from the overall score: "
            "low (<0.4), moderate (0.4-0.6), high (0.6-0.8), critical (>=0.8)"
        )
    )
    category_scores: dict[str, float] = Field(
        description=(
            "Per-domain risk scores in [0, 1]. "
            "Keys: medication, cardiovascular, infection, surgical, follow_up"
        )
    )
    risk_factors: list[RiskFactorResponse] = Field(
        description="Top contributing risk factors, ordered by (weight * value) descending (max 10)"
    )
    protective_factors: list[RiskFactorResponse] = Field(
        description="Identified protective / mitigating factors (max 5)"
    )
    recommendations: list[str] = Field(
        description="Actionable clinical recommendations based on the risk profile (max 5)"
    )
    model_name: str = Field(description="Name of the risk scoring model")
    model_version: str = Field(description="Version string of the deployed model")
    processing_time_ms: float = Field(ge=0, description="End-to-end inference latency in milliseconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "score": 0.83,
                "category": "critical",
                "category_scores": {
                    "medication": 0.85,
                    "cardiovascular": 0.90,
                    "infection": 0.20,
                    "surgical": 0.10,
                    "follow_up": 0.40,
                },
                "risk_factors": [
                    {
                        "name": "cardiac_arrest",
                        "description": "High-risk condition: acute myocardial infarction",
                        "weight": 0.95,
                        "value": 1.0,
                        "source": "text",
                        "evidence": "acute myocardial infarction",
                    }
                ],
                "protective_factors": [],
                "recommendations": [
                    "Immediate clinical review recommended",
                    "Consider escalation to specialist care",
                    "Medication review recommended - assess for interactions",
                    "Monitor closely for: cardiac_arrest, high_risk_med_warfarin",
                ],
                "model_name": "rule-based-risk",
                "model_version": "1.0.0",
                "processing_time_ms": 55.2,
            }
        }
    }
