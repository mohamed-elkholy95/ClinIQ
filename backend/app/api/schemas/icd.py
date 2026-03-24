"""ICD-10 prediction request and response schemas."""

from typing import Annotated

from pydantic import BaseModel, Field

MAX_TEXT_LENGTH = 100_000

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class ICDPredictionRequest(BaseModel):
    """Request body for ICD-10 code prediction."""

    text: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_TEXT_LENGTH,
            description="Clinical text to classify (plain text, up to 100 000 characters)",
        ),
    ]
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of ICD-10 predictions to return, ordered by confidence descending",
    )
    min_confidence: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Discard predictions whose confidence is below this threshold",
    )
    model: str = Field(
        default="sklearn-baseline",
        description=(
            "Classifier backend to use. "
            "One of 'sklearn-baseline', 'transformer' (Bio_ClinicalBERT), "
            "or 'hierarchical'. Defaults to the fast sklearn baseline."
        ),
    )
    include_chapter: bool = Field(
        default=True,
        description="Annotate each returned code with its ICD-10 chapter description",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": (
                    "62-year-old male admitted with acute onset chest pain radiating to the left arm. "
                    "ECG shows ST-segment elevation in leads V1-V4. History of type 2 diabetes and "
                    "hypertension. Started on aspirin 325 mg and heparin infusion."
                ),
                "top_k": 5,
                "min_confidence": 0.2,
                "model": "sklearn-baseline",
                "include_chapter": True,
            }
        }
    }


# ---------------------------------------------------------------------------
# Response atoms
# ---------------------------------------------------------------------------


class ICDCodeResponse(BaseModel):
    """A single ICD-10 code prediction with metadata."""

    code: str = Field(
        description="ICD-10-CM code string (e.g. 'I21.9')",
        examples=["I21.9", "E11.9", "I10"],
    )
    description: str | None = Field(
        default=None,
        description="Human-readable description of the ICD-10 code",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Model confidence score in [0, 1]",
    )
    chapter: str | None = Field(
        default=None,
        description="ICD-10 chapter the code belongs to (e.g. 'Diseases of the circulatory system')",
    )
    category: str | None = Field(
        default=None,
        description="ICD-10 category within the chapter, when available",
    )
    contributing_text: list[str] | None = Field(
        default=None,
        description="Text spans from the input that most influenced this prediction",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "code": "I21.9",
                "description": "Acute myocardial infarction, unspecified",
                "confidence": 0.87,
                "chapter": "Diseases of the circulatory system",
                "category": "Ischaemic heart diseases",
                "contributing_text": ["acute onset chest pain", "ST-segment elevation"],
            }
        }
    }


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------


class ICDPredictionResponse(BaseModel):
    """Top-level ICD-10 prediction response."""

    predictions: list[ICDCodeResponse] = Field(
        description="Predicted ICD-10 codes sorted by confidence descending"
    )
    prediction_count: int = Field(
        ge=0,
        description="Number of predictions returned (after top_k and min_confidence filtering)",
    )
    model_name: str = Field(description="Name of the classifier that produced these predictions")
    model_version: str = Field(description="Version string of the deployed model")
    processing_time_ms: float = Field(ge=0, description="End-to-end inference latency in milliseconds")
    document_summary: str | None = Field(
        default=None,
        description="Brief auto-generated summary of the document used during classification, if available",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "predictions": [
                    {
                        "code": "I21.9",
                        "description": "Acute myocardial infarction, unspecified",
                        "confidence": 0.87,
                        "chapter": "Diseases of the circulatory system",
                        "category": "Ischaemic heart diseases",
                        "contributing_text": ["ST-segment elevation", "chest pain"],
                    },
                    {
                        "code": "E11.9",
                        "description": "Type 2 diabetes mellitus without complications",
                        "confidence": 0.74,
                        "chapter": "Endocrine, nutritional and metabolic diseases",
                        "category": None,
                        "contributing_text": ["type 2 diabetes"],
                    },
                ],
                "prediction_count": 2,
                "model_name": "sklearn-baseline",
                "model_version": "1.0.0",
                "processing_time_ms": 38.1,
                "document_summary": None,
            }
        }
    }
