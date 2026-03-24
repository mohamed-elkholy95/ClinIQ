"""Summarization request and response schemas."""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

MAX_TEXT_LENGTH = 100_000

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class SummarizationRequest(BaseModel):
    """Request body for clinical text summarization."""

    text: Annotated[
        str,
        Field(
            min_length=10,
            max_length=MAX_TEXT_LENGTH,
            description="Clinical text to summarize (plain text, up to 100 000 characters)",
        ),
    ]
    detail_level: Literal["brief", "standard", "detailed"] = Field(
        default="standard",
        description=(
            "Controls the verbosity of the generated summary. "
            "'brief' targets ~50 words, 'standard' ~150 words, 'detailed' ~300 words."
        ),
    )
    model: Literal["extractive", "section-based", "abstractive", "hybrid"] = Field(
        default="extractive",
        description=(
            "Summarization backend to use. "
            "'extractive' selects key sentences (TextRank). "
            "'section-based' preserves clinical section structure. "
            "'abstractive' generates novel text (requires transformer). "
            "'hybrid' combines extractive pre-processing with abstractive generation."
        ),
    )
    max_length_words: int | None = Field(
        default=None,
        ge=10,
        le=2000,
        description=(
            "Hard cap on output word count. Overrides the default set by detail_level. "
            "Omit to use the detail_level default."
        ),
    )
    min_length_words: int | None = Field(
        default=None,
        ge=5,
        le=500,
        description=(
            "Minimum output word count. Overrides the detail_level default. "
            "Omit to use the detail_level default."
        ),
    )
    include_key_points: bool = Field(
        default=True,
        description="Return a separate list of key clinical bullet points alongside the summary prose",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": (
                    "CHIEF COMPLAINT: Shortness of breath and productive cough for 3 days.\n\n"
                    "HPI: 58-year-old female with COPD presents with worsening dyspnoea and increased "
                    "sputum production. Sputum is yellow-green. No haemoptysis. Denies fever. "
                    "O2 sat 91% on room air.\n\n"
                    "ASSESSMENT: COPD exacerbation with possible community-acquired pneumonia.\n\n"
                    "PLAN: Azithromycin 500 mg daily for 5 days, prednisone 40 mg daily for 5 days, "
                    "albuterol nebulisers q4h. Chest X-ray ordered."
                ),
                "detail_level": "standard",
                "model": "extractive",
                "max_length_words": None,
                "min_length_words": None,
                "include_key_points": True,
            }
        }
    }


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


class SummarizationResponse(BaseModel):
    """Top-level summarization response."""

    summary: str = Field(description="Generated summary text")
    key_points: list[str] | None = Field(
        default=None,
        description="Top key clinical bullet points extracted during summarization, if requested",
    )
    original_word_count: int = Field(
        ge=0,
        description="Word count of the original input text",
    )
    summary_word_count: int = Field(
        ge=0,
        description="Word count of the generated summary",
    )
    compression_ratio: float = Field(
        ge=0.0,
        description=(
            "Ratio of original_word_count to summary_word_count. "
            "A value of 4.0 means the summary is 4x shorter than the original."
        ),
    )
    summary_type: Literal["extractive", "abstractive", "hybrid"] = Field(
        description="The summarization strategy actually applied (may differ from the requested model for short inputs)"
    )
    model_name: str = Field(description="Name of the model that produced this summary")
    model_version: str = Field(description="Version string of the deployed model")
    processing_time_ms: float = Field(ge=0, description="End-to-end inference latency in milliseconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "summary": (
                    "58-year-old female with COPD presents with 3-day history of worsening dyspnoea "
                    "and productive yellow-green sputum. O2 sat 91% on room air. "
                    "Assessment: COPD exacerbation with possible CAP. "
                    "Plan: Azithromycin, prednisone, albuterol nebulisers, and chest X-ray."
                ),
                "key_points": [
                    "COPD exacerbation with possible community-acquired pneumonia",
                    "O2 saturation 91% on room air",
                    "Treatment: Azithromycin 500 mg + prednisone 40 mg for 5 days",
                ],
                "original_word_count": 98,
                "summary_word_count": 52,
                "compression_ratio": 1.88,
                "summary_type": "extractive",
                "model_name": "textrank",
                "model_version": "1.0.0",
                "processing_time_ms": 18.4,
            }
        }
    }
