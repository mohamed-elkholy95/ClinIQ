"""Full pipeline analysis request and response schemas."""

from typing import Annotated

from pydantic import BaseModel, Field, model_validator

from app.api.schemas.icd import ICDCodeResponse
from app.api.schemas.ner import EntityResponse
from app.api.schemas.risk import RiskFactorResponse
from app.api.schemas.summary import SummarizationResponse

MAX_TEXT_LENGTH = 100_000

# ---------------------------------------------------------------------------
# Pipeline configuration sub-models
# ---------------------------------------------------------------------------


class NERConfig(BaseModel):
    """NER stage configuration for the full pipeline."""

    enabled: bool = Field(default=True, description="Run named entity recognition")
    model: str = Field(
        default="rule-based",
        description="NER backend: 'rule-based', 'spacy', or 'transformer'",
    )
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for returned entities",
    )
    include_negated: bool = Field(default=True, description="Include entities in negated contexts")
    include_uncertain: bool = Field(default=True, description="Include entities flagged as uncertain")

    model_config = {
        "json_schema_extra": {
            "example": {"enabled": True, "model": "rule-based", "min_confidence": 0.5}
        }
    }


class ICDConfig(BaseModel):
    """ICD-10 prediction stage configuration for the full pipeline."""

    enabled: bool = Field(default=True, description="Run ICD-10 code prediction")
    model: str = Field(
        default="sklearn-baseline",
        description="ICD classifier backend: 'sklearn-baseline', 'transformer', or 'hierarchical'",
    )
    top_k: int = Field(default=10, ge=1, le=50, description="Maximum ICD codes to return")
    min_confidence: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for returned codes",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"enabled": True, "model": "sklearn-baseline", "top_k": 5}
        }
    }


class SummaryConfig(BaseModel):
    """Summarization stage configuration for the full pipeline."""

    enabled: bool = Field(default=True, description="Run clinical text summarization")
    model: str = Field(
        default="extractive",
        description="Summarizer backend: 'extractive', 'section-based', 'abstractive', or 'hybrid'",
    )
    detail_level: str = Field(
        default="standard",
        description="Verbosity level: 'brief', 'standard', or 'detailed'",
    )
    include_key_points: bool = Field(
        default=True,
        description="Include key bullet points in the summary output",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"enabled": True, "model": "extractive", "detail_level": "standard"}
        }
    }


class RiskConfig(BaseModel):
    """Risk scoring stage configuration for the full pipeline."""

    enabled: bool = Field(default=True, description="Run clinical risk scoring")
    category_weights: dict[str, float] | None = Field(
        default=None,
        description=(
            "Override per-category weights for the overall score aggregation. "
            "Keys: medication, cardiovascular, infection, surgical, follow_up. "
            "Omit to use server defaults."
        ),
    )

    model_config = {
        "json_schema_extra": {"example": {"enabled": True, "category_weights": None}}
    }


class PipelineConfig(BaseModel):
    """Top-level pipeline configuration passed in an AnalysisRequest."""

    ner: NERConfig = Field(default_factory=NERConfig, description="NER stage settings")
    icd: ICDConfig = Field(default_factory=ICDConfig, description="ICD-10 prediction stage settings")
    summary: SummaryConfig = Field(default_factory=SummaryConfig, description="Summarization stage settings")
    risk: RiskConfig = Field(default_factory=RiskConfig, description="Risk scoring stage settings")

    model_config = {
        "json_schema_extra": {
            "example": {
                "ner": {"enabled": True, "model": "rule-based", "min_confidence": 0.5},
                "icd": {"enabled": True, "model": "sklearn-baseline", "top_k": 5},
                "summary": {"enabled": True, "model": "extractive", "detail_level": "standard"},
                "risk": {"enabled": True, "category_weights": None},
            }
        }
    }


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class AnalysisRequest(BaseModel):
    """Request body for the full clinical NLP pipeline."""

    text: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_TEXT_LENGTH,
            description="Clinical text to process through the full pipeline (up to 100 000 characters)",
        ),
    ]
    config: PipelineConfig = Field(
        default_factory=PipelineConfig,
        description=(
            "Per-stage pipeline configuration. "
            "Omit entirely to run all stages with default settings."
        ),
    )
    document_id: str | None = Field(
        default=None,
        max_length=36,
        description=(
            "Optional client-supplied document identifier (e.g. a UUID or EMR record ID). "
            "Included verbatim in the response for correlation."
        ),
    )
    store_result: bool = Field(
        default=False,
        description=(
            "When True and the request is authenticated, persist the analysis result "
            "to the database and return a result_id."
        ),
    )

    @model_validator(mode="after")
    def at_least_one_stage_enabled(self) -> "AnalysisRequest":
        cfg = self.config
        if not any([cfg.ner.enabled, cfg.icd.enabled, cfg.summary.enabled, cfg.risk.enabled]):
            raise ValueError("At least one pipeline stage must be enabled")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": (
                    "CHIEF COMPLAINT: Chest pain.\n\n"
                    "HPI: 72-year-old male with known coronary artery disease presents with "
                    "acute onset substernal chest pain radiating to the jaw, onset 2 hours ago. "
                    "History of hypertension, hyperlipidaemia, and type 2 diabetes. "
                    "Current medications: atorvastatin 40 mg, metoprolol 50 mg BID, "
                    "metformin 1000 mg BID, aspirin 81 mg.\n\n"
                    "ASSESSMENT: STEMI. PLAN: Activate cath lab, heparin bolus, clopidogrel load."
                ),
                "config": {
                    "ner": {"enabled": True, "model": "rule-based", "min_confidence": 0.5},
                    "icd": {"enabled": True, "model": "sklearn-baseline", "top_k": 5},
                    "summary": {"enabled": True, "model": "extractive", "detail_level": "brief"},
                    "risk": {"enabled": True},
                },
                "document_id": "pat-20260324-001",
                "store_result": False,
            }
        }
    }


# ---------------------------------------------------------------------------
# Response atoms
# ---------------------------------------------------------------------------


class RiskSummary(BaseModel):
    """Compact risk summary embedded in AnalysisResponse."""

    score: float = Field(ge=0.0, le=1.0, description="Overall risk score in [0, 1]")
    category: str = Field(description="Qualitative risk level: low | moderate | high | critical")
    top_factors: list[RiskFactorResponse] = Field(
        description="Top 3 contributing risk factors"
    )
    recommendations: list[str] = Field(description="Up to 5 actionable clinical recommendations")

    model_config = {
        "json_schema_extra": {
            "example": {
                "score": 0.78,
                "category": "high",
                "top_factors": [],
                "recommendations": ["Urgent follow-up within 48-72 hours recommended"],
            }
        }
    }


class StageTiming(BaseModel):
    """Per-stage processing latencies."""

    ner_ms: float | None = Field(
        default=None,
        ge=0,
        description="NER stage latency in ms (null if skipped)",
    )
    icd_ms: float | None = Field(
        default=None,
        ge=0,
        description="ICD stage latency in ms (null if skipped)",
    )
    summary_ms: float | None = Field(
        default=None,
        ge=0,
        description="Summary stage latency in ms (null if skipped)",
    )
    risk_ms: float | None = Field(
        default=None,
        ge=0,
        description="Risk stage latency in ms (null if skipped)",
    )
    total_ms: float = Field(ge=0, description="Total end-to-end pipeline latency in ms")


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------


class AnalysisResponse(BaseModel):
    """Top-level full pipeline analysis response."""

    document_id: str | None = Field(
        default=None,
        description="Echo of the client-supplied document_id from the request",
    )
    result_id: str | None = Field(
        default=None,
        description="Server-assigned UUID for the persisted result (only set when store_result=True)",
    )
    text_length: int = Field(ge=0, description="Number of characters in the input text")

    # Stage results (null when stage was disabled)
    entities: list[EntityResponse] | None = Field(
        default=None,
        description="Named entities extracted by the NER stage. Null if NER was disabled.",
    )
    icd_codes: list[ICDCodeResponse] | None = Field(
        default=None,
        description="Predicted ICD-10 codes. Null if the ICD stage was disabled.",
    )
    summary: SummarizationResponse | None = Field(
        default=None,
        description="Clinical text summary. Null if the summarization stage was disabled.",
    )
    risk_score: RiskSummary | None = Field(
        default=None,
        description="Risk assessment result. Null if the risk scoring stage was disabled.",
    )

    timing: StageTiming = Field(description="Per-stage and total latency breakdown")

    model_config = {
        "json_schema_extra": {
            "example": {
                "document_id": "pat-20260324-001",
                "result_id": None,
                "text_length": 485,
                "entities": [
                    {
                        "text": "metoprolol",
                        "entity_type": "MEDICATION",
                        "start_char": 210,
                        "end_char": 220,
                        "confidence": 0.9,
                        "normalized_text": None,
                        "umls_cui": None,
                        "is_negated": False,
                        "is_uncertain": False,
                        "metadata": None,
                    }
                ],
                "icd_codes": [
                    {
                        "code": "I21.9",
                        "description": "Acute myocardial infarction, unspecified",
                        "confidence": 0.88,
                        "chapter": "Diseases of the circulatory system",
                        "category": None,
                        "contributing_text": None,
                    }
                ],
                "summary": {
                    "summary": "72-year-old male with CAD presents with STEMI. Activated cath lab.",
                    "key_points": ["STEMI", "Coronary artery disease", "Hypertension"],
                    "original_word_count": 85,
                    "summary_word_count": 14,
                    "compression_ratio": 6.07,
                    "summary_type": "extractive",
                    "model_name": "textrank",
                    "model_version": "1.0.0",
                    "processing_time_ms": 19.1,
                },
                "risk_score": {
                    "score": 0.78,
                    "category": "high",
                    "top_factors": [],
                    "recommendations": ["Urgent follow-up within 48-72 hours recommended"],
                },
                "timing": {
                    "ner_ms": 11.2,
                    "icd_ms": 39.5,
                    "summary_ms": 19.1,
                    "risk_ms": 8.3,
                    "total_ms": 78.1,
                },
            }
        }
    }
