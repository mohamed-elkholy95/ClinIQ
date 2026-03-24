"""Pydantic models for API requests and responses."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ==================== Shared Models ====================


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    error_code: str
    details: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    version: str
    models_loaded: dict[str, bool]
    uptime_seconds: float


# ==================== Entity Models ====================


class EntityBase(BaseModel):
    """Base entity model."""

    text: str
    entity_type: str
    start_char: int
    end_char: int
    confidence: float
    normalized_text: str | None = None
    umls_cui: str | None = None
    is_negated: bool = False
    is_uncertain: bool = False


class EntityResponse(EntityBase):
    """Entity in response."""

    pass


# ==================== ICD-10 Models ====================


class ICDCodePrediction(BaseModel):
    """ICD-10 code prediction."""

    code: str
    description: str | None = None
    confidence: float
    chapter: str | None = None
    category: str | None = None
    contributing_text: list[str] | None = None


# ==================== Summarization Models ====================


class SummaryResponse(BaseModel):
    """Summarization result."""

    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    summary_type: str
    key_points: list[str] | None = None


# ==================== Risk Models ====================


class RiskFactor(BaseModel):
    """Risk factor model."""

    name: str
    description: str
    weight: float
    value: float
    source: str
    evidence: str | None = None


class RiskScoreResponse(BaseModel):
    """Risk score response."""

    overall_score: float
    risk_level: str
    category_scores: dict[str, float]
    risk_factors: list[RiskFactor]
    protective_factors: list[RiskFactor]
    recommendations: list[str]


# ==================== Request Models ====================


class AnalyzeRequest(BaseModel):
    """Request for full analysis."""

    text: str = Field(..., min_length=10, max_length=100000)
    document_id: str | None = None
    enable_ner: bool = True
    enable_icd: bool = True
    enable_summarization: bool = True
    enable_risk: bool = True
    max_icd_codes: int = Field(10, ge=1, le=50)
    summary_max_length: int = Field(150, ge=50, le=500)


class NERRequest(BaseModel):
    """Request for NER only."""

    text: str = Field(..., min_length=10, max_length=100000)


class ICDRequest(BaseModel):
    """Request for ICD-10 prediction."""

    text: str = Field(..., min_length=10, max_length=100000)
    top_k: int = Field(10, ge=1, le=50)


class SummarizeRequest(BaseModel):
    """Request for summarization."""

    text: str = Field(..., min_length=10, max_length=100000)
    max_length: int = Field(150, ge=50, le=500)
    min_length: int = Field(30, ge=10, le=100)


class RiskRequest(BaseModel):
    """Request for risk scoring."""

    text: str = Field(..., min_length=10, max_length=100000)


class BatchRequest(BaseModel):
    """Request for batch processing."""

    documents: list[str] = Field(..., min_length=1, max_length=100)
    enable_ner: bool = True
    enable_icd: bool = True
    enable_summarization: bool = False
    enable_risk: bool = True


# ==================== Response Models ====================


class AnalysisResponse(BaseModel):
    """Full analysis response."""

    document_id: str | None = None
    text_hash: str

    entities: list[EntityResponse] = []
    icd_predictions: list[ICDCodePrediction] = []
    summary: SummaryResponse | None = None
    risk_score: RiskScoreResponse | None = None

    processing_time_ms: float
    component_times_ms: dict[str, float] = {}
    model_versions: dict[str, str] = {}


class NERResponse(BaseModel):
    """NER-only response."""

    entities: list[EntityResponse]
    entity_count: int
    entity_type_counts: dict[str, int]
    processing_time_ms: float
    model_version: str


class ICDResponse(BaseModel):
    """ICD-10 prediction response."""

    predictions: list[ICDCodePrediction]
    processing_time_ms: float
    model_version: str


class SummarizeResponse(BaseModel):
    """Summarization response."""

    summary: SummaryResponse
    processing_time_ms: float
    model_version: str


class RiskResponse(BaseModel):
    """Risk scoring response."""

    risk_score: RiskScoreResponse
    processing_time_ms: float
    model_version: str


class BatchJobResponse(BaseModel):
    """Batch job creation response."""

    job_id: str
    status: str
    total_documents: int
    message: str


class BatchStatusResponse(BaseModel):
    """Batch job status response."""

    job_id: str
    status: str
    total_documents: int
    processed_documents: int
    failed_documents: int
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result_url: str | None = None
    error_message: str | None = None


# ==================== Auth Models ====================


class TokenRequest(BaseModel):
    """Token request."""

    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int


class APIKeyCreate(BaseModel):
    """API key creation request."""

    name: str = Field(..., min_length=1, max_length=100)
    rate_limit: int = Field(100, ge=10, le=10000)
    expires_days: int | None = Field(None, ge=1, le=365)


class APIKeyResponse(BaseModel):
    """API key response."""

    id: str
    name: str
    key: str  # Only shown once on creation
    rate_limit: int
    expires_at: datetime | None
    created_at: datetime


# ==================== User Models ====================


class UserCreate(BaseModel):
    """User creation request."""

    email: str = Field(..., pattern=r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
    password: str = Field(..., min_length=8, max_length=100)
    full_name: str | None = Field(None, max_length=255)


class UserResponse(BaseModel):
    """User response."""

    id: str
    email: str
    full_name: str | None
    is_active: bool
    role: str
    created_at: datetime


# ==================== Model Management ====================


class ModelInfo(BaseModel):
    """Model information."""

    name: str
    version: str
    stage: str
    is_loaded: bool
    metrics: dict[str, float] | None = None
    deployed_at: datetime | None = None


class ModelsListResponse(BaseModel):
    """List of available models."""

    models: list[ModelInfo]
    default_versions: dict[str, str]
