"""Public re-exports for all API schemas."""

from app.api.schemas.analysis import AnalysisRequest, AnalysisResponse, PipelineConfig
from app.api.schemas.auth import (
    APIKeyCreate,
    APIKeyResponse,
    TokenRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
)
from app.api.schemas.batch import BatchDocumentItem, BatchPipelineConfig, BatchRequest, BatchStatusResponse
from app.api.schemas.common import (
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
    ModelInfo,
    PaginatedResponse,
    PaginationMeta,
    SuccessResponse,
)
from app.api.schemas.icd import ICDCodeResponse, ICDPredictionRequest, ICDPredictionResponse
from app.api.schemas.ner import EntityResponse, NERRequest, NERResponse
from app.api.schemas.risk import RiskDomainScore, RiskScoreRequest, RiskScoreResponse
from app.api.schemas.summary import SummarizationRequest, SummarizationResponse

__all__ = [
    # common
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
    "ModelInfo",
    "PaginatedResponse",
    "PaginationMeta",
    "SuccessResponse",
    # analysis
    "AnalysisRequest",
    "AnalysisResponse",
    "PipelineConfig",
    # ner
    "EntityResponse",
    "NERRequest",
    "NERResponse",
    # icd
    "ICDCodeResponse",
    "ICDPredictionRequest",
    "ICDPredictionResponse",
    # summary
    "SummarizationRequest",
    "SummarizationResponse",
    # risk
    "RiskDomainScore",
    "RiskScoreRequest",
    "RiskScoreResponse",
    # batch
    "BatchDocumentItem",
    "BatchPipelineConfig",
    "BatchRequest",
    "BatchStatusResponse",
    # auth
    "APIKeyCreate",
    "APIKeyResponse",
    "TokenRequest",
    "TokenResponse",
    "UserCreate",
    "UserResponse",
]
