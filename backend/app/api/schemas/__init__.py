"""Public re-exports for all ClinIQ API schemas.

Import from this module rather than individual sub-modules to keep endpoint
code stable if the internal file structure ever changes.

    from app.api.schemas import AnalysisRequest, AnalysisResponse
"""

# -- Analysis (full pipeline) --------------------------------------------------
from app.api.schemas.analysis import (
                                      AnalysisRequest,
                                      AnalysisResponse,
                                      ICDConfig,
                                      NERConfig,
                                      PipelineConfig,
                                      RiskConfig,
                                      RiskSummary,
                                      StageTiming,
                                      SummaryConfig,
)

# -- Auth & users --------------------------------------------------------------
from app.api.schemas.auth import (
                                      APIKeyCreate,
                                      APIKeyResponse,
                                      RefreshTokenRequest,
                                      TokenRequest,
                                      TokenResponse,
                                      UserCreate,
                                      UserResponse,
                                      UserUpdate,
)

# -- Batch processing ----------------------------------------------------------
from app.api.schemas.batch import (
                                      BatchDocument,
                                      BatchDocumentResult,
                                      BatchPipelineConfig,
                                      BatchRequest,
                                      BatchStatusResponse,
                                      BatchSubmitResponse,
)

# -- Common / shared -----------------------------------------------------------
from app.api.schemas.common import (
                                      ErrorDetail,
                                      ErrorResponse,
                                      HealthResponse,
                                      ModelInfo,
                                      PaginatedResponse,
                                      PaginationMeta,
                                      SuccessResponse,
)

# -- ICD-10 prediction ---------------------------------------------------------
from app.api.schemas.icd import ICDCodeResponse, ICDPredictionRequest, ICDPredictionResponse

# -- NER -----------------------------------------------------------------------
from app.api.schemas.ner import EntityResponse, NERRequest, NERResponse

# -- Risk scoring --------------------------------------------------------------
from app.api.schemas.risk import RiskFactorResponse, RiskScoreRequest, RiskScoreResponse

# -- Summarization -------------------------------------------------------------
from app.api.schemas.summary import SummarizationRequest, SummarizationResponse

__all__ = [
    # analysis
    "AnalysisRequest",
    "AnalysisResponse",
    "NERConfig",
    "ICDConfig",
    "SummaryConfig",
    "RiskConfig",
    "PipelineConfig",
    "RiskSummary",
    "StageTiming",
    # auth
    "TokenRequest",
    "TokenResponse",
    "RefreshTokenRequest",
    "UserCreate",
    "UserUpdate",
    "UserResponse",
    "APIKeyCreate",
    "APIKeyResponse",
    # batch
    "BatchDocument",
    "BatchPipelineConfig",
    "BatchRequest",
    "BatchSubmitResponse",
    "BatchStatusResponse",
    "BatchDocumentResult",
    # common
    "ErrorDetail",
    "ErrorResponse",
    "SuccessResponse",
    "PaginationMeta",
    "PaginatedResponse",
    "ModelInfo",
    "HealthResponse",
    # icd
    "ICDPredictionRequest",
    "ICDCodeResponse",
    "ICDPredictionResponse",
    # ner
    "NERRequest",
    "EntityResponse",
    "NERResponse",
    # risk
    "RiskScoreRequest",
    "RiskFactorResponse",
    "RiskScoreResponse",
    # summary
    "SummarizationRequest",
    "SummarizationResponse",
]
