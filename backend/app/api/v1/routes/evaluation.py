"""Evaluation and benchmarking endpoints.

Exposes the ClinIQ evaluation framework as a REST API, enabling
consumers to submit gold-standard annotations alongside predictions
and receive structured metric reports.  Useful for:

* **Model selection** — Compare candidate models on the same evaluation set
* **Regression testing** — Assert that pipeline updates don't degrade quality
* **Clinical validation** — Generate metric reports for regulatory review
* **Annotation QA** — Compute inter-annotator agreement (Cohen's Kappa)

Design decisions
----------------
* **Stateless evaluation** — All data is provided per-request; no
  server-side state is retained.  This simplifies scaling and avoids
  data retention concerns in HIPAA-regulated environments.
* **Batch-oriented** — Each endpoint accepts arrays of predictions
  and labels in a single request, reducing HTTP overhead for large
  evaluation sets.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.ml.utils.advanced_metrics import (
    compute_auprc,
    compute_calibration,
    compute_cohens_kappa,
    compute_hierarchical_icd_metrics,
    compute_mcc,
    compute_partial_ner_metrics,
    compute_rouge,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["evaluation"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class ClassificationEvalRequest(BaseModel):
    """Request body for binary classification evaluation."""

    y_true: list[int] = Field(
        ..., min_length=1, max_length=100_000,
        description="Ground truth binary labels (0 or 1).",
    )
    y_pred: list[int] = Field(
        ..., min_length=1, max_length=100_000,
        description="Predicted binary labels (0 or 1).",
    )
    y_prob: list[float] | None = Field(
        None,
        description="Predicted probabilities for calibration (optional).",
    )
    n_calibration_bins: int = Field(
        10, ge=1, le=100,
        description="Number of bins for calibration ECE.",
    )


class ClassificationEvalResponse(BaseModel):
    """Classification evaluation results."""

    mcc: float
    tp: int
    fp: int
    fn: int
    tn: int
    calibration: dict[str, Any] | None = None
    processing_time_ms: float


class KappaRequest(BaseModel):
    """Request body for inter-annotator agreement."""

    rater_a: list[str | int] = Field(
        ..., min_length=1, max_length=100_000,
        description="Labels from annotator A.",
    )
    rater_b: list[str | int] = Field(
        ..., min_length=1, max_length=100_000,
        description="Labels from annotator B.",
    )


class KappaResponse(BaseModel):
    """Cohen's Kappa agreement results."""

    kappa: float
    observed_agreement: float
    expected_agreement: float
    n_items: int
    processing_time_ms: float


class EntitySpan(BaseModel):
    """A single entity span annotation."""

    entity_type: str = Field(..., description="Entity type label.")
    start: int = Field(..., ge=0, description="Start character offset.")
    end: int = Field(..., ge=0, description="End character offset.")


class NEREvalRequest(BaseModel):
    """Request body for NER evaluation with partial matching."""

    gold_entities: list[EntitySpan] = Field(
        ..., max_length=10_000,
        description="Gold-standard entity annotations.",
    )
    pred_entities: list[EntitySpan] = Field(
        ..., max_length=10_000,
        description="Predicted entity annotations.",
    )
    overlap_threshold: float = Field(
        0.5, ge=0.0, le=1.0,
        description="Minimum Jaccard overlap for partial match credit.",
    )


class NEREvalResponse(BaseModel):
    """NER evaluation results with exact and partial metrics."""

    exact_f1: float
    partial_f1: float
    type_weighted_f1: float
    mean_overlap: float
    n_gold: int
    n_pred: int
    n_exact_matches: int
    n_partial_matches: int
    n_unmatched_pred: int
    n_unmatched_gold: int
    processing_time_ms: float


class ROUGERequest(BaseModel):
    """Request body for summarisation evaluation."""

    reference: str = Field(
        ..., min_length=1, max_length=500_000,
        description="Gold-standard reference summary.",
    )
    hypothesis: str = Field(
        ..., min_length=1, max_length=500_000,
        description="Model-generated summary.",
    )


class ROUGEResponse(BaseModel):
    """Full ROUGE evaluation results."""

    rouge1: dict[str, float]
    rouge2: dict[str, float]
    rougeL: dict[str, float]
    reference_length: int
    hypothesis_length: int
    length_ratio: float
    processing_time_ms: float


class ICDEvalRequest(BaseModel):
    """Request body for hierarchical ICD-10 evaluation."""

    gold_codes: list[str] = Field(
        ..., min_length=1, max_length=100_000,
        description="Ground truth ICD-10-CM codes.",
    )
    pred_codes: list[str] = Field(
        ..., min_length=1, max_length=100_000,
        description="Predicted ICD-10-CM codes.",
    )


class ICDEvalResponse(BaseModel):
    """Hierarchical ICD-10 evaluation results."""

    full_code_accuracy: float
    block_accuracy: float
    chapter_accuracy: float
    n_samples: int
    full_code_matches: int
    block_matches: int
    chapter_matches: int
    processing_time_ms: float


class AUPRCRequest(BaseModel):
    """Request body for AUPRC computation."""

    y_true: list[int] = Field(
        ..., min_length=1, max_length=100_000,
        description="Binary ground truth labels (0 or 1).",
    )
    y_scores: list[float] = Field(
        ..., min_length=1, max_length=100_000,
        description="Predicted scores/probabilities.",
    )
    label: str = Field(
        "positive",
        description="Human-readable label for the positive class.",
    )


class AUPRCResponse(BaseModel):
    """AUPRC evaluation result."""

    label: str
    auprc: float
    n_positive: int
    n_total: int
    processing_time_ms: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/evaluate/classification",
    response_model=ClassificationEvalResponse,
    summary="Evaluate binary classification predictions",
    description=(
        "Compute Matthews Correlation Coefficient (MCC), confusion matrix, "
        "and optional calibration metrics (ECE, Brier score) for binary "
        "classification predictions."
    ),
)
async def evaluate_classification(request: ClassificationEvalRequest) -> ClassificationEvalResponse:
    """Evaluate binary classification with MCC and calibration."""
    start = time.perf_counter()

    if len(request.y_true) != len(request.y_pred):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"y_true and y_pred must have equal length "
                   f"({len(request.y_true)} != {len(request.y_pred)})",
        )

    mcc_result = compute_mcc(request.y_true, request.y_pred)

    calibration = None
    if request.y_prob is not None:
        if len(request.y_prob) != len(request.y_true):
            from fastapi import HTTPException
            raise HTTPException(
                status_code=400,
                detail="y_prob must have same length as y_true",
            )
        cal = compute_calibration(
            request.y_true,
            request.y_prob,
            n_bins=request.n_calibration_bins,
        )
        calibration = {
            "expected_calibration_error": cal.expected_calibration_error,
            "brier_score": cal.brier_score,
            "n_bins": cal.n_bins,
            "bin_accuracies": cal.bin_accuracies,
            "bin_confidences": cal.bin_confidences,
            "bin_counts": cal.bin_counts,
        }

    elapsed = (time.perf_counter() - start) * 1000

    return ClassificationEvalResponse(
        mcc=mcc_result.mcc,
        tp=mcc_result.tp,
        fp=mcc_result.fp,
        fn=mcc_result.fn,
        tn=mcc_result.tn,
        calibration=calibration,
        processing_time_ms=round(elapsed, 2),
    )


@router.post(
    "/evaluate/agreement",
    response_model=KappaResponse,
    summary="Compute inter-annotator agreement (Cohen's Kappa)",
    description=(
        "Compute Cohen's Kappa coefficient between two annotators. "
        "Essential for validating annotation quality in clinical NLP "
        "datasets before model training."
    ),
)
async def evaluate_agreement(request: KappaRequest) -> KappaResponse:
    """Compute inter-annotator agreement."""
    start = time.perf_counter()

    if len(request.rater_a) != len(request.rater_b):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Rater sequences must have equal length "
                   f"({len(request.rater_a)} != {len(request.rater_b)})",
        )

    result = compute_cohens_kappa(request.rater_a, request.rater_b)
    elapsed = (time.perf_counter() - start) * 1000

    return KappaResponse(
        kappa=result.kappa,
        observed_agreement=result.observed_agreement,
        expected_agreement=result.expected_agreement,
        n_items=result.n_items,
        processing_time_ms=round(elapsed, 2),
    )


@router.post(
    "/evaluate/ner",
    response_model=NEREvalResponse,
    summary="Evaluate NER with partial span matching",
    description=(
        "Compute exact and partial entity span-matching metrics. "
        "Partial matching awards Jaccard overlap credit, revealing "
        "how close predictions are to correct boundaries."
    ),
)
async def evaluate_ner(request: NEREvalRequest) -> NEREvalResponse:
    """Evaluate NER predictions with partial span credit."""
    start = time.perf_counter()

    gold = [(e.entity_type, e.start, e.end) for e in request.gold_entities]
    pred = [(e.entity_type, e.start, e.end) for e in request.pred_entities]

    result = compute_partial_ner_metrics(
        gold, pred, overlap_threshold=request.overlap_threshold,
    )
    elapsed = (time.perf_counter() - start) * 1000

    return NEREvalResponse(
        exact_f1=result.exact_f1,
        partial_f1=result.partial_f1,
        type_weighted_f1=result.type_weighted_f1,
        mean_overlap=result.mean_overlap,
        n_gold=result.n_gold,
        n_pred=result.n_pred,
        n_exact_matches=result.n_exact_matches,
        n_partial_matches=result.n_partial_matches,
        n_unmatched_pred=result.n_unmatched_pred,
        n_unmatched_gold=result.n_unmatched_gold,
        processing_time_ms=round(elapsed, 2),
    )


@router.post(
    "/evaluate/rouge",
    response_model=ROUGEResponse,
    summary="Evaluate summarisation with full ROUGE",
    description=(
        "Compute ROUGE-1/2/L with full precision, recall, and F1 "
        "(not just recall). Includes length ratio for compression analysis."
    ),
)
async def evaluate_rouge(request: ROUGERequest) -> ROUGEResponse:
    """Evaluate summarisation quality."""
    start = time.perf_counter()

    result = compute_rouge(request.reference, request.hypothesis)
    elapsed = (time.perf_counter() - start) * 1000

    return ROUGEResponse(
        rouge1={"precision": result.rouge1.precision, "recall": result.rouge1.recall, "f1": result.rouge1.f1},
        rouge2={"precision": result.rouge2.precision, "recall": result.rouge2.recall, "f1": result.rouge2.f1},
        rougeL={"precision": result.rougeL.precision, "recall": result.rougeL.recall, "f1": result.rougeL.f1},
        reference_length=result.reference_length,
        hypothesis_length=result.hypothesis_length,
        length_ratio=result.length_ratio,
        processing_time_ms=round(elapsed, 2),
    )


@router.post(
    "/evaluate/icd",
    response_model=ICDEvalResponse,
    summary="Evaluate ICD-10 predictions hierarchically",
    description=(
        "Evaluate ICD-10-CM code predictions at three hierarchy levels: "
        "full code, 3-character block, and chapter. Reveals whether "
        "prediction errors are catastrophic or merely specificity mismatches."
    ),
)
async def evaluate_icd(request: ICDEvalRequest) -> ICDEvalResponse:
    """Evaluate ICD-10 predictions at multiple hierarchy levels."""
    start = time.perf_counter()

    if len(request.gold_codes) != len(request.pred_codes):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail=f"Code lists must have equal length "
                   f"({len(request.gold_codes)} != {len(request.pred_codes)})",
        )

    result = compute_hierarchical_icd_metrics(request.gold_codes, request.pred_codes)
    elapsed = (time.perf_counter() - start) * 1000

    return ICDEvalResponse(
        full_code_accuracy=result.full_code_accuracy,
        block_accuracy=result.block_accuracy,
        chapter_accuracy=result.chapter_accuracy,
        n_samples=result.n_samples,
        full_code_matches=result.full_code_matches,
        block_matches=result.block_matches,
        chapter_matches=result.chapter_matches,
        processing_time_ms=round(elapsed, 2),
    )


@router.post(
    "/evaluate/auprc",
    response_model=AUPRCResponse,
    summary="Compute Area Under Precision-Recall Curve",
    description=(
        "Compute AUPRC for binary classification. More informative than "
        "AUROC for imbalanced clinical datasets where negatives dominate."
    ),
)
async def evaluate_auprc(request: AUPRCRequest) -> AUPRCResponse:
    """Compute AUPRC for binary predictions."""
    start = time.perf_counter()

    if len(request.y_true) != len(request.y_scores):
        from fastapi import HTTPException
        raise HTTPException(
            status_code=400,
            detail="y_true and y_scores must have equal length",
        )

    result = compute_auprc(request.y_true, request.y_scores, label=request.label)
    elapsed = (time.perf_counter() - start) * 1000

    return AUPRCResponse(
        label=result.label,
        auprc=result.auprc,
        n_positive=result.n_positive,
        n_total=result.n_total,
        processing_time_ms=round(elapsed, 2),
    )


@router.get(
    "/evaluate/metrics",
    summary="List available evaluation metrics",
    description="Catalogue of evaluation metrics supported by the API.",
)
async def list_evaluation_metrics() -> dict[str, Any]:
    """Return the catalogue of available evaluation metrics."""
    return {
        "metrics": [
            {
                "name": "classification",
                "endpoint": "/evaluate/classification",
                "description": "MCC, confusion matrix, calibration (ECE, Brier score)",
                "use_case": "Binary classification model evaluation",
            },
            {
                "name": "agreement",
                "endpoint": "/evaluate/agreement",
                "description": "Cohen's Kappa inter-annotator agreement",
                "use_case": "Annotation quality validation",
            },
            {
                "name": "ner",
                "endpoint": "/evaluate/ner",
                "description": "Exact and partial span-matching NER metrics",
                "use_case": "Entity extraction evaluation with boundary credit",
            },
            {
                "name": "rouge",
                "endpoint": "/evaluate/rouge",
                "description": "ROUGE-1/2/L with full precision/recall/F1",
                "use_case": "Clinical summarisation quality assessment",
            },
            {
                "name": "icd",
                "endpoint": "/evaluate/icd",
                "description": "Hierarchical ICD-10 accuracy (chapter/block/full)",
                "use_case": "ICD-10-CM coding evaluation",
            },
            {
                "name": "auprc",
                "endpoint": "/evaluate/auprc",
                "description": "Area Under Precision-Recall Curve",
                "use_case": "Imbalanced binary classification evaluation",
            },
        ],
    }
