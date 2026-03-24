"""Document analysis service for the ClinIQ platform.

Provides a high-level async interface for analysing clinical text —
both single-document and batch — by delegating to the
:class:`~app.ml.pipeline.ClinicalPipeline`.  The service handles
pipeline lifecycle (lazy loading), error translation, and logging.
"""

from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import InferenceError
from app.ml.pipeline import ClinicalPipeline, PipelineConfig, PipelineResult
from app.ml.utils.text_preprocessing import preprocess_clinical_text

logger = logging.getLogger(__name__)
settings = get_settings()


def _text_hash(text: str) -> str:
    """Return a hex SHA-256 digest of *text* for deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AnalysisService:
    """Orchestrates clinical text analysis through the ML pipeline.

    The service lazily initialises the underlying
    :class:`ClinicalPipeline` on first use so that application startup
    remains fast.

    Parameters
    ----------
    pipeline : ClinicalPipeline | None
        An optional pre-built pipeline instance.  When *None* a new
        default pipeline is created on first analysis call.
    """

    def __init__(self, pipeline: ClinicalPipeline | None = None) -> None:
        self._pipeline = pipeline
        self._is_loaded = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_pipeline(self) -> ClinicalPipeline:
        """Lazily create and load the pipeline."""
        if self._pipeline is None:
            self._pipeline = ClinicalPipeline()
        if not self._is_loaded:
            self._pipeline.load()
            self._is_loaded = True
            logger.info("ML pipeline loaded successfully")
        return self._pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def analyze(
        self,
        text: str,
        document_id: str | None = None,
        config_override: PipelineConfig | None = None,
    ) -> dict[str, Any]:
        """Analyse a single clinical text document.

        Parameters
        ----------
        text:
            Raw clinical text to analyse.
        document_id:
            Optional external identifier to attach to the result.
        config_override:
            Optional per-request pipeline configuration.

        Returns
        -------
        dict[str, Any]
            Analysis result dictionary containing entities, ICD codes,
            summary, risk score, and processing metadata.

        Raises
        ------
        InferenceError
            If the underlying pipeline raises an exception.
        """
        pipeline = self._ensure_pipeline()

        start = time.perf_counter()
        try:
            cleaned = preprocess_clinical_text(text)
            result: PipelineResult = pipeline.process(cleaned, config_override)
            processing_ms = (time.perf_counter() - start) * 1_000

            logger.info(
                "Document analysed",
                extra={
                    "document_id": document_id,
                    "processing_ms": round(processing_ms, 2),
                    "text_hash": _text_hash(text),
                },
            )

            return {
                "document_id": document_id,
                "text_hash": _text_hash(text),
                "result": result,
                "processing_ms": round(processing_ms, 2),
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }
        except InferenceError:
            raise
        except Exception as exc:
            logger.error("Analysis failed: %s", exc, exc_info=True)
            raise InferenceError("pipeline", str(exc)) from exc

    async def batch_analyze(
        self,
        texts: list[str],
        config_override: PipelineConfig | None = None,
    ) -> list[dict[str, Any]]:
        """Analyse multiple clinical text documents sequentially.

        Parameters
        ----------
        texts:
            List of raw clinical text strings.
        config_override:
            Optional per-request pipeline configuration applied to
            every document in the batch.

        Returns
        -------
        list[dict[str, Any]]
            One result dictionary per input text, in the same order.

        Raises
        ------
        InferenceError
            If any individual analysis fails the whole batch is aborted.
        """
        if not texts:
            return []

        pipeline = self._ensure_pipeline()  # noqa: F841 — ensures loaded

        results: list[dict[str, Any]] = []
        for idx, text in enumerate(texts):
            try:
                result = await self.analyze(
                    text=text,
                    document_id=f"batch-{idx}",
                    config_override=config_override,
                )
                results.append(result)
            except Exception as exc:
                logger.error(
                    "Batch analysis failed at index %d: %s", idx, exc
                )
                raise InferenceError("pipeline", str(exc)) from exc

        logger.info("Batch analysis complete: %d documents", len(results))
        return results
