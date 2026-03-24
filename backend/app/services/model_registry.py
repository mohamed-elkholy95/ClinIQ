"""Singleton model registry for lazy-loading and caching ML models.

Provides a central place to initialise, load, and access ML model instances
so that API route handlers can call real inference instead of mock stubs.

Design decisions:
    - **Lazy loading** — Models are only loaded on first access, keeping
      application startup fast (important for container health checks).
    - **Double-checked locking** — The outer ``if`` avoids acquiring the lock
      on the hot path (model already loaded), while the inner ``if`` inside the
      lock prevents duplicate initialisation when two threads race on the first
      call.  This pattern is safe in CPython (GIL guarantees reference
      assignment is atomic), and the ``threading.Lock`` makes it portable.
    - **Thread-safety** — Uses a module-level ``threading.Lock``.  For the
      async FastAPI worker model where true parallelism is limited to threads
      in the threadpool executor, this is sufficient.  If we needed
      multi-process safety (e.g. gunicorn preforked workers), each worker
      would get its own copy — which is the desired behaviour for ML models.
    - **Separation from routes** — Keeping model lifecycle here rather than in
      route modules means routes stay thin and testable (just mock the registry
      function).

Usage from route handlers::

    from app.services.model_registry import get_ner_model, get_icd_model

    ner_model = get_ner_model()
    entities  = ner_model.extract_entities(text)
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.ml.icd.model import BaseICDClassifier
    from app.ml.ner.model import BaseNERModel
    from app.ml.risk.model import BaseRiskScorer
    from app.ml.summarization.model import BaseSummarizer

logger = logging.getLogger(__name__)

_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Internal state — populated lazily
# ---------------------------------------------------------------------------

_ner_model: BaseNERModel | None = None
_icd_model: BaseICDClassifier | None = None
_summarizer: BaseSummarizer | None = None
_risk_scorer: BaseRiskScorer | None = None


# ---------------------------------------------------------------------------
# Public accessors
# ---------------------------------------------------------------------------


def get_ner_model() -> BaseNERModel:
    """Return the cached NER model, loading on first call.

    Returns
    -------
    BaseNERModel
        A ready-to-use NER model (rule-based by default).
    """
    global _ner_model  # noqa: PLW0603
    if _ner_model is None:
        with _lock:
            if _ner_model is None:
                from app.ml.ner.model import RuleBasedNERModel

                logger.info("Initialising rule-based NER model…")
                model = RuleBasedNERModel()
                model.load()
                _ner_model = model
                logger.info("NER model ready.")
    return _ner_model


def get_icd_model() -> BaseICDClassifier:
    """Return the cached ICD-10 classifier, loading on first call.

    Returns
    -------
    BaseICDClassifier
        A ready-to-use ICD-10 classifier (rule-based by default).
    """
    global _icd_model  # noqa: PLW0603
    if _icd_model is None:
        with _lock:
            if _icd_model is None:
                from app.ml.icd.model import RuleBasedICDClassifier

                logger.info("Initialising rule-based ICD-10 classifier…")
                model = RuleBasedICDClassifier()
                model.load()
                _icd_model = model
                logger.info("ICD-10 classifier ready.")
    return _icd_model


def get_summarizer() -> BaseSummarizer:
    """Return the cached summarization model, loading on first call.

    Returns
    -------
    BaseSummarizer
        A ready-to-use clinical text summarizer.
    """
    global _summarizer  # noqa: PLW0603
    if _summarizer is None:
        with _lock:
            if _summarizer is None:
                from app.ml.summarization.model import ExtractiveSummarizer

                logger.info("Initialising extractive summarizer…")
                model = ExtractiveSummarizer()
                model.load()
                _summarizer = model
                logger.info("Summarizer ready.")
    return _summarizer


def get_risk_scorer() -> BaseRiskScorer:
    """Return the cached risk scoring model, loading on first call.

    Returns
    -------
    BaseRiskScorer
        A ready-to-use clinical risk scorer.
    """
    global _risk_scorer  # noqa: PLW0603
    if _risk_scorer is None:
        with _lock:
            if _risk_scorer is None:
                from app.ml.risk.model import RuleBasedRiskScorer

                logger.info("Initialising rule-based risk scorer…")
                model = RuleBasedRiskScorer()
                model.load()
                _risk_scorer = model
                logger.info("Risk scorer ready.")
    return _risk_scorer


def health_check() -> dict[str, bool]:
    """Return the load status of each model without triggering lazy loading.

    Useful for health/readiness endpoints that need to report which models
    are available without incurring the cost of loading them.

    Returns
    -------
    dict[str, bool]
        Mapping of model name to whether it is currently loaded and cached.
    """
    return {
        "ner": _ner_model is not None and _ner_model.is_loaded,
        "icd": _icd_model is not None and _icd_model.is_loaded,
        "summarizer": _summarizer is not None and _summarizer.is_loaded,
        "risk_scorer": _risk_scorer is not None and _risk_scorer.is_loaded,
    }


def reset_all() -> None:
    """Reset all cached models — useful in tests.

    Not thread-safe with concurrent inference; intended only for test
    setup/teardown.
    """
    global _ner_model, _icd_model, _summarizer, _risk_scorer  # noqa: PLW0603
    _ner_model = None
    _icd_model = None
    _summarizer = None
    _risk_scorer = None
    logger.info("Model registry reset.")
