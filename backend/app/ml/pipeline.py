"""ML pipeline orchestrator for the ClinIQ platform.

Coordinates NER, ICD-10 classification, summarization, risk scoring, and
dental analysis into a single :class:`ClinicalPipeline` that supports both
single-document and batch processing.  Each component is optional and loaded
lazily; failures in individual components produce partial results rather than
aborting the whole pipeline.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from app.ml.dental.model import DentalAssessment, DentalNERModel, PeriodontalRiskAssessor
from app.ml.icd.model import BaseICDClassifier, ICDPredictionResult
from app.ml.ner.model import BaseNERModel, Entity
from app.ml.risk.model import BaseRiskScorer, RiskAssessment
from app.ml.summarization.model import BaseSummarizer, SummarizationResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration & result data classes
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """Per-invocation configuration for :class:`ClinicalPipeline`.

    Attributes
    ----------
    enable_ner:
        Run the NER component.
    enable_icd:
        Run the ICD-10 classification component.
    enable_summarization:
        Run the summarization component.
    enable_risk:
        Run the risk scoring component.
    enable_dental:
        Run the dental NLP component.
    confidence_threshold:
        Minimum confidence for NER entities to be included in results.
    top_k_icd:
        Number of top ICD-10 predictions to return.
    detail_level:
        Summarization detail level (``"brief"``, ``"standard"``, ``"detailed"``).
    """

    enable_ner: bool = True
    enable_icd: bool = True
    enable_summarization: bool = True
    enable_risk: bool = True
    enable_dental: bool = False
    confidence_threshold: float = 0.5
    top_k_icd: int = 10
    detail_level: Literal["brief", "standard", "detailed"] = "standard"


@dataclass
class PipelineResult:
    """Complete result produced by :class:`ClinicalPipeline.process`.

    Attributes
    ----------
    document_id:
        Caller-supplied document identifier (may be ``None``).
    entities:
        NER-extracted :class:`~app.ml.ner.model.Entity` objects filtered by
        ``confidence_threshold``.
    icd_predictions:
        Top-k ICD-10 predictions as serialisable dicts.
    summary:
        :class:`~app.ml.summarization.model.SummarizationResult`, or ``None``
        if summarization was disabled or failed.
    risk_assessment:
        :class:`~app.ml.risk.model.RiskAssessment`, or ``None``.
    dental_assessment:
        :class:`~app.ml.dental.model.DentalAssessment`, or ``None``.
    processing_time_ms:
        Total wall-clock time for the pipeline call.
    model_versions:
        Mapping of component name to version string.
    component_errors:
        Mapping of component name to error message for any components that
        failed (partial-result mode).
    """

    document_id: str | None
    entities: list[Entity] = field(default_factory=list)
    icd_predictions: list[dict[str, Any]] = field(default_factory=list)
    summary: SummarizationResult | None = None
    risk_assessment: RiskAssessment | None = None
    dental_assessment: DentalAssessment | None = None
    processing_time_ms: float = 0.0
    model_versions: dict[str, str] = field(default_factory=dict)
    component_errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "document_id": self.document_id,
            "entities": [e.to_dict() for e in self.entities],
            "icd_predictions": self.icd_predictions,
            "summary": self.summary.to_dict() if self.summary else None,
            "risk_assessment": self.risk_assessment.to_dict()
            if self.risk_assessment
            else None,
            "dental_assessment": self.dental_assessment.to_dict()
            if self.dental_assessment
            else None,
            "processing_time_ms": self.processing_time_ms,
            "model_versions": self.model_versions,
            "component_errors": self.component_errors,
        }


# ---------------------------------------------------------------------------
# ClinicalPipeline
# ---------------------------------------------------------------------------


class ClinicalPipeline:
    """Orchestrates all ClinIQ ML models for end-to-end clinical text analysis.

    Models are injected via the constructor and loaded lazily on first use.
    Any component may be ``None``; in that case the corresponding output field
    in :class:`PipelineResult` remains ``None``.

    Parameters
    ----------
    ner_model:
        A :class:`~app.ml.ner.model.BaseNERModel` instance.
    icd_classifier:
        A :class:`~app.ml.icd.model.BaseICDClassifier` instance.
    summarizer:
        A :class:`~app.ml.summarization.model.BaseSummarizer` instance.
    risk_scorer:
        A :class:`~app.ml.risk.model.BaseRiskScorer` instance.
    dental_model:
        A :class:`~app.ml.dental.model.DentalNERModel` instance.  When
        provided together with a :class:`~app.ml.dental.model.PeriodontalRiskAssessor`
        the full dental assessment is produced.

    Example
    -------
    >>> pipeline = ClinicalPipeline(
    ...     ner_model=RuleBasedNERModel(),
    ...     risk_scorer=RuleBasedRiskScorer(),
    ...     summarizer=ExtractiveSummarizer(),
    ... )
    >>> result = pipeline.process(text, PipelineConfig(enable_dental=False))
    """

    def __init__(
        self,
        ner_model: BaseNERModel | None = None,
        icd_classifier: BaseICDClassifier | None = None,
        summarizer: BaseSummarizer | None = None,
        risk_scorer: BaseRiskScorer | None = None,
        dental_model: DentalNERModel | None = None,
        perio_assessor: PeriodontalRiskAssessor | None = None,
    ) -> None:
        self._ner_model = ner_model
        self._icd_classifier = icd_classifier
        self._summarizer = summarizer
        self._risk_scorer = risk_scorer
        self._dental_model = dental_model
        self._perio_assessor = perio_assessor
        self._is_loaded: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Eagerly load all non-None components."""
        logger.info("Loading ClinicalPipeline components …")
        start_time = time.time()
        errors: list[str] = []

        for name, component in self._components():
            try:
                component.ensure_loaded()
                logger.debug("Loaded component: %s", name)
            except Exception as exc:
                logger.error("Failed to load component '%s': %s", name, exc)
                errors.append(f"{name}: {exc}")

        self._is_loaded = True
        elapsed = (time.time() - start_time) * 1000
        logger.info("ClinicalPipeline loaded in %.1f ms (errors: %d)", elapsed, len(errors))
        if errors:
            logger.warning("Component load errors: %s", "; ".join(errors))

    @property
    def is_loaded(self) -> bool:
        """``True`` after :meth:`load` has been called."""
        return self._is_loaded

    def ensure_loaded(self) -> None:
        """Load if not yet loaded."""
        if not self._is_loaded:
            self.load()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(
        self,
        text: str,
        config: PipelineConfig | None = None,
        document_id: str | None = None,
    ) -> PipelineResult:
        """Run the full pipeline on *text*.

        Each enabled component is executed independently.  Failures are
        captured in :attr:`PipelineResult.component_errors` and do not
        abort the remaining components.

        Parameters
        ----------
        text:
            Raw clinical document text.
        config:
            Pipeline configuration; defaults to :class:`PipelineConfig`.
        document_id:
            Optional caller-supplied identifier included in the result.

        Returns
        -------
        PipelineResult
        """
        self.ensure_loaded()
        cfg = config or PipelineConfig()
        pipeline_start = time.time()

        result = PipelineResult(document_id=document_id)

        # --- 1. NER ---
        if cfg.enable_ner and self._ner_model is not None:
            result = self._run_ner(text, cfg, result)

        # --- 2. ICD-10 ---
        if cfg.enable_icd and self._icd_classifier is not None:
            result = self._run_icd(text, cfg, result)

        # --- 3. Summarization ---
        if cfg.enable_summarization and self._summarizer is not None:
            result = self._run_summarization(text, cfg, result)

        # --- 4. Risk scoring (can use NER and ICD outputs) ---
        if cfg.enable_risk and self._risk_scorer is not None:
            result = self._run_risk(text, cfg, result)

        # --- 5. Dental analysis ---
        if cfg.enable_dental and self._dental_model is not None:
            result = self._run_dental(text, cfg, result)

        # Populate model version metadata
        result.model_versions = self._collect_model_versions()
        result.processing_time_ms = (time.time() - pipeline_start) * 1000

        logger.debug(
            "ClinicalPipeline.process completed in %.1f ms (document_id=%s)",
            result.processing_time_ms,
            document_id,
        )
        return result

    def process_batch(
        self,
        texts: list[str],
        config: PipelineConfig | None = None,
        document_ids: list[str | None] | None = None,
    ) -> list[PipelineResult]:
        """Process a list of documents sequentially.

        Parameters
        ----------
        texts:
            List of raw clinical document texts.
        config:
            Shared pipeline configuration for all documents.
        document_ids:
            Optional list of identifiers aligned with *texts*.

        Returns
        -------
        list[PipelineResult]
        """
        self.ensure_loaded()
        doc_ids: list[str | None] = document_ids or [None] * len(texts)
        results: list[PipelineResult] = []

        for text, doc_id in zip(texts, doc_ids, strict=False):
            result = self.process(text, config=config, document_id=doc_id)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Component runners (each returns a (possibly updated) PipelineResult)
    # ------------------------------------------------------------------

    def _run_ner(
        self, text: str, cfg: PipelineConfig, result: PipelineResult
    ) -> PipelineResult:
        try:
            self._ner_model.ensure_loaded()  # type: ignore[union-attr]
            entities = self._ner_model.extract_entities(text)  # type: ignore[union-attr]
            # Filter by confidence threshold
            result.entities = [
                e for e in entities if e.confidence >= cfg.confidence_threshold
            ]
            logger.debug("NER extracted %d entities", len(result.entities))
        except Exception as exc:
            logger.error("NER component failed: %s", exc)
            result.component_errors["ner"] = str(exc)
        return result

    def _run_icd(
        self, text: str, cfg: PipelineConfig, result: PipelineResult
    ) -> PipelineResult:
        try:
            self._icd_classifier.ensure_loaded()  # type: ignore[union-attr]
            icd_result: ICDPredictionResult = self._icd_classifier.predict(  # type: ignore[union-attr]
                text, top_k=cfg.top_k_icd
            )
            result.icd_predictions = [p.to_dict() for p in icd_result.predictions]
            logger.debug("ICD-10 produced %d predictions", len(result.icd_predictions))
        except Exception as exc:
            logger.error("ICD component failed: %s", exc)
            result.component_errors["icd"] = str(exc)
        return result

    def _run_summarization(
        self, text: str, cfg: PipelineConfig, result: PipelineResult
    ) -> PipelineResult:
        try:
            self._summarizer.ensure_loaded()  # type: ignore[union-attr]
            result.summary = self._summarizer.summarize(  # type: ignore[union-attr]
                text, detail_level=cfg.detail_level
            )
            logger.debug("Summarization produced %d-char summary", len(result.summary.summary))
        except Exception as exc:
            logger.error("Summarization component failed: %s", exc)
            result.component_errors["summarization"] = str(exc)
        return result

    def _run_risk(
        self, text: str, cfg: PipelineConfig, result: PipelineResult
    ) -> PipelineResult:
        try:
            self._risk_scorer.ensure_loaded()  # type: ignore[union-attr]
            icd_codes = [
                p.get("code") for p in result.icd_predictions if p.get("code")
            ]
            result.risk_assessment = self._risk_scorer.assess_risk(  # type: ignore[union-attr]
                text,
                entities=result.entities or None,
                icd_codes=icd_codes or None,
            )
            logger.debug(
                "Risk assessment: %.1f (%s)",
                result.risk_assessment.overall_score,
                result.risk_assessment.risk_level,
            )
        except Exception as exc:
            logger.error("Risk scoring component failed: %s", exc)
            result.component_errors["risk"] = str(exc)
        return result

    def _run_dental(
        self, text: str, cfg: PipelineConfig, result: PipelineResult
    ) -> PipelineResult:
        try:
            self._dental_model.ensure_loaded()  # type: ignore[union-attr]
            dental_entities = self._dental_model.extract_entities(text)  # type: ignore[union-attr]

            # Periodontal risk (if assessor is available)
            perio_data: dict[str, Any] = {}
            perio_score = 0.0
            perio_classification = "Unknown"
            recommendations: list[str] = []
            if self._perio_assessor is not None:
                self._perio_assessor.ensure_loaded()
                perio_data = self._perio_assessor.assess(text, entities=dental_entities)
                perio_score = perio_data.get("risk_score", 0.0)
                perio_classification = perio_data.get("classification", "Unknown")
                recommendations = perio_data.get("recommendations", [])

            # Collect CDT codes from entities.
            # Build a reverse lookup (code → description) from the dental
            # model's class-level CDT_CODES dict which maps procedure names
            # to {"code": ..., "description": ...} dicts.
            code_to_desc: dict[str, str] = {}
            if hasattr(self._dental_model, "CDT_CODES"):
                for _proc, info in self._dental_model.CDT_CODES.items():
                    code_to_desc[info["code"]] = info["description"]

            suggested_cdt: dict[str, str] = {}
            for ent in dental_entities:
                cdt = ent.metadata.get("cdt_code") if ent.metadata else None
                if cdt and cdt in code_to_desc:
                    suggested_cdt[cdt] = code_to_desc[cdt]

            result.dental_assessment = DentalAssessment(
                entities=dental_entities,
                periodontal_risk_score=perio_score,
                periodontal_classification=perio_classification,
                cdt_codes=suggested_cdt,
                recommendations=recommendations,
                processing_time_ms=perio_data.get("processing_time_ms", 0.0),
                model_name=self._dental_model.model_name,
                model_version=self._dental_model.version,
            )
            logger.debug(
                "Dental NER extracted %d entities; perio score=%.1f",
                len(dental_entities),
                perio_score,
            )
        except Exception as exc:
            logger.error("Dental component failed: %s", exc)
            result.component_errors["dental"] = str(exc)
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _components(self) -> list[tuple[str, Any]]:
        """Return (name, component) for all non-None components that have
        ``ensure_loaded``."""
        pairs: list[tuple[str, Any]] = []
        if self._ner_model is not None:
            pairs.append(("ner", self._ner_model))
        if self._icd_classifier is not None:
            pairs.append(("icd", self._icd_classifier))
        if self._summarizer is not None:
            pairs.append(("summarizer", self._summarizer))
        if self._risk_scorer is not None:
            pairs.append(("risk", self._risk_scorer))
        if self._dental_model is not None:
            pairs.append(("dental", self._dental_model))
        if self._perio_assessor is not None:
            pairs.append(("perio", self._perio_assessor))
        return pairs

    def _collect_model_versions(self) -> dict[str, str]:
        """Collect version strings from all loaded components."""
        versions: dict[str, str] = {}
        component_map: list[tuple[str, Any]] = [
            ("ner", self._ner_model),
            ("icd", self._icd_classifier),
            ("summarizer", self._summarizer),
            ("risk", self._risk_scorer),
            ("dental", self._dental_model),
            ("perio", self._perio_assessor),
        ]
        for name, component in component_map:
            if component is not None and hasattr(component, "version"):
                versions[name] = component.version
        return versions
