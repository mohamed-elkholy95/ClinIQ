"""Unified ML pipeline orchestrating all models."""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.exceptions import InferenceError, ModelLoadError
from app.ml.icd.model import BaseICDClassifier, ICDPredictionResult, TransformerICDClassifier
from app.ml.ner.model import BaseNERModel, CompositeNERModel, Entity, RuleBasedNERModel
from app.ml.risk.scorer import RiskScore, RiskScorer
from app.ml.summarization.model import (
    BaseSummarizer,
    ExtractiveSummarizer,
    HybridSummarizer,
    SummaryResult,
)

logger = logging.getLogger(__name__)
settings = get_settings()


@dataclass
class PipelineConfig:
    """Configuration for the ML pipeline."""

    enable_ner: bool = True
    enable_icd: bool = True
    enable_summarization: bool = True
    enable_risk: bool = True

    # Model settings
    ner_model: str = "composite"
    icd_model: str = "transformer"
    summarizer_model: str = "hybrid"

    # Output settings
    max_icd_codes: int = 10
    summary_max_length: int = 150
    include_confidence: bool = True

    # Performance settings
    parallel_inference: bool = True
    cache_results: bool = True


@dataclass
class AnalysisResult:
    """Complete analysis result from the pipeline."""

    document_id: str | None = None
    text_hash: str = ""

    # Component results
    entities: list[Entity] = field(default_factory=list)
    icd_predictions: list[dict[str, Any]] = field(default_factory=list)
    summary: SummaryResult | None = None
    risk_score: RiskScore | None = None

    # Metadata
    processing_time_ms: float = 0.0
    component_times_ms: dict[str, float] = field(default_factory=dict)
    model_versions: dict[str, str] = field(default_factory=dict)
    pipeline_config: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "document_id": self.document_id,
            "text_hash": self.text_hash,
            "entities": [e.to_dict() for e in self.entities],
            "icd_predictions": self.icd_predictions,
            "summary": self.summary.to_dict() if self.summary else None,
            "risk_score": self.risk_score.to_dict() if self.risk_score else None,
            "processing_time_ms": self.processing_time_ms,
            "component_times_ms": self.component_times_ms,
            "model_versions": self.model_versions,
            "pipeline_config": self.pipeline_config,
        }


class MLPipeline:
    """Unified ML pipeline for clinical text analysis."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()

        # Initialize models (lazy loading)
        self._ner_model: BaseNERModel | None = None
        self._icd_model: BaseICDClassifier | None = None
        self._summarizer: BaseSummarizer | None = None
        self._risk_scorer: RiskScorer | None = None

        self._is_loaded = False

    def load(self) -> None:
        """Load all enabled models."""
        logger.info("Loading ML pipeline models...")
        start_time = time.time()

        try:
            if self.config.enable_ner:
                self._load_ner_model()

            if self.config.enable_icd:
                self._load_icd_model()

            if self.config.enable_summarization:
                self._load_summarizer()

            if self.config.enable_risk:
                self._load_risk_scorer()

            self._is_loaded = True
            load_time = time.time() - start_time
            logger.info(f"ML pipeline loaded in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load ML pipeline: {e}")
            raise ModelLoadError("pipeline", str(e))

    def _load_ner_model(self) -> None:
        """Load NER model."""
        if self.config.ner_model == "composite":
            # Composite model with rule-based and transformer
            rule_based = RuleBasedNERModel()
            self._ner_model = CompositeNERModel(
                models=[rule_based],
                voting="union",
            )
        else:
            self._ner_model = RuleBasedNERModel()

        self._ner_model.load()
        logger.info("Loaded NER model")

    def _load_icd_model(self) -> None:
        """Load ICD-10 classifier."""
        if self.config.icd_model == "transformer":
            self._icd_model = TransformerICDClassifier(
                device="cpu",  # Use "cuda" if GPU available
            )
        else:
            self._icd_model = TransformerICDClassifier()

        self._icd_model.load()
        logger.info("Loaded ICD-10 classifier")

    def _load_summarizer(self) -> None:
        """Load summarizer."""
        if self.config.summarizer_model == "hybrid":
            self._summarizer = HybridSummarizer()
        else:
            self._summarizer = ExtractiveSummarizer()

        self._summarizer.load()
        logger.info("Loaded summarizer")

    def _load_risk_scorer(self) -> None:
        """Load risk scorer."""
        self._risk_scorer = RiskScorer()
        logger.info("Loaded risk scorer")

    def is_loaded(self) -> bool:
        """Check if pipeline is loaded."""
        return self._is_loaded

    def analyze(
        self,
        text: str,
        document_id: str | None = None,
        config_override: PipelineConfig | None = None,
    ) -> AnalysisResult:
        """Analyze clinical text through the full pipeline."""
        config = config_override or self.config

        if not self._is_loaded:
            self.load()

        start_time = time.time()
        result = AnalysisResult(
            document_id=document_id,
            pipeline_config={
                "ner": config.enable_ner,
                "icd": config.enable_icd,
                "summarization": config.enable_summarization,
                "risk": config.enable_risk,
            },
        )

        # Compute text hash
        import hashlib

        result.text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]

        try:
            if config.parallel_inference:
                # Run components in parallel where possible
                self._analyze_parallel(text, result, config)
            else:
                # Run sequentially
                self._analyze_sequential(text, result, config)

        except Exception as e:
            logger.error(f"Pipeline analysis failed: {e}")
            raise InferenceError("pipeline", str(e))

        result.processing_time_ms = (time.time() - start_time) * 1000

        # Record model versions
        if self._ner_model:
            result.model_versions["ner"] = self._ner_model.version
        if self._icd_model:
            result.model_versions["icd"] = self._icd_model.version
        if self._summarizer:
            result.model_versions["summarizer"] = self._summarizer.version
        if self._risk_scorer:
            result.model_versions["risk"] = self._risk_scorer.version

        return result

    def _analyze_sequential(
        self,
        text: str,
        result: AnalysisResult,
        config: PipelineConfig,
    ) -> None:
        """Run analysis sequentially."""
        # NER
        if config.enable_ner and self._ner_model:
            start = time.time()
            result.entities = self._ner_model.extract_entities(text)
            result.component_times_ms["ner"] = (time.time() - start) * 1000

        # ICD-10
        if config.enable_icd and self._icd_model:
            start = time.time()
            icd_result = self._icd_model.predict(text, top_k=config.max_icd_codes)
            result.icd_predictions = [p.to_dict() for p in icd_result.predictions]
            result.component_times_ms["icd"] = icd_result.processing_time_ms

        # Summarization
        if config.enable_summarization and self._summarizer:
            start = time.time()
            result.summary = self._summarizer.summarize(
                text,
                max_length=config.summary_max_length,
            )
            result.component_times_ms["summarization"] = result.summary.processing_time_ms

        # Risk scoring (depends on NER and ICD)
        if config.enable_risk and self._risk_scorer:
            start = time.time()
            result.risk_score = self._risk_scorer.calculate_risk(
                text,
                entities=result.entities,
                icd_predictions=result.icd_predictions,
            )
            result.component_times_ms["risk"] = result.risk_score.processing_time_ms

    def _analyze_parallel(
        self,
        text: str,
        result: AnalysisResult,
        config: PipelineConfig,
    ) -> None:
        """Run independent components in parallel using asyncio."""
        import asyncio

        async def run_async():
            tasks = []

            # NER, ICD, and Summarization are independent
            if config.enable_ner and self._ner_model:
                tasks.append(self._run_ner_async(text))
            else:
                tasks.append(asyncio.sleep(0, result=None))

            if config.enable_icd and self._icd_model:
                tasks.append(self._run_icd_async(text, config.max_icd_codes))
            else:
                tasks.append(asyncio.sleep(0, result=None))

            if config.enable_summarization and self._summarizer:
                tasks.append(self._run_summarizer_async(text, config.summary_max_length))
            else:
                tasks.append(asyncio.sleep(0, result=None))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process NER result
            if results[0] and not isinstance(results[0], Exception):
                result.entities, result.component_times_ms["ner"] = results[0]

            # Process ICD result
            if results[1] and not isinstance(results[1], Exception):
                icd_result = results[1]
                result.icd_predictions = [p.to_dict() for p in icd_result.predictions]
                result.component_times_ms["icd"] = icd_result.processing_time_ms

            # Process summarization result
            if results[2] and not isinstance(results[2], Exception):
                result.summary, result.component_times_ms["summarization"] = results[2]

            # Risk scoring depends on above results
            if config.enable_risk and self._risk_scorer:
                start = time.time()
                result.risk_score = self._risk_scorer.calculate_risk(
                    text,
                    entities=result.entities,
                    icd_predictions=result.icd_predictions,
                )
                result.component_times_ms["risk"] = (time.time() - start) * 1000

        asyncio.run(run_async())

    async def _run_ner_async(self, text: str) -> tuple[list[Entity], float]:
        """Run NER in async context."""
        import asyncio

        loop = asyncio.get_event_loop()
        start = time.time()
        entities = await loop.run_in_executor(None, self._ner_model.extract_entities, text)
        elapsed = (time.time() - start) * 1000
        return entities, elapsed

    async def _run_icd_async(self, text: str, top_k: int) -> ICDPredictionResult:
        """Run ICD prediction in async context."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._icd_model.predict, text, top_k)

    async def _run_summarizer_async(
        self, text: str, max_length: int
    ) -> tuple[SummaryResult, float]:
        """Run summarization in async context."""
        import asyncio

        loop = asyncio.get_event_loop()
        start = time.time()
        summary = await loop.run_in_executor(
            None, self._summarizer.summarize, text, max_length
        )
        elapsed = (time.time() - start) * 1000
        return summary, elapsed

    def analyze_batch(
        self,
        texts: list[str],
        document_ids: list[str] | None = None,
    ) -> list[AnalysisResult]:
        """Analyze multiple documents."""
        results = []
        doc_ids = document_ids or [None] * len(texts)

        for text, doc_id in zip(texts, doc_ids):
            result = self.analyze(text, document_id=doc_id)
            results.append(result)

        return results

    # Convenience methods for single-component analysis

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract entities only."""
        if not self._is_loaded:
            self.load()
        if self._ner_model:
            return self._ner_model.extract_entities(text)
        return []

    def predict_icd_codes(self, text: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Predict ICD-10 codes only."""
        if not self._is_loaded:
            self.load()
        if self._icd_model:
            result = self._icd_model.predict(text, top_k=top_k)
            return [p.to_dict() for p in result.predictions]
        return []

    def summarize(self, text: str, max_length: int = 150) -> SummaryResult | None:
        """Generate summary only."""
        if not self._is_loaded:
            self.load()
        if self._summarizer:
            return self._summarizer.summarize(text, max_length=max_length)
        return None

    def calculate_risk(self, text: str) -> RiskScore | None:
        """Calculate risk score only."""
        if not self._is_loaded:
            self.load()
        if self._risk_scorer:
            return self._risk_scorer.calculate_risk(text)
        return None


# Singleton instance for the application
_pipeline_instance: MLPipeline | None = None


def get_pipeline() -> MLPipeline:
    """Get or create the ML pipeline singleton."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = MLPipeline()
    return _pipeline_instance
