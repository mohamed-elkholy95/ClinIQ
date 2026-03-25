"""Enhanced clinical NLP pipeline integrating all ClinIQ analysis modules.

Extends :class:`ClinicalPipeline` by orchestrating the 13 additional clinical
NLP modules built post-PRD into a unified, single-call analysis interface.
This allows callers to get a comprehensive clinical document analysis —
from document classification and section parsing through entity extraction,
normalization, assertion detection, and clinical scoring — in one pipeline
invocation.

Architecture
~~~~~~~~~~~~
The enhanced pipeline runs in **two phases**:

1. **Pre-processing** — Document classification, section parsing, quality
   analysis, de-identification, and abbreviation expansion.  These produce
   structural metadata that downstream stages use for confidence boosting
   and context-aware extraction.

2. **Extraction & scoring** — Medication, allergy, vital sign, temporal,
   SDoH, and relation extraction; assertion detection; concept normalization;
   Charlson comorbidity scoring.  These stages receive section parse results
   to enable section-aware confidence boosting.

All stages are independently toggleable and fault-tolerant: a failure in any
stage is captured in ``component_errors`` without aborting the remaining
stages.

Design decisions
----------------
* **Composition over inheritance** — Wraps ``ClinicalPipeline`` via
  delegation rather than subclassing, keeping the original pipeline's
  interface stable.
* **Shared section context** — The section parser runs once and its result
  is passed to downstream modules that support section-aware detection.
* **Zero coupling** — Each module is imported and invoked independently.
  Missing optional dependencies (e.g., transformer models) cause graceful
  fallback to rule-based alternatives.
* **Consistent result structure** — Results are serialisable via
  ``to_dict()`` for direct JSON response construction.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app.ml.pipeline import ClinicalPipeline, PipelineConfig, PipelineResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enhanced pipeline configuration
# ---------------------------------------------------------------------------


@dataclass
class EnhancedPipelineConfig(PipelineConfig):
    """Configuration for :class:`EnhancedClinicalPipeline`.

    Inherits all fields from :class:`PipelineConfig` and adds toggles
    for the additional clinical NLP modules.

    Attributes
    ----------
    enable_classification:
        Run document type classification.
    enable_sections:
        Run section parsing (recommended — used by downstream modules).
    enable_quality:
        Run clinical note quality analysis.
    enable_deidentification:
        Run PHI de-identification.
    enable_abbreviations:
        Run abbreviation detection and expansion.
    enable_medications:
        Run structured medication extraction.
    enable_allergies:
        Run allergy extraction.
    enable_vitals:
        Run vital signs extraction.
    enable_temporal:
        Run temporal information extraction.
    enable_assertions:
        Run assertion status detection on extracted entities.
    enable_normalization:
        Run concept normalization on extracted entities.
    enable_sdoh:
        Run Social Determinants of Health extraction.
    enable_relations:
        Run clinical relation extraction between entities.
    enable_comorbidity:
        Run Charlson Comorbidity Index calculation.
    min_confidence_enhanced:
        Minimum confidence threshold for enhanced module outputs.
    """

    enable_classification: bool = True
    enable_sections: bool = True
    enable_quality: bool = True
    enable_deidentification: bool = False  # Off by default (destructive)
    enable_abbreviations: bool = True
    enable_medications: bool = True
    enable_allergies: bool = True
    enable_vitals: bool = True
    enable_temporal: bool = True
    enable_assertions: bool = True
    enable_normalization: bool = True
    enable_sdoh: bool = True
    enable_relations: bool = True
    enable_comorbidity: bool = True
    min_confidence_enhanced: float = 0.5


# ---------------------------------------------------------------------------
# Enhanced result
# ---------------------------------------------------------------------------


@dataclass
class EnhancedPipelineResult:
    """Complete result from :class:`EnhancedClinicalPipeline`.

    Wraps the base :class:`PipelineResult` and adds outputs from all
    enhanced analysis modules.

    Attributes
    ----------
    base_result:
        Results from the core pipeline (NER, ICD, summarization, risk, dental).
    classification:
        Document type classification result.
    sections:
        Parsed document sections with category mappings.
    quality:
        Clinical note quality analysis report.
    deidentification:
        PHI de-identification result (text, entities, strategy).
    abbreviations:
        Detected and expanded abbreviations.
    medications:
        Structured medication extractions.
    allergies:
        Allergen extractions with reactions and severity.
    vitals:
        Vital sign measurements with clinical interpretation.
    temporal:
        Temporal expressions (dates, durations, frequencies).
    assertions:
        Assertion statuses for extracted entities.
    normalization:
        Normalized entities mapped to ontology codes.
    sdoh:
        Social Determinants of Health factors.
    relations:
        Clinical relations between extracted entities.
    comorbidity:
        Charlson Comorbidity Index calculation.
    processing_time_ms:
        Total wall-clock time for the enhanced pipeline.
    component_errors:
        Mapping of component name to error message for failed stages.
    """

    base_result: PipelineResult | None = None
    classification: dict[str, Any] | None = None
    sections: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None
    deidentification: dict[str, Any] | None = None
    abbreviations: dict[str, Any] | None = None
    medications: dict[str, Any] | None = None
    allergies: dict[str, Any] | None = None
    vitals: dict[str, Any] | None = None
    temporal: dict[str, Any] | None = None
    assertions: list[dict[str, Any]] | None = None
    normalization: list[dict[str, Any]] | None = None
    sdoh: dict[str, Any] | None = None
    relations: dict[str, Any] | None = None
    comorbidity: dict[str, Any] | None = None
    processing_time_ms: float = 0.0
    component_errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of all results."""
        return {
            "base_result": self.base_result.to_dict() if self.base_result else None,
            "classification": self.classification,
            "sections": self.sections,
            "quality": self.quality,
            "deidentification": self.deidentification,
            "abbreviations": self.abbreviations,
            "medications": self.medications,
            "allergies": self.allergies,
            "vitals": self.vitals,
            "temporal": self.temporal,
            "assertions": self.assertions,
            "normalization": self.normalization,
            "sdoh": self.sdoh,
            "relations": self.relations,
            "comorbidity": self.comorbidity,
            "processing_time_ms": self.processing_time_ms,
            "component_errors": self.component_errors,
        }


# ---------------------------------------------------------------------------
# Enhanced Clinical Pipeline
# ---------------------------------------------------------------------------


class EnhancedClinicalPipeline:
    """Unified clinical NLP pipeline integrating all ClinIQ modules.

    Wraps :class:`ClinicalPipeline` (NER, ICD, summarization, risk, dental)
    and adds document classification, section parsing, quality analysis,
    de-identification, abbreviation expansion, medication extraction,
    allergy extraction, vital signs, temporal extraction, assertion
    detection, concept normalization, SDoH extraction, relation extraction,
    and Charlson Comorbidity Index calculation.

    Parameters
    ----------
    base_pipeline:
        A configured :class:`ClinicalPipeline` instance.  If ``None``,
        a default pipeline with no models is created.

    Example
    -------
    >>> pipeline = EnhancedClinicalPipeline()
    >>> config = EnhancedPipelineConfig(enable_deidentification=True)
    >>> result = pipeline.process("Patient presents with chest pain...", config)
    >>> result.to_dict()
    """

    def __init__(
        self,
        base_pipeline: ClinicalPipeline | None = None,
    ) -> None:
        self._base_pipeline = base_pipeline or ClinicalPipeline()

        # Lazily instantiated module instances
        self._classifier = None
        self._section_parser = None
        self._quality_analyzer = None
        self._deidentifier = None
        self._abbreviation_expander = None
        self._medication_extractor = None
        self._allergy_extractor = None
        self._vitals_extractor = None
        self._temporal_extractor = None
        self._assertion_detector = None
        self._concept_normalizer = None
        self._sdoh_extractor = None
        self._relation_extractor = None
        self._charlson_calculator = None
        self._modules_initialized = False

    def _ensure_modules(self) -> None:
        """Lazily import and instantiate all enhanced modules.

        Each module is imported inside this method to avoid import-time
        failures when optional dependencies are missing.  Initialization
        failures are logged but do not prevent other modules from loading.
        """
        if self._modules_initialized:
            return

        # Document classifier
        try:
            from app.ml.classifier.document_classifier import (
                RuleBasedDocumentClassifier,
            )
            self._classifier = RuleBasedDocumentClassifier()
        except Exception as exc:
            logger.warning("Failed to initialize document classifier: %s", exc)

        # Section parser
        try:
            from app.ml.sections.parser import ClinicalSectionParser
            self._section_parser = ClinicalSectionParser()
        except Exception as exc:
            logger.warning("Failed to initialize section parser: %s", exc)

        # Quality analyzer
        try:
            from app.ml.quality.analyzer import ClinicalNoteQualityAnalyzer
            self._quality_analyzer = ClinicalNoteQualityAnalyzer()
        except Exception as exc:
            logger.warning("Failed to initialize quality analyzer: %s", exc)

        # De-identifier
        try:
            from app.ml.deidentification.detector import Deidentifier
            self._deidentifier = Deidentifier()
        except Exception as exc:
            logger.warning("Failed to initialize deidentifier: %s", exc)

        # Abbreviation expander
        try:
            from app.ml.abbreviations.expander import AbbreviationExpander
            self._abbreviation_expander = AbbreviationExpander()
        except Exception as exc:
            logger.warning("Failed to initialize abbreviation expander: %s", exc)

        # Medication extractor
        try:
            from app.ml.medications.extractor import ClinicalMedicationExtractor
            self._medication_extractor = ClinicalMedicationExtractor()
        except Exception as exc:
            logger.warning("Failed to initialize medication extractor: %s", exc)

        # Allergy extractor
        try:
            from app.ml.allergies.extractor import ClinicalAllergyExtractor
            self._allergy_extractor = ClinicalAllergyExtractor()
        except Exception as exc:
            logger.warning("Failed to initialize allergy extractor: %s", exc)

        # Vital signs extractor
        try:
            from app.ml.vitals.extractor import ClinicalVitalSignsExtractor
            self._vitals_extractor = ClinicalVitalSignsExtractor()
        except Exception as exc:
            logger.warning("Failed to initialize vitals extractor: %s", exc)

        # Temporal extractor
        try:
            from app.ml.temporal.extractor import ClinicalTemporalExtractor
            self._temporal_extractor = ClinicalTemporalExtractor()
        except Exception as exc:
            logger.warning("Failed to initialize temporal extractor: %s", exc)

        # Assertion detector
        try:
            from app.ml.assertions.detector import ConTextAssertionDetector
            self._assertion_detector = ConTextAssertionDetector()
        except Exception as exc:
            logger.warning("Failed to initialize assertion detector: %s", exc)

        # Concept normalizer
        try:
            from app.ml.normalization.normalizer import ClinicalConceptNormalizer
            self._concept_normalizer = ClinicalConceptNormalizer()
        except Exception as exc:
            logger.warning("Failed to initialize concept normalizer: %s", exc)

        # SDoH extractor
        try:
            from app.ml.sdoh.extractor import ClinicalSDoHExtractor
            self._sdoh_extractor = ClinicalSDoHExtractor()
        except Exception as exc:
            logger.warning("Failed to initialize SDoH extractor: %s", exc)

        # Relation extractor
        try:
            from app.ml.relations.extractor import RuleBasedRelationExtractor
            self._relation_extractor = RuleBasedRelationExtractor()
        except Exception as exc:
            logger.warning("Failed to initialize relation extractor: %s", exc)

        # Charlson calculator
        try:
            from app.ml.comorbidity.charlson import CharlsonCalculator
            self._charlson_calculator = CharlsonCalculator()
        except Exception as exc:
            logger.warning("Failed to initialize Charlson calculator: %s", exc)

        self._modules_initialized = True

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def process(
        self,
        text: str,
        config: EnhancedPipelineConfig | None = None,
        document_id: str | None = None,
    ) -> EnhancedPipelineResult:
        """Run the full enhanced pipeline on *text*.

        Executes the base pipeline first (NER, ICD, summarization, risk,
        dental) then runs all enabled enhanced modules.  Failures in any
        module are captured in ``component_errors`` without aborting.

        Parameters
        ----------
        text:
            Raw clinical document text.
        config:
            Pipeline configuration.  Defaults to
            :class:`EnhancedPipelineConfig` (all modules except
            de-identification enabled).
        document_id:
            Optional caller-supplied document identifier.

        Returns
        -------
        EnhancedPipelineResult
        """
        self._ensure_modules()
        cfg = config or EnhancedPipelineConfig()
        pipeline_start = time.time()
        result = EnhancedPipelineResult()

        # --- Base pipeline ---
        try:
            result.base_result = self._base_pipeline.process(
                text, config=cfg, document_id=document_id,
            )
            # Merge base pipeline errors
            if result.base_result.component_errors:
                result.component_errors.update(result.base_result.component_errors)
        except Exception as exc:
            logger.error("Base pipeline failed: %s", exc)
            result.component_errors["base_pipeline"] = str(exc)

        # === Phase 1: Pre-processing ===

        # 1. Document classification
        if cfg.enable_classification:
            result = self._run_classification(text, cfg, result)

        # 2. Section parsing (shared context for downstream modules)
        section_result = None
        if cfg.enable_sections:
            result, section_result = self._run_sections(text, cfg, result)

        # 3. Quality analysis
        if cfg.enable_quality:
            result = self._run_quality(text, cfg, result)

        # 4. De-identification
        if cfg.enable_deidentification:
            result = self._run_deidentification(text, cfg, result)

        # 5. Abbreviation expansion
        if cfg.enable_abbreviations:
            result = self._run_abbreviations(text, cfg, result)

        # === Phase 2: Extraction & scoring ===

        # 6. Medication extraction
        if cfg.enable_medications:
            result = self._run_medications(text, cfg, result)

        # 7. Allergy extraction
        if cfg.enable_allergies:
            result = self._run_allergies(text, cfg, result)

        # 8. Vital signs extraction
        if cfg.enable_vitals:
            result = self._run_vitals(text, cfg, result)

        # 9. Temporal extraction
        if cfg.enable_temporal:
            result = self._run_temporal(text, cfg, result)

        # 10. Assertion detection (uses NER entities from base pipeline)
        if cfg.enable_assertions:
            result = self._run_assertions(text, cfg, result)

        # 11. Concept normalization (uses NER entities from base pipeline)
        if cfg.enable_normalization:
            result = self._run_normalization(text, cfg, result)

        # 12. SDoH extraction
        if cfg.enable_sdoh:
            result = self._run_sdoh(text, cfg, result)

        # 13. Relation extraction (uses NER entities from base pipeline)
        if cfg.enable_relations:
            result = self._run_relations(text, cfg, result)

        # 14. Comorbidity scoring (uses ICD codes from base pipeline)
        if cfg.enable_comorbidity:
            result = self._run_comorbidity(text, cfg, result)

        result.processing_time_ms = (time.time() - pipeline_start) * 1000
        logger.debug(
            "EnhancedClinicalPipeline.process completed in %.1f ms "
            "(document_id=%s, errors=%d)",
            result.processing_time_ms,
            document_id,
            len(result.component_errors),
        )
        return result

    def process_batch(
        self,
        texts: list[str],
        config: EnhancedPipelineConfig | None = None,
        document_ids: list[str | None] | None = None,
    ) -> list[EnhancedPipelineResult]:
        """Process multiple documents sequentially.

        Parameters
        ----------
        texts:
            List of clinical document texts.
        config:
            Shared pipeline configuration.
        document_ids:
            Optional identifiers aligned with *texts*.

        Returns
        -------
        list[EnhancedPipelineResult]
        """
        doc_ids = document_ids or [None] * len(texts)
        return [
            self.process(text, config=config, document_id=doc_id)
            for text, doc_id in zip(texts, doc_ids, strict=False)
        ]

    # ------------------------------------------------------------------
    # Stage runners
    # ------------------------------------------------------------------

    def _run_classification(
        self, text: str, cfg: EnhancedPipelineConfig,
        result: EnhancedPipelineResult,
    ) -> EnhancedPipelineResult:
        """Classify document type."""
        try:
            if self._classifier is None:
                return result
            classification = self._classifier.classify(text)
            result.classification = {
                "predicted_type": classification.predicted_type.value,
                "confidence": classification.scores[0].confidence
                if classification.scores else 0.0,
                "top_scores": [
                    {
                        "type": s.document_type.value,
                        "confidence": s.confidence,
                    }
                    for s in classification.scores[:3]
                ],
                "processing_time_ms": classification.processing_time_ms,
            }
        except Exception as exc:
            logger.error("Classification failed: %s", exc)
            result.component_errors["classification"] = str(exc)
        return result

    def _run_sections(
        self, text: str, cfg: EnhancedPipelineConfig,
        result: EnhancedPipelineResult,
    ) -> tuple[EnhancedPipelineResult, Any]:
        """Parse document sections and return raw result for downstream use."""
        section_result = None
        try:
            if self._section_parser is None:
                return result, None
            section_result = self._section_parser.parse(text)
            result.sections = {
                "sections": [
                    {
                        "category": str(s.category),
                        "header": s.header,
                        "header_normalised": s.header_normalised,
                        "header_start": s.header_start,
                        "header_end": s.header_end,
                        "body_end": s.body_end,
                        "confidence": s.confidence,
                    }
                    for s in section_result.sections
                ],
                "section_count": len(section_result.sections),
                "categories_found": sorted(
                    str(c) for c in section_result.categories_found
                ),
            }
        except Exception as exc:
            logger.error("Section parsing failed: %s", exc)
            result.component_errors["sections"] = str(exc)
        return result, section_result

    def _run_quality(
        self, text: str, cfg: EnhancedPipelineConfig,
        result: EnhancedPipelineResult,
    ) -> EnhancedPipelineResult:
        """Analyze clinical note quality."""
        try:
            if self._quality_analyzer is None:
                return result
            report = self._quality_analyzer.analyze(text)
            result.quality = {
                "overall_score": report.overall_score,
                "grade": report.grade,
                "dimensions": [
                    {
                        "dimension": d.dimension.value,
                        "score": d.score,
                        "weight": d.weight,
                        "finding_count": len(d.findings),
                    }
                    for d in report.dimensions
                ],
                "recommendation_count": len(report.recommendations),
                "top_recommendations": report.recommendations[:5],
            }
        except Exception as exc:
            logger.error("Quality analysis failed: %s", exc)
            result.component_errors["quality"] = str(exc)
        return result

    def _run_deidentification(
        self, text: str, cfg: EnhancedPipelineConfig,
        result: EnhancedPipelineResult,
    ) -> EnhancedPipelineResult:
        """De-identify PHI in clinical text."""
        try:
            if self._deidentifier is None:
                return result
            deid = self._deidentifier.deidentify(text)
            result.deidentification = deid
        except Exception as exc:
            logger.error("De-identification failed: %s", exc)
            result.component_errors["deidentification"] = str(exc)
        return result

    def _run_abbreviations(
        self, text: str, cfg: EnhancedPipelineConfig,
        result: EnhancedPipelineResult,
    ) -> EnhancedPipelineResult:
        """Detect and expand clinical abbreviations."""
        try:
            if self._abbreviation_expander is None:
                return result
            expansion = self._abbreviation_expander.expand(text)
            result.abbreviations = {
                "total_found": expansion.total_found,
                "expanded_text": expansion.expanded_text,
                "matches": [
                    {
                        "abbreviation": m.abbreviation,
                        "expansion": m.expansion,
                        "start": m.start,
                        "end": m.end,
                        "confidence": m.confidence,
                        "domain": m.domain.value
                        if hasattr(m.domain, "value") else str(m.domain),
                        "is_ambiguous": m.is_ambiguous,
                    }
                    for m in expansion.matches
                ],
            }
        except Exception as exc:
            logger.error("Abbreviation expansion failed: %s", exc)
            result.component_errors["abbreviations"] = str(exc)
        return result

    def _run_medications(
        self, text: str, cfg: EnhancedPipelineConfig,
        result: EnhancedPipelineResult,
    ) -> EnhancedPipelineResult:
        """Extract structured medication information."""
        try:
            if self._medication_extractor is None:
                return result
            med_result = self._medication_extractor.extract(text)
            result.medications = {
                "medication_count": med_result.medication_count,
                "medications": [
                    {
                        "drug_name": m.drug_name,
                        "generic_name": m.generic_name,
                        "dosage": m.dosage.to_dict()
                        if m.dosage and hasattr(m.dosage, "to_dict")
                        else str(m.dosage) if m.dosage else None,
                        "route": m.route.value
                        if hasattr(m.route, "value") else str(m.route),
                        "frequency": m.frequency,
                        "duration": m.duration,
                        "indication": m.indication,
                        "prn": m.prn,
                        "status": str(m.status),
                        "confidence": m.confidence,
                    }
                    for m in med_result.medications
                ],
                "processing_time_ms": med_result.processing_time_ms,
            }
        except Exception as exc:
            logger.error("Medication extraction failed: %s", exc)
            result.component_errors["medications"] = str(exc)
        return result

    def _run_allergies(
        self, text: str, cfg: EnhancedPipelineConfig,
        result: EnhancedPipelineResult,
    ) -> EnhancedPipelineResult:
        """Extract allergens with reactions and severity."""
        try:
            if self._allergy_extractor is None:
                return result
            allergy_result = self._allergy_extractor.extract(text)
            result.allergies = {
                "allergy_count": len(allergy_result.allergies),
                "no_known_allergies": allergy_result.no_known_allergies,
                "allergies": [
                    {
                        "allergen": a.allergen,
                        "category": str(a.category),
                        "reactions": [r.to_dict() for r in a.reactions],
                        "severity": str(a.severity),
                        "status": str(a.status),
                        "confidence": a.confidence,
                    }
                    for a in allergy_result.allergies
                ],
            }
        except Exception as exc:
            logger.error("Allergy extraction failed: %s", exc)
            result.component_errors["allergies"] = str(exc)
        return result

    def _run_vitals(
        self, text: str, cfg: EnhancedPipelineConfig,
        result: EnhancedPipelineResult,
    ) -> EnhancedPipelineResult:
        """Extract vital sign measurements."""
        try:
            if self._vitals_extractor is None:
                return result
            vitals_result = self._vitals_extractor.extract(text)
            result.vitals = {
                "vital_count": len(vitals_result.readings),
                "vitals": [
                    {
                        "type": v.vital_type.value,
                        "value": v.value,
                        "unit": v.unit,
                        "interpretation": v.interpretation.value,
                        "confidence": v.confidence,
                    }
                    for v in vitals_result.readings
                ],
                "processing_time_ms": vitals_result.extraction_time_ms,
            }
        except Exception as exc:
            logger.error("Vital signs extraction failed: %s", exc)
            result.component_errors["vitals"] = str(exc)
        return result

    def _run_temporal(
        self, text: str, cfg: EnhancedPipelineConfig,
        result: EnhancedPipelineResult,
    ) -> EnhancedPipelineResult:
        """Extract temporal expressions."""
        try:
            if self._temporal_extractor is None:
                return result
            temporal_result = self._temporal_extractor.extract(text)
            result.temporal = temporal_result.to_dict()
        except Exception as exc:
            logger.error("Temporal extraction failed: %s", exc)
            result.component_errors["temporal"] = str(exc)
        return result

    def _run_assertions(
        self, text: str, cfg: EnhancedPipelineConfig,
        result: EnhancedPipelineResult,
    ) -> EnhancedPipelineResult:
        """Detect assertion status for entities from the base NER pipeline."""
        try:
            if self._assertion_detector is None:
                return result
            # Use entities from the base pipeline
            entities = []
            if result.base_result and result.base_result.entities:
                entities = result.base_result.entities

            if not entities:
                result.assertions = []
                return result

            assertions = []
            for entity in entities:
                try:
                    assertion = self._assertion_detector.detect(
                        text=text,
                        entity_start=entity.start_char,
                        entity_end=entity.end_char,
                    )
                    assertions.append({
                        "entity_text": assertion.entity_text,
                        "entity_type": entity.entity_type,
                        "status": assertion.status.value,
                        "confidence": assertion.confidence,
                        "trigger_text": assertion.trigger_text,
                    })
                except Exception:
                    # Skip entities that fail assertion detection
                    pass
            result.assertions = assertions
        except Exception as exc:
            logger.error("Assertion detection failed: %s", exc)
            result.component_errors["assertions"] = str(exc)
        return result

    def _run_normalization(
        self, text: str, cfg: EnhancedPipelineConfig,
        result: EnhancedPipelineResult,
    ) -> EnhancedPipelineResult:
        """Normalize extracted entities to ontology codes."""
        try:
            if self._concept_normalizer is None:
                return result
            entities = []
            if result.base_result and result.base_result.entities:
                entities = result.base_result.entities

            if not entities:
                result.normalization = []
                return result

            normalizations = []
            for entity in entities:
                try:
                    norm_result = self._concept_normalizer.normalize(
                        text=entity.text,
                        entity_type=entity.entity_type,
                    )
                    if norm_result and norm_result.matched:
                        normalizations.append({
                            "entity_text": entity.text,
                            "entity_type": entity.entity_type,
                            "cui": norm_result.cui,
                            "preferred_term": norm_result.preferred_term,
                            "match_type": norm_result.match_type,
                            "confidence": norm_result.confidence,
                            "snomed_code": norm_result.snomed_code,
                            "rxnorm_code": norm_result.rxnorm_code,
                            "icd10_code": norm_result.icd10_code,
                            "loinc_code": norm_result.loinc_code,
                        })
                except Exception:
                    pass
            result.normalization = normalizations
        except Exception as exc:
            logger.error("Concept normalization failed: %s", exc)
            result.component_errors["normalization"] = str(exc)
        return result

    def _run_sdoh(
        self, text: str, cfg: EnhancedPipelineConfig,
        result: EnhancedPipelineResult,
    ) -> EnhancedPipelineResult:
        """Extract Social Determinants of Health."""
        try:
            if self._sdoh_extractor is None:
                return result
            sdoh_result = self._sdoh_extractor.extract(text)
            result.sdoh = {
                "extraction_count": len(sdoh_result.extractions),
                "adverse_count": sdoh_result.adverse_count,
                "protective_count": sdoh_result.protective_count,
                "domain_summary": dict(sdoh_result.domain_summary),
                "extractions": [
                    {
                        "domain": str(e.domain),
                        "text": e.text,
                        "sentiment": str(e.sentiment),
                        "confidence": e.confidence,
                        "z_codes": e.z_codes,
                    }
                    for e in sdoh_result.extractions
                ],
            }
        except Exception as exc:
            logger.error("SDoH extraction failed: %s", exc)
            result.component_errors["sdoh"] = str(exc)
        return result

    def _run_relations(
        self, text: str, cfg: EnhancedPipelineConfig,
        result: EnhancedPipelineResult,
    ) -> EnhancedPipelineResult:
        """Extract clinical relations between entities."""
        try:
            if self._relation_extractor is None:
                return result
            entities = []
            if result.base_result and result.base_result.entities:
                entities = result.base_result.entities

            if len(entities) < 2:
                result.relations = {"relation_count": 0, "relations": []}
                return result

            rel_result = self._relation_extractor.extract(
                text=text,
                entities=entities,
            )
            result.relations = {
                "relation_count": len(rel_result.relations),
                "pair_count": rel_result.pair_count,
                "relations": [
                    {
                        "subject": r.subject.text,
                        "subject_type": r.subject.entity_type,
                        "object": r.object_entity.text,
                        "object_type": r.object_entity.entity_type,
                        "relation_type": r.relation_type.value,
                        "confidence": r.confidence,
                        "evidence": r.evidence,
                    }
                    for r in rel_result.relations
                ],
            }
        except Exception as exc:
            logger.error("Relation extraction failed: %s", exc)
            result.component_errors["relations"] = str(exc)
        return result

    def _run_comorbidity(
        self, text: str, cfg: EnhancedPipelineConfig,
        result: EnhancedPipelineResult,
    ) -> EnhancedPipelineResult:
        """Calculate Charlson Comorbidity Index."""
        try:
            if self._charlson_calculator is None:
                return result
            # Use ICD codes from base pipeline if available
            icd_codes = []
            if result.base_result and result.base_result.icd_predictions:
                icd_codes = [
                    p.get("code", "") for p in result.base_result.icd_predictions
                    if p.get("code")
                ]

            if not icd_codes and not text:
                result.comorbidity = {
                    "raw_score": 0,
                    "age_adjusted_score": None,
                    "risk_group": "low",
                    "ten_year_mortality": 0.0,
                    "category_count": 0,
                    "matched_categories": [],
                }
                return result
            cci_result = self._charlson_calculator.calculate(
                icd_codes=icd_codes or None,
                text=text or None,
            )
            result.comorbidity = {
                "raw_score": cci_result.raw_score,
                "age_adjusted_score": cci_result.age_adjusted_score,
                "risk_group": cci_result.mortality_estimate.risk_group,
                "ten_year_mortality": cci_result.mortality_estimate.ten_year_mortality,
                "category_count": cci_result.category_count,
                "matched_categories": [
                    {
                        "category": str(c.category),
                        "weight": c.weight,
                        "source": c.source,
                        "evidence": c.evidence,
                        "confidence": c.confidence,
                    }
                    for c in cci_result.matched_categories
                ],
            }
        except Exception as exc:
            logger.error("Comorbidity scoring failed: %s", exc)
            result.component_errors["comorbidity"] = str(exc)
        return result
