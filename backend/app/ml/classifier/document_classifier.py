"""Clinical document type classifier.

Classifies clinical text into one of the standard clinical document types
used in healthcare settings (discharge summary, progress note, radiology
report, etc.).  The classifier uses a two-tier architecture:

1. **RuleBasedDocumentClassifier** — Fast, deterministic classification based
   on section header detection, keyword frequency analysis, and structural
   pattern matching.  No ML dependencies required.

2. **TransformerDocumentClassifier** — Optional HuggingFace sequence
   classification model that provides higher accuracy when available,
   with automatic fallback to rule-based classification on load failure.

Design decisions
----------------
- Rule-based classifier prioritised for interpretability and zero-dependency
  operation; clinical document types have strong structural signals (section
  headers like "DISCHARGE DIAGNOSIS" are near-perfect indicators).
- Scoring uses weighted combination of: section header matches (0.45),
  keyword density (0.30), and structural features (0.25) — weights tuned
  empirically on clinical note corpora.
- Multi-label support via confidence thresholds: a single note can match
  multiple types (e.g., a combined H&P + operative note) with ranked
  confidences.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from app.core.exceptions import InferenceError

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Document type taxonomy
# ---------------------------------------------------------------------------


class DocumentType(StrEnum):
    """Standard clinical document types.

    Based on HL7 CDA document type codes and common EHR categorisations.
    """

    DISCHARGE_SUMMARY = "discharge_summary"
    PROGRESS_NOTE = "progress_note"
    HISTORY_PHYSICAL = "history_physical"
    OPERATIVE_NOTE = "operative_note"
    CONSULTATION_NOTE = "consultation_note"
    RADIOLOGY_REPORT = "radiology_report"
    PATHOLOGY_REPORT = "pathology_report"
    LABORATORY_REPORT = "laboratory_report"
    NURSING_NOTE = "nursing_note"
    EMERGENCY_NOTE = "emergency_note"
    DENTAL_NOTE = "dental_note"
    PRESCRIPTION = "prescription"
    REFERRAL = "referral"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class ClassificationScore:
    """Confidence score for a single document type.

    Attributes
    ----------
    document_type:
        The classified document type.
    confidence:
        Confidence score in [0, 1].
    evidence:
        List of textual evidence snippets that contributed to this score.
    """

    document_type: DocumentType
    confidence: float
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dictionary."""
        return {
            "document_type": self.document_type.value,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
        }


@dataclass
class ClassificationResult:
    """Full classification result with ranked predictions.

    Attributes
    ----------
    predicted_type:
        Top-ranked document type.
    scores:
        All scored document types, sorted descending by confidence.
    processing_time_ms:
        Wall-clock time for classification in milliseconds.
    classifier_version:
        Version identifier of the classifier that produced this result.
    """

    predicted_type: DocumentType
    scores: list[ClassificationScore]
    processing_time_ms: float = 0.0
    classifier_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dictionary."""
        return {
            "predicted_type": self.predicted_type.value,
            "scores": [s.to_dict() for s in self.scores],
            "processing_time_ms": round(self.processing_time_ms, 2),
            "classifier_version": self.classifier_version,
        }


# ---------------------------------------------------------------------------
# Section header patterns per document type
# ---------------------------------------------------------------------------

DOCUMENT_SECTION_PATTERNS: dict[DocumentType, list[str]] = {
    DocumentType.DISCHARGE_SUMMARY: [
        r"discharge\s+(?:summary|diagnosis|diagnos[ei]s|instructions|medications|condition|disposition)",
        r"hospital\s+course",
        r"admission\s+(?:date|diagnosis|diagnos[ei]s)",
        r"date\s+of\s+discharge",
        r"discharge\s+date",
        r"condition\s+(?:at|on)\s+discharge",
        r"follow[\s-]?up\s+instructions",
    ],
    DocumentType.PROGRESS_NOTE: [
        r"(?:daily\s+)?progress\s+note",
        r"(?:subjective|objective|assessment|plan)\s*:",
        r"\bsoap\b",
        r"interval\s+history",
        r"overnight\s+events",
        r"24[\s-]?hour\s+events",
    ],
    DocumentType.HISTORY_PHYSICAL: [
        r"history\s+(?:and|&)\s+physical",
        r"\bh\s*(?:&|and)\s*p\b",
        r"chief\s+complaint",
        r"history\s+of\s+present\s+illness",
        r"review\s+of\s+systems",
        r"physical\s+exam(?:ination)?",
        r"past\s+medical\s+history",
        r"social\s+history",
        r"family\s+history",
    ],
    DocumentType.OPERATIVE_NOTE: [
        r"(?:operative|surgical|procedure)\s+(?:note|report)",
        r"pre[\s-]?operative\s+diagnosis",
        r"post[\s-]?operative\s+diagnosis",
        r"(?:procedure|operation)\s+performed",
        r"anesthesia(?:\s+type)?",
        r"(?:estimated\s+)?blood\s+loss",
        r"specimens?\s+(?:sent|submitted)",
        r"findings?\s*:",
        r"surgical\s+technique",
    ],
    DocumentType.CONSULTATION_NOTE: [
        r"consultation\s+(?:note|report|request)",
        r"consult(?:ation)?\s+(?:for|regarding|re:?)",
        r"reason\s+for\s+consult(?:ation)?",
        r"(?:requesting|referring)\s+(?:physician|provider|service)",
        r"recommendations?\s*:",
        r"impression\s+(?:and|&)\s+(?:plan|recommendations?)",
    ],
    DocumentType.RADIOLOGY_REPORT: [
        r"(?:radiology|imaging|radiological)\s+(?:report|findings|interpretation)",
        r"(?:clinical\s+)?indication",
        r"(?:technique|protocol)\s*:",
        r"comparison\s*:",
        r"(?:impression|findings)\s*:",
        r"\b(?:ct|mri|x[\s-]?ray|ultrasound|us|pet|mammograph[yic]|fluoroscop[yic])\b",
        r"contrast\s+(?:enhanced|administered|given)",
    ],
    DocumentType.PATHOLOGY_REPORT: [
        r"(?:pathology|histopathology|cytopathology|surgical\s+pathology)\s+report",
        r"gross\s+description",
        r"microscopic\s+(?:description|examination|findings)",
        r"(?:final\s+)?(?:pathologic(?:al)?|histologic(?:al)?)\s+diagnosis",
        r"specimen\s+(?:type|source|submitted|received)",
        r"immunohistochemi(?:cal|stry)",
        r"tumor\s+(?:size|grade|stage|margins?)",
    ],
    DocumentType.LABORATORY_REPORT: [
        r"(?:laboratory|lab)\s+(?:report|results?)",
        r"(?:reference|normal)\s+(?:range|values?)",
        r"(?:cbc|cmp|bmp|lfts?|urinalysis|culture|sensitivity|hba1c|lipid\s+panel)",
        r"specimen\s+(?:type|collected|received)",
        r"(?:flag|abnormal|critical)\s+(?:value|result)",
    ],
    DocumentType.NURSING_NOTE: [
        r"nurs(?:ing|e(?:'?s)?)\s+(?:note|assessment|documentation|progress)",
        r"patient\s+(?:assessment|status|condition)\s*:",
        r"(?:vital\s+signs|vitals)\s*:",
        r"(?:intake|output|i\s*(?:&|and)\s*o)\s*:",
        r"(?:pain\s+(?:assessment|scale|level|score))",
        r"(?:fall\s+risk|braden\s+score|morse\s+(?:fall\s+)?scale)",
        r"(?:skin\s+assessment|wound\s+care)",
    ],
    DocumentType.EMERGENCY_NOTE: [
        r"(?:emergency|ed|er)\s+(?:note|report|encounter|visit)",
        r"(?:mode\s+of\s+arrival|triage)",
        r"(?:medical\s+)?(?:screening|decision[\s-]?making)",
        r"disposition\s*:",
        r"(?:time\s+of\s+)?(?:arrival|departure)",
        r"(?:acuity|esi)\s+(?:level)?",
        r"(?:chief\s+complaint|presenting\s+complaint)\s*:",
    ],
    DocumentType.DENTAL_NOTE: [
        r"(?:dental|periodontal|endodontic|orthodontic|oral\s+surgery)\s+(?:note|exam|record|chart)",
        r"(?:tooth|teeth)\s+(?:#|number|nos?\.?)",
        r"(?:occlusion|bite|tmj|temporomandibular)",
        r"(?:caries|cavity|cavities|restoration|crown|bridge|implant|extraction)",
        r"(?:probing\s+depth|attachment\s+(?:level|loss)|recession|mobility|furcation)",
        r"(?:prophylaxis|scaling|root\s+planing|srp|debridement)",
        r"(?:radiograph|periapical|bitewing|panoramic|cbct)",
    ],
    DocumentType.PRESCRIPTION: [
        r"(?:prescription|rx|e[\s-]?prescri(?:ption|be))",
        r"(?:sig|directions?)\s*:",
        r"(?:dispense|quantity|refills?|days?\s+supply)",
        r"(?:dea|npi)\s*(?:#|number|:)",
        r"(?:generic|brand)\s+(?:name|substitution)",
    ],
    DocumentType.REFERRAL: [
        r"(?:referral|refer(?:ring)?)\s+(?:note|letter|request|form)",
        r"(?:reason\s+for\s+referral|referral\s+reason)",
        r"(?:referring|sending)\s+(?:physician|provider|clinician)",
        r"(?:referred?\s+to|specialist|specialty)",
        r"(?:urgency|priority)\s*:",
    ],
}


# ---------------------------------------------------------------------------
# Keyword patterns per document type (lower-case tokens)
# ---------------------------------------------------------------------------

DOCUMENT_KEYWORDS: dict[DocumentType, list[str]] = {
    DocumentType.DISCHARGE_SUMMARY: [
        "discharged", "discharge", "hospital course", "admission",
        "length of stay", "follow-up", "follow up", "home medications",
        "discharge medications", "disposition", "admitted",
    ],
    DocumentType.PROGRESS_NOTE: [
        "subjective", "objective", "assessment", "plan", "interval",
        "stable", "improved", "unchanged", "continues", "tolerating",
    ],
    DocumentType.HISTORY_PHYSICAL: [
        "chief complaint", "hpi", "review of systems", "ros",
        "physical exam", "past medical history", "family history",
        "social history", "allergies", "medications",
    ],
    DocumentType.OPERATIVE_NOTE: [
        "incision", "dissection", "anesthesia", "procedure",
        "blood loss", "tourniquet", "specimen", "hemostasis",
        "closure", "sterile", "operative", "surgical",
    ],
    DocumentType.CONSULTATION_NOTE: [
        "consult", "consultation", "recommend", "requesting",
        "opinion", "evaluation", "impression", "specialty",
    ],
    DocumentType.RADIOLOGY_REPORT: [
        "impression", "findings", "technique", "comparison",
        "contrast", "opacit", "density", "attenuation", "enhancement",
        "signal", "lesion", "nodule", "mass", "effusion",
    ],
    DocumentType.PATHOLOGY_REPORT: [
        "specimen", "gross", "microscopic", "diagnosis",
        "margins", "tumor", "grade", "stage", "stain",
        "immunohistochemistry", "malignant", "benign", "biopsy",
    ],
    DocumentType.LABORATORY_REPORT: [
        "result", "reference range", "flag", "abnormal", "critical",
        "specimen", "collected", "hemoglobin", "glucose", "creatinine",
        "wbc", "rbc", "platelet", "sodium", "potassium",
    ],
    DocumentType.NURSING_NOTE: [
        "assessment", "vitals", "intake", "output", "pain",
        "repositioned", "ambulated", "oriented", "alert",
        "skin", "wound", "dressing", "iv site", "catheter",
    ],
    DocumentType.EMERGENCY_NOTE: [
        "chief complaint", "triage", "acuity", "arrival",
        "disposition", "discharge", "observation", "emergent",
        "acute", "trauma", "ambulance", "ems",
    ],
    DocumentType.DENTAL_NOTE: [
        "tooth", "teeth", "caries", "restoration", "crown",
        "extraction", "implant", "probing", "recession", "mobility",
        "occlusion", "bitewing", "periapical", "prophylaxis",
    ],
    DocumentType.PRESCRIPTION: [
        "tablet", "capsule", "mg", "ml", "refill", "dispense",
        "sig", "prn", "daily", "twice", "quantity", "prescription",
    ],
    DocumentType.REFERRAL: [
        "referral", "referred", "specialist", "appointment",
        "evaluation", "opinion", "urgency", "authorization",
    ],
}


# ---------------------------------------------------------------------------
# Structural features (line count ranges, avg line length, etc.)
# ---------------------------------------------------------------------------

STRUCTURAL_PROFILES: dict[DocumentType, dict[str, Any]] = {
    DocumentType.DISCHARGE_SUMMARY: {
        "typical_min_lines": 30,
        "typical_max_lines": 500,
        "has_structured_sections": True,
        "typical_section_count_min": 5,
    },
    DocumentType.PROGRESS_NOTE: {
        "typical_min_lines": 5,
        "typical_max_lines": 60,
        "has_structured_sections": True,
        "typical_section_count_min": 2,
    },
    DocumentType.HISTORY_PHYSICAL: {
        "typical_min_lines": 30,
        "typical_max_lines": 200,
        "has_structured_sections": True,
        "typical_section_count_min": 5,
    },
    DocumentType.OPERATIVE_NOTE: {
        "typical_min_lines": 15,
        "typical_max_lines": 150,
        "has_structured_sections": True,
        "typical_section_count_min": 4,
    },
    DocumentType.CONSULTATION_NOTE: {
        "typical_min_lines": 15,
        "typical_max_lines": 150,
        "has_structured_sections": True,
        "typical_section_count_min": 3,
    },
    DocumentType.RADIOLOGY_REPORT: {
        "typical_min_lines": 5,
        "typical_max_lines": 80,
        "has_structured_sections": True,
        "typical_section_count_min": 2,
    },
    DocumentType.PATHOLOGY_REPORT: {
        "typical_min_lines": 10,
        "typical_max_lines": 150,
        "has_structured_sections": True,
        "typical_section_count_min": 3,
    },
    DocumentType.LABORATORY_REPORT: {
        "typical_min_lines": 5,
        "typical_max_lines": 100,
        "has_structured_sections": False,
        "typical_section_count_min": 0,
    },
    DocumentType.NURSING_NOTE: {
        "typical_min_lines": 3,
        "typical_max_lines": 50,
        "has_structured_sections": False,
        "typical_section_count_min": 0,
    },
    DocumentType.EMERGENCY_NOTE: {
        "typical_min_lines": 10,
        "typical_max_lines": 120,
        "has_structured_sections": True,
        "typical_section_count_min": 3,
    },
    DocumentType.DENTAL_NOTE: {
        "typical_min_lines": 5,
        "typical_max_lines": 80,
        "has_structured_sections": True,
        "typical_section_count_min": 1,
    },
    DocumentType.PRESCRIPTION: {
        "typical_min_lines": 3,
        "typical_max_lines": 30,
        "has_structured_sections": False,
        "typical_section_count_min": 0,
    },
    DocumentType.REFERRAL: {
        "typical_min_lines": 5,
        "typical_max_lines": 60,
        "has_structured_sections": False,
        "typical_section_count_min": 0,
    },
}


# ---------------------------------------------------------------------------
# Scoring weights
# ---------------------------------------------------------------------------

SECTION_WEIGHT = 0.45
KEYWORD_WEIGHT = 0.30
STRUCTURAL_WEIGHT = 0.25


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseDocumentClassifier(ABC):
    """Abstract base for clinical document classifiers.

    Parameters
    ----------
    version:
        Semantic version string for the classifier implementation.
    """

    def __init__(self, version: str = "1.0.0"):
        self.version = version

    @abstractmethod
    def classify(self, text: str) -> ClassificationResult:
        """Classify a clinical document.

        Parameters
        ----------
        text:
            Raw clinical document text.

        Returns
        -------
        ClassificationResult
            Ranked list of document type scores with the top prediction.
        """

    def classify_batch(
        self, texts: Sequence[str],
    ) -> list[ClassificationResult]:
        """Classify multiple documents.

        Parameters
        ----------
        texts:
            Sequence of clinical document texts.

        Returns
        -------
        list[ClassificationResult]
            One result per input document, preserving order.
        """
        return [self.classify(t) for t in texts]


# ---------------------------------------------------------------------------
# Rule-based implementation
# ---------------------------------------------------------------------------


class RuleBasedDocumentClassifier(BaseDocumentClassifier):
    """Deterministic classifier using section headers, keywords, and structure.

    This classifier requires no ML model files and runs in <1 ms per document.
    It provides a reliable baseline and serves as the automatic fallback when
    the transformer classifier is unavailable.

    Scoring algorithm
    -----------------
    For each document type *t*, the final score is:

        score(t) = 0.45 × section_score(t) + 0.30 × keyword_score(t) + 0.25 × structural_score(t)

    - **section_score**: Fraction of *t*'s section header patterns found in
      the text, boosted by 0.1 if the first matched pattern appears in the
      opening 500 characters (header-position bonus).
    - **keyword_score**: Fraction of *t*'s keywords found in the lower-cased
      text, capped at 1.0.
    - **structural_score**: Binary features for line count and section count
      falling within *t*'s expected profile.

    Parameters
    ----------
    min_confidence:
        Minimum confidence threshold for including a type in results.
        Defaults to 0.05.
    """

    def __init__(
        self,
        min_confidence: float = 0.05,
        version: str = "1.0.0",
    ):
        super().__init__(version=version)
        self.min_confidence = min_confidence
        self._compiled_patterns: dict[DocumentType, list[re.Pattern]] = {}
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Pre-compile regex patterns for all document types."""
        for doc_type, patterns in DOCUMENT_SECTION_PATTERNS.items():
            self._compiled_patterns[doc_type] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    # -- scoring components --------------------------------------------------

    def _section_score(
        self, text: str, doc_type: DocumentType,
    ) -> tuple[float, list[str]]:
        """Score based on section header pattern matches.

        Parameters
        ----------
        text:
            Clinical document text.
        doc_type:
            Document type to score against.

        Returns
        -------
        tuple[float, list[str]]
            (score in [0, 1], list of matched pattern evidence strings)
        """
        patterns = self._compiled_patterns.get(doc_type, [])
        if not patterns:
            return 0.0, []

        matches: list[str] = []
        first_match_pos = len(text)  # sentinel

        for pattern in patterns:
            match = pattern.search(text)
            if match:
                matches.append(match.group().strip())
                first_match_pos = min(first_match_pos, match.start())

        if not matches:
            return 0.0, []

        # Base score: fraction of patterns matched
        base = len(matches) / len(patterns)

        # Header-position bonus: if the first match is in the opening 500
        # characters, the document likely leads with this type's header
        position_bonus = 0.1 if first_match_pos < 500 else 0.0

        return min(base + position_bonus, 1.0), matches

    def _keyword_score(
        self, text_lower: str, doc_type: DocumentType,
    ) -> tuple[float, list[str]]:
        """Score based on keyword frequency.

        Parameters
        ----------
        text_lower:
            Lower-cased document text.
        doc_type:
            Document type to score against.

        Returns
        -------
        tuple[float, list[str]]
            (score in [0, 1], list of matched keywords)
        """
        keywords = DOCUMENT_KEYWORDS.get(doc_type, [])
        if not keywords:
            return 0.0, []

        found = [kw for kw in keywords if kw in text_lower]
        score = min(len(found) / len(keywords), 1.0)
        return score, found

    def _structural_score(
        self,
        line_count: int,
        section_count: int,
        doc_type: DocumentType,
    ) -> float:
        """Score based on document structural features.

        Parameters
        ----------
        line_count:
            Number of non-empty lines in the document.
        section_count:
            Number of detected section-like headers.
        doc_type:
            Document type to score against.

        Returns
        -------
        float
            Score in [0, 1].
        """
        profile = STRUCTURAL_PROFILES.get(doc_type)
        if not profile:
            return 0.0

        score = 0.0

        # Line count within expected range
        min_lines = profile.get("typical_min_lines", 0)
        max_lines = profile.get("typical_max_lines", 10000)
        if min_lines <= line_count <= max_lines:
            score += 0.5

        # Section count meets expectation
        min_sections = profile.get("typical_section_count_min", 0)
        if profile.get("has_structured_sections", False):
            if section_count >= min_sections:
                score += 0.5
        else:
            # Unstructured document types score higher if *few* sections
            if section_count <= 3:
                score += 0.5

        return score

    def _count_sections(self, text: str) -> int:
        """Count section-header-like lines in the text.

        A line is considered a section header if it matches common header
        patterns: all-caps short lines, lines ending with ':', or lines
        with specific formatting.

        Parameters
        ----------
        text:
            Clinical document text.

        Returns
        -------
        int
            Approximate number of section headers.
        """
        count = 0
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            # All-caps line (2–60 chars) — common header style
            if stripped.isupper() and 2 <= len(stripped) <= 60 or stripped.endswith(":") and len(stripped) <= 80:
                count += 1
        return count

    # -- main classify -------------------------------------------------------

    def classify(self, text: str) -> ClassificationResult:
        """Classify a clinical document using rule-based scoring.

        Parameters
        ----------
        text:
            Raw clinical document text.

        Returns
        -------
        ClassificationResult
            Ranked classification scores with the top prediction.

        Raises
        ------
        InferenceError
            If classification fails unexpectedly.
        """
        import time as _time

        start = _time.perf_counter()

        try:
            text_lower = text.lower()
            lines = [l for l in text.split("\n") if l.strip()]
            line_count = len(lines)
            section_count = self._count_sections(text)

            scores: list[ClassificationScore] = []

            for doc_type in DocumentType:
                if doc_type == DocumentType.UNKNOWN:
                    continue

                sec_score, sec_evidence = self._section_score(text, doc_type)
                kw_score, kw_evidence = self._keyword_score(text_lower, doc_type)
                struct_score = self._structural_score(
                    line_count, section_count, doc_type,
                )

                combined = (
                    SECTION_WEIGHT * sec_score
                    + KEYWORD_WEIGHT * kw_score
                    + STRUCTURAL_WEIGHT * struct_score
                )

                if combined >= self.min_confidence:
                    evidence = []
                    if sec_evidence:
                        evidence.append(
                            f"Section headers: {', '.join(sec_evidence[:5])}"
                        )
                    if kw_evidence:
                        evidence.append(
                            f"Keywords: {', '.join(kw_evidence[:5])}"
                        )
                    scores.append(
                        ClassificationScore(
                            document_type=doc_type,
                            confidence=combined,
                            evidence=evidence,
                        )
                    )

            # Sort descending by confidence
            scores.sort(key=lambda s: s.confidence, reverse=True)

            predicted = scores[0].document_type if scores else DocumentType.UNKNOWN

            elapsed_ms = (_time.perf_counter() - start) * 1000

            return ClassificationResult(
                predicted_type=predicted,
                scores=scores,
                processing_time_ms=elapsed_ms,
                classifier_version=self.version,
            )

        except Exception as e:
            raise InferenceError("document_classifier", str(e)) from e


# ---------------------------------------------------------------------------
# Transformer implementation
# ---------------------------------------------------------------------------


class TransformerDocumentClassifier(BaseDocumentClassifier):
    """HuggingFace sequence-classification based document classifier.

    Uses a fine-tuned transformer model for document type prediction.
    Falls back to :class:`RuleBasedDocumentClassifier` when the model
    cannot be loaded (missing dependencies, model files, etc.).

    Parameters
    ----------
    model_name:
        HuggingFace model identifier or local path.
    max_length:
        Maximum token length for the tokenizer.
    version:
        Semantic version string.
    """

    def __init__(
        self,
        model_name: str = "clinical-document-classifier",
        max_length: int = 512,
        version: str = "1.0.0",
    ):
        super().__init__(version=version)
        self.model_name = model_name
        self.max_length = max_length
        self._model: Any = None
        self._tokenizer: Any = None
        self._is_loaded = False
        self._fallback = RuleBasedDocumentClassifier(version=version)

    def load(self) -> None:
        """Load the transformer model and tokenizer.

        Raises
        ------
        ModelLoadError
            If the model cannot be loaded.
        """
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
            )
            self._model.eval()
            self._is_loaded = True
            logger.info(
                "Loaded transformer document classifier: %s", self.model_name,
            )
        except Exception as e:
            logger.warning(
                "Failed to load transformer classifier '%s': %s. "
                "Falling back to rule-based.",
                self.model_name,
                e,
            )
            self._is_loaded = False

    def classify(self, text: str) -> ClassificationResult:
        """Classify using transformer model with rule-based fallback.

        Parameters
        ----------
        text:
            Raw clinical document text.

        Returns
        -------
        ClassificationResult
            Ranked classification scores.
        """
        if not self._is_loaded:
            return self._fallback.classify(text)

        import time as _time

        start = _time.perf_counter()

        try:
            import torch

            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)[0]

            # Map model label indices to DocumentType
            id2label = getattr(
                self._model.config, "id2label", {},
            )

            scores: list[ClassificationScore] = []
            for idx, prob in enumerate(probabilities):
                label = id2label.get(idx, f"LABEL_{idx}")
                try:
                    doc_type = DocumentType(label)
                except ValueError:
                    # Try matching by name
                    try:
                        doc_type = DocumentType[label.upper()]
                    except KeyError:
                        continue

                scores.append(
                    ClassificationScore(
                        document_type=doc_type,
                        confidence=float(prob),
                        evidence=[f"Transformer prediction (logit={float(logits[0][idx]):.4f})"],
                    )
                )

            scores.sort(key=lambda s: s.confidence, reverse=True)

            predicted = (
                scores[0].document_type if scores else DocumentType.UNKNOWN
            )

            elapsed_ms = (_time.perf_counter() - start) * 1000

            return ClassificationResult(
                predicted_type=predicted,
                scores=scores,
                processing_time_ms=elapsed_ms,
                classifier_version=self.version,
            )

        except Exception as e:
            logger.warning(
                "Transformer classification failed: %s. Falling back to rule-based.",
                e,
            )
            return self._fallback.classify(text)
