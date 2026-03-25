"""Clinical note quality analysis engine.

Evaluates clinical notes across five quality dimensions and produces a
composite score (0–100) with per-dimension breakdowns and actionable
recommendations.  Designed to run *before* the NLP inference pipeline so
that low-quality notes can be flagged, triaged, or rejected early.

Architecture
~~~~~~~~~~~~

The analyzer is decomposed into five independent scoring functions — one per
quality dimension — whose weighted scores are combined into the final quality
rating.  Each scorer operates on the raw text and returns a
:class:`QualityScore` with a numeric value (0–100), a confidence weight, and
a list of findings.  The composite :class:`QualityReport` aggregates all
dimensions and generates recommendations sorted by severity.

Design decisions
~~~~~~~~~~~~~~~~

1. **Zero ML dependencies** — All heuristics are deterministic regex /
   statistics-based so the module can run in <5 ms on typical notes without
   loading any model artifacts.
2. **Configurable weights** — Deployers can re-weight dimensions via
   :class:`QualityConfig` to match institutional priorities (e.g. a dental
   clinic may weight section completeness differently from an ED).
3. **Extensible findings** — Each finding carries a severity level and a
   human-readable message, enabling both automated filtering and clinician-
   facing dashboards.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class QualityDimension(str, Enum):
    """Quality dimensions evaluated by the analyzer."""

    COMPLETENESS = "completeness"
    READABILITY = "readability"
    STRUCTURE = "structure"
    INFORMATION_DENSITY = "information_density"
    CONSISTENCY = "consistency"


class FindingSeverity(str, Enum):
    """Severity level for quality findings."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Finding:
    """A single quality finding with context."""

    dimension: QualityDimension
    severity: FindingSeverity
    message: str
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe dictionary."""
        result: dict[str, Any] = {
            "dimension": self.dimension.value,
            "severity": self.severity.value,
            "message": self.message,
        }
        if self.detail:
            result["detail"] = self.detail
        return result


@dataclass
class QualityScore:
    """Score for a single quality dimension."""

    dimension: QualityDimension
    score: float  # 0–100
    weight: float  # contribution weight (sums to 1.0 across dimensions)
    findings: list[Finding] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe dictionary."""
        return {
            "dimension": self.dimension.value,
            "score": round(self.score, 2),
            "weight": round(self.weight, 3),
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass
class QualityReport:
    """Aggregate quality report for a clinical note."""

    overall_score: float  # 0–100 weighted composite
    grade: str  # A / B / C / D / F
    dimensions: list[QualityScore]
    recommendations: list[str]
    stats: dict[str, Any]  # word count, section count, etc.
    text_hash: str  # SHA-256 of input for deduplication / audit
    analysis_ms: float  # wall-clock time

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe dictionary."""
        return {
            "overall_score": round(self.overall_score, 2),
            "grade": self.grade,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "recommendations": self.recommendations,
            "stats": self.stats,
            "text_hash": self.text_hash,
            "analysis_ms": round(self.analysis_ms, 2),
        }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class QualityConfig:
    """Tunable parameters for quality analysis.

    Parameters
    ----------
    weights : dict[QualityDimension, float] | None
        Per-dimension weights.  Normalised to sum to 1.0 internally.
        Defaults give equal weight to all five dimensions.
    min_word_count : int
        Notes shorter than this receive a completeness penalty.
    max_abbreviation_ratio : float
        Fraction of tokens that are abbreviations before a warning fires.
    expected_sections : list[str] | None
        Section headers that a "complete" note should contain.  Case-
        insensitive matching against detected headers.
    """

    weights: dict[QualityDimension, float] | None = None
    min_word_count: int = 30
    max_abbreviation_ratio: float = 0.25
    expected_sections: list[str] | None = None

    def normalized_weights(self) -> dict[QualityDimension, float]:
        """Return weights normalised to sum to 1.0."""
        raw = self.weights or {d: 1.0 for d in QualityDimension}
        total = sum(raw.values()) or 1.0
        return {d: w / total for d, w in raw.items()}


# ---------------------------------------------------------------------------
# Expected section headers for different note types
# ---------------------------------------------------------------------------

DEFAULT_EXPECTED_SECTIONS = [
    "chief complaint",
    "history of present illness",
    "assessment",
    "plan",
]

# Canonical section headers recognised as valid clinical sections
KNOWN_SECTION_HEADERS: set[str] = {
    "chief complaint",
    "history of present illness",
    "hpi",
    "past medical history",
    "pmh",
    "medications",
    "allergies",
    "review of systems",
    "ros",
    "physical examination",
    "physical exam",
    "pe",
    "vital signs",
    "vitals",
    "assessment",
    "assessment and plan",
    "a/p",
    "plan",
    "impression",
    "diagnosis",
    "procedures",
    "operative findings",
    "laboratory",
    "labs",
    "imaging",
    "radiology",
    "family history",
    "social history",
    "disposition",
    "follow-up",
    "recommendations",
    "subjective",
    "objective",
    "chief concern",
    "dental history",
    "periodontal assessment",
    "oral examination",
}

# ---------------------------------------------------------------------------
# Medical term patterns for information-density scoring
# ---------------------------------------------------------------------------

# Regex fragments for common medical term suffixes/prefixes
_MEDICAL_TERM_PATTERN = re.compile(
    r"\b(?:"
    # Suffixes indicating medical terms
    r"\w+itis"  # inflammation
    r"|\w+osis"  # condition/disease
    r"|\w+emia"  # blood condition
    r"|\w+ectomy"  # surgical removal
    r"|\w+otomy"  # surgical incision
    r"|\w+oplasty"  # surgical repair
    r"|\w+oscopy"  # visual examination
    r"|\w+ology"  # study of
    r"|\w+pathy"  # disease
    r"|\w+penia"  # deficiency
    r"|\w+trophy"  # growth/nourishment
    r"|\w+algia"  # pain
    r"|\w+uria"  # urine condition
    r"|\w+megaly"  # enlargement
    r"|\w+rrhea"  # flow/discharge
    r"|\w+stasis"  # stopping
    # Common medical abbreviations
    r"|(?:BP|HR|RR|SpO2|WBC|RBC|HbA1c|BUN|GFR|INR|PT|PTT|CBC|BMP|CMP|LFT|TSH)"
    r"|(?:ECG|EKG|MRI|CT|CXR|ABG|EEG|EMG|PFT)"
    # Drug-class suffixes
    r"|\w+cillin"
    r"|\w+mycin"
    r"|\w+statin"
    r"|\w+pril"
    r"|\w+sartan"
    r"|\w+olol"
    r"|\w+azole"
    r"|\w+prazole"
    r")\b",
    re.IGNORECASE,
)

# Common clinical abbreviations (subset for density counting)
_ABBREVIATION_PATTERN = re.compile(
    r"\b(?:"
    r"pt|htn|dm|dm2|cad|chf|copd|ckd|sob|mi|cva|tia|dvt|pe|uti"
    r"|bid|tid|qid|prn|qhs|qd|po|iv|im|sq|sl|pr"
    r"|hpi|pmh|ros|a/p|r/o|f/u|s/p|h/o|c/o|w/u|d/c"
    r"|abd|ext|neuro|gi|gu|cv|resp|msk|derm|heent|ent"
    r"|yo|y/o|m/f|nkda|nad"
    r")\b",
    re.IGNORECASE,
)

# Sentence splitter (handles abbreviations like "Dr.", "vs.", etc.)
# Uses fixed-width lookbehinds to avoid regex errors on Python 3.12+.
_SENTENCE_RE = re.compile(
    r"(?<!Dr)(?<!Mr)(?<!Mrs)(?<!Ms)(?<!Jr)(?<!Sr)(?<!vs)"
    r"[.!?]"
    r"(?=\s+[A-Z]|\s*$)",
)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class ClinicalNoteQualityAnalyzer:
    """Scores clinical note quality across five dimensions.

    Parameters
    ----------
    config : QualityConfig | None
        Override default tuning parameters.  ``None`` uses sensible clinical
        defaults.

    Examples
    --------
    >>> analyzer = ClinicalNoteQualityAnalyzer()
    >>> report = analyzer.analyze("CHIEF COMPLAINT: Chest pain.\\n...")
    >>> report.overall_score  # 0–100
    82.5
    >>> report.grade
    'B'
    """

    def __init__(self, config: QualityConfig | None = None) -> None:
        self.config = config or QualityConfig()
        self._weights = self.config.normalized_weights()
        self._expected_sections = [
            s.lower()
            for s in (self.config.expected_sections or DEFAULT_EXPECTED_SECTIONS)
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> QualityReport:
        """Analyze a clinical note and return a quality report.

        Parameters
        ----------
        text : str
            Raw clinical note text.

        Returns
        -------
        QualityReport
            Composite quality report with per-dimension scores, findings,
            and recommendations.
        """
        start = time.perf_counter()

        # Pre-compute shared statistics
        stats = self._compute_stats(text)

        # Score each dimension
        scores = [
            self._score_completeness(text, stats),
            self._score_readability(text, stats),
            self._score_structure(text, stats),
            self._score_information_density(text, stats),
            self._score_consistency(text, stats),
        ]

        # Weighted composite
        overall = sum(s.score * s.weight for s in scores)
        overall = max(0.0, min(100.0, overall))

        grade = self._score_to_grade(overall)

        # Collect recommendations sorted by severity
        recommendations = self._generate_recommendations(scores)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        report = QualityReport(
            overall_score=overall,
            grade=grade,
            dimensions=scores,
            recommendations=recommendations,
            stats=stats,
            text_hash=text_hash,
            analysis_ms=elapsed_ms,
        )

        logger.info(
            "quality_analysis",
            extra={
                "overall_score": round(overall, 2),
                "grade": grade,
                "word_count": stats.get("word_count", 0),
                "elapsed_ms": round(elapsed_ms, 2),
            },
        )

        return report

    def analyze_batch(
        self, texts: list[str]
    ) -> list[QualityReport]:
        """Analyze multiple notes and return individual reports.

        Parameters
        ----------
        texts : list[str]
            List of clinical note texts.

        Returns
        -------
        list[QualityReport]
            One report per input text, in the same order.
        """
        return [self.analyze(t) for t in texts]

    # ------------------------------------------------------------------
    # Shared statistics
    # ------------------------------------------------------------------

    def _compute_stats(self, text: str) -> dict[str, Any]:
        """Pre-compute statistics shared across scorers."""
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        line_count = text.count("\n") + 1
        sentences = self._split_sentences(text)
        sentence_count = len(sentences)

        # Detect sections
        detected_sections = self._detect_sections(text)

        # Abbreviation tokens
        abbrev_matches = _ABBREVIATION_PATTERN.findall(text)
        abbreviation_count = len(abbrev_matches)
        abbreviation_ratio = abbreviation_count / max(word_count, 1)

        # Medical terms
        medical_matches = _MEDICAL_TERM_PATTERN.findall(text)
        medical_term_count = len(medical_matches)
        medical_term_ratio = medical_term_count / max(word_count, 1)

        return {
            "word_count": word_count,
            "char_count": char_count,
            "line_count": line_count,
            "sentence_count": sentence_count,
            "sentences": sentences,
            "detected_sections": detected_sections,
            "section_count": len(detected_sections),
            "abbreviation_count": abbreviation_count,
            "abbreviation_ratio": round(abbreviation_ratio, 4),
            "medical_term_count": medical_term_count,
            "medical_term_ratio": round(medical_term_ratio, 4),
        }

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences, handling clinical abbreviations."""
        # First split by newlines that likely separate sentences
        parts: list[str] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Further split by sentence-ending punctuation
            subs = _SENTENCE_RE.split(line)
            for sub in subs:
                sub = sub.strip()
                if sub:
                    parts.append(sub)
        return parts if parts else [text.strip()] if text.strip() else []

    def _detect_sections(self, text: str) -> list[str]:
        """Detect section headers in the note.

        Looks for lines that match the pattern ``HEADER:`` or ``**Header**``
        or lines in ALL CAPS that are short (< 60 chars).
        """
        sections: list[str] = []
        header_re = re.compile(
            r"^\s*(?:"
            r"([A-Z][A-Z /&\-]{2,})\s*:"  # ALL CAPS HEADER:
            r"|([A-Z][A-Za-z /&\-]+)\s*:"  # Title Case Header:
            r"|\*\*(.+?)\*\*"  # **Bold Header**
            r")\s*",
            re.MULTILINE,
        )
        for match in header_re.finditer(text):
            header = (match.group(1) or match.group(2) or match.group(3)).strip()
            sections.append(header.lower())
        return sections

    # ------------------------------------------------------------------
    # Dimension scorers
    # ------------------------------------------------------------------

    def _score_completeness(
        self, text: str, stats: dict[str, Any]
    ) -> QualityScore:
        """Score note completeness based on word count and section coverage.

        Evaluates:
        - Minimum word count threshold
        - Presence of expected clinical sections
        - Number of distinct sections
        """
        findings: list[Finding] = []
        score = 100.0
        word_count: int = stats["word_count"]
        detected: list[str] = stats["detected_sections"]

        # Word count check
        if word_count < self.config.min_word_count:
            penalty = min(40.0, (self.config.min_word_count - word_count) * 2.0)
            score -= penalty
            findings.append(
                Finding(
                    dimension=QualityDimension.COMPLETENESS,
                    severity=FindingSeverity.CRITICAL if word_count < 10 else FindingSeverity.WARNING,
                    message=f"Note is very short ({word_count} words)",
                    detail=f"Expected at least {self.config.min_word_count} words for meaningful analysis.",
                )
            )

        # Expected section coverage
        found_expected = 0
        for expected in self._expected_sections:
            matched = any(
                expected in sec or sec in expected
                for sec in detected
            )
            if matched:
                found_expected += 1
            else:
                findings.append(
                    Finding(
                        dimension=QualityDimension.COMPLETENESS,
                        severity=FindingSeverity.WARNING,
                        message=f"Missing expected section: {expected.title()}",
                        detail="Section not detected in the note. The NLP pipeline may produce less accurate results.",
                    )
                )

        if self._expected_sections:
            section_coverage = found_expected / len(self._expected_sections)
            # Weight: 40 points from section coverage
            score -= (1.0 - section_coverage) * 40.0

        # Bonus for having many distinct sections
        if len(detected) >= 6:
            score = min(100.0, score + 5.0)
            findings.append(
                Finding(
                    dimension=QualityDimension.COMPLETENESS,
                    severity=FindingSeverity.INFO,
                    message=f"Well-structured note with {len(detected)} sections",
                )
            )

        return QualityScore(
            dimension=QualityDimension.COMPLETENESS,
            score=max(0.0, min(100.0, score)),
            weight=self._weights[QualityDimension.COMPLETENESS],
            findings=findings,
        )

    def _score_readability(
        self, text: str, stats: dict[str, Any]
    ) -> QualityScore:
        """Score readability based on sentence length and abbreviation density.

        Evaluates:
        - Average sentence length (optimal: 10–25 words)
        - Abbreviation density
        - Very long sentences (> 50 words)
        - Very short "sentences" (< 3 words, not list items)
        """
        findings: list[Finding] = []
        score = 100.0
        sentences: list[str] = stats["sentences"]
        abbreviation_ratio: float = stats["abbreviation_ratio"]

        if not sentences:
            return QualityScore(
                dimension=QualityDimension.READABILITY,
                score=0.0,
                weight=self._weights[QualityDimension.READABILITY],
                findings=[
                    Finding(
                        dimension=QualityDimension.READABILITY,
                        severity=FindingSeverity.CRITICAL,
                        message="No sentences detected in the note",
                    )
                ],
            )

        # Sentence length analysis
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_length = statistics.mean(sentence_lengths) if sentence_lengths else 0

        if avg_length > 35:
            penalty = min(25.0, (avg_length - 35) * 2.0)
            score -= penalty
            findings.append(
                Finding(
                    dimension=QualityDimension.READABILITY,
                    severity=FindingSeverity.WARNING,
                    message=f"Average sentence length is high ({avg_length:.0f} words)",
                    detail="Long sentences may reduce NLP extraction accuracy. Consider shorter, more focused sentences.",
                )
            )
        elif avg_length < 3:
            score -= 15.0
            findings.append(
                Finding(
                    dimension=QualityDimension.READABILITY,
                    severity=FindingSeverity.WARNING,
                    message=f"Average sentence length is very low ({avg_length:.1f} words)",
                    detail="Extremely short text fragments may lack context for entity extraction.",
                )
            )

        # Very long sentences
        very_long = sum(1 for l in sentence_lengths if l > 50)
        if very_long > 0:
            score -= min(15.0, very_long * 5.0)
            findings.append(
                Finding(
                    dimension=QualityDimension.READABILITY,
                    severity=FindingSeverity.WARNING,
                    message=f"{very_long} sentence(s) exceed 50 words",
                    detail="Very long sentences may span multiple clinical concepts and reduce extraction precision.",
                )
            )

        # Abbreviation density
        if abbreviation_ratio > self.config.max_abbreviation_ratio:
            penalty = min(20.0, (abbreviation_ratio - self.config.max_abbreviation_ratio) * 100)
            score -= penalty
            findings.append(
                Finding(
                    dimension=QualityDimension.READABILITY,
                    severity=FindingSeverity.WARNING,
                    message=f"High abbreviation density ({abbreviation_ratio:.0%})",
                    detail=(
                        f"More than {self.config.max_abbreviation_ratio:.0%} of tokens are clinical "
                        "abbreviations. The system can handle abbreviations but expanded forms improve accuracy."
                    ),
                )
            )
        elif abbreviation_ratio > 0.05:
            findings.append(
                Finding(
                    dimension=QualityDimension.READABILITY,
                    severity=FindingSeverity.INFO,
                    message=f"Moderate abbreviation usage ({abbreviation_ratio:.0%})",
                )
            )

        return QualityScore(
            dimension=QualityDimension.READABILITY,
            score=max(0.0, min(100.0, score)),
            weight=self._weights[QualityDimension.READABILITY],
            findings=findings,
        )

    def _score_structure(
        self, text: str, stats: dict[str, Any]
    ) -> QualityScore:
        """Score structural quality of the note.

        Evaluates:
        - Consistent section header format
        - Whitespace ratio (too much or too little)
        - Line length variance (uniformity vs. ragged)
        - Use of numbered or bulleted lists
        """
        findings: list[Finding] = []
        score = 100.0
        detected: list[str] = stats["detected_sections"]
        line_count: int = stats["line_count"]
        char_count: int = stats["char_count"]

        # Whitespace ratio
        if char_count > 0:
            whitespace_chars = sum(1 for c in text if c in (" ", "\t", "\n", "\r"))
            ws_ratio = whitespace_chars / char_count

            if ws_ratio > 0.50:
                score -= 15.0
                findings.append(
                    Finding(
                        dimension=QualityDimension.STRUCTURE,
                        severity=FindingSeverity.WARNING,
                        message=f"Excessive whitespace ({ws_ratio:.0%} of content)",
                        detail="Large amounts of whitespace may indicate OCR artifacts or formatting issues.",
                    )
                )
            elif ws_ratio < 0.10 and char_count > 100:
                score -= 10.0
                findings.append(
                    Finding(
                        dimension=QualityDimension.STRUCTURE,
                        severity=FindingSeverity.INFO,
                        message="Very dense text with minimal whitespace",
                        detail="Consider adding line breaks between sections for better structure.",
                    )
                )

        # Section header presence
        if not detected and stats["word_count"] > 50:
            score -= 20.0
            findings.append(
                Finding(
                    dimension=QualityDimension.STRUCTURE,
                    severity=FindingSeverity.WARNING,
                    message="No section headers detected",
                    detail="Unstructured notes are harder to parse. Section headers improve extraction accuracy.",
                )
            )
        elif detected:
            # Check if headers match known clinical sections
            known_matches = sum(
                1 for s in detected if s in KNOWN_SECTION_HEADERS
            )
            if detected and known_matches / len(detected) < 0.5:
                score -= 10.0
                findings.append(
                    Finding(
                        dimension=QualityDimension.STRUCTURE,
                        severity=FindingSeverity.INFO,
                        message="Some section headers are non-standard",
                        detail="Using standard clinical section headers improves NLP accuracy.",
                    )
                )

        # List detection (numbered or bulleted)
        list_re = re.compile(r"^\s*(?:\d+[.)]\s|[-•*]\s)", re.MULTILINE)
        list_items = len(list_re.findall(text))
        if list_items >= 3:
            score = min(100.0, score + 5.0)
            findings.append(
                Finding(
                    dimension=QualityDimension.STRUCTURE,
                    severity=FindingSeverity.INFO,
                    message=f"Good use of lists ({list_items} list items detected)",
                )
            )

        # Line length variance — if very high, suggests mixed formatting
        lines = [l for l in text.split("\n") if l.strip()]
        if len(lines) >= 3:
            lengths = [len(l) for l in lines]
            if statistics.stdev(lengths) > 80:
                score -= 5.0
                findings.append(
                    Finding(
                        dimension=QualityDimension.STRUCTURE,
                        severity=FindingSeverity.INFO,
                        message="High variance in line lengths",
                        detail="Inconsistent line lengths may indicate mixed formatting or pasted content.",
                    )
                )

        return QualityScore(
            dimension=QualityDimension.STRUCTURE,
            score=max(0.0, min(100.0, score)),
            weight=self._weights[QualityDimension.STRUCTURE],
            findings=findings,
        )

    def _score_information_density(
        self, text: str, stats: dict[str, Any]
    ) -> QualityScore:
        """Score information density — medical term concentration.

        Evaluates:
        - Medical term ratio (higher is better for clinical notes)
        - Presence of numeric values (vitals, lab results)
        - Entity yield estimate
        """
        findings: list[Finding] = []
        score = 70.0  # Start at 70 — notes should earn their way up
        medical_term_ratio: float = stats["medical_term_ratio"]
        word_count: int = stats["word_count"]

        # Medical term density
        if medical_term_ratio >= 0.10:
            score += 25.0
            findings.append(
                Finding(
                    dimension=QualityDimension.INFORMATION_DENSITY,
                    severity=FindingSeverity.INFO,
                    message=f"High medical term density ({medical_term_ratio:.0%})",
                    detail="Rich clinical vocabulary should yield good NLP extraction results.",
                )
            )
        elif medical_term_ratio >= 0.05:
            score += 15.0
            findings.append(
                Finding(
                    dimension=QualityDimension.INFORMATION_DENSITY,
                    severity=FindingSeverity.INFO,
                    message=f"Moderate medical term density ({medical_term_ratio:.0%})",
                )
            )
        elif medical_term_ratio < 0.02 and word_count > 30:
            score -= 15.0
            findings.append(
                Finding(
                    dimension=QualityDimension.INFORMATION_DENSITY,
                    severity=FindingSeverity.WARNING,
                    message=f"Low medical term density ({medical_term_ratio:.0%})",
                    detail="The note may contain mostly administrative or non-clinical content.",
                )
            )

        # Numeric values (vitals, lab results, dosages)
        numeric_re = re.compile(r"\b\d+(?:\.\d+)?(?:\s*(?:mg|mcg|ml|g|%|mmHg|bpm|kg|lb|/\d+))\b", re.IGNORECASE)
        numeric_count = len(numeric_re.findall(text))
        if numeric_count >= 5:
            score += 5.0
            findings.append(
                Finding(
                    dimension=QualityDimension.INFORMATION_DENSITY,
                    severity=FindingSeverity.INFO,
                    message=f"Good numeric data density ({numeric_count} measurements)",
                )
            )
        elif numeric_count == 0 and word_count > 50:
            score -= 5.0
            findings.append(
                Finding(
                    dimension=QualityDimension.INFORMATION_DENSITY,
                    severity=FindingSeverity.INFO,
                    message="No numeric measurements detected",
                    detail="Vital signs, lab values, and dosages add clinical value.",
                )
            )

        return QualityScore(
            dimension=QualityDimension.INFORMATION_DENSITY,
            score=max(0.0, min(100.0, score)),
            weight=self._weights[QualityDimension.INFORMATION_DENSITY],
            findings=findings,
        )

    def _score_consistency(
        self, text: str, stats: dict[str, Any]
    ) -> QualityScore:
        """Score internal consistency of the note.

        Evaluates:
        - Duplicate paragraphs (copy-paste artifacts)
        - Contradictory assertion modifiers (e.g. "no chest pain" near "chest pain present")
        - Mixed tense usage indicators
        """
        findings: list[Finding] = []
        score = 100.0

        # Duplicate paragraph detection
        paragraphs = [
            p.strip()
            for p in re.split(r"\n\s*\n", text)
            if len(p.strip()) > 20
        ]
        if paragraphs:
            para_hashes = [hashlib.md5(p.lower().encode()).hexdigest() for p in paragraphs]
            duplicates = len(para_hashes) - len(set(para_hashes))
            if duplicates > 0:
                penalty = min(30.0, duplicates * 15.0)
                score -= penalty
                findings.append(
                    Finding(
                        dimension=QualityDimension.CONSISTENCY,
                        severity=FindingSeverity.WARNING,
                        message=f"{duplicates} duplicate paragraph(s) detected",
                        detail="Repeated content suggests copy-paste artifacts that may confuse NLP models.",
                    )
                )

        # Contradictory modifier detection
        # Look for pairs like "denies chest pain" + "chest pain" without negation nearby
        negated_terms = set()
        affirmed_terms = set()

        neg_re = re.compile(
            r"(?:no|not|denies?|without|negative for|absence of)\s+(\w+(?:\s+\w+)?)",
            re.IGNORECASE,
        )
        for match in neg_re.finditer(text):
            negated_terms.add(match.group(1).lower().strip())

        # Find affirmed terms (simple heuristic: terms after "presents with", "positive for", etc.)
        affirm_re = re.compile(
            r"(?:presents?\s+with|positive\s+for|complains?\s+of|reports?|notes?)\s+(\w+(?:\s+\w+)?)",
            re.IGNORECASE,
        )
        for match in affirm_re.finditer(text):
            affirmed_terms.add(match.group(1).lower().strip())

        contradictions = negated_terms & affirmed_terms
        if contradictions:
            # Only flag if there are clear contradictions (not just "no pain" + "pain management")
            real_contradictions = [
                c for c in contradictions
                if len(c.split()) >= 2 or c in {"pain", "fever", "cough", "nausea", "vomiting"}
            ]
            if real_contradictions:
                score -= min(20.0, len(real_contradictions) * 10.0)
                findings.append(
                    Finding(
                        dimension=QualityDimension.CONSISTENCY,
                        severity=FindingSeverity.WARNING,
                        message=f"Potential contradictions detected: {', '.join(sorted(real_contradictions)[:3])}",
                        detail="The same term appears in both negated and affirmed contexts. Review for accuracy.",
                    )
                )

        # Very short note with no findings = good consistency by default
        if not findings:
            findings.append(
                Finding(
                    dimension=QualityDimension.CONSISTENCY,
                    severity=FindingSeverity.INFO,
                    message="No consistency issues detected",
                )
            )

        return QualityScore(
            dimension=QualityDimension.CONSISTENCY,
            score=max(0.0, min(100.0, score)),
            weight=self._weights[QualityDimension.CONSISTENCY],
            findings=findings,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_to_grade(score: float) -> str:
        """Convert numeric score to letter grade.

        Parameters
        ----------
        score : float
            Overall quality score (0–100).

        Returns
        -------
        str
            Letter grade: A (≥90), B (≥80), C (≥70), D (≥60), F (<60).
        """
        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "F"

    @staticmethod
    def _generate_recommendations(scores: list[QualityScore]) -> list[str]:
        """Generate prioritised recommendations from all findings.

        Parameters
        ----------
        scores : list[QualityScore]
            Dimension scores with findings.

        Returns
        -------
        list[str]
            Recommendations sorted by severity (critical first), then by
            dimension weight.
        """
        severity_order = {
            FindingSeverity.CRITICAL: 0,
            FindingSeverity.WARNING: 1,
            FindingSeverity.INFO: 2,
        }

        all_findings: list[tuple[int, float, Finding]] = []
        for qs in scores:
            for f in qs.findings:
                if f.severity != FindingSeverity.INFO:
                    all_findings.append((severity_order[f.severity], -qs.weight, f))

        all_findings.sort(key=lambda x: (x[0], x[1]))

        recommendations: list[str] = []
        for _, _, finding in all_findings:
            rec = finding.message
            if finding.detail:
                rec += f" — {finding.detail}"
            recommendations.append(rec)

        return recommendations
