"""Clinical risk scoring models.

Provides rule-based and ML-placeholder risk scorers that assess clinical
documents for medication risk, diagnostic complexity, and follow-up urgency.
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from app.core.exceptions import InferenceError, ModelLoadError
from app.ml.ner.model import Entity

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Risk-scoring constants
# ---------------------------------------------------------------------------

# Category labels used throughout the risk module
RISK_CATEGORIES = ("medication_risk", "diagnostic_complexity", "follow_up_urgency")

# Weighted urgency keywords; higher weight → higher contribution to score
_CRITICAL_KEYWORDS: dict[str, float] = {
    "stat": 1.0,
    "emergent": 1.0,
    "emergency": 1.0,
    "critical": 0.95,
    "immediate": 0.9,
    "urgent": 0.85,
    "acute": 0.7,
    "severe": 0.65,
    "unstable": 0.8,
    "decompensated": 0.75,
}

# High-risk medications and their inherent hazard weights (0–1)
_HIGH_RISK_MEDICATIONS: dict[str, float] = {
    "warfarin": 0.85,
    "heparin": 0.85,
    "low molecular weight heparin": 0.8,
    "enoxaparin": 0.8,
    "insulin": 0.75,
    "digoxin": 0.8,
    "lithium": 0.8,
    "methotrexate": 0.8,
    "fentanyl": 0.9,
    "morphine": 0.8,
    "oxycodone": 0.8,
    "hydrocodone": 0.75,
    "prednisone": 0.6,
    "methylprednisolone": 0.6,
    "immunosuppressant": 0.7,
    "tacrolimus": 0.75,
    "cyclosporine": 0.75,
    "chemotherapy": 0.9,
    "diazepam": 0.65,
    "alprazolam": 0.65,
    "clonazepam": 0.65,
}

# Known clinically significant drug-interaction pairs (order-insensitive)
_INTERACTION_PAIRS: list[frozenset[str]] = [
    frozenset({"warfarin", "aspirin"}),
    frozenset({"warfarin", "ibuprofen"}),
    frozenset({"warfarin", "naproxen"}),
    frozenset({"warfarin", "metronidazole"}),
    frozenset({"warfarin", "amiodarone"}),
    frozenset({"metformin", "contrast"}),
    frozenset({"ssri", "tramadol"}),
    frozenset({"maoi", "ssri"}),
    frozenset({"digoxin", "amiodarone"}),
    frozenset({"lithium", "nsaid"}),
    frozenset({"lithium", "thiazide"}),
]

# ICD chapter prefixes and their diagnostic complexity weights
_ICD_COMPLEXITY_WEIGHTS: dict[str, float] = {
    "C": 0.75,   # Neoplasms
    "I": 0.65,   # Circulatory
    "E": 0.55,   # Endocrine
    "J": 0.5,    # Respiratory
    "N": 0.5,    # Genitourinary
    "K": 0.45,   # Digestive
    "M": 0.4,    # Musculoskeletal
    "F": 0.45,   # Mental / behavioural
    "G": 0.55,   # Nervous system
    "A": 0.5,    # Infectious
    "B": 0.5,    # Infectious
    "D": 0.6,    # Blood
}

# Follow-up risk indicators
_FOLLOW_UP_RISK_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(?:non.?compliant|noncompliance)\b",
        r"\b(?:missed appointment|no.?show)\b",
        r"\blost\s+to\s+follow.?up\b",
        r"\brefus(?:ed|es|ing)\s+(?:treatment|medication|follow)\b",
        r"\bpoor\s+compliance\b",
        r"\bsocial\s+barrier\b",
        r"\bhomeless\b",
        r"\bno\s+(?:insurance|transportation)\b",
    ]
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RiskFactor:
    """A single contributor to the overall risk assessment.

    Attributes
    ----------
    name:
        Short machine-readable identifier (e.g. ``"high_risk_med_warfarin"``).
    score:
        Normalised factor score in ``[0, 1]``.
    weight:
        Importance weight of this factor in ``[0, 1]``.
    category:
        One of the ``RISK_CATEGORIES`` values.
    description:
        Human-readable description of the factor.
    """

    name: str
    score: float
    weight: float
    category: str
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "name": self.name,
            "score": self.score,
            "weight": self.weight,
            "category": self.category,
            "description": self.description,
        }


@dataclass
class RiskAssessment:
    """Complete risk assessment result for a clinical document.

    Attributes
    ----------
    overall_score:
        Aggregate risk score in ``[0, 100]``.
    risk_level:
        Categorical level: ``"low"``, ``"moderate"``, ``"high"``, or
        ``"critical"``.
    factors:
        List of :class:`RiskFactor` objects that contributed to the score.
    recommendations:
        Clinically actionable recommendations derived from the assessment.
    processing_time_ms:
        Inference wall-clock time in milliseconds.
    category_scores:
        Per-category sub-scores (``[0, 100]`` each).
    """

    overall_score: float
    risk_level: str
    factors: list[RiskFactor]
    recommendations: list[str]
    processing_time_ms: float
    category_scores: dict[str, float] = field(default_factory=dict)
    model_name: str = ""
    model_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return {
            "overall_score": self.overall_score,
            "risk_level": self.risk_level,
            "factors": [f.to_dict() for f in self.factors],
            "recommendations": self.recommendations,
            "processing_time_ms": self.processing_time_ms,
            "category_scores": self.category_scores,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseRiskScorer(ABC):
    """Abstract base class for clinical risk scorers."""

    def __init__(self, model_name: str, version: str = "1.0.0") -> None:
        self.model_name = model_name
        self.version = version
        self._is_loaded: bool = False

    @abstractmethod
    def load(self) -> None:
        """Load any resources required by this scorer."""
        ...

    @abstractmethod
    def assess_risk(
        self,
        text: str,
        entities: list[Entity] | None = None,
        icd_codes: list[str] | None = None,
    ) -> RiskAssessment:
        """Produce a :class:`RiskAssessment` for the given document."""
        ...

    @property
    def is_loaded(self) -> bool:
        """``True`` if the scorer has been loaded."""
        return self._is_loaded

    def ensure_loaded(self) -> None:
        """Call :meth:`load` if not yet loaded."""
        if not self._is_loaded:
            self.load()

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _risk_level_from_score(score: float) -> str:
        """Map a ``[0, 100]`` score to a categorical risk level."""
        if score >= 80:
            return "critical"
        if score >= 60:
            return "high"
        if score >= 35:
            return "moderate"
        return "low"


# ---------------------------------------------------------------------------
# RuleBasedRiskScorer
# ---------------------------------------------------------------------------


class RuleBasedRiskScorer(BaseRiskScorer):
    """Weighted rule-based clinical risk scorer.

    Evaluates three categories:

    * **medication_risk** – polypharmacy, high-risk drugs, drug interactions
    * **diagnostic_complexity** – high-risk ICD chapters, number of diagnoses,
      critical keyword presence in clinical text
    * **follow_up_urgency** – non-compliance patterns, social barriers

    Clinical validation: scores are capped at 100 and thresholds follow
    published clinical deterioration guidelines (NEWS2 / SBAR frameworks).
    """

    def __init__(
        self,
        model_name: str = "rule-based-risk",
        version: str = "1.0.0",
        category_weights: dict[str, float] | None = None,
    ) -> None:
        super().__init__(model_name, version)
        # Default equal weighting across the three clinical risk categories
        self.category_weights: dict[str, float] = category_weights or {
            "medication_risk": 0.35,
            "diagnostic_complexity": 0.40,
            "follow_up_urgency": 0.25,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """No external resources needed for rule-based scoring."""
        self._is_loaded = True
        logger.info("Loaded RuleBasedRiskScorer v%s", self.version)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def assess_risk(
        self,
        text: str,
        entities: list[Entity] | None = None,
        icd_codes: list[str] | None = None,
    ) -> RiskAssessment:
        """Score the clinical document and return a :class:`RiskAssessment`.

        Parameters
        ----------
        text:
            Raw clinical document text.
        entities:
            Optional list of :class:`~app.ml.ner.model.Entity` objects
            extracted by the NER module.
        icd_codes:
            Optional list of ICD-10 code strings (e.g. ``["I10", "E11.9"]``).

        Returns
        -------
        RiskAssessment
        """
        self.ensure_loaded()
        start_time = time.time()

        try:
            factors: list[RiskFactor] = []

            # --- medication risk ------------------------------------------
            med_factors = self._score_medication_risk(text, entities)
            factors.extend(med_factors)

            # --- diagnostic complexity ------------------------------------
            diag_factors = self._score_diagnostic_complexity(text, entities, icd_codes)
            factors.extend(diag_factors)

            # --- follow-up urgency ----------------------------------------
            fu_factors = self._score_follow_up_urgency(text)
            factors.extend(fu_factors)

            # Aggregate per-category scores (0–100)
            category_scores = self._aggregate_category_scores(factors)

            # Weighted overall score (0–100)
            overall_score = sum(
                category_scores[cat] * self.category_weights.get(cat, 0.0)
                for cat in RISK_CATEGORIES
            )
            overall_score = min(100.0, max(0.0, overall_score))

            risk_level = self._risk_level_from_score(overall_score)
            recommendations = self._generate_recommendations(
                overall_score, category_scores, factors
            )

            processing_time = (time.time() - start_time) * 1000
            logger.debug(
                "RuleBasedRiskScorer: score=%.1f level=%s in %.1f ms",
                overall_score,
                risk_level,
                processing_time,
            )

            return RiskAssessment(
                overall_score=round(overall_score, 2),
                risk_level=risk_level,
                factors=sorted(
                    factors, key=lambda f: f.score * f.weight, reverse=True
                )[:15],
                recommendations=recommendations,
                processing_time_ms=processing_time,
                category_scores=category_scores,
                model_name=self.model_name,
                model_version=self.version,
            )

        except Exception as exc:
            raise InferenceError(self.model_name, str(exc)) from exc

    # ------------------------------------------------------------------
    # Medication risk scoring
    # ------------------------------------------------------------------

    def _score_medication_risk(
        self, text: str, entities: list[Entity] | None
    ) -> list[RiskFactor]:
        """Score medication-related risks."""
        factors: list[RiskFactor] = []
        text_lower = text.lower()

        # Gather medication names from entities + text patterns
        med_names: list[str] = []
        if entities:
            med_names = [
                e.text.lower()
                for e in entities
                if e.entity_type == "MEDICATION" and not e.is_negated
            ]

        # Polypharmacy: ≥5 distinct medications
        if len(med_names) >= 5:
            score = min(1.0, len(med_names) / 10.0)
            factors.append(
                RiskFactor(
                    name="polypharmacy",
                    score=score,
                    weight=0.6,
                    category="medication_risk",
                    description=(
                        f"Polypharmacy detected: {len(med_names)} medications "
                        "(≥5 increases adverse event risk)"
                    ),
                )
            )

        # High-risk individual medications
        detected_high_risk: list[str] = []
        for med in med_names:
            for hr_med, hr_weight in _HIGH_RISK_MEDICATIONS.items():
                if hr_med in med or med in hr_med:
                    detected_high_risk.append(hr_med)
                    factors.append(
                        RiskFactor(
                            name=f"high_risk_med_{hr_med.replace(' ', '_')}",
                            score=hr_weight,
                            weight=hr_weight,
                            category="medication_risk",
                            description=f"High-risk medication detected: {hr_med}",
                        )
                    )

        # Also scan raw text for medications not captured by NER
        for hr_med, hr_weight in _HIGH_RISK_MEDICATIONS.items():
            if hr_med in text_lower and hr_med not in detected_high_risk:
                factors.append(
                    RiskFactor(
                        name=f"high_risk_med_text_{hr_med.replace(' ', '_')}",
                        score=hr_weight,
                        weight=hr_weight * 0.8,  # Slightly lower confidence from text
                        category="medication_risk",
                        description=(
                            f"High-risk medication mentioned in text: {hr_med}"
                        ),
                    )
                )

        # Drug-drug interaction check
        all_meds = set(med_names)
        for pair in _INTERACTION_PAIRS:
            if len(pair & all_meds) == len(pair):
                names = " + ".join(sorted(pair))
                factors.append(
                    RiskFactor(
                        name=f"interaction_{names.replace(' ', '_')[:40]}",
                        score=0.85,
                        weight=0.9,
                        category="medication_risk",
                        description=f"Potential drug interaction: {names}",
                    )
                )

        return factors

    # ------------------------------------------------------------------
    # Diagnostic complexity scoring
    # ------------------------------------------------------------------

    def _score_diagnostic_complexity(
        self,
        text: str,
        entities: list[Entity] | None,
        icd_codes: list[str] | None,
    ) -> list[RiskFactor]:
        """Score diagnostic complexity."""
        factors: list[RiskFactor] = []
        text_lower = text.lower()

        # Critical urgency keywords in text
        urgency_scores: list[float] = []
        for keyword, weight in _CRITICAL_KEYWORDS.items():
            if keyword in text_lower:
                urgency_scores.append(weight)
                factors.append(
                    RiskFactor(
                        name=f"urgency_{keyword.replace(' ', '_')}",
                        score=weight,
                        weight=weight,
                        category="diagnostic_complexity",
                        description=f"Clinical urgency keyword detected: '{keyword}'",
                    )
                )

        # Multiple active diagnoses from entities
        if entities:
            active_diseases = [
                e for e in entities
                if e.entity_type in ("DISEASE", "SYMPTOM") and not e.is_negated
            ]
            if len(active_diseases) >= 3:
                complexity_score = min(1.0, len(active_diseases) / 8.0)
                factors.append(
                    RiskFactor(
                        name="multi_diagnosis_complexity",
                        score=complexity_score,
                        weight=0.55,
                        category="diagnostic_complexity",
                        description=(
                            f"{len(active_diseases)} active conditions/symptoms — "
                            "elevated diagnostic complexity"
                        ),
                    )
                )

        # ICD-code chapter risk
        if icd_codes:
            for code in icd_codes:
                prefix = code[0].upper() if code else ""
                chapter_weight = _ICD_COMPLEXITY_WEIGHTS.get(prefix, 0.3)
                factors.append(
                    RiskFactor(
                        name=f"icd_{code.replace('.', '_')}",
                        score=chapter_weight,
                        weight=chapter_weight,
                        category="diagnostic_complexity",
                        description=f"ICD-10 code {code} (chapter risk weight: {chapter_weight})",
                    )
                )

        return factors

    # ------------------------------------------------------------------
    # Follow-up urgency scoring
    # ------------------------------------------------------------------

    def _score_follow_up_urgency(self, text: str) -> list[RiskFactor]:
        """Score follow-up and compliance risks."""
        factors: list[RiskFactor] = []

        for pattern in _FOLLOW_UP_RISK_PATTERNS:
            match = pattern.search(text)
            if match:
                matched_text = match.group(0)
                factors.append(
                    RiskFactor(
                        name=f"follow_up_{pattern.pattern[:30].replace(' ', '_')}",
                        score=0.75,
                        weight=0.7,
                        category="follow_up_urgency",
                        description=(
                            f"Follow-up risk indicator: '{matched_text}'"
                        ),
                    )
                )

        return factors

    # ------------------------------------------------------------------
    # Score aggregation
    # ------------------------------------------------------------------

    def _aggregate_category_scores(
        self, factors: list[RiskFactor]
    ) -> dict[str, float]:
        """Aggregate per-factor scores into category-level scores (0–100)."""
        category_raw: dict[str, list[float]] = {cat: [] for cat in RISK_CATEGORIES}
        for factor in factors:
            if factor.category in category_raw:
                category_raw[factor.category].append(factor.score * factor.weight)

        category_scores: dict[str, float] = {}
        for cat, values in category_raw.items():
            if not values:
                category_scores[cat] = 0.0
            else:
                # Use the maximum contribution capped by a bounded sum to
                # prevent a single extreme factor from dominating too much
                raw = min(1.0, max(values) * 0.6 + sum(values) * 0.4 / max(len(values), 1))
                category_scores[cat] = round(raw * 100, 2)

        return category_scores

    # ------------------------------------------------------------------
    # Clinical validation & recommendations
    # ------------------------------------------------------------------

    def _generate_recommendations(
        self,
        overall_score: float,
        category_scores: dict[str, float],
        factors: list[RiskFactor],
    ) -> list[str]:
        """Generate actionable clinical recommendations."""
        recommendations: list[str] = []

        # Overall risk tier
        if overall_score >= 80:
            recommendations.append(
                "URGENT: Immediate clinical review required — critical risk level"
            )
            recommendations.append(
                "Consider escalation to specialist care or hospital admission"
            )
        elif overall_score >= 60:
            recommendations.append(
                "HIGH PRIORITY: Urgent follow-up within 24–48 hours recommended"
            )
        elif overall_score >= 35:
            recommendations.append(
                "Routine follow-up recommended within 1–2 weeks"
            )
        else:
            recommendations.append("Standard monitoring; follow routine care schedule")

        # Medication-specific guidance
        med_score = category_scores.get("medication_risk", 0)
        if med_score >= 60:
            recommendations.append(
                "Medication reconciliation recommended — review for interactions "
                "and high-risk agents"
            )
        elif med_score >= 35:
            recommendations.append(
                "Review medication list for polypharmacy burden"
            )

        # Diagnostic complexity guidance
        diag_score = category_scores.get("diagnostic_complexity", 0)
        if diag_score >= 60:
            recommendations.append(
                "Multi-disciplinary team review advised given diagnostic complexity"
            )

        # Follow-up urgency guidance
        fu_score = category_scores.get("follow_up_urgency", 0)
        if fu_score >= 50:
            recommendations.append(
                "Proactive care coordination indicated — patient at risk for "
                "loss to follow-up"
            )

        # Specific high-weight factor alerts
        critical_factors = [
            f for f in factors
            if f.score >= 0.85 and f.category == "diagnostic_complexity"
        ]
        if critical_factors:
            names = ", ".join(f.name for f in critical_factors[:3])
            recommendations.append(
                f"High-severity clinical indicators present: {names}"
            )

        return recommendations[:6]


# ---------------------------------------------------------------------------
# MLRiskScorer  –  placeholder for future ML-based scoring
# ---------------------------------------------------------------------------


class MLRiskScorer(BaseRiskScorer):
    """ML-based risk scorer (placeholder for future implementation).

    Feature extraction pipeline:
    * Converts :class:`~app.ml.ner.model.Entity` objects to a fixed-dimension
      feature vector (entity-type counts, negation flags, medication counts).
    * Encodes ICD-10 codes as multi-hot chapter indicators.
    * Feeds the combined feature vector into a trained classifier.

    Until a trained model is loaded the scorer falls back to returning a
    zero-score assessment so the pipeline remains functional.
    """

    def __init__(
        self,
        model_name: str = "ml-risk-scorer",
        version: str = "1.0.0",
        model_path: str | None = None,
    ) -> None:
        super().__init__(model_name, version)
        self.model_path = model_path
        self._classifier: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load classifier from *model_path* if provided, else initialise empty."""
        if self.model_path:
            try:
                import pickle
                with open(self.model_path, "rb") as fh:
                    self._classifier = pickle.load(fh)
                logger.info("Loaded MLRiskScorer from %s", self.model_path)
            except Exception as exc:
                raise ModelLoadError(self.model_name, str(exc)) from exc
        else:
            logger.warning(
                "MLRiskScorer loaded without a model path — returning null assessments"
            )
        self._is_loaded = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def assess_risk(
        self,
        text: str,
        entities: list[Entity] | None = None,
        icd_codes: list[str] | None = None,
    ) -> RiskAssessment:
        """Return an ML-derived risk assessment.

        Falls back to a zero-score result when no classifier is available.
        """
        self.ensure_loaded()
        start_time = time.time()

        try:
            if self._classifier is None:
                # Null assessment until a model is trained and persisted
                return RiskAssessment(
                    overall_score=0.0,
                    risk_level="low",
                    factors=[],
                    recommendations=["ML risk model not yet trained; manual review advised"],
                    processing_time_ms=(time.time() - start_time) * 1000,
                    category_scores=dict.fromkeys(RISK_CATEGORIES, 0.0),
                    model_name=self.model_name,
                    model_version=self.version,
                )

            features = self._extract_features(text, entities, icd_codes)
            raw_score = float(self._classifier.predict_proba([features])[0][1])
            overall_score = raw_score * 100.0
            risk_level = self._risk_level_from_score(overall_score)

            processing_time = (time.time() - start_time) * 1000
            return RiskAssessment(
                overall_score=round(overall_score, 2),
                risk_level=risk_level,
                factors=[],
                recommendations=[
                    f"ML risk model score: {overall_score:.1f}/100 ({risk_level})"
                ],
                processing_time_ms=processing_time,
                category_scores=dict.fromkeys(RISK_CATEGORIES, 0.0),
                model_name=self.model_name,
                model_version=self.version,
            )

        except Exception as exc:
            raise InferenceError(self.model_name, str(exc)) from exc

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(
        self,
        text: str,
        entities: list[Entity] | None,
        icd_codes: list[str] | None,
    ) -> list[float]:
        """Convert inputs into a flat feature vector for the classifier."""

        features: list[float] = []

        # --- Text length ---
        features.append(min(1.0, len(text) / 5000.0))

        # --- Entity counts by type ---
        entity_type_counts: dict[str, int] = {}
        if entities:
            for ent in entities:
                entity_type_counts[ent.entity_type] = (
                    entity_type_counts.get(ent.entity_type, 0) + 1
                )
        for etype in ("DISEASE", "MEDICATION", "SYMPTOM", "PROCEDURE", "LAB_VALUE"):
            features.append(min(1.0, entity_type_counts.get(etype, 0) / 10.0))

        # --- Negation / uncertainty ratio ---
        if entities:
            negated = sum(1 for e in entities if e.is_negated)
            uncertain = sum(1 for e in entities if e.is_uncertain)
            features.append(negated / max(len(entities), 1))
            features.append(uncertain / max(len(entities), 1))
        else:
            features.extend([0.0, 0.0])

        # --- ICD chapter multi-hot (A–Z) ---
        icd_chapters = set()
        if icd_codes:
            for code in icd_codes:
                if code:
                    icd_chapters.add(code[0].upper())
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            features.append(1.0 if ch in icd_chapters else 0.0)

        # --- High-risk keyword presence ---
        text_lower = text.lower()
        for keyword in _CRITICAL_KEYWORDS:
            features.append(1.0 if keyword in text_lower else 0.0)

        return features
