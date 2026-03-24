"""Risk Scoring system for clinical documents."""

import logging
from dataclasses import dataclass, field
from typing import Any

from app.ml.ner.model import Entity
from app.ml.utils.text_preprocessing import ClinicalTextPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class RiskFactor:
    """Represents a single risk factor."""

    name: str
    description: str
    weight: float
    value: float  # 0.0 to 1.0
    source: str  # "entity", "text", "derived"
    evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "weight": self.weight,
            "value": self.value,
            "source": self.source,
            "evidence": self.evidence,
        }


@dataclass
class RiskScore:
    """Complete risk assessment for a document."""

    overall_score: float  # 0.0 to 1.0
    risk_level: str  # "low", "moderate", "high", "critical"
    category_scores: dict[str, float]
    risk_factors: list[RiskFactor]
    protective_factors: list[RiskFactor]
    recommendations: list[str]
    processing_time_ms: float
    model_name: str
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_score": self.overall_score,
            "risk_level": self.risk_level,
            "category_scores": self.category_scores,
            "risk_factors": [rf.to_dict() for rf in self.risk_factors],
            "protective_factors": [pf.to_dict() for pf in self.protective_factors],
            "recommendations": self.recommendations,
            "processing_time_ms": self.processing_time_ms,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }


# Risk categories for clinical assessment
RISK_CATEGORIES = {
    "medication": {
        "description": "Medication-related risks",
        "factors": [
            "polypharmacy",
            "high_risk_medications",
            "drug_interactions",
            "non_compliance",
        ],
    },
    "cardiovascular": {
        "description": "Cardiovascular system risks",
        "factors": [
            "hypertension",
            "diabetes",
            "hyperlipidemia",
            "cardiac_history",
        ],
    },
    "infection": {
        "description": "Infection-related risks",
        "factors": [
            "immunosuppression",
            "chronic_infection",
            "recent_hospitalization",
        ],
    },
    "surgical": {
        "description": "Surgical/procedural risks",
        "factors": [
            "prior_surgery",
            "bleeding_risk",
            "anesthesia_risk",
        ],
    },
    "follow_up": {
        "description": "Follow-up compliance risks",
        "factors": [
            "missed_appointments",
            "poor_compliance",
            "social_barriers",
        ],
    },
}

# High-risk medical conditions and their weights
HIGH_RISK_CONDITIONS = {
    "acute myocardial infarction": 0.9,
    "stroke": 0.9,
    "pulmonary embolism": 0.9,
    "sepsis": 0.9,
    "acute respiratory failure": 0.85,
    "acute kidney injury": 0.8,
    "diabetic ketoacidosis": 0.8,
    "cardiac arrest": 0.95,
    "heart failure": 0.7,
    "coronary artery disease": 0.65,
    "chronic kidney disease": 0.6,
    "cirrhosis": 0.7,
    "cancer": 0.7,
    "hiv": 0.6,
    "copd exacerbation": 0.65,
    "pneumonia": 0.5,
    "urinary tract infection": 0.3,
    "hypertension": 0.4,
    "diabetes mellitus": 0.45,
    "obesity": 0.35,
}

# High-risk medications
HIGH_RISK_MEDICATIONS = {
    "warfarin": 0.7,
    "heparin": 0.7,
    "insulin": 0.6,
    "digoxin": 0.65,
    "lithium": 0.65,
    "methotrexate": 0.7,
    "prednisone": 0.5,
    "fentanyl": 0.75,
    "morphine": 0.7,
    "oxycodone": 0.7,
    "diazepam": 0.5,
    "alprazolam": 0.5,
}

# Urgency keywords in clinical text
URGENCY_KEYWORDS = {
    "critical": ["stat", "emergent", "critical", "immediate", "urgent", "emergency"],
    "high": ["acute", "severe", "unstable", "decompensated", "rapid"],
    "moderate": ["moderate", "concerning", "abnormal", "significant"],
}


class RiskScorer:
    """Calculate risk scores for clinical documents."""

    def __init__(
        self,
        model_name: str = "rule-based-risk",
        version: str = "1.0.0",
        category_weights: dict[str, float] | None = None,
    ):
        self.model_name = model_name
        self.version = version
        self.category_weights = category_weights or {
            "medication": 0.25,
            "cardiovascular": 0.25,
            "infection": 0.2,
            "surgical": 0.15,
            "follow_up": 0.15,
        }
        self.preprocessor = ClinicalTextPreprocessor()

    def calculate_risk(
        self,
        text: str,
        entities: list[Entity] | None = None,
        icd_predictions: list[dict] | None = None,
    ) -> RiskScore:
        """Calculate comprehensive risk score."""
        import time

        start_time = time.time()

        # Extract risk factors
        risk_factors = []
        protective_factors = []

        # Text-based risk factors
        text_factors = self._extract_text_risk_factors(text)
        risk_factors.extend(text_factors)

        # Entity-based risk factors
        if entities:
            entity_factors = self._extract_entity_risk_factors(entities)
            risk_factors.extend(entity_factors)

        # ICD-based risk factors
        if icd_predictions:
            icd_factors = self._extract_icd_risk_factors(icd_predictions)
            risk_factors.extend(icd_factors)

        # Calculate category scores
        category_scores = self._calculate_category_scores(risk_factors, text)

        # Calculate overall score
        overall_score = self._calculate_overall_score(category_scores, risk_factors)

        # Determine risk level
        risk_level = self._determine_risk_level(overall_score)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_score, category_scores, risk_factors
        )

        processing_time = (time.time() - start_time) * 1000

        return RiskScore(
            overall_score=overall_score,
            risk_level=risk_level,
            category_scores=category_scores,
            risk_factors=risk_factors[:10],  # Top 10
            protective_factors=protective_factors[:5],
            recommendations=recommendations,
            processing_time_ms=processing_time,
            model_name=self.model_name,
            model_version=self.version,
        )

    def _extract_text_risk_factors(self, text: str) -> list[RiskFactor]:
        """Extract risk factors from text patterns."""
        factors = []
        text_lower = text.lower()

        # Check for urgency keywords
        for level, keywords in URGENCY_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    factor = RiskFactor(
                        name=f"urgency_{keyword}",
                        description=f"Urgency keyword detected: {keyword}",
                        weight=0.8 if level == "critical" else 0.5 if level == "high" else 0.3,
                        value=1.0,
                        source="text",
                        evidence=keyword,
                    )
                    factors.append(factor)

        # Check for high-risk conditions in text
        for condition, weight in HIGH_RISK_CONDITIONS.items():
            if condition in text_lower:
                factor = RiskFactor(
                    name=condition.replace(" ", "_"),
                    description=f"High-risk condition: {condition}",
                    weight=weight,
                    value=1.0,
                    source="text",
                    evidence=condition,
                )
                factors.append(factor)

        # Count number of medical issues
        issue_indicators = [
            "diagnosis",
            "symptom",
            "abnormal",
            "positive for",
            "evidence of",
        ]
        issue_count = sum(1 for ind in issue_indicators if ind in text_lower)
        if issue_count >= 5:
            factors.append(
                RiskFactor(
                    name="multiple_issues",
                    description="Multiple medical issues identified",
                    weight=0.5,
                    value=min(1.0, issue_count / 10),
                    source="derived",
                    evidence=f"{issue_count} issues found",
                )
            )

        return factors

    def _extract_entity_risk_factors(self, entities: list[Entity]) -> list[RiskFactor]:
        """Extract risk factors from extracted entities."""
        factors = []

        # Count medications
        medications = [e for e in entities if e.entity_type == "MEDICATION"]
        if len(medications) >= 5:
            factors.append(
                RiskFactor(
                    name="polypharmacy",
                    description="Multiple medications (polypharmacy)",
                    weight=0.5,
                    value=min(1.0, len(medications) / 10),
                    source="entity",
                    evidence=f"{len(medications)} medications",
                )
            )

        # Check for high-risk medications
        for med in medications:
            med_lower = med.text.lower()
            for high_risk_med, weight in HIGH_RISK_MEDICATIONS.items():
                if high_risk_med in med_lower:
                    factors.append(
                        RiskFactor(
                            name=f"high_risk_med_{high_risk_med}",
                            description=f"High-risk medication: {med.text}",
                            weight=weight,
                            value=1.0,
                            source="entity",
                            evidence=med.text,
                        )
                    )

        # Check for disease entities
        diseases = [e for e in entities if e.entity_type == "DISEASE" and not e.is_negated]
        for disease in diseases:
            disease_lower = disease.text.lower()
            for condition, weight in HIGH_RISK_CONDITIONS.items():
                if condition in disease_lower or disease_lower in condition:
                    factors.append(
                        RiskFactor(
                            name=f"disease_{disease_lower[:20]}",
                            description=f"High-risk condition: {disease.text}",
                            weight=weight,
                            value=1.0,
                            source="entity",
                            evidence=disease.text,
                        )
                    )

        return factors

    def _extract_icd_risk_factors(self, icd_predictions: list[dict]) -> list[RiskFactor]:
        """Extract risk factors from ICD predictions."""
        factors = []

        for pred in icd_predictions[:10]:  # Top 10 predictions
            code = pred.get("code", "")
            description = pred.get("description", code)
            confidence = pred.get("confidence", 0)

            # Map ICD chapters to risk
            if code.startswith(("I",)):  # Circulatory
                weight = 0.6
            elif code.startswith(("E",)):  # Endocrine
                weight = 0.5
            elif code.startswith(("C", "D0", "D1", "D2", "D3", "D4")):  # Neoplasms
                weight = 0.7
            elif code.startswith(("J",)):  # Respiratory
                weight = 0.45
            elif code.startswith(("K",)):  # Digestive
                weight = 0.4
            else:
                weight = 0.3

            factors.append(
                RiskFactor(
                    name=f"icd_{code}",
                    description=f"ICD code: {code} - {description}",
                    weight=weight * confidence,
                    value=confidence,
                    source="icd",
                    evidence=code,
                )
            )

        return factors

    def _calculate_category_scores(
        self,
        risk_factors: list[RiskFactor],
        text: str,
    ) -> dict[str, float]:
        """Calculate scores for each risk category."""
        scores = {category: 0.0 for category in RISK_CATEGORIES}
        text_lower = text.lower()

        # Medication risks
        med_factors = [f for f in risk_factors if "med" in f.name.lower() or "polypharmacy" in f.name]
        if med_factors:
            scores["medication"] = min(1.0, sum(f.weight * f.value for f in med_factors) / 2)

        # Cardiovascular risks
        cv_keywords = ["hypertension", "cardiac", "heart", "coronary", "atrial", "myocardial"]
        cv_matches = sum(1 for kw in cv_keywords if kw in text_lower)
        scores["cardiovascular"] = min(1.0, cv_matches * 0.2)

        # Infection risks
        infection_keywords = ["infection", "sepsis", "bacterial", "viral", "immunocompromised"]
        inf_matches = sum(1 for kw in infection_keywords if kw in text_lower)
        scores["infection"] = min(1.0, inf_matches * 0.25)

        # Surgical risks
        surgical_keywords = ["surgery", "operative", "procedure", "incision", "post-op"]
        surg_matches = sum(1 for kw in surgical_keywords if kw in text_lower)
        scores["surgical"] = min(1.0, surg_matches * 0.2)

        # Follow-up risks
        fu_keywords = ["non-compliant", "missed", "lost to follow", "no show"]
        fu_matches = sum(1 for kw in fu_keywords if kw in text_lower)
        scores["follow_up"] = min(1.0, fu_matches * 0.3)

        return scores

    def _calculate_overall_score(
        self,
        category_scores: dict[str, float],
        risk_factors: list[RiskFactor],
    ) -> float:
        """Calculate overall risk score."""
        # Weighted average of category scores
        category_score = sum(
            category_scores[cat] * weight for cat, weight in self.category_weights.items()
        )

        # Boost from high-weight risk factors
        factor_boost = sum(
            f.weight * f.value for f in risk_factors if f.weight > 0.6
        ) / max(len([f for f in risk_factors if f.weight > 0.6]), 1)

        # Combine
        overall = (category_score * 0.7) + (factor_boost * 0.3)

        return min(1.0, max(0.0, overall))

    def _determine_risk_level(self, score: float) -> str:
        """Determine risk level from score."""
        if score >= 0.8:
            return "critical"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "moderate"
        else:
            return "low"

    def _generate_recommendations(
        self,
        overall_score: float,
        category_scores: dict[str, float],
        risk_factors: list[RiskFactor],
    ) -> list[str]:
        """Generate clinical recommendations based on risk assessment."""
        recommendations = []

        # General recommendations based on overall risk
        if overall_score >= 0.8:
            recommendations.append("Immediate clinical review recommended")
            recommendations.append("Consider escalation to specialist care")
        elif overall_score >= 0.6:
            recommendations.append("Urgent follow-up within 48-72 hours recommended")
        elif overall_score >= 0.4:
            recommendations.append("Routine follow-up recommended within 1-2 weeks")

        # Category-specific recommendations
        if category_scores.get("medication", 0) > 0.6:
            recommendations.append("Medication review recommended - assess for interactions")

        if category_scores.get("cardiovascular", 0) > 0.6:
            recommendations.append("Cardiology consultation may be indicated")

        if category_scores.get("infection", 0) > 0.6:
            recommendations.append("Consider infectious disease workup")

        # Factor-specific recommendations
        high_risk_factors = [f for f in risk_factors if f.weight > 0.7]
        if high_risk_factors:
            recommendations.append(
                f"Monitor closely for: {', '.join(f.name for f in high_risk_factors[:3])}"
            )

        return recommendations[:5]  # Max 5 recommendations
