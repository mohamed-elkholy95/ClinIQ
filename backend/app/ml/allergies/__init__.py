"""Clinical allergy extraction from free-text notes.

Extracts drug, food, and environmental allergies with associated reaction
types, severity levels, and NKDA (No Known Drug Allergies) detection.

The rule-based extractor identifies allergy mentions via pattern matching
against a curated dictionary of ~150 common allergens across three
categories, detects associated reactions (rash, anaphylaxis, GI upset,
etc.), and classifies severity (mild, moderate, severe, life-threatening).

Typical usage::

    from app.ml.allergies import ClinicalAllergyExtractor

    extractor = ClinicalAllergyExtractor()
    result = extractor.extract("Allergies: PCN (anaphylaxis), sulfa (rash)")
    for allergy in result.allergies:
        print(allergy.allergen, allergy.category, allergy.reactions)
"""

from app.ml.allergies.extractor import (
    AllergyCategory,
    AllergyResult,
    AllergySeverity,
    AllergyStatus,
    ClinicalAllergyExtractor,
    DetectedAllergy,
    ExtractionResult,
)

__all__ = [
    "AllergyCategory",
    "AllergyResult",
    "AllergySeverity",
    "AllergyStatus",
    "ClinicalAllergyExtractor",
    "DetectedAllergy",
    "ExtractionResult",
]
