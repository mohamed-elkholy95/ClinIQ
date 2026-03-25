"""Structured medication extraction from clinical text.

Parses medication mentions into normalized components: drug name, dosage,
route of administration, frequency, duration, indication, and PRN status.
Supports both free-text clinical narratives and structured medication lists.
"""

from app.ml.medications.extractor import (
    ClinicalMedicationExtractor,
    Dosage,
    MedicationMention,
    MedicationExtractionResult,
    RouteOfAdministration,
)

__all__ = [
    "ClinicalMedicationExtractor",
    "Dosage",
    "MedicationMention",
    "MedicationExtractionResult",
    "RouteOfAdministration",
]
