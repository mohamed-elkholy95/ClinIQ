"""Clinical note quality analysis module.

Scores the quality, completeness, and NLP-readiness of clinical notes before
they enter the inference pipeline.  Quality dimensions include:

* **Completeness** — presence of expected clinical sections (HPI, Assessment, Plan)
* **Readability** — sentence length, abbreviation density, spelling-like anomalies
* **Structure** — section header consistency, list/bullet usage, whitespace ratio
* **Information density** — medical term concentration, entity yield estimate
* **Consistency** — duplicate content, contradictory modifiers, temporal coherence

The main entry point is :class:`ClinicalNoteQualityAnalyzer`, which returns a
:class:`QualityReport` containing an overall quality score (0–100) and per-
dimension breakdowns with actionable recommendations.
"""

from app.ml.quality.analyzer import (
                                     ClinicalNoteQualityAnalyzer,
                                     QualityConfig,
                                     QualityDimension,
                                     QualityReport,
                                     QualityScore,
)

__all__ = [
    "ClinicalNoteQualityAnalyzer",
    "QualityConfig",
    "QualityDimension",
    "QualityReport",
    "QualityScore",
]
