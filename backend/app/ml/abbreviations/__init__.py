"""Clinical abbreviation detection and expansion module.

Identifies and expands medical abbreviations in clinical free text,
improving readability and downstream NLP accuracy. Covers 300+ abbreviations
across 12 clinical domains with context-aware disambiguation for ambiguous
abbreviations (e.g., "PE" → pulmonary embolism vs physical exam).
"""

from app.ml.abbreviations.expander import (
    AbbreviationConfig,
    AbbreviationExpander,
    AbbreviationMatch,
    AmbiguityResolution,
    ClinicalDomain,
    ExpansionResult,
)

__all__ = [
    "AbbreviationConfig",
    "AbbreviationExpander",
    "AbbreviationMatch",
    "AmbiguityResolution",
    "ClinicalDomain",
    "ExpansionResult",
]
