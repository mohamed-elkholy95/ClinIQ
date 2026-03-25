"""Clinical assertion detection module.

Implements NegEx/ConText-inspired assertion classification for clinical
entities, determining whether a medical concept is present, absent (negated),
possible (uncertain), conditional, hypothetical, or associated with someone
else (e.g., family history).
"""

from app.ml.assertions.detector import (
    AssertionDetector,
    AssertionResult,
    AssertionStatus,
    ConTextAssertionDetector,
    RuleBasedAssertionDetector,
    Trigger,
    TriggerType,
)

__all__ = [
    "AssertionDetector",
    "AssertionResult",
    "AssertionStatus",
    "ConTextAssertionDetector",
    "RuleBasedAssertionDetector",
    "Trigger",
    "TriggerType",
]
