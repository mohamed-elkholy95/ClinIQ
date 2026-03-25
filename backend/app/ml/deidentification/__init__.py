"""Protected Health Information (PHI) de-identification module.

Provides rule-based and pattern-based PHI detection and redaction for
clinical text, following the HIPAA Safe Harbor 18 identifier categories.
This module does NOT guarantee HIPAA compliance on its own—it is a
defence-in-depth layer designed to reduce accidental PHI exposure in
downstream analytics, logging, and model training pipelines.

Architecture
------------
- **PhiDetector** identifies PHI spans using regex patterns, contextual
  heuristics, and optional NER model integration.
- **Deidentifier** orchestrates detection and applies a configurable
  replacement strategy (redact, mask, or surrogate).
- Both classes are stateless and thread-safe once constructed.

HIPAA Safe Harbor identifiers covered:
  1. Names  2. Dates  3. Phone numbers  4. Fax numbers
  5. Email addresses  6. SSN  7. MRN  8. Health plan numbers
  9. Account numbers  10. Certificate/license numbers
  11. Vehicle identifiers  12. Device identifiers
  13. URLs  14. IP addresses  15. Biometric identifiers
  16. Photographs  17. Geographic data (zip codes)
  18. Ages over 89
"""

from app.ml.deidentification.detector import (
    DeidentificationConfig,
    Deidentifier,
    PhiDetector,
    PhiEntity,
    PhiType,
    ReplacementStrategy,
)

__all__ = [
    "DeidentificationConfig",
    "Deidentifier",
    "PhiDetector",
    "PhiEntity",
    "PhiType",
    "ReplacementStrategy",
]
