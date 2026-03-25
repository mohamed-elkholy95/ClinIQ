"""Clinical document section parsing and boundary detection.

Provides a unified section parser that identifies and segments clinical
documents into their constituent sections (e.g., "Chief Complaint",
"History of Present Illness", "Medications", "Assessment & Plan").

Multiple downstream modules (vitals, SDoH, assertions, medications, quality)
previously contained ad-hoc section detection logic.  This module
consolidates that into a single, reusable parser with a rich catalogue of
~60 clinical section header patterns covering standard note types (H&P,
discharge summaries, operative notes, dental notes).

Typical usage::

    from app.ml.sections import ClinicalSectionParser

    parser = ClinicalSectionParser()
    result = parser.parse("CHIEF COMPLAINT:\\nChest pain\\n\\nHPI:\\n...")
    for section in result.sections:
        print(section.header, section.category, section.span)
"""

from app.ml.sections.parser import (
    ClinicalSectionParser,
    SectionCategory,
    SectionParseResult,
    SectionSpan,
)

__all__ = [
    "ClinicalSectionParser",
    "SectionCategory",
    "SectionParseResult",
    "SectionSpan",
]
