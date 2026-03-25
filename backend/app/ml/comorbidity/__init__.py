"""Comorbidity scoring module.

Provides the Charlson Comorbidity Index (CCI) calculator and related
clinical scoring instruments for quantifying disease burden from
structured ICD-10-CM codes and free-text clinical narratives.
"""

from app.ml.comorbidity.charlson import (
    CCICategory,
    CCIResult,
    CharlsonCalculator,
    CharlsonConfig,
    ComorbidityMatch,
    MortalityEstimate,
)

__all__ = [
    "CCICategory",
    "CCIResult",
    "CharlsonCalculator",
    "CharlsonConfig",
    "ComorbidityMatch",
    "MortalityEstimate",
]
