"""Social Determinants of Health (SDoH) extraction module.

Extracts social and behavioural risk factors from unstructured clinical
text, categorising them across eight SDoH domains defined by Healthy
People 2030 and WHO frameworks:

* **Housing** — homelessness, housing instability, unsafe conditions
* **Employment** — unemployment, disability, job loss, retirement
* **Education** — literacy, educational attainment
* **Food Security** — food insecurity, malnutrition risk, food desert
* **Transportation** — transportation barriers to care
* **Social Support** — social isolation, caregiver burden, domestic violence
* **Substance Use** — tobacco, alcohol, illicit drug use, recovery status
* **Financial** — financial strain, insurance gaps, medical debt

Each extraction returns the domain, matched evidence text, sentiment
(positive/negative/neutral), confidence score, and any associated
ICD-10-CM Z-codes for social determinant documentation.
"""

from app.ml.sdoh.extractor import (
    ClinicalSDoHExtractor,
    SDoHDomain,
    SDoHExtraction,
    SDoHExtractionResult,
    SDoHSentiment,
)

__all__ = [
    "ClinicalSDoHExtractor",
    "SDoHDomain",
    "SDoHExtraction",
    "SDoHExtractionResult",
    "SDoHSentiment",
]
