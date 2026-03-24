"""ML utilities module initialization."""

from app.ml.utils.feature_engineering import (
    BagOfWordsExtractor,
    ClinicalFeatureExtractor,
    FeatureConfig,
    MEDICAL_STOPWORDS,
)
from app.ml.utils.text_preprocessing import (
    CLINICAL_SECTIONS,
    MEDICAL_ABBREVIATIONS,
    ClinicalTextPreprocessor,
    PreprocessingConfig,
    TextSection,
    preprocess_clinical_text,
)

__all__ = [
    # Text preprocessing
    "ClinicalTextPreprocessor",
    "PreprocessingConfig",
    "TextSection",
    "CLINICAL_SECTIONS",
    "MEDICAL_ABBREVIATIONS",
    "preprocess_clinical_text",
    # Feature engineering
    "ClinicalFeatureExtractor",
    "FeatureConfig",
    "BagOfWordsExtractor",
    "MEDICAL_STOPWORDS",
]
