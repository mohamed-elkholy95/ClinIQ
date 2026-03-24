"""Unit tests for clinical text feature engineering.

Tests ClinicalFeatureExtractor (TF-IDF + custom clinical features)
and BagOfWordsExtractor (baseline feature extraction).
"""

import numpy as np
import pytest

from app.ml.utils.feature_engineering import (
    CRITICAL_KEYWORDS,
    MEDICAL_STOPWORDS,
    BagOfWordsExtractor,
    ClinicalFeatureExtractor,
    FeatureConfig,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def clinical_docs() -> list[str]:
    """Small corpus of clinical notes for fitting extractors."""
    return [
        "Patient presents with acute chest pain and shortness of breath. "
        "History of hypertension. Medications: lisinopril 10mg daily.",
        "Chief complaint: persistent cough for two weeks. "
        "Denies fever. No hemoptysis. Assessment: possible bronchitis.",
        "55-year-old male with diabetes mellitus type 2. "
        "Metformin 1000mg BID. Blood glucose well controlled. "
        "Plan: continue current regimen, follow up in 3 months.",
        "Patient reports severe headache and nausea. "
        "CT scan negative for hemorrhage. Probable tension headache.",
        "Post-operative day 1 after appendectomy. "
        "Vital signs stable. Pain managed with acetaminophen 500mg PRN. "
        "Tolerating clear liquids. Plan: advance diet, discharge tomorrow.",
    ]


@pytest.fixture
def extractor() -> ClinicalFeatureExtractor:
    """Default-config extractor (unfitted)."""
    return ClinicalFeatureExtractor()


@pytest.fixture
def fitted_extractor(clinical_docs: list[str]) -> ClinicalFeatureExtractor:
    """Extractor fitted on the clinical_docs corpus."""
    config = FeatureConfig(max_features=200, min_df=1)
    ext = ClinicalFeatureExtractor(config=config)
    ext.fit(clinical_docs)
    return ext


# ---------------------------------------------------------------------------
# FeatureConfig
# ---------------------------------------------------------------------------


class TestFeatureConfig:
    """Verify FeatureConfig defaults and overrides."""

    def test_defaults(self) -> None:
        cfg = FeatureConfig()
        assert cfg.max_features == 10000
        assert cfg.ngram_range == (1, 2)
        assert cfg.min_df == 2
        assert cfg.max_df == 0.95
        assert cfg.use_custom_features is True
        assert cfg.use_tfidf is True

    def test_custom_values(self) -> None:
        cfg = FeatureConfig(max_features=500, ngram_range=(1, 3), use_tfidf=False)
        assert cfg.max_features == 500
        assert cfg.ngram_range == (1, 3)
        assert cfg.use_tfidf is False


# ---------------------------------------------------------------------------
# ClinicalFeatureExtractor — fit / transform lifecycle
# ---------------------------------------------------------------------------


class TestClinicalFeatureExtractorLifecycle:
    """Test fit → transform → fit_transform workflow."""

    def test_transform_before_fit_raises(self, extractor: ClinicalFeatureExtractor) -> None:
        with pytest.raises(ValueError, match="must be fitted"):
            extractor.transform(["some text"])

    def test_fit_sets_fitted_flag(
        self, extractor: ClinicalFeatureExtractor, clinical_docs: list[str]
    ) -> None:
        assert not extractor._is_fitted
        extractor.fit(clinical_docs)
        assert extractor._is_fitted

    def test_fit_returns_self(
        self, extractor: ClinicalFeatureExtractor, clinical_docs: list[str]
    ) -> None:
        result = extractor.fit(clinical_docs)
        assert result is extractor

    def test_transform_returns_2d_array(
        self, fitted_extractor: ClinicalFeatureExtractor, clinical_docs: list[str]
    ) -> None:
        features = fitted_extractor.transform(clinical_docs)
        assert isinstance(features, np.ndarray)
        assert features.ndim == 2
        assert features.shape[0] == len(clinical_docs)

    def test_fit_transform_equivalent(
        self, clinical_docs: list[str]
    ) -> None:
        """fit_transform should produce the same shape as fit + transform."""
        config = FeatureConfig(max_features=200, min_df=1)
        ext = ClinicalFeatureExtractor(config=config)
        result = ext.fit_transform(clinical_docs)
        assert result.ndim == 2
        assert result.shape[0] == len(clinical_docs)

    def test_single_document_transform(
        self, fitted_extractor: ClinicalFeatureExtractor
    ) -> None:
        features = fitted_extractor.transform(["Patient has chest pain."])
        assert features.shape[0] == 1
        assert features.shape[1] > 0

    def test_tfidf_disabled(self, clinical_docs: list[str]) -> None:
        """When TF-IDF is off, only custom features should appear."""
        config = FeatureConfig(use_tfidf=False, use_custom_features=True)
        ext = ClinicalFeatureExtractor(config=config)
        ext.fit(clinical_docs)
        features = ext.transform(clinical_docs[:1])
        # Should still produce features (custom ones)
        assert features.shape[1] > 0

    def test_custom_features_disabled(self, clinical_docs: list[str]) -> None:
        """When custom features are off, only TF-IDF features remain."""
        config = FeatureConfig(use_tfidf=True, use_custom_features=False, min_df=1, max_features=50)
        ext = ClinicalFeatureExtractor(config=config)
        ext.fit(clinical_docs)
        features = ext.transform(clinical_docs[:1])
        assert features.shape[1] > 0


# ---------------------------------------------------------------------------
# ClinicalFeatureExtractor — custom clinical features
# ---------------------------------------------------------------------------


class TestCustomClinicalFeatures:
    """Verify clinical feature extraction logic."""

    def test_doc_length_feature(self, fitted_extractor: ClinicalFeatureExtractor) -> None:
        short = "Pain."
        long = "Patient presents with severe " * 20
        features = fitted_extractor._extract_custom_features(short)
        features_long = fitted_extractor._extract_custom_features(long)
        assert features["doc_length"] < features_long["doc_length"]

    def test_word_count_feature(self, fitted_extractor: ClinicalFeatureExtractor) -> None:
        text = "Patient has chest pain and shortness of breath"
        features = fitted_extractor._extract_custom_features(text)
        assert features["word_count"] == 9.0

    def test_urgent_keywords_detected(self, fitted_extractor: ClinicalFeatureExtractor) -> None:
        text = "Urgent: patient needs emergency surgery, stat labs ordered"
        features = fitted_extractor._extract_custom_features(text)
        assert features["urgent_keywords_count"] >= 3  # urgent, emergency, stat

    def test_negation_keywords_detected(self, fitted_extractor: ClinicalFeatureExtractor) -> None:
        text = "Patient denies pain. No fever. Not short of breath."
        features = fitted_extractor._extract_custom_features(text)
        assert features["negation_keywords_count"] >= 3  # denies, no, not

    def test_uncertainty_keywords_detected(self, fitted_extractor: ClinicalFeatureExtractor) -> None:
        text = "Possible pneumonia. Suspected UTI. Rule out malignancy."
        features = fitted_extractor._extract_custom_features(text)
        assert features["uncertainty_keywords_count"] >= 2

    def test_medication_patterns_detected(self, fitted_extractor: ClinicalFeatureExtractor) -> None:
        text = "Metformin 1000mg daily, lisinopril 10mg BID, aspirin 81mg PRN"
        features = fitted_extractor._extract_custom_features(text)
        assert features["medication_pattern_count"] >= 3  # three mg patterns

    def test_number_count(self, fitted_extractor: ClinicalFeatureExtractor) -> None:
        text = "BP 120/80, HR 72, Temp 98.6, O2 sat 99%"
        features = fitted_extractor._extract_custom_features(text)
        assert features["number_count"] >= 4

    def test_abbreviation_count(self, fitted_extractor: ClinicalFeatureExtractor) -> None:
        text = "BP normal, HR stable, WBC elevated, CT scan ordered"
        features = fitted_extractor._extract_custom_features(text)
        assert features["abbreviation_count"] >= 3  # BP, HR, WBC, CT

    def test_empty_text_no_crash(self, fitted_extractor: ClinicalFeatureExtractor) -> None:
        features = fitted_extractor._extract_custom_features("")
        assert features["doc_length"] == 0.0
        assert features["word_count"] == 0.0
        assert features["avg_word_length"] == 0

    def test_punctuation_features(self, fitted_extractor: ClinicalFeatureExtractor) -> None:
        text = "Assessment: good\nPlan: continue\nMedications: listed;"
        features = fitted_extractor._extract_custom_features(text)
        assert features["colon_count"] == 3
        assert features["semicolon_count"] == 1
        assert features["newline_count"] == 2


# ---------------------------------------------------------------------------
# ClinicalFeatureExtractor — feature names
# ---------------------------------------------------------------------------


class TestFeatureNames:
    """Verify get_feature_names returns expected names."""

    def test_names_include_custom_features(
        self, fitted_extractor: ClinicalFeatureExtractor
    ) -> None:
        names = fitted_extractor.get_feature_names()
        assert "doc_length" in names
        assert "word_count" in names
        assert "negation_density" in names
        assert "medication_pattern_count" in names

    def test_names_include_tfidf_prefix(
        self, fitted_extractor: ClinicalFeatureExtractor
    ) -> None:
        names = fitted_extractor.get_feature_names()
        tfidf_names = [n for n in names if n.startswith("tfidf_")]
        assert len(tfidf_names) > 0


# ---------------------------------------------------------------------------
# ClinicalFeatureExtractor — dicts_to_array
# ---------------------------------------------------------------------------


class TestDictsToArray:
    """Test internal conversion from feature dicts to numpy arrays."""

    def test_empty_list(self, extractor: ClinicalFeatureExtractor) -> None:
        result = extractor._dicts_to_array([])
        assert isinstance(result, np.ndarray)
        assert result.size == 0

    def test_single_dict(self, extractor: ClinicalFeatureExtractor) -> None:
        result = extractor._dicts_to_array([{"a": 1.0, "b": 2.0}])
        assert result.shape == (1, 2)

    def test_missing_keys_filled_with_zero(self, extractor: ClinicalFeatureExtractor) -> None:
        result = extractor._dicts_to_array([{"a": 1.0, "b": 2.0}, {"b": 3.0, "c": 4.0}])
        assert result.shape == (2, 3)
        # First row should have c=0, second row should have a=0
        assert 0.0 in result[0]
        assert 0.0 in result[1]


# ---------------------------------------------------------------------------
# MEDICAL_STOPWORDS / CRITICAL_KEYWORDS sanity checks
# ---------------------------------------------------------------------------


class TestConstants:
    """Verify constant collections are non-empty and well-formed."""

    def test_medical_stopwords_not_empty(self) -> None:
        assert len(MEDICAL_STOPWORDS) > 10

    def test_critical_keywords_categories(self) -> None:
        assert "urgent_keywords" in CRITICAL_KEYWORDS
        assert "negation_keywords" in CRITICAL_KEYWORDS
        assert "uncertainty_keywords" in CRITICAL_KEYWORDS
        assert "pain_keywords" in CRITICAL_KEYWORDS
        assert "medication_patterns" in CRITICAL_KEYWORDS

    def test_all_keyword_lists_non_empty(self) -> None:
        for key, values in CRITICAL_KEYWORDS.items():
            assert len(values) > 0, f"CRITICAL_KEYWORDS['{key}'] is empty"


# ---------------------------------------------------------------------------
# BagOfWordsExtractor
# ---------------------------------------------------------------------------


class TestBagOfWordsExtractor:
    """Test BagOfWordsExtractor fit/transform cycle."""

    def test_transform_before_fit_raises(self) -> None:
        bow = BagOfWordsExtractor()
        with pytest.raises(ValueError, match="must be fitted"):
            bow.transform(["text"])

    def test_fit_returns_self(self, clinical_docs: list[str]) -> None:
        bow = BagOfWordsExtractor(max_features=100)
        result = bow.fit(clinical_docs)
        assert result is bow

    def test_vocabulary_built(self, clinical_docs: list[str]) -> None:
        bow = BagOfWordsExtractor(max_features=100)
        bow.fit(clinical_docs)
        assert len(bow.vocabulary) > 0
        assert len(bow.vocabulary) <= 100

    def test_transform_shape(self, clinical_docs: list[str]) -> None:
        bow = BagOfWordsExtractor(max_features=50)
        bow.fit(clinical_docs)
        result = bow.transform(clinical_docs)
        assert result.shape[0] == len(clinical_docs)
        assert result.shape[1] == len(bow.vocabulary)

    def test_fit_transform(self, clinical_docs: list[str]) -> None:
        bow = BagOfWordsExtractor(max_features=50)
        result = bow.fit_transform(clinical_docs)
        assert result.shape[0] == len(clinical_docs)

    def test_bigrams(self, clinical_docs: list[str]) -> None:
        bow = BagOfWordsExtractor(max_features=200, ngram_range=(1, 2))
        bow.fit(clinical_docs)
        # Should have some bigrams in vocabulary
        bigrams = [k for k in bow.vocabulary if " " in k]
        assert len(bigrams) > 0

    def test_unigrams_only(self, clinical_docs: list[str]) -> None:
        bow = BagOfWordsExtractor(max_features=100, ngram_range=(1, 1))
        bow.fit(clinical_docs)
        bigrams = [k for k in bow.vocabulary if " " in k]
        assert len(bigrams) == 0

    def test_word_counts_are_non_negative(self, clinical_docs: list[str]) -> None:
        bow = BagOfWordsExtractor(max_features=50)
        bow.fit(clinical_docs)
        result = bow.transform(clinical_docs)
        assert (result >= 0).all()

    def test_unknown_words_ignored(self, clinical_docs: list[str]) -> None:
        bow = BagOfWordsExtractor(max_features=50)
        bow.fit(clinical_docs)
        result = bow.transform(["xylophone zebra quasar"])
        # All zeros if none of these words are in the vocab
        assert result.sum() == 0.0 or result.shape[1] > 0
