"""Feature engineering for clinical text classification."""

import re
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from app.ml.utils.text_preprocessing import ClinicalTextPreprocessor

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    max_features: int = 10000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95
    use_custom_features: bool = True
    use_tfidf: bool = True


# Medical-specific stopwords (common words that don't help classification)
MEDICAL_STOPWORDS = {
    "patient",
    "patients",
    "history",
    "noted",
    "noted:",
    "reports",
    "states",
    "also",
    "without",
    "denies",
    "reports:",
    "present",
    "time",
    "today",
    "year",
    "years",
    "day",
    "days",
    "week",
    "weeks",
    "month",
    "months",
    "right",
    "left",
    "bilaterally",
    "following",
    "continued",
    "continue",
    "advised",
    "instructed",
}

# Critical medical keywords for various features
CRITICAL_KEYWORDS = {
    "urgent_keywords": [
        "urgent",
        "emergent",
        "stat",
        "immediate",
        "emergency",
        "critical",
        "acute",
    ],
    "negation_keywords": [
        "no",
        "not",
        "denies",
        "denied",
        "without",
        "negative",
        "absent",
        "none",
        "never",
    ],
    "uncertainty_keywords": [
        "possible",
        "probable",
        "likely",
        "suspected",
        "concern for",
        "rule out",
        "consider",
        "may have",
        "might",
        "perhaps",
        "?",
    ],
    "pain_keywords": [
        "pain",
        "ache",
        "tenderness",
        "discomfort",
        "hurt",
        "sore",
        "cramp",
        "spasm",
    ],
    "medication_patterns": [
        r"\b\d+\s*mg\b",
        r"\b\d+\s*ml\b",
        r"\b\d+\s*mcg\b",
        r"\b\d+\s*units?\b",
        r"\b(daily|bid|tid|qid|prn|qhs|weekly)\b",
    ],
}


class ClinicalFeatureExtractor:
    """Extract features from clinical text for ML models."""

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()
        self.preprocessor = ClinicalTextPreprocessor()
        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self._is_fitted = False

    def fit(self, documents: list[str]) -> "ClinicalFeatureExtractor":
        """Fit the TF-IDF vectorizer on training documents."""
        if self.config.use_tfidf:
            preprocessed = [self.preprocessor.preprocess(doc) for doc in documents]

            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.config.max_features,
                ngram_range=self.config.ngram_range,
                min_df=self.config.min_df,
                max_df=self.config.max_df,
                stop_words=list(MEDICAL_STOPWORDS),
                lowercase=True,
                sublinear_tf=True,
            )
            self.tfidf_vectorizer.fit(preprocessed)

        self._is_fitted = True
        return self

    def transform(self, documents: list[str]) -> "NDArray[np.float64]":
        """Transform documents to feature vectors."""
        if not self._is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")

        features_list = []

        for doc in documents:
            preprocessed = self.preprocessor.preprocess(doc)
            features = {}

            # TF-IDF features
            if self.config.use_tfidf and self.tfidf_vectorizer is not None:
                tfidf_features = self.tfidf_vectorizer.transform([preprocessed]).toarray()[0]
                for idx, val in enumerate(tfidf_features):
                    if val > 0:
                        features[f"tfidf_{idx}"] = float(val)

            # Custom features
            if self.config.use_custom_features:
                custom = self._extract_custom_features(preprocessed)
                features.update(custom)

            features_list.append(features)

        # Convert to dense array
        return self._dicts_to_array(features_list)

    def fit_transform(self, documents: list[str]) -> "NDArray[np.float64]":
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform(documents)

    def _extract_custom_features(self, text: str) -> dict[str, float]:
        """Extract custom clinical features from text."""
        features = {}
        text_lower = text.lower()

        # Document structure features
        features["doc_length"] = len(text)
        features["word_count"] = len(text.split())
        features["avg_word_length"] = np.mean([len(w) for w in text.split()]) if text.split() else 0

        # Section features
        sections = self.preprocessor.detect_sections(text)
        section_names = {s.name for s in sections}
        for section in ["chief_complaint", "hpi", "pmh", "assessment", "plan", "pe"]:
            features[f"has_{section}"] = float(section in section_names)

        # Keyword presence features
        for keyword_type, keywords in CRITICAL_KEYWORDS.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            features[f"{keyword_type}_count"] = count
            features[f"{keyword_type}_ratio"] = count / max(len(text.split()), 1) * 100

        # Medical entity density features
        features["number_count"] = len(re.findall(r"\b\d+(?:\.\d+)?\b", text))
        features["abbreviation_count"] = len(
            re.findall(r"\b[A-Z]{2,5}\b", text)
        )  # 2-5 uppercase letters

        # Negation and uncertainty patterns
        negation_count = sum(1 for kw in CRITICAL_KEYWORDS["negation_keywords"] if kw in text_lower)
        uncertainty_count = sum(
            1 for kw in CRITICAL_KEYWORDS["uncertainty_keywords"] if kw in text_lower
        )
        features["negation_density"] = negation_count / max(len(text.split()), 1) * 100
        features["uncertainty_density"] = uncertainty_count / max(len(text.split()), 1) * 100

        # Medication patterns
        med_matches = 0
        for pattern in CRITICAL_KEYWORDS["medication_patterns"]:
            med_matches += len(re.findall(pattern, text_lower))
        features["medication_pattern_count"] = med_matches

        # Punctuation features (can indicate clinical note style)
        features["colon_count"] = text.count(":")
        features["semicolon_count"] = text.count(";")
        features["newline_count"] = text.count("\n")

        return features

    def _dicts_to_array(self, features_list: list[dict]) -> "NDArray[np.float64]":
        """Convert list of feature dicts to numpy array."""
        if not features_list:
            return np.array([])

        # Get all feature names
        all_features = set()
        for features in features_list:
            all_features.update(features.keys())
        all_features = sorted(all_features)

        # Build array
        result = np.zeros((len(features_list), len(all_features)), dtype=np.float64)
        for i, features in enumerate(features_list):
            for j, feature_name in enumerate(all_features):
                result[i, j] = features.get(feature_name, 0.0)

        return result

    def get_feature_names(self) -> list[str]:
        """Get names of all features."""
        names = []

        if self.tfidf_vectorizer is not None:
            names.extend(f"tfidf_{name}" for name in self.tfidf_vectorizer.get_feature_names_out())

        # Custom feature names
        custom_features = [
            "doc_length",
            "word_count",
            "avg_word_length",
            "has_chief_complaint",
            "has_hpi",
            "has_pmh",
            "has_assessment",
            "has_plan",
            "has_pe",
            "urgent_keywords_count",
            "urgent_keywords_ratio",
            "negation_keywords_count",
            "negation_keywords_ratio",
            "uncertainty_keywords_count",
            "uncertainty_keywords_ratio",
            "pain_keywords_count",
            "pain_keywords_ratio",
            "medication_patterns_count",
            "medication_patterns_ratio",
            "number_count",
            "abbreviation_count",
            "negation_density",
            "uncertainty_density",
            "medication_pattern_count",
            "colon_count",
            "semicolon_count",
            "newline_count",
        ]
        names.extend(custom_features)

        return names


class BagOfWordsExtractor:
    """Simple bag of words extractor for baseline models."""

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple[int, int] = (1, 1),
    ):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vocabulary: dict[str, int] = {}
        self._is_fitted = False

    def fit(self, documents: list[str]) -> "BagOfWordsExtractor":
        """Build vocabulary from documents."""
        word_counts: Counter[str] = Counter()
        n = self.ngram_range[1]

        for doc in documents:
            words = doc.lower().split()
            for i in range(len(words)):
                for j in range(self.ngram_range[0], min(n + 1, len(words) - i + 1)):
                    ngram = " ".join(words[i : i + j])
                    word_counts[ngram] += 1

        # Select top features
        most_common = word_counts.most_common(self.max_features)
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}
        self._is_fitted = True
        return self

    def transform(self, documents: list[str]) -> "NDArray[np.float64]":
        """Transform documents to bag of words vectors."""
        if not self._is_fitted:
            raise ValueError("BagOfWordsExtractor must be fitted before transform")

        result = np.zeros((len(documents), len(self.vocabulary)), dtype=np.float64)
        n = self.ngram_range[1]

        for i, doc in enumerate(documents):
            words = doc.lower().split()
            for j in range(len(words)):
                for k in range(self.ngram_range[0], min(n + 1, len(words) - j + 1)):
                    ngram = " ".join(words[j : j + k])
                    if ngram in self.vocabulary:
                        result[i, self.vocabulary[ngram]] += 1

        return result

    def fit_transform(self, documents: list[str]) -> "NDArray[np.float64]":
        """Fit and transform in one step."""
        self.fit(documents)
        return self.transform(documents)
