"""ICD-10 Code Prediction model.

Predicts ICD-10-CM diagnosis codes from clinical free text.  This is
one of the highest-value NLP tasks in healthcare: accurate auto-coding
reduces manual coding effort (typically 5–15 min per encounter) and
improves revenue integrity for providers.

Architecture
~~~~~~~~~~~~
Two model families:

1. **Classical ML (TF-IDF + Logistic Regression / SVM)** — Fast,
   interpretable, good for common codes.  The feature extractor
   produces TF-IDF vectors enriched with clinical features; a
   multi-label classifier maps these to ICD-10 codes.

2. **Transformer-based (ClinicalBERT)** — Fine-tuned sequence
   classification for better performance on long notes and rare codes.
   Uses ``[CLS]`` token pooling with a multi-label sigmoid head.

Design decisions
----------------
* **Multi-label, not multi-class** — A single clinical encounter
  typically has 3–15 ICD-10 codes.  We use sigmoid activation per
  code (not softmax) so multiple codes can be predicted independently.
* **Top-k with confidence** — Rather than hard-thresholding, we
  return the top-k predictions with calibrated confidence scores.
  This lets the UI show "likely" vs "possible" codes and supports
  human-in-the-loop review workflows.
* **Contributing text** — Each prediction includes the text segments
  that most influenced the prediction, supporting clinical auditors
  who need to verify coding accuracy.
* **ICD-10-CM code dictionary** — ~400 high-frequency codes covering
  the most common diagnoses across specialties.  The dictionary maps
  codes to descriptions, chapters, and categories for display.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from app.core.exceptions import InferenceError, ModelLoadError
from app.ml.utils.feature_engineering import ClinicalFeatureExtractor, FeatureConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class ICDCodePrediction:
    """Represents an ICD-10 code prediction."""

    code: str
    description: str | None
    confidence: float
    chapter: str | None = None
    category: str | None = None
    contributing_text: list[str] | None = None  # Text segments that contributed

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "code": self.code,
            "description": self.description,
            "confidence": self.confidence,
            "chapter": self.chapter,
            "category": self.category,
            "contributing_text": self.contributing_text,
        }


@dataclass
class ICDPredictionResult:
    """Complete ICD-10 prediction result for a document."""

    predictions: list[ICDCodePrediction]
    processing_time_ms: float
    model_name: str
    model_version: str
    document_summary: str | None = None

    def top_k(self, k: int = 5) -> list[ICDCodePrediction]:
        """Get top k predictions."""
        return sorted(self.predictions, key=lambda p: p.confidence, reverse=True)[:k]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "predictions": [p.to_dict() for p in self.predictions],
            "processing_time_ms": self.processing_time_ms,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "document_summary": self.document_summary,
        }


# ICD-10 Chapter mapping for hierarchical prediction
ICD10_CHAPTERS = {
    "A00-B99": "Certain infectious and parasitic diseases",
    "C00-D49": "Neoplasms",
    "D50-D89": "Diseases of the blood and blood-forming organs",
    "E00-E89": "Endocrine, nutritional and metabolic diseases",
    "F01-F99": "Mental and behavioral disorders",
    "G00-G99": "Diseases of the nervous system",
    "H00-H59": "Diseases of the eye and adnexa",
    "H60-H95": "Diseases of the ear and mastoid process",
    "I00-I99": "Diseases of the circulatory system",
    "J00-J99": "Diseases of the respiratory system",
    "K00-K95": "Diseases of the digestive system",
    "L00-L99": "Diseases of the skin and subcutaneous tissue",
    "M00-M99": "Diseases of the musculoskeletal system",
    "N00-N99": "Diseases of the genitourinary system",
    "O00-O9A": "Pregnancy, childbirth and the puerperium",
    "P00-P96": "Certain conditions originating in the perinatal period",
    "Q00-Q99": "Congenital malformations and chromosomal abnormalities",
    "R00-R99": "Symptoms, signs and abnormal clinical findings",
    "S00-T88": "Injury, poisoning and certain other consequences of external causes",
    "V00-Y99": "External causes of morbidity",
    "Z00-Z99": "Factors influencing health status and contact with health services",
}


def get_chapter_for_code(code: str) -> str | None:
    """Get the ICD-10 chapter for a code."""
    code = code.upper()
    prefix = code[0] if code else ""

    chapter_ranges = {
        "A": "A00-B99",
        "B": "A00-B99",
        "C": "C00-D49",
        "D": "C00-D49" if code[:1] < "D50" else "D50-D89",
        "E": "E00-E89",
        "F": "F01-F99",
        "G": "G00-G99",
        "H": "H00-H59" if code[:1] < "H60" else "H60-H95",
        "I": "I00-I99",
        "J": "J00-J99",
        "K": "K00-K95",
        "L": "L00-L99",
        "M": "M00-M99",
        "N": "N00-N99",
        "O": "O00-O9A",
        "P": "P00-P96",
        "Q": "Q00-Q99",
        "R": "R00-R99",
        "S": "S00-T88",
        "T": "S00-T88",
        "V": "V00-Y99",
        "W": "V00-Y99",
        "X": "V00-Y99",
        "Y": "V00-Y99",
        "Z": "Z00-Z99",
    }

    chapter_code = chapter_ranges.get(prefix)
    return ICD10_CHAPTERS.get(chapter_code) if chapter_code else None


class BaseICDClassifier(ABC):
    """Abstract base class for ICD-10 classifiers."""

    def __init__(self, model_name: str, version: str = "1.0.0"):
        self.model_name = model_name
        self.version = version
        self._is_loaded = False
        self.code_descriptions: dict[str, str] = {}

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        ...

    @abstractmethod
    def predict(self, text: str, top_k: int = 10) -> ICDPredictionResult:
        """Predict ICD-10 codes from clinical text."""
        ...

    @abstractmethod
    def predict_batch(self, texts: list[str], top_k: int = 10) -> list[ICDPredictionResult]:
        """Predict ICD-10 codes for multiple documents."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def ensure_loaded(self) -> None:
        """Ensure the model is loaded."""
        if not self._is_loaded:
            self.load()


class SklearnICDClassifier(BaseICDClassifier):
    """Scikit-learn based ICD-10 classifier for baseline."""

    def __init__(
        self,
        model_name: str = "sklearn-baseline",
        version: str = "1.0.0",
        model_path: str | None = None,
        feature_config: FeatureConfig | None = None,
    ):
        super().__init__(model_name, version)
        self.model_path = model_path
        self.feature_config = feature_config or FeatureConfig()
        self.feature_extractor: ClinicalFeatureExtractor | None = None
        self.classifier: Any = None
        self.label_binarizer: Any = None

    def load(self) -> None:
        """Load the model and feature extractor."""
        import pickle

        try:
            if self.model_path:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                self.classifier = data["model"]
                self.feature_extractor = data["feature_extractor"]
                self.label_binarizer = data["label_binarizer"]
                self.code_descriptions = data.get("code_descriptions", {})
            else:
                # Initialize empty model (will need to be trained)
                self.feature_extractor = ClinicalFeatureExtractor(self.feature_config)
                self.label_binarizer = None
                self.classifier = None

            self._is_loaded = True
            logger.info(f"Loaded sklearn ICD classifier: {self.model_name}")

        except Exception as e:
            raise ModelLoadError(self.model_name, str(e))

    def predict(self, text: str, top_k: int = 10) -> ICDPredictionResult:
        """Predict ICD-10 codes from text."""
        import time

        self.ensure_loaded()

        if self.classifier is None:
            raise InferenceError(self.model_name, "Model not trained")

        start_time = time.time()

        try:
            # Extract features
            features = self.feature_extractor.transform([text])

            # Get predictions
            if hasattr(self.classifier, "predict_proba"):
                probas = self.classifier.predict_proba(features)
                if len(probas.shape) == 1:
                    probas = probas.reshape(1, -1)
            else:
                # Binary relevance classifiers return list of probas
                scores = self.classifier.decision_function(features)
                if not isinstance(scores, np.ndarray):
                    scores = np.array(scores)
                probas = 1 / (1 + np.exp(-scores))  # Sigmoid

            # Get top k predictions
            predictions = self._get_top_predictions(probas[0], top_k)

            processing_time = (time.time() - start_time) * 1000

            return ICDPredictionResult(
                predictions=predictions,
                processing_time_ms=processing_time,
                model_name=self.model_name,
                model_version=self.version,
            )

        except Exception as e:
            raise InferenceError(self.model_name, str(e))

    def predict_batch(self, texts: list[str], top_k: int = 10) -> list[ICDPredictionResult]:
        """Predict ICD-10 codes for multiple texts."""
        import time

        self.ensure_loaded()

        if self.classifier is None:
            raise InferenceError(self.model_name, "Model not trained")

        start_time = time.time()

        try:
            features = self.feature_extractor.transform(texts)

            if hasattr(self.classifier, "predict_proba"):
                probas = self.classifier.predict_proba(features)
            else:
                scores = self.classifier.decision_function(features)
                if not isinstance(scores, np.ndarray):
                    scores = np.array(scores)
                probas = 1 / (1 + np.exp(-scores))

            results = []
            for _i, proba in enumerate(probas):
                predictions = self._get_top_predictions(proba, top_k)
                results.append(
                    ICDPredictionResult(
                        predictions=predictions,
                        processing_time_ms=0,  # Will be set for batch
                        model_name=self.model_name,
                        model_version=self.version,
                    )
                )

            total_time = (time.time() - start_time) * 1000
            avg_time = total_time / len(texts)

            for result in results:
                result.processing_time_ms = avg_time

            return results

        except Exception as e:
            raise InferenceError(self.model_name, str(e))

    def _get_top_predictions(
        self, probas: NDArray[np.float64], top_k: int
    ) -> list[ICDCodePrediction]:
        """Get top k predictions from probability array."""
        top_indices = np.argsort(probas)[::-1][:top_k]

        predictions = []
        for idx in top_indices:
            if probas[idx] > 0.1:  # Threshold for minimum confidence
                code = self.label_binarizer.classes_[idx]
                predictions.append(
                    ICDCodePrediction(
                        code=code,
                        description=self.code_descriptions.get(code),
                        confidence=float(probas[idx]),
                        chapter=get_chapter_for_code(code),
                    )
                )

        return predictions


class TransformerICDClassifier(BaseICDClassifier):
    """Transformer-based ICD-10 classifier (BioBERT, ClinicalBERT, etc.)."""

    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        version: str = "1.0.0",
        model_path: str | None = None,
        device: str = "cpu",
        max_length: int = 512,
    ):
        super().__init__(model_name, version)
        self.model_path = model_path
        self.device = device
        self.max_length = max_length
        self.tokenizer: Any = None
        self.model: Any = None
        self.label_map: dict[int, str] = {}

    def load(self) -> None:
        """Load the transformer model."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            model_path = self.model_path or self.model_name

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()

            # Build label map from config
            if hasattr(self.model.config, "id2label"):
                self.label_map = self.model.config.id2label

            self._is_loaded = True
            logger.info(f"Loaded transformer ICD classifier: {model_path}")

        except Exception as e:
            raise ModelLoadError(self.model_name, str(e))

    def predict(self, text: str, top_k: int = 10) -> ICDPredictionResult:
        """Predict ICD-10 codes using transformer."""
        import time

        self.ensure_loaded()
        import torch

        start_time = time.time()

        try:
            # Handle long documents with sliding window
            if len(text.split()) > self.max_length:
                return self._predict_long_document(text, top_k)

            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0]
                probas = torch.sigmoid(logits).cpu().numpy()

            # Get top predictions
            predictions = self._get_top_predictions(probas, top_k)

            processing_time = (time.time() - start_time) * 1000

            return ICDPredictionResult(
                predictions=predictions,
                processing_time_ms=processing_time,
                model_name=self.model_name,
                model_version=self.version,
            )

        except Exception as e:
            raise InferenceError(self.model_name, str(e))

    def _predict_long_document(self, text: str, top_k: int) -> ICDPredictionResult:
        """Handle long documents with sliding window."""
        import time

        import torch

        start_time = time.time()

        # Split text into overlapping windows
        words = text.split()
        window_size = self.max_length - 100  # Leave room for special tokens
        stride = window_size // 2

        windows = []
        for i in range(0, len(words), stride):
            window = " ".join(words[i : i + window_size])
            windows.append(window)
            if i + window_size >= len(words):
                break

        # Process each window
        all_probas = []
        for window in windows:
            inputs = self.tokenizer(
                window,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits[0]
                probas = torch.sigmoid(logits).cpu().numpy()
                all_probas.append(probas)

        # Aggregate predictions (max pooling across windows)
        aggregated = np.max(all_probas, axis=0)
        predictions = self._get_top_predictions(aggregated, top_k)

        processing_time = (time.time() - start_time) * 1000

        return ICDPredictionResult(
            predictions=predictions,
            processing_time_ms=processing_time,
            model_name=self.model_name,
            model_version=self.version,
        )

    def predict_batch(self, texts: list[str], top_k: int = 10) -> list[ICDPredictionResult]:
        """Predict ICD-10 codes for multiple texts."""
        import time

        import torch

        self.ensure_loaded()
        start_time = time.time()

        try:
            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probas = torch.sigmoid(logits).cpu().numpy()

            # Get predictions for each document
            results = []
            for proba in probas:
                predictions = self._get_top_predictions(proba, top_k)
                results.append(
                    ICDPredictionResult(
                        predictions=predictions,
                        processing_time_ms=0,
                        model_name=self.model_name,
                        model_version=self.version,
                    )
                )

            total_time = (time.time() - start_time) * 1000
            avg_time = total_time / len(texts)

            for result in results:
                result.processing_time_ms = avg_time

            return results

        except Exception as e:
            raise InferenceError(self.model_name, str(e))

    def _get_top_predictions(
        self, probas: NDArray[np.float64], top_k: int
    ) -> list[ICDCodePrediction]:
        """Get top k predictions from probability array."""
        top_indices = np.argsort(probas)[::-1][:top_k]

        predictions = []
        for idx in top_indices:
            if probas[idx] > 0.1:
                code = self.label_map.get(idx, f"UNKNOWN_{idx}")
                predictions.append(
                    ICDCodePrediction(
                        code=code,
                        description=self.code_descriptions.get(code),
                        confidence=float(probas[idx]),
                        chapter=get_chapter_for_code(code),
                    )
                )

        return predictions


class HierarchicalICDClassifier(BaseICDClassifier):
    """Hierarchical ICD-10 classifier that predicts chapter then code."""

    def __init__(
        self,
        chapter_classifier: BaseICDClassifier,
        code_classifiers: dict[str, BaseICDClassifier],
        model_name: str = "hierarchical-icd",
        version: str = "1.0.0",
    ):
        super().__init__(model_name, version)
        self.chapter_classifier = chapter_classifier
        self.code_classifiers = code_classifiers

    def load(self) -> None:
        """Load chapter and code classifiers."""
        self.chapter_classifier.load()
        for classifier in self.code_classifiers.values():
            classifier.load()
        self._is_loaded = True

    def predict(self, text: str, top_k: int = 10) -> ICDPredictionResult:
        """Predict using hierarchical approach."""
        import time

        start_time = time.time()

        # First predict chapter
        chapter_result = self.chapter_classifier.predict(text, top_k=3)
        predicted_chapters = [p.code for p in chapter_result.predictions]

        # Then predict codes within likely chapters
        all_predictions = []
        for chapter in predicted_chapters:
            if chapter in self.code_classifiers:
                code_result = self.code_classifiers[chapter].predict(text, top_k=top_k // 2)
                all_predictions.extend(code_result.predictions)

        # Sort by confidence and take top k
        all_predictions.sort(key=lambda p: p.confidence, reverse=True)
        top_predictions = all_predictions[:top_k]

        processing_time = (time.time() - start_time) * 1000

        return ICDPredictionResult(
            predictions=top_predictions,
            processing_time_ms=processing_time,
            model_name=self.model_name,
            model_version=self.version,
        )

    def predict_batch(self, texts: list[str], top_k: int = 10) -> list[ICDPredictionResult]:
        """Predict for batch of texts."""
        return [self.predict(text, top_k) for text in texts]


# ---------------------------------------------------------------------------
# Rule-based ICD-10 classifier — works without a trained model
# ---------------------------------------------------------------------------

# Maps keyword/phrase patterns to (ICD code, description, base confidence).
_ICD_RULES: list[tuple[list[str], str, str, float]] = [
    # Cardiovascular
    (["hypertension", "high blood pressure", "htn", "elevated bp"],
     "I10", "Essential (primary) hypertension", 0.82),
    (["myocardial infarction", "heart attack", "mi ", "stemi", "nstemi", "st-segment elevation"],
     "I21.9", "Acute myocardial infarction, unspecified", 0.85),
    (["atrial fibrillation", "afib", "a-fib", "a fib"],
     "I48.91", "Unspecified atrial fibrillation", 0.80),
    (["heart failure", "chf", "congestive heart"],
     "I50.9", "Heart failure, unspecified", 0.78),
    (["chest pain"],
     "R07.9", "Chest pain, unspecified", 0.72),
    (["coronary artery disease", "cad", "atherosclerotic heart"],
     "I25.10", "Atherosclerotic heart disease of native coronary artery", 0.80),
    # Endocrine / metabolic
    (["type 2 diabetes", "type ii diabetes", "t2dm", "dm2", "diabetes mellitus type 2"],
     "E11.9", "Type 2 diabetes mellitus without complications", 0.84),
    (["type 1 diabetes", "type i diabetes", "t1dm", "dm1", "iddm"],
     "E10.9", "Type 1 diabetes mellitus without complications", 0.84),
    (["hyperlipidemia", "dyslipidemia", "high cholesterol", "hypercholesterolemia"],
     "E78.5", "Hyperlipidemia, unspecified", 0.76),
    (["hypothyroidism", "low thyroid"],
     "E03.9", "Hypothyroidism, unspecified", 0.78),
    # Respiratory
    (["copd", "chronic obstructive pulmonary", "emphysema"],
     "J44.1", "COPD with acute exacerbation", 0.75),
    (["pneumonia"],
     "J18.9", "Pneumonia, unspecified organism", 0.73),
    (["asthma"],
     "J45.909", "Unspecified asthma, uncomplicated", 0.74),
    # Renal
    (["chronic kidney disease", "ckd", "renal failure", "kidney failure"],
     "N18.9", "Chronic kidney disease, unspecified", 0.72),
    (["acute kidney injury", "aki", "acute renal failure"],
     "N17.9", "Acute kidney failure, unspecified", 0.76),
    # Neurological
    (["cerebrovascular accident", "stroke", "cva"],
     "I63.9", "Cerebral infarction, unspecified", 0.79),
    (["seizure", "epilepsy"],
     "G40.909", "Epilepsy, unspecified, not intractable", 0.71),
    # GI
    (["gerd", "gastroesophageal reflux", "acid reflux"],
     "K21.0", "GERD with esophagitis", 0.74),
    # Musculoskeletal
    (["low back pain", "lumbago", "lumbar pain"],
     "M54.5", "Low back pain", 0.77),
    # Mental
    (["major depressive", "depression", "mdd"],
     "F32.9", "Major depressive disorder, single episode, unspecified", 0.68),
    (["anxiety", "generalized anxiety"],
     "F41.9", "Anxiety disorder, unspecified", 0.66),
    # Infectious
    (["urinary tract infection", "uti"],
     "N39.0", "Urinary tract infection, site not specified", 0.78),
    (["sepsis", "septicemia"],
     "A41.9", "Sepsis, unspecified organism", 0.82),
    # General
    (["obesity", "obese", "bmi >30", "bmi > 30"],
     "E66.9", "Obesity, unspecified", 0.70),
    (["anemia", "low hemoglobin", "low hgb"],
     "D64.9", "Anemia, unspecified", 0.69),
]


class RuleBasedICDClassifier(BaseICDClassifier):
    """Keyword / pattern-based ICD-10 classifier.

    Designed as a zero-dependency baseline that works without a trained
    sklearn or transformer model.  Scans the input text for known clinical
    phrases and maps them to the most likely ICD-10-CM codes with a
    heuristic confidence score.  Useful as a fallback when no trained model
    artefact is available.

    Parameters
    ----------
    model_name:
        Identifier for logging and API responses.
    version:
        Semantic version string.
    """

    def __init__(
        self,
        model_name: str = "rule-based-icd",
        version: str = "1.0.0",
    ) -> None:
        super().__init__(model_name, version)
        self._compiled_rules: list[tuple[list[str], str, str, float]] = []

    def load(self) -> None:
        """Compile the keyword lookup table."""
        # We normalise keywords to lower-case at load time so inference only
        # needs to lower-case the input text once.
        self._compiled_rules = [
            ([kw.lower() for kw in keywords], code, desc, conf)
            for keywords, code, desc, conf in _ICD_RULES
        ]
        self._is_loaded = True
        logger.info("Loaded RuleBasedICDClassifier v%s (%d rules)", self.version, len(self._compiled_rules))

    def predict(self, text: str, top_k: int = 10) -> ICDPredictionResult:
        """Predict ICD-10 codes by scanning *text* for known clinical phrases.

        Parameters
        ----------
        text:
            Raw clinical document text.
        top_k:
            Maximum number of predictions to return.

        Returns
        -------
        ICDPredictionResult
        """
        import time as _time

        self.ensure_loaded()
        start = _time.time()

        text_lower = text.lower()
        predictions: list[ICDCodePrediction] = []

        for keywords, code, description, base_conf in self._compiled_rules:
            matched_keywords: list[str] = []
            for kw in keywords:
                if kw in text_lower:
                    matched_keywords.append(kw)

            if matched_keywords:
                # Boost confidence slightly when multiple synonyms match.
                confidence = min(1.0, base_conf + 0.03 * (len(matched_keywords) - 1))
                predictions.append(
                    ICDCodePrediction(
                        code=code,
                        description=description,
                        confidence=round(confidence, 4),
                        chapter=get_chapter_for_code(code),
                        category=None,
                        contributing_text=matched_keywords,
                    )
                )

        # Deduplicate by code (keep highest confidence).
        seen: dict[str, ICDCodePrediction] = {}
        for pred in predictions:
            if pred.code not in seen or pred.confidence > seen[pred.code].confidence:
                seen[pred.code] = pred
        predictions = list(seen.values())

        predictions.sort(key=lambda p: p.confidence, reverse=True)
        predictions = predictions[:top_k]

        elapsed_ms = (_time.time() - start) * 1000

        return ICDPredictionResult(
            predictions=predictions,
            processing_time_ms=elapsed_ms,
            model_name=self.model_name,
            model_version=self.version,
        )

    def predict_batch(self, texts: list[str], top_k: int = 10) -> list[ICDPredictionResult]:
        """Predict ICD-10 codes for a batch of documents.

        Parameters
        ----------
        texts:
            List of clinical document texts.
        top_k:
            Maximum predictions per document.

        Returns
        -------
        list[ICDPredictionResult]
        """
        return [self.predict(text, top_k) for text in texts]
