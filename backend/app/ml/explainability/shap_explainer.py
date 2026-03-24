"""SHAP-based model explainability for ClinIQ ML predictions.

Provides token-level attribution for ICD-10 code predictions using SHAP
KernelExplainer with TF-IDF features, and attention-weight extraction for
transformer-based models.  All explainers expose a common :class:`BaseExplainer`
interface so higher-level code can be written against a stable API.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class SHAPExplanation:
    """SHAP-based explanation for a single model prediction.

    Attributes
    ----------
    feature_attributions:
        Mapping of feature name (token or TF-IDF n-gram) to its SHAP value.
        Positive values push the prediction higher; negative values push it
        lower.
    base_value:
        Expected model output over the background dataset (SHAP base value).
    predicted_value:
        Actual model output for this input (base_value + sum of attributions).
    top_positive_features:
        List of ``(feature, shap_value)`` tuples for the highest-attribution
        features, sorted descending.
    top_negative_features:
        List of ``(feature, shap_value)`` tuples for the most-negative
        attribution features, sorted ascending.
    processing_time_ms:
        Wall-clock time taken to compute the explanation.
    """

    feature_attributions: dict[str, float]
    base_value: float
    predicted_value: float
    top_positive_features: list[tuple[str, float]] = field(default_factory=list)
    top_negative_features: list[tuple[str, float]] = field(default_factory=list)
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary."""
        return {
            "feature_attributions": self.feature_attributions,
            "base_value": self.base_value,
            "predicted_value": self.predicted_value,
            "top_positive_features": self.top_positive_features,
            "top_negative_features": self.top_negative_features,
            "processing_time_ms": self.processing_time_ms,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseExplainer(ABC):
    """Abstract base class for all ClinIQ explainers."""

    @abstractmethod
    def explain(self, text: str, model_output: Any) -> SHAPExplanation:
        """Produce a :class:`SHAPExplanation` for *text* given *model_output*.

        Parameters
        ----------
        text:
            Raw clinical document text that was fed to the model.
        model_output:
            The model's output for *text* — typically a float confidence score
            or a dict from :meth:`~app.ml.icd.model.ICDPredictionResult.to_dict`.

        Returns
        -------
        SHAPExplanation
        """


# ---------------------------------------------------------------------------
# Token-level SHAP explainer (TF-IDF + KernelExplainer)
# ---------------------------------------------------------------------------


class TokenSHAPExplainer(BaseExplainer):
    """Token-level attribution for ICD-10 predictions using SHAP KernelExplainer.

    This explainer works with any sklearn-compatible classifier that produces
    prediction probabilities.  It builds a TF-IDF representation of the input
    text, runs SHAP's model-agnostic KernelExplainer on that feature space, and
    maps SHAP values back to human-readable token/n-gram strings.

    Parameters
    ----------
    classifier:
        Any sklearn estimator with a ``predict_proba`` method.
    vectorizer:
        A fitted :class:`~sklearn.feature_extraction.text.TfidfVectorizer`.
        When ``None`` the explainer constructs and fits a minimal vectorizer
        from the background corpus.
    background_texts:
        Small corpus (20-100 documents) used as the SHAP background dataset.
        More background samples give more accurate baseline estimates at the
        cost of slower computation.
    top_k:
        Number of top positive and negative features to surface.
    n_background_samples:
        How many background samples to pass to KernelExplainer (sub-sampled
        from *background_texts* when the corpus is larger).
    """

    def __init__(
        self,
        classifier: Any,
        vectorizer: Any = None,
        background_texts: list[str] | None = None,
        top_k: int = 10,
        n_background_samples: int = 50,
    ) -> None:
        self.classifier = classifier
        self.vectorizer = vectorizer
        self.background_texts = background_texts or []
        self.top_k = top_k
        self.n_background_samples = n_background_samples
        self._shap_explainer: Any = None
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def explain(self, text: str, model_output: Any = None) -> SHAPExplanation:
        """Compute token-level SHAP attributions for *text*.

        Parameters
        ----------
        text:
            Clinical document text.
        model_output:
            Ignored; included for API consistency with :class:`BaseExplainer`.
            The explainer calls the classifier directly.

        Returns
        -------
        SHAPExplanation
        """
        start = time.time()

        try:
            import shap
        except ImportError:
            logger.warning("shap not installed — falling back to TF-IDF weight heuristic")
            return self._fallback_explain(text, start)

        vectorizer = self._get_vectorizer(text)
        explainer = self._get_shap_explainer(vectorizer)
        self._feature_names = vectorizer.get_feature_names_out().tolist()

        # Transform the target document
        x = vectorizer.transform([text])
        x_dense = x.toarray()

        # Compute SHAP values (returns array of shape [n_samples, n_features, n_classes]
        # or [n_samples, n_features] for binary)
        shap_values = explainer.shap_values(x_dense)

        # For multi-label or multi-class, take the class with the highest
        # predicted probability (the "primary" prediction)
        if isinstance(shap_values, list):
            # multi-class: list of arrays, one per class
            probas = self.classifier.predict_proba(x_dense)[0]
            best_class = int(np.argmax(probas))
            sv = shap_values[best_class][0]
            predicted_value = float(probas[best_class])
            base_value = float(explainer.expected_value[best_class])
        else:
            sv = shap_values[0]
            if hasattr(explainer, "expected_value"):
                ev = explainer.expected_value
                base_value = float(ev[0]) if hasattr(ev, "__len__") else float(ev)
            else:
                base_value = 0.0
            if hasattr(self.classifier, "predict_proba"):
                predicted_value = float(self.classifier.predict_proba(x_dense)[0].max())
            else:
                predicted_value = float(base_value + sv.sum())

        attributions = self._build_attribution_dict(sv)
        top_pos, top_neg = self._rank_features(attributions)

        return SHAPExplanation(
            feature_attributions=attributions,
            base_value=base_value,
            predicted_value=predicted_value,
            top_positive_features=top_pos,
            top_negative_features=top_neg,
            processing_time_ms=(time.time() - start) * 1000,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_vectorizer(self, text: str) -> Any:
        """Return a fitted TF-IDF vectorizer, constructing one if necessary."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        if self.vectorizer is not None:
            return self.vectorizer

        # Build a minimal vectorizer from background corpus + target text
        corpus = self.background_texts + [text] if self.background_texts else [text]
        vect = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )
        vect.fit(corpus)
        self.vectorizer = vect
        return vect

    def _get_shap_explainer(self, vectorizer: Any) -> Any:
        """Build or return cached KernelExplainer."""
        import shap

        if self._shap_explainer is not None:
            return self._shap_explainer

        # Build background matrix
        if self.background_texts:
            bg_texts = self.background_texts[: self.n_background_samples]
            background = vectorizer.transform(bg_texts).toarray()
        else:
            # Use a zero vector as a minimal background
            n_features = len(vectorizer.get_feature_names_out())
            background = np.zeros((1, n_features))

        def _predict_fn(x: NDArray[np.float64]) -> NDArray[np.float64]:
            if hasattr(self.classifier, "predict_proba"):
                return self.classifier.predict_proba(x)
            # Fallback: sigmoid of decision function
            scores = self.classifier.decision_function(x)
            return 1.0 / (1.0 + np.exp(-scores))

        self._shap_explainer = shap.KernelExplainer(_predict_fn, background)
        return self._shap_explainer

    def _build_attribution_dict(self, shap_values: NDArray[np.float64]) -> dict[str, float]:
        """Map SHAP values to feature names, filtering near-zero entries."""
        attributions: dict[str, float] = {}
        for name, val in zip(self._feature_names, shap_values):
            if abs(val) > 1e-6:
                attributions[name] = round(float(val), 6)
        return attributions

    def _rank_features(
        self, attributions: dict[str, float]
    ) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """Split attributions into top positive and top negative lists."""
        sorted_items = sorted(attributions.items(), key=lambda x: x[1], reverse=True)
        top_pos = [(k, v) for k, v in sorted_items if v > 0][: self.top_k]
        top_neg = [(k, v) for k, v in reversed(sorted_items) if v < 0][: self.top_k]
        return top_pos, top_neg

    def _fallback_explain(self, text: str, start: float) -> SHAPExplanation:
        """Heuristic TF-IDF-weight-based attribution when SHAP is unavailable."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        vect = TfidfVectorizer(max_features=200, ngram_range=(1, 2), sublinear_tf=True, min_df=1)
        vect.fit([text])
        x = vect.transform([text]).toarray()[0]
        names = vect.get_feature_names_out().tolist()

        attributions: dict[str, float] = {
            name: round(float(val), 6) for name, val in zip(names, x) if val > 0
        }
        top_pos, top_neg = self._rank_features(attributions)

        return SHAPExplanation(
            feature_attributions=attributions,
            base_value=0.0,
            predicted_value=float(x.sum()),
            top_positive_features=top_pos,
            top_negative_features=top_neg,
            processing_time_ms=(time.time() - start) * 1000,
        )


# ---------------------------------------------------------------------------
# Attention-weight explainer (transformer models)
# ---------------------------------------------------------------------------


class AttentionExplainer(BaseExplainer):
    """Extract attention weights from transformer models for explainability.

    Returns a token-level attention heatmap averaged across all heads and
    layers.  This provides an interpretable view of which sub-words the model
    "attended to" most strongly, suitable for highlighting in a clinical UI.

    Parameters
    ----------
    tokenizer:
        A Hugging Face tokenizer (e.g. ``AutoTokenizer``).
    model:
        A Hugging Face model with ``output_attentions=True`` support
        (e.g. ``AutoModelForSequenceClassification``).
    device:
        Torch device string (``"cpu"`` or ``"cuda"``).
    layer:
        Which transformer layer to extract attentions from.  ``-1`` uses
        the last layer; ``"mean"`` averages across all layers.
    aggregate:
        How to aggregate multi-head attentions: ``"mean"`` or ``"max"``.
    """

    def __init__(
        self,
        tokenizer: Any,
        model: Any,
        device: str = "cpu",
        layer: int | str = -1,
        aggregate: str = "mean",
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.layer = layer
        self.aggregate = aggregate

    def explain(self, text: str, model_output: Any = None) -> SHAPExplanation:
        """Compute attention-based token attributions for *text*.

        Parameters
        ----------
        text:
            Clinical document text.
        model_output:
            Ignored; included for API consistency.

        Returns
        -------
        SHAPExplanation
            ``feature_attributions`` maps sub-word tokens to their mean
            attention weight.  ``base_value`` is always 0.0 for attention
            heatmaps (no meaningful baseline).
        """
        start = time.time()

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for AttentionExplainer") from exc

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_offsets_mapping=False,
        )
        input_ids = inputs["input_ids"].to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        # outputs.attentions: tuple of (n_layers,) each [batch, heads, seq, seq]
        attentions = outputs.attentions  # tuple length = n_layers
        if attentions is None:
            logger.warning("Model did not return attentions — check output_attentions flag")
            return SHAPExplanation(
                feature_attributions={},
                base_value=0.0,
                predicted_value=0.0,
                processing_time_ms=(time.time() - start) * 1000,
            )

        # Stack: [n_layers, batch=1, heads, seq, seq]
        attn_stack = torch.stack(attentions, dim=0)  # [L, 1, H, S, S]
        attn_stack = attn_stack.squeeze(1)  # [L, H, S, S]

        # Select layer(s)
        if self.layer == "mean":
            attn_layer = attn_stack.mean(dim=0)  # [H, S, S]
        else:
            idx = int(self.layer)
            attn_layer = attn_stack[idx]  # [H, S, S]

        # Aggregate heads
        if self.aggregate == "mean":
            attn_heads = attn_layer.mean(dim=0)  # [S, S]
        else:
            attn_heads = attn_layer.max(dim=0).values  # [S, S]

        # Each token's importance = mean attention it received from all positions
        token_importance = attn_heads.mean(dim=0).cpu().numpy()  # [S]

        # Normalise to [0, 1]
        if token_importance.max() > 0:
            token_importance = token_importance / token_importance.max()

        attributions: dict[str, float] = {}
        for token, weight in zip(tokens, token_importance):
            # Skip special tokens
            if token in ("[CLS]", "[SEP]", "<s>", "</s>", "[PAD]"):
                continue
            clean_token = token.replace("##", "").replace("▁", "")
            if clean_token:
                # Accumulate for repeated sub-words
                attributions[clean_token] = attributions.get(clean_token, 0.0) + float(weight)

        # Round values
        attributions = {k: round(v, 6) for k, v in attributions.items()}

        sorted_items = sorted(attributions.items(), key=lambda x: x[1], reverse=True)
        top_pos = sorted_items[:10]
        top_neg: list[tuple[str, float]] = []  # Attention weights are non-negative

        # Predicted value from logits if available
        predicted_value = 0.0
        if hasattr(outputs, "logits"):
            import torch as th

            probas = th.sigmoid(outputs.logits[0]).cpu().numpy()
            predicted_value = float(probas.max())

        return SHAPExplanation(
            feature_attributions=attributions,
            base_value=0.0,
            predicted_value=predicted_value,
            top_positive_features=top_pos,
            top_negative_features=top_neg,
            processing_time_ms=(time.time() - start) * 1000,
        )


# ---------------------------------------------------------------------------
# API formatting helper
# ---------------------------------------------------------------------------


def format_explanation(explanation: SHAPExplanation, text: str) -> dict[str, Any]:
    """Format a :class:`SHAPExplanation` for an API response.

    Produces a ``highlighted_segments`` list that a frontend can use to
    render colour-coded text spans, along with the full attribution table
    and summary statistics.

    Parameters
    ----------
    explanation:
        The SHAP or attention explanation to format.
    text:
        The original clinical document text (used for span extraction).

    Returns
    -------
    dict
        A JSON-serialisable dictionary with the following keys:

        ``highlighted_segments``
            List of ``{text, start, end, attribution, direction}`` dicts
            where ``direction`` is ``"positive"``, ``"negative"``, or
            ``"neutral"``.
        ``top_positive_features``
            Top positive-attribution tokens with their SHAP values.
        ``top_negative_features``
            Top negative-attribution tokens with their SHAP values.
        ``base_value``
            SHAP base value (expected output).
        ``predicted_value``
            Model output for this input.
        ``attribution_sum``
            Sum of all feature attributions (should approximate
            ``predicted_value - base_value``).
        ``processing_time_ms``
            Explanation computation time.
    """
    highlighted_segments: list[dict[str, Any]] = []
    text_lower = text.lower()

    for feature, value in explanation.feature_attributions.items():
        if abs(value) < 0.001:
            continue

        # Try to find the feature string in the original text
        feat_lower = feature.lower()
        start = text_lower.find(feat_lower)
        if start == -1:
            # n-gram with space — try matching tokens individually
            continue

        end = start + len(feature)
        direction: str
        if value > 0:
            direction = "positive"
        elif value < 0:
            direction = "negative"
        else:
            direction = "neutral"

        highlighted_segments.append(
            {
                "text": text[start:end],
                "start": start,
                "end": end,
                "attribution": round(value, 6),
                "direction": direction,
            }
        )

    # Sort segments by position for easy rendering
    highlighted_segments.sort(key=lambda s: s["start"])

    attribution_sum = sum(explanation.feature_attributions.values())

    return {
        "highlighted_segments": highlighted_segments,
        "top_positive_features": [
            {"feature": f, "attribution": round(v, 6)}
            for f, v in explanation.top_positive_features
        ],
        "top_negative_features": [
            {"feature": f, "attribution": round(v, 6)}
            for f, v in explanation.top_negative_features
        ],
        "base_value": explanation.base_value,
        "predicted_value": explanation.predicted_value,
        "attribution_sum": round(attribution_sum, 6),
        "processing_time_ms": explanation.processing_time_ms,
    }
