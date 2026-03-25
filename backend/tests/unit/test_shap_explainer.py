"""Unit tests for the SHAP explainability module.

Tests SHAPExplanation dataclass, format_explanation helper, and the
TokenSHAPExplainer fallback path (when SHAP is not installed).
"""

from unittest.mock import MagicMock, patch

import numpy as np

from app.ml.explainability.shap_explainer import (
    SHAPExplanation,
    TokenSHAPExplainer,
    format_explanation,
)

# ---------------------------------------------------------------------------
# SHAPExplanation dataclass
# ---------------------------------------------------------------------------


class TestSHAPExplanation:
    """Test the SHAPExplanation data container."""

    def test_to_dict_all_fields(self) -> None:
        exp = SHAPExplanation(
            feature_attributions={"pain": 0.5, "normal": -0.2},
            base_value=0.3,
            predicted_value=0.6,
            top_positive_features=[("pain", 0.5)],
            top_negative_features=[("normal", -0.2)],
            processing_time_ms=12.5,
        )
        d = exp.to_dict()
        assert d["base_value"] == 0.3
        assert d["predicted_value"] == 0.6
        assert d["processing_time_ms"] == 12.5
        assert d["feature_attributions"]["pain"] == 0.5
        assert len(d["top_positive_features"]) == 1
        assert len(d["top_negative_features"]) == 1

    def test_default_fields(self) -> None:
        exp = SHAPExplanation(
            feature_attributions={},
            base_value=0.0,
            predicted_value=0.0,
        )
        assert exp.top_positive_features == []
        assert exp.top_negative_features == []
        assert exp.processing_time_ms == 0.0


# ---------------------------------------------------------------------------
# format_explanation
# ---------------------------------------------------------------------------


class TestFormatExplanation:
    """Test the API formatting helper."""

    def test_basic_formatting(self) -> None:
        exp = SHAPExplanation(
            feature_attributions={"pain": 0.5, "normal": -0.3},
            base_value=0.3,
            predicted_value=0.5,
            top_positive_features=[("pain", 0.5)],
            top_negative_features=[("normal", -0.3)],
            processing_time_ms=10.0,
        )
        text = "Patient reports pain but vitals normal"
        result = format_explanation(exp, text)

        assert "highlighted_segments" in result
        assert "top_positive_features" in result
        assert "top_negative_features" in result
        assert result["base_value"] == 0.3
        assert result["predicted_value"] == 0.5
        assert result["processing_time_ms"] == 10.0

    def test_highlighted_segments_have_positions(self) -> None:
        exp = SHAPExplanation(
            feature_attributions={"pain": 0.5},
            base_value=0.0,
            predicted_value=0.5,
            top_positive_features=[("pain", 0.5)],
        )
        text = "Patient has pain in chest"
        result = format_explanation(exp, text)
        segments = result["highlighted_segments"]
        assert len(segments) >= 1
        seg = segments[0]
        assert "start" in seg
        assert "end" in seg
        assert seg["direction"] == "positive"

    def test_negative_attribution_direction(self) -> None:
        exp = SHAPExplanation(
            feature_attributions={"normal": -0.4},
            base_value=0.5,
            predicted_value=0.1,
            top_negative_features=[("normal", -0.4)],
        )
        text = "Everything appears normal"
        result = format_explanation(exp, text)
        segments = result["highlighted_segments"]
        neg_segs = [s for s in segments if s["direction"] == "negative"]
        assert len(neg_segs) >= 1

    def test_attribution_sum(self) -> None:
        exp = SHAPExplanation(
            feature_attributions={"a": 0.3, "b": -0.1, "c": 0.2},
            base_value=0.0,
            predicted_value=0.4,
        )
        result = format_explanation(exp, "a b c")
        assert abs(result["attribution_sum"] - 0.4) < 0.001

    def test_features_not_in_text_skipped(self) -> None:
        exp = SHAPExplanation(
            feature_attributions={"nonexistent_token": 0.5},
            base_value=0.0,
            predicted_value=0.5,
        )
        text = "Patient has chest pain"
        result = format_explanation(exp, text)
        # Token not found in text → no highlighted segments
        assert len(result["highlighted_segments"]) == 0

    def test_near_zero_attributions_filtered(self) -> None:
        exp = SHAPExplanation(
            feature_attributions={"pain": 0.0005, "cough": 0.5},
            base_value=0.0,
            predicted_value=0.5,
        )
        text = "Patient has pain and cough"
        result = format_explanation(exp, text)
        # pain (0.0005) is below 0.001 threshold → filtered
        segment_texts = [s["text"].lower() for s in result["highlighted_segments"]]
        assert "pain" not in segment_texts

    def test_segments_sorted_by_position(self) -> None:
        exp = SHAPExplanation(
            feature_attributions={"cough": 0.3, "pain": 0.5},
            base_value=0.0,
            predicted_value=0.8,
        )
        text = "pain and cough"
        result = format_explanation(exp, text)
        segments = result["highlighted_segments"]
        if len(segments) >= 2:
            assert segments[0]["start"] <= segments[1]["start"]

    def test_top_features_formatting(self) -> None:
        exp = SHAPExplanation(
            feature_attributions={"x": 0.5},
            base_value=0.0,
            predicted_value=0.5,
            top_positive_features=[("x", 0.5)],
            top_negative_features=[("y", -0.3)],
        )
        result = format_explanation(exp, "x y")
        assert result["top_positive_features"][0]["feature"] == "x"
        assert result["top_negative_features"][0]["feature"] == "y"


# ---------------------------------------------------------------------------
# TokenSHAPExplainer — fallback path
# ---------------------------------------------------------------------------


class TestTokenSHAPExplainerFallback:
    """Test the fallback TF-IDF heuristic when SHAP is not installed."""

    def test_fallback_produces_explanation(self) -> None:
        classifier = MagicMock()
        explainer = TokenSHAPExplainer(classifier=classifier)

        with patch.dict("sys.modules", {"shap": None}):
            # Force ImportError on import shap
            with patch("builtins.__import__", side_effect=_mock_import_no_shap):
                result = explainer._fallback_explain("Patient has diabetes and pain", 0.0)

        assert isinstance(result, SHAPExplanation)
        assert result.base_value == 0.0
        assert len(result.feature_attributions) > 0
        assert result.processing_time_ms >= 0

    def test_fallback_attributions_non_negative(self) -> None:
        """TF-IDF weights are always ≥ 0."""
        classifier = MagicMock()
        explainer = TokenSHAPExplainer(classifier=classifier)
        result = explainer._fallback_explain("Chest pain with shortness of breath", 0.0)
        for val in result.feature_attributions.values():
            assert val >= 0

    def test_rank_features(self) -> None:
        classifier = MagicMock()
        explainer = TokenSHAPExplainer(classifier=classifier, top_k=2)
        attributions = {"a": 0.5, "b": 0.3, "c": -0.4, "d": -0.1, "e": 0.1}
        top_pos, top_neg = explainer._rank_features(attributions)
        assert len(top_pos) == 2
        assert top_pos[0][0] == "a"  # highest positive
        assert len(top_neg) == 2
        assert top_neg[0][0] == "c"  # most negative

    def test_build_attribution_dict_filters_near_zero(self) -> None:
        classifier = MagicMock()
        explainer = TokenSHAPExplainer(classifier=classifier)
        explainer._feature_names = ["a", "b", "c"]
        sv = np.array([0.5, 0.0000001, -0.3])
        result = explainer._build_attribution_dict(sv)
        assert "a" in result
        assert "b" not in result  # Near zero filtered
        assert "c" in result


import builtins as _real_builtins

_original_import = _real_builtins.__import__

def _mock_import_no_shap(name, *args, **kwargs):
    """Mock import that raises ImportError for 'shap'."""
    if name == "shap":
        raise ImportError("No module named 'shap'")
    return _original_import(name, *args, **kwargs)
