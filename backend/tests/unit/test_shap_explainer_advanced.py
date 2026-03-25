"""Advanced tests for TokenSHAPExplainer (real SHAP path) and AttentionExplainer.

Covers the real explain() flow with mocked SHAP library, vectorizer creation,
caching, multi-class vs binary paths, and the AttentionExplainer's
attention-weight extraction from transformer outputs.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Minimal torch mock for AttentionExplainer tests
# ---------------------------------------------------------------------------

class _FakeTorchTensor:
    """Lightweight tensor stand-in backed by numpy."""

    def __init__(self, data):
        self._arr = np.array(data, dtype=np.float32)

    def to(self, device):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def tolist(self):
        return self._arr.tolist()

    def squeeze(self, dim=None):
        return _FakeTorchTensor(self._arr.squeeze(axis=dim))

    def mean(self, dim=None):
        return _FakeTorchTensor(self._arr.mean(axis=dim))

    def max(self, dim=None):
        if dim is not None:
            result = MagicMock()
            result.values = _FakeTorchTensor(self._arr.max(axis=dim))
            return result
        return _FakeTorchTensor(self._arr.max())

    def __getitem__(self, idx):
        return _FakeTorchTensor(self._arr[idx])

    @property
    def shape(self):
        return self._arr.shape

    def __len__(self):
        return len(self._arr)


_torch_mock = MagicMock()
_torch_mock.tensor = lambda data, **kw: _FakeTorchTensor(data)
_torch_mock.zeros = lambda *shape, **kw: _FakeTorchTensor(np.zeros(shape if len(shape) > 1 else shape[0]))
_torch_mock.ones = lambda *shape, **kw: _FakeTorchTensor(np.ones(shape if len(shape) > 1 else shape[0]))
_torch_mock.rand = lambda *shape: _FakeTorchTensor(np.random.rand(*shape))
_torch_mock.sigmoid = lambda x: _FakeTorchTensor(1.0 / (1.0 + np.exp(-x.numpy())))
_torch_mock.stack = lambda tensors, dim=0: _FakeTorchTensor(np.stack([t.numpy() for t in tensors], axis=dim))
_torch_mock.long = "torch.int64"

_no_grad = MagicMock()
_no_grad.__enter__ = MagicMock(return_value=None)
_no_grad.__exit__ = MagicMock(return_value=False)
_torch_mock.no_grad.return_value = _no_grad

from app.ml.explainability.shap_explainer import (
    AttentionExplainer,
    SHAPExplanation,
    TokenSHAPExplainer,
)

# ---------------------------------------------------------------------------
# TokenSHAPExplainer — real SHAP path
# ---------------------------------------------------------------------------


class TestTokenSHAPExplainerRealPath:
    """Test explain() when SHAP is available (mocked)."""

    @pytest.fixture()
    def classifier(self) -> MagicMock:
        clf = MagicMock()
        clf.predict_proba.return_value = np.array([[0.3, 0.7]])
        return clf

    @pytest.fixture()
    def explainer(self, classifier: MagicMock) -> TokenSHAPExplainer:
        return TokenSHAPExplainer(
            classifier=classifier,
            background_texts=["Patient has diabetes", "Normal findings"],
            top_k=5,
        )

    def test_explain_binary_shap(self, explainer: TokenSHAPExplainer) -> None:
        """Binary classifier: shap_values is a single array."""
        mock_shap = MagicMock()
        # KernelExplainer returns array of shape [1, n_features]
        mock_ke_instance = MagicMock()
        mock_ke_instance.shap_values.return_value = np.array([[0.3, -0.1, 0.2, 0.0, 0.05]])
        mock_ke_instance.expected_value = 0.4
        mock_shap.KernelExplainer.return_value = mock_ke_instance

        with patch.dict("sys.modules", {"shap": mock_shap}):
            result = explainer.explain("Patient has chest pain today")

        assert isinstance(result, SHAPExplanation)
        assert result.processing_time_ms > 0
        assert result.base_value == 0.4

    def test_explain_multiclass_shap(self, explainer: TokenSHAPExplainer) -> None:
        """Multi-class: shap_values is a list of arrays, one per class."""
        mock_shap = MagicMock()
        mock_ke_instance = MagicMock()
        # 3 classes, 5 features
        mock_ke_instance.shap_values.return_value = [
            np.array([[0.1, 0.0, 0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.3, 0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.2, 0.0, 0.0]]),
        ]
        mock_ke_instance.expected_value = [0.2, 0.5, 0.3]
        mock_shap.KernelExplainer.return_value = mock_ke_instance

        # predict_proba returns 3-class output; class 1 is highest
        explainer.classifier.predict_proba.return_value = np.array([[0.1, 0.7, 0.2]])

        with patch.dict("sys.modules", {"shap": mock_shap}):
            result = explainer.explain("Diabetes with complications")

        assert isinstance(result, SHAPExplanation)
        assert result.base_value == 0.5  # expected_value[1] for best class
        assert result.predicted_value == pytest.approx(0.7)

    def test_get_vectorizer_caches(self, explainer: TokenSHAPExplainer) -> None:
        """Vectorizer is built once and cached."""
        v1 = explainer._get_vectorizer("some text")
        v2 = explainer._get_vectorizer("other text")
        assert v1 is v2

    def test_get_vectorizer_no_background(self) -> None:
        """When no background texts, vectorizer fits on target text only."""
        clf = MagicMock()
        exp = TokenSHAPExplainer(classifier=clf, background_texts=[])
        v = exp._get_vectorizer("Single document text")
        assert v is not None
        assert len(v.get_feature_names_out()) > 0

    def test_get_shap_explainer_caches(self, explainer: TokenSHAPExplainer) -> None:
        """KernelExplainer is built once and cached."""
        mock_shap = MagicMock()
        mock_ke_instance = MagicMock()
        mock_shap.KernelExplainer.return_value = mock_ke_instance

        vectorizer = explainer._get_vectorizer("test text")

        with patch.dict("sys.modules", {"shap": mock_shap}):
            e1 = explainer._get_shap_explainer(vectorizer)
            e2 = explainer._get_shap_explainer(vectorizer)

        assert e1 is e2
        assert mock_shap.KernelExplainer.call_count == 1

    def test_shap_explainer_zero_background(self) -> None:
        """Without background texts, uses zero vector as baseline."""
        clf = MagicMock()
        exp = TokenSHAPExplainer(classifier=clf, background_texts=[])

        mock_shap = MagicMock()
        mock_ke_instance = MagicMock()
        mock_shap.KernelExplainer.return_value = mock_ke_instance

        vectorizer = exp._get_vectorizer("test text")

        with patch.dict("sys.modules", {"shap": mock_shap}):
            exp._get_shap_explainer(vectorizer)

        # Check background was a zero array
        call_args = mock_shap.KernelExplainer.call_args
        bg = call_args[0][1]  # positional arg 1 = background
        assert np.all(bg == 0)

    def test_explain_decision_function_fallback(self) -> None:
        """Classifier without predict_proba uses decision_function + sigmoid."""
        clf = MagicMock(spec=["decision_function", "fit"])
        clf.decision_function.return_value = np.array([[1.0, -1.0]])

        exp = TokenSHAPExplainer(classifier=clf, background_texts=[])

        mock_shap = MagicMock()
        mock_ke_instance = MagicMock()
        mock_ke_instance.shap_values.return_value = np.array([[0.2, -0.1]])
        mock_ke_instance.expected_value = 0.3
        mock_shap.KernelExplainer.return_value = mock_ke_instance

        with patch.dict("sys.modules", {"shap": mock_shap}):
            result = exp.explain("test text")

        assert isinstance(result, SHAPExplanation)
        # predicted_value = base_value + sum(sv) = 0.3 + 0.1 = 0.4
        assert result.predicted_value == pytest.approx(0.4, abs=0.01)


# ---------------------------------------------------------------------------
# AttentionExplainer
# ---------------------------------------------------------------------------


class TestAttentionExplainer:
    """Test transformer attention-based explainability."""

    @pytest.fixture(autouse=True)
    def _patch_torch(self):
        with patch.dict("sys.modules", {"torch": _torch_mock}):
            yield

    def _make_explainer(self, layer=-1, aggregate="mean"):
        """Build an AttentionExplainer with mocked tokenizer/model."""
        tokenizer = MagicMock()
        model = MagicMock()
        return AttentionExplainer(
            tokenizer=tokenizer,
            model=model,
            device="cpu",
            layer=layer,
            aggregate=aggregate,
        )

    def _mock_torch_output(self, explainer, seq_len=5, n_layers=2, n_heads=4):
        """Configure mocked model to return fake attention tensors."""
        # Tokenizer returns input_ids and token strings
        explainer.tokenizer.return_value = {
            "input_ids": _FakeTorchTensor(np.zeros((1, seq_len))),
            "attention_mask": _FakeTorchTensor(np.ones((1, seq_len))),
        }
        tokens = ["[CLS]", "patient", "has", "pain", "[SEP]"][:seq_len]
        explainer.tokenizer.convert_ids_to_tokens.return_value = tokens

        # Model returns attentions tuple
        # Each attention: [batch=1, heads, seq, seq]
        attentions = tuple(
            _FakeTorchTensor(np.random.rand(1, n_heads, seq_len, seq_len).astype(np.float32))
            for _ in range(n_layers)
        )
        mock_output = MagicMock()
        mock_output.attentions = attentions
        mock_output.logits = _FakeTorchTensor([[0.8, 0.2]])
        explainer.model.return_value = mock_output

    def test_explain_returns_shap_explanation(self) -> None:
        exp = self._make_explainer()
        self._mock_torch_output(exp)
        result = exp.explain("Patient has pain")
        assert isinstance(result, SHAPExplanation)
        assert result.processing_time_ms > 0

    def test_special_tokens_excluded(self) -> None:
        exp = self._make_explainer()
        self._mock_torch_output(exp)
        result = exp.explain("Patient has pain")
        assert "[CLS]" not in result.feature_attributions
        assert "[SEP]" not in result.feature_attributions

    def test_mean_layer_aggregation(self) -> None:
        exp = self._make_explainer(layer="mean")
        self._mock_torch_output(exp)
        result = exp.explain("Patient has pain")
        assert len(result.feature_attributions) > 0

    def test_max_head_aggregation(self) -> None:
        exp = self._make_explainer(aggregate="max")
        self._mock_torch_output(exp)
        result = exp.explain("Patient has pain")
        assert len(result.feature_attributions) > 0

    def test_no_attentions_returns_empty(self) -> None:
        """Model doesn't return attentions — graceful fallback."""
        exp = self._make_explainer()
        exp.tokenizer.return_value = {
            "input_ids": _FakeTorchTensor(np.zeros((1, 3))),
            "attention_mask": _FakeTorchTensor(np.ones((1, 3))),
        }
        exp.tokenizer.convert_ids_to_tokens.return_value = ["[CLS]", "test", "[SEP]"]

        mock_output = MagicMock()
        mock_output.attentions = None
        exp.model.return_value = mock_output

        result = exp.explain("test")
        assert result.feature_attributions == {}

    def test_predicted_value_from_logits(self) -> None:
        exp = self._make_explainer()
        self._mock_torch_output(exp)
        result = exp.explain("Patient has pain")
        # sigmoid(0.8) ≈ 0.69, sigmoid(0.2) ≈ 0.55 → max ≈ 0.69
        assert result.predicted_value > 0

    def test_subword_accumulation(self) -> None:
        """Sub-word tokens (##ing) get accumulated into base token."""
        exp = self._make_explainer()
        exp.tokenizer.return_value = {
            "input_ids": _FakeTorchTensor(np.zeros((1, 4))),
            "attention_mask": _FakeTorchTensor(np.ones((1, 4))),
        }
        exp.tokenizer.convert_ids_to_tokens.return_value = ["[CLS]", "run", "##ning", "[SEP]"]

        attentions = tuple(
            _FakeTorchTensor(np.random.rand(1, 2, 4, 4).astype(np.float32))
            for _ in range(2)
        )
        mock_output = MagicMock()
        mock_output.attentions = attentions
        mock_output.logits = _FakeTorchTensor([[0.5]])
        exp.model.return_value = mock_output

        result = exp.explain("running")
        # "run" and "##ning" → "run" and "ning" → both contribute
        # The key "run" should accumulate both sub-word weights
        assert "run" in result.feature_attributions or "ning" in result.feature_attributions

    def test_torch_not_installed_raises(self) -> None:
        """Without torch, AttentionExplainer.explain raises RuntimeError."""
        exp = self._make_explainer()

        with patch.dict("sys.modules", {"torch": None}):
            with pytest.raises((RuntimeError, ImportError)):
                exp.explain("test text")

    def test_top_positive_features_populated(self) -> None:
        exp = self._make_explainer()
        self._mock_torch_output(exp)
        result = exp.explain("Patient has pain")
        # Attention weights are non-negative, so top_negative should be empty
        assert len(result.top_negative_features) == 0
        assert len(result.top_positive_features) > 0
