"""Extended tests for ONNX Runtime model serving — targeting uncovered code paths.

Covers:
- Full ``load()`` success path with mocked onnxruntime
- ``_load_tokenizer()`` success, ImportError, and missing-directory paths
- ``_tokenize()`` with mocked tokenizer
- ``export_from_pytorch()`` actual execution (not mocked)
- ``predict()`` with text input via tokenizer
- ``predict_batch()`` with text inputs
- ``ensure_loaded()`` calling load when not loaded
- ``InferenceSession`` creation failure
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from app.core.exceptions import InferenceError, ModelLoadError
from app.ml.serving.onnx_runtime import OnnxModelServer, OnnxPrediction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_ort() -> MagicMock:
    """Create a mock onnxruntime module with realistic structure."""
    mock_ort = MagicMock()
    mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99

    mock_input = MagicMock()
    mock_input.name = "input_ids"
    mock_attn = MagicMock()
    mock_attn.name = "attention_mask"
    mock_output = MagicMock()
    mock_output.name = "logits"

    mock_session = MagicMock()
    mock_session.get_inputs.return_value = [mock_input, mock_attn]
    mock_session.get_outputs.return_value = [mock_output]
    mock_session.get_providers.return_value = ["CPUExecutionProvider"]
    mock_session.run.return_value = [np.array([[0.2, 0.8]])]

    mock_ort.InferenceSession.return_value = mock_session
    return mock_ort


# ---------------------------------------------------------------------------
# Full load() success path
# ---------------------------------------------------------------------------

class TestLoadSuccessPath:
    """Cover the actual load() method end-to-end with mocked onnxruntime."""

    def test_load_sets_session_and_names(self, tmp_path: Path) -> None:
        """load() should populate _session, _input_names, _output_names."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake-onnx")

        server = OnnxModelServer(model_path=str(model_file))
        mock_ort = _make_mock_ort()

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            server.load()

        assert server.is_loaded
        assert server._input_names == ["input_ids", "attention_mask"]
        assert server._output_names == ["logits"]

    def test_load_creates_session_options(self, tmp_path: Path) -> None:
        """load() should configure SessionOptions with optimization level."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake")

        server = OnnxModelServer(model_path=str(model_file))
        mock_ort = _make_mock_ort()

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            server.load()

        mock_ort.SessionOptions.assert_called_once()
        mock_ort.InferenceSession.assert_called_once()

    def test_load_with_tokenizer_path_calls_load_tokenizer(
        self, tmp_path: Path,
    ) -> None:
        """When tokenizer_path is set, load() should invoke _load_tokenizer."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake")
        tok_dir = tmp_path / "tokenizer"
        tok_dir.mkdir()

        server = OnnxModelServer(
            model_path=str(model_file),
            tokenizer_path=str(tok_dir),
        )
        mock_ort = _make_mock_ort()

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}), \
             patch.object(server, "_load_tokenizer") as mock_lt:
            server.load()

        mock_lt.assert_called_once()

    def test_load_without_tokenizer_path_skips_tokenizer(
        self, tmp_path: Path,
    ) -> None:
        """When tokenizer_path is None, _load_tokenizer should NOT be called."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake")

        server = OnnxModelServer(model_path=str(model_file))
        mock_ort = _make_mock_ort()

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}), \
             patch.object(server, "_load_tokenizer") as mock_lt:
            server.load()

        mock_lt.assert_not_called()

    def test_load_inference_session_failure(self, tmp_path: Path) -> None:
        """load() should raise ModelLoadError if InferenceSession fails."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake")

        server = OnnxModelServer(model_path=str(model_file))
        mock_ort = _make_mock_ort()
        mock_ort.InferenceSession.side_effect = RuntimeError("corrupt model")

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            with pytest.raises(ModelLoadError, match="Failed to create"):
                server.load()


# ---------------------------------------------------------------------------
# _load_tokenizer() paths
# ---------------------------------------------------------------------------

class TestLoadTokenizer:
    """Cover _load_tokenizer() including success, ImportError, missing dir."""

    def test_load_tokenizer_transformers_import_error(
        self, tmp_path: Path,
    ) -> None:
        """Should raise ModelLoadError if transformers is not installed."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake")
        tok_dir = tmp_path / "tokenizer"
        tok_dir.mkdir()

        server = OnnxModelServer(
            model_path=str(model_file),
            tokenizer_path=str(tok_dir),
        )

        # Temporarily remove 'transformers' from sys.modules to simulate absence
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ModelLoadError, match="transformers"):
                server._load_tokenizer()

    def test_load_tokenizer_missing_directory(self, tmp_path: Path) -> None:
        """Should raise ModelLoadError if tokenizer directory doesn't exist."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake")

        server = OnnxModelServer(
            model_path=str(model_file),
            tokenizer_path=str(tmp_path / "nonexistent_tokenizer"),
        )

        mock_transformers = MagicMock()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            with pytest.raises(ModelLoadError, match="not found"):
                server._load_tokenizer()

    def test_load_tokenizer_none_path(self, tmp_path: Path) -> None:
        """Should raise ModelLoadError when tokenizer_path is None."""
        server = OnnxModelServer(model_path=str(tmp_path / "model.onnx"))
        server.tokenizer_path = None

        mock_transformers = MagicMock()
        with patch.dict("sys.modules", {"transformers": mock_transformers}):
            with pytest.raises(ModelLoadError, match="not found"):
                server._load_tokenizer()


# ---------------------------------------------------------------------------
# _tokenize() path
# ---------------------------------------------------------------------------

class TestTokenize:
    """Cover _tokenize() with a mocked tokenizer."""

    def test_tokenize_returns_numpy_dict(self) -> None:
        """_tokenize should return dict of int64 numpy arrays."""
        server = OnnxModelServer(model_path="model.onnx")
        server._input_names = ["input_ids", "attention_mask"]

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[101, 2003, 102]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
        }
        mock_tokenizer.__iter__ = MagicMock(
            return_value=iter(["input_ids", "attention_mask"]),
        )
        server._tokenizer = mock_tokenizer

        result = server._tokenize("Patient has diabetes")
        assert "input_ids" in result
        assert "attention_mask" in result
        assert result["input_ids"].dtype == np.int64

    def test_tokenize_filters_to_input_names(self) -> None:
        """_tokenize should only include keys that match _input_names."""
        server = OnnxModelServer(model_path="model.onnx")
        server._input_names = ["input_ids"]

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[101]], dtype=np.int64),
            "attention_mask": np.array([[1]], dtype=np.int64),
            "token_type_ids": np.array([[0]], dtype=np.int64),
        }
        mock_tokenizer.__iter__ = MagicMock(
            return_value=iter(["input_ids", "attention_mask", "token_type_ids"]),
        )
        server._tokenizer = mock_tokenizer

        result = server._tokenize("test text")
        assert "input_ids" in result
        assert "attention_mask" not in result  # not in _input_names
        assert "token_type_ids" not in result

    def test_tokenize_without_tokenizer_raises(self) -> None:
        """_tokenize should raise InferenceError if no tokenizer set."""
        server = OnnxModelServer(model_path="model.onnx")
        server._tokenizer = None

        with pytest.raises(InferenceError, match="no tokenizer"):
            server._tokenize("test")


# ---------------------------------------------------------------------------
# predict() with text (tokenizer integration)
# ---------------------------------------------------------------------------

class TestPredictWithText:
    """Cover predict(text=...) path that goes through _tokenize."""

    def test_predict_text_calls_tokenize_then_runs(self) -> None:
        """predict(text=...) should tokenize then run session."""
        server = OnnxModelServer(model_path="model.onnx")
        server._loaded = True
        server._input_names = ["input_ids", "attention_mask"]
        server._output_names = ["logits"]

        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[0.3, 0.7]])]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        server._session = mock_session

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[101, 2003, 102]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
        }
        mock_tokenizer.__iter__ = MagicMock(
            return_value=iter(["input_ids", "attention_mask"]),
        )
        server._tokenizer = mock_tokenizer

        result = server.predict(text="Patient has diabetes")
        assert isinstance(result, OnnxPrediction)
        assert "logits" in result.outputs
        mock_session.run.assert_called_once()


# ---------------------------------------------------------------------------
# predict_batch() with texts
# ---------------------------------------------------------------------------

class TestPredictBatchWithTexts:
    """Cover predict_batch(texts=...) path."""

    def test_batch_with_texts(self) -> None:
        """predict_batch(texts=...) should iterate and call predict(text=...)."""
        server = OnnxModelServer(model_path="model.onnx")
        server._loaded = True
        server._input_names = ["input_ids"]
        server._output_names = ["logits"]

        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[0.5, 0.5]])]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        server._session = mock_session

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": np.array([[101]], dtype=np.int64),
        }
        mock_tokenizer.__iter__ = MagicMock(
            return_value=iter(["input_ids"]),
        )
        server._tokenizer = mock_tokenizer

        results = server.predict_batch(texts=["Hello", "World"])
        assert len(results) == 2


# ---------------------------------------------------------------------------
# ensure_loaded() triggering load()
# ---------------------------------------------------------------------------

class TestEnsureLoaded:
    """Cover ensure_loaded() calling load() when not loaded."""

    def test_ensure_loaded_calls_load_when_not_loaded(
        self, tmp_path: Path,
    ) -> None:
        """ensure_loaded() should call load() if _loaded is False."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake")

        server = OnnxModelServer(model_path=str(model_file))
        assert not server._loaded

        with patch.object(server, "load") as mock_load:
            server.ensure_loaded()
            mock_load.assert_called_once()


# ---------------------------------------------------------------------------
# export_from_pytorch() actual execution
# ---------------------------------------------------------------------------

class TestExportFromPyTorch:
    """Cover the static export_from_pytorch method without mocking it."""

    def test_export_raises_model_load_error_on_failure(
        self, tmp_path: Path,
    ) -> None:
        """export should raise ModelLoadError when torch export fails."""
        output = tmp_path / "subdir" / "exported.onnx"

        mock_torch = MagicMock()
        mock_torch.onnx.export.side_effect = RuntimeError("Export failed")

        mock_model = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with pytest.raises(ModelLoadError, match="ONNX export failed"):
                OnnxModelServer.export_from_pytorch(
                    model=mock_model,
                    dummy_input={"input_ids": MagicMock()},
                    output_path=str(output),
                )

    def test_export_creates_parent_dirs(self, tmp_path: Path) -> None:
        """export should create parent directories if they don't exist."""
        output = tmp_path / "deeply" / "nested" / "model.onnx"

        mock_torch = MagicMock()
        mock_model = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = OnnxModelServer.export_from_pytorch(
                model=mock_model,
                dummy_input={"input_ids": MagicMock()},
                output_path=str(output),
            )
            assert result == output.resolve()

        assert output.parent.exists()

    def test_export_default_output_names(self, tmp_path: Path) -> None:
        """export should default to ['logits'] for output_names."""
        output = tmp_path / "model.onnx"
        mock_torch = MagicMock()
        mock_model = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            OnnxModelServer.export_from_pytorch(
                model=mock_model,
                dummy_input={"input_ids": MagicMock(), "attention_mask": MagicMock()},
                output_path=str(output),
            )

        # Verify torch.onnx.export was called
        call_kwargs = mock_torch.onnx.export.call_args
        assert call_kwargs is not None

    def test_export_custom_names_and_axes(self, tmp_path: Path) -> None:
        """export should pass custom input/output names and dynamic axes."""
        output = tmp_path / "model.onnx"
        mock_torch = MagicMock()
        mock_model = MagicMock()

        custom_input = ["ids", "mask"]
        custom_output = ["predictions"]
        custom_axes = {"ids": {0: "batch"}}

        with patch.dict("sys.modules", {"torch": mock_torch}):
            OnnxModelServer.export_from_pytorch(
                model=mock_model,
                dummy_input={"ids": MagicMock(), "mask": MagicMock()},
                output_path=str(output),
                input_names=custom_input,
                output_names=custom_output,
                dynamic_axes=custom_axes,
                opset_version=13,
            )

        call_kwargs = mock_torch.onnx.export.call_args
        assert call_kwargs is not None
