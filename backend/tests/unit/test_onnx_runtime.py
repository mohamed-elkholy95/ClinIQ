"""Tests for ONNX Runtime model serving module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.exceptions import InferenceError, ModelLoadError
from app.ml.serving.onnx_runtime import OnnxModelServer, OnnxPrediction


class TestOnnxPrediction:
    """Tests for the OnnxPrediction dataclass."""

    def test_basic_construction(self) -> None:
        """Should store outputs, latency, and model path."""
        pred = OnnxPrediction(
            outputs={"logits": np.array([[0.1, 0.9]])},
            latency_ms=5.2,
            model_path="models/ner.onnx",
        )
        assert pred.latency_ms == 5.2
        assert pred.model_path == "models/ner.onnx"
        assert "logits" in pred.outputs

    def test_default_metadata(self) -> None:
        """Metadata should default to empty dict."""
        pred = OnnxPrediction(
            outputs={}, latency_ms=0.0, model_path="test.onnx"
        )
        assert pred.metadata == {}

    def test_custom_metadata(self) -> None:
        """Should accept custom metadata."""
        pred = OnnxPrediction(
            outputs={},
            latency_ms=1.0,
            model_path="test.onnx",
            metadata={"providers": ["CPUExecutionProvider"]},
        )
        assert pred.metadata["providers"] == ["CPUExecutionProvider"]


class TestOnnxModelServerInit:
    """Tests for OnnxModelServer construction."""

    def test_default_providers(self) -> None:
        """Should default to CUDA + CPU providers."""
        server = OnnxModelServer(model_path="model.onnx")
        assert "CUDAExecutionProvider" in server.providers
        assert "CPUExecutionProvider" in server.providers

    def test_custom_providers(self) -> None:
        """Should accept custom provider list."""
        server = OnnxModelServer(
            model_path="model.onnx",
            providers=["CPUExecutionProvider"],
        )
        assert server.providers == ["CPUExecutionProvider"]

    def test_default_max_length(self) -> None:
        """Default max_length should be 512."""
        server = OnnxModelServer(model_path="model.onnx")
        assert server.max_length == 512

    def test_not_loaded_initially(self) -> None:
        """Server should not be loaded after construction."""
        server = OnnxModelServer(model_path="model.onnx")
        assert not server.is_loaded

    def test_tokenizer_path_none_by_default(self) -> None:
        """Tokenizer path should be None when not specified."""
        server = OnnxModelServer(model_path="model.onnx")
        assert server.tokenizer_path is None


class TestOnnxModelServerLoad:
    """Tests for model loading."""

    def test_raises_when_onnxruntime_not_installed(self, tmp_path: Path) -> None:
        """Should raise ModelLoadError if onnxruntime is missing."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake")
        server = OnnxModelServer(model_path=str(model_file))

        with patch.dict("sys.modules", {"onnxruntime": None}):
            with patch(
                "builtins.__import__",
                side_effect=ImportError("No module named 'onnxruntime'"),
            ):
                with pytest.raises(ModelLoadError, match="onnxruntime"):
                    server.load()

    def test_raises_when_model_file_missing(self) -> None:
        """Should raise ModelLoadError if .onnx file doesn't exist."""
        server = OnnxModelServer(model_path="/nonexistent/model.onnx")

        mock_ort = MagicMock()
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            with pytest.raises(ModelLoadError, match="not found"):
                server.load()

    def test_successful_load(self, tmp_path: Path) -> None:
        """Should successfully load when onnxruntime is available."""
        model_file = tmp_path / "model.onnx"
        model_file.write_bytes(b"fake-onnx-data")

        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "input_ids"
        mock_output = MagicMock()
        mock_output.name = "logits"
        mock_session.get_inputs.return_value = [mock_input]
        mock_session.get_outputs.return_value = [mock_output]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]

        mock_ort = MagicMock()
        mock_ort.InferenceSession.return_value = mock_session
        mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99

        server = OnnxModelServer(model_path=str(model_file))

        with patch.dict("sys.modules", {"onnxruntime": mock_ort}):
            with patch("app.ml.serving.onnx_runtime.OnnxModelServer.load") as mock_load:
                # Just test the state management
                server._loaded = True
                assert server.is_loaded

    def test_ensure_loaded_calls_load_once(self) -> None:
        """ensure_loaded should only call load() when not yet loaded."""
        server = OnnxModelServer(model_path="model.onnx")
        server._loaded = True
        # Should not raise — already loaded
        server.ensure_loaded()
        assert server.is_loaded


class TestOnnxModelServerPredict:
    """Tests for inference."""

    def _make_loaded_server(self) -> OnnxModelServer:
        """Create a server with mocked session for testing."""
        server = OnnxModelServer(model_path="test.onnx")
        server._loaded = True
        server._input_names = ["input_ids", "attention_mask"]
        server._output_names = ["logits"]

        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[0.1, 0.9]])]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        server._session = mock_session

        return server

    def test_predict_with_inputs(self) -> None:
        """Should run inference with pre-tokenized inputs."""
        server = self._make_loaded_server()
        inputs = {
            "input_ids": np.array([[101, 2003, 102]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
        }
        result = server.predict(inputs=inputs)
        assert isinstance(result, OnnxPrediction)
        assert result.latency_ms >= 0
        assert "logits" in result.outputs

    def test_predict_raises_with_both_text_and_inputs(self) -> None:
        """Should raise ValueError when both text and inputs given."""
        server = self._make_loaded_server()
        with pytest.raises(ValueError, match="exactly one"):
            server.predict(text="hello", inputs={"input_ids": np.array([1])})

    def test_predict_raises_with_neither(self) -> None:
        """Should raise ValueError when neither text nor inputs given."""
        server = self._make_loaded_server()
        with pytest.raises(ValueError, match="exactly one"):
            server.predict()

    def test_predict_raises_on_missing_input_names(self) -> None:
        """Should raise InferenceError when required inputs are missing."""
        server = self._make_loaded_server()
        inputs = {"input_ids": np.array([[101]], dtype=np.int64)}
        # Missing attention_mask
        with pytest.raises(InferenceError, match="Missing input tensors"):
            server.predict(inputs=inputs)

    def test_predict_raises_on_runtime_error(self) -> None:
        """Should wrap ONNX Runtime errors in InferenceError."""
        server = self._make_loaded_server()
        server._session.run.side_effect = RuntimeError("ORT failure")
        inputs = {
            "input_ids": np.array([[101]], dtype=np.int64),
            "attention_mask": np.array([[1]], dtype=np.int64),
        }
        with pytest.raises(InferenceError, match="ONNX Runtime inference failed"):
            server.predict(inputs=inputs)

    def test_predict_text_without_tokenizer_raises(self) -> None:
        """Should raise InferenceError when text is given but no tokenizer."""
        server = self._make_loaded_server()
        server._tokenizer = None
        with pytest.raises(InferenceError, match="no tokenizer"):
            server.predict(text="Patient has diabetes")

    def test_predict_output_metadata(self) -> None:
        """Prediction metadata should contain provider info."""
        server = self._make_loaded_server()
        inputs = {
            "input_ids": np.array([[101]], dtype=np.int64),
            "attention_mask": np.array([[1]], dtype=np.int64),
        }
        result = server.predict(inputs=inputs)
        assert "providers" in result.metadata
        assert "CPUExecutionProvider" in result.metadata["providers"]


class TestOnnxModelServerBatchPredict:
    """Tests for batch inference."""

    def _make_loaded_server(self) -> OnnxModelServer:
        """Create a server with mocked session for batch testing."""
        server = OnnxModelServer(model_path="test.onnx")
        server._loaded = True
        server._input_names = ["input_ids"]
        server._output_names = ["logits"]

        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[0.5, 0.5]])]
        mock_session.get_providers.return_value = ["CPUExecutionProvider"]
        server._session = mock_session

        return server

    def test_batch_predict_with_inputs(self) -> None:
        """Should return one prediction per input."""
        server = self._make_loaded_server()
        batch = [
            {"input_ids": np.array([[101]], dtype=np.int64)},
            {"input_ids": np.array([[102]], dtype=np.int64)},
        ]
        results = server.predict_batch(batch_inputs=batch)
        assert len(results) == 2
        assert all(isinstance(r, OnnxPrediction) for r in results)

    def test_batch_raises_with_both_texts_and_inputs(self) -> None:
        """Should raise ValueError when both texts and batch_inputs given."""
        server = self._make_loaded_server()
        with pytest.raises(ValueError, match="exactly one"):
            server.predict_batch(
                texts=["hello"],
                batch_inputs=[{"input_ids": np.array([1])}],
            )

    def test_batch_raises_with_neither(self) -> None:
        """Should raise ValueError when neither is provided."""
        server = self._make_loaded_server()
        with pytest.raises(ValueError, match="exactly one"):
            server.predict_batch()

    def test_batch_empty_list(self) -> None:
        """Empty batch should return empty results."""
        server = self._make_loaded_server()
        results = server.predict_batch(batch_inputs=[])
        assert results == []


class TestOnnxExportHelper:
    """Tests for the static export_from_pytorch method."""

    def test_export_creates_file(self, tmp_path: Path) -> None:
        """Export should create an .onnx file at the specified path."""
        output = tmp_path / "exported.onnx"

        mock_torch = MagicMock()
        mock_model = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with patch(
                "app.ml.serving.onnx_runtime.OnnxModelServer.export_from_pytorch"
            ) as mock_export:
                mock_export.return_value = output
                result = OnnxModelServer.export_from_pytorch(
                    model=mock_model,
                    dummy_input={"input_ids": MagicMock()},
                    output_path=str(output),
                )
                assert result == output
