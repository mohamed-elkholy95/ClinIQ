"""ONNX Runtime model serving for production inference.

Provides a unified ``OnnxModelServer`` that loads exported ``.onnx`` models
and runs tokenize-then-infer pipelines with automatic session caching,
input validation, and Prometheus-compatible latency recording.

Design Decisions
----------------
* **Lazy loading** — models are loaded on first ``predict()`` call, not at
  import time.  This keeps application startup fast and avoids allocating
  GPU/CPU resources for models that may never be used.
* **Session caching** — a single ``InferenceSession`` is kept in memory per
  model path.  ONNX Runtime manages its own thread pool internally, so we
  avoid re-creating sessions on every request.
* **Tokenizer agnostic** — the server accepts pre-tokenised numpy arrays *or*
  raw text (when a HuggingFace tokenizer path is provided).  This lets
  callers choose between flexibility and control.
* **Graceful degradation** — when ``onnxruntime`` is not installed the module
  raises ``ModelLoadError`` with a clear message rather than an opaque
  ``ImportError``.

Usage
-----
>>> server = OnnxModelServer(
...     model_path="models/ner/ner_model.onnx",
...     tokenizer_path="models/ner/tokenizer",
... )
>>> server.load()
>>> result = server.predict("Patient presents with acute chest pain.")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from app.core.exceptions import InferenceError, ModelLoadError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class OnnxPrediction:
    """Container for a single ONNX model prediction.

    Attributes
    ----------
    outputs : dict[str, NDArray]
        Raw output tensors keyed by ONNX output node name.
    latency_ms : float
        Wall-clock inference time in milliseconds (excludes tokenization).
    model_path : str
        Path to the ONNX model that produced this prediction.
    """

    outputs: dict[str, NDArray]
    latency_ms: float
    model_path: str
    metadata: dict[str, Any] = field(default_factory=dict)


class OnnxModelServer:
    """High-performance model serving via ONNX Runtime.

    Parameters
    ----------
    model_path : str | Path
        Filesystem path to the exported ``.onnx`` model file.
    tokenizer_path : str | Path | None
        Optional path to a HuggingFace tokenizer directory.  When provided,
        ``predict(text)`` will auto-tokenize the input.  When ``None`` the
        caller must supply pre-tokenized numpy arrays.
    providers : list[str] | None
        ONNX Runtime execution providers in priority order.
        Defaults to ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.
    max_length : int
        Maximum token sequence length for the tokenizer (default 512).
    """

    def __init__(
        self,
        model_path: str | Path,
        tokenizer_path: str | Path | None = None,
        providers: list[str] | None = None,
        max_length: int = 512,
    ) -> None:
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else None
        self.providers = providers or ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.max_length = max_length

        self._session: Any | None = None
        self._tokenizer: Any | None = None
        self._loaded = False
        self._input_names: list[str] = []
        self._output_names: list[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load the ONNX model and optional tokenizer into memory.

        Raises
        ------
        ModelLoadError
            If ``onnxruntime`` is not installed, the model file is missing,
            or the session fails to initialise.
        """
        try:
            import onnxruntime as ort  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ModelLoadError(
                str(self.model_path),
                reason=(
                    "onnxruntime is not installed. "
                    "Install with: pip install onnxruntime  "
                    "(or onnxruntime-gpu for CUDA support)"
                ),
            ) from exc

        if not self.model_path.exists():
            raise ModelLoadError(
                str(self.model_path),
                reason=f"Model file not found: {self.model_path}",
            )

        try:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            sess_options.intra_op_num_threads = 4

            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=self.providers,
            )
            self._input_names = [inp.name for inp in self._session.get_inputs()]
            self._output_names = [out.name for out in self._session.get_outputs()]
            logger.info(
                "ONNX model loaded: %s  inputs=%s  outputs=%s  providers=%s",
                self.model_path.name,
                self._input_names,
                self._output_names,
                self._session.get_providers(),
            )
        except Exception as exc:
            raise ModelLoadError(
                str(self.model_path),
                reason=f"Failed to create InferenceSession: {exc}",
            ) from exc

        # Load tokenizer if configured
        if self.tokenizer_path is not None:
            self._load_tokenizer()

        self._loaded = True

    def _load_tokenizer(self) -> None:
        """Load a HuggingFace tokenizer from disk.

        Raises
        ------
        ModelLoadError
            If the ``transformers`` library is not installed or the
            tokenizer directory is invalid.
        """
        try:
            from transformers import AutoTokenizer  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ModelLoadError(
                str(self.model_path),
                reason="transformers library required for tokenizer: pip install transformers",
            ) from exc

        if not self.tokenizer_path or not self.tokenizer_path.exists():
            raise ModelLoadError(
                str(self.model_path),
                reason=f"Tokenizer directory not found: {self.tokenizer_path}",
            )

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_path))
        logger.info("Tokenizer loaded from %s", self.tokenizer_path)

    def ensure_loaded(self) -> None:
        """Load the model if not already loaded.

        This is the recommended entry point — it's safe to call
        repeatedly and will only do work the first time.
        """
        if not self._loaded:
            self.load()

    @property
    def is_loaded(self) -> bool:
        """Whether the ONNX session is currently active."""
        return self._loaded

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        text: str | None = None,
        inputs: dict[str, NDArray] | None = None,
    ) -> OnnxPrediction:
        """Run inference on the loaded ONNX model.

        Parameters
        ----------
        text : str, optional
            Raw clinical text.  Requires a tokenizer to be configured.
        inputs : dict[str, NDArray], optional
            Pre-tokenized model inputs as ``{input_name: array}``.
            Exactly one of ``text`` or ``inputs`` must be provided.

        Returns
        -------
        OnnxPrediction
            Prediction result with raw output tensors and timing info.

        Raises
        ------
        InferenceError
            If the model is not loaded, inputs are invalid, or ONNX
            Runtime raises during execution.
        ValueError
            If both or neither of ``text`` / ``inputs`` are provided.
        """
        self.ensure_loaded()

        if text is not None and inputs is not None:
            raise ValueError("Provide exactly one of 'text' or 'inputs', not both")
        if text is None and inputs is None:
            raise ValueError("Provide exactly one of 'text' or 'inputs'")

        # Tokenize if raw text was provided
        if text is not None:
            inputs = self._tokenize(text)

        assert inputs is not None  # for type checker

        # Validate input names
        missing = set(self._input_names) - set(inputs.keys())
        if missing:
            raise InferenceError(
                str(self.model_path),
                reason=f"Missing input tensors: {missing}. Expected: {self._input_names}",
            )

        # Run inference
        try:
            start = time.perf_counter()
            raw_outputs = self._session.run(
                self._output_names,
                {name: inputs[name] for name in self._input_names},
            )
            latency_ms = (time.perf_counter() - start) * 1000
        except Exception as exc:
            raise InferenceError(
                str(self.model_path),
                reason=f"ONNX Runtime inference failed: {exc}",
            ) from exc

        outputs = dict(zip(self._output_names, raw_outputs))

        logger.debug(
            "ONNX inference complete: model=%s latency=%.2fms",
            self.model_path.name,
            latency_ms,
        )

        return OnnxPrediction(
            outputs=outputs,
            latency_ms=latency_ms,
            model_path=str(self.model_path),
            metadata={
                "providers": self._session.get_providers() if self._session else [],
                "input_names": self._input_names,
                "output_names": self._output_names,
            },
        )

    def predict_batch(
        self,
        texts: list[str] | None = None,
        batch_inputs: list[dict[str, NDArray]] | None = None,
    ) -> list[OnnxPrediction]:
        """Run inference on multiple inputs sequentially.

        Parameters
        ----------
        texts : list[str], optional
            List of raw clinical texts to process.
        batch_inputs : list[dict[str, NDArray]], optional
            List of pre-tokenized input dicts.

        Returns
        -------
        list[OnnxPrediction]
            One prediction per input, in order.

        Raises
        ------
        ValueError
            If both or neither of ``texts`` / ``batch_inputs`` are provided.
        """
        if texts is not None and batch_inputs is not None:
            raise ValueError("Provide exactly one of 'texts' or 'batch_inputs'")
        if texts is None and batch_inputs is None:
            raise ValueError("Provide exactly one of 'texts' or 'batch_inputs'")

        results: list[OnnxPrediction] = []
        items: list[tuple[str | None, dict[str, NDArray] | None]]

        if texts is not None:
            items = [(t, None) for t in texts]
        else:
            assert batch_inputs is not None
            items = [(None, inp) for inp in batch_inputs]

        for text_item, input_item in items:
            results.append(self.predict(text=text_item, inputs=input_item))

        return results

    # ------------------------------------------------------------------
    # Tokenization
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> dict[str, NDArray]:
        """Tokenize raw text into model input tensors.

        Parameters
        ----------
        text : str
            Raw clinical text to tokenize.

        Returns
        -------
        dict[str, NDArray]
            Token IDs, attention masks, and (optionally) token type IDs
            as int64 numpy arrays with batch dimension.

        Raises
        ------
        InferenceError
            If no tokenizer is configured.
        """
        if self._tokenizer is None:
            raise InferenceError(
                str(self.model_path),
                reason=(
                    "Cannot tokenize raw text: no tokenizer configured. "
                    "Either provide pre-tokenized inputs or set tokenizer_path."
                ),
            )

        encoding = self._tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        # Convert to dict of numpy arrays (ONNX Runtime expects numpy)
        inputs: dict[str, NDArray] = {}
        for key in encoding:
            if key in self._input_names:
                inputs[key] = np.array(encoding[key], dtype=np.int64)

        return inputs

    # ------------------------------------------------------------------
    # Export helpers (static)
    # ------------------------------------------------------------------

    @staticmethod
    def export_from_pytorch(
        model: Any,
        dummy_input: dict[str, Any],
        output_path: str | Path,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        opset_version: int = 14,
    ) -> Path:
        """Export a PyTorch model to ONNX format.

        Parameters
        ----------
        model
            A PyTorch ``nn.Module`` in eval mode.
        dummy_input : dict[str, Any]
            Example input tensors for tracing.
        output_path : str | Path
            Where to write the ``.onnx`` file.
        input_names : list[str], optional
            Names for the ONNX graph inputs.
        output_names : list[str], optional
            Names for the ONNX graph outputs.
        dynamic_axes : dict, optional
            Dynamic axis specifications for variable-length sequences.
        opset_version : int
            ONNX opset version (default 14 for broad compatibility).

        Returns
        -------
        Path
            Absolute path to the exported ``.onnx`` file.

        Raises
        ------
        ModelLoadError
            If PyTorch or ONNX export fails.
        """
        import torch  # type: ignore[import-untyped]

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if input_names is None:
            input_names = list(dummy_input.keys())
        if output_names is None:
            output_names = ["logits"]
        if dynamic_axes is None:
            dynamic_axes = {name: {0: "batch", 1: "sequence"} for name in input_names}
            for out_name in output_names:
                dynamic_axes[out_name] = {0: "batch"}

        try:
            model.eval()
            torch.onnx.export(
                model,
                tuple(dummy_input.values()),
                str(output_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=opset_version,
                do_constant_folding=True,
            )
            logger.info("ONNX model exported to %s", output_path)
            return output_path.resolve()
        except Exception as exc:
            raise ModelLoadError(
                "pytorch-export",
                reason=f"ONNX export failed: {exc}",
            ) from exc
