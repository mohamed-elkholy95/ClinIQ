"""Clinical text summarization models.

Provides extractive (TextRank + clinical relevance weighting) and abstractive
(HuggingFace BART/T5 wrapper) summarizers that accept a ``detail_level``
parameter to control output length.
"""

from __future__ import annotations

import logging
import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from app.core.exceptions import InferenceError, ModelLoadError
from app.ml.utils.text_preprocessing import ClinicalTextPreprocessor

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Target fraction of original sentences to include at each detail level
_DETAIL_RATIO: dict[str, float] = {
    "brief": 0.15,
    "standard": 0.30,
    "detailed": 0.50,
}

# Hard upper bounds on sentence count per detail level
_DETAIL_SENTENCE_CAP: dict[str, int] = {
    "brief": 5,
    "standard": 12,
    "detailed": 25,
}

# Min/max token lengths for abstractive generation per detail level
_ABSTRACTIVE_LENGTHS: dict[str, tuple[int, int]] = {
    "brief": (30, 80),
    "standard": (80, 200),
    "detailed": (150, 400),
}

# Maximum tokens fed to an abstractive model in one pass (leaves room for
# special tokens and generation overhead)
_CHUNK_MAX_TOKENS = 900

# Section names that carry elevated clinical weight during summarisation
_HIGH_PRIORITY_SECTIONS = frozenset(
    {
        "assessment",
        "plan",
        "impression",
        "assessment and plan",
        "a/p",
        "chief_complaint",
        "chief complaint",
        "cc",
    }
)

# Patterns that mark sentences as clinically important
_CLINICAL_IMPORTANCE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\b(?:diagnosis|diagnosed|impression)\b",
        r"\b(?:treatment|therapy|plan|prescrib|recommend)\b",
        r"\b(?:significant|critical|urgent|emergent|stat|acute)\b",
        r"\b(?:increased|elevated|decreased|abnormal|positive)\b",
        r"\b(?:medication|drug|dose|dosage|mg|mcg)\b",
        r"\b(?:follow.?up|refer(?:ral)?|consult)\b",
        r"\b(?:procedure|surgery|operation|biopsy)\b",
        r"\b(?:allerg(?:y|ic)|reaction|contraindic)\b",
    ]
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SummarizationResult:
    """Result of a summarization operation.

    Attributes
    ----------
    summary:
        The generated summary text.
    key_findings:
        A short list of the most clinically significant sentences / findings
        extracted from the original document.
    detail_level:
        The requested detail level that produced this result.
    processing_time_ms:
        Wall-clock time for inference in milliseconds.
    model_name:
        Identifier of the model that produced this result.
    model_version:
        Version string of the model.
    sentence_count_original:
        Number of sentences in the source document.
    sentence_count_summary:
        Number of sentences in the produced summary.
    metadata:
        Optional free-form metadata dict (e.g. chunk_count for abstractive).
    """

    summary: str
    key_findings: list[str]
    detail_level: Literal["brief", "standard", "detailed"]
    processing_time_ms: float
    model_name: str
    model_version: str
    sentence_count_original: int = 0
    sentence_count_summary: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "summary": self.summary,
            "key_findings": self.key_findings,
            "detail_level": self.detail_level,
            "processing_time_ms": self.processing_time_ms,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "sentence_count_original": self.sentence_count_original,
            "sentence_count_summary": self.sentence_count_summary,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseSummarizer(ABC):
    """Abstract base class for all clinical text summarizers."""

    def __init__(self, model_name: str, version: str = "1.0.0") -> None:
        self.model_name = model_name
        self.version = version
        self._is_loaded: bool = False

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        ...

    @abstractmethod
    def summarize(
        self,
        text: str,
        detail_level: Literal["brief", "standard", "detailed"] = "standard",
    ) -> SummarizationResult:
        """Summarise *text* at the requested *detail_level*."""
        ...

    @property
    def is_loaded(self) -> bool:
        """``True`` if the model has been loaded."""
        return self._is_loaded

    def ensure_loaded(self) -> None:
        """Call :meth:`load` if the model is not yet loaded."""
        if not self._is_loaded:
            self.load()

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _target_sentence_count(
        self,
        total: int,
        detail_level: Literal["brief", "standard", "detailed"],
    ) -> int:
        """Return how many sentences the summary should contain."""
        ratio = _DETAIL_RATIO[detail_level]
        target = max(1, math.ceil(total * ratio))
        return min(target, _DETAIL_SENTENCE_CAP[detail_level])

    def _extract_key_findings(
        self,
        sentences: list[str],
        top_n: int = 5,
    ) -> list[str]:
        """Return the *top_n* sentences most likely to be key clinical findings."""
        scored: list[tuple[float, str]] = []
        for sent in sentences:
            score = sum(
                1.0 for pat in _CLINICAL_IMPORTANCE_PATTERNS if pat.search(sent)
            )
            if score > 0:
                scored.append((score, sent))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:top_n]]


# ---------------------------------------------------------------------------
# ExtractiveSummarizer  –  TextRank + clinical relevance weighting
# ---------------------------------------------------------------------------


class ExtractiveSummarizer(BaseSummarizer):
    """TextRank-based extractive summariser with clinical relevance weighting.

    Algorithm:
    1. Segment the document into sentences with :class:`ClinicalTextPreprocessor`.
    2. Compute a TF-IDF matrix over the sentence vocabulary (sklearn
       ``TfidfVectorizer``).
    3. Build a sentence-similarity graph from pairwise cosine similarities of
       the TF-IDF vectors.
    4. Assign a clinical importance *bias* score to each node.  Sentences from
       Assessment / Plan sections and those matching ``_CLINICAL_IMPORTANCE_PATTERNS``
       receive higher bias.
    5. Run personalised PageRank until convergence.
    6. Select the top-ranked sentences and emit them in their original
       document order.
    """

    def __init__(
        self,
        model_name: str = "extractive-textrank",
        version: str = "1.0.0",
        damping: float = 0.85,
        max_iter: int = 100,
        convergence_threshold: float = 1e-4,
    ) -> None:
        super().__init__(model_name, version)
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_threshold = convergence_threshold
        self._preprocessor = ClinicalTextPreprocessor()
        self._vectorizer: Any = None  # sklearn TfidfVectorizer; created at load()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Initialise the TF-IDF vectorizer (no disk I/O required)."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                stop_words="english",
                max_features=5000,
                sublinear_tf=True,
            )
            self._is_loaded = True
            logger.info("Loaded ExtractiveSummarizer v%s", self.version)
        except Exception as exc:
            raise ModelLoadError(self.model_name, str(exc)) from exc

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def summarize(
        self,
        text: str,
        detail_level: Literal["brief", "standard", "detailed"] = "standard",
    ) -> SummarizationResult:
        """Summarise *text* using biased TextRank.

        Parameters
        ----------
        text:
            Raw clinical document text.
        detail_level:
            ``"brief"``, ``"standard"``, or ``"detailed"`` – controls the
            fraction of sentences retained.

        Returns
        -------
        SummarizationResult
        """
        self.ensure_loaded()
        start_time = time.time()

        try:
            cleaned = self._preprocessor.preprocess(text)
            sentences = self._preprocessor.segment_sentences(cleaned)
            sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

            if not sentences:
                return self._empty_result(text, detail_level, start_time)

            if len(sentences) == 1:
                processing_time = (time.time() - start_time) * 1000
                return SummarizationResult(
                    summary=sentences[0],
                    key_findings=sentences[:1],
                    detail_level=detail_level,
                    processing_time_ms=processing_time,
                    model_name=self.model_name,
                    model_version=self.version,
                    sentence_count_original=1,
                    sentence_count_summary=1,
                )

            # Build TF-IDF similarity matrix
            tfidf_matrix = self._vectorizer.fit_transform(sentences).toarray()
            sim_matrix = self._cosine_similarity_matrix(tfidf_matrix)

            # Compute per-sentence clinical bias
            bias = self._compute_bias_scores(sentences, cleaned)

            # Run personalised PageRank
            scores = self._pagerank(sim_matrix, bias)

            # Select top-k sentences, restore original order
            target = self._target_sentence_count(len(sentences), detail_level)
            top_indices = sorted(
                range(len(sentences)),
                key=lambda i: scores[i],
                reverse=True,
            )[:target]
            selected_indices = sorted(top_indices)

            summary_sentences = [sentences[i] for i in selected_indices]
            summary = " ".join(summary_sentences)
            key_findings = self._extract_key_findings(sentences)

            processing_time = (time.time() - start_time) * 1000
            logger.debug(
                "ExtractiveSummarizer: %d -> %d sentences in %.1f ms",
                len(sentences),
                len(summary_sentences),
                processing_time,
            )

            return SummarizationResult(
                summary=summary,
                key_findings=key_findings,
                detail_level=detail_level,
                processing_time_ms=processing_time,
                model_name=self.model_name,
                model_version=self.version,
                sentence_count_original=len(sentences),
                sentence_count_summary=len(summary_sentences),
            )

        except Exception as exc:
            raise InferenceError(self.model_name, str(exc)) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cosine_similarity_matrix(self, matrix: Any) -> Any:
        """Return the pairwise cosine similarity matrix for *matrix* rows."""
        import numpy as np

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalised = matrix / norms
        return normalised @ normalised.T

    def _compute_bias_scores(
        self, sentences: list[str], full_text: str
    ) -> Any:
        """Return a per-sentence bias vector for personalised PageRank.

        Bias is elevated for sentences that:
        * belong to Assessment / Plan sections
        * match ``_CLINICAL_IMPORTANCE_PATTERNS``
        """
        import numpy as np

        sections = self._preprocessor.detect_sections(full_text)
        section_spans = [(s.start_char, s.end_char, s.name) for s in sections]

        bias = np.ones(len(sentences))
        search_start = 0

        for idx, sent in enumerate(sentences):
            # Pattern-based importance score
            pattern_score = sum(
                1.0 for pat in _CLINICAL_IMPORTANCE_PATTERNS if pat.search(sent)
            )
            bias[idx] += pattern_score * 0.2

            # Section-based boost: locate sentence in original text
            pos = full_text.find(sent, search_start)
            if pos != -1:
                search_start = pos
                for sec_start, sec_end, sec_name in section_spans:
                    if sec_start <= pos < sec_end:
                        if sec_name in _HIGH_PRIORITY_SECTIONS:
                            bias[idx] += 1.5
                        break

        # Re-scale so bias sums to len(sentences), preserving PageRank properties
        total = bias.sum()
        if total > 0:
            bias = bias * (len(sentences) / total)
        return bias

    def _pagerank(self, sim_matrix: Any, bias: Any) -> Any:
        """Run personalised PageRank on the sentence similarity graph."""
        import numpy as np

        n = sim_matrix.shape[0]

        # Remove self-loops and row-normalise
        np.fill_diagonal(sim_matrix, 0.0)
        row_sums = sim_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        transition = sim_matrix / row_sums

        # Normalise bias to a probability vector
        bias_norm = bias / bias.sum() if bias.sum() > 0 else np.full(n, 1.0 / n)

        scores = np.full(n, 1.0 / n)
        for _ in range(self.max_iter):
            new_scores = (
                (1 - self.damping) * bias_norm
                + self.damping * transition.T @ scores
            )
            delta = float(np.abs(new_scores - scores).sum())
            scores = new_scores
            if delta < self.convergence_threshold:
                break

        return scores

    def _empty_result(
        self,
        text: str,
        detail_level: Literal["brief", "standard", "detailed"],
        start_time: float,
    ) -> SummarizationResult:
        return SummarizationResult(
            summary=text[:500] if text else "",
            key_findings=[],
            detail_level=detail_level,
            processing_time_ms=(time.time() - start_time) * 1000,
            model_name=self.model_name,
            model_version=self.version,
        )


# ---------------------------------------------------------------------------
# AbstractiveSummarizer  –  HuggingFace BART / T5 wrapper
# ---------------------------------------------------------------------------


class AbstractiveSummarizer(BaseSummarizer):
    """HuggingFace BART/T5 abstractive summarizer for clinical text.

    Long documents are split into overlapping chunks that fit within the
    model's context window.  Each chunk is summarised independently.  When
    the concatenated partial summaries are still too long, a second
    summarisation pass is applied (hierarchical / two-stage approach).

    Parameters
    ----------
    model_name:
        HuggingFace model identifier or path (default ``facebook/bart-large-cnn``).
    version:
        Semantic version string.
    model_path:
        Optional local path; overrides *model_name* if provided.
    device:
        ``"cpu"`` or ``"cuda"``.
    chunk_overlap_tokens:
        Number of overlapping tokens between adjacent chunks to preserve
        context at chunk boundaries.
    """

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        version: str = "1.0.0",
        model_path: str | None = None,
        device: str = "cpu",
        chunk_overlap_tokens: int = 50,
    ) -> None:
        super().__init__(model_name, version)
        self.model_path = model_path
        self.device = device
        self.chunk_overlap_tokens = chunk_overlap_tokens
        self._pipeline: Any = None
        self._tokenizer: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Download / load the HuggingFace summarization pipeline."""
        try:
            from transformers import AutoTokenizer, pipeline

            path = self.model_path or self.model_name
            self._tokenizer = AutoTokenizer.from_pretrained(path)
            self._pipeline = pipeline(
                "summarization",
                model=path,
                tokenizer=self._tokenizer,
                device=0 if self.device == "cuda" else -1,
                framework="pt",
            )
            self._is_loaded = True
            logger.info("Loaded AbstractiveSummarizer: %s", path)
        except Exception as exc:
            raise ModelLoadError(self.model_name, str(exc)) from exc

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def summarize(
        self,
        text: str,
        detail_level: Literal["brief", "standard", "detailed"] = "standard",
    ) -> SummarizationResult:
        """Generate an abstractive summary, chunking long documents as needed.

        Parameters
        ----------
        text:
            Raw clinical document text.
        detail_level:
            Controls ``min_length`` / ``max_length`` passed to the model.

        Returns
        -------
        SummarizationResult
        """
        self.ensure_loaded()
        start_time = time.time()

        try:
            preprocessor = ClinicalTextPreprocessor()
            cleaned = preprocessor.preprocess(text)

            min_len, max_len = _ABSTRACTIVE_LENGTHS[detail_level]
            chunks = self._chunk_text(cleaned)

            if len(chunks) == 1:
                raw_summary = self._summarize_chunk(chunks[0], min_len, max_len)
            else:
                # Summarise each chunk, then optionally do a second pass
                partial_summaries = [
                    self._summarize_chunk(
                        chunk,
                        max(10, min_len // len(chunks)),
                        max_len,
                    )
                    for chunk in chunks
                ]
                combined = " ".join(partial_summaries)
                combined_tokens = len(self._tokenizer.encode(combined))
                if combined_tokens > _CHUNK_MAX_TOKENS:
                    raw_summary = self._summarize_chunk(combined, min_len, max_len)
                else:
                    raw_summary = combined

            sentences = preprocessor.segment_sentences(cleaned)
            key_findings = self._extract_key_findings(sentences)

            processing_time = (time.time() - start_time) * 1000
            logger.debug(
                "AbstractiveSummarizer: completed in %.1f ms (%d chunk(s))",
                processing_time,
                len(chunks),
            )

            return SummarizationResult(
                summary=raw_summary,
                key_findings=key_findings,
                detail_level=detail_level,
                processing_time_ms=processing_time,
                model_name=self.model_name,
                model_version=self.version,
                sentence_count_original=len(sentences),
                sentence_count_summary=len(
                    preprocessor.segment_sentences(raw_summary)
                ),
                metadata={"chunk_count": len(chunks)},
            )

        except Exception as exc:
            raise InferenceError(self.model_name, str(exc)) from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str) -> list[str]:
        """Split *text* into overlapping token-bounded chunks."""
        token_ids: list[int] = self._tokenizer.encode(
            text, add_special_tokens=False
        )
        if len(token_ids) <= _CHUNK_MAX_TOKENS:
            return [text]

        chunks: list[str] = []
        step = _CHUNK_MAX_TOKENS - self.chunk_overlap_tokens
        for start in range(0, len(token_ids), step):
            chunk_ids = token_ids[start : start + _CHUNK_MAX_TOKENS]
            chunk_text = self._tokenizer.decode(chunk_ids, skip_special_tokens=True)
            chunks.append(chunk_text)
            if start + _CHUNK_MAX_TOKENS >= len(token_ids):
                break

        logger.debug(
            "AbstractiveSummarizer: split into %d chunks", len(chunks)
        )
        return chunks

    def _summarize_chunk(
        self, chunk: str, min_length: int, max_length: int
    ) -> str:
        """Run the HuggingFace pipeline on a single text chunk."""
        result = self._pipeline(
            chunk,
            min_length=min_length,
            max_length=max_length,
            do_sample=False,
            truncation=True,
        )
        return result[0]["summary_text"]
