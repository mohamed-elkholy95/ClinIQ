"""Clinical Text Summarization models."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.core.exceptions import InferenceError, ModelLoadError
from app.ml.utils.text_preprocessing import ClinicalTextPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class SummaryResult:
    """Result of summarization."""

    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    processing_time_ms: float
    model_name: str
    model_version: str
    summary_type: str  # "extractive" or "abstractive"
    key_points: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "summary": self.summary,
            "original_length": self.original_length,
            "summary_length": self.summary_length,
            "compression_ratio": self.compression_ratio,
            "processing_time_ms": self.processing_time_ms,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "summary_type": self.summary_type,
            "key_points": self.key_points,
        }


class BaseSummarizer(ABC):
    """Abstract base class for summarizers."""

    def __init__(self, model_name: str, version: str = "1.0.0"):
        self.model_name = model_name
        self.version = version
        self._is_loaded = False
        self.preprocessor = ClinicalTextPreprocessor()

    @abstractmethod
    def load(self) -> None:
        """Load the model."""
        ...

    @abstractmethod
    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
    ) -> SummaryResult:
        """Generate summary of clinical text."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self._is_loaded:
            self.load()


class ExtractiveSummarizer(BaseSummarizer):
    """Extractive summarization using TextRank-like algorithm."""

    def __init__(
        self,
        model_name: str = "textrank",
        version: str = "1.0.0",
        damping: float = 0.85,
        max_iterations: int = 100,
    ):
        super().__init__(model_name, version)
        self.damping = damping
        self.max_iterations = max_iterations

    def load(self) -> None:
        """Load resources (minimal for extractive)."""
        import nltk

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords", quiet=True)

        self._is_loaded = True
        logger.info(f"Loaded extractive summarizer: {self.model_name}")

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
    ) -> SummaryResult:
        """Generate extractive summary."""
        import time

        self.ensure_loaded()
        import nltk

        start_time = time.time()

        # Preprocess
        preprocessed = self.preprocessor.preprocess(text)
        sentences = nltk.sent_tokenize(preprocessed)

        if len(sentences) <= 2:
            return SummaryResult(
                summary=preprocessed,
                original_length=len(text.split()),
                summary_length=len(preprocessed.split()),
                compression_ratio=1.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                model_name=self.model_name,
                model_version=self.version,
                summary_type="extractive",
            )

        # Build similarity matrix
        similarity_matrix = self._build_similarity_matrix(sentences)

        # Apply PageRank-like algorithm
        scores = self._textrank(similarity_matrix)

        # Select top sentences
        ranked_sentences = sorted(
            ((scores[i], s, i) for i, s in enumerate(sentences)),
            reverse=True,
        )

        # Build summary with length constraint
        summary_sentences = []
        current_length = 0

        for score, sentence, original_idx in ranked_sentences:
            sentence_length = len(sentence.split())
            if current_length + sentence_length <= max_length:
                summary_sentences.append((original_idx, sentence))
                current_length += sentence_length

        # Sort by original position for coherence
        summary_sentences.sort(key=lambda x: x[0])
        summary = " ".join(s[1] for s in summary_sentences)

        # Extract key points (top 3 sentences by score)
        key_points = [s for _, s, _ in ranked_sentences[:3]]

        processing_time = (time.time() - start_time) * 1000
        original_length = len(text.split())
        summary_length = len(summary.split())

        return SummaryResult(
            summary=summary,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=original_length / summary_length if summary_length > 0 else 1.0,
            processing_time_ms=processing_time,
            model_name=self.model_name,
            model_version=self.version,
            summary_type="extractive",
            key_points=key_points,
        )

    def _build_similarity_matrix(self, sentences: list[str]) -> np.ndarray:
        """Build sentence similarity matrix."""
        import nltk
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        try:
            stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            stop_words = set(stopwords.words("english"))

        n = len(sentences)
        similarity_matrix = np.zeros((n, n))

        # Tokenize and clean sentences
        tokenized = []
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            words = [w for w in words if w.isalnum() and w not in stop_words]
            tokenized.append(words)

        # Compute similarity using overlap coefficient
        for i in range(n):
            for j in range(i + 1, n):
                if not tokenized[i] or not tokenized[j]:
                    continue

                set_i = set(tokenized[i])
                set_j = set(tokenized[j])

                intersection = len(set_i & set_j)
                min_len = min(len(set_i), len(set_j))

                if min_len > 0:
                    similarity = intersection / min_len
                    similarity_matrix[i][j] = similarity
                    similarity_matrix[j][i] = similarity

        return similarity_matrix

    def _textrank(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Apply TextRank algorithm to similarity matrix."""
        n = len(similarity_matrix)

        # Normalize rows
        for i in range(n):
            row_sum = similarity_matrix[i].sum()
            if row_sum > 0:
                similarity_matrix[i] /= row_sum

        # Initialize scores uniformly
        scores = np.ones(n) / n

        # Power iteration
        for _ in range(self.max_iterations):
            new_scores = (1 - self.damping) / n + self.damping * similarity_matrix.T @ scores
            if np.abs(new_scores - scores).sum() < 1e-4:
                break
            scores = new_scores

        return scores


class SectionBasedSummarizer(BaseSummarizer):
    """Summarizer that extracts key sections from clinical notes."""

    # Priority order for clinical sections
    SECTION_PRIORITY = [
        "chief_complaint",
        "hpi",
        "assessment",
        "plan",
        "pmh",
        "pe",
        "ros",
        "labs",
        "medications",
        "allergies",
        "fh",
        "sh",
    ]

    def __init__(
        self,
        model_name: str = "section-based",
        version: str = "1.0.0",
        max_section_length: int = 100,
    ):
        super().__init__(model_name, version)
        self.max_section_length = max_section_length

    def load(self) -> None:
        """Load resources."""
        self._is_loaded = True
        logger.info(f"Loaded section-based summarizer: {self.model_name}")

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
    ) -> SummaryResult:
        """Generate summary based on section extraction."""
        import time

        start_time = time.time()

        # Detect sections
        sections = self.preprocessor.detect_sections(text)
        section_map = {s.name: s for s in sections}

        # Extract priority sections
        summary_parts = []
        current_length = 0
        key_points = []

        for section_name in self.SECTION_PRIORITY:
            if section_name in section_map:
                section = section_map[section_name]
                section_text = section.content

                # Truncate if needed
                words = section_text.split()
                if len(words) > self.max_section_length:
                    section_text = " ".join(words[: self.max_section_length]) + "..."

                section_words = len(section_text.split())

                if current_length + section_words <= max_length:
                    summary_parts.append(f"[{section_name.upper()}] {section_text}")
                    current_length += section_words

                    if len(key_points) < 3:
                        key_points.append(section_text[:100] + "...")

        summary = "\n\n".join(summary_parts)

        processing_time = (time.time() - start_time) * 1000
        original_length = len(text.split())
        summary_length = len(summary.split())

        return SummaryResult(
            summary=summary,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=original_length / summary_length if summary_length > 0 else 1.0,
            processing_time_ms=processing_time,
            model_name=self.model_name,
            model_version=self.version,
            summary_type="extractive",
            key_points=key_points,
        )


class AbstractiveSummarizer(BaseSummarizer):
    """Abstractive summarization using transformer models."""

    def __init__(
        self,
        model_name: str = "facebook/bart-base",
        version: str = "1.0.0",
        model_path: str | None = None,
        device: str = "cpu",
    ):
        super().__init__(model_name, version)
        self.model_path = model_path
        self.device = device
        self.tokenizer: Any = None
        self.model: Any = None

    def load(self) -> None:
        """Load the transformer model."""
        try:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            model_path = self.model_path or self.model_name

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()

            self._is_loaded = True
            logger.info(f"Loaded abstractive summarizer: {model_path}")

        except Exception as e:
            raise ModelLoadError(self.model_name, str(e))

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
    ) -> SummaryResult:
        """Generate abstractive summary."""
        import time

        self.ensure_loaded()
        import torch

        start_time = time.time()

        try:
            # Preprocess
            preprocessed = self.preprocessor.preprocess(text)

            # Tokenize
            inputs = self.tokenizer(
                preprocessed,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate
            with torch.no_grad():
                summary_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )

            # Decode
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            processing_time = (time.time() - start_time) * 1000
            original_length = len(text.split())
            summary_length = len(summary.split())

            return SummaryResult(
                summary=summary,
                original_length=original_length,
                summary_length=summary_length,
                compression_ratio=original_length / summary_length if summary_length > 0 else 1.0,
                processing_time_ms=processing_time,
                model_name=self.model_name,
                model_version=self.version,
                summary_type="abstractive",
            )

        except Exception as e:
            raise InferenceError(self.model_name, str(e))


class HybridSummarizer(BaseSummarizer):
    """Combines extractive and abstractive summarization."""

    def __init__(
        self,
        extractive: BaseSummarizer | None = None,
        abstractive: BaseSummarizer | None = None,
        model_name: str = "hybrid",
        version: str = "1.0.0",
    ):
        super().__init__(model_name, version)
        self.extractive = extractive or ExtractiveSummarizer()
        self.abstractive = abstractive

    def load(self) -> None:
        """Load sub-models."""
        self.extractive.load()
        if self.abstractive:
            self.abstractive.load()
        self._is_loaded = True

    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
    ) -> SummaryResult:
        """Generate hybrid summary."""
        import time

        start_time = time.time()

        # First get extractive summary
        extractive_result = self.extractive.summarize(text, max_length=max_length * 2)

        if not self.abstractive:
            return extractive_result

        # Then apply abstractive to extractive result
        if len(extractive_result.summary.split()) > max_length:
            abstractive_result = self.abstractive.summarize(
                extractive_result.summary,
                max_length=max_length,
                min_length=min_length,
            )
            final_summary = abstractive_result.summary
            summary_type = "hybrid"
        else:
            final_summary = extractive_result.summary
            summary_type = "extractive"

        processing_time = (time.time() - start_time) * 1000
        original_length = len(text.split())
        summary_length = len(final_summary.split())

        return SummaryResult(
            summary=final_summary,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=original_length / summary_length if summary_length > 0 else 1.0,
            processing_time_ms=processing_time,
            model_name=self.model_name,
            model_version=self.version,
            summary_type=summary_type,
            key_points=extractive_result.key_points,
        )
