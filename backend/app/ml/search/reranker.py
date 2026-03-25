"""Cross-encoder re-ranker for clinical document search.

Initial retrieval (BM25 + TF-IDF) casts a wide net and returns ~10-50
candidates efficiently.  The re-ranker applies a more expensive but
more accurate scoring model to those candidates and returns a refined
top-k.  This two-stage architecture (retrieve → re-rank) is standard
in production search systems because it balances latency and quality.

Architecture
------------
* **Cross-encoder scoring** — Unlike the bi-encoder approach of TF-IDF
  (where query and document are encoded independently), a cross-encoder
  jointly attends to both query and document in a single forward pass.
  This captures token-level interactions that bi-encoders miss but is
  too slow for full-corpus scoring.
* **Rule-based clinical scorer** — A lightweight, no-dependency scorer
  that uses medical term overlap, section-header proximity, and
  abbreviation awareness to re-rank candidates.  This is the default
  when no transformer model is available and is fast enough (<1 ms per
  candidate) for real-time use.
* **Pluggable interface** — ``ReRanker`` is an abstract base with two
  concrete implementations: ``ClinicalRuleReRanker`` (default, no ML
  dependencies) and ``TransformerReRanker`` (requires a HuggingFace
  cross-encoder model).

Design decisions
----------------
* **Abbreviation-aware matching** — The re-ranker uses the same
  abbreviation dictionary from ``query_expansion.py`` to detect when a
  query says "htn" but the document says "hypertension" (or vice versa),
  boosting the score accordingly.
* **Section weighting** — Clinical notes have structure (Assessment,
  Plan, HPI, etc.).  Text near "Assessment" or "Plan" section headers
  is weighted higher, mirroring how clinicians read notes.
* **Normalised output** — All re-rankers return scores in [0, 1] so
  downstream code can apply a uniform threshold.
"""

from __future__ import annotations

import abc
import logging
import math
import re
from dataclasses import dataclass, field

from app.ml.search.query_expansion import (
    _ABBREVIATION_TO_FULL,
    _FULL_TO_ABBREVIATION,
    _SYNONYM_LOOKUP,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReRankCandidate:
    """A document candidate to be re-ranked.

    Attributes
    ----------
    doc_id:
        Application-level document identifier.
    text:
        Full or snippet text of the document.
    initial_score:
        Score from the first-stage retriever (e.g. hybrid BM25 + TF-IDF).
    metadata:
        Optional additional metadata (e.g. document type, specialty).
    """

    doc_id: str
    text: str
    initial_score: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ReRankedResult:
    """A re-ranked search result.

    Attributes
    ----------
    doc_id:
        Application-level document identifier.
    score:
        Final re-ranked score in [0, 1].
    initial_score:
        Original first-stage retrieval score.
    text:
        Document text (for display).
    score_components:
        Breakdown of scoring factors (for debugging/explainability).
    """

    doc_id: str
    score: float
    initial_score: float
    text: str
    score_components: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ReRanker(abc.ABC):
    """Abstract re-ranker interface.

    Subclasses must implement ``score_pair`` which takes a query and
    a single document and returns a relevance score in [0, 1].
    """

    @abc.abstractmethod
    def score_pair(self, query: str, document: str) -> float:
        """Score a single query-document pair.

        Parameters
        ----------
        query:
            Search query string.
        document:
            Document text.

        Returns
        -------
        float
            Relevance score in [0, 1].
        """

    def rerank(
        self,
        query: str,
        candidates: list[ReRankCandidate],
        top_k: int = 10,
        initial_weight: float = 0.3,
    ) -> list[ReRankedResult]:
        """Re-rank a list of candidates.

        The final score interpolates the re-ranker score with the
        initial retrieval score:

            final = (1 - initial_weight) * reranker_score + initial_weight * initial_score

        Parameters
        ----------
        query:
            Search query.
        candidates:
            First-stage retrieval candidates.
        top_k:
            Maximum results to return.
        initial_weight:
            Weight given to the initial retrieval score (0-1).

        Returns
        -------
        list[ReRankedResult]
            Re-ranked results sorted by final score descending.
        """
        if not candidates:
            return []

        results: list[ReRankedResult] = []
        for candidate in candidates:
            reranker_score = self.score_pair(query, candidate.text)
            final_score = (
                (1 - initial_weight) * reranker_score
                + initial_weight * candidate.initial_score
            )

            results.append(ReRankedResult(
                doc_id=candidate.doc_id,
                score=round(min(max(final_score, 0.0), 1.0), 4),
                initial_score=candidate.initial_score,
                text=candidate.text,
                score_components={
                    "reranker": round(reranker_score, 4),
                    "initial": round(candidate.initial_score, 4),
                    "initial_weight": initial_weight,
                },
            ))

        # Sort by final score descending
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]


# ---------------------------------------------------------------------------
# Clinical rule-based re-ranker (no ML dependencies)
# ---------------------------------------------------------------------------

# Section headers that indicate high-value clinical content
_HIGH_VALUE_SECTIONS = re.compile(
    r"\b(assessment|plan|impression|diagnosis|conclusion|recommendation"
    r"|assessment and plan|a/?p|final diagnosis|discharge diagnosis)\b",
    re.IGNORECASE,
)

_MODERATE_VALUE_SECTIONS = re.compile(
    r"\b(history of present illness|hpi|chief complaint"
    r"|reason for visit|clinical findings|results)\b",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


class ClinicalRuleReRanker(ReRanker):
    """Rule-based re-ranker using medical term matching and section weighting.

    Scoring components:
    1. **Term overlap** (0.35) — Fraction of query tokens found in document
    2. **Abbreviation match** (0.20) — Bonus for abbreviation ↔ full-form matches
    3. **Synonym match** (0.15) — Bonus for synonym-group matches
    4. **Section proximity** (0.15) — Bonus for matches near high-value sections
    5. **Coverage density** (0.15) — Ratio of matched unique terms to doc length

    Parameters
    ----------
    section_boost:
        Multiplier for matches in Assessment/Plan sections.
    abbreviation_boost:
        Score bonus for abbreviation ↔ full-form matches.
    """

    def __init__(
        self,
        section_boost: float = 1.5,
        abbreviation_boost: float = 0.15,
    ) -> None:
        self.section_boost = section_boost
        self.abbreviation_boost = abbreviation_boost

    def score_pair(self, query: str, document: str) -> float:
        """Score a query-document pair using clinical rule heuristics.

        Parameters
        ----------
        query:
            Search query text.
        document:
            Document text.

        Returns
        -------
        float
            Relevance score in [0, 1].
        """
        if not query.strip() or not document.strip():
            return 0.0

        q_tokens = set(_TOKEN_RE.findall(query.lower()))
        d_tokens = _TOKEN_RE.findall(document.lower())
        d_token_set = set(d_tokens)

        if not q_tokens or not d_tokens:
            return 0.0

        # Component 1: Direct term overlap
        direct_matches = q_tokens & d_token_set
        term_overlap = len(direct_matches) / len(q_tokens) if q_tokens else 0.0

        # Component 2: Abbreviation matching
        abbr_score = 0.0
        abbr_matches = 0
        for qt in q_tokens:
            # Check if query token is an abbreviation with full form in doc
            full = _ABBREVIATION_TO_FULL.get(qt)
            if full:
                full_tokens = set(_TOKEN_RE.findall(full.lower()))
                if full_tokens & d_token_set:
                    abbr_matches += 1

            # Check if query token's full form has abbreviation in doc
            abbr = _FULL_TO_ABBREVIATION.get(qt)
            if abbr and abbr in d_token_set:
                abbr_matches += 1

        if q_tokens:
            abbr_score = min(abbr_matches / len(q_tokens), 1.0)

        # Component 3: Synonym matching
        syn_score = 0.0
        syn_matches = 0
        for qt in q_tokens:
            synonyms = _SYNONYM_LOOKUP.get(qt, set())
            for syn in synonyms:
                syn_tokens = set(_TOKEN_RE.findall(syn.lower()))
                if syn_tokens & d_token_set:
                    syn_matches += 1
                    break  # Count each query token at most once
        if q_tokens:
            syn_score = min(syn_matches / len(q_tokens), 1.0)

        # Component 4: Section proximity
        section_score = 0.0
        doc_lower = document.lower()
        if _HIGH_VALUE_SECTIONS.search(doc_lower):
            # Check if query terms appear near these sections
            for match in _HIGH_VALUE_SECTIONS.finditer(doc_lower):
                # Look at 200 chars after the section header
                window = doc_lower[match.start():match.start() + 300]
                window_tokens = set(_TOKEN_RE.findall(window))
                overlap = q_tokens & window_tokens
                if overlap:
                    section_score = max(
                        section_score,
                        len(overlap) / len(q_tokens) * self.section_boost,
                    )
            section_score = min(section_score, 1.0)
        elif _MODERATE_VALUE_SECTIONS.search(doc_lower):
            for match in _MODERATE_VALUE_SECTIONS.finditer(doc_lower):
                window = doc_lower[match.start():match.start() + 300]
                window_tokens = set(_TOKEN_RE.findall(window))
                overlap = q_tokens & window_tokens
                if overlap:
                    section_score = max(
                        section_score,
                        len(overlap) / len(q_tokens) * 0.7,
                    )
            section_score = min(section_score, 1.0)

        # Component 5: Coverage density
        # How many unique matched terms appear relative to document length
        all_matches = direct_matches | {
            qt for qt in q_tokens
            if _ABBREVIATION_TO_FULL.get(qt)
            and set(_TOKEN_RE.findall(_ABBREVIATION_TO_FULL[qt])) & d_token_set
        }
        # Use log to dampen long-document advantage
        doc_len_factor = math.log(len(d_tokens) + 1)
        density = len(all_matches) / max(doc_len_factor, 1.0)
        density_score = min(density, 1.0)

        # Weighted combination
        final = (
            0.35 * term_overlap
            + 0.20 * abbr_score
            + 0.15 * syn_score
            + 0.15 * section_score
            + 0.15 * density_score
        )

        return min(max(final, 0.0), 1.0)


class TransformerReRanker(ReRanker):
    """Cross-encoder re-ranker using a HuggingFace transformer model.

    Wraps a ``cross-encoder/ms-marco-MiniLM-L-6-v2`` or similar model
    for pairwise query-document scoring.  Falls back to the rule-based
    re-ranker if the model fails to load.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier for the cross-encoder.
    max_length:
        Maximum input sequence length (tokens).
    device:
        Torch device (``"cpu"``, ``"cuda"``).  Default auto-detects.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self._device = device
        self._fallback = ClinicalRuleReRanker()
        self._loaded = False

    def _load_model(self) -> bool:
        """Attempt to load the cross-encoder model.

        Returns
        -------
        bool
            True if model loaded successfully.
        """
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import]

            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self._device,
            )
            self._loaded = True
            logger.info("Loaded cross-encoder model: %s", self.model_name)
            return True
        except ImportError:
            logger.warning(
                "sentence-transformers not installed; "
                "using rule-based re-ranker fallback"
            )
            return False
        except Exception:
            logger.exception("Failed to load cross-encoder model: %s", self.model_name)
            return False

    def score_pair(self, query: str, document: str) -> float:
        """Score a query-document pair using the cross-encoder.

        Falls back to rule-based scoring if the model is unavailable.

        Parameters
        ----------
        query:
            Search query.
        document:
            Document text.

        Returns
        -------
        float
            Relevance score in [0, 1].
        """
        if not self._loaded:
            if not self._load_model():
                return self._fallback.score_pair(query, document)

        try:
            # Truncate document to max_length chars to avoid OOM
            truncated = document[:self.max_length * 4]
            raw_score = self._model.predict([(query, truncated)])  # type: ignore[union-attr]
            # Sigmoid normalisation to [0, 1]
            score = float(raw_score[0]) if hasattr(raw_score, '__len__') else float(raw_score)
            normalised = 1.0 / (1.0 + math.exp(-score))
            return min(max(normalised, 0.0), 1.0)
        except Exception:
            logger.exception("Cross-encoder inference failed; using fallback")
            return self._fallback.score_pair(query, document)
