"""Clinical relation extraction between medical entities.

Identifies semantic relationships between pairs of medical entities found in
clinical text.  Supports both rule-based pattern matching (zero-dependency,
<1 ms per pair) and an optional transformer-based classifier that can be
fine-tuned on labelled relation data.

Design decisions
----------------
* **Rule-based first** — Most clinical relations follow predictable syntactic
  patterns ("Patient is on *metformin* for *diabetes*"), so rule-based
  extraction provides strong baseline performance without ML overhead.
* **Configurable relation types** — New relation categories can be registered
  via :pyattr:`RELATION_TYPES` without modifying extractor logic.
* **Entity-pair windowing** — Only entity pairs within a configurable token
  distance are considered, reducing quadratic blowup on long documents.
* **Directional relations** — Each :class:`Relation` records a subject/object
  ordering that reflects clinical semantics (e.g. *medication* treats
  *disease*, not the reverse).

Architecture
-----------
::

    Entities ─┬─► PairFilter (distance + type compatibility)
              │
              └─► RuleBasedRelationExtractor ──► Relation[]
                         │
              ┌──────────┘
              ▼
    TransformerRelationExtractor (optional fine-tuned model)
"""

from __future__ import annotations

import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from app.ml.ner.model import Entity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Relation taxonomy
# ---------------------------------------------------------------------------

class RelationType(str, Enum):
    """Semantic relation categories between medical entities.

    Each relation is directional: *subject* → relation → *object*.
    """

    TREATS = "treats"                    # medication → disease
    CAUSES = "causes"                    # disease/procedure → symptom
    DIAGNOSES = "diagnoses"              # test → disease
    CONTRAINDICATES = "contraindicates"  # disease → medication
    ADMINISTERED_FOR = "administered_for"  # procedure → disease
    DOSAGE_OF = "dosage_of"              # dosage → medication
    LOCATION_OF = "location_of"          # anatomy → disease/procedure
    RESULT_OF = "result_of"              # lab_value → test
    WORSENS = "worsens"                  # disease → disease/symptom
    PREVENTS = "prevents"                # medication → disease
    MONITORS = "monitors"                # test → disease/medication
    SIDE_EFFECT_OF = "side_effect_of"    # symptom → medication


# Which (subject_type, object_type) pairs are valid for each relation.
# This prevents nonsensical pairings like DOSAGE treats ANATOMY.
RELATION_TYPE_CONSTRAINTS: dict[RelationType, list[tuple[str, str]]] = {
    RelationType.TREATS: [
        ("MEDICATION", "DISEASE"),
        ("MEDICATION", "SYMPTOM"),
        ("PROCEDURE", "DISEASE"),
    ],
    RelationType.CAUSES: [
        ("DISEASE", "SYMPTOM"),
        ("PROCEDURE", "SYMPTOM"),
        ("MEDICATION", "SYMPTOM"),
    ],
    RelationType.DIAGNOSES: [
        ("TEST", "DISEASE"),
        ("LAB_VALUE", "DISEASE"),
    ],
    RelationType.CONTRAINDICATES: [
        ("DISEASE", "MEDICATION"),
        ("DISEASE", "PROCEDURE"),
    ],
    RelationType.ADMINISTERED_FOR: [
        ("PROCEDURE", "DISEASE"),
        ("PROCEDURE", "SYMPTOM"),
    ],
    RelationType.DOSAGE_OF: [
        ("DOSAGE", "MEDICATION"),
    ],
    RelationType.LOCATION_OF: [
        ("ANATOMY", "DISEASE"),
        ("ANATOMY", "PROCEDURE"),
        ("ANATOMY", "SYMPTOM"),
    ],
    RelationType.RESULT_OF: [
        ("LAB_VALUE", "TEST"),
    ],
    RelationType.WORSENS: [
        ("DISEASE", "DISEASE"),
        ("DISEASE", "SYMPTOM"),
        ("MEDICATION", "DISEASE"),
    ],
    RelationType.PREVENTS: [
        ("MEDICATION", "DISEASE"),
        ("PROCEDURE", "DISEASE"),
    ],
    RelationType.MONITORS: [
        ("TEST", "DISEASE"),
        ("TEST", "MEDICATION"),
    ],
    RelationType.SIDE_EFFECT_OF: [
        ("SYMPTOM", "MEDICATION"),
    ],
}


@dataclass
class Relation:
    """A directed semantic relation between two medical entities.

    Parameters
    ----------
    subject:
        The source entity (e.g. the medication).
    object_entity:
        The target entity (e.g. the disease).
    relation_type:
        The semantic relationship linking subject to object.
    confidence:
        Extraction confidence in [0, 1].
    evidence:
        The text span connecting the two entities.
    metadata:
        Arbitrary extra fields (e.g. extraction method, model version).
    """

    subject: Entity
    object_entity: Entity
    relation_type: RelationType
    confidence: float
    evidence: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "subject": self.subject.to_dict(),
            "object": self.object_entity.to_dict(),
            "relation_type": self.relation_type.value,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


@dataclass
class RelationExtractionResult:
    """Container for a batch of extracted relations.

    Parameters
    ----------
    relations:
        Extracted :class:`Relation` instances.
    entity_count:
        How many entities were supplied.
    pair_count:
        How many entity pairs were evaluated.
    processing_time_ms:
        Wall-clock extraction time.
    model_name:
        Which extractor produced the results.
    model_version:
        Version of the extractor.
    """

    relations: list[Relation]
    entity_count: int
    pair_count: int
    processing_time_ms: float
    model_name: str
    model_version: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "relations": [r.to_dict() for r in self.relations],
            "entity_count": self.entity_count,
            "pair_count": self.pair_count,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "model_name": self.model_name,
            "model_version": self.model_version,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseRelationExtractor(ABC):
    """Interface for clinical relation extractors."""

    @abstractmethod
    def extract(
        self,
        text: str,
        entities: list[Entity],
        *,
        max_distance: int = 150,
        min_confidence: float = 0.3,
    ) -> RelationExtractionResult:
        """Extract relations between *entities* found in *text*.

        Parameters
        ----------
        text:
            Original clinical text.
        entities:
            Pre-extracted medical entities with character offsets.
        max_distance:
            Maximum character distance between two entities to be
            considered a candidate pair.
        min_confidence:
            Discard relations below this threshold.

        Returns
        -------
        RelationExtractionResult
        """


# ---------------------------------------------------------------------------
# Linguistic pattern library
# ---------------------------------------------------------------------------

# Patterns map a regex (applied to the text *between* two entities) to a
# relation type and base confidence.  The regex is case-insensitive.
# Order matters: first match wins for a given pair.

_TREAT_PATTERNS: list[tuple[str, float]] = [
    (r"\b(?:for|to treat|treating|treats|used for|indicated for|prescribed for)\b", 0.85),
    (r"\b(?:started on|initiated|placed on|given for|administered for)\b", 0.80),
    (r"\b(?:managed with|controlled with|responding to)\b", 0.75),
]

_CAUSE_PATTERNS: list[tuple[str, float]] = [
    (r"\b(?:caused by|due to|secondary to|resulting from|attributed to)\b", 0.85),
    (r"\b(?:leads? to|resulted in|causing|produces?)\b", 0.75),
    (r"\b(?:associated with|related to|complicated by)\b", 0.60),
]

_DIAGNOSE_PATTERNS: list[tuple[str, float]] = [
    (r"\b(?:revealed|showed|demonstrated|confirmed|consistent with)\b", 0.80),
    (r"\b(?:positive for|suggestive of|indicative of|diagnostic of)\b", 0.85),
    (r"\b(?:ruled out|negative for|excludes?)\b", 0.70),
]

_LOCATION_PATTERNS: list[tuple[str, float]] = [
    (r"\b(?:in the|of the|involving|affecting|located in)\b", 0.70),
    (r"\b(?:left|right|bilateral)\b", 0.60),
]

_DOSAGE_PATTERNS: list[tuple[str, float]] = [
    (r"\s*$", 0.90),  # Dosage immediately precedes or follows medication
    (r"\b(?:of|dose|dosage)\b", 0.85),
]

_RESULT_PATTERNS: list[tuple[str, float]] = [
    (r"\b(?:showed|was|result|level|value)\b", 0.75),
    (r"[:\s]*$", 0.70),  # Lab value directly after test name with colon
]

_SIDE_EFFECT_PATTERNS: list[tuple[str, float]] = [
    (r"\b(?:side effect of|adverse effect of|adverse reaction to)\b", 0.90),
    (r"\b(?:after (?:starting|taking|beginning)|since (?:starting|taking))\b", 0.70),
]

_PREVENT_PATTERNS: list[tuple[str, float]] = [
    (r"\b(?:for prevention|to prevent|prophylaxis|prophylactic)\b", 0.85),
    (r"\b(?:preventive|protective|reduces? risk)\b", 0.75),
]

_MONITOR_PATTERNS: list[tuple[str, float]] = [
    (r"\b(?:to monitor|monitoring|for monitoring|follow[- ]?up)\b", 0.80),
    (r"\b(?:checked|rechecked|repeated|trending)\b", 0.70),
]

_WORSEN_PATTERNS: list[tuple[str, float]] = [
    (r"\b(?:worsened by|exacerbated by|aggravated by)\b", 0.85),
    (r"\b(?:worsening|deteriorat|decompensated)\b", 0.70),
]

_CONTRAINDICATE_PATTERNS: list[tuple[str, float]] = [
    (r"\b(?:contraindicated|avoid|hold|discontinue|withhold)\b", 0.80),
    (r"\b(?:allergy to|allergic to|intolerant)\b", 0.85),
]


# Master mapping: relation_type → list of (subject_type, object_type, patterns)
# Each entry describes *which entity types* the patterns apply to and the
# direction of the relation.
RELATION_PATTERNS: dict[RelationType, list[tuple[list[tuple[str, str]], list[tuple[str, float]]]]] = {
    RelationType.TREATS: [
        ([("MEDICATION", "DISEASE"), ("MEDICATION", "SYMPTOM")], _TREAT_PATTERNS),
    ],
    RelationType.CAUSES: [
        ([("DISEASE", "SYMPTOM"), ("PROCEDURE", "SYMPTOM"), ("MEDICATION", "SYMPTOM")], _CAUSE_PATTERNS),
    ],
    RelationType.DIAGNOSES: [
        ([("TEST", "DISEASE"), ("LAB_VALUE", "DISEASE")], _DIAGNOSE_PATTERNS),
    ],
    RelationType.LOCATION_OF: [
        ([("ANATOMY", "DISEASE"), ("ANATOMY", "PROCEDURE"), ("ANATOMY", "SYMPTOM")], _LOCATION_PATTERNS),
    ],
    RelationType.DOSAGE_OF: [
        ([("DOSAGE", "MEDICATION")], _DOSAGE_PATTERNS),
    ],
    RelationType.RESULT_OF: [
        ([("LAB_VALUE", "TEST")], _RESULT_PATTERNS),
    ],
    RelationType.SIDE_EFFECT_OF: [
        ([("SYMPTOM", "MEDICATION")], _SIDE_EFFECT_PATTERNS),
    ],
    RelationType.PREVENTS: [
        ([("MEDICATION", "DISEASE"), ("PROCEDURE", "DISEASE")], _PREVENT_PATTERNS),
    ],
    RelationType.MONITORS: [
        ([("TEST", "DISEASE"), ("TEST", "MEDICATION")], _MONITOR_PATTERNS),
    ],
    RelationType.WORSENS: [
        ([("DISEASE", "DISEASE"), ("DISEASE", "SYMPTOM"), ("MEDICATION", "DISEASE")], _WORSEN_PATTERNS),
    ],
    RelationType.CONTRAINDICATES: [
        ([("DISEASE", "MEDICATION"), ("DISEASE", "PROCEDURE")], _CONTRAINDICATE_PATTERNS),
    ],
}


# ---------------------------------------------------------------------------
# Rule-based extractor
# ---------------------------------------------------------------------------

class RuleBasedRelationExtractor(BaseRelationExtractor):
    """Extract entity relations using syntactic pattern matching.

    This extractor requires no ML models and runs in sub-millisecond time
    per entity pair.  It works by:

    1. Generating candidate pairs from entities within *max_distance*.
    2. Checking entity-type compatibility against :data:`RELATION_TYPE_CONSTRAINTS`.
    3. Matching the intervening text against relation-specific regex patterns.
    4. Applying proximity and co-sentence bonuses to the confidence score.

    Parameters
    ----------
    model_name:
        Identifier for provenance tracking.
    version:
        Semantic version string.
    """

    MODEL_NAME = "rule-based-relations"
    VERSION = "1.0.0"

    def __init__(
        self,
        model_name: str | None = None,
        version: str | None = None,
    ) -> None:
        self.model_name = model_name or self.MODEL_NAME
        self.version = version or self.VERSION
        # Pre-compile all patterns for speed
        self._compiled_patterns: dict[RelationType, list[tuple[list[tuple[str, str]], list[tuple[re.Pattern[str], float]]]]] = {}
        for rel_type, entries in RELATION_PATTERNS.items():
            compiled_entries = []
            for type_pairs, raw_patterns in entries:
                compiled = [(re.compile(pat, re.IGNORECASE), conf) for pat, conf in raw_patterns]
                compiled_entries.append((type_pairs, compiled))
            self._compiled_patterns[rel_type] = compiled_entries

    def extract(
        self,
        text: str,
        entities: list[Entity],
        *,
        max_distance: int = 150,
        min_confidence: float = 0.3,
    ) -> RelationExtractionResult:
        """Extract relations from *text* given pre-extracted *entities*.

        Parameters
        ----------
        text:
            The original clinical note.
        entities:
            Entities with character offsets.
        max_distance:
            Maximum character gap between entity pair to consider.
        min_confidence:
            Drop relations below this threshold.

        Returns
        -------
        RelationExtractionResult
            Sorted by descending confidence.
        """
        start = time.perf_counter()

        if len(entities) < 2:
            elapsed = (time.perf_counter() - start) * 1000
            return RelationExtractionResult(
                relations=[],
                entity_count=len(entities),
                pair_count=0,
                processing_time_ms=elapsed,
                model_name=self.model_name,
                model_version=self.version,
            )

        # Sort entities by position for windowed pairing
        sorted_entities = sorted(entities, key=lambda e: e.start_char)

        # Generate candidate pairs within distance window
        pairs: list[tuple[Entity, Entity]] = []
        for i, e1 in enumerate(sorted_entities):
            for j in range(i + 1, len(sorted_entities)):
                e2 = sorted_entities[j]
                # Distance between end of first and start of second
                gap = e2.start_char - e1.end_char
                if gap > max_distance:
                    break  # Sorted, so all further entities are farther
                if gap < 0:
                    continue  # Overlapping entities
                pairs.append((e1, e2))

        # Evaluate each pair
        relations: list[Relation] = []
        for e1, e2 in pairs:
            found = self._match_pair(text, e1, e2, min_confidence)
            relations.extend(found)

        # De-duplicate: keep highest-confidence relation per unique
        # (subject_start, object_start, relation_type) triple.
        seen: dict[tuple[int, int, str], Relation] = {}
        for rel in relations:
            key = (rel.subject.start_char, rel.object_entity.start_char, rel.relation_type.value)
            if key not in seen or rel.confidence > seen[key].confidence:
                seen[key] = rel
        unique_relations = sorted(seen.values(), key=lambda r: r.confidence, reverse=True)

        elapsed = (time.perf_counter() - start) * 1000

        return RelationExtractionResult(
            relations=unique_relations,
            entity_count=len(entities),
            pair_count=len(pairs),
            processing_time_ms=elapsed,
            model_name=self.model_name,
            model_version=self.version,
        )

    def _match_pair(
        self,
        text: str,
        e1: Entity,
        e2: Entity,
        min_confidence: float,
    ) -> list[Relation]:
        """Try all relation patterns for one entity pair.

        Both orderings (e1→e2 and e2→e1) are tested because the textual
        order doesn't always match the semantic direction.
        """
        results: list[Relation] = []
        # Text between the two entities
        between_text = text[e1.end_char:e2.start_char]

        for rel_type, entries in self._compiled_patterns.items():
            for type_pairs, compiled_patterns in entries:
                # Try e1 as subject, e2 as object
                if (e1.entity_type, e2.entity_type) in type_pairs:
                    rel = self._try_patterns(
                        rel_type, compiled_patterns, between_text,
                        subject=e1, obj=e2, min_confidence=min_confidence,
                    )
                    if rel:
                        results.append(rel)

                # Try reversed: e2 as subject, e1 as object
                if (e2.entity_type, e1.entity_type) in type_pairs:
                    rel = self._try_patterns(
                        rel_type, compiled_patterns, between_text,
                        subject=e2, obj=e1, min_confidence=min_confidence,
                    )
                    if rel:
                        results.append(rel)

        return results

    def _try_patterns(
        self,
        rel_type: RelationType,
        patterns: list[tuple[re.Pattern[str], float]],
        between_text: str,
        *,
        subject: Entity,
        obj: Entity,
        min_confidence: float,
    ) -> Relation | None:
        """Apply compiled patterns to the between-text.

        Returns the first matching :class:`Relation` or ``None``.
        """
        for pattern, base_conf in patterns:
            if pattern.search(between_text):
                # Proximity bonus: closer entities → higher confidence
                distance = abs(obj.start_char - subject.end_char)
                proximity_bonus = max(0.0, 0.1 * (1.0 - distance / 150))

                # Co-sentence bonus: if no sentence boundary between them
                sentence_bonus = 0.05 if "." not in between_text.strip() else 0.0

                confidence = min(1.0, base_conf + proximity_bonus + sentence_bonus)

                if confidence < min_confidence:
                    continue

                return Relation(
                    subject=subject,
                    object_entity=obj,
                    relation_type=rel_type,
                    confidence=confidence,
                    evidence=between_text.strip()[:200],
                    metadata={
                        "method": "rule_based",
                        "pattern": pattern.pattern,
                        "proximity_bonus": round(proximity_bonus, 4),
                        "sentence_bonus": sentence_bonus,
                    },
                )
        return None


# ---------------------------------------------------------------------------
# Transformer-based extractor (optional)
# ---------------------------------------------------------------------------

class TransformerRelationExtractor(BaseRelationExtractor):
    """Relation extraction using a fine-tuned transformer classifier.

    This extractor wraps a HuggingFace sequence-classification model trained
    on entity-pair contexts.  Input format follows the standard RE format::

        [CLS] <subject> [SEP] <between_context> [SEP] <object> [CLS]

    Falls back to :class:`RuleBasedRelationExtractor` if the model cannot be
    loaded (missing weights, missing transformers library, etc.).

    Parameters
    ----------
    model_id:
        HuggingFace model identifier or local path.
    label_map:
        Mapping from integer label indices to :class:`RelationType` values.
        If ``None``, attempts to read from model config.
    device:
        Torch device string (``"cpu"``, ``"cuda:0"``).
    """

    MODEL_NAME = "transformer-relations"
    VERSION = "1.0.0"

    def __init__(
        self,
        model_id: str = "cliniq/relation-classifier",
        label_map: dict[int, RelationType] | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.label_map = label_map or {}
        self.model_name = self.MODEL_NAME
        self.version = self.VERSION
        self._model: Any = None
        self._tokenizer: Any = None
        self._fallback = RuleBasedRelationExtractor()
        self._loaded = False
        self._load_failed = False

    def load(self) -> None:
        """Attempt to load the transformer model and tokenizer.

        If loading fails, sets ``_load_failed`` so subsequent calls use the
        rule-based fallback without retrying.
        """
        if self._loaded or self._load_failed:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            self._model.to(self.device)
            self._model.eval()

            # Build label map from config if not provided
            if not self.label_map and hasattr(self._model.config, "id2label"):
                for idx_str, label_str in self._model.config.id2label.items():
                    try:
                        self.label_map[int(idx_str)] = RelationType(label_str)
                    except (ValueError, KeyError):
                        logger.warning("Unknown relation label in model config: %s", label_str)

            self._loaded = True
            logger.info("Loaded relation extraction model: %s", self.model_id)
        except Exception:
            logger.warning(
                "Failed to load transformer relation model '%s'; "
                "falling back to rule-based extraction.",
                self.model_id,
                exc_info=True,
            )
            self._load_failed = True

    def extract(
        self,
        text: str,
        entities: list[Entity],
        *,
        max_distance: int = 150,
        min_confidence: float = 0.3,
    ) -> RelationExtractionResult:
        """Extract relations, using transformer if available.

        Parameters
        ----------
        text:
            Original clinical text.
        entities:
            Pre-extracted entities with character offsets.
        max_distance:
            Max character gap for candidate pairs.
        min_confidence:
            Minimum confidence threshold.

        Returns
        -------
        RelationExtractionResult
        """
        self.load()

        if not self._loaded:
            result = self._fallback.extract(
                text, entities,
                max_distance=max_distance,
                min_confidence=min_confidence,
            )
            # Update provenance to indicate fallback
            result.model_name = f"{self.model_name} (fallback: rule-based)"
            return result

        return self._extract_with_model(
            text, entities,
            max_distance=max_distance,
            min_confidence=min_confidence,
        )

    def _extract_with_model(
        self,
        text: str,
        entities: list[Entity],
        *,
        max_distance: int,
        min_confidence: float,
    ) -> RelationExtractionResult:
        """Run transformer inference on candidate entity pairs."""
        import torch

        start = time.perf_counter()

        sorted_entities = sorted(entities, key=lambda e: e.start_char)
        pairs: list[tuple[Entity, Entity]] = []
        for i, e1 in enumerate(sorted_entities):
            for j in range(i + 1, len(sorted_entities)):
                e2 = sorted_entities[j]
                gap = e2.start_char - e1.end_char
                if gap > max_distance:
                    break
                if gap < 0:
                    continue
                pairs.append((e1, e2))

        relations: list[Relation] = []
        for subj, obj in pairs:
            between = text[subj.end_char:obj.start_char].strip()
            input_text = f"{subj.text} [SEP] {between} [SEP] {obj.text}"

            inputs = self._tokenizer(
                input_text,
                return_tensors="pt",
                max_length=128,
                truncation=True,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                logits = self._model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze()

            best_idx = int(probs.argmax())
            confidence = float(probs[best_idx])

            if confidence < min_confidence:
                continue

            rel_type = self.label_map.get(best_idx)
            if rel_type is None:
                continue

            # Validate type constraints
            valid_pairs = RELATION_TYPE_CONSTRAINTS.get(rel_type, [])
            if (subj.entity_type, obj.entity_type) not in valid_pairs and \
               (obj.entity_type, subj.entity_type) not in valid_pairs:
                continue

            # Determine direction based on constraints
            if (subj.entity_type, obj.entity_type) in valid_pairs:
                final_subj, final_obj = subj, obj
            else:
                final_subj, final_obj = obj, subj

            relations.append(Relation(
                subject=final_subj,
                object_entity=final_obj,
                relation_type=rel_type,
                confidence=confidence,
                evidence=between[:200],
                metadata={
                    "method": "transformer",
                    "model_id": self.model_id,
                    "label_index": best_idx,
                },
            ))

        # Sort by confidence descending
        relations.sort(key=lambda r: r.confidence, reverse=True)

        elapsed = (time.perf_counter() - start) * 1000

        return RelationExtractionResult(
            relations=relations,
            entity_count=len(entities),
            pair_count=len(pairs),
            processing_time_ms=elapsed,
            model_name=self.model_name,
            model_version=self.version,
        )
