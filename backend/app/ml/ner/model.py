"""Medical Named Entity Recognition model wrapper."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from app.core.exceptions import InferenceError, ModelLoadError
from app.ml.utils.text_preprocessing import ClinicalTextPreprocessor

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents an extracted medical entity."""

    text: str
    entity_type: str
    start_char: int
    end_char: int
    confidence: float
    normalized_text: str | None = None
    umls_cui: str | None = None
    is_negated: bool = False
    is_uncertain: bool = False
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "normalized_text": self.normalized_text,
            "umls_cui": self.umls_cui,
            "is_negated": self.is_negated,
            "is_uncertain": self.is_uncertain,
            "metadata": self.metadata,
        }


# Standard entity types for medical NER
ENTITY_TYPES = {
    "DISEASE": "Diseases and disorders",
    "SYMPTOM": "Signs and symptoms",
    "MEDICATION": "Medications and drugs",
    "DOSAGE": "Medication dosages",
    "PROCEDURE": "Medical procedures",
    "ANATOMY": "Anatomical structures",
    "LAB_VALUE": "Laboratory values and results",
    "TEST": "Medical tests and examinations",
    "TREATMENT": "Treatments and therapies",
    "DEVICE": "Medical devices",
    "BODY_PART": "Body parts and regions",
    "DURATION": "Time durations",
    "FREQUENCY": "Frequency of events",
    "TEMPORAL": "Temporal expressions",
}


class BaseNERModel(ABC):
    """Abstract base class for NER models."""

    def __init__(self, model_name: str, version: str = "1.0.0"):
        self.model_name = model_name
        self.version = version
        self._is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""
        ...

    @abstractmethod
    def extract_entities(self, text: str) -> list[Entity]:
        """Extract medical entities from text."""
        ...

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded

    def ensure_loaded(self) -> None:
        """Ensure the model is loaded."""
        if not self._is_loaded:
            self.load()


class RuleBasedNERModel(BaseNERModel):
    """Rule-based NER model using patterns and dictionaries."""

    def __init__(
        self,
        model_name: str = "rule-based",
        version: str = "1.0.0",
        patterns: dict[str, list[str]] | None = None,
    ):
        super().__init__(model_name, version)
        self.patterns = patterns or self._default_patterns()
        self.preprocessor = ClinicalTextPreprocessor()
        self._compiled_patterns: dict[str, list] = {}

    def _default_patterns(self) -> dict[str, list[str]]:
        """Default entity patterns."""
        return {
            "MEDICATION": [
                r"\b(?:aspirin|ibuprofen|acetaminophen|metformin|lisinopril|atorvastatin|omeprazole|amlodipine|metoprolol|losartan)\b",
                r"\b(?:penicillin|amoxicillin|azithromycin|ciprofloxacin|doxycycline)\b",
                r"\b(?:prednisone|methylprednisolone|dexamethasone)\b",
                r"\b(?:insulin|glipizide|glyburide)\b",
                r"\b\w+(?:cillin|mycin|statin|pril|sartan|olol|pine|zole|prazole)\b",
            ],
            "DOSAGE": [
                r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|ml|g|units?|tabs?|capsules?)\b",
                r"\b(?:daily|bid|tid|qid|prn|qhs|weekly|monthly)\b",
            ],
            "LAB_VALUE": [
                r"\b(?:HbA1c|A1C)\s*(?::|:)?\s*\d+(?:\.\d+)?(?:\s*%?)?",
                r"\b(?:glucose|blood sugar)\s*(?::|:)?\s*\d+(?:\.\d+)?",
                r"\b(?:BP|blood pressure)\s*(?::|:)?\s*\d+/\d+",
                r"\b(?:WBC|RBC|hemoglobin|hematocrit|platelets?)\s*(?::|:)?\s*\d+(?:\.\d+)?",
            ],
            "PROCEDURE": [
                r"\b(?:ECG|EKG|MRI|CT scan|X-ray|ultrasound|echocardiogram|colonoscopy|endoscopy|biopsy)\b",
                r"\b\w+ectomy\b",
                r"\b\w+otomy\b",
                r"\b\w+oplasty\b",
            ],
            "TEMPORAL": [
                r"\b\d+\s*(?:days?|weeks?|months?|years?)\s*(?:ago|prior|before|earlier)\b",
                r"\b(?:yesterday|today|last week|last month|last year)\b",
                r"\b(?:since|for the past|over the last)\s+\d+\s*(?:days?|weeks?|months?)\b",
            ],
        }

    def load(self) -> None:
        """Compile patterns for faster matching."""
        import re

        for entity_type, pattern_list in self.patterns.items():
            self._compiled_patterns[entity_type] = [
                re.compile(p, re.IGNORECASE) for p in pattern_list
            ]
        self._is_loaded = True
        logger.info(f"Loaded rule-based NER model v{self.version}")

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract entities using pattern matching."""
        self.ensure_loaded()
        entities = []

        for entity_type, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entities.append(
                        Entity(
                            text=match.group(),
                            entity_type=entity_type,
                            start_char=match.start(),
                            end_char=match.end(),
                            confidence=0.9,  # High confidence for rule matches
                        )
                    )

        # Sort by position and remove overlaps
        entities = self._resolve_overlaps(entities)

        # Detect negation and uncertainty
        entities = self._detect_modifiers(text, entities)

        return entities

    def _resolve_overlaps(self, entities: list[Entity]) -> list[Entity]:
        """Resolve overlapping entities by keeping higher confidence ones."""
        if not entities:
            return entities

        # Sort by start position
        entities.sort(key=lambda e: (e.start_char, -e.confidence))

        resolved = []
        for entity in entities:
            # Check if this entity overlaps with any already resolved
            overlaps = False
            for existing in resolved:
                if self._entities_overlap(entity, existing):
                    overlaps = True
                    break

            if not overlaps:
                resolved.append(entity)

        return resolved

    def _entities_overlap(self, e1: Entity, e2: Entity) -> bool:
        """Check if two entities overlap."""
        return e1.start_char < e2.end_char and e2.start_char < e1.end_char

    def _detect_modifiers(self, text: str, entities: list[Entity]) -> list[Entity]:
        """Detect negation and uncertainty modifiers."""
        negation_patterns = [
            r"\bno\s+",
            r"\bnot\s+",
            r"\bdenies?\s+",
            r"\bwithout\s+",
            r"\bnegative\s+for\s+",
            r"\babsence\s+of\s+",
        ]

        uncertainty_patterns = [
            r"\bpossible\s+",
            r"\bprobable\s+",
            r"\blikely\s+",
            r"\bsuspected\s+",
            r"\bmay\s+have\s+",
            r"\bmight\s+be\s+",
            r"\bconcern\s+(?:for|about)\s+",
            r"\brule\s+out\s+",
        ]

        import re

        for entity in entities:
            # Check text before entity for modifiers
            context_start = max(0, entity.start_char - 50)
            context = text[context_start : entity.start_char].lower()

            for pattern in negation_patterns:
                if re.search(pattern, context):
                    entity.is_negated = True
                    break

            for pattern in uncertainty_patterns:
                if re.search(pattern, context):
                    entity.is_uncertain = True
                    break

        return entities


class SpacyNERModel(BaseNERModel):
    """NER model using scispaCy."""

    def __init__(
        self,
        model_name: str = "en_ner_bc5cdr_md",
        version: str = "0.5.3",
        model_path: str | None = None,
    ):
        super().__init__(model_name, version)
        self.model_path = model_path
        self.nlp: Any = None
        self.preprocessor = ClinicalTextPreprocessor()

    def load(self) -> None:
        """Load the scispaCy model."""
        try:
            import spacy

            if self.model_path:
                self.nlp = spacy.load(self.model_path)
            else:
                # Try to load by name
                self.nlp = spacy.load(self.model_name)

            self._is_loaded = True
            logger.info(f"Loaded scispaCy model: {self.model_name}")
        except Exception as e:
            raise ModelLoadError(self.model_name, str(e))

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract entities using scispaCy."""
        self.ensure_loaded()

        try:
            doc = self.nlp(text)
            entities = []

            for ent in doc.ents:
                # Map scispaCy entity types to our standard types
                entity_type = self._map_entity_type(ent.label_)

                entities.append(
                    Entity(
                        text=ent.text,
                        entity_type=entity_type,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        confidence=0.85,  # Default confidence for scispaCy
                        normalized_text=None,
                    )
                )

            # Detect negation and uncertainty
            entities = self._detect_modifiers(text, entities)

            return entities

        except Exception as e:
            raise InferenceError(self.model_name, str(e))

    def _map_entity_type(self, scispacy_type: str) -> str:
        """Map scispaCy entity types to standard types."""
        mapping = {
            "DISEASE": "DISEASE",
            "CHEMICAL": "MEDICATION",
            "ORGAN": "ANATOMY",
            "CELL_TYPE": "ANATOMY",
            "CELL_LINE": "ANATOMY",
            "PROTEIN": "MEDICATION",
            "DNA": "ANATOMY",
            "RNA": "ANATOMY",
        }
        return mapping.get(scispacy_type, scispacy_type)

    def _detect_modifiers(self, text: str, entities: list[Entity]) -> list[Entity]:
        """Detect negation using NegEx-like patterns."""
        # Simplified negation detection
        import re

        for entity in entities:
            context_start = max(0, entity.start_char - 50)
            context = text[context_start : entity.start_char].lower()

            if re.search(r"\b(no|not|denies?|without|negative)\s+", context):
                entity.is_negated = True

            if re.search(r"\b(possible|probable|likely|suspected|may|might)\s+", context):
                entity.is_uncertain = True

        return entities


class TransformerNERModel(BaseNERModel):
    """NER model using fine-tuned transformers (BioBERT, etc.)."""

    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-base-cased-v1.1",
        version: str = "1.0.0",
        model_path: str | None = None,
        device: str = "cpu",
    ):
        super().__init__(model_name, version)
        self.model_path = model_path
        self.device = device
        self.tokenizer: Any = None
        self.model: Any = None
        self.label_map: dict[int, str] = {}

    def load(self) -> None:
        """Load the transformer model."""
        try:
            from transformers import AutoModelForTokenClassification, AutoTokenizer

            model_path = self.model_path or self.model_name

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()

            # Build label map from model config
            if hasattr(self.model.config, "id2label"):
                self.label_map = self.model.config.id2label

            self._is_loaded = True
            logger.info(f"Loaded transformer NER model: {model_path}")

        except Exception as e:
            raise ModelLoadError(self.model_name, str(e))

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract entities using transformer model."""
        self.ensure_loaded()

        try:
            import torch

            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                return_offsets_mapping=True,
            )

            offset_mapping = inputs.pop("offset_mapping")[0]
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)[0]

            # Extract entities using BIO tagging
            entities = self._extract_from_bio_tags(
                predictions.cpu().numpy(),
                offset_mapping.cpu().numpy(),
                text,
            )

            return entities

        except Exception as e:
            raise InferenceError(self.model_name, str(e))

    def _extract_from_bio_tags(
        self,
        predictions: np.ndarray,
        offsets: np.ndarray,
        text: str,
    ) -> list[Entity]:
        """Extract entities from BIO-tagged predictions."""
        entities = []
        current_entity: dict[str, Any] | None = None

        for i, (pred, (start, end)) in enumerate(zip(predictions, offsets, strict=False)):
            if start == end:  # Special token
                continue

            label = self.label_map.get(pred, "O")

            if label.startswith("B-"):
                # Beginning of new entity
                if current_entity:
                    entities.append(self._create_entity(current_entity, text))
                current_entity = {
                    "type": label[2:],
                    "start": start,
                    "end": end,
                    "tokens": [i],
                }
            elif label.startswith("I-") and current_entity:
                # Continuation of entity
                current_entity["end"] = end
                current_entity["tokens"].append(i)
            else:
                # Outside entity
                if current_entity:
                    entities.append(self._create_entity(current_entity, text))
                    current_entity = None

        # Don't forget last entity
        if current_entity:
            entities.append(self._create_entity(current_entity, text))

        return entities

    def _create_entity(self, entity_info: dict, text: str) -> Entity:
        """Create Entity object from entity info dict."""
        entity_text = text[entity_info["start"] : entity_info["end"]]
        return Entity(
            text=entity_text,
            entity_type=entity_info["type"],
            start_char=entity_info["start"],
            end_char=entity_info["end"],
            confidence=0.9,  # Could compute from logits
        )


class CompositeNERModel(BaseNERModel):
    """Combine multiple NER models for better coverage."""

    def __init__(
        self,
        models: list[BaseNERModel],
        voting: str = "union",
        model_name: str = "composite-ner",
        version: str = "1.0.0",
    ):
        super().__init__(model_name, version)
        self.models = models
        self.voting = voting  # "union", "intersection", "majority"

    def load(self) -> None:
        """Load all sub-models."""
        for model in self.models:
            model.load()
        self._is_loaded = True

    def extract_entities(self, text: str) -> list[Entity]:
        """Extract entities using all models."""
        all_entities = []

        for model in self.models:
            entities = model.extract_entities(text)
            all_entities.append(entities)

        if self.voting == "union":
            return self._union_vote(all_entities)
        elif self.voting == "intersection":
            return self._intersection_vote(all_entities)
        else:  # majority
            return self._majority_vote(all_entities)

    def _union_vote(self, all_entities: list[list[Entity]]) -> list[Entity]:
        """Return all entities from all models."""
        result = []
        seen = set()

        for entities in all_entities:
            for entity in entities:
                key = (entity.text, entity.entity_type, entity.start_char)
                if key not in seen:
                    seen.add(key)
                    result.append(entity)

        return self._resolve_overlaps(result)

    def _intersection_vote(self, all_entities: list[list[Entity]]) -> list[Entity]:
        """Return only entities found by all models."""
        if not all_entities:
            return []

        # Find entities present in all model outputs
        entity_sets = [
            {(e.text, e.entity_type, e.start_char, e.end_char) for e in entities}
            for entities in all_entities
        ]

        common = set.intersection(*entity_sets) if entity_sets else set()

        # Return entities that are in the common set
        result = []
        for entity in all_entities[0]:
            key = (entity.text, entity.entity_type, entity.start_char, entity.end_char)
            if key in common:
                result.append(entity)

        return result

    def _majority_vote(self, all_entities: list[list[Entity]]) -> list[Entity]:
        """Return entities found by majority of models."""
        from collections import Counter

        # Count occurrences
        entity_counts: Counter = Counter()
        entity_map: dict[tuple, Entity] = {}

        for entities in all_entities:
            seen_this_model = set()
            for entity in entities:
                key = (entity.text, entity.entity_type, entity.start_char, entity.end_char)
                if key not in seen_this_model:
                    seen_this_model.add(key)
                    entity_counts[key] += 1
                    if key not in entity_map:
                        entity_map[key] = entity

        # Keep entities with majority support
        threshold = len(all_entities) / 2
        result = [
            entity_map[key]
            for key, count in entity_counts.items()
            if count > threshold
        ]

        return self._resolve_overlaps(result)

    def _resolve_overlaps(self, entities: list[Entity]) -> list[Entity]:
        """Resolve overlapping entities."""
        if not entities:
            return entities

        entities.sort(key=lambda e: (e.start_char, -e.confidence))
        resolved = []

        for entity in entities:
            overlaps = False
            for existing in resolved:
                if entity.start_char < existing.end_char and existing.start_char < entity.end_char:
                    overlaps = True
                    break
            if not overlaps:
                resolved.append(entity)

        return resolved
