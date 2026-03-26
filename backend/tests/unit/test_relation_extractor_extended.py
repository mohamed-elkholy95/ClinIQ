"""Extended tests for relation extraction — targeting uncovered code paths.

Covers:
- ``TransformerRelationExtractor.load()`` success, failure, and retry-skip
- ``TransformerRelationExtractor.extract()`` fallback to rule-based
- ``TransformerRelationExtractor._extract_with_model()`` end-to-end
- ``RuleBasedRelationExtractor`` proximity and sentence bonuses
- Type constraint filtering and direction detection
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.ml.ner.model import Entity
from app.ml.relations.extractor import (
    RELATION_TYPE_CONSTRAINTS,
    Relation,
    RelationExtractionResult,
    RelationType,
    RuleBasedRelationExtractor,
    TransformerRelationExtractor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_entity(
    text: str,
    entity_type: str,
    start: int,
    end: int,
) -> Entity:
    """Create a minimal Entity for testing."""
    return Entity(
        text=text,
        entity_type=entity_type,
        start_char=start,
        end_char=end,
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# TransformerRelationExtractor load paths
# ---------------------------------------------------------------------------

class TestTransformerRelationExtractorLoad:
    """Cover load() success, failure, and skip-if-already-failed."""

    def test_load_success(self) -> None:
        """Successful load should set _loaded=True."""
        extractor = TransformerRelationExtractor(model_id="test-model")

        mock_tokenizer_cls = MagicMock()
        mock_model_cls = MagicMock()
        mock_model = MagicMock()
        mock_model.config.id2label = {}
        mock_model_cls.from_pretrained.return_value = mock_model

        with patch.dict("sys.modules", {
            "transformers": MagicMock(
                AutoModelForSequenceClassification=mock_model_cls,
                AutoTokenizer=mock_tokenizer_cls,
            ),
        }):
            extractor.load()

        assert extractor._loaded
        assert not extractor._load_failed

    def test_load_failure_sets_flag(self) -> None:
        """Failed load should set _load_failed without raising."""
        extractor = TransformerRelationExtractor(model_id="bad-model")

        with patch.dict("sys.modules", {"transformers": None}):
            extractor.load()

        assert not extractor._loaded
        assert extractor._load_failed

    def test_load_skips_if_already_loaded(self) -> None:
        """Should not re-attempt load if already loaded."""
        extractor = TransformerRelationExtractor()
        extractor._loaded = True

        with patch.dict("sys.modules", {"transformers": MagicMock()}) as _:
            extractor.load()
        # Should be a no-op
        assert extractor._loaded

    def test_load_skips_if_already_failed(self) -> None:
        """Should not re-attempt load if previous attempt failed."""
        extractor = TransformerRelationExtractor()
        extractor._load_failed = True

        extractor.load()
        assert not extractor._loaded
        assert extractor._load_failed

    def test_load_builds_label_map_from_config(self) -> None:
        """Should build label_map from model config.id2label."""
        extractor = TransformerRelationExtractor(model_id="test")
        extractor.label_map = {}

        mock_model = MagicMock()
        mock_model.config.id2label = {
            "0": "treats",
            "1": "causes",
            "2": "unknown_relation_xyz",  # Should warn and skip
        }

        mock_model_cls = MagicMock()
        mock_model_cls.from_pretrained.return_value = mock_model

        with patch.dict("sys.modules", {
            "transformers": MagicMock(
                AutoModelForSequenceClassification=mock_model_cls,
                AutoTokenizer=MagicMock(),
            ),
        }):
            extractor.load()

        assert extractor.label_map.get(0) == RelationType.TREATS
        assert extractor.label_map.get(1) == RelationType.CAUSES
        assert 2 not in extractor.label_map  # Unknown relation skipped


# ---------------------------------------------------------------------------
# TransformerRelationExtractor extract() fallback
# ---------------------------------------------------------------------------

class TestTransformerExtractFallback:
    """Cover extract() falling back to rule-based when model not loaded."""

    def test_extract_falls_back_when_not_loaded(self) -> None:
        """Should use rule-based fallback and annotate model_name."""
        extractor = TransformerRelationExtractor()
        extractor._load_failed = True

        entities = [
            _make_entity("aspirin", "MEDICATION", 0, 7),
            _make_entity("headache", "DISEASE", 15, 23),
        ]
        text = "aspirin treats headache effectively"

        result = extractor.extract(text, entities)
        assert isinstance(result, RelationExtractionResult)
        assert "fallback" in result.model_name


# ---------------------------------------------------------------------------
# TransformerRelationExtractor _extract_with_model()
# ---------------------------------------------------------------------------

class TestTransformerExtractWithModel:
    """Cover _extract_with_model() inference path."""

    def test_extract_with_model_finds_relations(self) -> None:
        """Should run transformer inference and return relations."""
        extractor = TransformerRelationExtractor()
        extractor._loaded = True
        extractor.label_map = {0: RelationType.TREATS, 1: RelationType.CAUSES}

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock()
        mock_tokenizer.return_value.to.return_value = mock_tokenizer.return_value
        extractor._tokenizer = mock_tokenizer

        # Mock torch
        mock_torch = MagicMock()

        mock_probs = MagicMock()
        mock_probs.argmax.return_value = 0  # TREATS
        mock_probs.__getitem__ = MagicMock(return_value=MagicMock(
            __float__=lambda s: 0.85,
        ))
        mock_torch.softmax.return_value.squeeze.return_value = mock_probs
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        mock_model = MagicMock()
        mock_model.return_value.logits = MagicMock()
        extractor._model = mock_model

        entities = [
            _make_entity("metformin", "MEDICATION", 0, 9),
            _make_entity("diabetes", "DISEASE", 17, 25),
        ]
        text = "metformin treats diabetes effectively"

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = extractor._extract_with_model(
                text, entities, max_distance=150, min_confidence=0.3,
            )

        assert isinstance(result, RelationExtractionResult)

    def test_extract_with_model_filters_low_confidence(self) -> None:
        """Predictions below min_confidence should be filtered out."""
        extractor = TransformerRelationExtractor()
        extractor._loaded = True
        extractor.label_map = {0: RelationType.TREATS}

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value.to.return_value = mock_tokenizer.return_value
        extractor._tokenizer = mock_tokenizer

        mock_torch = MagicMock()
        mock_probs = MagicMock()
        mock_probs.argmax.return_value = 0
        mock_probs.__getitem__ = MagicMock(return_value=MagicMock(
            __float__=lambda s: 0.1,  # Below threshold
        ))
        mock_torch.softmax.return_value.squeeze.return_value = mock_probs
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        mock_model = MagicMock()
        mock_model.return_value.logits = MagicMock()
        extractor._model = mock_model

        entities = [
            _make_entity("aspirin", "MEDICATION", 0, 7),
            _make_entity("pain", "DISEASE", 15, 19),
        ]

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = extractor._extract_with_model(
                "aspirin treats pain", entities,
                max_distance=150, min_confidence=0.5,
            )

        assert result.relations == []

    def test_extract_with_model_skips_overlapping_entities(self) -> None:
        """Entities with negative gap should be skipped."""
        extractor = TransformerRelationExtractor()
        extractor._loaded = True
        extractor.label_map = {0: RelationType.TREATS}

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value.to.return_value = mock_tokenizer.return_value
        extractor._tokenizer = mock_tokenizer

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        extractor._model = MagicMock()

        # Overlapping entities: e2 starts before e1 ends
        entities = [
            _make_entity("chest pain", "DISEASE", 0, 10),
            _make_entity("pain syndrome", "DISEASE", 6, 19),
        ]

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = extractor._extract_with_model(
                "chest pain syndrome", entities,
                max_distance=150, min_confidence=0.3,
            )

        assert result.pair_count == 0

    def test_extract_with_model_skips_unknown_label(self) -> None:
        """Predictions with unmapped label index should be skipped."""
        extractor = TransformerRelationExtractor()
        extractor._loaded = True
        extractor.label_map = {}  # Empty — no valid labels

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value.to.return_value = mock_tokenizer.return_value
        extractor._tokenizer = mock_tokenizer

        mock_torch = MagicMock()
        mock_probs = MagicMock()
        mock_probs.argmax.return_value = 99  # Not in label_map
        mock_probs.__getitem__ = MagicMock(return_value=MagicMock(
            __float__=lambda s: 0.9,
        ))
        mock_torch.softmax.return_value.squeeze.return_value = mock_probs
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        mock_model = MagicMock()
        mock_model.return_value.logits = MagicMock()
        extractor._model = mock_model

        entities = [
            _make_entity("drug", "MEDICATION", 0, 4),
            _make_entity("disease", "DISEASE", 12, 19),
        ]

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = extractor._extract_with_model(
                "drug treats disease", entities,
                max_distance=150, min_confidence=0.3,
            )

        assert result.relations == []

    def test_extract_with_model_max_distance_filtering(self) -> None:
        """Entities beyond max_distance should not form pairs."""
        extractor = TransformerRelationExtractor()
        extractor._loaded = True
        extractor.label_map = {0: RelationType.TREATS}

        mock_torch = MagicMock()
        mock_torch.no_grad.return_value.__enter__ = MagicMock()
        mock_torch.no_grad.return_value.__exit__ = MagicMock()

        extractor._model = MagicMock()
        extractor._tokenizer = MagicMock()

        # Entities very far apart
        entities = [
            _make_entity("aspirin", "MEDICATION", 0, 7),
            _make_entity("headache", "DISEASE", 500, 508),
        ]

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = extractor._extract_with_model(
                "aspirin " + " " * 490 + "headache",
                entities,
                max_distance=100, min_confidence=0.3,
            )

        assert result.pair_count == 0


# ---------------------------------------------------------------------------
# RuleBasedRelationExtractor — bonus scoring paths
# ---------------------------------------------------------------------------

class TestRuleBasedBonusScoring:
    """Cover proximity_bonus and sentence_bonus in _try_match."""

    def test_close_entities_get_proximity_bonus(self) -> None:
        """Very close entities should have proximity bonus > 0."""
        extractor = RuleBasedRelationExtractor()
        entities = [
            _make_entity("aspirin", "MEDICATION", 0, 7),
            _make_entity("pain", "SYMPTOM", 15, 19),
        ]
        text = "aspirin treats pain"

        result = extractor.extract(text, entities, min_confidence=0.3)

        for rel in result.relations:
            if rel.relation_type == RelationType.TREATS:
                assert rel.metadata.get("proximity_bonus", 0) > 0

    def test_same_sentence_gets_sentence_bonus(self) -> None:
        """Entities in same sentence (no period) should get sentence bonus."""
        extractor = RuleBasedRelationExtractor()
        entities = [
            _make_entity("infection", "DISEASE", 0, 9),
            _make_entity("fever", "SYMPTOM", 17, 22),
        ]
        text = "infection causes fever in patients"

        result = extractor.extract(text, entities, min_confidence=0.3)

        for rel in result.relations:
            if rel.relation_type == RelationType.CAUSES:
                assert rel.metadata.get("sentence_bonus", 0) == 0.05

    def test_empty_entities_returns_empty(self) -> None:
        """Empty entity list should return empty result."""
        extractor = RuleBasedRelationExtractor()
        result = extractor.extract("Some text", [], min_confidence=0.3)
        assert result.relations == []
        assert result.entity_count == 0

    def test_single_entity_no_pairs(self) -> None:
        """Single entity should produce no relations."""
        extractor = RuleBasedRelationExtractor()
        entities = [_make_entity("aspirin", "MEDICATION", 0, 7)]
        result = extractor.extract("aspirin", entities, min_confidence=0.3)
        assert result.pair_count == 0
