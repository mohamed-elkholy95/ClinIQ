"""Unit tests for SpacyNERModel and TransformerNERModel.

Tests entity extraction via scispaCy (mocked) and transformer NER models
(mocked) including negation/uncertainty detection, BIO tag extraction,
entity type mapping, and error propagation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.exceptions import InferenceError, ModelLoadError
from app.ml.ner.model import (
    Entity,
    SpacyNERModel,
    TransformerNERModel,
    ENTITY_TYPES,
)


# ---------------------------------------------------------------------------
# Entity dataclass
# ---------------------------------------------------------------------------

class TestEntity:
    """Tests for the Entity dataclass."""

    def test_to_dict_full(self) -> None:
        e = Entity(
            text="metformin", entity_type="MEDICATION",
            start_char=10, end_char=19, confidence=0.95,
            normalized_text="metformin hydrochloride",
            umls_cui="C0025598", is_negated=True, is_uncertain=False,
            metadata={"source": "rule"},
        )
        d = e.to_dict()
        assert d["text"] == "metformin"
        assert d["is_negated"] is True
        assert d["umls_cui"] == "C0025598"
        assert d["metadata"] == {"source": "rule"}

    def test_to_dict_defaults(self) -> None:
        e = Entity(text="aspirin", entity_type="MEDICATION",
                    start_char=0, end_char=7, confidence=0.8)
        d = e.to_dict()
        assert d["normalized_text"] is None
        assert d["is_negated"] is False
        assert d["is_uncertain"] is False


class TestEntityTypes:
    """Verify standard entity type definitions."""

    def test_disease_exists(self) -> None:
        assert "DISEASE" in ENTITY_TYPES

    def test_medication_exists(self) -> None:
        assert "MEDICATION" in ENTITY_TYPES

    def test_minimum_types(self) -> None:
        assert len(ENTITY_TYPES) >= 10


# ---------------------------------------------------------------------------
# SpacyNERModel (mocked — no real spacy installation needed)
# ---------------------------------------------------------------------------

class TestSpacyNERModel:
    """Tests for the scispaCy NER model wrapper."""

    def _make_mock_ent(self, text: str, label: str, start: int, end: int) -> MagicMock:
        ent = MagicMock()
        ent.text = text
        ent.label_ = label
        ent.start_char = start
        ent.end_char = end
        return ent

    def test_load_success(self) -> None:
        mock_spacy = MagicMock()
        mock_spacy.load.return_value = MagicMock()
        with patch.dict("sys.modules", {"spacy": mock_spacy}):
            model = SpacyNERModel(model_name="en_ner_bc5cdr_md")
            model.load()
            assert model.is_loaded

    def test_load_from_path(self) -> None:
        mock_spacy = MagicMock()
        with patch.dict("sys.modules", {"spacy": mock_spacy}):
            model = SpacyNERModel(model_path="/fake/model")
            model.load()
            mock_spacy.load.assert_called_once_with("/fake/model")

    def test_load_failure_raises(self) -> None:
        mock_spacy = MagicMock()
        mock_spacy.load.side_effect = OSError("model not found")
        with patch.dict("sys.modules", {"spacy": mock_spacy}):
            model = SpacyNERModel()
            with pytest.raises(ModelLoadError):
                model.load()

    def test_extract_entities(self) -> None:
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.ents = [
            self._make_mock_ent("hypertension", "DISEASE", 0, 12),
            self._make_mock_ent("metformin", "CHEMICAL", 25, 34),
        ]
        mock_nlp.return_value = mock_doc

        model = SpacyNERModel()
        model.nlp = mock_nlp
        model._is_loaded = True

        entities = model.extract_entities("hypertension managed with metformin daily")
        assert len(entities) == 2
        assert entities[0].entity_type == "DISEASE"
        assert entities[1].entity_type == "MEDICATION"  # CHEMICAL maps to MEDICATION

    def test_entity_type_mapping(self) -> None:
        model = SpacyNERModel()
        assert model._map_entity_type("CHEMICAL") == "MEDICATION"
        assert model._map_entity_type("ORGAN") == "ANATOMY"
        assert model._map_entity_type("DISEASE") == "DISEASE"
        assert model._map_entity_type("UNKNOWN_TYPE") == "UNKNOWN_TYPE"

    def test_negation_detection(self) -> None:
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.ents = [
            self._make_mock_ent("diabetes", "DISEASE", 15, 23),
        ]
        mock_nlp.return_value = mock_doc

        model = SpacyNERModel()
        model.nlp = mock_nlp
        model._is_loaded = True

        entities = model.extract_entities("Patient denies diabetes or related symptoms")
        assert len(entities) == 1
        assert entities[0].is_negated is True

    def test_uncertainty_detection(self) -> None:
        # "possible" is at char 21; "pneumonia" starts at char 30
        text = "Chest X-ray suggests possible pneumonia in right lower lobe"
        start = text.index("pneumonia")  # 30
        end = start + len("pneumonia")   # 39

        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.ents = [
            self._make_mock_ent("pneumonia", "DISEASE", start, end),
        ]
        mock_nlp.return_value = mock_doc

        model = SpacyNERModel()
        model.nlp = mock_nlp
        model._is_loaded = True

        entities = model.extract_entities(text)
        assert entities[0].is_uncertain is True

    def test_inference_error_propagates(self) -> None:
        mock_nlp = MagicMock(side_effect=RuntimeError("NLP failed"))

        model = SpacyNERModel()
        model.nlp = mock_nlp
        model._is_loaded = True

        with pytest.raises(InferenceError):
            model.extract_entities("some text")


# ---------------------------------------------------------------------------
# TransformerNERModel (mocked)
# ---------------------------------------------------------------------------

class TestTransformerNERModel:
    """Tests for the transformer-based NER model."""

    def test_load_success(self) -> None:
        mock_model = MagicMock()
        mock_model.config.id2label = {0: "O", 1: "B-DISEASE", 2: "I-DISEASE"}
        mock_tokenizer = MagicMock()

        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.return_value = mock_tokenizer
        mock_transformers.AutoModelForTokenClassification.from_pretrained.return_value = mock_model

        mock_torch = MagicMock()

        with patch.dict("sys.modules", {
            "torch": mock_torch,
            "transformers": mock_transformers,
        }):
            model = TransformerNERModel(model_name="test-bert")
            model.load()
            assert model.is_loaded
            assert model.label_map == {0: "O", 1: "B-DISEASE", 2: "I-DISEASE"}

    def test_load_from_path(self) -> None:
        mock_model = MagicMock()
        mock_model.config.id2label = {}
        mock_transformers = MagicMock()
        mock_transformers.AutoModelForTokenClassification.from_pretrained.return_value = mock_model

        with patch.dict("sys.modules", {
            "torch": MagicMock(),
            "transformers": mock_transformers,
        }):
            model = TransformerNERModel(model_path="/custom/path")
            model.load()
            mock_transformers.AutoTokenizer.from_pretrained.assert_called_with("/custom/path")

    def test_load_failure_raises(self) -> None:
        mock_transformers = MagicMock()
        mock_transformers.AutoTokenizer.from_pretrained.side_effect = OSError("not found")

        with patch.dict("sys.modules", {
            "torch": MagicMock(),
            "transformers": mock_transformers,
        }):
            model = TransformerNERModel()
            with pytest.raises(ModelLoadError):
                model.load()

    def test_extract_from_bio_tags(self) -> None:
        """Test BIO tag extraction logic directly."""
        model = TransformerNERModel()
        model.label_map = {0: "O", 1: "B-DISEASE", 2: "I-DISEASE", 3: "B-MEDICATION"}

        predictions = np.array([0, 1, 2, 0, 3, 0])
        offsets = np.array([
            [0, 0],    # CLS
            [0, 12],   # "hypertension" B-DISEASE
            [12, 15],  # "..." I-DISEASE
            [16, 20],  # "with" O
            [21, 30],  # "metformin" B-MEDICATION
            [0, 0],    # SEP
        ])

        text = "hypertension... with metformin"
        entities = model._extract_from_bio_tags(predictions, offsets, text)

        assert len(entities) == 2
        assert entities[0].entity_type == "DISEASE"
        assert entities[0].start_char == 0
        assert entities[0].end_char == 15
        assert entities[1].entity_type == "MEDICATION"

    def test_extract_entities_inference_error(self) -> None:
        mock_torch = MagicMock()
        mock_torch.no_grad.side_effect = RuntimeError("cuda error")

        model = TransformerNERModel()
        model._is_loaded = True
        model.tokenizer = MagicMock()
        model.tokenizer.return_value = {"input_ids": MagicMock(), "offset_mapping": MagicMock()}

        with patch.dict("sys.modules", {"torch": mock_torch}):
            with pytest.raises(InferenceError):
                model.extract_entities("some text")
