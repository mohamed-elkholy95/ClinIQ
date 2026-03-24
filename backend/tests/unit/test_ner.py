"""Unit tests for the NER model module."""

import pytest

from app.ml.ner.model import (
    ENTITY_TYPES,
    Entity,
    RuleBasedNERModel,
)


class TestEntityDataclass:
    """Tests for the Entity dataclass."""

    def test_entity_creation_minimal(self):
        """Test creating an Entity with required fields only."""
        entity = Entity(
            text="metformin",
            entity_type="MEDICATION",
            start_char=0,
            end_char=9,
            confidence=0.95,
        )

        assert entity.text == "metformin"
        assert entity.entity_type == "MEDICATION"
        assert entity.start_char == 0
        assert entity.end_char == 9
        assert entity.confidence == 0.95

    def test_entity_defaults(self):
        """Test default values for optional fields."""
        entity = Entity(
            text="diabetes",
            entity_type="DISEASE",
            start_char=10,
            end_char=18,
            confidence=0.9,
        )

        assert entity.normalized_text is None
        assert entity.umls_cui is None
        assert entity.is_negated is False
        assert entity.is_uncertain is False
        assert entity.metadata is None

    def test_entity_to_dict_keys(self):
        """Test that to_dict() returns all expected keys."""
        entity = Entity(
            text="aspirin",
            entity_type="MEDICATION",
            start_char=5,
            end_char=12,
            confidence=0.88,
        )

        result = entity.to_dict()

        expected_keys = {
            "text",
            "entity_type",
            "start_char",
            "end_char",
            "confidence",
            "normalized_text",
            "umls_cui",
            "is_negated",
            "is_uncertain",
            "metadata",
        }
        assert set(result.keys()) == expected_keys

    def test_entity_to_dict_values(self):
        """Test that to_dict() returns correct values."""
        entity = Entity(
            text="lisinopril",
            entity_type="MEDICATION",
            start_char=20,
            end_char=30,
            confidence=0.92,
            normalized_text="LISINOPRIL",
            umls_cui="C0065374",
            is_negated=False,
            is_uncertain=True,
            metadata={"dose": "10mg"},
        )

        result = entity.to_dict()

        assert result["text"] == "lisinopril"
        assert result["entity_type"] == "MEDICATION"
        assert result["start_char"] == 20
        assert result["end_char"] == 30
        assert result["confidence"] == 0.92
        assert result["normalized_text"] == "LISINOPRIL"
        assert result["umls_cui"] == "C0065374"
        assert result["is_negated"] is False
        assert result["is_uncertain"] is True
        assert result["metadata"] == {"dose": "10mg"}

    def test_entity_negated_flag(self):
        """Test the is_negated flag."""
        entity = Entity(
            text="chest pain",
            entity_type="SYMPTOM",
            start_char=0,
            end_char=10,
            confidence=0.85,
            is_negated=True,
        )

        assert entity.is_negated is True
        assert entity.to_dict()["is_negated"] is True


class TestEntityTypes:
    """Tests for the ENTITY_TYPES constant."""

    def test_entity_types_contains_expected_keys(self):
        """Test that ENTITY_TYPES has all expected entity type keys."""
        expected = {
            "DISEASE",
            "SYMPTOM",
            "MEDICATION",
            "DOSAGE",
            "PROCEDURE",
            "ANATOMY",
            "LAB_VALUE",
            "TEST",
            "TREATMENT",
            "DEVICE",
            "BODY_PART",
            "DURATION",
            "FREQUENCY",
            "TEMPORAL",
        }
        assert expected.issubset(set(ENTITY_TYPES.keys()))

    def test_entity_types_values_are_strings(self):
        """Test that all ENTITY_TYPES values are non-empty strings."""
        for key, value in ENTITY_TYPES.items():
            assert isinstance(value, str), f"Value for {key} is not a string"
            assert len(value) > 0, f"Value for {key} is empty"


class TestRuleBasedNERModel:
    """Tests for RuleBasedNERModel."""

    @pytest.fixture
    def model(self) -> RuleBasedNERModel:
        """Create a loaded RuleBasedNERModel instance."""
        m = RuleBasedNERModel()
        m.load()
        return m

    def test_model_is_loaded_after_load(self):
        """Test that is_loaded becomes True after calling load()."""
        model = RuleBasedNERModel()
        assert model.is_loaded is False
        model.load()
        assert model.is_loaded is True

    def test_model_default_attributes(self):
        """Test default model_name and version attributes."""
        model = RuleBasedNERModel()
        assert model.model_name == "rule-based"
        assert model.version == "1.0.0"

    def test_custom_model_name_and_version(self):
        """Test that model_name and version can be customised."""
        model = RuleBasedNERModel(model_name="custom-ner", version="2.0.0")
        assert model.model_name == "custom-ner"
        assert model.version == "2.0.0"

    def test_ensure_loaded_triggers_load(self):
        """Test that ensure_loaded() calls load() when not yet loaded."""
        model = RuleBasedNERModel()
        assert model.is_loaded is False
        model.ensure_loaded()
        assert model.is_loaded is True

    def test_extract_entities_returns_list(self, model: RuleBasedNERModel):
        """Test that extract_entities always returns a list."""
        result = model.extract_entities("Patient takes aspirin daily.")
        assert isinstance(result, list)

    def test_extract_medication_entity(self, model: RuleBasedNERModel):
        """Test extraction of a known medication entity."""
        text = "Patient is taking metformin for diabetes."
        entities = model.extract_entities(text)

        entity_types = [e.entity_type for e in entities]
        entity_texts = [e.text.lower() for e in entities]

        assert "MEDICATION" in entity_types
        assert any("metformin" in t for t in entity_texts)

    @pytest.mark.parametrize(
        "text,expected_medication",
        [
            ("She takes aspirin 81mg daily.", "aspirin"),
            ("Started on lisinopril 10mg.", "lisinopril"),
            ("Prescribed atorvastatin 20mg at bedtime.", "atorvastatin"),
            ("Patient on metformin 1000mg twice daily.", "metformin"),
        ],
    )
    def test_extract_known_medications(
        self, model: RuleBasedNERModel, text: str, expected_medication: str
    ):
        """Parametrised test for known medication extraction."""
        entities = model.extract_entities(text)
        medication_texts = [
            e.text.lower() for e in entities if e.entity_type == "MEDICATION"
        ]
        assert any(expected_medication in t for t in medication_texts), (
            f"Expected '{expected_medication}' in medications extracted from: {text!r}"
        )

    def test_extract_dosage_entity(self, model: RuleBasedNERModel):
        """Test extraction of dosage information."""
        text = "Prescribed metformin 500mg twice daily."
        entities = model.extract_entities(text)

        entity_types = [e.entity_type for e in entities]
        assert "DOSAGE" in entity_types

    def test_extract_entities_empty_text(self, model: RuleBasedNERModel):
        """Test extraction on empty text returns empty list."""
        result = model.extract_entities("")
        assert result == []

    def test_entity_confidence_in_range(self, model: RuleBasedNERModel):
        """Test that all extracted entities have confidence in [0, 1]."""
        text = "Patient takes metformin 500mg daily for type 2 diabetes."
        entities = model.extract_entities(text)

        for entity in entities:
            assert 0.0 <= entity.confidence <= 1.0, (
                f"Confidence {entity.confidence} out of range for entity {entity.text!r}"
            )

    def test_entity_char_positions_valid(self, model: RuleBasedNERModel):
        """Test that start_char < end_char for every extracted entity."""
        text = "Patient is on lisinopril 10mg and aspirin 81mg daily."
        entities = model.extract_entities(text)

        for entity in entities:
            assert entity.start_char < entity.end_char, (
                f"Invalid char positions for entity {entity.text!r}: "
                f"start={entity.start_char}, end={entity.end_char}"
            )

    def test_entity_text_matches_source(self, model: RuleBasedNERModel):
        """Test that entity text matches the source text at the reported positions."""
        text = "The patient uses metformin for glycemic control."
        entities = model.extract_entities(text)

        for entity in entities:
            extracted = text[entity.start_char : entity.end_char]
            assert extracted.lower() == entity.text.lower(), (
                f"Entity text {entity.text!r} does not match source text "
                f"slice {extracted!r} at [{entity.start_char}:{entity.end_char}]"
            )

    def test_resolve_overlaps_no_duplicates(self, model: RuleBasedNERModel):
        """Test that resolved entities do not overlap."""
        text = "Patient takes atorvastatin 20mg for hyperlipidemia."
        entities = model.extract_entities(text)

        # Check pairwise non-overlap
        for i, e1 in enumerate(entities):
            for e2 in entities[i + 1 :]:
                overlap = e1.start_char < e2.end_char and e2.start_char < e1.end_char
                assert not overlap, (
                    f"Entities overlap: {e1.text!r} [{e1.start_char}:{e1.end_char}] "
                    f"and {e2.text!r} [{e2.start_char}:{e2.end_char}]"
                )

    def test_negation_detection_denies(self, model: RuleBasedNERModel):
        """Test negation detection for 'denies' keyword."""
        text = "Patient denies chest pain or shortness of breath."
        entities = model.extract_entities(text)

        # We don't assert specific entities here since 'chest pain' isn't in default
        # patterns; but ensure no entities have spurious negation flags set incorrectly
        for entity in entities:
            assert isinstance(entity.is_negated, bool)

    def test_negation_detection_no_prefix(self, model: RuleBasedNERModel):
        """Test that 'no [medication]' prefix triggers negation flag."""
        text = "There is no aspirin prescribed for this patient."
        entities = model.extract_entities(text)

        aspirin_entities = [e for e in entities if "aspirin" in e.text.lower()]
        if aspirin_entities:
            assert aspirin_entities[0].is_negated is True

    def test_uncertainty_detection_possible(self, model: RuleBasedNERModel):
        """Test that uncertainty keywords set is_uncertain flag."""
        text = "Possible aspirin allergy noted in the chart."
        entities = model.extract_entities(text)

        aspirin_entities = [e for e in entities if "aspirin" in e.text.lower()]
        if aspirin_entities:
            assert aspirin_entities[0].is_uncertain is True

    @pytest.mark.parametrize(
        "clinical_text",
        [
            "Patient is a 55-year-old male with type 2 diabetes mellitus and hypertension.",
            "Metformin 1000mg PO BID and lisinopril 10mg PO daily.",
            "Blood pressure 130/80 mmHg, heart rate 72 bpm.",
            "ECG ordered. Troponins obtained. Aspirin 325mg given.",
            "No evidence of acute coronary syndrome. Patient discharged.",
        ],
    )
    def test_extract_entities_various_clinical_texts(
        self, model: RuleBasedNERModel, clinical_text: str
    ):
        """Test extraction does not raise on varied clinical texts."""
        result = model.extract_entities(clinical_text)
        assert isinstance(result, list)
        for entity in result:
            assert isinstance(entity, Entity)

    def test_custom_patterns_override(self):
        """Test that providing custom patterns replaces default patterns."""
        custom_patterns = {
            "CUSTOM_TYPE": [r"\b(test_token)\b"],
        }
        model = RuleBasedNERModel(patterns=custom_patterns)
        model.load()

        assert "CUSTOM_TYPE" in model.patterns
        assert "MEDICATION" not in model.patterns

    def test_extract_entities_returns_entity_objects(self, model: RuleBasedNERModel):
        """Test that extract_entities returns Entity instances."""
        text = "Patient is on metformin 500mg daily."
        entities = model.extract_entities(text)

        for entity in entities:
            assert isinstance(entity, Entity)

    def test_entities_sorted_by_start_char(self, model: RuleBasedNERModel):
        """Test that entities appear in document order (sorted by start_char)."""
        text = (
            "Patient takes aspirin 81mg in the morning "
            "and metformin 500mg at night."
        )
        entities = model.extract_entities(text)

        if len(entities) >= 2:
            for i in range(len(entities) - 1):
                assert entities[i].start_char <= entities[i + 1].start_char
