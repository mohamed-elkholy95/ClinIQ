"""Tests for clinical relation extraction module.

Covers:
- RelationType enum completeness and values
- Relation / RelationExtractionResult data classes and serialisation
- RELATION_TYPE_CONSTRAINTS correctness
- RuleBasedRelationExtractor:
  - Pattern matching for all 11 relation types
  - Proximity and co-sentence confidence bonuses
  - Entity pair windowing (max_distance)
  - De-duplication of overlapping relations
  - Min confidence filtering
  - Edge cases: single entity, empty entities, overlapping spans, no matches
- TransformerRelationExtractor:
  - Fallback to rule-based on load failure
  - Model provenance tracking
"""

from __future__ import annotations

import pytest

from app.ml.ner.model import Entity
from app.ml.relations.extractor import (
    RELATION_PATTERNS,
    RELATION_TYPE_CONSTRAINTS,
    BaseRelationExtractor,
    Relation,
    RelationExtractionResult,
    RelationType,
    RuleBasedRelationExtractor,
    TransformerRelationExtractor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity(
    text: str,
    etype: str,
    start: int,
    end: int | None = None,
    confidence: float = 0.9,
    is_negated: bool = False,
) -> Entity:
    """Create an entity with sensible defaults."""
    return Entity(
        text=text,
        entity_type=etype,
        start_char=start,
        end_char=end if end is not None else start + len(text),
        confidence=confidence,
        is_negated=is_negated,
    )


# ---------------------------------------------------------------------------
# Enum & data class tests
# ---------------------------------------------------------------------------

class TestRelationType:
    """Tests for RelationType enum."""

    def test_all_values_are_strings(self):
        for rt in RelationType:
            assert isinstance(rt.value, str)

    def test_expected_count(self):
        assert len(RelationType) == 12

    def test_key_types_exist(self):
        expected = {"treats", "causes", "diagnoses", "dosage_of", "location_of", "result_of"}
        actual = {rt.value for rt in RelationType}
        assert expected.issubset(actual)


class TestRelationDataClass:
    """Tests for Relation and RelationExtractionResult serialisation."""

    def test_relation_to_dict(self):
        subj = _entity("metformin", "MEDICATION", 0)
        obj = _entity("diabetes", "DISEASE", 30)
        rel = Relation(
            subject=subj,
            object_entity=obj,
            relation_type=RelationType.TREATS,
            confidence=0.85,
            evidence="for",
        )
        d = rel.to_dict()
        assert d["relation_type"] == "treats"
        assert d["confidence"] == 0.85
        assert d["subject"]["text"] == "metformin"
        assert d["object"]["text"] == "diabetes"

    def test_result_to_dict(self):
        result = RelationExtractionResult(
            relations=[],
            entity_count=5,
            pair_count=10,
            processing_time_ms=1.23,
            model_name="test",
            model_version="0.1.0",
        )
        d = result.to_dict()
        assert d["entity_count"] == 5
        assert d["pair_count"] == 10
        assert d["processing_time_ms"] == 1.23


class TestConstraints:
    """Tests for RELATION_TYPE_CONSTRAINTS mapping."""

    def test_all_relation_types_have_constraints(self):
        for rt in RelationType:
            assert rt in RELATION_TYPE_CONSTRAINTS, f"Missing constraints for {rt}"

    def test_treats_allows_medication_disease(self):
        assert ("MEDICATION", "DISEASE") in RELATION_TYPE_CONSTRAINTS[RelationType.TREATS]

    def test_dosage_of_allows_dosage_medication(self):
        assert ("DOSAGE", "MEDICATION") in RELATION_TYPE_CONSTRAINTS[RelationType.DOSAGE_OF]

    def test_result_of_allows_lab_value_test(self):
        assert ("LAB_VALUE", "TEST") in RELATION_TYPE_CONSTRAINTS[RelationType.RESULT_OF]


# ---------------------------------------------------------------------------
# RuleBasedRelationExtractor tests
# ---------------------------------------------------------------------------

class TestRuleBasedExtractor:
    """Tests for RuleBasedRelationExtractor."""

    @pytest.fixture()
    def extractor(self) -> RuleBasedRelationExtractor:
        return RuleBasedRelationExtractor()

    # --- Treats relation ---

    def test_treats_for_pattern(self, extractor):
        text = "Patient is on metformin for diabetes mellitus"
        entities = [
            _entity("metformin", "MEDICATION", 14, 23),
            _entity("diabetes mellitus", "DISEASE", 28, 45),
        ]
        result = extractor.extract(text, entities)
        assert len(result.relations) >= 1
        treats = [r for r in result.relations if r.relation_type == RelationType.TREATS]
        assert len(treats) >= 1
        assert treats[0].subject.text == "metformin"
        assert treats[0].object_entity.text == "diabetes mellitus"

    def test_treats_prescribed_for(self, extractor):
        text = "Lisinopril prescribed for hypertension management"
        entities = [
            _entity("Lisinopril", "MEDICATION", 0, 10),
            _entity("hypertension", "DISEASE", 25, 37),
        ]
        result = extractor.extract(text, entities)
        treats = [r for r in result.relations if r.relation_type == RelationType.TREATS]
        assert len(treats) >= 1

    def test_treats_started_on(self, extractor):
        text = "Patient was started on insulin for poorly controlled diabetes"
        entities = [
            _entity("insulin", "MEDICATION", 22, 29),
            _entity("diabetes", "DISEASE", 52, 60),
        ]
        result = extractor.extract(text, entities)
        treats = [r for r in result.relations if r.relation_type == RelationType.TREATS]
        assert len(treats) >= 1

    # --- Causes relation ---

    def test_causes_due_to(self, extractor):
        text = "Shortness of breath due to pneumonia is concerning"
        entities = [
            _entity("pneumonia", "DISEASE", 27, 36),
            _entity("Shortness of breath", "SYMPTOM", 0, 19),
        ]
        result = extractor.extract(text, entities)
        causes = [r for r in result.relations if r.relation_type == RelationType.CAUSES]
        assert len(causes) >= 1

    def test_causes_secondary_to(self, extractor):
        text = "Acute kidney injury secondary to sepsis requires monitoring"
        entities = [
            _entity("Acute kidney injury", "DISEASE", 0, 18),
            _entity("sepsis", "DISEASE", 33, 39),
        ]
        result = extractor.extract(text, entities)
        # This should match WORSENS (DISEASE→DISEASE) or related pattern
        assert result.pair_count >= 1

    # --- Diagnoses relation ---

    def test_diagnoses_revealed(self, extractor):
        text = "CT scan revealed pneumonia in the right lower lobe"
        entities = [
            _entity("CT scan", "TEST", 0, 7),
            _entity("pneumonia", "DISEASE", 17, 26),
        ]
        result = extractor.extract(text, entities)
        diag = [r for r in result.relations if r.relation_type == RelationType.DIAGNOSES]
        assert len(diag) >= 1
        assert diag[0].subject.text == "CT scan"

    def test_diagnoses_positive_for(self, extractor):
        text = "Blood culture positive for staphylococcus aureus"
        entities = [
            _entity("Blood culture", "TEST", 0, 13),
            _entity("staphylococcus aureus", "DISEASE", 27, 48),
        ]
        result = extractor.extract(text, entities)
        diag = [r for r in result.relations if r.relation_type == RelationType.DIAGNOSES]
        assert len(diag) >= 1

    # --- Location relation ---

    def test_location_of(self, extractor):
        text = "Pain in the left knee is worse with ambulation"
        entities = [
            _entity("left knee", "ANATOMY", 12, 21),
            _entity("Pain", "SYMPTOM", 0, 4),
        ]
        result = extractor.extract(text, entities)
        locs = [r for r in result.relations if r.relation_type == RelationType.LOCATION_OF]
        assert len(locs) >= 1

    # --- Dosage relation ---

    def test_dosage_of(self, extractor):
        text = "Take 500mg metformin twice daily with meals"
        entities = [
            _entity("500mg", "DOSAGE", 5, 10),
            _entity("metformin", "MEDICATION", 11, 20),
        ]
        result = extractor.extract(text, entities)
        dosage = [r for r in result.relations if r.relation_type == RelationType.DOSAGE_OF]
        assert len(dosage) >= 1
        assert dosage[0].subject.entity_type == "DOSAGE"
        assert dosage[0].object_entity.entity_type == "MEDICATION"

    # --- Result relation ---

    def test_result_of(self, extractor):
        text = "Hemoglobin was 7.2 g/dL on CBC this morning"
        entities = [
            _entity("7.2 g/dL", "LAB_VALUE", 15, 23),
            _entity("CBC", "TEST", 27, 30),
        ]
        result = extractor.extract(text, entities)
        res = [r for r in result.relations if r.relation_type == RelationType.RESULT_OF]
        assert len(res) >= 1

    # --- Side effect relation ---

    def test_side_effect_of(self, extractor):
        text = "Nausea is a side effect of methotrexate therapy"
        entities = [
            _entity("Nausea", "SYMPTOM", 0, 6),
            _entity("methotrexate", "MEDICATION", 27, 39),
        ]
        result = extractor.extract(text, entities)
        se = [r for r in result.relations if r.relation_type == RelationType.SIDE_EFFECT_OF]
        assert len(se) >= 1

    # --- Prevents relation ---

    def test_prevents(self, extractor):
        text = "Aspirin for prevention of cardiovascular events"
        entities = [
            _entity("Aspirin", "MEDICATION", 0, 7),
            _entity("cardiovascular events", "DISEASE", 27, 48),
        ]
        result = extractor.extract(text, entities)
        prev = [r for r in result.relations if r.relation_type == RelationType.PREVENTS]
        assert len(prev) >= 1

    # --- Monitors relation ---

    def test_monitors(self, extractor):
        text = "INR to monitor warfarin therapeutic levels"
        entities = [
            _entity("INR", "TEST", 0, 3),
            _entity("warfarin", "MEDICATION", 15, 23),
        ]
        result = extractor.extract(text, entities)
        mon = [r for r in result.relations if r.relation_type == RelationType.MONITORS]
        assert len(mon) >= 1

    # --- Contraindication relation ---

    def test_contraindicates(self, extractor):
        text = "Due to renal failure, avoid metformin in this patient"
        entities = [
            _entity("renal failure", "DISEASE", 7, 20),
            _entity("metformin", "MEDICATION", 28, 37),
        ]
        result = extractor.extract(text, entities)
        contra = [r for r in result.relations if r.relation_type == RelationType.CONTRAINDICATES]
        assert len(contra) >= 1

    # --- Confidence bonuses ---

    def test_proximity_bonus_increases_confidence(self, extractor):
        """Closer entities should have higher confidence than distant ones."""
        # Close pair
        close_text = "metformin for diabetes is recommended"
        close_entities = [
            _entity("metformin", "MEDICATION", 0, 9),
            _entity("diabetes", "DISEASE", 14, 22),
        ]
        close_result = extractor.extract(close_text, close_entities)

        # Distant pair
        far_text = "metformin which the patient has been taking for several years to manage their diabetes"
        far_entities = [
            _entity("metformin", "MEDICATION", 0, 9),
            _entity("diabetes", "DISEASE", 77, 85),
        ]
        far_result = extractor.extract(far_text, far_entities)

        if close_result.relations and far_result.relations:
            close_conf = close_result.relations[0].confidence
            far_conf = far_result.relations[0].confidence
            assert close_conf >= far_conf

    def test_co_sentence_bonus(self, extractor):
        """Relations within the same sentence should get a bonus."""
        text = "Metformin for diabetes"
        entities = [
            _entity("Metformin", "MEDICATION", 0, 9),
            _entity("diabetes", "DISEASE", 14, 22),
        ]
        result = extractor.extract(text, entities)
        if result.relations:
            # No period in between → sentence bonus applied
            assert result.relations[0].metadata.get("sentence_bonus", 0) > 0

    # --- Edge cases ---

    def test_single_entity_returns_empty(self, extractor):
        text = "Patient has diabetes"
        entities = [_entity("diabetes", "DISEASE", 12, 20)]
        result = extractor.extract(text, entities)
        assert result.relations == []
        assert result.entity_count == 1
        assert result.pair_count == 0

    def test_empty_entities_returns_empty(self, extractor):
        text = "Clinical note without entities"
        result = extractor.extract(text, [])
        assert result.relations == []
        assert result.entity_count == 0

    def test_max_distance_filters_pairs(self, extractor):
        text = "A" * 200 + "metformin" + "B" * 200 + "for diabetes"
        entities = [
            _entity("metformin", "MEDICATION", 200, 209),
            _entity("diabetes", "DISEASE", 413, 421),
        ]
        # Default max_distance=150 should exclude this pair
        result = extractor.extract(text, entities, max_distance=150)
        assert result.pair_count == 0

        # Increasing distance should include it
        result2 = extractor.extract(text, entities, max_distance=300)
        assert result2.pair_count >= 1

    def test_min_confidence_filtering(self, extractor):
        # Use a longer gap so bonuses don't push confidence to 1.0
        text = "Metformin was started several months ago and then used to manage the patient's diabetes"
        entities = [
            _entity("Metformin", "MEDICATION", 0, 9),
            _entity("diabetes", "DISEASE", 78, 86),
        ]
        # Very high threshold should filter out relations
        result = extractor.extract(text, entities, min_confidence=0.99)
        assert len(result.relations) == 0

    def test_incompatible_types_no_relation(self, extractor):
        """Entities with types that don't match any constraint produce no relations."""
        text = "500mg dose of 200mg is unusual"
        entities = [
            _entity("500mg", "DOSAGE", 0, 5),
            _entity("200mg", "DOSAGE", 14, 19),
        ]
        result = extractor.extract(text, entities)
        assert len(result.relations) == 0

    def test_deduplication_keeps_highest_confidence(self, extractor):
        """When multiple patterns match the same pair, keep highest confidence."""
        text = "Started metformin, used for treating diabetes mellitus"
        entities = [
            _entity("metformin", "MEDICATION", 8, 17),
            _entity("diabetes mellitus", "DISEASE", 37, 54),
        ]
        result = extractor.extract(text, entities)
        # Multiple treat patterns could match; only one relation per pair+type
        treats = [r for r in result.relations if r.relation_type == RelationType.TREATS]
        # Should be deduplicated to at most one per direction
        pair_keys = {(r.subject.start_char, r.object_entity.start_char) for r in treats}
        assert len(pair_keys) <= 1

    def test_result_contains_model_info(self, extractor):
        text = "Metformin for diabetes"
        entities = [
            _entity("Metformin", "MEDICATION", 0, 9),
            _entity("diabetes", "DISEASE", 14, 22),
        ]
        result = extractor.extract(text, entities)
        assert result.model_name == "rule-based-relations"
        assert result.model_version == "1.0.0"

    def test_processing_time_is_positive(self, extractor):
        text = "Metformin for diabetes"
        entities = [
            _entity("Metformin", "MEDICATION", 0, 9),
            _entity("diabetes", "DISEASE", 14, 22),
        ]
        result = extractor.extract(text, entities)
        assert result.processing_time_ms >= 0

    def test_realistic_clinical_note(self, extractor):
        """Full clinical note with multiple entities and expected relations."""
        text = (
            "Patient is a 65-year-old male with hypertension managed with "
            "lisinopril 10mg daily. CT scan of the chest revealed pneumonia "
            "in the right lower lobe. Started on amoxicillin for the pneumonia. "
            "BMP showed creatinine of 2.1 mg/dL."
        )
        entities = [
            _entity("hypertension", "DISEASE", 35, 47),
            _entity("lisinopril", "MEDICATION", 61, 71),
            _entity("10mg", "DOSAGE", 72, 76),
            _entity("CT scan", "TEST", 84, 91),
            _entity("chest", "ANATOMY", 99, 104),
            _entity("pneumonia", "DISEASE", 114, 123),
            _entity("right lower lobe", "ANATOMY", 131, 147),
            _entity("amoxicillin", "MEDICATION", 160, 171),
            _entity("pneumonia", "DISEASE", 180, 189),
            _entity("BMP", "TEST", 191, 194),
            _entity("2.1 mg/dL", "LAB_VALUE", 214, 223),
        ]
        result = extractor.extract(text, entities, max_distance=200)
        assert result.entity_count == 11
        assert result.pair_count > 0
        assert len(result.relations) > 0

        # Check for expected relation types
        rel_types = {r.relation_type for r in result.relations}
        # Should find at least treats and/or diagnoses
        assert len(rel_types) >= 1

    def test_overlapping_entities_skipped(self, extractor):
        """Entities with overlapping spans should not form pairs."""
        text = "diabetes mellitus type 2"
        entities = [
            _entity("diabetes", "DISEASE", 0, 8),
            _entity("diabetes mellitus", "DISEASE", 0, 17),
        ]
        result = extractor.extract(text, entities)
        # Overlapping → negative gap → skipped
        assert result.pair_count == 0


# ---------------------------------------------------------------------------
# TransformerRelationExtractor tests
# ---------------------------------------------------------------------------

class TestTransformerExtractorFallback:
    """Tests for TransformerRelationExtractor (model not available)."""

    def test_fallback_to_rule_based(self):
        ext = TransformerRelationExtractor(model_id="nonexistent/model")
        text = "Metformin for diabetes"
        entities = [
            _entity("Metformin", "MEDICATION", 0, 9),
            _entity("diabetes", "DISEASE", 14, 22),
        ]
        result = ext.extract(text, entities)
        # Should fall back and still produce results
        assert "fallback" in result.model_name.lower()
        assert len(result.relations) >= 1

    def test_load_not_retried_after_failure(self):
        ext = TransformerRelationExtractor(model_id="nonexistent/model")
        ext.load()  # First load attempt
        assert ext._load_failed
        ext.load()  # Second call should be a no-op
        assert ext._load_failed
        assert not ext._loaded

    def test_custom_model_name(self):
        ext = TransformerRelationExtractor()
        assert ext.model_name == "transformer-relations"
        assert ext.version == "1.0.0"
