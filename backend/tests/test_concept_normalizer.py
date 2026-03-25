"""Tests for the clinical concept normalization engine.

Covers:
- Enum completeness and data structure validation
- Dictionary integrity (no duplicate CUIs in same group, all aliases unique-ish)
- Exact match on preferred terms
- Alias match (abbreviations, synonyms, brand names)
- Fuzzy matching with similarity threshold
- Type-aware filtering
- Batch normalization
- CUI reverse lookup
- Statistics tracking
- Edge cases (empty input, whitespace, case insensitivity)
- NormalizationResult serialization
- Dictionary stats helper
"""

from __future__ import annotations

import pytest

from app.ml.normalization.normalizer import (
    BatchNormalizationResult,
    ClinicalConceptNormalizer,
    ConceptEntry,
    EntityTypeGroup,
    NormalizationResult,
    NormalizerConfig,
    OntologySource,
    _ALIAS_INDEX,
    _CONCEPT_DATA,
    _EXACT_INDEX,
    _GROUP_INDEX,
    get_dictionary_stats,
)


# ---------------------------------------------------------------------------
# Enum & data structure tests
# ---------------------------------------------------------------------------


class TestEnumsAndDataclasses:
    """Validate enum completeness and dataclass behaviour."""

    def test_ontology_source_values(self) -> None:
        assert set(OntologySource) == {
            OntologySource.UMLS,
            OntologySource.SNOMED_CT,
            OntologySource.RXNORM,
            OntologySource.ICD10CM,
            OntologySource.LOINC,
        }

    def test_entity_type_group_values(self) -> None:
        assert set(EntityTypeGroup) == {
            EntityTypeGroup.CONDITION,
            EntityTypeGroup.MEDICATION,
            EntityTypeGroup.PROCEDURE,
            EntityTypeGroup.ANATOMY,
            EntityTypeGroup.LAB,
        }

    def test_concept_entry_frozen(self) -> None:
        entry = ConceptEntry(
            cui="C0000001",
            preferred_term="Test Concept",
            aliases=("test",),
        )
        with pytest.raises(AttributeError):
            entry.cui = "C9999999"  # type: ignore[misc]

    def test_normalization_result_to_dict(self) -> None:
        result = NormalizationResult(
            input_text="HTN",
            matched=True,
            cui="C0020538",
            preferred_term="Hypertension",
            confidence=1.0,
            match_type="exact",
            snomed_code="38341003",
            icd10_code="I10",
        )
        d = result.to_dict()
        assert d["input_text"] == "HTN"
        assert d["matched"] is True
        assert d["codes"]["umls_cui"] == "C0020538"
        assert d["codes"]["snomed_ct"] == "38341003"
        assert d["codes"]["icd10_cm"] == "I10"
        assert d["confidence"] == 1.0
        assert d["match_type"] == "exact"

    def test_normalization_result_default(self) -> None:
        result = NormalizationResult(input_text="unknown entity")
        assert result.matched is False
        assert result.cui is None
        assert result.confidence == 0.0
        assert result.match_type == "none"

    def test_batch_result_to_dict(self) -> None:
        r1 = NormalizationResult(input_text="HTN", matched=True, confidence=1.0)
        r2 = NormalizationResult(input_text="xyz", matched=False, confidence=0.0)
        batch = BatchNormalizationResult(
            results=[r1, r2],
            total=2,
            matched_count=1,
            match_rate=0.5,
            processing_time_ms=1.23,
        )
        d = batch.to_dict()
        assert len(d["results"]) == 2
        assert d["summary"]["total"] == 2
        assert d["summary"]["matched"] == 1
        assert d["summary"]["unmatched"] == 1
        assert d["summary"]["match_rate"] == 0.5


# ---------------------------------------------------------------------------
# Dictionary integrity tests
# ---------------------------------------------------------------------------


class TestDictionaryIntegrity:
    """Validate the concept dictionary is well-formed."""

    def test_all_concepts_have_cui(self) -> None:
        for concept in _CONCEPT_DATA:
            assert concept.cui, f"Concept '{concept.preferred_term}' missing CUI"
            assert concept.cui.startswith("C"), f"CUI '{concept.cui}' doesn't start with C"

    def test_all_concepts_have_preferred_term(self) -> None:
        for concept in _CONCEPT_DATA:
            assert concept.preferred_term.strip(), f"CUI {concept.cui} has empty preferred_term"

    def test_preferred_terms_indexed(self) -> None:
        for concept in _CONCEPT_DATA:
            key = concept.preferred_term.lower().strip()
            assert key in _EXACT_INDEX, f"'{concept.preferred_term}' not in exact index"

    def test_aliases_indexed(self) -> None:
        for concept in _CONCEPT_DATA:
            for alias in concept.aliases:
                key = alias.lower().strip()
                assert key in _ALIAS_INDEX, f"Alias '{alias}' not in alias index"

    def test_group_index_populated(self) -> None:
        for group in EntityTypeGroup:
            assert len(_GROUP_INDEX[group]) > 0, f"No concepts in group {group.value}"

    def test_dictionary_has_conditions(self) -> None:
        conditions = [c for c in _CONCEPT_DATA if c.type_group == EntityTypeGroup.CONDITION]
        assert len(conditions) >= 40

    def test_dictionary_has_medications(self) -> None:
        meds = [c for c in _CONCEPT_DATA if c.type_group == EntityTypeGroup.MEDICATION]
        assert len(meds) >= 30

    def test_dictionary_has_procedures(self) -> None:
        procs = [c for c in _CONCEPT_DATA if c.type_group == EntityTypeGroup.PROCEDURE]
        assert len(procs) >= 10

    def test_dictionary_has_labs(self) -> None:
        labs = [c for c in _CONCEPT_DATA if c.type_group == EntityTypeGroup.LAB]
        assert len(labs) >= 15

    def test_dictionary_has_anatomy(self) -> None:
        anatomy = [c for c in _CONCEPT_DATA if c.type_group == EntityTypeGroup.ANATOMY]
        assert len(anatomy) >= 5


# ---------------------------------------------------------------------------
# Exact match tests
# ---------------------------------------------------------------------------


class TestExactMatch:
    """Test exact match on preferred terms."""

    def setup_method(self) -> None:
        self.normalizer = ClinicalConceptNormalizer()

    def test_exact_match_hypertension(self) -> None:
        result = self.normalizer.normalize("Hypertension")
        assert result.matched is True
        assert result.cui == "C0020538"
        assert result.confidence == 1.0
        assert result.match_type == "exact"
        assert result.snomed_code == "38341003"
        assert result.icd10_code == "I10"

    def test_exact_match_case_insensitive(self) -> None:
        result = self.normalizer.normalize("HYPERTENSION")
        assert result.matched is True
        assert result.cui == "C0020538"

    def test_exact_match_with_whitespace(self) -> None:
        result = self.normalizer.normalize("  hypertension  ")
        assert result.matched is True
        assert result.cui == "C0020538"

    def test_exact_match_metformin(self) -> None:
        result = self.normalizer.normalize("Metformin")
        assert result.matched is True
        assert result.cui == "C0025598"
        assert result.rxnorm_code == "6809"

    def test_exact_match_complete_blood_count(self) -> None:
        result = self.normalizer.normalize("Complete Blood Count")
        assert result.matched is True
        assert result.loinc_code == "26604-6"

    def test_exact_match_dental_caries(self) -> None:
        result = self.normalizer.normalize("Dental Caries")
        assert result.matched is True
        assert result.snomed_code == "80967001"
        assert result.icd10_code == "K02.9"

    def test_exact_match_electrocardiogram(self) -> None:
        result = self.normalizer.normalize("Electrocardiogram")
        assert result.matched is True
        assert result.snomed_code == "29303009"

    def test_exact_match_heart(self) -> None:
        result = self.normalizer.normalize("Heart")
        assert result.matched is True
        assert result.semantic_type == "Body Part, Organ, or Organ Component"

    def test_exact_match_returns_semantic_type(self) -> None:
        result = self.normalizer.normalize("Asthma")
        assert result.semantic_type == "Disease or Syndrome"

    def test_exact_match_depression(self) -> None:
        result = self.normalizer.normalize("Major Depressive Disorder")
        assert result.matched is True
        assert result.icd10_code == "F33.0"


# ---------------------------------------------------------------------------
# Alias match tests
# ---------------------------------------------------------------------------


class TestAliasMatch:
    """Test alias / abbreviation matching."""

    def setup_method(self) -> None:
        self.normalizer = ClinicalConceptNormalizer()

    def test_alias_htn(self) -> None:
        result = self.normalizer.normalize("HTN")
        assert result.matched is True
        assert result.cui == "C0020538"
        assert result.preferred_term == "Hypertension"
        assert result.match_type == "alias"
        assert result.confidence == 0.95

    def test_alias_mi(self) -> None:
        result = self.normalizer.normalize("MI")
        assert result.matched is True
        assert result.preferred_term == "Myocardial Infarction"

    def test_alias_heart_attack(self) -> None:
        result = self.normalizer.normalize("heart attack")
        assert result.matched is True
        assert result.cui == "C0027051"

    def test_alias_brand_name_lipitor(self) -> None:
        result = self.normalizer.normalize("Lipitor")
        assert result.matched is True
        assert result.preferred_term == "Atorvastatin"

    def test_alias_brand_name_zoloft(self) -> None:
        result = self.normalizer.normalize("zoloft")
        assert result.matched is True
        assert result.preferred_term == "Sertraline"

    def test_alias_sobr(self) -> None:
        result = self.normalizer.normalize("SOB")
        assert result.matched is True
        assert result.preferred_term == "Dyspnea"

    def test_alias_cbc(self) -> None:
        result = self.normalizer.normalize("CBC")
        assert result.matched is True
        assert result.preferred_term == "Complete Blood Count"

    def test_alias_copd(self) -> None:
        result = self.normalizer.normalize("COPD")
        assert result.matched is True
        assert result.preferred_term == "Chronic Obstructive Pulmonary Disease"

    def test_alias_cabg(self) -> None:
        result = self.normalizer.normalize("CABG")
        assert result.matched is True
        assert result.preferred_term == "Coronary Artery Bypass Graft"

    def test_alias_covid(self) -> None:
        result = self.normalizer.normalize("covid")
        assert result.matched is True
        assert result.preferred_term == "COVID-19"

    def test_alias_ozempic(self) -> None:
        result = self.normalizer.normalize("Ozempic")
        assert result.matched is True
        assert result.preferred_term == "Semaglutide"

    def test_alias_z_pack(self) -> None:
        result = self.normalizer.normalize("z-pack")
        assert result.matched is True
        assert result.preferred_term == "Azithromycin"

    def test_alias_tmj(self) -> None:
        result = self.normalizer.normalize("TMJ")
        assert result.matched is True
        assert result.preferred_term == "Temporomandibular Joint Disorder"

    def test_alias_srp(self) -> None:
        result = self.normalizer.normalize("SRP")
        assert result.matched is True
        assert result.preferred_term == "Scaling and Root Planing"

    def test_alias_a1c(self) -> None:
        result = self.normalizer.normalize("A1C")
        assert result.matched is True
        assert result.preferred_term == "Hemoglobin A1c"

    def test_alias_with_trailing_period(self) -> None:
        result = self.normalizer.normalize("HTN.")
        assert result.matched is True
        assert result.preferred_term == "Hypertension"


# ---------------------------------------------------------------------------
# Fuzzy match tests
# ---------------------------------------------------------------------------


class TestFuzzyMatch:
    """Test fuzzy matching with similarity threshold."""

    def setup_method(self) -> None:
        self.normalizer = ClinicalConceptNormalizer()

    def test_fuzzy_match_typo(self) -> None:
        result = self.normalizer.normalize("hypertensoin")
        assert result.matched is True
        assert result.preferred_term == "Hypertension"
        assert result.match_type == "fuzzy"
        assert result.confidence >= 0.80

    def test_fuzzy_match_partial_name(self) -> None:
        result = self.normalizer.normalize("myocardial infarct")
        assert result.matched is True
        assert result.preferred_term == "Myocardial Infarction"
        assert result.match_type == "fuzzy"

    def test_fuzzy_match_below_threshold_misses(self) -> None:
        config = NormalizerConfig(min_similarity=0.99)
        normalizer = ClinicalConceptNormalizer(config)
        result = normalizer.normalize("hypertensoin")
        assert result.matched is False

    def test_fuzzy_match_disabled(self) -> None:
        config = NormalizerConfig(enable_fuzzy=False)
        normalizer = ClinicalConceptNormalizer(config)
        result = normalizer.normalize("hypertensoin")
        assert result.matched is False
        assert result.match_type == "none"

    def test_fuzzy_match_alternatives(self) -> None:
        result = self.normalizer.normalize("osteoarthrtis")
        assert result.matched is True
        # Alternatives may or may not be present depending on threshold
        assert isinstance(result.alternatives, list)

    def test_fuzzy_match_low_threshold_more_hits(self) -> None:
        config = NormalizerConfig(min_similarity=0.60)
        normalizer = ClinicalConceptNormalizer(config)
        result = normalizer.normalize("asthm")
        assert result.matched is True


# ---------------------------------------------------------------------------
# Type-aware filtering tests
# ---------------------------------------------------------------------------


class TestTypeAwareFiltering:
    """Test entity-type-aware concept filtering."""

    def setup_method(self) -> None:
        self.normalizer = ClinicalConceptNormalizer()

    def test_disease_type_matches_condition(self) -> None:
        result = self.normalizer.normalize("Hypertension", entity_type="DISEASE")
        assert result.matched is True
        assert result.cui == "C0020538"

    def test_medication_type_matches_drug(self) -> None:
        result = self.normalizer.normalize("Metformin", entity_type="MEDICATION")
        assert result.matched is True
        assert result.cui == "C0025598"

    def test_type_mismatch_blocks_exact(self) -> None:
        # Hypertension is a CONDITION, not a MEDICATION
        result = self.normalizer.normalize("Hypertension", entity_type="MEDICATION")
        # Should not match as exact because type doesn't match
        assert result.match_type != "exact" or not result.matched

    def test_procedure_type_matches(self) -> None:
        result = self.normalizer.normalize("MRI", entity_type="PROCEDURE")
        assert result.matched is True
        assert result.preferred_term == "Magnetic Resonance Imaging"

    def test_lab_type_matches(self) -> None:
        result = self.normalizer.normalize("CBC", entity_type="LAB_VALUE")
        assert result.matched is True
        assert result.preferred_term == "Complete Blood Count"

    def test_no_type_constraint_matches_anything(self) -> None:
        result = self.normalizer.normalize("Hypertension")
        assert result.matched is True

    def test_unknown_entity_type_no_constraint(self) -> None:
        result = self.normalizer.normalize("Hypertension", entity_type="UNKNOWN_TYPE")
        # Unknown type → no type constraint (no mapping in _ENTITY_TYPE_TO_GROUP)
        assert result.matched is True

    def test_symptom_type_matches_condition(self) -> None:
        result = self.normalizer.normalize("Dyspnea", entity_type="SYMPTOM")
        assert result.matched is True

    def test_body_part_type_matches_anatomy(self) -> None:
        result = self.normalizer.normalize("Heart", entity_type="BODY_PART")
        assert result.matched is True

    def test_test_type_matches_lab(self) -> None:
        result = self.normalizer.normalize("Hemoglobin A1c", entity_type="TEST")
        assert result.matched is True


# ---------------------------------------------------------------------------
# Batch normalization tests
# ---------------------------------------------------------------------------


class TestBatchNormalization:
    """Test batch normalization."""

    def setup_method(self) -> None:
        self.normalizer = ClinicalConceptNormalizer()

    def test_batch_multiple_entities(self) -> None:
        entities = [
            {"text": "HTN", "entity_type": "DISEASE"},
            {"text": "metformin", "entity_type": "MEDICATION"},
            {"text": "CBC", "entity_type": "LAB_VALUE"},
            {"text": "nonexistent_entity_xyz"},
        ]
        result = self.normalizer.normalize_batch(entities)
        assert result.total == 4
        assert result.matched_count == 3
        assert result.match_rate == 0.75
        assert result.processing_time_ms >= 0
        assert len(result.results) == 4
        assert result.results[0].matched is True
        assert result.results[3].matched is False

    def test_batch_preserves_order(self) -> None:
        entities = [
            {"text": "Asthma"},
            {"text": "Metformin"},
            {"text": "ECG"},
        ]
        result = self.normalizer.normalize_batch(entities)
        assert result.results[0].preferred_term == "Asthma"
        assert result.results[1].preferred_term == "Metformin"
        assert result.results[2].preferred_term == "Electrocardiogram"

    def test_batch_empty_list(self) -> None:
        result = self.normalizer.normalize_batch([])
        assert result.total == 0
        assert result.matched_count == 0
        assert result.match_rate == 0.0

    def test_batch_all_matched(self) -> None:
        entities = [
            {"text": "HTN"},
            {"text": "DM"},
            {"text": "COPD"},
        ]
        result = self.normalizer.normalize_batch(entities)
        assert result.match_rate == 1.0

    def test_batch_result_to_dict(self) -> None:
        entities = [{"text": "HTN"}, {"text": "xyz_unknown"}]
        result = self.normalizer.normalize_batch(entities)
        d = result.to_dict()
        assert "results" in d
        assert "summary" in d
        assert d["summary"]["total"] == 2
        assert d["summary"]["matched"] == 1


# ---------------------------------------------------------------------------
# CUI lookup tests
# ---------------------------------------------------------------------------


class TestCUILookup:
    """Test reverse CUI lookup."""

    def setup_method(self) -> None:
        self.normalizer = ClinicalConceptNormalizer()

    def test_lookup_existing_cui(self) -> None:
        concept = self.normalizer.lookup_cui("C0020538")
        assert concept is not None
        assert concept.preferred_term == "Hypertension"
        assert "htn" in concept.aliases

    def test_lookup_case_insensitive(self) -> None:
        concept = self.normalizer.lookup_cui("c0020538")
        assert concept is not None
        assert concept.preferred_term == "Hypertension"

    def test_lookup_nonexistent_cui(self) -> None:
        concept = self.normalizer.lookup_cui("C9999999")
        assert concept is None

    def test_lookup_with_whitespace(self) -> None:
        concept = self.normalizer.lookup_cui("  C0020538  ")
        assert concept is not None

    def test_lookup_medication_cui(self) -> None:
        concept = self.normalizer.lookup_cui("C0025598")
        assert concept is not None
        assert concept.preferred_term == "Metformin"
        assert concept.rxnorm_code == "6809"


# ---------------------------------------------------------------------------
# Statistics tests
# ---------------------------------------------------------------------------


class TestStatistics:
    """Test normalization statistics tracking."""

    def test_stats_tracking(self) -> None:
        normalizer = ClinicalConceptNormalizer()
        normalizer.reset_stats()

        normalizer.normalize("Hypertension")  # exact
        normalizer.normalize("HTN")           # alias
        normalizer.normalize("xyz_unknown")   # miss

        stats = normalizer.get_stats()
        assert stats["total"] == 3
        assert stats["exact_hits"] == 1
        assert stats["alias_hits"] == 1
        assert stats["misses"] == 1
        assert stats["match_rate"] == pytest.approx(2 / 3, abs=0.01)

    def test_stats_reset(self) -> None:
        normalizer = ClinicalConceptNormalizer()
        normalizer.normalize("Hypertension")
        normalizer.reset_stats()
        stats = normalizer.get_stats()
        assert stats["total"] == 0
        assert stats["exact_hits"] == 0

    def test_stats_include_dictionary_size(self) -> None:
        normalizer = ClinicalConceptNormalizer()
        stats = normalizer.get_stats()
        assert stats["dictionary_size"] == len(_CONCEPT_DATA)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def setup_method(self) -> None:
        self.normalizer = ClinicalConceptNormalizer()

    def test_empty_string(self) -> None:
        result = self.normalizer.normalize("")
        assert result.matched is False
        assert result.match_type == "none"

    def test_whitespace_only(self) -> None:
        result = self.normalizer.normalize("   ")
        assert result.matched is False

    def test_single_character(self) -> None:
        result = self.normalizer.normalize("x")
        # Might match something via fuzzy, or not
        assert isinstance(result.matched, bool)

    def test_very_long_text(self) -> None:
        result = self.normalizer.normalize("a" * 10000)
        assert result.matched is False

    def test_special_characters(self) -> None:
        result = self.normalizer.normalize("@#$%^&*()")
        assert result.matched is False

    def test_numeric_input(self) -> None:
        result = self.normalizer.normalize("12345")
        assert isinstance(result, NormalizationResult)

    def test_mixed_case_alias(self) -> None:
        result = self.normalizer.normalize("hTn")
        assert result.matched is True
        assert result.preferred_term == "Hypertension"

    def test_abbreviation_with_periods(self) -> None:
        # "A.F." with trailing period stripped
        result = self.normalizer.normalize("afib.")
        assert result.matched is True
        assert result.preferred_term == "Atrial Fibrillation"

    def test_multi_word_collapse_whitespace(self) -> None:
        result = self.normalizer.normalize("heart   attack")
        assert result.matched is True
        assert result.preferred_term == "Myocardial Infarction"


# ---------------------------------------------------------------------------
# Dictionary stats helper tests
# ---------------------------------------------------------------------------


class TestDictionaryStats:
    """Test the get_dictionary_stats() helper."""

    def test_returns_total_concepts(self) -> None:
        stats = get_dictionary_stats()
        assert stats["total_concepts"] == len(_CONCEPT_DATA)
        assert stats["total_concepts"] > 100

    def test_returns_total_aliases(self) -> None:
        stats = get_dictionary_stats()
        assert stats["total_aliases"] > 100

    def test_returns_type_group_counts(self) -> None:
        stats = get_dictionary_stats()
        groups = stats["by_type_group"]
        assert "CONDITION" in groups
        assert "MEDICATION" in groups
        assert "PROCEDURE" in groups
        assert "LAB" in groups
        assert "ANATOMY" in groups

    def test_returns_ontology_coverage(self) -> None:
        stats = get_dictionary_stats()
        coverage = stats["ontology_coverage"]
        assert coverage["snomed_ct"] > 0
        assert coverage["rxnorm"] > 0
        assert coverage["icd10_cm"] > 0
        assert coverage["loinc"] > 0


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestNormalizerConfig:
    """Test normalizer configuration options."""

    def test_default_config(self) -> None:
        config = NormalizerConfig()
        assert config.min_similarity == 0.80
        assert config.max_alternatives == 3
        assert config.enable_fuzzy is True
        assert config.type_aware is True

    def test_custom_config(self) -> None:
        config = NormalizerConfig(
            min_similarity=0.90,
            max_alternatives=5,
            enable_fuzzy=False,
            type_aware=False,
        )
        normalizer = ClinicalConceptNormalizer(config)
        assert normalizer.config.min_similarity == 0.90
        assert normalizer.config.enable_fuzzy is False

    def test_type_aware_disabled(self) -> None:
        config = NormalizerConfig(type_aware=False)
        normalizer = ClinicalConceptNormalizer(config)
        # Should match even with mismatched type
        result = normalizer.normalize("Hypertension", entity_type="MEDICATION")
        assert result.matched is True  # type filtering disabled


# ---------------------------------------------------------------------------
# Realistic clinical scenario tests
# ---------------------------------------------------------------------------


class TestRealisticScenarios:
    """Test with realistic clinical entity mentions."""

    def setup_method(self) -> None:
        self.normalizer = ClinicalConceptNormalizer()

    def test_discharge_summary_entities(self) -> None:
        """Normalize entities from a typical discharge summary."""
        entities = [
            {"text": "CHF", "entity_type": "DISEASE"},
            {"text": "AFib", "entity_type": "DISEASE"},
            {"text": "lisinopril", "entity_type": "MEDICATION"},
            {"text": "metoprolol", "entity_type": "MEDICATION"},
            {"text": "BNP", "entity_type": "LAB_VALUE"},
            {"text": "Echo", "entity_type": "PROCEDURE"},
        ]
        result = self.normalizer.normalize_batch(entities)
        assert result.matched_count >= 5  # Most should match
        assert result.results[0].preferred_term == "Congestive Heart Failure"
        assert result.results[1].preferred_term == "Atrial Fibrillation"

    def test_dental_note_entities(self) -> None:
        """Normalize entities from a dental progress note."""
        entities = [
            {"text": "caries"},
            {"text": "periodontal disease"},
            {"text": "SRP"},
            {"text": "root canal"},
            {"text": "TMJ"},
        ]
        result = self.normalizer.normalize_batch(entities)
        assert result.matched_count >= 4

    def test_lab_report_entities(self) -> None:
        """Normalize lab test names."""
        entities = [
            {"text": "CBC", "entity_type": "TEST"},
            {"text": "BMP", "entity_type": "TEST"},
            {"text": "TSH", "entity_type": "TEST"},
            {"text": "HbA1c", "entity_type": "TEST"},
            {"text": "LFTs", "entity_type": "TEST"},
        ]
        result = self.normalizer.normalize_batch(entities)
        assert result.matched_count >= 4
