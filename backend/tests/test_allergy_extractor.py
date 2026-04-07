"""Tests for the clinical allergy extraction module.

Covers allergen detection, reaction identification, severity classification,
NKDA detection, negation handling, section awareness, and edge cases.
"""

from __future__ import annotations

import pytest

from app.ml.allergies.extractor import (
    _ALL_ALLERGENS,
    _REACTIONS,
    _SURFACE_TO_ENTRY,
    AllergyCategory,
    AllergyResult,
    AllergySeverity,
    AllergyStatus,
    ClinicalAllergyExtractor,
    DetectedAllergy,
    ExtractionResult,
)

# ---------------------------------------------------------------------------
# Enum completeness
# ---------------------------------------------------------------------------

class TestEnums:
    """Ensure enums cover expected values."""

    def test_allergy_categories(self) -> None:
        assert AllergyCategory.DRUG == "drug"
        assert AllergyCategory.FOOD == "food"
        assert AllergyCategory.ENVIRONMENTAL == "environmental"
        assert len(AllergyCategory) == 3

    def test_severity_levels(self) -> None:
        assert AllergySeverity.MILD == "mild"
        assert AllergySeverity.MODERATE == "moderate"
        assert AllergySeverity.SEVERE == "severe"
        assert AllergySeverity.LIFE_THREATENING == "life_threatening"
        assert AllergySeverity.UNKNOWN == "unknown"
        assert len(AllergySeverity) == 5

    def test_allergy_statuses(self) -> None:
        assert AllergyStatus.ACTIVE == "active"
        assert AllergyStatus.TOLERATED == "tolerated"
        assert AllergyStatus.HISTORICAL == "historical"
        assert len(AllergyStatus) == 3


# ---------------------------------------------------------------------------
# Dataclass serialisation
# ---------------------------------------------------------------------------

class TestDataclasses:
    """Test dataclass serialisation."""

    def test_allergy_result_to_dict(self) -> None:
        r = AllergyResult(reaction="anaphylaxis", severity=AllergySeverity.LIFE_THREATENING)
        d = r.to_dict()
        assert d["reaction"] == "anaphylaxis"
        assert d["severity"] == "life_threatening"

    def test_detected_allergy_to_dict(self) -> None:
        a = DetectedAllergy(
            allergen="penicillin",
            allergen_raw="PCN",
            category=AllergyCategory.DRUG,
            reactions=[AllergyResult("rash", AllergySeverity.MODERATE)],
            severity=AllergySeverity.MODERATE,
            start=10,
            end=13,
            confidence=0.85,
        )
        d = a.to_dict()
        assert d["allergen"] == "penicillin"
        assert d["category"] == "drug"
        assert len(d["reactions"]) == 1
        assert d["confidence"] == 0.85

    def test_extraction_result_to_dict(self) -> None:
        r = ExtractionResult(
            allergies=[
                DetectedAllergy(
                    allergen="aspirin", allergen_raw="aspirin",
                    category=AllergyCategory.DRUG, start=0, end=7,
                )
            ],
            no_known_allergies=False,
            text_length=50,
        )
        d = r.to_dict()
        assert d["allergy_count"] == 1
        assert "drug" in d["categories"]


# ---------------------------------------------------------------------------
# Dictionary integrity
# ---------------------------------------------------------------------------

class TestDictionary:
    """Test allergen dictionary coverage and structure."""

    def test_minimum_drug_allergens(self) -> None:
        drug_count = sum(1 for e in _ALL_ALLERGENS if e.category == AllergyCategory.DRUG)
        assert drug_count >= 75

    def test_minimum_food_allergens(self) -> None:
        food_count = sum(1 for e in _ALL_ALLERGENS if e.category == AllergyCategory.FOOD)
        assert food_count >= 12

    def test_minimum_environmental_allergens(self) -> None:
        env_count = sum(1 for e in _ALL_ALLERGENS if e.category == AllergyCategory.ENVIRONMENTAL)
        assert env_count >= 8

    def test_minimum_reactions(self) -> None:
        assert len(_REACTIONS) >= 30

    def test_surface_forms_all_lowercase(self) -> None:
        for form in _SURFACE_TO_ENTRY:
            assert form == form.lower(), f"Surface form not lowercase: {form}"

    def test_no_duplicate_canonical_names(self) -> None:
        canonicals = [e.canonical for e in _ALL_ALLERGENS]
        # Allow latex to appear in both drug and environmental
        # (it's actually only in _DRUG_ALLERGENS with ENVIRONMENTAL category)
        assert len(canonicals) == len(set(canonicals))

    def test_get_allergen_count(self) -> None:
        counts = ClinicalAllergyExtractor.get_allergen_count()
        assert "drug" in counts
        assert "food" in counts
        assert "environmental" in counts
        assert sum(counts.values()) == len(_ALL_ALLERGENS)

    def test_get_reaction_count(self) -> None:
        assert ClinicalAllergyExtractor.get_reaction_count() == len(_REACTIONS)


# ---------------------------------------------------------------------------
# Drug allergy detection
# ---------------------------------------------------------------------------

class TestDrugAllergyDetection:
    """Test detection of drug allergens."""

    @pytest.fixture()
    def extractor(self) -> ClinicalAllergyExtractor:
        return ClinicalAllergyExtractor()

    def test_penicillin_canonical(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Patient is allergic to penicillin.")
        assert len(result.allergies) >= 1
        assert any(a.allergen == "penicillin" for a in result.allergies)

    def test_pcn_abbreviation(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Allergies: PCN - anaphylaxis")
        pcn = next((a for a in result.allergies if a.allergen == "penicillin"), None)
        assert pcn is not None
        assert pcn.allergen_raw.lower() == "pcn"

    def test_sulfa_detection(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Allergic to sulfa drugs, develops rash.")
        assert any(a.allergen == "sulfonamides" for a in result.allergies)

    def test_aspirin_with_alias(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Cannot take ASA due to GI bleeding.")
        assert any(a.allergen == "aspirin" for a in result.allergies)

    def test_ibuprofen_brand_name(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Allergic to Advil, causes hives.")
        assert any(a.allergen == "ibuprofen" for a in result.allergies)

    def test_contrast_dye(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Allergic to IV contrast, had anaphylaxis.")
        assert any(a.allergen == "iodine contrast" for a in result.allergies)

    def test_codeine_opioid(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Allergy to codeine - nausea and vomiting.")
        assert any(a.allergen == "codeine" for a in result.allergies)

    def test_drug_category_assigned(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Allergic to amoxicillin.")
        for a in result.allergies:
            if a.allergen == "amoxicillin":
                assert a.category == AllergyCategory.DRUG

    def test_multiple_drug_allergies(self, extractor: ClinicalAllergyExtractor) -> None:
        text = "Allergies: PCN, sulfa, codeine, aspirin"
        result = extractor.extract(text)
        allergens = {a.allergen for a in result.allergies}
        assert "penicillin" in allergens
        assert "sulfonamides" in allergens
        assert "codeine" in allergens
        assert "aspirin" in allergens


# ---------------------------------------------------------------------------
# Food allergy detection
# ---------------------------------------------------------------------------

class TestFoodAllergyDetection:
    """Test detection of food allergens."""

    @pytest.fixture()
    def extractor(self) -> ClinicalAllergyExtractor:
        return ClinicalAllergyExtractor()

    def test_peanut_allergy(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Severe peanut allergy with anaphylaxis.")
        assert any(a.allergen == "peanuts" and a.category == AllergyCategory.FOOD
                    for a in result.allergies)

    def test_shellfish_allergy(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Allergic to shellfish, hives and swelling.")
        assert any(a.allergen == "shellfish" for a in result.allergies)

    def test_tree_nut_alias(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Allergic to walnuts and cashews.")
        allergens = {a.allergen for a in result.allergies}
        assert "tree nuts" in allergens

    def test_dairy_allergy(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Dairy allergy - GI upset.")
        assert any(a.allergen == "milk" for a in result.allergies)


# ---------------------------------------------------------------------------
# Environmental allergy detection
# ---------------------------------------------------------------------------

class TestEnvironmentalAllergyDetection:
    """Test detection of environmental allergens."""

    @pytest.fixture()
    def extractor(self) -> ClinicalAllergyExtractor:
        return ClinicalAllergyExtractor()

    def test_pollen_allergy(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Seasonal allergies to pollen.")
        assert any(a.allergen == "pollen" and a.category == AllergyCategory.ENVIRONMENTAL
                    for a in result.allergies)

    def test_dust_mite_allergy(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Allergic to dust mites.")
        assert any(a.allergen == "dust mites" for a in result.allergies)

    def test_bee_sting_allergy(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("History of anaphylaxis from bee stings.")
        assert any(a.allergen == "bee stings" for a in result.allergies)

    def test_latex_allergy(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Latex allergy - use non-latex gloves.")
        assert any(a.allergen == "latex" for a in result.allergies)


# ---------------------------------------------------------------------------
# Reaction detection
# ---------------------------------------------------------------------------

class TestReactionDetection:
    """Test detection of reactions associated with allergies."""

    @pytest.fixture()
    def extractor(self) -> ClinicalAllergyExtractor:
        return ClinicalAllergyExtractor()

    def test_anaphylaxis_detected(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("PCN - anaphylaxis")
        pcn = next(a for a in result.allergies if a.allergen == "penicillin")
        assert any(r.reaction == "anaphylaxis" for r in pcn.reactions)

    def test_rash_detected(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Sulfa causes rash.")
        sulfa = next(a for a in result.allergies if a.allergen == "sulfonamides")
        assert any(r.reaction == "rash" for r in sulfa.reactions)

    def test_hives_detected(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Amoxicillin: hives and itching.")
        amox = next(a for a in result.allergies if a.allergen == "amoxicillin")
        reactions = {r.reaction for r in amox.reactions}
        assert "hives" in reactions

    def test_gi_upset_detected(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Codeine - nausea and vomiting.")
        cod = next(a for a in result.allergies if a.allergen == "codeine")
        reactions = {r.reaction for r in cod.reactions}
        assert "nausea" in reactions or "vomiting" in reactions

    def test_multiple_reactions(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Penicillin causes hives, angioedema, and dyspnea.")
        pcn = next(a for a in result.allergies if a.allergen == "penicillin")
        assert len(pcn.reactions) >= 2

    def test_no_reactions_unknown_severity(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Allergic to lisinopril.")
        lis = next(a for a in result.allergies if a.allergen == "lisinopril")
        assert lis.severity == AllergySeverity.UNKNOWN


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

class TestSeverityClassification:
    """Test severity inference from reactions."""

    @pytest.fixture()
    def extractor(self) -> ClinicalAllergyExtractor:
        return ClinicalAllergyExtractor()

    def test_life_threatening_from_anaphylaxis(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("PCN - anaphylaxis")
        pcn = next(a for a in result.allergies if a.allergen == "penicillin")
        assert pcn.severity == AllergySeverity.LIFE_THREATENING

    def test_severe_from_angioedema(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("ACE inhibitor - angioedema")
        acei = next(a for a in result.allergies if a.allergen == "ace inhibitors")
        assert acei.severity == AllergySeverity.SEVERE

    def test_moderate_from_rash(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Sulfa: rash")
        sulfa = next(a for a in result.allergies if a.allergen == "sulfonamides")
        assert sulfa.severity == AllergySeverity.MODERATE

    def test_mild_from_nausea(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Codeine causes nausea.")
        cod = next(a for a in result.allergies if a.allergen == "codeine")
        assert cod.severity == AllergySeverity.MILD

    def test_severity_modifier_upgrade(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Amoxicillin: severe rash requiring hospitalisation.")
        amox = next(a for a in result.allergies if a.allergen == "amoxicillin")
        assert amox.severity == AllergySeverity.SEVERE

    def test_max_severity_multiple_reactions(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Penicillin: rash progressing to anaphylaxis.")
        pcn = next(a for a in result.allergies if a.allergen == "penicillin")
        assert pcn.severity == AllergySeverity.LIFE_THREATENING


# ---------------------------------------------------------------------------
# NKDA detection
# ---------------------------------------------------------------------------

class TestNKDADetection:
    """Test No Known Drug Allergies detection."""

    @pytest.fixture()
    def extractor(self) -> ClinicalAllergyExtractor:
        return ClinicalAllergyExtractor()

    def test_nkda_abbreviation(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Allergies: NKDA")
        assert result.no_known_allergies is True
        assert result.nkda_evidence == "NKDA"

    def test_nka_abbreviation(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("NKA")
        assert result.no_known_allergies is True

    def test_no_known_drug_allergies_text(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("No known drug allergies.")
        assert result.no_known_allergies is True

    def test_denies_allergies(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Patient denies any allergies.")
        assert result.no_known_allergies is True

    def test_no_allergies(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("No allergies.")
        assert result.no_known_allergies is True

    def test_no_nkda_when_allergies_present(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Allergic to penicillin.")
        assert result.no_known_allergies is False


# ---------------------------------------------------------------------------
# Negation / toleration handling
# ---------------------------------------------------------------------------

class TestNegationHandling:
    """Test that negated allergy mentions are marked as tolerated."""

    @pytest.fixture()
    def extractor(self) -> ClinicalAllergyExtractor:
        return ClinicalAllergyExtractor()

    def test_tolerates_pattern(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Patient tolerates penicillin without issues.")
        pcn = next((a for a in result.allergies if a.allergen == "penicillin"), None)
        assert pcn is not None
        assert pcn.status == AllergyStatus.TOLERATED

    def test_not_allergic_to(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Not allergic to sulfa.")
        sulfa = next((a for a in result.allergies if a.allergen == "sulfonamides"), None)
        assert sulfa is not None
        assert sulfa.status == AllergyStatus.TOLERATED

    def test_historical_allergy(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Previously allergic to amoxicillin as a child.")
        amox = next((a for a in result.allergies if a.allergen == "amoxicillin"), None)
        assert amox is not None
        assert amox.status == AllergyStatus.HISTORICAL

    def test_active_by_default(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Allergic to penicillin.")
        pcn = next(a for a in result.allergies if a.allergen == "penicillin")
        assert pcn.status == AllergyStatus.ACTIVE


# ---------------------------------------------------------------------------
# Section awareness
# ---------------------------------------------------------------------------

class TestSectionAwareness:
    """Test confidence boosting inside allergy sections."""

    @pytest.fixture()
    def extractor(self) -> ClinicalAllergyExtractor:
        return ClinicalAllergyExtractor()

    def test_higher_confidence_in_allergy_section(self, extractor: ClinicalAllergyExtractor) -> None:
        # In section.
        text_in = "ALLERGIES:\nPenicillin - rash"
        result_in = extractor.extract(text_in)
        # Outside section.
        text_out = "Patient mentioned penicillin causes rash."
        result_out = extractor.extract(text_out)

        pcn_in = next(a for a in result_in.allergies if a.allergen == "penicillin")
        pcn_out = next(a for a in result_out.allergies if a.allergen == "penicillin")
        assert pcn_in.confidence > pcn_out.confidence

    def test_drug_allergies_header(self, extractor: ClinicalAllergyExtractor) -> None:
        text = "DRUG ALLERGIES:\nSulfa - hives"
        result = extractor.extract(text)
        sulfa = next(a for a in result.allergies if a.allergen == "sulfonamides")
        # Should have section boost.
        assert sulfa.confidence >= 0.80


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

class TestConfidenceScoring:
    """Test confidence computation rules."""

    @pytest.fixture()
    def extractor(self) -> ClinicalAllergyExtractor:
        return ClinicalAllergyExtractor()

    def test_drug_allergen_boost(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Allergic to penicillin.")
        pcn = next(a for a in result.allergies if a.allergen == "penicillin")
        # Base 0.70 + drug 0.05 = 0.75
        assert pcn.confidence >= 0.75

    def test_reaction_boost(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Penicillin causes anaphylaxis.")
        pcn = next(a for a in result.allergies if a.allergen == "penicillin")
        # Base 0.70 + drug 0.05 + reaction 0.10 = 0.85
        assert pcn.confidence >= 0.85

    def test_section_plus_reaction_boost(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("ALLERGIES:\nPenicillin - anaphylaxis")
        pcn = next(a for a in result.allergies if a.allergen == "penicillin")
        # Base 0.70 + drug 0.05 + reaction 0.10 + section 0.10 = 0.95
        assert pcn.confidence >= 0.95

    def test_min_confidence_filtering(self) -> None:
        extractor = ClinicalAllergyExtractor(min_confidence=0.90)
        result = extractor.extract("Patient mentioned aspirin once.")
        # Low confidence mentions should be filtered.
        assert len(result.allergies) == 0 or all(a.confidence >= 0.90 for a in result.allergies)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    """Test allergen deduplication."""

    @pytest.fixture()
    def extractor(self) -> ClinicalAllergyExtractor:
        return ClinicalAllergyExtractor()

    def test_same_allergen_deduped(self, extractor: ClinicalAllergyExtractor) -> None:
        text = "Allergic to penicillin. PCN causes rash. Penicillin anaphylaxis."
        result = extractor.extract(text)
        pcn_count = sum(1 for a in result.allergies if a.allergen == "penicillin")
        assert pcn_count == 1

    def test_keeps_highest_confidence(self, extractor: ClinicalAllergyExtractor) -> None:
        text = "ALLERGIES:\nPCN - anaphylaxis\n\nOther mention of penicillin in history."
        result = extractor.extract(text)
        pcn = next(a for a in result.allergies if a.allergen == "penicillin")
        # The section-boosted one should win.
        assert pcn.confidence >= 0.85


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------

class TestBatchExtraction:
    """Test batch allergy extraction."""

    def test_batch_returns_per_document(self) -> None:
        extractor = ClinicalAllergyExtractor()
        texts = [
            "Allergies: PCN",
            "NKDA",
            "Allergic to peanuts and shellfish.",
        ]
        results = extractor.extract_batch(texts)
        assert len(results) == 3
        assert len(results[0].allergies) >= 1
        assert results[1].no_known_allergies is True
        assert len(results[2].allergies) >= 2

    def test_batch_preserves_order(self) -> None:
        extractor = ClinicalAllergyExtractor()
        texts = ["Allergic to aspirin.", "NKDA", "Allergic to codeine."]
        results = extractor.extract_batch(texts)
        assert results[0].allergies[0].allergen == "aspirin"
        assert results[1].no_known_allergies is True
        assert results[2].allergies[0].allergen == "codeine"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for allergy extraction."""

    @pytest.fixture()
    def extractor(self) -> ClinicalAllergyExtractor:
        return ClinicalAllergyExtractor()

    def test_empty_string(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("")
        assert len(result.allergies) == 0
        assert result.text_length == 0

    def test_whitespace_only(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("   \n\t  ")
        assert len(result.allergies) == 0

    def test_no_allergens_found(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Patient presents with chest pain and shortness of breath.")
        assert len(result.allergies) == 0
        assert result.no_known_allergies is False

    def test_case_insensitive_detection(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("PENICILLIN causes ANAPHYLAXIS")
        assert any(a.allergen == "penicillin" for a in result.allergies)

    def test_allergen_in_parentheses(self, extractor: ClinicalAllergyExtractor) -> None:
        result = extractor.extract("Allergies: penicillin (rash)")
        pcn = next(a for a in result.allergies if a.allergen == "penicillin")
        assert any(r.reaction == "rash" for r in pcn.reactions)

    def test_comma_separated_list(self, extractor: ClinicalAllergyExtractor) -> None:
        text = "Allergies: PCN, sulfa, codeine, latex"
        result = extractor.extract(text)
        assert len(result.allergies) >= 4

    def test_nkda_with_specific_allergies(self, extractor: ClinicalAllergyExtractor) -> None:
        """NKDA can coexist with detected allergens (contradictory but real)."""
        text = "NKDA. However, patient reports penicillin causes rash."
        result = extractor.extract(text)
        assert result.no_known_allergies is True
        assert len(result.allergies) >= 1


# ---------------------------------------------------------------------------
# Realistic clinical notes
# ---------------------------------------------------------------------------

class TestRealisticNotes:
    """Test with realistic clinical allergy documentation."""

    @pytest.fixture()
    def extractor(self) -> ClinicalAllergyExtractor:
        return ClinicalAllergyExtractor()

    def test_standard_allergy_list(self, extractor: ClinicalAllergyExtractor) -> None:
        text = (
            "ALLERGIES:\n"
            "1. Penicillin - anaphylaxis (confirmed 2019)\n"
            "2. Sulfa - rash\n"
            "3. Codeine - nausea/vomiting\n"
            "4. Latex - contact dermatitis\n"
            "5. Shellfish - hives, throat swelling\n"
        )
        result = extractor.extract(text)
        assert len(result.allergies) >= 5

        # Check specific allergies.
        pcn = next(a for a in result.allergies if a.allergen == "penicillin")
        assert pcn.severity == AllergySeverity.LIFE_THREATENING

        sulfa = next(a for a in result.allergies if a.allergen == "sulfonamides")
        assert sulfa.severity == AllergySeverity.MODERATE

    def test_discharge_summary_allergies(self, extractor: ClinicalAllergyExtractor) -> None:
        text = (
            "DISCHARGE SUMMARY\n\n"
            "ALLERGIES: PCN (anaphylaxis), ASA (GI bleeding), "
            "Contrast dye (urticaria)\n\n"
            "MEDICATIONS:\n"
            "Lisinopril 20mg daily\n"
        )
        result = extractor.extract(text)
        allergens = {a.allergen for a in result.allergies}
        assert "penicillin" in allergens
        assert "aspirin" in allergens
        assert "iodine contrast" in allergens

    def test_nkda_note(self, extractor: ClinicalAllergyExtractor) -> None:
        text = (
            "ALLERGIES: NKDA\n\n"
            "MEDICATIONS:\n"
            "Metformin 1000mg BID\n"
            "Lisinopril 10mg daily\n"
        )
        result = extractor.extract(text)
        assert result.no_known_allergies is True

    def test_mixed_categories(self, extractor: ClinicalAllergyExtractor) -> None:
        text = (
            "Drug Allergies:\n"
            "- Penicillin (anaphylaxis)\n"
            "- Sulfa (rash)\n\n"
            "Food Allergies:\n"
            "- Peanuts (throat swelling)\n"
            "- Shellfish (hives)\n\n"
            "Environmental Allergies:\n"
            "- Pollen (seasonal rhinitis)\n"
            "- Dust mites\n"
        )
        result = extractor.extract(text)
        categories = {a.category for a in result.allergies}
        assert AllergyCategory.DRUG in categories
        assert AllergyCategory.FOOD in categories
        assert AllergyCategory.ENVIRONMENTAL in categories
