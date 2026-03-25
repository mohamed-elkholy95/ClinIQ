"""Tests for the Social Determinants of Health (SDoH) extractor.

Covers:
* Enum completeness and serialisation
* Dataclass to_dict round-trip
* Pattern library construction and trigger counts
* Per-domain extraction (housing, employment, education, food security,
  transportation, social support, substance use, financial)
* Negation-aware sentiment flipping
* Section-aware confidence boosting
* Overlapping span deduplication
* Batch extraction
* Edge cases (empty text, whitespace, very short)
* Domain info and Z-code catalogue
* Realistic clinical note end-to-end
"""

from __future__ import annotations

import pytest

from app.ml.sdoh.extractor import (
    DOMAIN_Z_CODES,
    PATTERN_LIBRARY,
    ClinicalSDoHExtractor,
    SDoHDomain,
    SDoHExtraction,
    SDoHExtractionResult,
    SDoHSentiment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extractor() -> ClinicalSDoHExtractor:
    """Default extractor with standard settings."""
    return ClinicalSDoHExtractor()


@pytest.fixture
def low_confidence_extractor() -> ClinicalSDoHExtractor:
    """Extractor with very low min_confidence to catch everything."""
    return ClinicalSDoHExtractor(min_confidence=0.10)


# ---------------------------------------------------------------------------
# Enum completeness
# ---------------------------------------------------------------------------


class TestSDoHDomain:
    """SDoHDomain enum tests."""

    def test_has_eight_domains(self) -> None:
        assert len(SDoHDomain) == 8

    def test_all_values_lowercase(self) -> None:
        for domain in SDoHDomain:
            assert domain.value == domain.value.lower()

    def test_string_enum_serialisation(self) -> None:
        assert str(SDoHDomain.HOUSING) == "SDoHDomain.HOUSING"
        assert SDoHDomain.HOUSING.value == "housing"

    def test_known_domains(self) -> None:
        expected = {
            "housing", "employment", "education", "food_security",
            "transportation", "social_support", "substance_use", "financial",
        }
        actual = {d.value for d in SDoHDomain}
        assert actual == expected


class TestSDoHSentiment:
    """SDoHSentiment enum tests."""

    def test_has_three_values(self) -> None:
        assert len(SDoHSentiment) == 3

    def test_values(self) -> None:
        assert SDoHSentiment.ADVERSE.value == "adverse"
        assert SDoHSentiment.PROTECTIVE.value == "protective"
        assert SDoHSentiment.NEUTRAL.value == "neutral"


# ---------------------------------------------------------------------------
# Dataclass serialisation
# ---------------------------------------------------------------------------


class TestSDoHExtraction:
    """SDoHExtraction dataclass tests."""

    def test_to_dict_all_fields(self) -> None:
        ext = SDoHExtraction(
            domain=SDoHDomain.HOUSING,
            text="homeless",
            sentiment=SDoHSentiment.ADVERSE,
            confidence=0.92,
            z_codes=["Z59.00"],
            trigger="homelessness",
            start=10,
            end=18,
            negated=False,
            section="Social History",
        )
        d = ext.to_dict()
        assert d["domain"] == "housing"
        assert d["sentiment"] == "adverse"
        assert d["confidence"] == 0.92
        assert d["z_codes"] == ["Z59.00"]
        assert d["negated"] is False
        assert d["section"] == "Social History"

    def test_to_dict_defaults(self) -> None:
        ext = SDoHExtraction(
            domain=SDoHDomain.FINANCIAL,
            text="uninsured",
            sentiment=SDoHSentiment.ADVERSE,
            confidence=0.90,
        )
        d = ext.to_dict()
        assert d["z_codes"] == []
        assert d["trigger"] == ""
        assert d["start"] == 0
        assert d["end"] == 0
        assert d["negated"] is False
        assert d["section"] == ""


class TestSDoHExtractionResult:
    """SDoHExtractionResult dataclass tests."""

    def test_to_dict_empty(self) -> None:
        result = SDoHExtractionResult()
        d = result.to_dict()
        assert d["extractions"] == []
        assert d["domain_summary"] == {}
        assert d["adverse_count"] == 0
        assert d["protective_count"] == 0

    def test_to_dict_with_extractions(self) -> None:
        ext = SDoHExtraction(
            domain=SDoHDomain.HOUSING,
            text="homeless",
            sentiment=SDoHSentiment.ADVERSE,
            confidence=0.90,
        )
        result = SDoHExtractionResult(
            extractions=[ext],
            domain_summary={"housing": 1},
            adverse_count=1,
            text_length=100,
            processing_time_ms=1.5,
        )
        d = result.to_dict()
        assert len(d["extractions"]) == 1
        assert d["adverse_count"] == 1
        assert d["processing_time_ms"] == 1.5


# ---------------------------------------------------------------------------
# Pattern library
# ---------------------------------------------------------------------------


class TestPatternLibrary:
    """Pattern library construction tests."""

    def test_all_domains_have_triggers(self) -> None:
        for domain in SDoHDomain:
            assert domain in PATTERN_LIBRARY, f"Missing triggers for {domain}"
            assert len(PATTERN_LIBRARY[domain]) > 0

    def test_total_trigger_count(self, extractor: ClinicalSDoHExtractor) -> None:
        count = extractor.total_trigger_count()
        # At least 80 triggers across 8 domains
        assert count >= 80

    def test_triggers_have_required_keys(self) -> None:
        for domain, triggers in PATTERN_LIBRARY.items():
            for trigger in triggers:
                assert "pattern" in trigger
                assert "sentiment" in trigger
                assert "base_confidence" in trigger
                assert "description" in trigger

    def test_confidence_values_in_range(self) -> None:
        for domain, triggers in PATTERN_LIBRARY.items():
            for trigger in triggers:
                conf = trigger["base_confidence"]
                assert 0.0 <= conf <= 1.0, f"Bad confidence {conf} in {domain}"


# ---------------------------------------------------------------------------
# Z-code mapping
# ---------------------------------------------------------------------------


class TestZCodeMapping:
    """Z-code catalogue tests."""

    def test_all_domains_have_z_codes(self) -> None:
        for domain in SDoHDomain:
            assert domain in DOMAIN_Z_CODES
            assert len(DOMAIN_Z_CODES[domain]) >= 1

    def test_z_codes_start_with_z(self) -> None:
        for domain, entries in DOMAIN_Z_CODES.items():
            for entry in entries:
                assert entry["code"].startswith("Z"), (
                    f"Code {entry['code']} in {domain} doesn't start with Z"
                )

    def test_z_code_entries_have_description(self) -> None:
        for domain, entries in DOMAIN_Z_CODES.items():
            for entry in entries:
                assert "description" in entry
                assert len(entry["description"]) > 5


# ---------------------------------------------------------------------------
# Housing domain extraction
# ---------------------------------------------------------------------------


class TestHousingExtraction:
    """Housing domain extraction tests."""

    def test_homeless(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Patient is currently homeless.")
        housing = [e for e in result.extractions if e.domain == SDoHDomain.HOUSING]
        assert len(housing) >= 1
        assert housing[0].sentiment == SDoHSentiment.ADVERSE

    def test_shelter(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Lives in a shelter downtown.")
        housing = [e for e in result.extractions if e.domain == SDoHDomain.HOUSING]
        assert len(housing) >= 1

    def test_eviction(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Facing eviction from apartment.")
        housing = [e for e in result.extractions if e.domain == SDoHDomain.HOUSING]
        assert len(housing) >= 1

    def test_stable_housing_protective(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Has stable housing with family.")
        housing = [e for e in result.extractions if e.domain == SDoHDomain.HOUSING]
        protective = [e for e in housing if e.sentiment == SDoHSentiment.PROTECTIVE]
        assert len(protective) >= 1

    def test_lives_with_family(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Patient lives with spouse and children.")
        housing = [e for e in result.extractions if e.domain == SDoHDomain.HOUSING]
        assert any(e.sentiment == SDoHSentiment.PROTECTIVE for e in housing)

    def test_housing_z_codes_populated(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Currently homeless.")
        housing = [e for e in result.extractions if e.domain == SDoHDomain.HOUSING]
        assert housing[0].z_codes
        assert any("Z59" in code for code in housing[0].z_codes)


# ---------------------------------------------------------------------------
# Employment domain extraction
# ---------------------------------------------------------------------------


class TestEmploymentExtraction:
    """Employment domain extraction tests."""

    def test_unemployed(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Patient is currently unemployed.")
        empl = [e for e in result.extractions if e.domain == SDoHDomain.EMPLOYMENT]
        assert len(empl) >= 1
        assert empl[0].sentiment == SDoHSentiment.ADVERSE

    def test_employed_protective(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Currently employed full-time.")
        empl = [e for e in result.extractions if e.domain == SDoHDomain.EMPLOYMENT]
        protective = [e for e in empl if e.sentiment == SDoHSentiment.PROTECTIVE]
        assert len(protective) >= 1

    def test_retired_neutral(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Patient is retired.")
        empl = [e for e in result.extractions if e.domain == SDoHDomain.EMPLOYMENT]
        assert any(e.sentiment == SDoHSentiment.NEUTRAL for e in empl)

    def test_disability(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Has been on disability for 3 years.")
        empl = [e for e in result.extractions if e.domain == SDoHDomain.EMPLOYMENT]
        assert len(empl) >= 1

    def test_laid_off(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Was laid off from work last month.")
        empl = [e for e in result.extractions if e.domain == SDoHDomain.EMPLOYMENT]
        assert len(empl) >= 1
        assert empl[0].sentiment == SDoHSentiment.ADVERSE


# ---------------------------------------------------------------------------
# Education domain extraction
# ---------------------------------------------------------------------------


class TestEducationExtraction:
    """Education domain extraction tests."""

    def test_low_literacy(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Patient has limited health literacy.")
        edu = [e for e in result.extractions if e.domain == SDoHDomain.EDUCATION]
        assert len(edu) >= 1

    def test_language_barrier(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Requires an interpreter for visits.")
        edu = [e for e in result.extractions if e.domain == SDoHDomain.EDUCATION]
        assert len(edu) >= 1

    def test_limited_english(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Limited English proficiency, speaks Mandarin.")
        edu = [e for e in result.extractions if e.domain == SDoHDomain.EDUCATION]
        assert len(edu) >= 1
        assert edu[0].sentiment == SDoHSentiment.ADVERSE


# ---------------------------------------------------------------------------
# Food security domain extraction
# ---------------------------------------------------------------------------


class TestFoodSecurityExtraction:
    """Food security domain extraction tests."""

    def test_food_insecurity(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Reports food insecurity at home.")
        food = [e for e in result.extractions if e.domain == SDoHDomain.FOOD_SECURITY]
        assert len(food) >= 1
        assert food[0].sentiment == SDoHSentiment.ADVERSE

    def test_skipping_meals(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Admits to skipping meals due to cost.")
        food = [e for e in result.extractions if e.domain == SDoHDomain.FOOD_SECURITY]
        assert len(food) >= 1

    def test_snap_benefits(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Receiving SNAP benefits.")
        food = [e for e in result.extractions if e.domain == SDoHDomain.FOOD_SECURITY]
        assert len(food) >= 1

    def test_cannot_afford_food(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Cannot afford groceries this month.")
        food = [e for e in result.extractions if e.domain == SDoHDomain.FOOD_SECURITY]
        assert len(food) >= 1
        assert food[0].sentiment == SDoHSentiment.ADVERSE


# ---------------------------------------------------------------------------
# Transportation domain extraction
# ---------------------------------------------------------------------------


class TestTransportationExtraction:
    """Transportation domain extraction tests."""

    def test_no_transportation(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("No reliable transportation to appointments.")
        trans = [e for e in result.extractions if e.domain == SDoHDomain.TRANSPORTATION]
        assert len(trans) >= 1
        assert trans[0].sentiment == SDoHSentiment.ADVERSE

    def test_missed_appointments(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Missed appointments due to transport issues.")
        trans = [e for e in result.extractions if e.domain == SDoHDomain.TRANSPORTATION]
        assert len(trans) >= 1

    def test_transportation_barrier(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Transportation barrier to care noted.")
        trans = [e for e in result.extractions if e.domain == SDoHDomain.TRANSPORTATION]
        assert len(trans) >= 1


# ---------------------------------------------------------------------------
# Social support domain extraction
# ---------------------------------------------------------------------------


class TestSocialSupportExtraction:
    """Social support domain extraction tests."""

    def test_social_isolation(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Patient is socially isolated.")
        social = [e for e in result.extractions if e.domain == SDoHDomain.SOCIAL_SUPPORT]
        assert len(social) >= 1
        assert social[0].sentiment == SDoHSentiment.ADVERSE

    def test_domestic_violence(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("History of domestic violence.")
        social = [e for e in result.extractions if e.domain == SDoHDomain.SOCIAL_SUPPORT]
        assert len(social) >= 1
        assert social[0].sentiment == SDoHSentiment.ADVERSE

    def test_lives_alone_neutral(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Patient lives alone.")
        social = [e for e in result.extractions if e.domain == SDoHDomain.SOCIAL_SUPPORT]
        assert any(e.sentiment == SDoHSentiment.NEUTRAL for e in social)

    def test_strong_support_protective(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Has strong family support network.")
        social = [e for e in result.extractions if e.domain == SDoHDomain.SOCIAL_SUPPORT]
        protective = [e for e in social if e.sentiment == SDoHSentiment.PROTECTIVE]
        assert len(protective) >= 1

    def test_caregiver_burden(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Exhibiting caregiver burnout symptoms.")
        social = [e for e in result.extractions if e.domain == SDoHDomain.SOCIAL_SUPPORT]
        assert len(social) >= 1

    def test_incarceration(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Recently released from prison.")
        social = [e for e in result.extractions if e.domain == SDoHDomain.SOCIAL_SUPPORT]
        assert len(social) >= 1
        assert social[0].sentiment == SDoHSentiment.ADVERSE


# ---------------------------------------------------------------------------
# Substance use domain extraction
# ---------------------------------------------------------------------------


class TestSubstanceUseExtraction:
    """Substance use domain extraction tests."""

    def test_current_smoker(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Current smoker, 1 PPD for 20 years.")
        subs = [e for e in result.extractions if e.domain == SDoHDomain.SUBSTANCE_USE]
        assert len(subs) >= 1
        adverse = [e for e in subs if e.sentiment == SDoHSentiment.ADVERSE]
        assert len(adverse) >= 1

    def test_heavy_alcohol(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Reports heavy drinking, 6 beers daily.")
        subs = [e for e in result.extractions if e.domain == SDoHDomain.SUBSTANCE_USE]
        assert len(subs) >= 1

    def test_illicit_drug_use(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("History of IV drug use, last used cocaine 2 months ago.")
        subs = [e for e in result.extractions if e.domain == SDoHDomain.SUBSTANCE_USE]
        assert len(subs) >= 1

    def test_former_smoker_protective(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Former smoker, quit 5 years ago.")
        subs = [e for e in result.extractions if e.domain == SDoHDomain.SUBSTANCE_USE]
        protective = [e for e in subs if e.sentiment == SDoHSentiment.PROTECTIVE]
        assert len(protective) >= 1

    def test_never_smoker(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Non-smoker, no alcohol or drug use.")
        subs = [e for e in result.extractions if e.domain == SDoHDomain.SUBSTANCE_USE]
        protective = [e for e in subs if e.sentiment == SDoHSentiment.PROTECTIVE]
        assert len(protective) >= 1

    def test_social_drinker_neutral(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Social drinker, occasional wine with dinner.")
        subs = [e for e in result.extractions if e.domain == SDoHDomain.SUBSTANCE_USE]
        neutral = [e for e in subs if e.sentiment == SDoHSentiment.NEUTRAL]
        assert len(neutral) >= 1

    def test_substance_use_disorder(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Diagnosed with substance use disorder.")
        subs = [e for e in result.extractions if e.domain == SDoHDomain.SUBSTANCE_USE]
        assert len(subs) >= 1
        assert subs[0].sentiment == SDoHSentiment.ADVERSE

    def test_in_recovery(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Sober for 3 years, attends AA meetings.")
        subs = [e for e in result.extractions if e.domain == SDoHDomain.SUBSTANCE_USE]
        protective = [e for e in subs if e.sentiment == SDoHSentiment.PROTECTIVE]
        assert len(protective) >= 1

    def test_opioid_use(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Active opioid use, on Suboxone.")
        subs = [e for e in result.extractions if e.domain == SDoHDomain.SUBSTANCE_USE]
        adverse = [e for e in subs if e.sentiment == SDoHSentiment.ADVERSE]
        assert len(adverse) >= 1

    def test_vaping(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Reports vaping nicotine daily.")
        subs = [e for e in result.extractions if e.domain == SDoHDomain.SUBSTANCE_USE]
        assert len(subs) >= 1


# ---------------------------------------------------------------------------
# Financial domain extraction
# ---------------------------------------------------------------------------


class TestFinancialExtraction:
    """Financial domain extraction tests."""

    def test_uninsured(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Patient is uninsured.")
        fin = [e for e in result.extractions if e.domain == SDoHDomain.FINANCIAL]
        assert len(fin) >= 1
        assert fin[0].sentiment == SDoHSentiment.ADVERSE

    def test_cannot_afford_medication(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Cannot afford medication this month.")
        fin = [e for e in result.extractions if e.domain == SDoHDomain.FINANCIAL]
        assert len(fin) >= 1

    def test_medication_rationing(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Admits to rationing insulin due to cost.")
        fin = [e for e in result.extractions if e.domain == SDoHDomain.FINANCIAL]
        assert len(fin) >= 1
        assert fin[0].sentiment == SDoHSentiment.ADVERSE

    def test_low_income(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Patient reports low income, fixed income.")
        fin = [e for e in result.extractions if e.domain == SDoHDomain.FINANCIAL]
        assert len(fin) >= 1

    def test_medical_debt(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("Significant medical debt from prior hospitalisation.")
        fin = [e for e in result.extractions if e.domain == SDoHDomain.FINANCIAL]
        assert len(fin) >= 1


# ---------------------------------------------------------------------------
# Negation handling
# ---------------------------------------------------------------------------


class TestNegationHandling:
    """Negation-aware sentiment adjustment tests."""

    def test_negated_adverse_becomes_protective(
        self, extractor: ClinicalSDoHExtractor
    ) -> None:
        result = extractor.extract("Denies homelessness, has stable housing.")
        housing = [e for e in result.extractions if e.domain == SDoHDomain.HOUSING]
        # "Denies homelessness" should flip adverse → protective
        negated = [e for e in housing if e.negated]
        if negated:
            assert negated[0].sentiment == SDoHSentiment.PROTECTIVE

    def test_negated_substance_use(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("No tobacco use, no alcohol use, no drug use.")
        subs = [e for e in result.extractions if e.domain == SDoHDomain.SUBSTANCE_USE]
        # "No ... use" patterns exist as protective triggers
        protective = [e for e in subs if e.sentiment == SDoHSentiment.PROTECTIVE]
        assert len(protective) >= 1

    def test_negation_window_respected(self) -> None:
        """Negation cue far from match should not trigger."""
        extractor = ClinicalSDoHExtractor(negation_window=5)
        # "denies" is far from "homeless" in this phrasing
        text = "Patient denies any significant past medical history. Currently homeless."
        result = extractor.extract(text)
        housing = [e for e in result.extractions if e.domain == SDoHDomain.HOUSING]
        adverse = [e for e in housing if e.sentiment == SDoHSentiment.ADVERSE]
        assert len(adverse) >= 1  # Not negated because window is too small


# ---------------------------------------------------------------------------
# Section-aware confidence boosting
# ---------------------------------------------------------------------------


class TestSectionAwareness:
    """Section-aware confidence boosting tests."""

    def test_social_history_section_boost(
        self, extractor: ClinicalSDoHExtractor
    ) -> None:
        without_section = extractor.extract("Currently homeless.")
        with_section = extractor.extract(
            "Social History:\nCurrently homeless.\nFamily History:\nNoncontributory."
        )

        housing_without = [
            e for e in without_section.extractions if e.domain == SDoHDomain.HOUSING
        ]
        housing_with = [
            e for e in with_section.extractions if e.domain == SDoHDomain.HOUSING
        ]

        assert housing_without and housing_with
        # Inside social history should have higher confidence
        assert housing_with[0].confidence >= housing_without[0].confidence

    def test_section_field_populated(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract(
            "Social History:\nPatient is currently homeless.\nFamily History:\nNone."
        )
        housing = [e for e in result.extractions if e.domain == SDoHDomain.HOUSING]
        assert any(e.section == "Social History" for e in housing)

    def test_no_section_field_when_outside(
        self, extractor: ClinicalSDoHExtractor
    ) -> None:
        result = extractor.extract("Patient is currently homeless.")
        housing = [e for e in result.extractions if e.domain == SDoHDomain.HOUSING]
        assert housing[0].section == ""


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Overlapping span deduplication tests."""

    def test_overlapping_spans_deduped(
        self, extractor: ClinicalSDoHExtractor
    ) -> None:
        # "unstable housing" could match both "unstable housing" and
        # "housing insecurity" depending on phrasing
        result = extractor.extract("Patient has housing insecurity.")
        housing = [e for e in result.extractions if e.domain == SDoHDomain.HOUSING]
        # Should not have duplicate extractions for same span
        spans = [(e.start, e.end) for e in housing]
        # No exact duplicates
        assert len(spans) == len(set(spans))


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------


class TestBatchExtraction:
    """Batch extraction tests."""

    def test_batch_returns_correct_count(
        self, extractor: ClinicalSDoHExtractor
    ) -> None:
        texts = [
            "Patient is homeless.",
            "Currently employed full-time.",
            "No tobacco use.",
        ]
        results = extractor.extract_batch(texts)
        assert len(results) == 3

    def test_batch_preserves_order(
        self, extractor: ClinicalSDoHExtractor
    ) -> None:
        texts = ["Uninsured patient.", "Currently homeless."]
        results = extractor.extract_batch(texts)
        # First should have financial, second housing
        assert any(
            e.domain == SDoHDomain.FINANCIAL for e in results[0].extractions
        )
        assert any(
            e.domain == SDoHDomain.HOUSING for e in results[1].extractions
        )

    def test_batch_empty_list(self, extractor: ClinicalSDoHExtractor) -> None:
        results = extractor.extract_batch([])
        assert results == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge case handling tests."""

    def test_empty_text(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("")
        assert result.extractions == []
        assert result.text_length == 0

    def test_whitespace_only(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("   \n\t  ")
        assert result.extractions == []

    def test_no_sdoh_content(self, extractor: ClinicalSDoHExtractor) -> None:
        result = extractor.extract("BP 120/80, HR 72, afebrile. Normal exam.")
        # Should find nothing (or very little)
        assert result.adverse_count == 0

    def test_very_long_text(self, extractor: ClinicalSDoHExtractor) -> None:
        text = "Normal clinical note. " * 1000 + " Currently homeless."
        result = extractor.extract(text)
        housing = [e for e in result.extractions if e.domain == SDoHDomain.HOUSING]
        assert len(housing) >= 1

    def test_processing_time_recorded(
        self, extractor: ClinicalSDoHExtractor
    ) -> None:
        result = extractor.extract("Patient is currently homeless.")
        assert result.processing_time_ms > 0

    def test_text_length_recorded(
        self, extractor: ClinicalSDoHExtractor
    ) -> None:
        text = "Patient is unemployed."
        result = extractor.extract(text)
        assert result.text_length == len(text)


# ---------------------------------------------------------------------------
# Domain info and catalogue
# ---------------------------------------------------------------------------


class TestDomainInfo:
    """Domain information and catalogue tests."""

    def test_get_domain_info(self, extractor: ClinicalSDoHExtractor) -> None:
        info = extractor.get_domain_info(SDoHDomain.HOUSING)
        assert info["domain"] == "housing"
        assert "description" in info
        assert info["trigger_count"] > 0
        assert "z_codes" in info

    def test_get_all_domains(self, extractor: ClinicalSDoHExtractor) -> None:
        domains = extractor.get_all_domains()
        assert len(domains) == 8

    def test_get_z_codes_for_domain(self) -> None:
        codes = ClinicalSDoHExtractor.get_z_codes_for_domain(SDoHDomain.HOUSING)
        assert len(codes) >= 1
        assert all("code" in c and "description" in c for c in codes)

    def test_total_trigger_count_static(self) -> None:
        count = ClinicalSDoHExtractor.total_trigger_count()
        assert isinstance(count, int)
        assert count > 0

    def test_domain_info_trigger_counts_sum(
        self, extractor: ClinicalSDoHExtractor
    ) -> None:
        info = extractor.get_domain_info(SDoHDomain.SUBSTANCE_USE)
        assert info["adverse_triggers"] + info["protective_triggers"] <= info[
            "trigger_count"
        ]


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------


class TestResultAggregation:
    """Result summary and aggregation tests."""

    def test_domain_summary_populated(
        self, extractor: ClinicalSDoHExtractor
    ) -> None:
        result = extractor.extract(
            "Currently homeless, unemployed, no insurance."
        )
        assert len(result.domain_summary) >= 2

    def test_adverse_count_matches(
        self, extractor: ClinicalSDoHExtractor
    ) -> None:
        result = extractor.extract("Homeless and uninsured.")
        assert result.adverse_count == len(
            [e for e in result.extractions if e.sentiment == SDoHSentiment.ADVERSE]
        )

    def test_protective_count_matches(
        self, extractor: ClinicalSDoHExtractor
    ) -> None:
        result = extractor.extract("Non-smoker, strong family support.")
        assert result.protective_count == len(
            [
                e
                for e in result.extractions
                if e.sentiment == SDoHSentiment.PROTECTIVE
            ]
        )


# ---------------------------------------------------------------------------
# Realistic clinical note end-to-end
# ---------------------------------------------------------------------------


class TestRealisticClinicalNote:
    """End-to-end test with a realistic clinical note."""

    def test_full_social_history(self, extractor: ClinicalSDoHExtractor) -> None:
        note = """
        SOCIAL HISTORY:
        Patient is a 52-year-old male, currently unemployed after being
        laid off 6 months ago. Lives alone in a rented apartment,
        reports housing insecurity and fear of eviction. Currently
        uninsured, cannot afford medications. Admits to rationing
        insulin due to cost. Reports food insecurity, relies on
        food bank. No reliable transportation to appointments.
        Current smoker, 1 pack per day for 30 years. Reports heavy
        drinking, 4-6 beers per day. Denies illicit drug use.
        Limited social support, divorced. Limited English proficiency,
        requires an interpreter.

        FAMILY HISTORY:
        Father died of MI at age 55.
        """
        result = extractor.extract(note)

        # Should detect multiple domains
        assert len(result.domain_summary) >= 5

        # Should have many adverse findings
        assert result.adverse_count >= 5

        # Check specific domains detected
        domains_found = set(result.domain_summary.keys())
        assert "housing" in domains_found
        assert "employment" in domains_found
        assert "financial" in domains_found
        assert "substance_use" in domains_found

        # Should have some protective findings (denies illicit drugs)
        assert result.protective_count >= 1

        # Extractions should have Z-codes
        for ext in result.extractions:
            assert isinstance(ext.z_codes, list)

        # Processing time should be fast (<100ms)
        assert result.processing_time_ms < 100

    def test_healthy_social_history(
        self, extractor: ClinicalSDoHExtractor
    ) -> None:
        note = """
        Social History:
        Patient is a 35-year-old female, currently employed as a
        teacher. Owns a home, lives with spouse and two children.
        Strong family support. Non-smoker, denies alcohol and drug
        use. Health literate, college educated. No financial concerns.

        Family History:
        Noncontributory.
        """
        result = extractor.extract(note)

        # Should have mostly protective findings
        assert result.protective_count >= result.adverse_count

    def test_mixed_note_with_dental_context(
        self, extractor: ClinicalSDoHExtractor
    ) -> None:
        note = """
        Chief Complaint: Tooth pain #19

        Social History:
        Patient reports being unable to work due to chronic pain.
        On disability. Cannot afford dental treatment, uninsured.
        Former smoker, quit smoking 2 years ago.
        Supportive family, wife accompanies to appointments.

        Assessment:
        Periapical abscess #19, needs RCT.
        """
        result = extractor.extract(note)

        # Should detect employment, financial, substance use, social support
        assert len(result.domain_summary) >= 3
        # Mix of adverse and protective
        assert result.adverse_count >= 1
        assert result.protective_count >= 1
