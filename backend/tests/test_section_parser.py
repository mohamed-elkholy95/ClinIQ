"""Tests for the unified clinical section parser.

Covers section detection, category mapping, span computation, position
queries, and edge cases across multiple header formats.
"""

from __future__ import annotations

import pytest

from app.ml.sections.parser import (
    _HEADER_TO_CATEGORY,
    ClinicalSectionParser,
    SectionCategory,
    SectionParseResult,
    SectionSpan,
    _normalise_header,
)

# ---------------------------------------------------------------------------
# Enum completeness
# ---------------------------------------------------------------------------

class TestSectionCategoryEnum:
    """Ensure the category enum covers all expected clinical sections."""

    def test_has_chief_complaint(self) -> None:
        assert SectionCategory.CHIEF_COMPLAINT == "chief_complaint"

    def test_has_hpi(self) -> None:
        assert SectionCategory.HISTORY_PRESENT_ILLNESS == "history_of_present_illness"

    def test_has_vital_signs(self) -> None:
        assert SectionCategory.VITAL_SIGNS == "vital_signs"

    def test_has_medications(self) -> None:
        assert SectionCategory.MEDICATIONS == "medications"

    def test_has_allergies(self) -> None:
        assert SectionCategory.ALLERGIES == "allergies"

    def test_has_assessment_and_plan(self) -> None:
        assert SectionCategory.ASSESSMENT_AND_PLAN == "assessment_and_plan"

    def test_has_dental_sections(self) -> None:
        assert SectionCategory.DENTAL_HISTORY == "dental_history"
        assert SectionCategory.PERIODONTAL_ASSESSMENT == "periodontal_assessment"
        assert SectionCategory.ORAL_EXAMINATION == "oral_examination"

    def test_has_unknown(self) -> None:
        assert SectionCategory.UNKNOWN == "unknown"

    def test_minimum_category_count(self) -> None:
        """At least 35 section categories defined."""
        assert len(SectionCategory) >= 35


# ---------------------------------------------------------------------------
# Dataclass serialisation
# ---------------------------------------------------------------------------

class TestSectionSpan:
    """Tests for SectionSpan dataclass."""

    def test_span_property(self) -> None:
        s = SectionSpan(
            header="CHIEF COMPLAINT",
            header_normalised="chief complaint",
            category=SectionCategory.CHIEF_COMPLAINT,
            header_start=0,
            header_end=18,
            body_end=50,
            confidence=1.0,
        )
        assert s.span == (0, 50)

    def test_to_dict(self) -> None:
        s = SectionSpan(
            header="HPI",
            header_normalised="hpi",
            category=SectionCategory.HISTORY_PRESENT_ILLNESS,
            header_start=10,
            header_end=15,
            body_end=100,
            confidence=0.85,
        )
        d = s.to_dict()
        assert d["header"] == "HPI"
        assert d["category"] == "history_of_present_illness"
        assert d["header_start"] == 10
        assert d["confidence"] == 0.85

    def test_frozen(self) -> None:
        s = SectionSpan(
            header="X", header_normalised="x",
            category=SectionCategory.UNKNOWN,
            header_start=0, header_end=1, body_end=10,
            confidence=0.5,
        )
        with pytest.raises(AttributeError):
            s.header = "Y"  # type: ignore[misc]


class TestSectionParseResult:
    """Tests for SectionParseResult dataclass."""

    def test_empty_result(self) -> None:
        r = SectionParseResult()
        d = r.to_dict()
        assert d["section_count"] == 0
        assert d["sections"] == []

    def test_to_dict_with_sections(self) -> None:
        s = SectionSpan(
            header="PLAN", header_normalised="plan",
            category=SectionCategory.PLAN,
            header_start=0, header_end=6, body_end=50,
            confidence=1.0,
        )
        r = SectionParseResult(
            sections=[s],
            preamble_end=0,
            categories_found={SectionCategory.PLAN},
            text_length=50,
        )
        d = r.to_dict()
        assert d["section_count"] == 1
        assert "plan" in d["categories_found"]


# ---------------------------------------------------------------------------
# Header normalisation
# ---------------------------------------------------------------------------

class TestNormalisation:
    """Tests for header normalisation and category lookup."""

    def test_normalise_strips_colon(self) -> None:
        assert _normalise_header("CHIEF COMPLAINT:") == "chief complaint"

    def test_normalise_lowercase(self) -> None:
        assert _normalise_header("History of Present Illness") == "history of present illness"

    def test_normalise_strips_asterisks(self) -> None:
        assert _normalise_header("**Medications**") == "medications"

    def test_category_mapping_count(self) -> None:
        """At least 60 header→category mappings in the dictionary."""
        assert len(_HEADER_TO_CATEGORY) >= 60


# ---------------------------------------------------------------------------
# Colon-terminated header detection
# ---------------------------------------------------------------------------

class TestColonHeaders:
    """Tests for HEADER: pattern detection."""

    def test_allcaps_colon(self) -> None:
        text = "CHIEF COMPLAINT:\nChest pain"
        parser = ClinicalSectionParser()
        result = parser.parse(text)
        assert len(result.sections) >= 1
        assert result.sections[0].category == SectionCategory.CHIEF_COMPLAINT

    def test_title_case_colon(self) -> None:
        text = "History of Present Illness:\nPatient presents with chest pain"
        parser = ClinicalSectionParser()
        result = parser.parse(text)
        assert any(s.category == SectionCategory.HISTORY_PRESENT_ILLNESS for s in result.sections)

    def test_bold_header(self) -> None:
        text = "**Medications**\nLisinopril 10mg daily"
        parser = ClinicalSectionParser()
        result = parser.parse(text)
        assert any(s.category == SectionCategory.MEDICATIONS for s in result.sections)

    def test_confidence_is_1_for_colon(self) -> None:
        text = "VITAL SIGNS:\nBP 120/80"
        parser = ClinicalSectionParser()
        result = parser.parse(text)
        assert result.sections[0].confidence == 1.0


# ---------------------------------------------------------------------------
# ALL-CAPS line detection
# ---------------------------------------------------------------------------

class TestAllCapsHeaders:
    """Tests for short ALL-CAPS line detection."""

    def test_allcaps_no_colon(self) -> None:
        text = "ASSESSMENT\nHypertension, uncontrolled"
        parser = ClinicalSectionParser()
        result = parser.parse(text)
        assert len(result.sections) >= 1
        assert result.sections[0].category == SectionCategory.ASSESSMENT

    def test_confidence_is_085(self) -> None:
        text = "ASSESSMENT\nHypertension"
        parser = ClinicalSectionParser()
        result = parser.parse(text)
        assert result.sections[0].confidence == 0.85


# ---------------------------------------------------------------------------
# Multiple sections
# ---------------------------------------------------------------------------

class TestMultipleSections:
    """Tests for parsing documents with multiple sections."""

    MULTI_SECTION_NOTE = (
        "CHIEF COMPLAINT:\n"
        "Chest pain\n\n"
        "HPI:\n"
        "55yo M presents with acute onset chest pain.\n\n"
        "PAST MEDICAL HISTORY:\n"
        "Hypertension, DM2\n\n"
        "MEDICATIONS:\n"
        "Lisinopril 10mg daily\n\n"
        "ALLERGIES:\n"
        "Penicillin - anaphylaxis\n\n"
        "VITAL SIGNS:\n"
        "BP 120/80, HR 78\n\n"
        "PHYSICAL EXAM:\n"
        "Alert and oriented\n\n"
        "ASSESSMENT AND PLAN:\n"
        "Acute chest pain, ACS workup"
    )

    def test_detects_all_sections(self) -> None:
        parser = ClinicalSectionParser()
        result = parser.parse(self.MULTI_SECTION_NOTE)
        assert len(result.sections) == 8

    def test_categories_found(self) -> None:
        parser = ClinicalSectionParser()
        result = parser.parse(self.MULTI_SECTION_NOTE)
        expected = {
            SectionCategory.CHIEF_COMPLAINT,
            SectionCategory.HISTORY_PRESENT_ILLNESS,
            SectionCategory.PAST_MEDICAL_HISTORY,
            SectionCategory.MEDICATIONS,
            SectionCategory.ALLERGIES,
            SectionCategory.VITAL_SIGNS,
            SectionCategory.PHYSICAL_EXAM,
            SectionCategory.ASSESSMENT_AND_PLAN,
        }
        assert result.categories_found == expected

    def test_sections_ordered_by_position(self) -> None:
        parser = ClinicalSectionParser()
        result = parser.parse(self.MULTI_SECTION_NOTE)
        starts = [s.header_start for s in result.sections]
        assert starts == sorted(starts)

    def test_section_spans_cover_body(self) -> None:
        parser = ClinicalSectionParser()
        result = parser.parse(self.MULTI_SECTION_NOTE)
        for i, section in enumerate(result.sections):
            assert section.header_end <= section.body_end
            if i + 1 < len(result.sections):
                # Body ends where next header starts.
                assert section.body_end == result.sections[i + 1].header_start

    def test_last_section_extends_to_end(self) -> None:
        parser = ClinicalSectionParser()
        result = parser.parse(self.MULTI_SECTION_NOTE)
        assert result.sections[-1].body_end == len(self.MULTI_SECTION_NOTE)


# ---------------------------------------------------------------------------
# Position queries
# ---------------------------------------------------------------------------

class TestPositionQueries:
    """Tests for in_section() and get_section_at()."""

    NOTE = "MEDICATIONS:\nLisinopril 10mg\n\nALLERGIES:\nPenicillin - anaphylaxis"

    def test_in_section_medications(self) -> None:
        parser = ClinicalSectionParser()
        # "Lisinopril" is inside MEDICATIONS section.
        pos = self.NOTE.index("Lisinopril")
        assert parser.in_section(
            self.NOTE, pos, {SectionCategory.MEDICATIONS}
        )

    def test_not_in_section(self) -> None:
        parser = ClinicalSectionParser()
        pos = self.NOTE.index("Lisinopril")
        assert not parser.in_section(
            self.NOTE, pos, {SectionCategory.ALLERGIES}
        )

    def test_get_section_at(self) -> None:
        parser = ClinicalSectionParser()
        pos = self.NOTE.index("Penicillin")
        section = parser.get_section_at(self.NOTE, pos)
        assert section is not None
        assert section.category == SectionCategory.ALLERGIES

    def test_get_section_at_preamble(self) -> None:
        text = "Some preamble text\n\nCHIEF COMPLAINT:\nChest pain"
        parser = ClinicalSectionParser()
        section = parser.get_section_at(text, 5)
        assert section is None

    def test_precomputed_result(self) -> None:
        parser = ClinicalSectionParser()
        result = parser.parse(self.NOTE)
        pos = self.NOTE.index("Penicillin")
        assert parser.in_section(
            self.NOTE, pos, {SectionCategory.ALLERGIES}, _result=result
        )


# ---------------------------------------------------------------------------
# Category descriptions
# ---------------------------------------------------------------------------

class TestCategoryDescriptions:
    """Tests for get_category_descriptions()."""

    def test_all_categories_have_descriptions(self) -> None:
        descriptions = ClinicalSectionParser.get_category_descriptions()
        for cat in SectionCategory:
            assert cat in descriptions, f"Missing description for {cat}"

    def test_descriptions_are_non_empty(self) -> None:
        descriptions = ClinicalSectionParser.get_category_descriptions()
        for cat, desc in descriptions.items():
            assert len(desc) > 10, f"Description too short for {cat}"


# ---------------------------------------------------------------------------
# min_confidence filtering
# ---------------------------------------------------------------------------

class TestMinConfidence:
    """Tests for min_confidence parameter."""

    def test_filters_low_confidence_sections(self) -> None:
        # ALL-CAPS without colon = 0.85 confidence.
        text = "ASSESSMENT\nHypertension"
        parser = ClinicalSectionParser(min_confidence=0.90)
        result = parser.parse(text)
        assert len(result.sections) == 0

    def test_includes_high_confidence(self) -> None:
        text = "ASSESSMENT:\nHypertension"  # Colon = 1.0
        parser = ClinicalSectionParser(min_confidence=0.90)
        result = parser.parse(text)
        assert len(result.sections) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for the section parser."""

    def test_empty_string(self) -> None:
        parser = ClinicalSectionParser()
        result = parser.parse("")
        assert len(result.sections) == 0
        assert result.text_length == 0

    def test_none_like_empty(self) -> None:
        parser = ClinicalSectionParser()
        result = parser.parse("   ")
        assert len(result.sections) == 0

    def test_no_sections(self) -> None:
        parser = ClinicalSectionParser()
        result = parser.parse("Patient presents with chest pain and shortness of breath.")
        assert len(result.sections) == 0
        assert result.preamble_end == len("Patient presents with chest pain and shortness of breath.")

    def test_preamble_before_first_section(self) -> None:
        text = "Patient info here\n\nCHIEF COMPLAINT:\nChest pain"
        parser = ClinicalSectionParser()
        result = parser.parse(text)
        assert result.preamble_end > 0
        assert result.preamble_end == result.sections[0].header_start

    def test_unknown_header(self) -> None:
        text = "BANANA SECTION:\nSome text"
        parser = ClinicalSectionParser()
        result = parser.parse(text)
        assert len(result.sections) >= 1
        assert result.sections[0].category == SectionCategory.UNKNOWN

    def test_abbreviation_headers(self) -> None:
        """Common abbreviations like PMH, FH, SH should resolve."""
        text = "PMH:\nHTN, DM\n\nFH:\nCAD in father\n\nSH:\nNon-smoker"
        parser = ClinicalSectionParser()
        result = parser.parse(text)
        categories = {s.category for s in result.sections}
        assert SectionCategory.PAST_MEDICAL_HISTORY in categories
        assert SectionCategory.FAMILY_HISTORY in categories
        assert SectionCategory.SOCIAL_HISTORY in categories

    def test_dental_sections(self) -> None:
        text = (
            "DENTAL HISTORY:\nLast visit 6 months ago\n\n"
            "PERIODONTAL ASSESSMENT:\nProbing depths 3-5mm\n\n"
            "ORAL EXAMINATION:\nNo lesions"
        )
        parser = ClinicalSectionParser()
        result = parser.parse(text)
        categories = {s.category for s in result.sections}
        assert SectionCategory.DENTAL_HISTORY in categories
        assert SectionCategory.PERIODONTAL_ASSESSMENT in categories
        assert SectionCategory.ORAL_EXAMINATION in categories

    def test_soap_sections(self) -> None:
        text = "SUBJECTIVE:\nPatient reports pain\n\nOBJECTIVE:\nAlert, oriented"
        parser = ClinicalSectionParser()
        result = parser.parse(text)
        categories = {s.category for s in result.sections}
        assert SectionCategory.SUBJECTIVE in categories
        assert SectionCategory.OBJECTIVE in categories


# ---------------------------------------------------------------------------
# Discharge summary sections
# ---------------------------------------------------------------------------

class TestDischargeSummary:
    """Tests for discharge-specific section detection."""

    DISCHARGE_NOTE = (
        "DISCHARGE DIAGNOSIS:\n"
        "Acute MI\n\n"
        "HOSPITAL COURSE:\n"
        "Patient admitted for ACS.\n\n"
        "DISCHARGE MEDICATIONS:\n"
        "Aspirin 81mg\n\n"
        "DISCHARGE INSTRUCTIONS:\n"
        "Follow up in 1 week\n\n"
        "FOLLOW UP:\n"
        "Cardiology in 2 weeks"
    )

    def test_discharge_sections(self) -> None:
        parser = ClinicalSectionParser()
        result = parser.parse(self.DISCHARGE_NOTE)
        categories = {s.category for s in result.sections}
        assert SectionCategory.DISCHARGE_DIAGNOSIS in categories
        assert SectionCategory.HOSPITAL_COURSE in categories
        assert SectionCategory.DISCHARGE_MEDICATIONS in categories
        assert SectionCategory.DISCHARGE_INSTRUCTIONS in categories
        assert SectionCategory.FOLLOW_UP in categories


# ---------------------------------------------------------------------------
# Realistic clinical note
# ---------------------------------------------------------------------------

class TestRealisticNote:
    """Test with a realistic H&P note."""

    HP_NOTE = (
        "CHIEF COMPLAINT:\n"
        "Shortness of breath x 2 days\n\n"
        "HISTORY OF PRESENT ILLNESS:\n"
        "72-year-old female presents with progressive dyspnea over the past "
        "48 hours. She reports associated orthopnea and paroxysmal nocturnal "
        "dyspnea. She has a history of CHF with last echo showing EF 35%.\n\n"
        "PAST MEDICAL HISTORY:\n"
        "1. CHF (EF 35%)\n"
        "2. HTN\n"
        "3. DM2\n"
        "4. CKD stage 3\n\n"
        "MEDICATIONS:\n"
        "Lisinopril 20mg daily\n"
        "Metoprolol 50mg BID\n"
        "Furosemide 40mg daily\n"
        "Metformin 1000mg BID\n\n"
        "ALLERGIES:\n"
        "PCN - anaphylaxis\n"
        "Sulfa - rash\n\n"
        "VITAL SIGNS:\n"
        "BP 158/92, HR 98, RR 24, SpO2 91% on RA, Temp 98.4°F\n\n"
        "PHYSICAL EXAM:\n"
        "General: Mild respiratory distress\n"
        "Lungs: Bilateral basilar crackles\n"
        "CV: S3 gallop, no murmurs\n"
        "Ext: 2+ pitting edema bilaterally\n\n"
        "LABS:\n"
        "BNP 1450, Cr 1.8, BUN 32\n\n"
        "ASSESSMENT AND PLAN:\n"
        "1. Acute decompensated CHF\n"
        "   - IV furosemide 40mg\n"
        "   - Strict I/O\n"
        "2. CKD stage 3 - monitor renal function\n"
        "3. DM2 - hold metformin\n"
    )

    def test_section_count(self) -> None:
        parser = ClinicalSectionParser()
        result = parser.parse(self.HP_NOTE)
        assert len(result.sections) >= 9

    def test_key_categories_found(self) -> None:
        parser = ClinicalSectionParser()
        result = parser.parse(self.HP_NOTE)
        categories = result.categories_found
        assert SectionCategory.CHIEF_COMPLAINT in categories
        assert SectionCategory.HISTORY_PRESENT_ILLNESS in categories
        assert SectionCategory.MEDICATIONS in categories
        assert SectionCategory.VITAL_SIGNS in categories
        assert SectionCategory.ASSESSMENT_AND_PLAN in categories

    def test_medication_position_query(self) -> None:
        parser = ClinicalSectionParser()
        # "Lisinopril" should be inside MEDICATIONS section.
        pos = self.HP_NOTE.index("Lisinopril")
        assert parser.in_section(
            self.HP_NOTE, pos, {SectionCategory.MEDICATIONS}
        )

    def test_vitals_position_query(self) -> None:
        parser = ClinicalSectionParser()
        pos = self.HP_NOTE.index("BP 158/92")
        assert parser.in_section(
            self.HP_NOTE, pos, {SectionCategory.VITAL_SIGNS}
        )
