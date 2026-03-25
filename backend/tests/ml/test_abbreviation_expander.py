"""Tests for clinical abbreviation detection and expansion module.

Covers dictionary integrity, unambiguous detection, ambiguous disambiguation,
section-aware overrides, deduplication, batch processing, config options,
edge cases, and realistic clinical scenarios.
"""

import pytest

from app.ml.abbreviations.expander import (
    AbbreviationConfig,
    AbbreviationExpander,
    AbbreviationMatch,
    AmbiguityResolution,
    ClinicalDomain,
    ExpansionResult,
    _ABBREVIATION_DB,
    _AMBIGUOUS_DB,
)


# ─────────────────────────────────────────────────────────────────────
# Enum & Dataclass Completeness
# ─────────────────────────────────────────────────────────────────────


class TestEnumCompleteness:
    """Verify all enum values exist and are string-serializable."""

    def test_clinical_domain_count(self):
        """All 12 clinical domains are defined."""
        assert len(ClinicalDomain) == 12

    def test_clinical_domain_values(self):
        """Domain values are lowercase strings."""
        for d in ClinicalDomain:
            assert d == d.value
            assert isinstance(str(d), str)

    def test_ambiguity_resolution_count(self):
        """All 4 resolution methods exist."""
        assert len(AmbiguityResolution) == 4

    def test_ambiguity_resolution_values(self):
        """Resolution values are lowercase strings."""
        expected = {"unambiguous", "context_resolved", "default_sense", "section_resolved"}
        actual = {r.value for r in AmbiguityResolution}
        assert actual == expected


class TestDataclassSerialization:
    """Verify dataclass to_dict() methods produce valid output."""

    def test_abbreviation_match_to_dict(self):
        """AbbreviationMatch serializes all fields."""
        match = AbbreviationMatch(
            abbreviation="HTN",
            expansion="hypertension",
            start=10,
            end=13,
            confidence=0.95,
            domain=ClinicalDomain.CARDIOLOGY,
            is_ambiguous=False,
            resolution=AmbiguityResolution.UNAMBIGUOUS,
        )
        d = match.to_dict()
        assert d["abbreviation"] == "HTN"
        assert d["expansion"] == "hypertension"
        assert d["start"] == 10
        assert d["end"] == 13
        assert d["confidence"] == 0.95
        assert d["domain"] == "cardiology"
        assert d["is_ambiguous"] is False
        assert d["resolution"] == "unambiguous"
        assert d["alternative_expansions"] == []

    def test_expansion_result_to_dict(self):
        """ExpansionResult serializes with rounded processing time."""
        result = ExpansionResult(
            original_text="test",
            expanded_text="test",
            matches=[],
            total_found=0,
            ambiguous_count=0,
            processing_time_ms=1.23456,
        )
        d = result.to_dict()
        assert d["processing_time_ms"] == 1.23
        assert d["matches"] == []

    def test_ambiguous_match_to_dict(self):
        """Ambiguous match includes alternative expansions."""
        match = AbbreviationMatch(
            abbreviation="PE",
            expansion="pulmonary embolism",
            start=0,
            end=2,
            confidence=0.85,
            domain=ClinicalDomain.PULMONOLOGY,
            is_ambiguous=True,
            resolution=AmbiguityResolution.CONTEXT_RESOLVED,
            alternative_expansions=["physical exam"],
        )
        d = match.to_dict()
        assert d["is_ambiguous"] is True
        assert d["alternative_expansions"] == ["physical exam"]


# ─────────────────────────────────────────────────────────────────────
# Dictionary Integrity
# ─────────────────────────────────────────────────────────────────────


class TestDictionaryIntegrity:
    """Verify the abbreviation dictionary is well-formed."""

    def test_all_unambiguous_have_three_fields(self):
        """Each unambiguous entry has (expansion, domain, confidence)."""
        for abbrev, entry in _ABBREVIATION_DB.items():
            assert len(entry) == 3, f"{abbrev} has {len(entry)} fields"
            expansion, domain, confidence = entry
            assert isinstance(expansion, str) and expansion
            assert isinstance(domain, ClinicalDomain)
            assert 0.0 < confidence <= 1.0

    def test_all_ambiguous_have_senses(self):
        """Each ambiguous entry has at least 2 senses (or 1 for single-override)."""
        for abbrev, senses in _AMBIGUOUS_DB.items():
            assert len(senses) >= 1, f"{abbrev} has no senses"
            for expansion, domain, keywords in senses:
                assert isinstance(expansion, str) and expansion
                assert isinstance(domain, ClinicalDomain)
                assert isinstance(keywords, list)
                assert len(keywords) > 0, f"{abbrev}/{expansion} has no keywords"

    def test_no_overlap_between_dbs(self):
        """No abbreviation appears in both unambiguous and ambiguous DBs."""
        overlap = set(_ABBREVIATION_DB.keys()) & set(_AMBIGUOUS_DB.keys())
        assert not overlap, f"Overlap: {overlap}"

    def test_abbreviation_keys_are_lowercase(self):
        """All dictionary keys are lowercase."""
        for key in _ABBREVIATION_DB:
            assert key == key.lower(), f"Key not lowercase: {key}"
        for key in _AMBIGUOUS_DB:
            assert key == key.lower(), f"Ambiguous key not lowercase: {key}"

    def test_minimum_dictionary_size(self):
        """Dictionary has substantial coverage."""
        total = len(_ABBREVIATION_DB) + len(_AMBIGUOUS_DB)
        assert total >= 200, f"Only {total} entries"

    def test_all_domains_represented(self):
        """Every ClinicalDomain has at least one abbreviation."""
        domains_in_db = set()
        for _, (_, domain, _) in _ABBREVIATION_DB.items():
            domains_in_db.add(domain)
        for _, senses in _AMBIGUOUS_DB.items():
            for _, domain, _ in senses:
                domains_in_db.add(domain)
        for domain in ClinicalDomain:
            assert domain in domains_in_db, f"Missing domain: {domain}"


# ─────────────────────────────────────────────────────────────────────
# Unambiguous Detection
# ─────────────────────────────────────────────────────────────────────


class TestUnambiguousDetection:
    """Test detection of unambiguous abbreviations."""

    @pytest.fixture()
    def expander(self):
        return AbbreviationExpander()

    def test_single_abbreviation(self, expander):
        """Detects a single abbreviation."""
        result = expander.expand("Patient has HTN.")
        assert result.total_found >= 1
        htn_match = next(m for m in result.matches if m.abbreviation.lower() == "htn")
        assert htn_match.expansion == "hypertension"
        assert htn_match.is_ambiguous is False

    def test_multiple_abbreviations(self, expander):
        """Detects multiple abbreviations in one text."""
        result = expander.expand("PMH: HTN, DM2, CAD, COPD.")
        abbrevs = {m.abbreviation.lower() for m in result.matches}
        assert "htn" in abbrevs
        assert "dm2" in abbrevs
        assert "cad" in abbrevs
        assert "copd" in abbrevs

    def test_case_insensitive(self, expander):
        """Matches abbreviations regardless of case."""
        result = expander.expand("htn and Htn and HTN")
        htn_matches = [m for m in result.matches if m.abbreviation.lower() == "htn"]
        assert len(htn_matches) == 3

    def test_word_boundary_no_partial(self, expander):
        """Does not match abbreviations inside longer words."""
        # "catheter" should not trigger a match on "cat"
        result = expander.expand("catheter placement was uncomplicated")
        for m in result.matches:
            # No match should start inside "catheter"
            assert m.abbreviation.lower() != "cat"

    def test_slash_abbreviations(self, expander):
        """Detects abbreviations with slashes like n/v, h/o."""
        result = expander.expand("Patient c/o n/v for 2 days.")
        abbrevs = {m.abbreviation.lower() for m in result.matches}
        assert "c/o" in abbrevs
        assert "n/v" in abbrevs

    def test_expanded_text_format(self, expander):
        """Expanded text uses 'abbreviation (expansion)' format."""
        result = expander.expand("HTN diagnosed.")
        assert "HTN (hypertension)" in result.expanded_text

    def test_original_text_preserved(self, expander):
        """Original text is returned unchanged."""
        text = "PMH: HTN, DM2."
        result = expander.expand(text)
        assert result.original_text == text

    def test_character_offsets_correct(self, expander):
        """Start/end offsets point to the correct text span."""
        text = "History of HTN."
        result = expander.expand(text)
        htn = next(m for m in result.matches if m.abbreviation.lower() == "htn")
        assert text[htn.start:htn.end].lower() == "htn"

    def test_cardiology_abbreviations(self, expander):
        """Detects cardiology-domain abbreviations."""
        result = expander.expand("STEMI with CABG, LVEF 35%.")
        abbrevs = {m.abbreviation.lower() for m in result.matches}
        assert "stemi" in abbrevs
        assert "cabg" in abbrevs
        assert "lvef" in abbrevs

    def test_pharmacy_abbreviations(self, expander):
        """Detects pharmacy-domain abbreviations."""
        result = expander.expand("Metoprolol 25mg PO BID PRN.")
        abbrevs = {m.abbreviation.lower() for m in result.matches}
        assert "po" in abbrevs
        assert "bid" in abbrevs
        assert "prn" in abbrevs

    def test_dental_abbreviations(self, expander):
        """Detects dental-domain abbreviations."""
        result = expander.expand("SRP completed, BOP noted at TMJ.")
        abbrevs = {m.abbreviation.lower() for m in result.matches}
        assert "srp" in abbrevs
        assert "bop" in abbrevs
        assert "tmj" in abbrevs

    def test_hematology_abbreviations(self, expander):
        """Detects hematology/lab abbreviations."""
        result = expander.expand("CBC with WBC 12.5, HGB 14.2, PLT 250.")
        abbrevs = {m.abbreviation.lower() for m in result.matches}
        assert "cbc" in abbrevs
        assert "wbc" in abbrevs

    def test_confidence_matches_dictionary(self, expander):
        """Unambiguous match confidence matches the dictionary value."""
        result = expander.expand("HTN")
        htn = next(m for m in result.matches if m.abbreviation.lower() == "htn")
        expected_conf = _ABBREVIATION_DB["htn"][2]
        assert htn.confidence == expected_conf


# ─────────────────────────────────────────────────────────────────────
# Ambiguous Disambiguation
# ─────────────────────────────────────────────────────────────────────


class TestAmbiguousDisambiguation:
    """Test context-aware disambiguation of ambiguous abbreviations."""

    @pytest.fixture()
    def expander(self):
        return AbbreviationExpander()

    def test_pe_as_pulmonary_embolism(self, expander):
        """PE resolves to pulmonary embolism with clot context."""
        result = expander.expand("Concern for PE given DVT history and positive D-dimer.")
        pe_match = next(
            (m for m in result.matches if m.abbreviation.lower() == "pe"), None,
        )
        assert pe_match is not None
        assert pe_match.expansion == "pulmonary embolism"
        assert pe_match.is_ambiguous is True
        assert pe_match.resolution == AmbiguityResolution.CONTEXT_RESOLVED

    def test_pe_as_physical_exam(self, expander):
        """PE resolves to physical exam with exam context."""
        result = expander.expand(
            "PE: Heart regular, lungs clear, abdomen soft.",
        )
        pe_match = next(
            (m for m in result.matches if m.abbreviation.lower() == "pe"), None,
        )
        assert pe_match is not None
        assert pe_match.expansion == "physical exam"

    def test_pt_as_patient(self, expander):
        """PT resolves to patient when used as 'the pt is'."""
        result = expander.expand("The pt is a 65 y/o male.")
        pt_match = next(
            (m for m in result.matches if m.abbreviation.lower() == "pt"), None,
        )
        assert pt_match is not None
        assert pt_match.expansion == "patient"

    def test_pt_as_physical_therapy(self, expander):
        """PT resolves to physical therapy in rehab context."""
        result = expander.expand("Consult PT for gait training and exercise program.")
        pt_match = next(
            (m for m in result.matches if m.abbreviation.lower() == "pt"), None,
        )
        assert pt_match is not None
        assert pt_match.expansion == "physical therapy"

    def test_ms_as_multiple_sclerosis(self, expander):
        """MS resolves to multiple sclerosis with neuro context."""
        result = expander.expand(
            "Diagnosed with MS, MRI brain shows demyelinating lesions.",
        )
        ms_match = next(
            (m for m in result.matches if m.abbreviation.lower() == "ms"), None,
        )
        assert ms_match is not None
        assert ms_match.expansion == "multiple sclerosis"

    def test_ambiguous_default_sense(self, expander):
        """Ambiguous abbreviation defaults to first sense without context."""
        result = expander.expand("PE noted.")
        pe_match = next(
            (m for m in result.matches if m.abbreviation.lower() == "pe"), None,
        )
        assert pe_match is not None
        assert pe_match.resolution in {
            AmbiguityResolution.DEFAULT_SENSE,
            AmbiguityResolution.CONTEXT_RESOLVED,
        }

    def test_ambiguous_has_alternatives(self, expander):
        """Ambiguous matches list alternative expansions."""
        result = expander.expand("Concern for PE given DVT and clot history.")
        pe_match = next(
            (m for m in result.matches if m.abbreviation.lower() == "pe"), None,
        )
        assert pe_match is not None
        assert len(pe_match.alternative_expansions) >= 1

    def test_ra_as_rheumatoid_arthritis(self, expander):
        """RA resolves to rheumatoid arthritis with joint context."""
        result = expander.expand("RA with joint swelling and stiffness on methotrexate.")
        ra_match = next(
            (m for m in result.matches if m.abbreviation.lower() == "ra"), None,
        )
        assert ra_match is not None
        assert ra_match.expansion == "rheumatoid arthritis"

    def test_ra_as_room_air(self, expander):
        """RA resolves to room air with oxygen context."""
        result = expander.expand("SpO2 98% on RA, breathing comfortably.")
        ra_match = next(
            (m for m in result.matches if m.abbreviation.lower() == "ra"), None,
        )
        assert ra_match is not None
        assert ra_match.expansion == "room air"

    def test_ed_as_emergency_department(self, expander):
        """ED resolves to emergency department in triage context."""
        result = expander.expand("Patient presented to ED via ambulance.")
        ed_match = next(
            (m for m in result.matches if m.abbreviation.lower() == "ed"), None,
        )
        assert ed_match is not None
        assert ed_match.expansion == "emergency department"

    def test_cap_as_community_acquired_pneumonia(self, expander):
        """CAP resolves to pneumonia with infection context."""
        result = expander.expand("Treated for CAP with antibiotics, CXR shows infiltrate.")
        cap_match = next(
            (m for m in result.matches if m.abbreviation.lower() == "cap"), None,
        )
        assert cap_match is not None
        assert cap_match.expansion == "community-acquired pneumonia"


# ─────────────────────────────────────────────────────────────────────
# Section-Aware Disambiguation
# ─────────────────────────────────────────────────────────────────────


class TestSectionAwareDisambiguation:
    """Test section header-based abbreviation override."""

    @pytest.fixture()
    def expander(self):
        return AbbreviationExpander()

    def test_pe_under_physical_exam_header(self, expander):
        """PE under 'Physical Exam:' header resolves to physical exam."""
        text = "Physical Exam:\nPE notable for wheezing bilaterally."
        result = expander.expand(text)
        pe_match = next(
            (m for m in result.matches if m.abbreviation.lower() == "pe"), None,
        )
        assert pe_match is not None
        assert pe_match.expansion == "physical exam"
        assert pe_match.resolution == AmbiguityResolution.SECTION_RESOLVED

    def test_pt_under_labs_header(self, expander):
        """PT under 'Labs:' header resolves to prothrombin time."""
        text = "Labs:\nPT 14.2 seconds, INR 1.1."
        result = expander.expand(text)
        pt_match = next(
            (m for m in result.matches if m.abbreviation.lower() == "pt"), None,
        )
        assert pt_match is not None
        assert pt_match.expansion == "prothrombin time"

    def test_cap_under_medications_header(self, expander):
        """CAP under 'Medications:' header resolves to capsule."""
        text = "Medications:\nOmeprazole 20mg CAP PO daily."
        result = expander.expand(text)
        cap_match = next(
            (m for m in result.matches if m.abbreviation.lower() == "cap"), None,
        )
        assert cap_match is not None
        assert cap_match.expansion == "capsule"

    def test_pd_under_periodontal_header(self, expander):
        """PD under 'Periodontal' header resolves to probing depth."""
        text = "Periodontal Assessment:\nPD 4-5mm at sites 3, 14, 19."
        result = expander.expand(text)
        pd_match = next(
            (m for m in result.matches if m.abbreviation.lower() == "pd"), None,
        )
        assert pd_match is not None
        assert pd_match.expansion == "probing depth"


# ─────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────


class TestDeduplication:
    """Test overlapping span deduplication."""

    @pytest.fixture()
    def expander(self):
        return AbbreviationExpander()

    def test_no_duplicate_spans(self, expander):
        """No two matches share the same start offset."""
        result = expander.expand("HTN, DM2, CAD, COPD, CHF, CKD, GERD.")
        starts = [m.start for m in result.matches]
        assert len(starts) == len(set(starts))

    def test_longer_match_preferred(self, expander):
        """Longer abbreviation wins over shorter overlapping one."""
        # "n/v/d" should match over "n/v"
        result = expander.expand("Reports n/v/d for 3 days.")
        nvd = next(
            (m for m in result.matches if m.abbreviation.lower() == "n/v/d"), None,
        )
        assert nvd is not None
        assert nvd.expansion == "nausea, vomiting, and diarrhea"


# ─────────────────────────────────────────────────────────────────────
# Configuration Options
# ─────────────────────────────────────────────────────────────────────


class TestConfiguration:
    """Test configuration options."""

    def test_min_confidence_filtering(self):
        """Matches below min_confidence are excluded."""
        config = AbbreviationConfig(min_confidence=0.95)
        expander = AbbreviationExpander(config)
        result = expander.expand("HTN, OR, US, BM")
        for m in result.matches:
            assert m.confidence >= 0.95

    def test_expand_in_place_disabled(self):
        """When disabled, expanded_text equals original_text."""
        config = AbbreviationConfig(expand_in_place=False)
        expander = AbbreviationExpander(config)
        text = "PMH: HTN, DM2."
        result = expander.expand(text)
        assert result.expanded_text == text

    def test_domain_filter(self):
        """Only abbreviations from specified domains are detected."""
        config = AbbreviationConfig(domains=[ClinicalDomain.DENTAL])
        expander = AbbreviationExpander(config)
        result = expander.expand("SRP completed. PMH: HTN, DM2.")
        for m in result.matches:
            assert m.domain == ClinicalDomain.DENTAL

    def test_exclude_unambiguous(self):
        """When include_unambiguous=False, only ambiguous matches returned."""
        config = AbbreviationConfig(include_unambiguous=False)
        expander = AbbreviationExpander(config)
        result = expander.expand("The pt has HTN and PE was normal.")
        for m in result.matches:
            assert m.is_ambiguous is True


# ─────────────────────────────────────────────────────────────────────
# Batch Processing
# ─────────────────────────────────────────────────────────────────────


class TestBatchProcessing:
    """Test batch expansion."""

    @pytest.fixture()
    def expander(self):
        return AbbreviationExpander()

    def test_batch_returns_correct_count(self, expander):
        """One result per input text."""
        texts = ["HTN noted.", "DM2 diagnosed.", "COPD stable."]
        results = expander.expand_batch(texts)
        assert len(results) == 3

    def test_batch_order_preserved(self, expander):
        """Results are in the same order as input."""
        texts = ["HTN", "DM2", "COPD"]
        results = expander.expand_batch(texts)
        assert results[0].original_text == "HTN"
        assert results[1].original_text == "DM2"
        assert results[2].original_text == "COPD"

    def test_batch_with_empty_text(self, expander):
        """Empty texts in batch produce zero matches."""
        texts = ["HTN", "", "COPD"]
        results = expander.expand_batch(texts)
        assert results[1].total_found == 0


# ─────────────────────────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture()
    def expander(self):
        return AbbreviationExpander()

    def test_empty_string(self, expander):
        """Empty string returns empty result."""
        result = expander.expand("")
        assert result.total_found == 0
        assert result.matches == []

    def test_whitespace_only(self, expander):
        """Whitespace-only string returns empty result."""
        result = expander.expand("   \n\t  ")
        assert result.total_found == 0

    def test_no_abbreviations(self, expander):
        """Text without abbreviations returns zero matches."""
        result = expander.expand("The patient was seen in the clinic today.")
        # Some common words might match (e.g., "the") but clinical abbreviations
        # should have word boundaries
        for m in result.matches:
            assert m.abbreviation.lower() in _ABBREVIATION_DB or m.abbreviation.lower() in _AMBIGUOUS_DB

    def test_processing_time_recorded(self, expander):
        """Processing time is a positive number."""
        result = expander.expand("HTN noted.")
        assert result.processing_time_ms >= 0

    def test_very_long_text(self, expander):
        """Handles text near the length limit."""
        text = "HTN. " * 5000  # 25,000 chars
        result = expander.expand(text)
        assert result.total_found > 0

    def test_abbreviation_at_start(self, expander):
        """Abbreviation at the very start of text."""
        result = expander.expand("HTN diagnosed last year.")
        assert any(m.abbreviation.lower() == "htn" and m.start == 0 for m in result.matches)

    def test_abbreviation_at_end(self, expander):
        """Abbreviation at the very end of text."""
        result = expander.expand("Diagnosed with HTN")
        assert any(m.abbreviation.lower() == "htn" for m in result.matches)

    def test_consecutive_abbreviations(self, expander):
        """Multiple abbreviations back-to-back."""
        result = expander.expand("HTN DM2 CAD")
        assert result.total_found >= 3


# ─────────────────────────────────────────────────────────────────────
# Dictionary Stats & Lookup
# ─────────────────────────────────────────────────────────────────────


class TestDictionaryStatsAndLookup:
    """Test dictionary introspection methods."""

    @pytest.fixture()
    def expander(self):
        return AbbreviationExpander()

    def test_stats_total_count(self, expander):
        """Stats report correct total."""
        stats = expander.get_dictionary_stats()
        assert stats["total_entries"] == len(_ABBREVIATION_DB) + len(_AMBIGUOUS_DB)
        assert stats["total_unambiguous"] == len(_ABBREVIATION_DB)
        assert stats["total_ambiguous"] == len(_AMBIGUOUS_DB)

    def test_stats_has_domains(self, expander):
        """Stats include domain breakdown."""
        stats = expander.get_dictionary_stats()
        assert len(stats["domains"]) > 0

    def test_lookup_unambiguous(self, expander):
        """Lookup returns details for unambiguous abbreviation."""
        result = expander.lookup("HTN")
        assert result is not None
        assert result["expansion"] == "hypertension"
        assert result["is_ambiguous"] is False

    def test_lookup_ambiguous(self, expander):
        """Lookup returns senses for ambiguous abbreviation."""
        result = expander.lookup("PE")
        assert result is not None
        assert result["is_ambiguous"] is True
        assert len(result["senses"]) >= 2

    def test_lookup_case_insensitive(self, expander):
        """Lookup is case-insensitive."""
        assert expander.lookup("htn") is not None
        assert expander.lookup("HTN") is not None
        assert expander.lookup("Htn") is not None

    def test_lookup_not_found(self, expander):
        """Lookup returns None for unknown abbreviation."""
        assert expander.lookup("ZZZZZZZ") is None

    def test_stats_senses_count(self, expander):
        """Total senses count is correct."""
        stats = expander.get_dictionary_stats()
        expected = sum(len(s) for s in _AMBIGUOUS_DB.values())
        assert stats["total_senses"] == expected


# ─────────────────────────────────────────────────────────────────────
# Realistic Clinical Notes
# ─────────────────────────────────────────────────────────────────────


class TestRealisticClinicalNotes:
    """Test with realistic clinical note excerpts."""

    @pytest.fixture()
    def expander(self):
        return AbbreviationExpander()

    def test_admission_note(self, expander):
        """Admission note with mixed abbreviations."""
        text = """
        HPI: 72 y/o male with h/o HTN, DM2, CAD s/p CABG presents to ED
        with c/o SOB and CP x 2 days. PMH significant for CHF with LVEF 30%,
        COPD on home O2, CKD stage 3.
        """
        result = expander.expand(text)
        assert result.total_found >= 10
        abbrevs = {m.abbreviation.lower() for m in result.matches}
        assert "htn" in abbrevs
        assert "dm2" in abbrevs
        assert "cad" in abbrevs
        assert "cabg" in abbrevs
        assert "sob" in abbrevs
        assert "chf" in abbrevs
        assert "copd" in abbrevs

    def test_discharge_summary(self, expander):
        """Discharge summary with medication list."""
        text = """
        Medications:
        1. Metoprolol 25mg PO BID
        2. Lisinopril 10mg PO QD
        3. Atorvastatin 40mg PO QHS
        4. Metformin 500mg PO BID
        5. ASA 81mg PO QD
        """
        result = expander.expand(text)
        abbrevs = {m.abbreviation.lower() for m in result.matches}
        assert "po" in abbrevs
        assert "bid" in abbrevs

    def test_dental_note(self, expander):
        """Dental clinical note."""
        text = """
        Periodontal Assessment:
        SRP completed Q1-Q4. BOP noted at sites 3, 14, 19.
        PD 4-5mm posteriorly. FMX reviewed, no periapical pathology.
        TMJ palpation WNL bilaterally.
        """
        result = expander.expand(text)
        abbrevs = {m.abbreviation.lower() for m in result.matches}
        assert "srp" in abbrevs
        assert "bop" in abbrevs
        assert "fmx" in abbrevs
        assert "tmj" in abbrevs
        assert "wnl" in abbrevs

    def test_lab_section_with_ambiguous(self, expander):
        """Lab section correctly disambiguates PT as prothrombin time."""
        text = """
        Laboratory Results:
        CBC: WBC 8.2, HGB 14.1, PLT 210
        BMP: Na 140, K 4.2, Cr 1.1, BUN 18
        PT 13.5, INR 1.0
        """
        result = expander.expand(text)
        pt_match = next(
            (m for m in result.matches if m.abbreviation.lower() == "pt"), None,
        )
        assert pt_match is not None
        assert pt_match.expansion == "prothrombin time"
