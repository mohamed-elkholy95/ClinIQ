"""Tests for medical query expansion module.

Validates abbreviation expansion, synonym lookup, bidirectional mapping,
multi-word phrase handling, expansion caps, and edge cases.
"""

from __future__ import annotations

import pytest

from app.ml.search.query_expansion import (
    ExpandedQuery,
    MedicalQueryExpander,
    _ABBREVIATION_TO_FULL,
    _FULL_TO_ABBREVIATION,
    _SYNONYM_LOOKUP,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def expander() -> MedicalQueryExpander:
    """Default query expander."""
    return MedicalQueryExpander(max_expansions=8)


@pytest.fixture
def small_expander() -> MedicalQueryExpander:
    """Expander with tight expansion cap."""
    return MedicalQueryExpander(max_expansions=2)


# ---------------------------------------------------------------------------
# Abbreviation dictionary tests
# ---------------------------------------------------------------------------


class TestAbbreviationDictionary:
    """Validate the static abbreviation mappings."""

    def test_common_cardiovascular_abbreviations(self) -> None:
        """HTN, MI, CHF should all have full forms."""
        assert _ABBREVIATION_TO_FULL["htn"] == "hypertension"
        assert _ABBREVIATION_TO_FULL["mi"] == "myocardial infarction"
        assert _ABBREVIATION_TO_FULL["chf"] == "congestive heart failure"

    def test_common_endocrine_abbreviations(self) -> None:
        """DM, T2DM should map correctly."""
        assert _ABBREVIATION_TO_FULL["dm"] == "diabetes mellitus"
        assert _ABBREVIATION_TO_FULL["t2dm"] == "type 2 diabetes mellitus"

    def test_dental_abbreviations(self) -> None:
        """Dental abbreviations should be present."""
        assert _ABBREVIATION_TO_FULL["perio"] == "periodontal"
        assert _ABBREVIATION_TO_FULL["srp"] == "scaling and root planing"
        assert _ABBREVIATION_TO_FULL["rct"] == "root canal treatment"
        assert _ABBREVIATION_TO_FULL["bop"] == "bleeding on probing"

    def test_reverse_mapping_consistency(self) -> None:
        """Every abbreviation should have a reverse lookup entry."""
        for abbr, full in _ABBREVIATION_TO_FULL.items():
            reverse = _FULL_TO_ABBREVIATION.get(full.lower())
            assert reverse is not None, f"No reverse for {full}"

    def test_lab_abbreviations(self) -> None:
        """Lab/diagnostic abbreviations."""
        assert _ABBREVIATION_TO_FULL["cbc"] == "complete blood count"
        assert _ABBREVIATION_TO_FULL["mri"] == "magnetic resonance imaging"
        assert _ABBREVIATION_TO_FULL["ekg"] == "electrocardiogram"

    def test_ekg_ecg_both_map_to_electrocardiogram(self) -> None:
        """Both EKG and ECG are valid abbreviations."""
        assert _ABBREVIATION_TO_FULL["ekg"] == "electrocardiogram"
        assert _ABBREVIATION_TO_FULL["ecg"] == "electrocardiogram"


# ---------------------------------------------------------------------------
# Synonym lookup tests
# ---------------------------------------------------------------------------


class TestSynonymLookup:
    """Validate the synonym group lookups."""

    def test_hypertension_synonyms(self) -> None:
        """Hypertension should have high blood pressure as a synonym."""
        syns = _SYNONYM_LOOKUP.get("hypertension", set())
        assert "high blood pressure" in syns or "htn" in syns

    def test_diabetes_synonyms(self) -> None:
        """Diabetes should link to DM and related terms."""
        syns = _SYNONYM_LOOKUP.get("diabetes mellitus", set())
        assert "diabetes" in syns or "dm" in syns

    def test_spelling_variants(self) -> None:
        """British/American spelling variants should be synonyms."""
        assert "anaemia" in _SYNONYM_LOOKUP.get("anemia", set())
        assert "anemia" in _SYNONYM_LOOKUP.get("anaemia", set())
        assert "haemorrhage" in _SYNONYM_LOOKUP.get("hemorrhage", set())

    def test_dental_synonyms(self) -> None:
        """Dental synonym groups should be populated."""
        syns = _SYNONYM_LOOKUP.get("dental caries", set())
        assert "tooth decay" in syns or "cavity" in syns

    def test_symptom_synonyms(self) -> None:
        """Symptom groups should link common terms."""
        sob_syns = _SYNONYM_LOOKUP.get("shortness of breath", set())
        assert "dyspnea" in sob_syns or "sob" in sob_syns

    def test_bidirectional_synonyms(self) -> None:
        """If A is synonym of B, then B should be synonym of A."""
        for term, synonyms in _SYNONYM_LOOKUP.items():
            for syn in synonyms:
                reverse = _SYNONYM_LOOKUP.get(syn, set())
                assert term in reverse, (
                    f"'{term}' is in synonyms of '{syn}' but not vice versa"
                )


# ---------------------------------------------------------------------------
# Expander — abbreviation expansion tests
# ---------------------------------------------------------------------------


class TestAbbreviationExpansion:
    """Test abbreviation → full-form expansion."""

    def test_single_abbreviation_expanded(self, expander: MedicalQueryExpander) -> None:
        """A query with 'htn' should expand to include 'hypertension'."""
        result = expander.expand("patient with htn")
        assert "hypertension" in result.expanded_terms

    def test_multiple_abbreviations(self, expander: MedicalQueryExpander) -> None:
        """Multiple abbreviations in one query should all expand."""
        result = expander.expand("htn dm copd")
        terms = set(result.expanded_terms)
        assert "hypertension" in terms
        assert "diabetes mellitus" in terms
        assert "chronic obstructive pulmonary disease" in terms

    def test_abbreviation_already_present_not_duplicated(
        self, expander: MedicalQueryExpander,
    ) -> None:
        """If the full form is already in the query, don't add it again."""
        result = expander.expand("patient with hypertension")
        assert "hypertension" not in result.expanded_terms

    def test_full_form_to_abbreviation(self, expander: MedicalQueryExpander) -> None:
        """Full form in query should add abbreviation."""
        result = expander.expand("diabetes")
        # "diabetes" as single token should find synonyms including "dm"
        has_dm = "dm" in result.expanded_terms
        has_diabetic = "diabetic" in result.expanded_terms
        assert has_dm or has_diabetic

    def test_case_insensitive(self, expander: MedicalQueryExpander) -> None:
        """Expansion should work regardless of case."""
        result = expander.expand("HTN and DM")
        assert result.expansion_count > 0

    def test_dental_abbreviation_expansion(
        self, expander: MedicalQueryExpander,
    ) -> None:
        """Dental abbreviations should expand."""
        result = expander.expand("srp needed")
        assert "scaling and root planing" in result.expanded_terms


# ---------------------------------------------------------------------------
# Expander — synonym expansion tests
# ---------------------------------------------------------------------------


class TestSynonymExpansion:
    """Test synonym group expansion."""

    def test_synonym_expansion(self, expander: MedicalQueryExpander) -> None:
        """Single-word synonym should expand."""
        result = expander.expand("dyspnea")
        terms = set(result.expanded_terms)
        assert "shortness of breath" in terms or "sob" in terms

    def test_spelling_variant_expansion(
        self, expander: MedicalQueryExpander,
    ) -> None:
        """British spelling should expand to American (or vice versa)."""
        result = expander.expand("anaemia symptoms")
        assert "anemia" in result.expanded_terms

    def test_synonym_not_duplicated_if_present(
        self, expander: MedicalQueryExpander,
    ) -> None:
        """Synonyms already in the query should not be added."""
        result = expander.expand("anemia anaemia")
        # Neither should appear as expanded because both are already present
        for term in result.expanded_terms:
            assert term not in ("anemia", "anaemia") or term not in "anemia anaemia"


# ---------------------------------------------------------------------------
# Expander — edge cases
# ---------------------------------------------------------------------------


class TestExpansionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_query(self, expander: MedicalQueryExpander) -> None:
        """Empty query should return empty expansion."""
        result = expander.expand("")
        assert result.expansion_count == 0
        assert result.expanded_query == ""

    def test_whitespace_only_query(self, expander: MedicalQueryExpander) -> None:
        """Whitespace-only query should return unchanged."""
        result = expander.expand("   ")
        assert result.expansion_count == 0

    def test_no_medical_terms(self, expander: MedicalQueryExpander) -> None:
        """Non-medical query should have no expansions."""
        result = expander.expand("weather today sunshine")
        assert result.expansion_count == 0

    def test_max_expansions_cap(self, small_expander: MedicalQueryExpander) -> None:
        """Expansion should not exceed max_expansions limit."""
        result = small_expander.expand("htn dm copd chf cad afib sob")
        assert result.expansion_count <= 2

    def test_expanded_query_contains_original(
        self, expander: MedicalQueryExpander,
    ) -> None:
        """Expanded query should start with the original query."""
        result = expander.expand("patient with htn")
        assert result.expanded_query.startswith("patient with htn")

    def test_expansion_sources_populated(
        self, expander: MedicalQueryExpander,
    ) -> None:
        """Each expanded term should have a source explanation."""
        result = expander.expand("htn treatment")
        for term in result.expanded_terms:
            assert term in result.expansion_sources

    def test_expanded_query_dataclass_fields(
        self, expander: MedicalQueryExpander,
    ) -> None:
        """ExpandedQuery should have all expected fields."""
        result = expander.expand("dm management")
        assert isinstance(result, ExpandedQuery)
        assert isinstance(result.original, str)
        assert isinstance(result.expanded_terms, list)
        assert isinstance(result.expansion_sources, dict)
        assert isinstance(result.expanded_query, str)
        assert isinstance(result.expansion_count, int)


# ---------------------------------------------------------------------------
# Expander — configuration tests
# ---------------------------------------------------------------------------


class TestExpanderConfiguration:
    """Test expander configuration options."""

    def test_abbreviations_disabled(self) -> None:
        """With abbreviations disabled, should not expand abbreviations directly."""
        exp = MedicalQueryExpander(include_abbreviations=False)
        result = exp.expand("htn treatment")
        # "hypertension" might still appear via synonym groups (htn is in some),
        # but direct abbreviation expansion should be skipped. Check that the
        # source is NOT "abbreviation expansion" if hypertension is present.
        for term in result.expanded_terms:
            source = result.expansion_sources.get(term, "")
            assert "abbreviation expansion" not in source

    def test_synonyms_disabled(self) -> None:
        """With synonyms disabled, should not expand them."""
        exp = MedicalQueryExpander(include_synonyms=False)
        result = exp.expand("dyspnea")
        # Should only get abbreviation expansions, not synonym groups
        for term in result.expanded_terms:
            assert term not in ("shortness of breath", "breathlessness")

    def test_both_disabled_no_expansion(self) -> None:
        """With both disabled, should produce zero expansions."""
        exp = MedicalQueryExpander(
            include_abbreviations=False, include_synonyms=False,
        )
        result = exp.expand("htn dm copd")
        assert result.expansion_count == 0

    def test_max_expansions_zero(self) -> None:
        """max_expansions=0 should produce no expansions."""
        exp = MedicalQueryExpander(max_expansions=0)
        result = exp.expand("htn dm copd")
        assert result.expansion_count == 0


# ---------------------------------------------------------------------------
# Helper method tests
# ---------------------------------------------------------------------------


class TestHelperMethods:
    """Test individual lookup helper methods."""

    def test_get_abbreviation(self, expander: MedicalQueryExpander) -> None:
        """get_abbreviation should return the abbreviation for a full term."""
        assert expander.get_abbreviation("hypertension") == "htn"
        assert expander.get_abbreviation("diabetes mellitus") == "dm"

    def test_get_abbreviation_not_found(
        self, expander: MedicalQueryExpander,
    ) -> None:
        """get_abbreviation should return None for unknown terms."""
        assert expander.get_abbreviation("xyzzy") is None

    def test_get_full_form(self, expander: MedicalQueryExpander) -> None:
        """get_full_form should return the full form for an abbreviation."""
        assert expander.get_full_form("htn") == "hypertension"
        assert expander.get_full_form("dm") == "diabetes mellitus"

    def test_get_full_form_not_found(
        self, expander: MedicalQueryExpander,
    ) -> None:
        """get_full_form should return None for unknown abbreviations."""
        assert expander.get_full_form("xyzzy") is None

    def test_get_synonyms(self, expander: MedicalQueryExpander) -> None:
        """get_synonyms should return a set of synonyms."""
        syns = expander.get_synonyms("anemia")
        assert isinstance(syns, set)
        assert "anaemia" in syns

    def test_get_synonyms_empty(self, expander: MedicalQueryExpander) -> None:
        """get_synonyms should return empty set for unknown terms."""
        syns = expander.get_synonyms("xyzzy")
        assert syns == set()

    def test_get_synonyms_case_insensitive(
        self, expander: MedicalQueryExpander,
    ) -> None:
        """get_synonyms should be case-insensitive."""
        assert expander.get_synonyms("ANEMIA") == expander.get_synonyms("anemia")
