"""Tests for Charlson Comorbidity Index calculator.

Covers:
- Enum completeness and data integrity
- ICD-10-CM code prefix matching (all 17 categories)
- Free-text extraction (all 17 categories)
- Hierarchical exclusion rules (3 pairs)
- Age adjustment computation
- 10-year mortality estimation
- Configuration options
- Edge cases (empty input, unknown codes, overlapping conditions)
- Batch calculation
- End-to-end realistic clinical scenarios
"""

import math

import pytest

from app.ml.comorbidity.charlson import (
    _HIERARCHICAL_PAIRS,
    _PREFIX_LOOKUP,
    _SORTED_PREFIXES,
    _TEXT_PATTERNS,
    CATEGORY_DESCRIPTIONS,
    CATEGORY_WEIGHTS,
    ICD10_PREFIXES,
    CCICategory,
    CharlsonCalculator,
    CharlsonConfig,
    ComorbidityMatch,
    MortalityEstimate,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def calculator() -> CharlsonCalculator:
    """Create a fresh CharlsonCalculator instance."""
    return CharlsonCalculator()


# ---------------------------------------------------------------------------
# Enum and registry completeness
# ---------------------------------------------------------------------------


class TestEnumCompleteness:
    """Verify all 17 categories are consistently defined everywhere."""

    def test_cci_category_has_17_members(self) -> None:
        assert len(CCICategory) == 17

    def test_all_categories_have_weights(self) -> None:
        for cat in CCICategory:
            assert cat in CATEGORY_WEIGHTS, f"{cat} missing from CATEGORY_WEIGHTS"

    def test_all_categories_have_descriptions(self) -> None:
        for cat in CCICategory:
            assert cat in CATEGORY_DESCRIPTIONS, f"{cat} missing from CATEGORY_DESCRIPTIONS"

    def test_all_categories_have_icd10_prefixes(self) -> None:
        for cat in CCICategory:
            assert cat in ICD10_PREFIXES, f"{cat} missing from ICD10_PREFIXES"
            assert len(ICD10_PREFIXES[cat]) > 0, f"{cat} has empty prefix list"

    def test_all_categories_have_text_patterns(self) -> None:
        for cat in CCICategory:
            assert cat in _TEXT_PATTERNS, f"{cat} missing from _TEXT_PATTERNS"
            assert len(_TEXT_PATTERNS[cat]) > 0, f"{cat} has empty pattern list"

    def test_weights_are_valid_values(self) -> None:
        valid_weights = {1, 2, 3, 6}
        for cat, weight in CATEGORY_WEIGHTS.items():
            assert weight in valid_weights, f"{cat} has invalid weight {weight}"

    def test_prefix_lookup_built_correctly(self) -> None:
        total_prefixes = sum(len(v) for v in ICD10_PREFIXES.values())
        assert len(_PREFIX_LOOKUP) == total_prefixes

    def test_sorted_prefixes_longest_first(self) -> None:
        for i in range(len(_SORTED_PREFIXES) - 1):
            assert len(_SORTED_PREFIXES[i]) >= len(_SORTED_PREFIXES[i + 1])


# ---------------------------------------------------------------------------
# Dataclass serialisation
# ---------------------------------------------------------------------------


class TestDataclasses:
    """Test dataclass construction and to_dict methods."""

    def test_comorbidity_match_to_dict(self) -> None:
        match = ComorbidityMatch(
            category=CCICategory.DEMENTIA,
            weight=1,
            source="icd_code",
            evidence="G30.1",
            confidence=1.0,
            description="Test description",
        )
        d = match.to_dict()
        assert d["category"] == "dementia"
        assert d["weight"] == 1
        assert d["source"] == "icd_code"
        assert d["confidence"] == 1.0

    def test_mortality_estimate_to_dict(self) -> None:
        est = MortalityEstimate(
            ten_year_mortality=0.12345,
            ten_year_survival=0.87655,
            risk_group="mild",
        )
        d = est.to_dict()
        assert d["ten_year_mortality"] == 0.1235
        assert d["ten_year_survival"] == 0.8766
        assert d["risk_group"] == "mild"

    def test_cci_result_to_dict(self, calculator: CharlsonCalculator) -> None:
        result = calculator.calculate(icd_codes=["I21.0"])
        d = result.to_dict()
        assert "raw_score" in d
        assert "matched_categories" in d
        assert "mortality_estimate" in d
        assert isinstance(d["matched_categories"], list)


# ---------------------------------------------------------------------------
# ICD-10-CM code matching
# ---------------------------------------------------------------------------


class TestICDCodeMatching:
    """Test prefix-based ICD-10-CM → CCI category resolution."""

    @pytest.mark.parametrize(
        "code,expected_category",
        [
            ("I21.0", CCICategory.MYOCARDIAL_INFARCTION),
            ("I22.1", CCICategory.MYOCARDIAL_INFARCTION),
            ("I252", CCICategory.MYOCARDIAL_INFARCTION),
            ("I50.9", CCICategory.CONGESTIVE_HEART_FAILURE),
            ("I420", CCICategory.CONGESTIVE_HEART_FAILURE),
            ("I70.1", CCICategory.PERIPHERAL_VASCULAR),
            ("I71.3", CCICategory.PERIPHERAL_VASCULAR),
            ("I63.9", CCICategory.CEREBROVASCULAR),
            ("G45.0", CCICategory.CEREBROVASCULAR),
            ("F01.5", CCICategory.DEMENTIA),
            ("G30.0", CCICategory.DEMENTIA),
            ("J44.1", CCICategory.CHRONIC_PULMONARY),
            ("J45.2", CCICategory.CHRONIC_PULMONARY),
            ("M05.7", CCICategory.RHEUMATIC),
            ("M32.1", CCICategory.RHEUMATIC),
            ("K25.0", CCICategory.PEPTIC_ULCER),
            ("K73.0", CCICategory.MILD_LIVER),
            ("E11.9", CCICategory.DIABETES_UNCOMPLICATED),
            ("E10.9", CCICategory.DIABETES_UNCOMPLICATED),
            ("E11.2", CCICategory.DIABETES_COMPLICATED),
            ("E10.5", CCICategory.DIABETES_COMPLICATED),
            ("G81.0", CCICategory.HEMIPLEGIA),
            ("G82.2", CCICategory.HEMIPLEGIA),
            ("N18.3", CCICategory.RENAL),
            ("N19", CCICategory.RENAL),
            ("C34.1", CCICategory.MALIGNANCY),
            ("C50.9", CCICategory.MALIGNANCY),
            ("K704", CCICategory.MODERATE_SEVERE_LIVER),
            ("I850", CCICategory.MODERATE_SEVERE_LIVER),
            ("C78.0", CCICategory.METASTATIC_TUMOR),
            ("C80.0", CCICategory.METASTATIC_TUMOR),
            ("B20", CCICategory.AIDS_HIV),
        ],
    )
    def test_code_maps_to_correct_category(
        self, calculator: CharlsonCalculator, code: str, expected_category: CCICategory
    ) -> None:
        matches = calculator.match_codes([code])
        assert len(matches) == 1
        assert matches[0].category == expected_category
        assert matches[0].source == "icd_code"
        assert matches[0].confidence == 1.0

    def test_unknown_code_returns_empty(self, calculator: CharlsonCalculator) -> None:
        matches = calculator.match_codes(["Z00.0"])
        assert len(matches) == 0

    def test_empty_code_returns_empty(self, calculator: CharlsonCalculator) -> None:
        matches = calculator.match_codes([""])
        assert len(matches) == 0

    def test_code_normalisation_removes_dots(self, calculator: CharlsonCalculator) -> None:
        """I21.0 and I210 should both match."""
        m1 = calculator.match_codes(["I21.0"])
        m2 = calculator.match_codes(["I210"])
        assert m1[0].category == m2[0].category

    def test_code_normalisation_case_insensitive(self, calculator: CharlsonCalculator) -> None:
        m1 = calculator.match_codes(["i21.0"])
        m2 = calculator.match_codes(["I21.0"])
        assert m1[0].category == m2[0].category

    def test_multiple_codes_same_category(self, calculator: CharlsonCalculator) -> None:
        """Multiple MI codes should still map to same category."""
        matches = calculator.match_codes(["I21.0", "I22.1"])
        assert all(m.category == CCICategory.MYOCARDIAL_INFARCTION for m in matches)

    def test_multiple_codes_different_categories(self, calculator: CharlsonCalculator) -> None:
        matches = calculator.match_codes(["I21.0", "E11.9", "N18.3"])
        categories = {m.category for m in matches}
        assert CCICategory.MYOCARDIAL_INFARCTION in categories
        assert CCICategory.DIABETES_UNCOMPLICATED in categories
        assert CCICategory.RENAL in categories


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


class TestTextExtraction:
    """Test keyword-based comorbidity extraction from free text."""

    @pytest.mark.parametrize(
        "text,expected_category",
        [
            ("History of myocardial infarction in 2019", CCICategory.MYOCARDIAL_INFARCTION),
            ("Patient has STEMI", CCICategory.MYOCARDIAL_INFARCTION),
            ("Diagnosed with congestive heart failure", CCICategory.CONGESTIVE_HEART_FAILURE),
            ("CHF with reduced ejection fraction", CCICategory.CONGESTIVE_HEART_FAILURE),
            ("Peripheral vascular disease, claudication", CCICategory.PERIPHERAL_VASCULAR),
            ("History of CVA with mild residua", CCICategory.CEREBROVASCULAR),
            ("TIA in 2020", CCICategory.CEREBROVASCULAR),
            ("Progressive dementia, likely Alzheimer", CCICategory.DEMENTIA),
            ("Severe COPD on home oxygen", CCICategory.CHRONIC_PULMONARY),
            ("Rheumatoid arthritis on methotrexate", CCICategory.RHEUMATIC),
            ("History of peptic ulcer disease", CCICategory.PEPTIC_ULCER),
            ("Chronic hepatitis B carrier", CCICategory.MILD_LIVER),
            ("Type 2 diabetes mellitus", CCICategory.DIABETES_UNCOMPLICATED),
            ("Diabetic nephropathy with CKD", CCICategory.DIABETES_COMPLICATED),
            ("Hemiplegia from prior stroke", CCICategory.HEMIPLEGIA),
            ("End-stage renal disease on dialysis", CCICategory.RENAL),
            ("Metastatic colon cancer", CCICategory.MALIGNANCY),
            ("Metastatic disease to liver and lungs", CCICategory.METASTATIC_TUMOR),
            ("HIV positive on antiretroviral therapy", CCICategory.AIDS_HIV),
        ],
    )
    def test_text_matches_category(
        self, calculator: CharlsonCalculator, text: str, expected_category: CCICategory
    ) -> None:
        matches = calculator.extract_from_text(text)
        categories = {m.category for m in matches}
        assert expected_category in categories

    def test_text_match_confidence_below_icd(self, calculator: CharlsonCalculator) -> None:
        """Text matches should have lower confidence than ICD codes."""
        matches = calculator.extract_from_text("History of myocardial infarction")
        assert all(m.confidence < 1.0 for m in matches)
        assert all(m.source == "text" for m in matches)

    def test_text_multi_word_higher_confidence(self, calculator: CharlsonCalculator) -> None:
        matches_long = calculator.extract_from_text("congestive heart failure")
        matches_short = calculator.extract_from_text("CHF")
        # Multi-word → higher confidence.
        assert matches_long[0].confidence > matches_short[0].confidence

    def test_text_min_confidence_threshold(self, calculator: CharlsonCalculator) -> None:
        """Raising threshold should filter out low-confidence matches."""
        all_matches = calculator.extract_from_text("MI, COPD, RA", min_confidence=0.0)
        filtered = calculator.extract_from_text("MI, COPD, RA", min_confidence=0.90)
        assert len(filtered) < len(all_matches)

    def test_text_no_matches(self, calculator: CharlsonCalculator) -> None:
        matches = calculator.extract_from_text("Patient is healthy, no complaints")
        assert len(matches) == 0

    def test_text_deduplication_per_category(self, calculator: CharlsonCalculator) -> None:
        """Multiple mentions of same category should not duplicate."""
        text = "CHF, congestive heart failure, heart failure"
        matches = calculator.extract_from_text(text)
        chf_matches = [m for m in matches if m.category == CCICategory.CONGESTIVE_HEART_FAILURE]
        assert len(chf_matches) == 1


# ---------------------------------------------------------------------------
# Hierarchical exclusion
# ---------------------------------------------------------------------------


class TestHierarchicalExclusion:
    """Test that mild categories are excluded when severe variants present."""

    def test_diabetes_exclusion(self, calculator: CharlsonCalculator) -> None:
        """Uncomplicated diabetes excluded when complicated present."""
        result = calculator.calculate(
            icd_codes=["E11.9", "E11.2"],  # uncomplicated + complicated
            config=CharlsonConfig(include_text_extraction=False),
        )
        matched_cats = {m.category for m in result.matched_categories}
        excluded_cats = {m.category for m in result.excluded_categories}
        assert CCICategory.DIABETES_COMPLICATED in matched_cats
        assert CCICategory.DIABETES_UNCOMPLICATED in excluded_cats

    def test_liver_exclusion(self, calculator: CharlsonCalculator) -> None:
        """Mild liver excluded when moderate/severe liver present."""
        result = calculator.calculate(
            icd_codes=["K73.0", "K704"],  # mild + severe
            config=CharlsonConfig(include_text_extraction=False),
        )
        matched_cats = {m.category for m in result.matched_categories}
        excluded_cats = {m.category for m in result.excluded_categories}
        assert CCICategory.MODERATE_SEVERE_LIVER in matched_cats
        assert CCICategory.MILD_LIVER in excluded_cats

    def test_malignancy_exclusion(self, calculator: CharlsonCalculator) -> None:
        """Malignancy excluded when metastatic tumor present."""
        result = calculator.calculate(
            icd_codes=["C34.1", "C78.0"],  # primary + metastatic
            config=CharlsonConfig(include_text_extraction=False),
        )
        matched_cats = {m.category for m in result.matched_categories}
        excluded_cats = {m.category for m in result.excluded_categories}
        assert CCICategory.METASTATIC_TUMOR in matched_cats
        assert CCICategory.MALIGNANCY in excluded_cats

    def test_exclusion_disabled(self, calculator: CharlsonCalculator) -> None:
        """Both categories counted when exclusion disabled."""
        result = calculator.calculate(
            icd_codes=["E11.9", "E11.2"],
            config=CharlsonConfig(
                include_text_extraction=False,
                hierarchical_exclusion=False,
            ),
        )
        matched_cats = {m.category for m in result.matched_categories}
        assert CCICategory.DIABETES_UNCOMPLICATED in matched_cats
        assert CCICategory.DIABETES_COMPLICATED in matched_cats
        assert len(result.excluded_categories) == 0

    def test_exclusion_only_lower_removed(self, calculator: CharlsonCalculator) -> None:
        """Only the lower-weight variant should be excluded."""
        result = calculator.calculate(
            icd_codes=["E11.9", "E11.2"],
            config=CharlsonConfig(include_text_extraction=False),
        )
        # Weight 1 (uncomplicated) excluded, weight 2 (complicated) kept.
        assert result.raw_score == 2  # Only complicated diabetes counts.

    def test_all_hierarchical_pairs_defined(self) -> None:
        """All 3 hierarchical pairs should be defined."""
        assert len(_HIERARCHICAL_PAIRS) == 3
        for lower, higher in _HIERARCHICAL_PAIRS:
            assert CATEGORY_WEIGHTS[lower] < CATEGORY_WEIGHTS[higher]


# ---------------------------------------------------------------------------
# Age adjustment
# ---------------------------------------------------------------------------


class TestAgeAdjustment:
    """Test age-based point calculation."""

    @pytest.mark.parametrize(
        "age,expected_points",
        [
            (30, 0),
            (40, 0),
            (49, 0),
            (50, 1),
            (59, 1),
            (60, 2),
            (69, 2),
            (70, 3),
            (79, 3),
            (80, 4),
            (90, 4),
            (100, 4),
        ],
    )
    def test_age_points(self, age: int, expected_points: int) -> None:
        points = CharlsonCalculator._compute_age_points(age)
        assert points == expected_points

    def test_age_adjusted_score(self, calculator: CharlsonCalculator) -> None:
        result = calculator.calculate(
            icd_codes=["I21.0"],  # weight 1
            config=CharlsonConfig(age_adjust=True, patient_age=72),
        )
        assert result.raw_score == 1
        assert result.age_adjusted_score == 1 + 3  # age 72 → 3 points

    def test_no_age_adjustment_without_age(self, calculator: CharlsonCalculator) -> None:
        result = calculator.calculate(
            icd_codes=["I21.0"],
            config=CharlsonConfig(age_adjust=True, patient_age=None),
        )
        assert result.age_adjusted_score is None

    def test_no_age_adjustment_when_disabled(self, calculator: CharlsonCalculator) -> None:
        result = calculator.calculate(
            icd_codes=["I21.0"],
            config=CharlsonConfig(age_adjust=False, patient_age=72),
        )
        assert result.age_adjusted_score is None


# ---------------------------------------------------------------------------
# Mortality estimation
# ---------------------------------------------------------------------------


class TestMortalityEstimation:
    """Test 10-year mortality formula and risk groups."""

    def test_score_0_low_risk(self) -> None:
        est = CharlsonCalculator._estimate_mortality(0)
        assert est.risk_group == "low"
        assert est.ten_year_mortality < 0.05

    def test_score_1_mild_risk(self) -> None:
        est = CharlsonCalculator._estimate_mortality(1)
        assert est.risk_group == "mild"

    def test_score_2_mild_risk(self) -> None:
        est = CharlsonCalculator._estimate_mortality(2)
        assert est.risk_group == "mild"

    def test_score_3_moderate_risk(self) -> None:
        est = CharlsonCalculator._estimate_mortality(3)
        assert est.risk_group == "moderate"

    def test_score_5_severe_risk(self) -> None:
        est = CharlsonCalculator._estimate_mortality(5)
        assert est.risk_group == "severe"

    def test_mortality_survival_sum_to_one(self) -> None:
        for score in range(0, 15):
            est = CharlsonCalculator._estimate_mortality(score)
            assert abs(est.ten_year_mortality + est.ten_year_survival - 1.0) < 1e-10

    def test_mortality_increases_with_score(self) -> None:
        prev_mortality = 0.0
        for score in range(0, 10):
            est = CharlsonCalculator._estimate_mortality(score)
            assert est.ten_year_mortality >= prev_mortality
            prev_mortality = est.ten_year_mortality

    def test_mortality_formula_matches_charlson(self) -> None:
        """Verify against the manual formula for CCI=3."""
        cci = 3
        exponent = math.exp(cci * 0.9)
        expected_survival = 0.983 ** exponent
        expected_mortality = 1.0 - expected_survival

        est = CharlsonCalculator._estimate_mortality(cci)
        assert abs(est.ten_year_mortality - expected_mortality) < 1e-10

    def test_mortality_clamped(self) -> None:
        """Very high scores should not exceed 1.0."""
        est = CharlsonCalculator._estimate_mortality(50)
        assert 0.0 <= est.ten_year_mortality <= 1.0
        assert 0.0 <= est.ten_year_survival <= 1.0


# ---------------------------------------------------------------------------
# Full calculation
# ---------------------------------------------------------------------------


class TestFullCalculation:
    """End-to-end calculation tests."""

    def test_single_code(self, calculator: CharlsonCalculator) -> None:
        result = calculator.calculate(
            icd_codes=["I21.0"],
            config=CharlsonConfig(include_text_extraction=False),
        )
        assert result.raw_score == 1
        assert result.category_count == 1
        assert result.processing_time_ms >= 0

    def test_multiple_codes_different_categories(self, calculator: CharlsonCalculator) -> None:
        result = calculator.calculate(
            icd_codes=["I21.0", "E11.9", "J44.1"],
            config=CharlsonConfig(include_text_extraction=False),
        )
        assert result.raw_score == 3  # 1 + 1 + 1
        assert result.category_count == 3

    def test_duplicate_category_counted_once(self, calculator: CharlsonCalculator) -> None:
        """Two MI codes should not double-count."""
        result = calculator.calculate(
            icd_codes=["I21.0", "I22.1"],
            config=CharlsonConfig(include_text_extraction=False),
        )
        assert result.raw_score == 1
        assert result.category_count == 1

    def test_text_only_calculation(self, calculator: CharlsonCalculator) -> None:
        result = calculator.calculate(
            text="Patient has COPD and type 2 diabetes mellitus",
        )
        assert result.raw_score >= 2
        assert result.category_count >= 2

    def test_icd_takes_priority_over_text(self, calculator: CharlsonCalculator) -> None:
        """When code and text both match same category, ICD code is used."""
        result = calculator.calculate(
            icd_codes=["I21.0"],
            text="History of myocardial infarction",
        )
        mi_matches = [
            m for m in result.matched_categories
            if m.category == CCICategory.MYOCARDIAL_INFARCTION
        ]
        assert len(mi_matches) == 1
        assert mi_matches[0].source == "icd_code"  # ICD wins.

    def test_text_adds_uncoded_conditions(self, calculator: CharlsonCalculator) -> None:
        """Text extraction catches conditions not in the code list."""
        result = calculator.calculate(
            icd_codes=["I21.0"],
            text="Also has COPD and dementia",
        )
        categories = {m.category for m in result.matched_categories}
        assert CCICategory.MYOCARDIAL_INFARCTION in categories
        assert CCICategory.CHRONIC_PULMONARY in categories
        assert CCICategory.DEMENTIA in categories

    def test_text_extraction_disabled(self, calculator: CharlsonCalculator) -> None:
        result = calculator.calculate(
            icd_codes=["I21.0"],
            text="Also has COPD and dementia",
            config=CharlsonConfig(include_text_extraction=False),
        )
        assert result.category_count == 1  # Only MI from ICD.

    def test_no_input_raises(self, calculator: CharlsonCalculator) -> None:
        with pytest.raises(ValueError, match="At least one"):
            calculator.calculate()

    def test_empty_lists_raise(self, calculator: CharlsonCalculator) -> None:
        with pytest.raises(ValueError, match="At least one"):
            calculator.calculate(icd_codes=[], text="")

    def test_config_recorded_in_result(self, calculator: CharlsonCalculator) -> None:
        result = calculator.calculate(
            icd_codes=["I21.0"],
            config=CharlsonConfig(age_adjust=True, patient_age=65),
        )
        assert result.config["age_adjust"] is True
        assert result.config["patient_age"] == 65

    def test_matched_categories_sorted_by_weight_desc(
        self, calculator: CharlsonCalculator
    ) -> None:
        result = calculator.calculate(
            icd_codes=["I21.0", "C78.0", "N18.3"],  # weight 1, 6, 2
            config=CharlsonConfig(include_text_extraction=False),
        )
        weights = [m.weight for m in result.matched_categories]
        assert weights == sorted(weights, reverse=True)


# ---------------------------------------------------------------------------
# Realistic clinical scenarios
# ---------------------------------------------------------------------------


class TestRealisticScenarios:
    """End-to-end tests with realistic patient data."""

    def test_complex_multimorbid_patient(self, calculator: CharlsonCalculator) -> None:
        """Elderly patient with multiple comorbidities."""
        result = calculator.calculate(
            icd_codes=[
                "I21.4",   # MI (1)
                "I50.9",   # CHF (1)
                "J44.1",   # COPD (1)
                "E11.2",   # Diabetes complicated (2)
                "N18.4",   # CKD (2)
                "C78.7",   # Metastatic tumor (6)
            ],
            config=CharlsonConfig(
                age_adjust=True,
                patient_age=78,
                include_text_extraction=False,
            ),
        )
        # MI(1) + CHF(1) + COPD(1) + DM-complicated(2) + CKD(2) + metastatic(6) = 13
        assert result.raw_score == 13
        # Age 78 → 3 points → adjusted = 16
        assert result.age_adjusted_score == 16
        assert result.mortality_estimate.risk_group == "severe"

    def test_healthy_patient_with_only_text(self, calculator: CharlsonCalculator) -> None:
        text = (
            "65-year-old male presents for annual wellness exam. "
            "No significant past medical history. "
            "Vitals within normal limits."
        )
        result = calculator.calculate(text=text)
        assert result.raw_score == 0
        assert result.mortality_estimate.risk_group == "low"

    def test_mixed_code_and_text_with_exclusion(self, calculator: CharlsonCalculator) -> None:
        """Codes provide some conditions, text catches others."""
        result = calculator.calculate(
            icd_codes=["E11.9", "E11.5"],  # DM uncomplicated + complicated
            text="Patient also has chronic hepatitis B and COPD",
            config=CharlsonConfig(hierarchical_exclusion=True),
        )
        matched = {m.category for m in result.matched_categories}
        # DM uncomplicated should be excluded (diabetes complicated present).
        assert CCICategory.DIABETES_COMPLICATED in matched
        assert CCICategory.DIABETES_UNCOMPLICATED not in matched
        # Text-extracted conditions should be present.
        assert CCICategory.MILD_LIVER in matched
        assert CCICategory.CHRONIC_PULMONARY in matched


# ---------------------------------------------------------------------------
# Category info static method
# ---------------------------------------------------------------------------


class TestCategoryInfo:
    """Test the static category info helper."""

    def test_returns_17_categories(self) -> None:
        info = CharlsonCalculator.get_category_info()
        assert len(info) == 17

    def test_info_structure(self) -> None:
        info = CharlsonCalculator.get_category_info()
        for entry in info:
            assert "category" in entry
            assert "weight" in entry
            assert "description" in entry
            assert isinstance(entry["weight"], int)
