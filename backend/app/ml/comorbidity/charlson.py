"""Charlson Comorbidity Index (CCI) calculator.

Implements the Charlson–Deyo adaptation of the original Charlson
Comorbidity Index (Charlson et al., 1987) using ICD-10-CM code mappings
based on Quan et al. (2005) coding algorithms.

The CCI assigns integer weights (1–6) to 17 disease categories and sums
them into a composite score that predicts 10-year mortality risk.  This
module supports:

* **ICD-10-CM code matching** — prefix-based lookup against ~200 validated
  code prefixes across 17 categories.
* **Free-text extraction** — keyword/phrase matching for common clinical
  terminology when structured codes are unavailable.
* **Age adjustment** — optional Charlson–Deyo age-based point addition
  (1 point per decade above 40, capped at 4).
* **10-year mortality estimation** — Charlson's original exponential
  survival formula: estimated_mortality = 1 − 0.983^(e^(CCI × 0.9)).

Design decisions
----------------
* **Prefix matching, not exact match**: ICD-10-CM codes are hierarchical;
  ``I21`` covers ``I21.0`` through ``I21.9``.  Prefix matching handles the
  full specificity range without enumerating every leaf code.
* **Weight-based deduplication**: When multiple codes map to the same
  category, the category is counted only once (highest-weight match wins).
* **Text extraction is supplementary**: It catches conditions documented
  in notes but not yet coded.  Confidence is lower (0.70–0.85 vs. 1.0
  for ICD-coded conditions) to reflect this.
* **Immutable category registry**: The ``CHARLSON_CATEGORIES`` dict is
  built once at import time from ``_build_category_registry()`` and never
  mutated at runtime.

References
----------
1. Charlson ME, Pompei P, Ales KL, MacKenzie CR. A new method of
   classifying prognostic comorbidity in longitudinal studies.
   J Chronic Dis. 1987;40(5):373-383.
2. Quan H, Sundararajan V, Halfon P, et al. Coding algorithms for
   defining comorbidities in ICD-9-CM and ICD-10-CM data. Med Care.
   2005;43(11):1130-1139.
3. Deyo RA, Cherkin DC, Ciol MA. Adapting a clinical comorbidity index
   for use with ICD-9-CM administrative databases. J Clin Epidemiol.
   1992;45(6):613-619.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and data classes
# ---------------------------------------------------------------------------


class CCICategory(StrEnum):
    """The 17 Charlson Comorbidity Index disease categories.

    Each category carries an integer weight (1, 2, 3, or 6) reflecting
    its relative contribution to mortality risk.  Values are the
    human-readable category names used in API responses and model cards.
    """

    MYOCARDIAL_INFARCTION = "myocardial_infarction"
    CONGESTIVE_HEART_FAILURE = "congestive_heart_failure"
    PERIPHERAL_VASCULAR = "peripheral_vascular_disease"
    CEREBROVASCULAR = "cerebrovascular_disease"
    DEMENTIA = "dementia"
    CHRONIC_PULMONARY = "chronic_pulmonary_disease"
    RHEUMATIC = "rheumatic_disease"
    PEPTIC_ULCER = "peptic_ulcer_disease"
    MILD_LIVER = "mild_liver_disease"
    DIABETES_UNCOMPLICATED = "diabetes_uncomplicated"
    DIABETES_COMPLICATED = "diabetes_complicated"
    HEMIPLEGIA = "hemiplegia_paraplegia"
    RENAL = "renal_disease"
    MALIGNANCY = "malignancy"
    MODERATE_SEVERE_LIVER = "moderate_severe_liver_disease"
    METASTATIC_TUMOR = "metastatic_solid_tumor"
    AIDS_HIV = "aids_hiv"


# Category weights per the original Charlson scoring (1987).
CATEGORY_WEIGHTS: dict[CCICategory, int] = {
    CCICategory.MYOCARDIAL_INFARCTION: 1,
    CCICategory.CONGESTIVE_HEART_FAILURE: 1,
    CCICategory.PERIPHERAL_VASCULAR: 1,
    CCICategory.CEREBROVASCULAR: 1,
    CCICategory.DEMENTIA: 1,
    CCICategory.CHRONIC_PULMONARY: 1,
    CCICategory.RHEUMATIC: 1,
    CCICategory.PEPTIC_ULCER: 1,
    CCICategory.MILD_LIVER: 1,
    CCICategory.DIABETES_UNCOMPLICATED: 1,
    CCICategory.DIABETES_COMPLICATED: 2,
    CCICategory.HEMIPLEGIA: 2,
    CCICategory.RENAL: 2,
    CCICategory.MALIGNANCY: 2,
    CCICategory.MODERATE_SEVERE_LIVER: 3,
    CCICategory.METASTATIC_TUMOR: 6,
    CCICategory.AIDS_HIV: 6,
}

# Category human-readable descriptions for API display.
CATEGORY_DESCRIPTIONS: dict[CCICategory, str] = {
    CCICategory.MYOCARDIAL_INFARCTION: "History of definite or probable myocardial infarction",
    CCICategory.CONGESTIVE_HEART_FAILURE: "Congestive heart failure with exertional or paroxysmal dyspnea",
    CCICategory.PERIPHERAL_VASCULAR: "Intermittent claudication, bypass, or aneurysm >6 cm",
    CCICategory.CEREBROVASCULAR: "Cerebrovascular accident (CVA) with minor or no residua, or TIA",
    CCICategory.DEMENTIA: "Chronic cognitive deficit including Alzheimer's disease",
    CCICategory.CHRONIC_PULMONARY: "COPD, asthma, or other chronic pulmonary disease",
    CCICategory.RHEUMATIC: "Rheumatoid arthritis, SLE, polymyositis, or mixed connective tissue disease",
    CCICategory.PEPTIC_ULCER: "Any history of treatment for peptic ulcer disease",
    CCICategory.MILD_LIVER: "Chronic hepatitis or cirrhosis without portal hypertension",
    CCICategory.DIABETES_UNCOMPLICATED: "Diabetes treated with insulin or oral agents, no end-organ damage",
    CCICategory.DIABETES_COMPLICATED: "Diabetes with retinopathy, neuropathy, nephropathy, or poor control",
    CCICategory.HEMIPLEGIA: "Hemiplegia or paraplegia from any cause",
    CCICategory.RENAL: "Chronic kidney disease (creatinine >3 mg/dL, dialysis, transplant, uremia)",
    CCICategory.MALIGNANCY: "Any malignancy including lymphoma, leukemia (excluding skin cancer)",
    CCICategory.MODERATE_SEVERE_LIVER: "Cirrhosis with portal hypertension or variceal bleeding history",
    CCICategory.METASTATIC_TUMOR: "Metastatic solid tumor of any origin",
    CCICategory.AIDS_HIV: "AIDS or HIV with CD4 <200 or AIDS-defining illness",
}


# ---------------------------------------------------------------------------
# ICD-10-CM code prefix mappings (Quan et al. 2005)
# ---------------------------------------------------------------------------

# Each key is a CCICategory; each value is a tuple of ICD-10-CM code
# prefixes.  A patient code matches if it starts with any listed prefix.
# Prefixes are stored upper-cased and stripped of dots for normalisation.

ICD10_PREFIXES: dict[CCICategory, tuple[str, ...]] = {
    CCICategory.MYOCARDIAL_INFARCTION: (
        "I21", "I22", "I252",
    ),
    CCICategory.CONGESTIVE_HEART_FAILURE: (
        "I099", "I110", "I130", "I132", "I255", "I420",
        "I425", "I426", "I427", "I428", "I429", "I43",
        "I50", "P290",
    ),
    CCICategory.PERIPHERAL_VASCULAR: (
        "I70", "I71", "I731", "I738", "I739", "I771",
        "I790", "I792", "K551", "K558", "K559", "Z958", "Z959",
    ),
    CCICategory.CEREBROVASCULAR: (
        "G45", "G46", "H340", "H341", "H342",
        "I60", "I61", "I62", "I63", "I64", "I65", "I66",
        "I67", "I68", "I69",
    ),
    CCICategory.DEMENTIA: (
        "F00", "F01", "F02", "F03", "F051", "G30", "G311",
    ),
    CCICategory.CHRONIC_PULMONARY: (
        "I278", "I279",
        "J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47",
        "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67",
        "J684", "J701", "J703",
    ),
    CCICategory.RHEUMATIC: (
        "M05", "M06", "M315", "M32", "M33", "M34",
        "M351", "M353", "M360",
    ),
    CCICategory.PEPTIC_ULCER: (
        "K25", "K26", "K27", "K28",
    ),
    CCICategory.MILD_LIVER: (
        "B18", "K700", "K701", "K702", "K703", "K709",
        "K713", "K714", "K715",
        "K717", "K73", "K74",
        "K760", "K762", "K763", "K764",
        "K768", "K769", "Z944",
    ),
    CCICategory.DIABETES_UNCOMPLICATED: (
        "E100", "E101", "E106", "E108", "E109",
        "E110", "E111", "E116", "E118", "E119",
        "E120", "E121", "E126", "E128", "E129",
        "E130", "E131", "E136", "E138", "E139",
        "E140", "E141", "E146", "E148", "E149",
    ),
    CCICategory.DIABETES_COMPLICATED: (
        "E102", "E103", "E104", "E105", "E107",
        "E112", "E113", "E114", "E115", "E117",
        "E122", "E123", "E124", "E125", "E127",
        "E132", "E133", "E134", "E135", "E137",
        "E142", "E143", "E144", "E145", "E147",
    ),
    CCICategory.HEMIPLEGIA: (
        "G041", "G114", "G801", "G802",
        "G81", "G82", "G830", "G831", "G832", "G833", "G834",
        "G839",
    ),
    CCICategory.RENAL: (
        "I120", "I131",
        "N032", "N033", "N034", "N035", "N036", "N037",
        "N052", "N053", "N054", "N055", "N056", "N057",
        "N18", "N19", "N250",
        "Z490", "Z491", "Z492", "Z940", "Z992",
    ),
    CCICategory.MALIGNANCY: (
        "C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09",
        "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18", "C19",
        "C20", "C21", "C22", "C23", "C24", "C25", "C26",
        "C30", "C31", "C32", "C33", "C34",
        "C37", "C38", "C39",
        "C40", "C41", "C43",
        "C45", "C46", "C47", "C48", "C49",
        "C50", "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58",
        "C60", "C61", "C62", "C63", "C64", "C65", "C66", "C67", "C68",
        "C69", "C70", "C71", "C72", "C73", "C74", "C75",
        "C76",
        "C81", "C82", "C83", "C84", "C85", "C88",
        "C90", "C91", "C92", "C93", "C94", "C95", "C96", "C97",
    ),
    CCICategory.MODERATE_SEVERE_LIVER: (
        "I850", "I859", "I864", "I982",
        "K704", "K711", "K721", "K729", "K765", "K766", "K767",
    ),
    CCICategory.METASTATIC_TUMOR: (
        "C77", "C78", "C79", "C80",
    ),
    CCICategory.AIDS_HIV: (
        "B20", "B21", "B22", "B24",
    ),
}

# ---------------------------------------------------------------------------
# Text-based keyword patterns for each category
# ---------------------------------------------------------------------------
# These compiled regexes match common clinical phrases for conditions that
# may appear in free-text notes without accompanying ICD codes.
# Word-boundary (\b) anchoring prevents substring false positives.

_TEXT_PATTERNS: dict[CCICategory, list[re.Pattern[str]]] = {
    CCICategory.MYOCARDIAL_INFARCTION: [
        re.compile(r"\b(?:myocardial\s+infarction|heart\s+attack|STEMI|NSTEMI|MI)\b", re.I),
    ],
    CCICategory.CONGESTIVE_HEART_FAILURE: [
        re.compile(r"\b(?:congestive\s+heart\s+failure|CHF|heart\s+failure|systolic\s+dysfunction|diastolic\s+dysfunction|HFrEF|HFpEF)\b", re.I),
    ],
    CCICategory.PERIPHERAL_VASCULAR: [
        re.compile(r"\b(?:peripheral\s+vascular\s+disease|PVD|peripheral\s+arterial\s+disease|PAD|claudication|aortic\s+aneurysm)\b", re.I),
    ],
    CCICategory.CEREBROVASCULAR: [
        re.compile(r"\b(?:cerebrovascular\s+accident|CVA|stroke|transient\s+ischemic\s+attack|TIA|cerebral\s+infarction)\b", re.I),
    ],
    CCICategory.DEMENTIA: [
        re.compile(r"\b(?:dementia|Alzheimer|cognitive\s+impairment|Lewy\s+body|vascular\s+dementia|frontotemporal)\b", re.I),
    ],
    CCICategory.CHRONIC_PULMONARY: [
        re.compile(r"\b(?:COPD|chronic\s+obstructive\s+pulmonary|emphysema|chronic\s+bronchitis|asthma|pulmonary\s+fibrosis)\b", re.I),
    ],
    CCICategory.RHEUMATIC: [
        re.compile(r"\b(?:rheumatoid\s+arthritis|RA|systemic\s+lupus|SLE|polymyositis|dermatomyositis|scleroderma|mixed\s+connective\s+tissue)\b", re.I),
    ],
    CCICategory.PEPTIC_ULCER: [
        re.compile(r"\b(?:peptic\s+ulcer|gastric\s+ulcer|duodenal\s+ulcer|GI\s+bleed)\b", re.I),
    ],
    CCICategory.MILD_LIVER: [
        re.compile(r"\b(?:chronic\s+hepatitis|hepatitis\s+[BC]|cirrhosis|fatty\s+liver|NAFLD|NASH)\b", re.I),
    ],
    CCICategory.DIABETES_UNCOMPLICATED: [
        re.compile(r"\b(?:diabetes\s+mellitus|type\s+[12]\s+diabetes|DM[12]?|T[12]DM)\b", re.I),
    ],
    CCICategory.DIABETES_COMPLICATED: [
        re.compile(r"\b(?:diabetic\s+(?:retinopathy|neuropathy|nephropathy|foot\s+ulcer|ketoacidosis)|DKA|HHS)\b", re.I),
    ],
    CCICategory.HEMIPLEGIA: [
        re.compile(r"\b(?:hemiplegia|paraplegia|quadriplegia|tetraplegia|spinal\s+cord\s+injury)\b", re.I),
    ],
    CCICategory.RENAL: [
        re.compile(r"\b(?:chronic\s+kidney\s+disease|CKD|end[- ]stage\s+renal|ESRD|dialysis|renal\s+failure|kidney\s+transplant)\b", re.I),
    ],
    CCICategory.MALIGNANCY: [
        re.compile(r"\b(?:cancer|carcinoma|lymphoma|leukemia|leukaemia|malignancy|melanoma|sarcoma|myeloma)\b", re.I),
    ],
    CCICategory.MODERATE_SEVERE_LIVER: [
        re.compile(r"\b(?:portal\s+hypertension|variceal\s+bleed|esophageal\s+varices|hepatic\s+encephalopathy|ascites|liver\s+failure)\b", re.I),
    ],
    CCICategory.METASTATIC_TUMOR: [
        re.compile(r"\b(?:metastatic|metastasis|metastases|stage\s+IV\s+cancer|mets\s+to)\b", re.I),
    ],
    CCICategory.AIDS_HIV: [
        re.compile(r"\b(?:AIDS|HIV\s+positive|HIV/AIDS|human\s+immunodeficiency\s+virus)\b", re.I),
    ],
}


# ---------------------------------------------------------------------------
# Pre-built lookup structures for O(1) prefix matching
# ---------------------------------------------------------------------------

def _build_prefix_lookup() -> dict[str, CCICategory]:
    """Build a flat prefix → category lookup dictionary.

    Returns
    -------
    dict[str, CCICategory]
        Maps each normalised ICD-10-CM prefix to its CCI category.
    """
    lookup: dict[str, CCICategory] = {}
    for category, prefixes in ICD10_PREFIXES.items():
        for prefix in prefixes:
            normalised = prefix.upper().replace(".", "")
            lookup[normalised] = category
    return lookup


_PREFIX_LOOKUP: dict[str, CCICategory] = _build_prefix_lookup()

# Sorted prefixes by length descending for longest-match-first resolution.
_SORTED_PREFIXES: list[str] = sorted(_PREFIX_LOOKUP.keys(), key=len, reverse=True)


# ---------------------------------------------------------------------------
# Configuration and result data classes
# ---------------------------------------------------------------------------


@dataclass
class CharlsonConfig:
    """Configuration for Charlson Comorbidity Index calculation.

    Parameters
    ----------
    age_adjust : bool
        Whether to add age-based points (1 per decade above 40, max 4).
    patient_age : int | None
        Patient age in years for age adjustment.
    include_text_extraction : bool
        Whether to also scan free text for comorbidity keywords.
    text_confidence_threshold : float
        Minimum confidence for text-extracted conditions (0.0–1.0).
    hierarchical_exclusion : bool
        When True, if both mild and severe versions of a condition are
        detected (e.g., mild liver + severe liver), only the higher-weight
        version counts.  This follows the standard Charlson convention.
    """

    age_adjust: bool = False
    patient_age: int | None = None
    include_text_extraction: bool = True
    text_confidence_threshold: float = 0.70
    hierarchical_exclusion: bool = True


@dataclass
class ComorbidityMatch:
    """A single matched comorbidity condition.

    Parameters
    ----------
    category : CCICategory
        CCI disease category.
    weight : int
        Charlson weight for this category (1, 2, 3, or 6).
    source : str
        How this match was identified: ``"icd_code"`` or ``"text"``.
    evidence : str
        The specific code or text phrase that triggered the match.
    confidence : float
        Match confidence (1.0 for ICD codes, lower for text matches).
    description : str
        Human-readable description of the category.
    """

    category: CCICategory
    weight: int
    source: str
    evidence: str
    confidence: float
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "category": self.category.value,
            "weight": self.weight,
            "source": self.source,
            "evidence": self.evidence,
            "confidence": round(self.confidence, 4),
            "description": self.description,
        }


@dataclass
class MortalityEstimate:
    """Estimated 10-year mortality from the CCI score.

    The formula uses Charlson's original exponential survival model:
    ``estimated_mortality = 1 − 0.983 ^ exp(CCI × 0.9)``

    Parameters
    ----------
    ten_year_mortality : float
        Estimated 10-year mortality probability (0.0–1.0).
    ten_year_survival : float
        Estimated 10-year survival probability (0.0–1.0).
    risk_group : str
        Categorical risk group: ``"low"`` (0), ``"mild"`` (1–2),
        ``"moderate"`` (3–4), ``"severe"`` (≥5).
    """

    ten_year_mortality: float
    ten_year_survival: float
    risk_group: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "ten_year_mortality": round(self.ten_year_mortality, 4),
            "ten_year_survival": round(self.ten_year_survival, 4),
            "risk_group": self.risk_group,
        }


@dataclass
class CCIResult:
    """Complete Charlson Comorbidity Index calculation result.

    Parameters
    ----------
    raw_score : int
        Sum of Charlson weights for detected categories.
    age_adjusted_score : int | None
        Score with age adjustment points, or None if age not provided.
    matched_categories : list[ComorbidityMatch]
        All matched comorbidity conditions with evidence.
    excluded_categories : list[ComorbidityMatch]
        Categories removed by hierarchical exclusion rules.
    mortality_estimate : MortalityEstimate
        10-year mortality/survival estimates.
    category_count : int
        Number of distinct CCI categories detected.
    processing_time_ms : float
        Calculation time in milliseconds.
    config : dict[str, Any]
        Configuration used for this calculation.
    """

    raw_score: int
    age_adjusted_score: int | None
    matched_categories: list[ComorbidityMatch]
    excluded_categories: list[ComorbidityMatch]
    mortality_estimate: MortalityEstimate
    category_count: int
    processing_time_ms: float
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "raw_score": self.raw_score,
            "age_adjusted_score": self.age_adjusted_score,
            "matched_categories": [m.to_dict() for m in self.matched_categories],
            "excluded_categories": [e.to_dict() for e in self.excluded_categories],
            "mortality_estimate": self.mortality_estimate.to_dict(),
            "category_count": self.category_count,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "config": self.config,
        }


# ---------------------------------------------------------------------------
# Hierarchical exclusion pairs
# ---------------------------------------------------------------------------
# When both a mild and severe variant are detected, only the higher-weight
# category should count.  The tuple format is (lower_weight, higher_weight).

_HIERARCHICAL_PAIRS: list[tuple[CCICategory, CCICategory]] = [
    (CCICategory.DIABETES_UNCOMPLICATED, CCICategory.DIABETES_COMPLICATED),
    (CCICategory.MILD_LIVER, CCICategory.MODERATE_SEVERE_LIVER),
    (CCICategory.MALIGNANCY, CCICategory.METASTATIC_TUMOR),
]


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


class CharlsonCalculator:
    """Charlson Comorbidity Index calculator.

    Accepts ICD-10-CM codes and/or free-text clinical narratives and
    computes the standard CCI score with optional age adjustment and
    10-year mortality estimates.

    Examples
    --------
    >>> calc = CharlsonCalculator()
    >>> result = calc.calculate(
    ...     icd_codes=["I21.0", "E11.9", "N18.3"],
    ...     config=CharlsonConfig(age_adjust=True, patient_age=72),
    ... )
    >>> result.raw_score
    4
    >>> result.age_adjusted_score
    7
    """

    def __init__(self) -> None:
        """Initialise the calculator with pre-built lookup tables."""
        self._prefix_lookup = _PREFIX_LOOKUP
        self._sorted_prefixes = _SORTED_PREFIXES
        self._text_patterns = _TEXT_PATTERNS

    # -- Public API --------------------------------------------------------

    def calculate(
        self,
        icd_codes: list[str] | None = None,
        text: str | None = None,
        config: CharlsonConfig | None = None,
    ) -> CCIResult:
        """Calculate the Charlson Comorbidity Index.

        Parameters
        ----------
        icd_codes : list[str] | None
            ICD-10-CM codes (e.g., ``["I21.0", "E11.9"]``).
        text : str | None
            Free-text clinical narrative to scan for comorbidities.
        config : CharlsonConfig | None
            Calculation configuration; defaults to ``CharlsonConfig()``.

        Returns
        -------
        CCIResult
            Complete CCI calculation result.

        Raises
        ------
        ValueError
            If neither ``icd_codes`` nor ``text`` is provided.
        """
        if not icd_codes and not text:
            msg = "At least one of icd_codes or text must be provided"
            raise ValueError(msg)

        if config is None:
            config = CharlsonConfig()

        start = time.perf_counter()

        # Collect all matches keyed by category for deduplication.
        category_matches: dict[CCICategory, ComorbidityMatch] = {}

        # Phase 1: ICD-10-CM code matching (highest confidence).
        if icd_codes:
            for code in icd_codes:
                match = self._match_icd_code(code)
                if match is not None:
                    existing = category_matches.get(match.category)
                    if existing is None or match.confidence > existing.confidence:
                        category_matches[match.category] = match

        # Phase 2: Text-based extraction (supplementary).
        if text and config.include_text_extraction:
            text_matches = self._extract_from_text(text, config.text_confidence_threshold)
            for match in text_matches:
                # Only add text match if no ICD-coded match exists for this
                # category — ICD codes are authoritative.
                if match.category not in category_matches:
                    category_matches[match.category] = match

        # Phase 3: Hierarchical exclusion.
        excluded: list[ComorbidityMatch] = []
        if config.hierarchical_exclusion:
            for lower, higher in _HIERARCHICAL_PAIRS:
                if lower in category_matches and higher in category_matches:
                    excluded.append(category_matches.pop(lower))

        # Phase 4: Compute raw score.
        raw_score = sum(m.weight for m in category_matches.values())

        # Phase 5: Age adjustment (Charlson–Deyo).
        age_adjusted: int | None = None
        if config.age_adjust and config.patient_age is not None:
            age_points = self._compute_age_points(config.patient_age)
            age_adjusted = raw_score + age_points

        # Phase 6: Mortality estimation (uses the best available score).
        effective_score = age_adjusted if age_adjusted is not None else raw_score
        mortality = self._estimate_mortality(effective_score)

        elapsed_ms = (time.perf_counter() - start) * 1000

        matched_list = sorted(
            category_matches.values(),
            key=lambda m: (-m.weight, m.category.value),
        )

        config_dict: dict[str, Any] = {
            "age_adjust": config.age_adjust,
            "patient_age": config.patient_age,
            "include_text_extraction": config.include_text_extraction,
            "text_confidence_threshold": config.text_confidence_threshold,
            "hierarchical_exclusion": config.hierarchical_exclusion,
        }

        return CCIResult(
            raw_score=raw_score,
            age_adjusted_score=age_adjusted,
            matched_categories=matched_list,
            excluded_categories=excluded,
            mortality_estimate=mortality,
            category_count=len(matched_list),
            processing_time_ms=elapsed_ms,
            config=config_dict,
        )

    def match_codes(self, icd_codes: list[str]) -> list[ComorbidityMatch]:
        """Match a list of ICD-10-CM codes to CCI categories.

        Parameters
        ----------
        icd_codes : list[str]
            ICD-10-CM codes to match.

        Returns
        -------
        list[ComorbidityMatch]
            Matched comorbidity conditions (may contain duplicate categories
            if multiple codes map to the same category).
        """
        matches: list[ComorbidityMatch] = []
        for code in icd_codes:
            match = self._match_icd_code(code)
            if match is not None:
                matches.append(match)
        return matches

    def extract_from_text(self, text: str, min_confidence: float = 0.70) -> list[ComorbidityMatch]:
        """Extract comorbidity conditions from free-text clinical narrative.

        Parameters
        ----------
        text : str
            Clinical narrative text.
        min_confidence : float
            Minimum confidence threshold for matches.

        Returns
        -------
        list[ComorbidityMatch]
            Text-extracted comorbidity conditions.
        """
        return self._extract_from_text(text, min_confidence)

    # -- Internal methods --------------------------------------------------

    def _match_icd_code(self, code: str) -> ComorbidityMatch | None:
        """Match a single ICD-10-CM code to a CCI category.

        Uses longest-prefix-first matching so that more specific prefixes
        take priority (e.g., ``E112`` matches diabetes_complicated before
        ``E11`` could match diabetes_uncomplicated if it were a prefix).

        Parameters
        ----------
        code : str
            An ICD-10-CM code (e.g., ``"I21.0"``).

        Returns
        -------
        ComorbidityMatch | None
            Matched condition, or None if code doesn't map to any CCI category.
        """
        normalised = code.upper().replace(".", "").strip()
        if not normalised:
            return None

        for prefix in self._sorted_prefixes:
            if normalised.startswith(prefix):
                category = self._prefix_lookup[prefix]
                return ComorbidityMatch(
                    category=category,
                    weight=CATEGORY_WEIGHTS[category],
                    source="icd_code",
                    evidence=code,
                    confidence=1.0,
                    description=CATEGORY_DESCRIPTIONS[category],
                )
        return None

    def _extract_from_text(
        self, text: str, min_confidence: float
    ) -> list[ComorbidityMatch]:
        """Extract comorbidities from free-text using keyword patterns.

        Confidence is assigned based on pattern specificity:
        - Multi-word medical terms (e.g., "myocardial infarction"): 0.85
        - Short abbreviations (e.g., "MI", "CHF"): 0.75
        - Generic terms (e.g., "cancer"): 0.70

        Parameters
        ----------
        text : str
            Clinical narrative text.
        min_confidence : float
            Minimum confidence to include a match.

        Returns
        -------
        list[ComorbidityMatch]
            Matched conditions from text analysis.
        """
        matches: list[ComorbidityMatch] = []
        seen_categories: set[CCICategory] = set()

        for category, patterns in self._text_patterns.items():
            if category in seen_categories:
                continue

            for pattern in patterns:
                m = pattern.search(text)
                if m:
                    matched_text = m.group(0)
                    confidence = self._text_match_confidence(matched_text)
                    if confidence >= min_confidence:
                        matches.append(ComorbidityMatch(
                            category=category,
                            weight=CATEGORY_WEIGHTS[category],
                            source="text",
                            evidence=matched_text,
                            confidence=confidence,
                            description=CATEGORY_DESCRIPTIONS[category],
                        ))
                        seen_categories.add(category)
                    break  # One match per category is enough.

        return matches

    @staticmethod
    def _text_match_confidence(matched_text: str) -> float:
        """Estimate confidence based on matched text characteristics.

        Longer, more specific medical phrases get higher confidence
        than short abbreviations that could be ambiguous.

        Parameters
        ----------
        matched_text : str
            The actual text that matched a regex pattern.

        Returns
        -------
        float
            Confidence score between 0.70 and 0.85.
        """
        word_count = len(matched_text.split())
        if word_count >= 3:
            return 0.85  # Multi-word medical terms are specific.
        if word_count == 2:
            return 0.80
        if len(matched_text) >= 4:
            return 0.75  # Longer abbreviations (COPD, AIDS, etc.).
        return 0.70  # Short abbreviations (MI, RA, etc.).

    @staticmethod
    def _compute_age_points(age: int) -> int:
        """Compute age adjustment points per Charlson–Deyo.

        Parameters
        ----------
        age : int
            Patient age in years.

        Returns
        -------
        int
            Age points (0–4): 1 point per decade above 40.
        """
        if age < 40:
            return 0
        # 40–49 → 0 in some variants, but Charlson-Deyo standard is:
        # 50–59 → 1, 60–69 → 2, 70–79 → 3, ≥80 → 4
        # Using the 50+ variant which is most widely adopted.
        if age < 50:
            return 0
        decades = (age - 50) // 10 + 1
        return min(decades, 4)

    @staticmethod
    def _estimate_mortality(cci_score: int) -> MortalityEstimate:
        """Estimate 10-year mortality from CCI score.

        Uses Charlson's original formula:
            estimated_mortality = 1 − 0.983^(e^(CCI × 0.9))

        Parameters
        ----------
        cci_score : int
            Charlson Comorbidity Index score (raw or age-adjusted).

        Returns
        -------
        MortalityEstimate
            Mortality and survival estimates with risk grouping.
        """
        import math

        # Charlson survival formula.
        exponent = math.exp(cci_score * 0.9)
        survival = 0.983 ** exponent
        mortality = 1.0 - survival

        # Clamp to [0, 1] for numerical safety.
        mortality = max(0.0, min(1.0, mortality))
        survival = max(0.0, min(1.0, survival))

        # Risk group classification based on total score.
        if cci_score == 0:
            risk_group = "low"
        elif cci_score <= 2:
            risk_group = "mild"
        elif cci_score <= 4:
            risk_group = "moderate"
        else:
            risk_group = "severe"

        return MortalityEstimate(
            ten_year_mortality=mortality,
            ten_year_survival=survival,
            risk_group=risk_group,
        )

    @staticmethod
    def get_category_info() -> list[dict[str, Any]]:
        """Return information about all 17 CCI categories.

        Returns
        -------
        list[dict[str, Any]]
            Category name, weight, and description for each category.
        """
        return [
            {
                "category": cat.value,
                "weight": CATEGORY_WEIGHTS[cat],
                "description": CATEGORY_DESCRIPTIONS[cat],
            }
            for cat in CCICategory
        ]
