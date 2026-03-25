"""Social Determinants of Health (SDoH) extractor.

Implements a rule-based extraction engine that identifies social and
behavioural risk factors from unstructured clinical narrative.  The
extractor operates across eight SDoH domains aligned with the Healthy
People 2030 framework and maps findings to ICD-10-CM Z-codes (Z55–Z65)
for structured documentation.

Design decisions
----------------
* **Rule-based first**: Clinical SDoH documentation varies wildly in
  style (social history sections, intake forms, nursing notes).  A
  pattern-library approach with curated triggers provides high precision
  on common phrasings without requiring labelled training data.
* **Sentiment-aware extraction**: Social factors can be *protective*
  ("lives with supportive spouse") or *adverse* ("currently homeless").
  Tagging sentiment enables downstream risk stratification.
* **Section-aware boosting**: SDoH mentions inside a "Social History"
  section receive a confidence boost because false positives are less
  likely in that context.
* **Negation handling**: Patterns preceded by negation cues ("denies",
  "no history of") are tagged with POSITIVE sentiment (protective) or
  suppressed depending on context, preventing false adverse findings.
* **Z-code mapping**: Each domain maps to one or more ICD-10-CM Z-codes
  so downstream coding modules can auto-suggest social-determinant
  documentation codes (e.g., Z59.0 Homelessness).

References
----------
1. Healthy People 2030: Social Determinants of Health.
   https://health.gov/healthypeople/priority-areas/social-determinants-health
2. WHO Commission on Social Determinants of Health (2008).
3. ICD-10-CM Z55-Z65: Persons with potential health hazards related to
   socioeconomic and psychosocial circumstances.
4. Patra BG, et al. "Extracting social determinants of health from
   electronic health records using NLP." J Am Med Inform Assoc. 2021.
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


class SDoHDomain(StrEnum):
    """Eight social determinant domains aligned with Healthy People 2030.

    Values are lowercase identifiers used in API responses and Z-code
    mapping tables.
    """

    HOUSING = "housing"
    EMPLOYMENT = "employment"
    EDUCATION = "education"
    FOOD_SECURITY = "food_security"
    TRANSPORTATION = "transportation"
    SOCIAL_SUPPORT = "social_support"
    SUBSTANCE_USE = "substance_use"
    FINANCIAL = "financial"


class SDoHSentiment(StrEnum):
    """Whether the extracted factor is adverse, protective, or neutral.

    * ADVERSE — risk factor present (e.g., "currently homeless")
    * PROTECTIVE — social strength present (e.g., "strong family support")
    * NEUTRAL — informational, unclear direction (e.g., "lives alone")
    """

    ADVERSE = "adverse"
    PROTECTIVE = "protective"
    NEUTRAL = "neutral"


@dataclass
class SDoHExtraction:
    """A single SDoH factor extracted from clinical text.

    Attributes
    ----------
    domain : SDoHDomain
        Which of the 8 SDoH domains this factor belongs to.
    text : str
        The matched substring from the clinical note.
    sentiment : SDoHSentiment
        Whether the factor is adverse, protective, or neutral.
    confidence : float
        Confidence score in [0.0, 1.0].
    z_codes : list[str]
        ICD-10-CM Z-codes associated with this domain finding.
    trigger : str
        The pattern/keyword that triggered this extraction.
    start : int
        Character offset of the match start in the original text.
    end : int
        Character offset of the match end in the original text.
    negated : bool
        Whether the match was preceded by a negation cue.
    section : str
        Section header context, if detected (e.g., "Social History").

    Returns
    -------
    dict[str, Any]
        Serialisable dictionary when ``to_dict()`` is called.
    """

    domain: SDoHDomain
    text: str
    sentiment: SDoHSentiment
    confidence: float
    z_codes: list[str] = field(default_factory=list)
    trigger: str = ""
    start: int = 0
    end: int = 0
    negated: bool = False
    section: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            All fields with enum values converted to their string
            representations.
        """
        return {
            "domain": self.domain.value,
            "text": self.text,
            "sentiment": self.sentiment.value,
            "confidence": round(self.confidence, 4),
            "z_codes": list(self.z_codes),
            "trigger": self.trigger,
            "start": self.start,
            "end": self.end,
            "negated": self.negated,
            "section": self.section,
        }


@dataclass
class SDoHExtractionResult:
    """Container for a complete SDoH extraction run.

    Attributes
    ----------
    extractions : list[SDoHExtraction]
        All SDoH factors found in the text.
    domain_summary : dict[str, int]
        Count of extractions per domain.
    adverse_count : int
        Number of adverse factors detected.
    protective_count : int
        Number of protective factors detected.
    text_length : int
        Character length of the input text.
    processing_time_ms : float
        Wall-clock processing time in milliseconds.
    """

    extractions: list[SDoHExtraction] = field(default_factory=list)
    domain_summary: dict[str, int] = field(default_factory=dict)
    adverse_count: int = 0
    protective_count: int = 0
    text_length: int = 0
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dictionary.

        Returns
        -------
        dict[str, Any]
            Complete extraction result with nested extraction dicts.
        """
        return {
            "extractions": [e.to_dict() for e in self.extractions],
            "domain_summary": dict(self.domain_summary),
            "adverse_count": self.adverse_count,
            "protective_count": self.protective_count,
            "text_length": self.text_length,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Z-code mapping per domain
# ---------------------------------------------------------------------------

#: ICD-10-CM Z-codes for social determinant documentation.
#: Each domain maps to one or more codes from the Z55–Z65 range.
DOMAIN_Z_CODES: dict[SDoHDomain, list[dict[str, str]]] = {
    SDoHDomain.HOUSING: [
        {"code": "Z59.00", "description": "Homelessness, unspecified"},
        {"code": "Z59.01", "description": "Sheltered homelessness"},
        {"code": "Z59.02", "description": "Unsheltered homelessness"},
        {"code": "Z59.1", "description": "Inadequate housing"},
        {"code": "Z59.81", "description": "Housing instability, housed"},
    ],
    SDoHDomain.EMPLOYMENT: [
        {"code": "Z56.0", "description": "Unemployment, unspecified"},
        {"code": "Z56.1", "description": "Change of job"},
        {"code": "Z56.2", "description": "Threat of job loss"},
        {"code": "Z56.9", "description": "Unspecified problems related to employment"},
    ],
    SDoHDomain.EDUCATION: [
        {"code": "Z55.0", "description": "Illiteracy and low-level literacy"},
        {"code": "Z55.9", "description": "Problems related to education and literacy, unspecified"},
    ],
    SDoHDomain.FOOD_SECURITY: [
        {"code": "Z59.41", "description": "Food insecurity"},
        {"code": "Z59.48", "description": "Other specified lack of adequate food"},
    ],
    SDoHDomain.TRANSPORTATION: [
        {"code": "Z59.82", "description": "Transportation insecurity"},
    ],
    SDoHDomain.SOCIAL_SUPPORT: [
        {"code": "Z60.2", "description": "Problems related to living alone"},
        {"code": "Z60.4", "description": "Social exclusion and rejection"},
        {"code": "Z63.0", "description": "Problems in relationship with spouse or partner"},
        {"code": "Z63.4", "description": "Disappearance and death of family member"},
        {"code": "Z65.3", "description": "Problems related to other legal circumstances"},
    ],
    SDoHDomain.SUBSTANCE_USE: [
        {"code": "Z72.0", "description": "Tobacco use"},
        {"code": "Z71.41", "description": "Alcohol abuse counseling and surveillance"},
        {"code": "Z71.51", "description": "Drug abuse counseling and surveillance"},
    ],
    SDoHDomain.FINANCIAL: [
        {"code": "Z59.5", "description": "Extreme poverty"},
        {"code": "Z59.6", "description": "Low income"},
        {"code": "Z59.7", "description": "Insufficient social insurance and welfare support"},
    ],
}


# ---------------------------------------------------------------------------
# Pattern library
# ---------------------------------------------------------------------------

#: Negation cues that flip sentiment or suppress extraction.
NEGATION_CUES: list[re.Pattern[str]] = [
    re.compile(
        r"\b(?:no|not|never|denies|denied|without|negative for|"
        r"no history of|no hx of|does not|has not|hasn't|doesn't|"
        r"none|absent|quit|stopped|former|previously)\b",
        re.IGNORECASE,
    ),
]

#: Section headers that indicate SDoH content is likely.
SOCIAL_HISTORY_HEADERS: re.Pattern[str] = re.compile(
    r"(?:^|\n)\s*(?:SOCIAL\s+HISTORY|Social\s+History|SH:|"
    r"PSYCHOSOCIAL|Psychosocial|SOCIAL\s+HX|Social\s+Hx|"
    r"BEHAVIORAL\s+HEALTH|Behavioral\s+Health|"
    r"SUBSTANCE\s+(?:USE|ABUSE)\s+HISTORY|"
    r"SOCIAL\s+DETERMINANTS)\s*[:\-]?\s*",
    re.IGNORECASE,
)

#: Scope terminators that end a social history section.
SECTION_TERMINATORS: re.Pattern[str] = re.compile(
    r"(?:^|\n)\s*(?:FAMILY\s+HISTORY|Family\s+History|FH:|"
    r"REVIEW\s+OF\s+SYSTEMS|ROS:|PHYSICAL\s+EXAM|PE:|"
    r"ASSESSMENT|A/P:|PLAN|MEDICATIONS|ALLERGIES|"
    r"PAST\s+MEDICAL\s+HISTORY|PMH:)\s*[:\-]?\s*",
    re.IGNORECASE,
)


def _build_pattern_library() -> dict[SDoHDomain, list[dict[str, Any]]]:
    """Build the compiled regex pattern library for all SDoH domains.

    Each entry contains:
    * ``pattern`` — compiled regex for the trigger phrase
    * ``sentiment`` — default sentiment (may be flipped by negation)
    * ``base_confidence`` — confidence before section/negation adjustment
    * ``description`` — human-readable trigger label

    Returns
    -------
    dict[SDoHDomain, list[dict[str, Any]]]
        Mapping from domain to list of trigger dictionaries.
    """
    library: dict[SDoHDomain, list[dict[str, Any]]] = {}

    # --- HOUSING ---
    library[SDoHDomain.HOUSING] = [
        # Adverse
        _trigger(r"(?:currently\s+)?homeless", SDoHSentiment.ADVERSE, 0.92, "homelessness"),
        _trigger(r"lives?\s+(?:on\s+the\s+street|in\s+(?:a\s+)?shelter)", SDoHSentiment.ADVERSE, 0.90, "shelter/street"),
        _trigger(r"housing\s+insecur(?:e|ity)", SDoHSentiment.ADVERSE, 0.90, "housing insecurity"),
        _trigger(r"unstable\s+housing", SDoHSentiment.ADVERSE, 0.88, "unstable housing"),
        _trigger(r"evict(?:ed|ion)", SDoHSentiment.ADVERSE, 0.90, "eviction"),
        _trigger(r"(?:at\s+risk\s+of|facing)\s+(?:eviction|homelessness)", SDoHSentiment.ADVERSE, 0.88, "eviction risk"),
        _trigger(r"couch\s+surf(?:ing|er|s)", SDoHSentiment.ADVERSE, 0.85, "couch surfing"),
        _trigger(r"unsafe\s+(?:living|housing)\s+conditions?", SDoHSentiment.ADVERSE, 0.88, "unsafe housing"),
        _trigger(r"(?:overcrowd|overcrowd)(?:ed|ing)", SDoHSentiment.ADVERSE, 0.82, "overcrowding"),
        _trigger(r"(?:mold|lead\s+paint|pest|rodent|bed\s*bug)", SDoHSentiment.ADVERSE, 0.78, "housing hazard"),
        _trigger(r"transitional\s+housing", SDoHSentiment.NEUTRAL, 0.80, "transitional housing"),
        # Protective
        _trigger(r"stable\s+housing", SDoHSentiment.PROTECTIVE, 0.85, "stable housing"),
        _trigger(r"(?:own|owns)\s+(?:a\s+)?home", SDoHSentiment.PROTECTIVE, 0.82, "homeowner"),
        _trigger(r"lives?\s+with\s+(?:family|spouse|partner|wife|husband)", SDoHSentiment.PROTECTIVE, 0.80, "lives with family"),
    ]

    # --- EMPLOYMENT ---
    library[SDoHDomain.EMPLOYMENT] = [
        _trigger(r"(?:currently\s+)?unemployed", SDoHSentiment.ADVERSE, 0.90, "unemployed"),
        _trigger(r"lost\s+(?:his|her|their)\s+job", SDoHSentiment.ADVERSE, 0.88, "job loss"),
        _trigger(r"laid\s+off", SDoHSentiment.ADVERSE, 0.88, "laid off"),
        _trigger(r"on\s+disability", SDoHSentiment.ADVERSE, 0.85, "on disability"),
        _trigger(r"unable\s+to\s+work", SDoHSentiment.ADVERSE, 0.88, "unable to work"),
        _trigger(r"work(?:place|ing)\s+(?:injury|accident|stress)", SDoHSentiment.ADVERSE, 0.82, "workplace issue"),
        _trigger(r"(?:job|occupational)\s+(?:hazard|exposure)", SDoHSentiment.ADVERSE, 0.80, "occupational hazard"),
        # Protective / neutral
        _trigger(r"(?:currently\s+)?employed", SDoHSentiment.PROTECTIVE, 0.82, "employed"),
        _trigger(r"(?:full|part)[- ]time\s+(?:employ(?:ed|ment)|work|job)", SDoHSentiment.PROTECTIVE, 0.80, "employment type"),
        _trigger(r"retired", SDoHSentiment.NEUTRAL, 0.82, "retired"),
        _trigger(r"works?\s+as\s+(?:a\s+)?[\w\s]+", SDoHSentiment.PROTECTIVE, 0.78, "occupation"),
    ]

    # --- EDUCATION ---
    library[SDoHDomain.EDUCATION] = [
        _trigger(r"(?:low|limited)\s+(?:health\s+)?literacy", SDoHSentiment.ADVERSE, 0.88, "low literacy"),
        _trigger(r"(?:did\s+not|didn'?t)\s+(?:finish|complete)\s+(?:high\s+)?school", SDoHSentiment.ADVERSE, 0.85, "incomplete education"),
        _trigger(r"(?:no|limited)\s+formal\s+education", SDoHSentiment.ADVERSE, 0.85, "no formal education"),
        _trigger(r"(?:difficulty|trouble)\s+(?:reading|understanding)\s+(?:instructions|labels|medications?)", SDoHSentiment.ADVERSE, 0.82, "reading difficulty"),
        _trigger(r"(?:requires?|needs?)\s+(?:an?\s+)?interpreter", SDoHSentiment.ADVERSE, 0.80, "language barrier"),
        _trigger(r"(?:limited|no)\s+English\s+proficiency", SDoHSentiment.ADVERSE, 0.85, "limited English"),
        # Protective
        _trigger(r"(?:college|university)\s+(?:educated|graduate|degree)", SDoHSentiment.PROTECTIVE, 0.78, "college educated"),
        _trigger(r"health\s+literate", SDoHSentiment.PROTECTIVE, 0.80, "health literate"),
    ]

    # --- FOOD SECURITY ---
    library[SDoHDomain.FOOD_SECURITY] = [
        _trigger(r"food\s+insecur(?:e|ity)", SDoHSentiment.ADVERSE, 0.92, "food insecurity"),
        _trigger(r"(?:not\s+enough|inadequate|lack(?:s|ing)?)\s+(?:food|nutrition|meals?)", SDoHSentiment.ADVERSE, 0.88, "inadequate food"),
        _trigger(r"(?:skips?|skipping|miss(?:es|ing)?)\s+meals?", SDoHSentiment.ADVERSE, 0.85, "skipping meals"),
        _trigger(r"(?:cannot|can'?t|unable\s+to)\s+afford\s+(?:food|groceries|meals?)", SDoHSentiment.ADVERSE, 0.90, "cannot afford food"),
        _trigger(r"(?:food\s+(?:bank|pantry|stamps?)|SNAP|WIC|EBT)", SDoHSentiment.ADVERSE, 0.78, "food assistance"),
        _trigger(r"(?:malnourish(?:ed|ment)|underweight\s+due\s+to)", SDoHSentiment.ADVERSE, 0.88, "malnutrition"),
        _trigger(r"food\s+desert", SDoHSentiment.ADVERSE, 0.85, "food desert"),
        # Protective
        _trigger(r"(?:adequate|good|balanced)\s+(?:nutrition|diet)", SDoHSentiment.PROTECTIVE, 0.78, "adequate nutrition"),
    ]

    # --- TRANSPORTATION ---
    library[SDoHDomain.TRANSPORTATION] = [
        _trigger(r"(?:no|lack(?:s|ing)?)\s+(?:reliable\s+)?transportation", SDoHSentiment.ADVERSE, 0.90, "no transportation"),
        _trigger(r"transportation\s+(?:barrier|insecur(?:e|ity)|issue|problem|difficult)", SDoHSentiment.ADVERSE, 0.88, "transportation barrier"),
        _trigger(r"(?:cannot|can'?t|unable\s+to)\s+(?:get\s+to|afford)\s+(?:appointments?|clinic|hospital|pharmacy)", SDoHSentiment.ADVERSE, 0.88, "cannot reach care"),
        _trigger(r"miss(?:es|ed|ing)?\s+appointments?\s+(?:due\s+to|because\s+of)\s+(?:transport|ride|travel)", SDoHSentiment.ADVERSE, 0.85, "missed appointments"),
        _trigger(r"relies?\s+on\s+(?:public\s+transit|bus|others?\s+for\s+rides?)", SDoHSentiment.NEUTRAL, 0.75, "relies on public transit"),
        # Protective
        _trigger(r"(?:has\s+(?:own|reliable)\s+)?(?:car|vehicle|transportation)", SDoHSentiment.PROTECTIVE, 0.75, "has transportation"),
    ]

    # --- SOCIAL SUPPORT ---
    library[SDoHDomain.SOCIAL_SUPPORT] = [
        _trigger(r"(?:socially?\s+)?isolat(?:ed|ion)", SDoHSentiment.ADVERSE, 0.88, "social isolation"),
        _trigger(r"lives?\s+alone", SDoHSentiment.NEUTRAL, 0.75, "lives alone"),
        _trigger(r"(?:no|limited|lack(?:s|ing)?)\s+(?:social\s+)?support", SDoHSentiment.ADVERSE, 0.88, "lack of support"),
        _trigger(r"caregiver\s+(?:burden|burnout|stress|fatigue)", SDoHSentiment.ADVERSE, 0.85, "caregiver burden"),
        _trigger(r"(?:domestic|intimate\s+partner)\s+(?:violence|abuse)", SDoHSentiment.ADVERSE, 0.92, "domestic violence"),
        _trigger(r"(?:elder|child)\s+(?:abuse|neglect)", SDoHSentiment.ADVERSE, 0.92, "abuse/neglect"),
        _trigger(r"(?:recently\s+)?(?:divorced|separated|widowed)", SDoHSentiment.ADVERSE, 0.75, "relationship loss"),
        _trigger(r"(?:incarcerat(?:ed|ion)|recently\s+released\s+from\s+(?:prison|jail|detention))", SDoHSentiment.ADVERSE, 0.85, "incarceration"),
        _trigger(r"(?:undocumented|immigration)\s+(?:status|concern|fear)", SDoHSentiment.ADVERSE, 0.80, "immigration concern"),
        # Protective
        _trigger(r"(?:strong|good|adequate)\s+(?:social\s+|family\s+)?support", SDoHSentiment.PROTECTIVE, 0.85, "good support"),
        _trigger(r"(?:supportive|involved)\s+(?:family|spouse|partner|friends?)", SDoHSentiment.PROTECTIVE, 0.82, "supportive family"),
        _trigger(r"(?:attends?|active\s+in)\s+(?:church|community|support\s+group)", SDoHSentiment.PROTECTIVE, 0.78, "community involvement"),
    ]

    # --- SUBSTANCE USE ---
    library[SDoHDomain.SUBSTANCE_USE] = [
        # Tobacco
        _trigger(r"(?:current(?:ly)?|active)\s+(?:smoker|tobacco\s+use)", SDoHSentiment.ADVERSE, 0.92, "current smoker"),
        _trigger(r"smokes?\s+(?:\d+\s+)?(?:cigarettes?|packs?|ppd|pack[- ]years?)", SDoHSentiment.ADVERSE, 0.90, "smoking quantity"),
        _trigger(r"(?:chewing|smokeless)\s+tobacco", SDoHSentiment.ADVERSE, 0.88, "smokeless tobacco"),
        _trigger(r"vap(?:es?|ing)", SDoHSentiment.ADVERSE, 0.82, "vaping"),
        # Alcohol
        _trigger(r"(?:heavy|excessive|binge)\s+(?:drink(?:er|ing)|alcohol)", SDoHSentiment.ADVERSE, 0.90, "heavy drinking"),
        _trigger(r"alcohol\s+(?:abuse|dependence|use\s+disorder|misuse)", SDoHSentiment.ADVERSE, 0.92, "alcohol use disorder"),
        _trigger(r"(?:drinks?|consumes?)\s+\d+\s+(?:beers?|drinks?|glasses?|bottles?)\s+(?:per|a|daily|weekly)", SDoHSentiment.ADVERSE, 0.82, "alcohol consumption"),
        _trigger(r"CAGE\s+(?:score\s+)?(?:\d|positive)", SDoHSentiment.ADVERSE, 0.85, "CAGE positive"),
        _trigger(r"AUDIT(?:-C)?\s+(?:score\s+)?(?:[4-9]|\d{2,}|positive)", SDoHSentiment.ADVERSE, 0.85, "AUDIT positive"),
        # Illicit drugs
        _trigger(r"(?:illicit|recreational|IV|intravenous)\s+drug\s+use", SDoHSentiment.ADVERSE, 0.92, "illicit drug use"),
        _trigger(r"(?:cocaine|heroin|methamphetamine|meth|fentanyl|opioid)\s+(?:use|abuse|dependence)", SDoHSentiment.ADVERSE, 0.92, "specific drug use"),
        _trigger(r"(?:marijuana|cannabis)\s+(?:use|daily|regularly)", SDoHSentiment.ADVERSE, 0.78, "cannabis use"),
        _trigger(r"substance\s+(?:abuse|use\s+disorder|dependence)", SDoHSentiment.ADVERSE, 0.90, "substance use disorder"),
        _trigger(r"history\s+of\s+(?:drug|substance|opioid)\s+(?:abuse|use|overdose)", SDoHSentiment.ADVERSE, 0.85, "substance history"),
        _trigger(r"needle\s+(?:shar(?:e|ing)|track\s+marks?)", SDoHSentiment.ADVERSE, 0.90, "IV drug signs"),
        # Protective / recovery
        _trigger(r"(?:former|ex)[- ]smoker", SDoHSentiment.PROTECTIVE, 0.82, "former smoker"),
        _trigger(r"quit\s+(?:smoking|tobacco|drinking|alcohol)\s+\d+", SDoHSentiment.PROTECTIVE, 0.85, "quit substance"),
        _trigger(r"(?:sober|sobriety|in\s+recovery|clean)\s+(?:for\s+)?\d+", SDoHSentiment.PROTECTIVE, 0.85, "in recovery"),
        _trigger(r"(?:never|non)[- ]?smoker", SDoHSentiment.PROTECTIVE, 0.90, "never smoker"),
        _trigger(r"(?:denies|no)\s+(?:tobacco|alcohol|drug|substance)\s+use", SDoHSentiment.PROTECTIVE, 0.88, "denies substance use"),
        _trigger(r"social(?:ly)?\s+(?:drink(?:er|s|ing))", SDoHSentiment.NEUTRAL, 0.72, "social drinker"),
    ]

    # --- FINANCIAL ---
    library[SDoHDomain.FINANCIAL] = [
        _trigger(r"(?:financial|economic)\s+(?:strain|hardship|difficult|stress|instability|burden)", SDoHSentiment.ADVERSE, 0.90, "financial strain"),
        _trigger(r"(?:cannot|can'?t|unable\s+to)\s+afford\s+(?:medication|treatment|care|insurance|copay|rent)", SDoHSentiment.ADVERSE, 0.90, "cannot afford care"),
        _trigger(r"(?:uninsured|no\s+(?:health\s+)?insurance|lapsed?\s+(?:coverage|insurance))", SDoHSentiment.ADVERSE, 0.90, "uninsured"),
        _trigger(r"(?:under|un)insured", SDoHSentiment.ADVERSE, 0.88, "underinsured"),
        _trigger(r"medical\s+(?:debt|bills?|bankruptcy)", SDoHSentiment.ADVERSE, 0.88, "medical debt"),
        _trigger(r"(?:medicaid|charity\s+care|sliding\s+(?:scale|fee)|patient\s+assistance)", SDoHSentiment.NEUTRAL, 0.75, "financial assistance"),
        _trigger(r"(?:low|below\s+poverty|fixed)\s+income", SDoHSentiment.ADVERSE, 0.85, "low income"),
        _trigger(r"(?:ration(?:s|ing)?|stretch(?:es|ing)?|split(?:s|ting)?)\s+(?:medication|insulin|pills?)", SDoHSentiment.ADVERSE, 0.92, "medication rationing"),
        _trigger(r"cost[- ]?related\s+(?:non)?adherence", SDoHSentiment.ADVERSE, 0.90, "cost-related nonadherence"),
    ]

    return library


def _trigger(
    pattern: str,
    sentiment: SDoHSentiment,
    base_confidence: float,
    description: str,
) -> dict[str, Any]:
    """Create a single trigger dictionary with a compiled regex.

    Parameters
    ----------
    pattern : str
        Regex pattern string (will be compiled case-insensitive).
    sentiment : SDoHSentiment
        Default sentiment for this trigger.
    base_confidence : float
        Confidence score before section/negation adjustments.
    description : str
        Human-readable label for the trigger.

    Returns
    -------
    dict[str, Any]
        Trigger dictionary with compiled ``pattern`` key.
    """
    return {
        "pattern": re.compile(pattern, re.IGNORECASE),
        "sentiment": sentiment,
        "base_confidence": base_confidence,
        "description": description,
    }


# Build once at import time (immutable after construction).
PATTERN_LIBRARY: dict[SDoHDomain, list[dict[str, Any]]] = _build_pattern_library()


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class ClinicalSDoHExtractor:
    """Rule-based Social Determinants of Health extractor.

    Scans clinical text for SDoH indicators across 8 domains, applies
    section-aware confidence boosting and negation-aware sentiment
    adjustment, and maps findings to ICD-10-CM Z-codes.

    Parameters
    ----------
    min_confidence : float
        Minimum confidence threshold for returned extractions.
    section_boost : float
        Confidence bonus applied when match is inside a Social History
        section.
    negation_window : int
        Maximum characters before a match to search for negation cues.
    context_window : int
        Characters before and after a match to include in context.

    Examples
    --------
    >>> extractor = ClinicalSDoHExtractor()
    >>> result = extractor.extract("Patient is currently homeless, denies tobacco use.")
    >>> len(result.extractions) >= 2
    True
    """

    def __init__(
        self,
        min_confidence: float = 0.50,
        section_boost: float = 0.05,
        negation_window: int = 40,
        context_window: int = 50,
    ) -> None:
        self.min_confidence = min_confidence
        self.section_boost = section_boost
        self.negation_window = negation_window
        self.context_window = context_window
        self._pattern_library = PATTERN_LIBRARY

    def extract(self, text: str) -> SDoHExtractionResult:
        """Extract SDoH factors from clinical text.

        Parameters
        ----------
        text : str
            Raw clinical note or social history section.

        Returns
        -------
        SDoHExtractionResult
            Container with all extractions, domain summary, and timing.
        """
        start_time = time.perf_counter()

        if not text or not text.strip():
            return SDoHExtractionResult(
                text_length=0,
                processing_time_ms=0.0,
            )

        # Detect social history section boundaries for confidence boosting.
        social_sections = self._find_social_sections(text)

        raw_extractions: list[SDoHExtraction] = []

        for domain, triggers in self._pattern_library.items():
            z_code_entries = DOMAIN_Z_CODES.get(domain, [])
            z_codes = [entry["code"] for entry in z_code_entries]

            for trigger_info in triggers:
                pattern: re.Pattern[str] = trigger_info["pattern"]
                default_sentiment: SDoHSentiment = trigger_info["sentiment"]
                base_conf: float = trigger_info["base_confidence"]
                description: str = trigger_info["description"]

                for match in pattern.finditer(text):
                    match_start = match.start()
                    match_end = match.end()
                    matched_text = match.group(0)

                    # Check section context.
                    in_social = self._in_social_section(
                        match_start, social_sections
                    )
                    section_name = "Social History" if in_social else ""

                    # Check negation context.
                    negated = self._check_negation(text, match_start)

                    # Adjust confidence.
                    confidence = base_conf
                    if in_social:
                        confidence = min(confidence + self.section_boost, 1.0)

                    # Adjust sentiment based on negation.
                    sentiment = default_sentiment
                    if negated:
                        if default_sentiment == SDoHSentiment.ADVERSE:
                            # Negated adverse → protective
                            # (e.g., "denies homelessness" → protective)
                            sentiment = SDoHSentiment.PROTECTIVE
                            confidence *= 0.85  # Slight reduction for inferred
                        elif default_sentiment == SDoHSentiment.PROTECTIVE:
                            # Negated protective → adverse
                            # (e.g., "no stable housing" → adverse)
                            sentiment = SDoHSentiment.ADVERSE
                            confidence *= 0.85

                    if confidence < self.min_confidence:
                        continue

                    extraction = SDoHExtraction(
                        domain=domain,
                        text=matched_text,
                        sentiment=sentiment,
                        confidence=confidence,
                        z_codes=z_codes,
                        trigger=description,
                        start=match_start,
                        end=match_end,
                        negated=negated,
                        section=section_name,
                    )
                    raw_extractions.append(extraction)

        # Deduplicate overlapping extractions (keep highest confidence).
        extractions = self._deduplicate(raw_extractions)

        # Sort by confidence descending, then by position.
        extractions.sort(key=lambda e: (-e.confidence, e.start))

        # Build domain summary.
        domain_summary: dict[str, int] = {}
        adverse_count = 0
        protective_count = 0
        for ext in extractions:
            domain_summary[ext.domain.value] = (
                domain_summary.get(ext.domain.value, 0) + 1
            )
            if ext.sentiment == SDoHSentiment.ADVERSE:
                adverse_count += 1
            elif ext.sentiment == SDoHSentiment.PROTECTIVE:
                protective_count += 1

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return SDoHExtractionResult(
            extractions=extractions,
            domain_summary=domain_summary,
            adverse_count=adverse_count,
            protective_count=protective_count,
            text_length=len(text),
            processing_time_ms=elapsed_ms,
        )

    def extract_batch(
        self,
        texts: list[str],
    ) -> list[SDoHExtractionResult]:
        """Extract SDoH factors from multiple clinical texts.

        Parameters
        ----------
        texts : list[str]
            List of clinical note strings.

        Returns
        -------
        list[SDoHExtractionResult]
            One result per input text, in order.
        """
        return [self.extract(text) for text in texts]

    def get_domain_info(self, domain: SDoHDomain) -> dict[str, Any]:
        """Return metadata for a specific SDoH domain.

        Parameters
        ----------
        domain : SDoHDomain
            The domain to describe.

        Returns
        -------
        dict[str, Any]
            Domain name, description, Z-codes, and trigger count.
        """
        descriptions = {
            SDoHDomain.HOUSING: "Housing stability, homelessness, and living conditions",
            SDoHDomain.EMPLOYMENT: "Employment status, occupational hazards, and work ability",
            SDoHDomain.EDUCATION: "Educational attainment, literacy, and language barriers",
            SDoHDomain.FOOD_SECURITY: "Food access, nutrition adequacy, and food assistance",
            SDoHDomain.TRANSPORTATION: "Transportation access and barriers to care",
            SDoHDomain.SOCIAL_SUPPORT: "Social networks, isolation, family dynamics, and safety",
            SDoHDomain.SUBSTANCE_USE: "Tobacco, alcohol, and drug use patterns and recovery",
            SDoHDomain.FINANCIAL: "Financial strain, insurance status, and affordability",
        }
        z_entries = DOMAIN_Z_CODES.get(domain, [])
        triggers = self._pattern_library.get(domain, [])

        return {
            "domain": domain.value,
            "description": descriptions.get(domain, ""),
            "z_codes": z_entries,
            "trigger_count": len(triggers),
            "adverse_triggers": sum(
                1 for t in triggers if t["sentiment"] == SDoHSentiment.ADVERSE
            ),
            "protective_triggers": sum(
                1 for t in triggers if t["sentiment"] == SDoHSentiment.PROTECTIVE
            ),
        }

    def get_all_domains(self) -> list[dict[str, Any]]:
        """Return metadata for all 8 SDoH domains.

        Returns
        -------
        list[dict[str, Any]]
            List of domain info dictionaries.
        """
        return [self.get_domain_info(d) for d in SDoHDomain]

    @staticmethod
    def get_z_codes_for_domain(domain: SDoHDomain) -> list[dict[str, str]]:
        """Return Z-code entries for a specific domain.

        Parameters
        ----------
        domain : SDoHDomain
            The SDoH domain.

        Returns
        -------
        list[dict[str, str]]
            List of {code, description} dictionaries.
        """
        return DOMAIN_Z_CODES.get(domain, [])

    @staticmethod
    def total_trigger_count() -> int:
        """Return total number of trigger patterns across all domains.

        Returns
        -------
        int
            Sum of triggers in the pattern library.
        """
        return sum(len(triggers) for triggers in PATTERN_LIBRARY.values())

    # -------------------------------------------------------------------
    # Internal methods
    # -------------------------------------------------------------------

    def _find_social_sections(
        self, text: str
    ) -> list[tuple[int, int]]:
        """Identify character ranges of social history sections.

        Parameters
        ----------
        text : str
            Full clinical note text.

        Returns
        -------
        list[tuple[int, int]]
            List of (start, end) tuples for social history sections.
        """
        sections: list[tuple[int, int]] = []
        for header_match in SOCIAL_HISTORY_HEADERS.finditer(text):
            start = header_match.end()
            # Find the next section terminator.
            term_match = SECTION_TERMINATORS.search(text, start)
            end = term_match.start() if term_match else len(text)
            sections.append((start, end))
        return sections

    def _in_social_section(
        self, position: int, sections: list[tuple[int, int]]
    ) -> bool:
        """Check whether a character position falls inside a social section.

        Parameters
        ----------
        position : int
            Character offset to check.
        sections : list[tuple[int, int]]
            Social history section boundaries.

        Returns
        -------
        bool
            True if position is within any social section.
        """
        return any(start <= position < end for start, end in sections)

    def _check_negation(self, text: str, match_start: int) -> bool:
        """Check for negation cues in the window preceding a match.

        Parameters
        ----------
        text : str
            Full clinical text.
        match_start : int
            Character offset of the pattern match start.

        Returns
        -------
        bool
            True if a negation cue is found in the preceding window.
        """
        window_start = max(0, match_start - self.negation_window)
        window = text[window_start:match_start]
        return any(cue.search(window) for cue in NEGATION_CUES)

    def _deduplicate(
        self, extractions: list[SDoHExtraction]
    ) -> list[SDoHExtraction]:
        """Remove overlapping extractions, keeping highest confidence.

        Two extractions overlap if their character spans intersect and
        they belong to the same domain.

        Parameters
        ----------
        extractions : list[SDoHExtraction]
            Raw list of extractions (may contain overlaps).

        Returns
        -------
        list[SDoHExtraction]
            Deduplicated list.
        """
        if not extractions:
            return []

        # Sort by confidence descending so we keep the best match.
        sorted_exts = sorted(extractions, key=lambda e: -e.confidence)
        kept: list[SDoHExtraction] = []
        used_spans: list[tuple[SDoHDomain, int, int]] = []

        for ext in sorted_exts:
            overlaps = False
            for domain, s, e in used_spans:
                if ext.domain == domain and ext.start < e and ext.end > s:
                    overlaps = True
                    break
            if not overlaps:
                kept.append(ext)
                used_spans.append((ext.domain, ext.start, ext.end))

        return kept
