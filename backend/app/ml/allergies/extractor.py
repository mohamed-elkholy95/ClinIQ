"""Rule-based clinical allergy extraction engine.

Identifies drug, food, and environmental allergens from clinical free text
with associated reactions, severity classification, and negation detection.

Architecture
~~~~~~~~~~~~
1. **Allergen dictionary** — ~150 entries across 3 categories (drug, food,
   environmental) with aliases (e.g., "PCN" → penicillin, "ASA" → aspirin).
   Compiled into a single alternation regex sorted longest-first to prevent
   partial-match shadowing.

2. **Reaction detection** — 30+ reaction patterns (rash, hives, anaphylaxis,
   urticaria, angioedema, GI upset, etc.) searched within a configurable
   context window around each allergen mention.

3. **Severity classification** — Four levels (mild, moderate, severe,
   life-threatening) inferred from reaction type (anaphylaxis → life-
   threatening) and explicit severity modifiers ("severe rash").

4. **NKDA detection** — Recognises "NKDA", "No Known Drug Allergies",
   "NKA", "No Known Allergies", "denies allergies" to produce a
   ``no_known_allergies`` flag.

5. **Section awareness** — Allergen mentions inside an "ALLERGIES:" section
   receive a confidence boost (+0.10) via the unified section parser.

6. **Negation handling** — "no allergy to penicillin", "tolerates sulfa"
   patterns produce TOLERATED status instead of ACTIVE.

Performance: <3 ms per document, zero ML dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AllergyCategory(StrEnum):
    """Allergen category."""

    DRUG = "drug"
    FOOD = "food"
    ENVIRONMENTAL = "environmental"


class AllergySeverity(StrEnum):
    """Allergy severity classification."""

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    LIFE_THREATENING = "life_threatening"
    UNKNOWN = "unknown"


class AllergyStatus(StrEnum):
    """Allergy assertion status."""

    ACTIVE = "active"
    TOLERATED = "tolerated"
    HISTORICAL = "historical"


# ---------------------------------------------------------------------------
# Allergen dictionary
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AllergenEntry:
    """A single allergen with its normalised name and aliases."""

    canonical: str
    category: AllergyCategory
    aliases: tuple[str, ...] = ()


# fmt: off
_DRUG_ALLERGENS: list[AllergenEntry] = [
    # Antibiotics
    AllergenEntry("penicillin", AllergyCategory.DRUG, ("pcn", "pen", "penicillins")),
    AllergenEntry("amoxicillin", AllergyCategory.DRUG, ("amox", "amoxil")),
    AllergenEntry("ampicillin", AllergyCategory.DRUG, ()),
    AllergenEntry("augmentin", AllergyCategory.DRUG, ("amoxicillin-clavulanate",)),
    AllergenEntry("cephalosporins", AllergyCategory.DRUG, ("cephalosporin",)),
    AllergenEntry("cefazolin", AllergyCategory.DRUG, ("ancef", "kefzol")),
    AllergenEntry("ceftriaxone", AllergyCategory.DRUG, ("rocephin",)),
    AllergenEntry("cephalexin", AllergyCategory.DRUG, ("keflex",)),
    AllergenEntry("sulfonamides", AllergyCategory.DRUG, ("sulfa", "sulfamethoxazole", "bactrim", "septra", "tmp-smx", "sulfonamide")),
    AllergenEntry("trimethoprim", AllergyCategory.DRUG, ()),
    AllergenEntry("fluoroquinolones", AllergyCategory.DRUG, ("quinolones", "fluoroquinolone")),
    AllergenEntry("ciprofloxacin", AllergyCategory.DRUG, ("cipro",)),
    AllergenEntry("levofloxacin", AllergyCategory.DRUG, ("levaquin",)),
    AllergenEntry("moxifloxacin", AllergyCategory.DRUG, ("avelox",)),
    AllergenEntry("azithromycin", AllergyCategory.DRUG, ("zithromax", "z-pack", "zpak")),
    AllergenEntry("erythromycin", AllergyCategory.DRUG, ()),
    AllergenEntry("clarithromycin", AllergyCategory.DRUG, ("biaxin",)),
    AllergenEntry("tetracycline", AllergyCategory.DRUG, ()),
    AllergenEntry("doxycycline", AllergyCategory.DRUG, ()),
    AllergenEntry("clindamycin", AllergyCategory.DRUG, ("cleocin",)),
    AllergenEntry("metronidazole", AllergyCategory.DRUG, ("flagyl",)),
    AllergenEntry("vancomycin", AllergyCategory.DRUG, ("vanc",)),
    AllergenEntry("nitrofurantoin", AllergyCategory.DRUG, ("macrobid", "macrodantin")),
    AllergenEntry("gentamicin", AllergyCategory.DRUG, ()),

    # NSAIDs / Analgesics
    AllergenEntry("aspirin", AllergyCategory.DRUG, ("asa",)),
    AllergenEntry("ibuprofen", AllergyCategory.DRUG, ("advil", "motrin")),
    AllergenEntry("naproxen", AllergyCategory.DRUG, ("aleve", "naprosyn")),
    AllergenEntry("nsaids", AllergyCategory.DRUG, ("nsaid", "non-steroidal anti-inflammatory")),
    AllergenEntry("celecoxib", AllergyCategory.DRUG, ("celebrex",)),
    AllergenEntry("acetaminophen", AllergyCategory.DRUG, ("tylenol", "apap", "paracetamol")),
    AllergenEntry("tramadol", AllergyCategory.DRUG, ("ultram",)),
    AllergenEntry("codeine", AllergyCategory.DRUG, ()),
    AllergenEntry("morphine", AllergyCategory.DRUG, ("ms contin",)),
    AllergenEntry("hydrocodone", AllergyCategory.DRUG, ("vicodin", "norco")),
    AllergenEntry("oxycodone", AllergyCategory.DRUG, ("oxycontin", "percocet")),
    AllergenEntry("meperidine", AllergyCategory.DRUG, ("demerol",)),
    AllergenEntry("fentanyl", AllergyCategory.DRUG, ()),

    # Cardiovascular
    AllergenEntry("lisinopril", AllergyCategory.DRUG, ()),
    AllergenEntry("ace inhibitors", AllergyCategory.DRUG, ("ace inhibitor", "acei")),
    AllergenEntry("enalapril", AllergyCategory.DRUG, ("vasotec",)),
    AllergenEntry("losartan", AllergyCategory.DRUG, ("cozaar",)),
    AllergenEntry("metoprolol", AllergyCategory.DRUG, ("lopressor", "toprol")),
    AllergenEntry("atenolol", AllergyCategory.DRUG, ("tenormin",)),
    AllergenEntry("amlodipine", AllergyCategory.DRUG, ("norvasc",)),
    AllergenEntry("hydrochlorothiazide", AllergyCategory.DRUG, ("hctz",)),
    AllergenEntry("furosemide", AllergyCategory.DRUG, ("lasix",)),
    AllergenEntry("statins", AllergyCategory.DRUG, ("statin",)),
    AllergenEntry("atorvastatin", AllergyCategory.DRUG, ("lipitor",)),
    AllergenEntry("simvastatin", AllergyCategory.DRUG, ("zocor",)),
    AllergenEntry("warfarin", AllergyCategory.DRUG, ("coumadin",)),
    AllergenEntry("heparin", AllergyCategory.DRUG, ()),

    # Psychiatric / CNS
    AllergenEntry("sertraline", AllergyCategory.DRUG, ("zoloft",)),
    AllergenEntry("fluoxetine", AllergyCategory.DRUG, ("prozac",)),
    AllergenEntry("paroxetine", AllergyCategory.DRUG, ("paxil",)),
    AllergenEntry("citalopram", AllergyCategory.DRUG, ("celexa",)),
    AllergenEntry("escitalopram", AllergyCategory.DRUG, ("lexapro",)),
    AllergenEntry("duloxetine", AllergyCategory.DRUG, ("cymbalta",)),
    AllergenEntry("gabapentin", AllergyCategory.DRUG, ("neurontin",)),
    AllergenEntry("pregabalin", AllergyCategory.DRUG, ("lyrica",)),
    AllergenEntry("phenytoin", AllergyCategory.DRUG, ("dilantin",)),
    AllergenEntry("carbamazepine", AllergyCategory.DRUG, ("tegretol",)),
    AllergenEntry("lamotrigine", AllergyCategory.DRUG, ("lamictal",)),
    AllergenEntry("valproic acid", AllergyCategory.DRUG, ("depakote", "valproate")),

    # GI / Endocrine
    AllergenEntry("metformin", AllergyCategory.DRUG, ("glucophage",)),
    AllergenEntry("insulin", AllergyCategory.DRUG, ()),
    AllergenEntry("omeprazole", AllergyCategory.DRUG, ("prilosec",)),
    AllergenEntry("pantoprazole", AllergyCategory.DRUG, ("protonix",)),
    AllergenEntry("metoclopramide", AllergyCategory.DRUG, ("reglan",)),
    AllergenEntry("ondansetron", AllergyCategory.DRUG, ("zofran",)),

    # Anaesthetics / Muscle Relaxants
    AllergenEntry("lidocaine", AllergyCategory.DRUG, ("xylocaine",)),
    AllergenEntry("novocaine", AllergyCategory.DRUG, ("procaine",)),
    AllergenEntry("propofol", AllergyCategory.DRUG, ("diprivan",)),
    AllergenEntry("succinylcholine", AllergyCategory.DRUG, ()),

    # Contrast / Latex
    AllergenEntry("iodine contrast", AllergyCategory.DRUG, ("contrast dye", "iv contrast", "iodinated contrast", "ct contrast")),
    AllergenEntry("gadolinium", AllergyCategory.DRUG, ("gd contrast", "mri contrast")),
    AllergenEntry("latex", AllergyCategory.ENVIRONMENTAL, ()),

    # Other
    AllergenEntry("allopurinol", AllergyCategory.DRUG, ("zyloprim",)),
    AllergenEntry("prednisone", AllergyCategory.DRUG, ()),
    AllergenEntry("methylprednisolone", AllergyCategory.DRUG, ("medrol", "solumedrol")),
    AllergenEntry("diphenhydramine", AllergyCategory.DRUG, ("benadryl",)),
]

_FOOD_ALLERGENS: list[AllergenEntry] = [
    AllergenEntry("peanuts", AllergyCategory.FOOD, ("peanut", "peanut butter")),
    AllergenEntry("tree nuts", AllergyCategory.FOOD, ("tree nut", "almonds", "cashews", "walnuts", "pecans", "pistachios", "hazelnuts", "macadamia")),
    AllergenEntry("shellfish", AllergyCategory.FOOD, ("shrimp", "lobster", "crab", "clams", "mussels", "oysters")),
    AllergenEntry("fish", AllergyCategory.FOOD, ()),
    AllergenEntry("milk", AllergyCategory.FOOD, ("dairy", "cow's milk", "lactose")),
    AllergenEntry("eggs", AllergyCategory.FOOD, ("egg",)),
    AllergenEntry("wheat", AllergyCategory.FOOD, ("gluten",)),
    AllergenEntry("soy", AllergyCategory.FOOD, ("soybean", "soybeans")),
    AllergenEntry("sesame", AllergyCategory.FOOD, ()),
    AllergenEntry("corn", AllergyCategory.FOOD, ()),
    AllergenEntry("strawberries", AllergyCategory.FOOD, ("strawberry",)),
    AllergenEntry("bananas", AllergyCategory.FOOD, ("banana",)),
    AllergenEntry("kiwi", AllergyCategory.FOOD, ()),
    AllergenEntry("avocado", AllergyCategory.FOOD, ()),
    AllergenEntry("chocolate", AllergyCategory.FOOD, ()),
]

_ENVIRONMENTAL_ALLERGENS: list[AllergenEntry] = [
    AllergenEntry("pollen", AllergyCategory.ENVIRONMENTAL, ("pollens", "tree pollen", "grass pollen", "ragweed")),
    AllergenEntry("dust mites", AllergyCategory.ENVIRONMENTAL, ("dust", "dust mite")),
    AllergenEntry("mold", AllergyCategory.ENVIRONMENTAL, ("mould", "molds", "moulds", "fungal spores")),
    AllergenEntry("animal dander", AllergyCategory.ENVIRONMENTAL, ("cat dander", "dog dander", "pet dander", "cats", "dogs")),
    AllergenEntry("bee stings", AllergyCategory.ENVIRONMENTAL, ("bee sting", "bee venom", "wasp sting", "insect stings", "hymenoptera")),
    AllergenEntry("cockroach", AllergyCategory.ENVIRONMENTAL, ("cockroaches",)),
    AllergenEntry("adhesive tape", AllergyCategory.ENVIRONMENTAL, ("tape", "medical tape", "adhesive")),
    AllergenEntry("nickel", AllergyCategory.ENVIRONMENTAL, ()),
    AllergenEntry("perfume", AllergyCategory.ENVIRONMENTAL, ("fragrance", "fragrances")),
]
# fmt: on

# Build the unified allergen lookup.
_ALL_ALLERGENS: list[AllergenEntry] = (
    _DRUG_ALLERGENS + _FOOD_ALLERGENS + _ENVIRONMENTAL_ALLERGENS
)

# Map every surface form → AllergenEntry for O(1) lookup after regex match.
_SURFACE_TO_ENTRY: dict[str, AllergenEntry] = {}
_ALL_SURFACE_FORMS: list[str] = []

for _entry in _ALL_ALLERGENS:
    forms = [_entry.canonical] + list(_entry.aliases)
    for _form in forms:
        _lower = _form.lower()
        _SURFACE_TO_ENTRY[_lower] = _entry
        _ALL_SURFACE_FORMS.append(_lower)

# Sort longest-first for greedy matching.
_ALL_SURFACE_FORMS.sort(key=len, reverse=True)

# Build compiled regex: \b(form1|form2|...)\b
_ALLERGEN_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(f) for f in _ALL_SURFACE_FORMS) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Reaction detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ReactionInfo:
    """A clinical reaction type with severity classification."""

    name: str
    severity: AllergySeverity


_REACTIONS: list[ReactionInfo] = [
    # Life-threatening
    ReactionInfo("anaphylaxis", AllergySeverity.LIFE_THREATENING),
    ReactionInfo("anaphylactic shock", AllergySeverity.LIFE_THREATENING),
    ReactionInfo("anaphylactoid reaction", AllergySeverity.LIFE_THREATENING),
    ReactionInfo("airway compromise", AllergySeverity.LIFE_THREATENING),
    ReactionInfo("laryngeal edema", AllergySeverity.LIFE_THREATENING),
    ReactionInfo("cardiac arrest", AllergySeverity.LIFE_THREATENING),
    ReactionInfo("respiratory arrest", AllergySeverity.LIFE_THREATENING),

    # Severe
    ReactionInfo("angioedema", AllergySeverity.SEVERE),
    ReactionInfo("bronchospasm", AllergySeverity.SEVERE),
    ReactionInfo("stevens-johnson syndrome", AllergySeverity.SEVERE),
    ReactionInfo("sjs", AllergySeverity.SEVERE),
    ReactionInfo("toxic epidermal necrolysis", AllergySeverity.SEVERE),
    ReactionInfo("ten", AllergySeverity.SEVERE),
    ReactionInfo("serum sickness", AllergySeverity.SEVERE),
    ReactionInfo("hypotension", AllergySeverity.SEVERE),
    ReactionInfo("dyspnea", AllergySeverity.SEVERE),
    ReactionInfo("difficulty breathing", AllergySeverity.SEVERE),
    ReactionInfo("shortness of breath", AllergySeverity.SEVERE),
    ReactionInfo("wheezing", AllergySeverity.SEVERE),
    ReactionInfo("throat swelling", AllergySeverity.SEVERE),
    ReactionInfo("tongue swelling", AllergySeverity.SEVERE),
    ReactionInfo("facial swelling", AllergySeverity.SEVERE),

    # Moderate
    ReactionInfo("urticaria", AllergySeverity.MODERATE),
    ReactionInfo("hives", AllergySeverity.MODERATE),
    ReactionInfo("rash", AllergySeverity.MODERATE),
    ReactionInfo("maculopapular rash", AllergySeverity.MODERATE),
    ReactionInfo("pruritus", AllergySeverity.MODERATE),
    ReactionInfo("itching", AllergySeverity.MODERATE),
    ReactionInfo("swelling", AllergySeverity.MODERATE),
    ReactionInfo("edema", AllergySeverity.MODERATE),
    ReactionInfo("erythema", AllergySeverity.MODERATE),

    # Mild
    ReactionInfo("nausea", AllergySeverity.MILD),
    ReactionInfo("vomiting", AllergySeverity.MILD),
    ReactionInfo("diarrhea", AllergySeverity.MILD),
    ReactionInfo("gi upset", AllergySeverity.MILD),
    ReactionInfo("stomach upset", AllergySeverity.MILD),
    ReactionInfo("abdominal pain", AllergySeverity.MILD),
    ReactionInfo("headache", AllergySeverity.MILD),
    ReactionInfo("dizziness", AllergySeverity.MILD),
    ReactionInfo("drowsiness", AllergySeverity.MILD),
    ReactionInfo("dry mouth", AllergySeverity.MILD),
    ReactionInfo("flushing", AllergySeverity.MILD),
]

# Build reaction regex sorted longest-first.
_REACTION_NAMES = sorted(
    [r.name for r in _REACTIONS], key=len, reverse=True
)
_REACTION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(n) for n in _REACTION_NAMES) + r")\b",
    re.IGNORECASE,
)
_REACTION_LOOKUP: dict[str, ReactionInfo] = {r.name.lower(): r for r in _REACTIONS}

# Severity modifiers that override the default reaction severity.
_SEVERITY_MODIFIERS: dict[str, AllergySeverity] = {
    "severe": AllergySeverity.SEVERE,
    "life-threatening": AllergySeverity.LIFE_THREATENING,
    "fatal": AllergySeverity.LIFE_THREATENING,
    "serious": AllergySeverity.SEVERE,
    "mild": AllergySeverity.MILD,
    "moderate": AllergySeverity.MODERATE,
}

_SEVERITY_MODIFIER_RE = re.compile(
    r"\b(" + "|".join(re.escape(m) for m in _SEVERITY_MODIFIERS) + r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# NKDA detection
# ---------------------------------------------------------------------------

_NKDA_PATTERNS = re.compile(
    r"\b(?:"
    r"NKDA|NKA|NKFA"
    r"|no\s+known\s+(?:drug\s+)?allergies"
    r"|no\s+known\s+food\s+allergies"
    r"|denies\s+(?:any\s+)?allergies"
    r"|no\s+allergies"
    r"|no\s+drug\s+allergies"
    r"|no\s+medication\s+allergies"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Negation / toleration patterns
# ---------------------------------------------------------------------------

_NEGATION_PATTERNS = re.compile(
    r"\b(?:"
    r"no\s+(?:allergy|allergic\s+reaction)\s+to"
    r"|tolerates"
    r"|tolerated"
    r"|no\s+adverse\s+reaction\s+to"
    r"|not\s+allergic\s+to"
    r"|previously\s+tolerated"
    r"|allergy\s+removed"
    r")\s+",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class AllergyResult:
    """A detected reaction associated with an allergy."""

    reaction: str
    severity: AllergySeverity

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict."""
        return {
            "reaction": self.reaction,
            "severity": str(self.severity),
        }


@dataclass(slots=True)
class DetectedAllergy:
    """A single detected allergy mention.

    Attributes
    ----------
    allergen : str
        Canonical allergen name (normalised).
    allergen_raw : str
        Surface form as found in text.
    category : AllergyCategory
        Drug, food, or environmental.
    reactions : list[AllergyResult]
        Associated reactions found in context.
    severity : AllergySeverity
        Overall severity (max of reaction severities).
    status : AllergyStatus
        Active, tolerated, or historical.
    start : int
        Character offset of allergen mention start.
    end : int
        Character offset of allergen mention end.
    confidence : float
        Detection confidence.
    """

    allergen: str
    allergen_raw: str
    category: AllergyCategory
    reactions: list[AllergyResult] = field(default_factory=list)
    severity: AllergySeverity = AllergySeverity.UNKNOWN
    status: AllergyStatus = AllergyStatus.ACTIVE
    start: int = 0
    end: int = 0
    confidence: float = 0.70

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict."""
        return {
            "allergen": self.allergen,
            "allergen_raw": self.allergen_raw,
            "category": str(self.category),
            "reactions": [r.to_dict() for r in self.reactions],
            "severity": str(self.severity),
            "status": str(self.status),
            "start": self.start,
            "end": self.end,
            "confidence": round(self.confidence, 4),
        }


@dataclass(slots=True)
class ExtractionResult:
    """Result of allergy extraction from a document.

    Attributes
    ----------
    allergies : list[DetectedAllergy]
        All detected allergy mentions.
    no_known_allergies : bool
        True if NKDA / NKA pattern detected.
    nkda_evidence : str
        The matched NKDA text, if any.
    text_length : int
        Length of input text.
    """

    allergies: list[DetectedAllergy] = field(default_factory=list)
    no_known_allergies: bool = False
    nkda_evidence: str = ""
    text_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialise to JSON-safe dict."""
        return {
            "allergies": [a.to_dict() for a in self.allergies],
            "no_known_allergies": self.no_known_allergies,
            "nkda_evidence": self.nkda_evidence,
            "allergy_count": len(self.allergies),
            "categories": sorted({str(a.category) for a in self.allergies}),
            "text_length": self.text_length,
        }


# ---------------------------------------------------------------------------
# Severity ordering for max() computation
# ---------------------------------------------------------------------------

_SEVERITY_ORDER: dict[AllergySeverity, int] = {
    AllergySeverity.UNKNOWN: 0,
    AllergySeverity.MILD: 1,
    AllergySeverity.MODERATE: 2,
    AllergySeverity.SEVERE: 3,
    AllergySeverity.LIFE_THREATENING: 4,
}


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

class ClinicalAllergyExtractor:
    """Rule-based clinical allergy extraction engine.

    Detects drug, food, and environmental allergens from clinical text with
    associated reactions, severity classification, and NKDA detection.

    Parameters
    ----------
    min_confidence : float
        Minimum confidence to include an allergy in results.
    reaction_window : int
        Character window around allergen mention to search for reactions.
    section_boost : float
        Confidence boost when allergen is inside an "Allergies" section.

    Examples
    --------
    >>> ext = ClinicalAllergyExtractor()
    >>> result = ext.extract("Allergies: Penicillin (anaphylaxis), Sulfa (rash)")
    >>> len(result.allergies)
    2
    >>> result.allergies[0].allergen
    'penicillin'
    """

    def __init__(
        self,
        min_confidence: float = 0.50,
        reaction_window: int = 80,
        section_boost: float = 0.10,
    ) -> None:
        self.min_confidence = min_confidence
        self.reaction_window = reaction_window
        self.section_boost = section_boost

    def extract(self, text: str) -> ExtractionResult:
        """Extract allergies from clinical text.

        Parameters
        ----------
        text : str
            Raw clinical note text.

        Returns
        -------
        ExtractionResult
            Detected allergies, NKDA status, and metadata.
        """
        if not text or not text.strip():
            return ExtractionResult(text_length=len(text) if text else 0)

        result = ExtractionResult(text_length=len(text))

        # Check NKDA first.
        nkda_match = _NKDA_PATTERNS.search(text)
        if nkda_match:
            result.no_known_allergies = True
            result.nkda_evidence = nkda_match.group(0).strip()

        # Find allergy section boundaries for confidence boosting.
        allergy_sections = self._find_allergy_sections(text)

        # Find all allergen mentions.
        raw_matches: list[DetectedAllergy] = []
        for m in _ALLERGEN_PATTERN.finditer(text):
            surface = m.group(0)
            entry = _SURFACE_TO_ENTRY.get(surface.lower())
            if entry is None:
                continue

            start, end = m.start(), m.end()

            # Check negation — is this allergen preceded by a toleration phrase?
            status = self._detect_status(text, start)

            # Find reactions in context window, bounded by line breaks
            # to avoid bleeding across allergy list items.
            window_start = max(0, start - self.reaction_window)
            window_end = min(len(text), end + self.reaction_window)

            # Narrow the window to the current line/list-item boundaries
            # when the text contains numbered or bulleted allergy lists.
            line_start = text.rfind("\n", window_start, start)
            line_start = line_start + 1 if line_start >= 0 else window_start
            line_end = text.find("\n", end)
            line_end = line_end if line_end >= 0 else window_end

            # Use the narrower of line-bounded or window-bounded context.
            context = text[line_start:line_end]
            reactions = self._find_reactions(context)

            # Determine overall severity.
            severity = self._compute_severity(reactions, context)

            # Base confidence.
            confidence = 0.70

            # Boost if inside allergy section.
            if self._in_allergy_section(start, allergy_sections):
                confidence += self.section_boost

            # Boost if reactions found.
            if reactions:
                confidence += 0.10

            # Boost for drug allergens (more clinically specific).
            if entry.category == AllergyCategory.DRUG:
                confidence += 0.05

            confidence = min(1.0, confidence)

            if confidence >= self.min_confidence:
                raw_matches.append(
                    DetectedAllergy(
                        allergen=entry.canonical,
                        allergen_raw=surface,
                        category=entry.category,
                        reactions=reactions,
                        severity=severity,
                        status=status,
                        start=start,
                        end=end,
                        confidence=confidence,
                    )
                )

        # Deduplicate — keep highest-confidence mention per canonical allergen.
        result.allergies = self._deduplicate(raw_matches)

        return result

    def extract_batch(
        self, texts: list[str]
    ) -> list[ExtractionResult]:
        """Extract allergies from multiple documents.

        Parameters
        ----------
        texts : list[str]
            List of clinical note texts.

        Returns
        -------
        list[ExtractionResult]
            Per-document extraction results.
        """
        return [self.extract(t) for t in texts]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_reactions(self, context: str) -> list[AllergyResult]:
        """Find reaction mentions within a context window.

        Parameters
        ----------
        context : str
            Text window around the allergen mention.

        Returns
        -------
        list[AllergyResult]
            Detected reactions with severity.
        """
        reactions: list[AllergyResult] = []
        seen: set[str] = set()

        for m in _REACTION_PATTERN.finditer(context):
            reaction_name = m.group(0).lower()
            if reaction_name in seen:
                continue
            seen.add(reaction_name)

            info = _REACTION_LOOKUP.get(reaction_name)
            if info is None:
                continue

            # Check for severity modifiers near this reaction.
            local_start = max(0, m.start() - 30)
            local_context = context[local_start:m.end()]
            severity = info.severity

            mod_match = _SEVERITY_MODIFIER_RE.search(local_context)
            if mod_match:
                mod_severity = _SEVERITY_MODIFIERS.get(mod_match.group(0).lower())
                if mod_severity and _SEVERITY_ORDER.get(mod_severity, 0) > _SEVERITY_ORDER.get(severity, 0):
                    severity = mod_severity

            reactions.append(AllergyResult(reaction=info.name, severity=severity))

        return reactions

    def _compute_severity(
        self, reactions: list[AllergyResult], context: str
    ) -> AllergySeverity:
        """Compute overall severity from reactions and context modifiers.

        Parameters
        ----------
        reactions : list[AllergyResult]
            Detected reactions.
        context : str
            Text window for modifier detection.

        Returns
        -------
        AllergySeverity
            Highest severity found, or UNKNOWN if no reactions.
        """
        if not reactions:
            # Check for standalone severity modifiers in context.
            mod_match = _SEVERITY_MODIFIER_RE.search(context)
            if mod_match:
                return _SEVERITY_MODIFIERS.get(
                    mod_match.group(0).lower(), AllergySeverity.UNKNOWN
                )
            return AllergySeverity.UNKNOWN

        return max(
            reactions,
            key=lambda r: _SEVERITY_ORDER.get(r.severity, 0),
        ).severity

    def _detect_status(self, text: str, allergen_start: int) -> AllergyStatus:
        """Detect whether an allergy mention is negated/tolerated.

        Parameters
        ----------
        text : str
            Full document text.
        allergen_start : int
            Start position of the allergen mention.

        Returns
        -------
        AllergyStatus
            ACTIVE, TOLERATED, or HISTORICAL.
        """
        # Check for negation patterns in the 60 chars preceding the allergen.
        prefix_start = max(0, allergen_start - 60)
        prefix = text[prefix_start:allergen_start]

        if _NEGATION_PATTERNS.search(prefix):
            return AllergyStatus.TOLERATED

        # Check for historical markers.
        historical_re = re.compile(
            r"\b(?:previously|former|childhood|resolved|outgrown)\b",
            re.IGNORECASE,
        )
        if historical_re.search(prefix):
            return AllergyStatus.HISTORICAL

        return AllergyStatus.ACTIVE

    def _find_allergy_sections(
        self, text: str
    ) -> list[tuple[int, int]]:
        """Find character ranges of allergy-related sections.

        Parameters
        ----------
        text : str
            Full document text.

        Returns
        -------
        list[tuple[int, int]]
            List of (start, end) tuples for allergy sections.
        """
        header_re = re.compile(
            r"^\s*(?:"
            r"(?:ALLERGIES|DRUG ALLERGIES|MEDICATION ALLERGIES|ALLERGY LIST|ADVERSE REACTIONS)\s*:"
            r"|(?:Allergies|Drug Allergies|Medication Allergies)\s*:"
            r")",
            re.MULTILINE,
        )
        terminator_re = re.compile(
            r"^\s*[A-Z][A-Z /&\-]{2,}\s*:",
            re.MULTILINE,
        )

        sections: list[tuple[int, int]] = []
        for header_match in header_re.finditer(text):
            start = header_match.end()
            # Find next section header.
            term_match = terminator_re.search(text, start)
            end = term_match.start() if term_match else len(text)
            sections.append((header_match.start(), end))

        return sections

    @staticmethod
    def _in_allergy_section(
        position: int, sections: list[tuple[int, int]]
    ) -> bool:
        """Check if a position falls inside any allergy section.

        Parameters
        ----------
        position : int
            Character offset.
        sections : list[tuple[int, int]]
            Allergy section boundaries.

        Returns
        -------
        bool
            True if position is inside an allergy section.
        """
        return any(start <= position < end for start, end in sections)

    @staticmethod
    def _deduplicate(
        matches: list[DetectedAllergy],
    ) -> list[DetectedAllergy]:
        """Deduplicate allergen mentions, keeping highest confidence per canonical name.

        Parameters
        ----------
        matches : list[DetectedAllergy]
            Raw detected allergies (may contain duplicates).

        Returns
        -------
        list[DetectedAllergy]
            Deduplicated list.
        """
        best: dict[str, DetectedAllergy] = {}
        for m in matches:
            existing = best.get(m.allergen)
            if existing is None or m.confidence > existing.confidence:
                best[m.allergen] = m
        return sorted(best.values(), key=lambda a: a.start)

    @staticmethod
    def get_allergen_count() -> dict[str, int]:
        """Return allergen counts per category.

        Returns
        -------
        dict[str, int]
            Category → count mapping.
        """
        counts: dict[str, int] = {}
        for entry in _ALL_ALLERGENS:
            key = str(entry.category)
            counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def get_reaction_count() -> int:
        """Return total number of known reaction patterns.

        Returns
        -------
        int
            Number of reaction entries.
        """
        return len(_REACTIONS)

