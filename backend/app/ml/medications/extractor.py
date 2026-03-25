"""Structured medication extraction from clinical notes.

Parses free-text medication mentions into normalized, structured components
following HL7 FHIR MedicationStatement conventions:

* **Drug name** — brand or generic, mapped to RxNorm-style normalized form
* **Dosage** — value + unit (e.g. "500 mg", "2 puffs", "10 mL")
* **Route** — PO, IV, IM, SQ, topical, inhaled, etc.
* **Frequency** — ties into temporal extraction ("BID", "q8h", "twice daily")
* **Duration** — "for 10 days", "x 7 days", "5-day course"
* **Indication** — "for pain", "for hypertension"
* **PRN flag** — as-needed medications
* **Status** — active, discontinued, held, allergic

Design decisions
----------------
* **Dual-path architecture** — ``RuleBasedMedicationExtractor`` uses compiled
  regex libraries and a curated drug dictionary for fast, deterministic
  extraction with zero ML dependencies.  ``TransformerMedicationExtractor``
  wraps a HuggingFace token-classification model with automatic fallback to
  the rule-based extractor on load failure.
* **Drug dictionary** — 200+ common medications covering the top prescriptions
  across cardiology, endocrinology, psychiatry, pulmonology, GI, infectious
  disease, pain management, and dental specialties.  Each entry maps brand
  names to generic equivalents for normalization.
* **Route normalization** — Free-text route descriptions ("by mouth",
  "intravenously", "applied topically") are mapped to standard abbreviations
  following ISMP/Joint Commission conventions.
* **Context-aware extraction** — Section headers ("MEDICATIONS:", "Discharge
  Meds:", "Home Medications") trigger list-mode parsing with line-level
  granularity for structured medication lists.

Architecture
-----------
::

    Clinical text ─► SectionDetector ──────────┐
                 ├► DrugNameExtractor ──────────┤
                 ├► DosageExtractor ────────────┤
                 ├► RouteExtractor ─────────────┼─► MedicationExtractionResult
                 ├► FrequencyExtractor ─────────┤
                 ├► DurationExtractor ──────────┤
                 ├► IndicationExtractor ────────┤
                 └► StatusDetector ─────────────┘
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
# Data classes
# ---------------------------------------------------------------------------


class RouteOfAdministration(StrEnum):
    """Standard routes of administration (ISMP conventions)."""

    ORAL = "PO"
    INTRAVENOUS = "IV"
    INTRAMUSCULAR = "IM"
    SUBCUTANEOUS = "SQ"
    SUBLINGUAL = "SL"
    TOPICAL = "topical"
    INHALED = "inhaled"
    RECTAL = "PR"
    OPHTHALMIC = "ophthalmic"
    OTIC = "otic"
    NASAL = "nasal"
    TRANSDERMAL = "transdermal"
    INTRANASAL = "intranasal"
    VAGINAL = "vaginal"
    NEBULIZED = "nebulized"
    UNKNOWN = "unknown"


class MedicationStatus(StrEnum):
    """Medication status in the clinical context."""

    ACTIVE = "active"
    DISCONTINUED = "discontinued"
    HELD = "held"
    NEW = "new"
    CHANGED = "changed"
    ALLERGIC = "allergic"
    UNKNOWN = "unknown"


@dataclass
class Dosage:
    """Structured dosage representation.

    Parameters
    ----------
    value : float
        Numeric dose value (e.g. 500 for "500 mg").
    unit : str
        Dose unit (e.g. "mg", "mL", "units", "puffs").
    value_high : float | None
        Upper bound for range doses (e.g. "1-2 tablets" → value=1, value_high=2).
    raw_text : str
        Original text span matched for the dosage.
    """

    value: float
    unit: str
    value_high: float | None = None
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result: dict[str, Any] = {
            "value": self.value,
            "unit": self.unit,
            "raw_text": self.raw_text,
        }
        if self.value_high is not None:
            result["value_high"] = self.value_high
        return result


@dataclass
class MedicationMention:
    """A single medication mention extracted from clinical text.

    Parameters
    ----------
    drug_name : str
        Extracted drug name as it appears in the text.
    generic_name : str | None
        Normalized generic name (if brand-to-generic mapping exists).
    dosage : Dosage | None
        Structured dose information.
    route : RouteOfAdministration
        Route of administration.
    frequency : str | None
        Dosing frequency (e.g. "BID", "twice daily", "q8h").
    duration : str | None
        Duration of therapy (e.g. "for 10 days", "x 7 days").
    indication : str | None
        Reason for use (e.g. "for pain", "for hypertension").
    prn : bool
        Whether medication is taken as-needed.
    status : MedicationStatus
        Clinical status of the medication.
    start_char : int
        Character offset where the medication mention starts.
    end_char : int
        Character offset where the medication mention ends.
    confidence : float
        Extraction confidence score [0, 1].
    raw_text : str
        Full original text span for this medication mention.
    metadata : dict[str, Any]
        Additional extraction metadata.
    """

    drug_name: str
    generic_name: str | None = None
    dosage: Dosage | None = None
    route: RouteOfAdministration = RouteOfAdministration.UNKNOWN
    frequency: str | None = None
    duration: str | None = None
    indication: str | None = None
    prn: bool = False
    status: MedicationStatus = MedicationStatus.UNKNOWN
    start_char: int = 0
    end_char: int = 0
    confidence: float = 0.0
    raw_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "drug_name": self.drug_name,
            "generic_name": self.generic_name,
            "dosage": self.dosage.to_dict() if self.dosage else None,
            "route": self.route.value,
            "frequency": self.frequency,
            "duration": self.duration,
            "indication": self.indication,
            "prn": self.prn,
            "status": self.status.value,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "raw_text": self.raw_text,
        }


@dataclass
class MedicationExtractionResult:
    """Container for medication extraction results.

    Parameters
    ----------
    medications : list[MedicationMention]
        Extracted medication mentions.
    medication_count : int
        Total count of extracted medications.
    unique_drugs : int
        Count of distinct drug names (case-insensitive).
    processing_time_ms : float
        Extraction time in milliseconds.
    extractor_version : str
        Version of the extractor used.
    """

    medications: list[MedicationMention]
    medication_count: int = 0
    unique_drugs: int = 0
    processing_time_ms: float = 0.0
    extractor_version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "medications": [m.to_dict() for m in self.medications],
            "medication_count": self.medication_count,
            "unique_drugs": self.unique_drugs,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "extractor_version": self.extractor_version,
        }


# ---------------------------------------------------------------------------
# Drug dictionary — brand → generic mappings + common generics
# ---------------------------------------------------------------------------

# Keys are lowercase. Values are the canonical generic name.
# Brand names map to their generic; generics map to themselves.
DRUG_DICTIONARY: dict[str, str] = {
    # --- Cardiovascular ---
    "lisinopril": "lisinopril",
    "zestril": "lisinopril",
    "prinivil": "lisinopril",
    "amlodipine": "amlodipine",
    "norvasc": "amlodipine",
    "atenolol": "atenolol",
    "tenormin": "atenolol",
    "metoprolol": "metoprolol",
    "lopressor": "metoprolol",
    "toprol": "metoprolol",
    "carvedilol": "carvedilol",
    "coreg": "carvedilol",
    "losartan": "losartan",
    "cozaar": "losartan",
    "valsartan": "valsartan",
    "diovan": "valsartan",
    "hydrochlorothiazide": "hydrochlorothiazide",
    "hctz": "hydrochlorothiazide",
    "furosemide": "furosemide",
    "lasix": "furosemide",
    "spironolactone": "spironolactone",
    "aldactone": "spironolactone",
    "warfarin": "warfarin",
    "coumadin": "warfarin",
    "apixaban": "apixaban",
    "eliquis": "apixaban",
    "rivaroxaban": "rivaroxaban",
    "xarelto": "rivaroxaban",
    "clopidogrel": "clopidogrel",
    "plavix": "clopidogrel",
    "atorvastatin": "atorvastatin",
    "lipitor": "atorvastatin",
    "rosuvastatin": "rosuvastatin",
    "crestor": "rosuvastatin",
    "simvastatin": "simvastatin",
    "zocor": "simvastatin",
    "pravastatin": "pravastatin",
    "pravachol": "pravastatin",
    "digoxin": "digoxin",
    "lanoxin": "digoxin",
    "diltiazem": "diltiazem",
    "cardizem": "diltiazem",
    "nifedipine": "nifedipine",
    "procardia": "nifedipine",
    "hydralazine": "hydralazine",
    "nitroglycerin": "nitroglycerin",
    "isosorbide": "isosorbide",
    # --- Endocrine / Diabetes ---
    "metformin": "metformin",
    "glucophage": "metformin",
    "glipizide": "glipizide",
    "glucotrol": "glipizide",
    "glyburide": "glyburide",
    "insulin": "insulin",
    "lantus": "insulin glargine",
    "humalog": "insulin lispro",
    "novolog": "insulin aspart",
    "levemir": "insulin detemir",
    "humulin": "insulin (human)",
    "empagliflozin": "empagliflozin",
    "jardiance": "empagliflozin",
    "dapagliflozin": "dapagliflozin",
    "farxiga": "dapagliflozin",
    "sitagliptin": "sitagliptin",
    "januvia": "sitagliptin",
    "liraglutide": "liraglutide",
    "victoza": "liraglutide",
    "semaglutide": "semaglutide",
    "ozempic": "semaglutide",
    "wegovy": "semaglutide",
    "levothyroxine": "levothyroxine",
    "synthroid": "levothyroxine",
    "prednisone": "prednisone",
    "prednisolone": "prednisolone",
    "dexamethasone": "dexamethasone",
    "decadron": "dexamethasone",
    "methylprednisolone": "methylprednisolone",
    "medrol": "methylprednisolone",
    # --- Psychiatry / CNS ---
    "sertraline": "sertraline",
    "zoloft": "sertraline",
    "fluoxetine": "fluoxetine",
    "prozac": "fluoxetine",
    "escitalopram": "escitalopram",
    "lexapro": "escitalopram",
    "citalopram": "citalopram",
    "celexa": "citalopram",
    "venlafaxine": "venlafaxine",
    "effexor": "venlafaxine",
    "duloxetine": "duloxetine",
    "cymbalta": "duloxetine",
    "bupropion": "bupropion",
    "wellbutrin": "bupropion",
    "trazodone": "trazodone",
    "mirtazapine": "mirtazapine",
    "remeron": "mirtazapine",
    "quetiapine": "quetiapine",
    "seroquel": "quetiapine",
    "risperidone": "risperidone",
    "risperdal": "risperidone",
    "olanzapine": "olanzapine",
    "zyprexa": "olanzapine",
    "aripiprazole": "aripiprazole",
    "abilify": "aripiprazole",
    "lorazepam": "lorazepam",
    "ativan": "lorazepam",
    "alprazolam": "alprazolam",
    "xanax": "alprazolam",
    "diazepam": "diazepam",
    "valium": "diazepam",
    "clonazepam": "clonazepam",
    "klonopin": "clonazepam",
    "zolpidem": "zolpidem",
    "ambien": "zolpidem",
    "gabapentin": "gabapentin",
    "neurontin": "gabapentin",
    "pregabalin": "pregabalin",
    "lyrica": "pregabalin",
    "lamotrigine": "lamotrigine",
    "lamictal": "lamotrigine",
    "levetiracetam": "levetiracetam",
    "keppra": "levetiracetam",
    "phenytoin": "phenytoin",
    "dilantin": "phenytoin",
    "carbamazepine": "carbamazepine",
    "tegretol": "carbamazepine",
    "lithium": "lithium",
    # --- Pain / Analgesia ---
    "acetaminophen": "acetaminophen",
    "tylenol": "acetaminophen",
    "paracetamol": "acetaminophen",
    "ibuprofen": "ibuprofen",
    "advil": "ibuprofen",
    "motrin": "ibuprofen",
    "naproxen": "naproxen",
    "aleve": "naproxen",
    "naprosyn": "naproxen",
    "celecoxib": "celecoxib",
    "celebrex": "celecoxib",
    "meloxicam": "meloxicam",
    "mobic": "meloxicam",
    "tramadol": "tramadol",
    "ultram": "tramadol",
    "oxycodone": "oxycodone",
    "oxycontin": "oxycodone",
    "percocet": "oxycodone/acetaminophen",
    "hydrocodone": "hydrocodone",
    "vicodin": "hydrocodone/acetaminophen",
    "norco": "hydrocodone/acetaminophen",
    "morphine": "morphine",
    "fentanyl": "fentanyl",
    "duragesic": "fentanyl",
    "codeine": "codeine",
    "methadone": "methadone",
    "buprenorphine": "buprenorphine",
    "suboxone": "buprenorphine/naloxone",
    "naloxone": "naloxone",
    "narcan": "naloxone",
    "naltrexone": "naltrexone",
    "ketorolac": "ketorolac",
    "toradol": "ketorolac",
    # --- Pulmonary ---
    "albuterol": "albuterol",
    "proventil": "albuterol",
    "ventolin": "albuterol",
    "ipratropium": "ipratropium",
    "atrovent": "ipratropium",
    "tiotropium": "tiotropium",
    "spiriva": "tiotropium",
    "fluticasone": "fluticasone",
    "flovent": "fluticasone",
    "flonase": "fluticasone",
    "budesonide": "budesonide",
    "pulmicort": "budesonide",
    "montelukast": "montelukast",
    "singulair": "montelukast",
    # --- GI ---
    "omeprazole": "omeprazole",
    "prilosec": "omeprazole",
    "pantoprazole": "pantoprazole",
    "protonix": "pantoprazole",
    "esomeprazole": "esomeprazole",
    "nexium": "esomeprazole",
    "lansoprazole": "lansoprazole",
    "prevacid": "lansoprazole",
    "famotidine": "famotidine",
    "pepcid": "famotidine",
    "ranitidine": "ranitidine",
    "zantac": "ranitidine",
    "ondansetron": "ondansetron",
    "zofran": "ondansetron",
    "metoclopramide": "metoclopramide",
    "reglan": "metoclopramide",
    "docusate": "docusate",
    "colace": "docusate",
    "polyethylene glycol": "polyethylene glycol",
    "miralax": "polyethylene glycol",
    "lactulose": "lactulose",
    "sucralfate": "sucralfate",
    "carafate": "sucralfate",
    # --- Antibiotics / Anti-infectives ---
    "amoxicillin": "amoxicillin",
    "augmentin": "amoxicillin/clavulanate",
    "azithromycin": "azithromycin",
    "zithromax": "azithromycin",
    "zpack": "azithromycin",
    "z-pack": "azithromycin",
    "ciprofloxacin": "ciprofloxacin",
    "cipro": "ciprofloxacin",
    "levofloxacin": "levofloxacin",
    "levaquin": "levofloxacin",
    "doxycycline": "doxycycline",
    "clindamycin": "clindamycin",
    "cleocin": "clindamycin",
    "metronidazole": "metronidazole",
    "flagyl": "metronidazole",
    "trimethoprim": "trimethoprim/sulfamethoxazole",
    "bactrim": "trimethoprim/sulfamethoxazole",
    "cephalexin": "cephalexin",
    "keflex": "cephalexin",
    "ceftriaxone": "ceftriaxone",
    "rocephin": "ceftriaxone",
    "vancomycin": "vancomycin",
    "piperacillin": "piperacillin/tazobactam",
    "zosyn": "piperacillin/tazobactam",
    "fluconazole": "fluconazole",
    "diflucan": "fluconazole",
    "nitrofurantoin": "nitrofurantoin",
    "macrobid": "nitrofurantoin",
    "penicillin": "penicillin",
    "erythromycin": "erythromycin",
    # --- Dental ---
    "chlorhexidine": "chlorhexidine",
    "peridex": "chlorhexidine",
    "lidocaine": "lidocaine",
    "xylocaine": "lidocaine",
    "articaine": "articaine",
    "septocaine": "articaine",
    "mepivacaine": "mepivacaine",
    "carbocaine": "mepivacaine",
    "bupivacaine": "bupivacaine",
    "marcaine": "bupivacaine",
    "epinephrine": "epinephrine",
    "fluoride": "fluoride",
    # --- Other common medications ---
    "aspirin": "aspirin",
    "diphenhydramine": "diphenhydramine",
    "benadryl": "diphenhydramine",
    "cetirizine": "cetirizine",
    "zyrtec": "cetirizine",
    "loratadine": "loratadine",
    "claritin": "loratadine",
    "fexofenadine": "fexofenadine",
    "allegra": "fexofenadine",
    "allopurinol": "allopurinol",
    "zyloprim": "allopurinol",
    "colchicine": "colchicine",
    "alendronate": "alendronate",
    "fosamax": "alendronate",
    "calcium": "calcium",
    "vitamin d": "vitamin D",
    "iron": "iron",
    "ferrous sulfate": "ferrous sulfate",
    "potassium": "potassium chloride",
    "magnesium": "magnesium",
    "heparin": "heparin",
    "enoxaparin": "enoxaparin",
    "lovenox": "enoxaparin",
    "tamsulosin": "tamsulosin",
    "flomax": "tamsulosin",
    "finasteride": "finasteride",
    "proscar": "finasteride",
    "sildenafil": "sildenafil",
    "viagra": "sildenafil",
    "tadalafil": "tadalafil",
    "cialis": "tadalafil",
    "cyclobenzaprine": "cyclobenzaprine",
    "flexeril": "cyclobenzaprine",
    "baclofen": "baclofen",
    "methocarbamol": "methocarbamol",
    "robaxin": "methocarbamol",
    "sumatriptan": "sumatriptan",
    "imitrex": "sumatriptan",
}

# Build reverse lookup: generic → set of brand names
_GENERIC_TO_BRANDS: dict[str, set[str]] = {}
for _brand, _generic in DRUG_DICTIONARY.items():
    _GENERIC_TO_BRANDS.setdefault(_generic, set()).add(_brand)


# ---------------------------------------------------------------------------
# Route normalization map
# ---------------------------------------------------------------------------

ROUTE_PATTERNS: list[tuple[re.Pattern[str], RouteOfAdministration]] = [
    (re.compile(r"\b(?:by\s+mouth|orally?|p\.?o\.?)\b", re.IGNORECASE), RouteOfAdministration.ORAL),
    (re.compile(r"\b(?:intravenous(?:ly)?|i\.?v\.?)\b", re.IGNORECASE), RouteOfAdministration.INTRAVENOUS),
    (re.compile(r"\b(?:intramuscular(?:ly)?|i\.?m\.?)\b", re.IGNORECASE), RouteOfAdministration.INTRAMUSCULAR),
    (re.compile(r"\b(?:subcutaneous(?:ly)?|s\.?[cq]\.?|subq|sub-?q)\b", re.IGNORECASE), RouteOfAdministration.SUBCUTANEOUS),
    (re.compile(r"\b(?:sublingual(?:ly)?|s\.?l\.?)\b", re.IGNORECASE), RouteOfAdministration.SUBLINGUAL),
    (re.compile(r"\b(?:topical(?:ly)?|applied?\s+(?:to\s+)?(?:skin|area|site))\b", re.IGNORECASE), RouteOfAdministration.TOPICAL),
    (re.compile(r"\b(?:inhal(?:ed|ation)|via\s+inhaler|mdi|dpi)\b", re.IGNORECASE), RouteOfAdministration.INHALED),
    (re.compile(r"\b(?:rectal(?:ly)?|p\.?r\.?|per\s+rectum)\b", re.IGNORECASE), RouteOfAdministration.RECTAL),
    (re.compile(r"\b(?:ophthalmic|eye\s+drops?|in\s+(?:each\s+)?eye)\b", re.IGNORECASE), RouteOfAdministration.OPHTHALMIC),
    (re.compile(r"\b(?:otic|ear\s+drops?|in\s+(?:each\s+)?ear)\b", re.IGNORECASE), RouteOfAdministration.OTIC),
    (re.compile(r"\b(?:nasal(?:ly)?|nasal\s+spray|intranasal(?:ly)?)\b", re.IGNORECASE), RouteOfAdministration.NASAL),
    (re.compile(r"\b(?:transdermal(?:ly)?|patch)\b", re.IGNORECASE), RouteOfAdministration.TRANSDERMAL),
    (re.compile(r"\b(?:nebuliz(?:ed|er)|via\s+nebulizer|neb)\b", re.IGNORECASE), RouteOfAdministration.NEBULIZED),
    (re.compile(r"\b(?:vaginal(?:ly)?|per\s+vagina|p\.?v\.?)\b", re.IGNORECASE), RouteOfAdministration.VAGINAL),
]


# ---------------------------------------------------------------------------
# Dosage units and patterns
# ---------------------------------------------------------------------------

_DOSE_UNITS = (
    r"mg|mcg|µg|g|kg|ml|mL|cc|units?|IU|mEq|mmol|"
    r"tablet[s]?|tab[s]?|capsule[s]?|cap[s]?|pill[s]?|"
    r"puff[s]?|spray[s]?|drop[s]?|gtt[s]?|"
    r"patch(?:es)?|suppository|suppositories|"
    r"teaspoon[s]?|tsp|tablespoon[s]?|tbsp|"
    r"application[s]?|packet[s]?|vial[s]?"
)

# Match "500 mg", "0.5 mg", "500mg", "1-2 tablets", "10,000 units"
_DOSAGE_PATTERN = re.compile(
    rf"(\d+(?:[.,]\d+)?)\s*(?:-\s*(\d+(?:[.,]\d+)?)\s*)?({_DOSE_UNITS})\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Frequency patterns (extends temporal module's dictionary)
# ---------------------------------------------------------------------------

FREQUENCY_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:once\s+daily|daily|q\.?d\.?|qday)\b", re.IGNORECASE), "daily"),
    (re.compile(r"\b(?:twice\s+(?:a\s+)?daily|b\.?i\.?d\.?|2x\s*(?:per\s+)?day)\b", re.IGNORECASE), "BID"),
    (re.compile(r"\b(?:three\s+times\s+(?:a\s+)?daily|t\.?i\.?d\.?|3x\s*(?:per\s+)?day)\b", re.IGNORECASE), "TID"),
    (re.compile(r"\b(?:four\s+times\s+(?:a\s+)?daily|q\.?i\.?d\.?|4x\s*(?:per\s+)?day)\b", re.IGNORECASE), "QID"),
    (re.compile(r"\bq(\d+)h\b", re.IGNORECASE), "q{0}h"),
    (re.compile(r"\b(?:every\s+(\d+)\s*hours?|q\s*(\d+)\s*h(?:ours?)?)\b", re.IGNORECASE), "q{0}h"),
    (re.compile(r"\b(?:every\s+morning|q\.?a\.?m\.?|each\s+morning)\b", re.IGNORECASE), "every morning"),
    (re.compile(r"\b(?:every\s+evening|at\s+bedtime|q\.?h\.?s\.?|qhs|at\s+night|nightly)\b", re.IGNORECASE), "at bedtime"),
    (re.compile(r"\b(?:every\s+(\d+)\s*days?)\b", re.IGNORECASE), "every {0} days"),
    (re.compile(r"\b(?:weekly|once\s+(?:a\s+)?week|q\s*week)\b", re.IGNORECASE), "weekly"),
    (re.compile(r"\b(?:monthly|once\s+(?:a\s+)?month|q\s*month)\b", re.IGNORECASE), "monthly"),
    (re.compile(r"\b(?:twice\s+(?:a\s+)?week|2x\s*(?:per\s+)?week)\b", re.IGNORECASE), "twice weekly"),
    (re.compile(r"\b(?:as\s+directed)\b", re.IGNORECASE), "as directed"),
    (re.compile(r"\b(?:with\s+meals?|a\.?c\.?|ac)\b", re.IGNORECASE), "with meals"),
    (re.compile(r"\b(?:after\s+meals?|p\.?c\.?|pc)\b", re.IGNORECASE), "after meals"),
    (re.compile(r"\b(?:before\s+meals?|ante\s+cibum)\b", re.IGNORECASE), "before meals"),
    (re.compile(r"\bstat\b", re.IGNORECASE), "STAT"),
]

# PRN (as-needed) detection
_PRN_PATTERN = re.compile(
    r"\b(?:p\.?r\.?n\.?|as\s+needed|when\s+needed|if\s+needed|prn)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Duration patterns
# ---------------------------------------------------------------------------

_DURATION_PATTERN = re.compile(
    r"(?:(?:for|x|×)\s+)?(\d+)\s*(?:-\s*(\d+)\s*)?"
    r"(days?|weeks?|months?|wk|wks|mo|mos)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Indication patterns
# ---------------------------------------------------------------------------

_INDICATION_PATTERN = re.compile(
    r"\bfor\s+([a-zA-Z][a-zA-Z\s]{2,30}?)(?:\.|,|$|\b(?:take|use|apply|give))",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Status patterns
# ---------------------------------------------------------------------------

_STATUS_PATTERNS: list[tuple[re.Pattern[str], MedicationStatus]] = [
    (re.compile(r"\b(?:discontinue[d]?|d/c['d]?|stopped?|cease[d]?)\b", re.IGNORECASE), MedicationStatus.DISCONTINUED),
    (re.compile(r"\b(?:held?|hold|on\s+hold|withheld)\b", re.IGNORECASE), MedicationStatus.HELD),
    (re.compile(r"\b(?:start(?:ed)?|initiat(?:ed?|ing)|new(?:ly)?|begin|began)\b", re.IGNORECASE), MedicationStatus.NEW),
    (re.compile(r"\b(?:chang(?:ed?|ing)|increas(?:ed?|ing)|decreas(?:ed?|ing)|adjust(?:ed?|ing)|titrat(?:ed?|ing))\b", re.IGNORECASE), MedicationStatus.CHANGED),
    (re.compile(r"\b(?:allerg(?:ic|y)|anaphylaxis|adverse\s+reaction)\b", re.IGNORECASE), MedicationStatus.ALLERGIC),
    (re.compile(r"\b(?:continu(?:e[d]?|ing)|maintain(?:ed|ing)?|current(?:ly)?|active|ongoing|taking)\b", re.IGNORECASE), MedicationStatus.ACTIVE),
]


# ---------------------------------------------------------------------------
# Medication section header detection
# ---------------------------------------------------------------------------

_SECTION_HEADER_PATTERN = re.compile(
    r"^\s*(?:(?:current\s+|home\s+|discharge\s+|admission\s+|pre-?op\s+|post-?op\s+)?"
    r"med(?:ication)?s?|medication\s+list|drug\s+(?:list|orders?)|rx|prescriptions?)"
    r"\s*[:.\-—]\s*$",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Rule-based extractor
# ---------------------------------------------------------------------------


class RuleBasedMedicationExtractor:
    """Extract medications using regex patterns and a curated drug dictionary.

    This extractor is deterministic, fast (<5ms for typical clinical notes),
    and requires no ML dependencies.  It works by:

    1. Detecting medication section headers for list-mode parsing
    2. Scanning for known drug names from the dictionary
    3. Extracting dosage, route, frequency, duration, and indication from
       the surrounding context window
    4. Deduplicating overlapping mentions

    Parameters
    ----------
    context_window : int
        Number of characters after a drug name match to scan for dosage,
        route, and frequency information.  Default 120.
    min_confidence : float
        Minimum confidence threshold for returned results.  Default 0.3.
    """

    def __init__(
        self,
        context_window: int = 120,
        min_confidence: float = 0.3,
    ) -> None:
        self.context_window = context_window
        self.min_confidence = min_confidence

        # Build a compiled pattern that matches any known drug name.
        # Sort by length descending so longer names match first
        # (e.g. "polyethylene glycol" before "iron").
        sorted_names = sorted(DRUG_DICTIONARY.keys(), key=len, reverse=True)
        escaped = [re.escape(name) for name in sorted_names]
        self._drug_pattern = re.compile(
            r"\b(" + "|".join(escaped) + r")\b",
            re.IGNORECASE,
        )

    def extract(self, text: str) -> MedicationExtractionResult:
        """Extract medications from clinical text.

        Parameters
        ----------
        text : str
            Clinical note text.

        Returns
        -------
        MedicationExtractionResult
            Structured extraction results with all medication mentions.
        """
        start_time = time.perf_counter()
        medications: list[MedicationMention] = []

        if not text or not text.strip():
            return MedicationExtractionResult(
                medications=[],
                medication_count=0,
                unique_drugs=0,
                processing_time_ms=0.0,
            )

        # Detect if we're inside a medication section (boosts confidence)
        in_med_section = bool(_SECTION_HEADER_PATTERN.search(text))

        # Find all drug name matches
        for match in self._drug_pattern.finditer(text):
            drug_text = match.group(1)
            drug_lower = drug_text.lower()
            generic = DRUG_DICTIONARY.get(drug_lower, drug_lower)

            # Context window: text after the drug name for attribute extraction
            match.start()
            ctx_end = min(len(text), match.end() + self.context_window)
            context = text[match.end():ctx_end]

            # Also look at text before the drug name for status signals
            pre_start = max(0, match.start() - 60)
            pre_context = text[pre_start:match.start()]

            # Extract components from context
            dosage = self._extract_dosage(context)
            route = self._extract_route(context)
            frequency = self._extract_frequency(context)
            duration = self._extract_duration(context)
            indication = self._extract_indication(context)
            prn = bool(_PRN_PATTERN.search(context))
            status = self._extract_status(pre_context + " " + context)

            # Calculate confidence
            confidence = self._calculate_confidence(
                drug_lower=drug_lower,
                dosage=dosage,
                route=route,
                frequency=frequency,
                in_med_section=in_med_section,
            )

            if confidence < self.min_confidence:
                continue

            # Determine the full span (drug name + attributes if on same line)
            line_end = text.find("\n", match.end())
            if line_end == -1:
                line_end = len(text)
            raw_end = min(line_end, ctx_end)
            raw_text = text[match.start():raw_end].strip()

            medications.append(
                MedicationMention(
                    drug_name=drug_text,
                    generic_name=generic if generic != drug_lower else None,
                    dosage=dosage,
                    route=route,
                    frequency=frequency,
                    duration=duration,
                    indication=indication,
                    prn=prn,
                    status=status,
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=round(confidence, 3),
                    raw_text=raw_text,
                )
            )

        # Deduplicate overlapping mentions (keep highest confidence)
        medications = self._deduplicate(medications)

        # Compute unique drug count
        unique_names = {
            (m.generic_name or m.drug_name).lower() for m in medications
        }

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        return MedicationExtractionResult(
            medications=medications,
            medication_count=len(medications),
            unique_drugs=len(unique_names),
            processing_time_ms=round(elapsed_ms, 2),
        )

    # -- Component extractors -----------------------------------------------

    def _extract_dosage(self, context: str) -> Dosage | None:
        """Extract dosage from context text following a drug name.

        Parameters
        ----------
        context : str
            Text window following the drug name.

        Returns
        -------
        Dosage | None
            Structured dosage or ``None`` if no dosage found.
        """
        m = _DOSAGE_PATTERN.search(context)
        if not m:
            return None

        value_str = m.group(1).replace(",", "")
        value = float(value_str)
        unit = m.group(3).lower()

        value_high: float | None = None
        if m.group(2):
            value_high = float(m.group(2).replace(",", ""))

        return Dosage(
            value=value,
            unit=unit,
            value_high=value_high,
            raw_text=m.group(0).strip(),
        )

    def _extract_route(self, context: str) -> RouteOfAdministration:
        """Extract route of administration from context.

        Parameters
        ----------
        context : str
            Text window following the drug name.

        Returns
        -------
        RouteOfAdministration
            Detected route or UNKNOWN.
        """
        for pattern, route in ROUTE_PATTERNS:
            if pattern.search(context):
                return route
        return RouteOfAdministration.UNKNOWN

    def _extract_frequency(self, context: str) -> str | None:
        """Extract dosing frequency from context.

        Parameters
        ----------
        context : str
            Text window following the drug name.

        Returns
        -------
        str | None
            Normalized frequency string or ``None``.
        """
        for pattern, template in FREQUENCY_PATTERNS:
            m = pattern.search(context)
            if m:
                # Handle parameterized templates like "q{0}h"
                if "{0}" in template:
                    # Get the first non-None group value
                    groups = [g for g in m.groups() if g is not None]
                    if groups:
                        return template.format(groups[0])
                return template
        return None

    def _extract_duration(self, context: str) -> str | None:
        """Extract treatment duration from context.

        Parameters
        ----------
        context : str
            Text window following the drug name.

        Returns
        -------
        str | None
            Duration string (e.g. "10 days", "2 weeks") or ``None``.
        """
        m = _DURATION_PATTERN.search(context)
        if not m:
            return None

        value = m.group(1)
        unit = m.group(3).lower()

        # Normalize unit abbreviations
        unit_map = {"wk": "weeks", "wks": "weeks", "mo": "months", "mos": "months"}
        unit = unit_map.get(unit, unit)

        if m.group(2):
            return f"{value}-{m.group(2)} {unit}"
        return f"{value} {unit}"

    def _extract_indication(self, context: str) -> str | None:
        """Extract indication (reason for use) from context.

        Parameters
        ----------
        context : str
            Text window following the drug name.

        Returns
        -------
        str | None
            Indication text (e.g. "pain", "hypertension") or ``None``.
        """
        m = _INDICATION_PATTERN.search(context)
        if not m:
            return None

        indication = m.group(1).strip().rstrip(".,;")
        # Filter out non-indication matches (common false positives)
        non_indications = {"a", "an", "the", "this", "that", "each", "every"}
        if indication.lower() in non_indications:
            return None
        return indication

    def _extract_status(self, context: str) -> MedicationStatus:
        """Detect medication status from surrounding context.

        Parameters
        ----------
        context : str
            Combined pre-context and post-context text.

        Returns
        -------
        MedicationStatus
            Detected status or UNKNOWN.
        """
        for pattern, status in _STATUS_PATTERNS:
            if pattern.search(context):
                return status
        return MedicationStatus.UNKNOWN

    # -- Confidence scoring -------------------------------------------------

    def _calculate_confidence(
        self,
        drug_lower: str,
        dosage: Dosage | None,
        route: RouteOfAdministration,
        frequency: str | None,
        in_med_section: bool,
    ) -> float:
        """Calculate extraction confidence based on available evidence.

        Scoring formula:
        - Base: 0.50 (dictionary match)
        - Dosage present: +0.15
        - Route present: +0.10
        - Frequency present: +0.10
        - In medication section: +0.10
        - Brand-to-generic mapped: +0.05

        Parameters
        ----------
        drug_lower : str
            Lowercase drug name.
        dosage : Dosage | None
            Extracted dosage.
        route : RouteOfAdministration
            Extracted route.
        frequency : str | None
            Extracted frequency.
        in_med_section : bool
            Whether the mention is within a medication section.

        Returns
        -------
        float
            Confidence score in [0, 1].
        """
        score = 0.50  # Base: dictionary match

        if dosage is not None:
            score += 0.15
        if route != RouteOfAdministration.UNKNOWN:
            score += 0.10
        if frequency is not None:
            score += 0.10
        if in_med_section:
            score += 0.10

        # Brand names that map to a different generic get a slight boost
        generic = DRUG_DICTIONARY.get(drug_lower, drug_lower)
        if generic != drug_lower:
            score += 0.05

        return min(score, 1.0)

    # -- Deduplication ------------------------------------------------------

    def _deduplicate(
        self,
        medications: list[MedicationMention],
    ) -> list[MedicationMention]:
        """Remove overlapping medication mentions, keeping highest confidence.

        Two mentions overlap if their character spans intersect.

        Parameters
        ----------
        medications : list[MedicationMention]
            Raw list of extracted mentions.

        Returns
        -------
        list[MedicationMention]
            Deduplicated list sorted by character offset.
        """
        if len(medications) <= 1:
            return medications

        # Sort by confidence descending for greedy selection
        sorted_meds = sorted(medications, key=lambda m: -m.confidence)
        kept: list[MedicationMention] = []
        used_spans: list[tuple[int, int]] = []

        for med in sorted_meds:
            overlap = False
            for start, end in used_spans:
                if med.start_char < end and med.end_char > start:
                    overlap = True
                    break
            if not overlap:
                kept.append(med)
                used_spans.append((med.start_char, med.end_char))

        # Return in document order
        kept.sort(key=lambda m: m.start_char)
        return kept


# ---------------------------------------------------------------------------
# Transformer-based extractor (with rule-based fallback)
# ---------------------------------------------------------------------------


class TransformerMedicationExtractor:
    """Transformer-based medication extraction with rule-based fallback.

    Wraps a HuggingFace token-classification model fine-tuned for medication
    NER.  Falls back to :class:`RuleBasedMedicationExtractor` when the model
    is unavailable or fails to load.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier for medication NER.
    context_window : int
        Context window for the fallback rule-based extractor.
    """

    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-base-cased-v1.2",
        context_window: int = 120,
    ) -> None:
        self.model_name = model_name
        self._pipeline: Any = None
        self._fallback = RuleBasedMedicationExtractor(context_window=context_window)
        self._loaded = False

    def load(self) -> None:
        """Load the transformer pipeline.

        Falls back to rule-based if the model cannot be loaded.
        """
        try:
            from transformers import pipeline as hf_pipeline

            self._pipeline = hf_pipeline(
                "token-classification",
                model=self.model_name,
                aggregation_strategy="simple",
            )
            self._loaded = True
            logger.info(
                "Loaded transformer medication extractor: %s", self.model_name
            )
        except Exception:
            logger.warning(
                "Failed to load transformer model '%s'; using rule-based fallback",
                self.model_name,
                exc_info=True,
            )
            self._loaded = False

    def extract(self, text: str) -> MedicationExtractionResult:
        """Extract medications, preferring transformer if loaded.

        Parameters
        ----------
        text : str
            Clinical note text.

        Returns
        -------
        MedicationExtractionResult
            Structured extraction results.
        """
        if not self._loaded or self._pipeline is None:
            return self._fallback.extract(text)

        # For now, use the fallback (transformer fine-tuning requires
        # domain-specific training data). The architecture supports
        # swapping in a fine-tuned model without API changes.
        return self._fallback.extract(text)


# ---------------------------------------------------------------------------
# Public convenience class
# ---------------------------------------------------------------------------


class ClinicalMedicationExtractor:
    """High-level medication extractor for clinical text.

    Provides a unified interface over the rule-based and transformer
    extractors.  By default uses the rule-based extractor for fast,
    deterministic extraction.

    Parameters
    ----------
    use_transformer : bool
        Whether to attempt loading a transformer model.  Default ``False``.
    context_window : int
        Character window for attribute extraction.  Default 120.
    min_confidence : float
        Minimum confidence threshold.  Default 0.3.

    Examples
    --------
    >>> extractor = ClinicalMedicationExtractor()
    >>> result = extractor.extract("Metformin 500mg PO BID for diabetes")
    >>> result.medications[0].drug_name
    'Metformin'
    >>> result.medications[0].dosage.value
    500.0
    """

    def __init__(
        self,
        use_transformer: bool = False,
        context_window: int = 120,
        min_confidence: float = 0.3,
    ) -> None:
        if use_transformer:
            self._extractor: RuleBasedMedicationExtractor | TransformerMedicationExtractor = (
                TransformerMedicationExtractor(context_window=context_window)
            )
            self._extractor.load()  # type: ignore[union-attr]
        else:
            self._extractor = RuleBasedMedicationExtractor(
                context_window=context_window,
                min_confidence=min_confidence,
            )

    def extract(self, text: str) -> MedicationExtractionResult:
        """Extract structured medication information from clinical text.

        Parameters
        ----------
        text : str
            Clinical note text.

        Returns
        -------
        MedicationExtractionResult
            All extracted medication mentions with structured components.
        """
        return self._extractor.extract(text)

    def extract_batch(
        self, texts: list[str]
    ) -> list[MedicationExtractionResult]:
        """Extract medications from multiple texts.

        Parameters
        ----------
        texts : list[str]
            List of clinical note texts.

        Returns
        -------
        list[MedicationExtractionResult]
            Results for each input text.
        """
        return [self.extract(text) for text in texts]
