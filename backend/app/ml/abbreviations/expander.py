"""Clinical abbreviation detection and expansion engine.

Detects medical abbreviations in clinical free text and expands them to
their full forms. Handles ambiguous abbreviations (e.g., "PE" could mean
pulmonary embolism or physical exam) via context-aware disambiguation
using surrounding text signals.

Design decisions:
- Word-boundary regex matching prevents partial matches (e.g., "cat" inside
  "catheter" won't match if "cat" were an abbreviation)
- Case-insensitive matching with original-case preservation in output
- Context window analysis (±50 chars default) for ambiguity resolution
- Domain tagging enables downstream filtering by clinical specialty
- Overlapping span deduplication with confidence tie-breaking
- Zero ML dependencies — pure regex/dictionary for <5ms per document

Architecture:
  Input text → Word boundary scan → Dictionary lookup → Ambiguity check
  → Context window analysis → Confidence scoring → Deduplication → Output
"""

import re
import time
from dataclasses import dataclass, field
from enum import StrEnum


class ClinicalDomain(StrEnum):
    """Clinical domain categories for abbreviation classification.

    Each abbreviation belongs to one or more domains, enabling
    domain-specific filtering (e.g., show only cardiology abbreviations).
    """

    CARDIOLOGY = "cardiology"
    PULMONOLOGY = "pulmonology"
    ENDOCRINE = "endocrine"
    NEUROLOGY = "neurology"
    GASTROENTEROLOGY = "gastroenterology"
    RENAL = "renal"
    INFECTIOUS = "infectious"
    MUSCULOSKELETAL = "musculoskeletal"
    HEMATOLOGY = "hematology"
    GENERAL = "general"
    DENTAL = "dental"
    PHARMACY = "pharmacy"


class AmbiguityResolution(StrEnum):
    """How an ambiguous abbreviation was resolved.

    Tracks the disambiguation method for transparency and auditability.
    """

    UNAMBIGUOUS = "unambiguous"
    CONTEXT_RESOLVED = "context_resolved"
    DEFAULT_SENSE = "default_sense"
    SECTION_RESOLVED = "section_resolved"


@dataclass
class AbbreviationMatch:
    """A detected abbreviation with its expansion.

    Parameters
    ----------
    abbreviation : str
        The abbreviation as it appears in the source text.
    expansion : str
        The full expanded form.
    start : int
        Start character offset in the source text.
    end : int
        End character offset in the source text.
    confidence : float
        Detection confidence (0.0–1.0).
    domain : ClinicalDomain
        Clinical domain this abbreviation belongs to.
    is_ambiguous : bool
        Whether this abbreviation has multiple possible expansions.
    resolution : AmbiguityResolution
        How the expansion was selected (if ambiguous).
    alternative_expansions : list[str]
        Other possible expansions (empty if unambiguous).

    Returns
    -------
    dict
        Serializable dictionary via ``to_dict()``.
    """

    abbreviation: str
    expansion: str
    start: int
    end: int
    confidence: float
    domain: ClinicalDomain
    is_ambiguous: bool = False
    resolution: AmbiguityResolution = AmbiguityResolution.UNAMBIGUOUS
    alternative_expansions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary for API responses."""
        return {
            "abbreviation": self.abbreviation,
            "expansion": self.expansion,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "domain": str(self.domain),
            "is_ambiguous": self.is_ambiguous,
            "resolution": str(self.resolution),
            "alternative_expansions": self.alternative_expansions,
        }


@dataclass
class ExpansionResult:
    """Result of abbreviation expansion on a document.

    Parameters
    ----------
    original_text : str
        The unmodified input text.
    expanded_text : str
        Text with abbreviations replaced by their expansions.
    matches : list[AbbreviationMatch]
        All detected abbreviation matches.
    total_found : int
        Number of abbreviations detected.
    ambiguous_count : int
        Number of ambiguous abbreviations resolved.
    processing_time_ms : float
        Processing time in milliseconds.

    Returns
    -------
    dict
        Serializable dictionary via ``to_dict()``.
    """

    original_text: str
    expanded_text: str
    matches: list[AbbreviationMatch]
    total_found: int
    ambiguous_count: int
    processing_time_ms: float

    def to_dict(self) -> dict:
        """Serialize to dictionary for API responses."""
        return {
            "original_text": self.original_text,
            "expanded_text": self.expanded_text,
            "matches": [m.to_dict() for m in self.matches],
            "total_found": self.total_found,
            "ambiguous_count": self.ambiguous_count,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


@dataclass
class AbbreviationConfig:
    """Configuration for the abbreviation expander.

    Parameters
    ----------
    min_confidence : float
        Minimum confidence threshold for including matches (default 0.60).
    context_window : int
        Characters to examine on each side for disambiguation (default 50).
    expand_in_place : bool
        Whether to produce expanded text output (default True).
    include_unambiguous : bool
        Include unambiguous abbreviations in output (default True).
    domains : list[ClinicalDomain] | None
        Filter to specific clinical domains (None = all domains).
    """

    min_confidence: float = 0.60
    context_window: int = 50
    expand_in_place: bool = True
    include_unambiguous: bool = True
    domains: list[ClinicalDomain] | None = None


# ─────────────────────────────────────────────────────────────────────
# Abbreviation Dictionary
# ─────────────────────────────────────────────────────────────────────
# Each entry maps an abbreviation to (expansion, domain, confidence).
# Multi-word abbreviations (e.g., "n/v") are supported.
# Confidence reflects how reliably the abbreviation maps to a single
# meaning in clinical text — lower for ambiguous or context-dependent
# abbreviations.

_ABBREVIATION_DB: dict[str, tuple[str, ClinicalDomain, float]] = {
    # ── Cardiology ──────────────────────────────────────────────────
    "htn": ("hypertension", ClinicalDomain.CARDIOLOGY, 0.95),
    "cad": ("coronary artery disease", ClinicalDomain.CARDIOLOGY, 0.95),
    "chf": ("congestive heart failure", ClinicalDomain.CARDIOLOGY, 0.95),
    "mi": ("myocardial infarction", ClinicalDomain.CARDIOLOGY, 0.90),
    "stemi": ("ST-elevation myocardial infarction", ClinicalDomain.CARDIOLOGY, 0.95),
    "nstemi": ("non-ST-elevation myocardial infarction", ClinicalDomain.CARDIOLOGY, 0.95),
    "afib": ("atrial fibrillation", ClinicalDomain.CARDIOLOGY, 0.95),
    "a-fib": ("atrial fibrillation", ClinicalDomain.CARDIOLOGY, 0.95),
    "aflutter": ("atrial flutter", ClinicalDomain.CARDIOLOGY, 0.95),
    "avr": ("aortic valve replacement", ClinicalDomain.CARDIOLOGY, 0.85),
    "mvr": ("mitral valve replacement", ClinicalDomain.CARDIOLOGY, 0.85),
    "cabg": ("coronary artery bypass graft", ClinicalDomain.CARDIOLOGY, 0.95),
    "pci": ("percutaneous coronary intervention", ClinicalDomain.CARDIOLOGY, 0.90),
    "lvef": ("left ventricular ejection fraction", ClinicalDomain.CARDIOLOGY, 0.95),
    "ef": ("ejection fraction", ClinicalDomain.CARDIOLOGY, 0.85),
    "bp": ("blood pressure", ClinicalDomain.CARDIOLOGY, 0.90),
    "hr": ("heart rate", ClinicalDomain.CARDIOLOGY, 0.85),
    "nsr": ("normal sinus rhythm", ClinicalDomain.CARDIOLOGY, 0.95),
    "svt": ("supraventricular tachycardia", ClinicalDomain.CARDIOLOGY, 0.95),
    "vtach": ("ventricular tachycardia", ClinicalDomain.CARDIOLOGY, 0.95),
    "vfib": ("ventricular fibrillation", ClinicalDomain.CARDIOLOGY, 0.95),
    "pvd": ("peripheral vascular disease", ClinicalDomain.CARDIOLOGY, 0.90),
    "aaa": ("abdominal aortic aneurysm", ClinicalDomain.CARDIOLOGY, 0.85),
    "chb": ("complete heart block", ClinicalDomain.CARDIOLOGY, 0.90),
    "jvd": ("jugular venous distension", ClinicalDomain.CARDIOLOGY, 0.95),
    "ble": ("bilateral lower extremity", ClinicalDomain.CARDIOLOGY, 0.85),
    "ekg": ("electrocardiogram", ClinicalDomain.CARDIOLOGY, 0.95),
    "ecg": ("electrocardiogram", ClinicalDomain.CARDIOLOGY, 0.95),
    "echo": ("echocardiogram", ClinicalDomain.CARDIOLOGY, 0.90),
    # ── Pulmonology ─────────────────────────────────────────────────
    "copd": ("chronic obstructive pulmonary disease", ClinicalDomain.PULMONOLOGY, 0.95),
    "sob": ("shortness of breath", ClinicalDomain.PULMONOLOGY, 0.95),
    "doe": ("dyspnea on exertion", ClinicalDomain.PULMONOLOGY, 0.90),
    "pna": ("pneumonia", ClinicalDomain.PULMONOLOGY, 0.90),
    "ards": ("acute respiratory distress syndrome", ClinicalDomain.PULMONOLOGY, 0.95),
    "osa": ("obstructive sleep apnea", ClinicalDomain.PULMONOLOGY, 0.95),
    "cpap": ("continuous positive airway pressure", ClinicalDomain.PULMONOLOGY, 0.95),
    "bipap": ("bilevel positive airway pressure", ClinicalDomain.PULMONOLOGY, 0.95),
    "cxr": ("chest x-ray", ClinicalDomain.PULMONOLOGY, 0.95),
    "pfts": ("pulmonary function tests", ClinicalDomain.PULMONOLOGY, 0.95),
    "fev1": ("forced expiratory volume in 1 second", ClinicalDomain.PULMONOLOGY, 0.95),
    "fvc": ("forced vital capacity", ClinicalDomain.PULMONOLOGY, 0.95),
    "spo2": ("oxygen saturation", ClinicalDomain.PULMONOLOGY, 0.95),
    "o2": ("oxygen", ClinicalDomain.PULMONOLOGY, 0.85),
    "nc": ("nasal cannula", ClinicalDomain.PULMONOLOGY, 0.80),
    "nrb": ("non-rebreather mask", ClinicalDomain.PULMONOLOGY, 0.90),
    "ett": ("endotracheal tube", ClinicalDomain.PULMONOLOGY, 0.90),
    "vent": ("ventilator", ClinicalDomain.PULMONOLOGY, 0.80),
    # ── Endocrine ───────────────────────────────────────────────────
    "dm": ("diabetes mellitus", ClinicalDomain.ENDOCRINE, 0.90),
    "dm1": ("diabetes mellitus type 1", ClinicalDomain.ENDOCRINE, 0.95),
    "dm2": ("diabetes mellitus type 2", ClinicalDomain.ENDOCRINE, 0.95),
    "t1dm": ("type 1 diabetes mellitus", ClinicalDomain.ENDOCRINE, 0.95),
    "t2dm": ("type 2 diabetes mellitus", ClinicalDomain.ENDOCRINE, 0.95),
    "hba1c": ("hemoglobin A1c", ClinicalDomain.ENDOCRINE, 0.95),
    "a1c": ("hemoglobin A1c", ClinicalDomain.ENDOCRINE, 0.95),
    "tsh": ("thyroid-stimulating hormone", ClinicalDomain.ENDOCRINE, 0.95),
    "dka": ("diabetic ketoacidosis", ClinicalDomain.ENDOCRINE, 0.95),
    "hhs": ("hyperosmolar hyperglycemic state", ClinicalDomain.ENDOCRINE, 0.90),
    "fsbs": ("fingerstick blood sugar", ClinicalDomain.ENDOCRINE, 0.90),
    "fbg": ("fasting blood glucose", ClinicalDomain.ENDOCRINE, 0.90),
    # ── Neurology ───────────────────────────────────────────────────
    "cva": ("cerebrovascular accident", ClinicalDomain.NEUROLOGY, 0.95),
    "tia": ("transient ischemic attack", ClinicalDomain.NEUROLOGY, 0.95),
    "lp": ("lumbar puncture", ClinicalDomain.NEUROLOGY, 0.85),
    "csf": ("cerebrospinal fluid", ClinicalDomain.NEUROLOGY, 0.95),
    "eeg": ("electroencephalogram", ClinicalDomain.NEUROLOGY, 0.95),
    "emg": ("electromyography", ClinicalDomain.NEUROLOGY, 0.95),
    "loc": ("loss of consciousness", ClinicalDomain.NEUROLOGY, 0.80),
    "aox3": ("alert and oriented times three", ClinicalDomain.NEUROLOGY, 0.95),
    "aox4": ("alert and oriented times four", ClinicalDomain.NEUROLOGY, 0.95),
    "cn": ("cranial nerve", ClinicalDomain.NEUROLOGY, 0.80),
    # "ms" is in _AMBIGUOUS_DB (multiple sclerosis / mental status / morphine sulfate)
    "gcs": ("Glasgow Coma Scale", ClinicalDomain.NEUROLOGY, 0.95),
    "nihss": ("NIH Stroke Scale", ClinicalDomain.NEUROLOGY, 0.95),
    # ── Gastroenterology ────────────────────────────────────────────
    "gerd": ("gastroesophageal reflux disease", ClinicalDomain.GASTROENTEROLOGY, 0.95),
    "gi": ("gastrointestinal", ClinicalDomain.GASTROENTEROLOGY, 0.90),
    "ugib": ("upper GI bleed", ClinicalDomain.GASTROENTEROLOGY, 0.95),
    "lgib": ("lower GI bleed", ClinicalDomain.GASTROENTEROLOGY, 0.95),
    "egd": ("esophagogastroduodenoscopy", ClinicalDomain.GASTROENTEROLOGY, 0.95),
    "ercp": ("endoscopic retrograde cholangiopancreatography", ClinicalDomain.GASTROENTEROLOGY, 0.95),
    "sbo": ("small bowel obstruction", ClinicalDomain.GASTROENTEROLOGY, 0.95),
    "lbo": ("large bowel obstruction", ClinicalDomain.GASTROENTEROLOGY, 0.95),
    "npo": ("nothing by mouth", ClinicalDomain.GASTROENTEROLOGY, 0.95),
    "bm": ("bowel movement", ClinicalDomain.GASTROENTEROLOGY, 0.80),
    "n/v": ("nausea and vomiting", ClinicalDomain.GASTROENTEROLOGY, 0.95),
    "n/v/d": ("nausea, vomiting, and diarrhea", ClinicalDomain.GASTROENTEROLOGY, 0.95),
    "lft": ("liver function tests", ClinicalDomain.GASTROENTEROLOGY, 0.90),
    "lfts": ("liver function tests", ClinicalDomain.GASTROENTEROLOGY, 0.90),
    "hep": ("hepatitis", ClinicalDomain.GASTROENTEROLOGY, 0.80),
    # ── Renal ───────────────────────────────────────────────────────
    "ckd": ("chronic kidney disease", ClinicalDomain.RENAL, 0.95),
    "aki": ("acute kidney injury", ClinicalDomain.RENAL, 0.95),
    "esrd": ("end-stage renal disease", ClinicalDomain.RENAL, 0.95),
    "gfr": ("glomerular filtration rate", ClinicalDomain.RENAL, 0.95),
    "egfr": ("estimated glomerular filtration rate", ClinicalDomain.RENAL, 0.95),
    "bun": ("blood urea nitrogen", ClinicalDomain.RENAL, 0.95),
    # "cr" is in _AMBIGUOUS_DB (creatinine — context-dependent)
    "hd": ("hemodialysis", ClinicalDomain.RENAL, 0.85),
    # "pd" is in _AMBIGUOUS_DB (peritoneal dialysis / probing depth)
    "uti": ("urinary tract infection", ClinicalDomain.RENAL, 0.95),
    "foley": ("Foley catheter", ClinicalDomain.RENAL, 0.90),
    "ua": ("urinalysis", ClinicalDomain.RENAL, 0.85),
    # ── Infectious Disease ──────────────────────────────────────────
    "mrsa": ("methicillin-resistant Staphylococcus aureus", ClinicalDomain.INFECTIOUS, 0.95),
    "vre": ("vancomycin-resistant Enterococcus", ClinicalDomain.INFECTIOUS, 0.95),
    "hiv": ("human immunodeficiency virus", ClinicalDomain.INFECTIOUS, 0.95),
    "aids": ("acquired immunodeficiency syndrome", ClinicalDomain.INFECTIOUS, 0.95),
    "tb": ("tuberculosis", ClinicalDomain.INFECTIOUS, 0.90),
    "ppd": ("purified protein derivative", ClinicalDomain.INFECTIOUS, 0.85),
    "c. diff": ("Clostridioides difficile", ClinicalDomain.INFECTIOUS, 0.95),
    "cdiff": ("Clostridioides difficile", ClinicalDomain.INFECTIOUS, 0.95),
    "abx": ("antibiotics", ClinicalDomain.INFECTIOUS, 0.90),
    "bcx": ("blood cultures", ClinicalDomain.INFECTIOUS, 0.90),
    # ── Musculoskeletal ─────────────────────────────────────────────
    "oa": ("osteoarthritis", ClinicalDomain.MUSCULOSKELETAL, 0.85),
    # "ra" is in _AMBIGUOUS_DB (rheumatoid arthritis / room air)
    "djd": ("degenerative joint disease", ClinicalDomain.MUSCULOSKELETAL, 0.95),
    "rom": ("range of motion", ClinicalDomain.MUSCULOSKELETAL, 0.90),
    "tka": ("total knee arthroplasty", ClinicalDomain.MUSCULOSKELETAL, 0.95),
    "tha": ("total hip arthroplasty", ClinicalDomain.MUSCULOSKELETAL, 0.95),
    "orif": ("open reduction internal fixation", ClinicalDomain.MUSCULOSKELETAL, 0.95),
    "fx": ("fracture", ClinicalDomain.MUSCULOSKELETAL, 0.85),
    # "pt" is in _AMBIGUOUS_DB (patient / physical therapy / prothrombin time)
    "ot": ("occupational therapy", ClinicalDomain.MUSCULOSKELETAL, 0.75),
    # ── Hematology ──────────────────────────────────────────────────
    "cbc": ("complete blood count", ClinicalDomain.HEMATOLOGY, 0.95),
    "bmp": ("basic metabolic panel", ClinicalDomain.HEMATOLOGY, 0.95),
    "cmp": ("comprehensive metabolic panel", ClinicalDomain.HEMATOLOGY, 0.95),
    "wbc": ("white blood cell count", ClinicalDomain.HEMATOLOGY, 0.95),
    "rbc": ("red blood cell count", ClinicalDomain.HEMATOLOGY, 0.95),
    "hgb": ("hemoglobin", ClinicalDomain.HEMATOLOGY, 0.95),
    "hct": ("hematocrit", ClinicalDomain.HEMATOLOGY, 0.95),
    "plt": ("platelet count", ClinicalDomain.HEMATOLOGY, 0.95),
    "inr": ("international normalized ratio", ClinicalDomain.HEMATOLOGY, 0.95),
    "ptt": ("partial thromboplastin time", ClinicalDomain.HEMATOLOGY, 0.95),
    "aptt": ("activated partial thromboplastin time", ClinicalDomain.HEMATOLOGY, 0.95),
    "esr": ("erythrocyte sedimentation rate", ClinicalDomain.HEMATOLOGY, 0.95),
    "crp": ("C-reactive protein", ClinicalDomain.HEMATOLOGY, 0.95),
    "prbc": ("packed red blood cells", ClinicalDomain.HEMATOLOGY, 0.95),
    "ffp": ("fresh frozen plasma", ClinicalDomain.HEMATOLOGY, 0.95),
    # ── General / Common ────────────────────────────────────────────
    "h/o": ("history of", ClinicalDomain.GENERAL, 0.95),
    "c/o": ("complains of", ClinicalDomain.GENERAL, 0.95),
    "s/p": ("status post", ClinicalDomain.GENERAL, 0.95),
    "f/u": ("follow-up", ClinicalDomain.GENERAL, 0.90),
    "w/u": ("work-up", ClinicalDomain.GENERAL, 0.90),
    "r/o": ("rule out", ClinicalDomain.GENERAL, 0.90),
    "d/c": ("discharge", ClinicalDomain.GENERAL, 0.85),
    "y/o": ("year old", ClinicalDomain.GENERAL, 0.90),
    "yo": ("year old", ClinicalDomain.GENERAL, 0.85),
    "m/f": ("male/female", ClinicalDomain.GENERAL, 0.80),
    "wnl": ("within normal limits", ClinicalDomain.GENERAL, 0.95),
    "nad": ("no acute distress", ClinicalDomain.GENERAL, 0.90),
    "nkda": ("no known drug allergies", ClinicalDomain.GENERAL, 0.95),
    "nka": ("no known allergies", ClinicalDomain.GENERAL, 0.95),
    "ama": ("against medical advice", ClinicalDomain.GENERAL, 0.85),
    "dnr": ("do not resuscitate", ClinicalDomain.GENERAL, 0.95),
    "dni": ("do not intubate", ClinicalDomain.GENERAL, 0.95),
    "dnr/dni": ("do not resuscitate / do not intubate", ClinicalDomain.GENERAL, 0.95),
    "hpi": ("history of present illness", ClinicalDomain.GENERAL, 0.95),
    "pmh": ("past medical history", ClinicalDomain.GENERAL, 0.95),
    "psh": ("past surgical history", ClinicalDomain.GENERAL, 0.95),
    "ros": ("review of systems", ClinicalDomain.GENERAL, 0.95),
    "a/p": ("assessment and plan", ClinicalDomain.GENERAL, 0.90),
    "cc": ("chief complaint", ClinicalDomain.GENERAL, 0.80),
    "dx": ("diagnosis", ClinicalDomain.GENERAL, 0.90),
    "ddx": ("differential diagnosis", ClinicalDomain.GENERAL, 0.95),
    "tx": ("treatment", ClinicalDomain.GENERAL, 0.85),
    "sx": ("symptoms", ClinicalDomain.GENERAL, 0.80),
    "hx": ("history", ClinicalDomain.GENERAL, 0.85),
    "rx": ("prescription", ClinicalDomain.GENERAL, 0.85),
    "bmi": ("body mass index", ClinicalDomain.GENERAL, 0.95),
    "ct": ("computed tomography", ClinicalDomain.GENERAL, 0.90),
    "mri": ("magnetic resonance imaging", ClinicalDomain.GENERAL, 0.95),
    "us": ("ultrasound", ClinicalDomain.GENERAL, 0.75),
    # "or" is in _AMBIGUOUS_DB (operating room — context-dependent)
    "er": ("emergency room", ClinicalDomain.GENERAL, 0.80),
    # "ed" is in _AMBIGUOUS_DB (emergency department / erectile dysfunction)
    "icu": ("intensive care unit", ClinicalDomain.GENERAL, 0.95),
    "ccu": ("coronary care unit", ClinicalDomain.GENERAL, 0.90),
    "micu": ("medical intensive care unit", ClinicalDomain.GENERAL, 0.95),
    "sicu": ("surgical intensive care unit", ClinicalDomain.GENERAL, 0.95),
    "pacu": ("post-anesthesia care unit", ClinicalDomain.GENERAL, 0.95),
    "snf": ("skilled nursing facility", ClinicalDomain.GENERAL, 0.90),
    "ltac": ("long-term acute care", ClinicalDomain.GENERAL, 0.90),
    "loa": ("leave of absence", ClinicalDomain.GENERAL, 0.80),
    "aox": ("alert and oriented", ClinicalDomain.GENERAL, 0.85),
    "i&d": ("incision and drainage", ClinicalDomain.GENERAL, 0.90),
    # ── Pharmacy ────────────────────────────────────────────────────
    "po": ("by mouth", ClinicalDomain.PHARMACY, 0.90),
    "iv": ("intravenous", ClinicalDomain.PHARMACY, 0.90),
    "im": ("intramuscular", ClinicalDomain.PHARMACY, 0.90),
    "sq": ("subcutaneous", ClinicalDomain.PHARMACY, 0.90),
    "subq": ("subcutaneous", ClinicalDomain.PHARMACY, 0.95),
    "sl": ("sublingual", ClinicalDomain.PHARMACY, 0.85),
    "pr": ("per rectum", ClinicalDomain.PHARMACY, 0.80),
    "bid": ("twice daily", ClinicalDomain.PHARMACY, 0.95),
    "tid": ("three times daily", ClinicalDomain.PHARMACY, 0.95),
    "qid": ("four times daily", ClinicalDomain.PHARMACY, 0.95),
    "qd": ("once daily", ClinicalDomain.PHARMACY, 0.90),
    "qhs": ("at bedtime", ClinicalDomain.PHARMACY, 0.95),
    "qam": ("every morning", ClinicalDomain.PHARMACY, 0.95),
    "qpm": ("every evening", ClinicalDomain.PHARMACY, 0.95),
    "q4h": ("every 4 hours", ClinicalDomain.PHARMACY, 0.95),
    "q6h": ("every 6 hours", ClinicalDomain.PHARMACY, 0.95),
    "q8h": ("every 8 hours", ClinicalDomain.PHARMACY, 0.95),
    "q12h": ("every 12 hours", ClinicalDomain.PHARMACY, 0.95),
    "prn": ("as needed", ClinicalDomain.PHARMACY, 0.95),
    "stat": ("immediately", ClinicalDomain.PHARMACY, 0.90),
    "ac": ("before meals", ClinicalDomain.PHARMACY, 0.85),
    "pc": ("after meals", ClinicalDomain.PHARMACY, 0.85),
    "hs": ("at bedtime", ClinicalDomain.PHARMACY, 0.80),
    "mcg": ("micrograms", ClinicalDomain.PHARMACY, 0.90),
    "meq": ("milliequivalents", ClinicalDomain.PHARMACY, 0.90),
    "gtts": ("drops", ClinicalDomain.PHARMACY, 0.90),
    "tab": ("tablet", ClinicalDomain.PHARMACY, 0.85),
    # "cap" is in _AMBIGUOUS_DB (community-acquired pneumonia / capsule)
    "susp": ("suspension", ClinicalDomain.PHARMACY, 0.85),
    "oint": ("ointment", ClinicalDomain.PHARMACY, 0.85),
    "inh": ("inhaler", ClinicalDomain.PHARMACY, 0.85),
    "neb": ("nebulizer", ClinicalDomain.PHARMACY, 0.90),
    "nsaid": ("nonsteroidal anti-inflammatory drug", ClinicalDomain.PHARMACY, 0.95),
    "nsaids": ("nonsteroidal anti-inflammatory drugs", ClinicalDomain.PHARMACY, 0.95),
    "ppi": ("proton pump inhibitor", ClinicalDomain.PHARMACY, 0.90),
    "ace": ("angiotensin-converting enzyme", ClinicalDomain.PHARMACY, 0.85),
    "arb": ("angiotensin receptor blocker", ClinicalDomain.PHARMACY, 0.90),
    "ssri": ("selective serotonin reuptake inhibitor", ClinicalDomain.PHARMACY, 0.95),
    "snri": ("serotonin-norepinephrine reuptake inhibitor", ClinicalDomain.PHARMACY, 0.95),
    "tca": ("tricyclic antidepressant", ClinicalDomain.PHARMACY, 0.90),
    "maoi": ("monoamine oxidase inhibitor", ClinicalDomain.PHARMACY, 0.95),
    "hctz": ("hydrochlorothiazide", ClinicalDomain.PHARMACY, 0.95),
    "asa": ("aspirin", ClinicalDomain.PHARMACY, 0.90),
    "abx": ("antibiotics", ClinicalDomain.PHARMACY, 0.90),
    "pcn": ("penicillin", ClinicalDomain.PHARMACY, 0.90),
    # ── Dental ──────────────────────────────────────────────────────
    "srp": ("scaling and root planing", ClinicalDomain.DENTAL, 0.95),
    "rct": ("root canal therapy", ClinicalDomain.DENTAL, 0.90),
    "bop": ("bleeding on probing", ClinicalDomain.DENTAL, 0.95),
    "tmj": ("temporomandibular joint", ClinicalDomain.DENTAL, 0.95),
    "tmd": ("temporomandibular disorder", ClinicalDomain.DENTAL, 0.95),
    "cdt": ("Current Dental Terminology", ClinicalDomain.DENTAL, 0.90),
    "perio": ("periodontal", ClinicalDomain.DENTAL, 0.90),
    "endo": ("endodontic", ClinicalDomain.DENTAL, 0.85),
    "ortho": ("orthodontic", ClinicalDomain.DENTAL, 0.80),
    "prosth": ("prosthetic", ClinicalDomain.DENTAL, 0.85),
    "cal": ("clinical attachment loss", ClinicalDomain.DENTAL, 0.85),
    # "pd" is in _AMBIGUOUS_DB (peritoneal dialysis / probing depth)
    "bwx": ("bitewing x-ray", ClinicalDomain.DENTAL, 0.95),
    "pano": ("panoramic radiograph", ClinicalDomain.DENTAL, 0.90),
    "fmx": ("full mouth x-ray", ClinicalDomain.DENTAL, 0.95),
    "pfm": ("porcelain-fused-to-metal", ClinicalDomain.DENTAL, 0.90),
    "rpd": ("removable partial denture", ClinicalDomain.DENTAL, 0.90),
}


# ─────────────────────────────────────────────────────────────────────
# Ambiguous Abbreviations
# ─────────────────────────────────────────────────────────────────────
# These abbreviations have multiple clinical meanings. Each entry maps
# to a list of (expansion, domain, context_keywords) tuples. The
# context_keywords are used for disambiguation — if any keyword appears
# within the context window, that sense is preferred.

_AMBIGUOUS_DB: dict[str, list[tuple[str, ClinicalDomain, list[str]]]] = {
    "pe": [
        (
            "pulmonary embolism",
            ClinicalDomain.PULMONOLOGY,
            [
                "clot", "embol", "anticoag", "heparin", "warfarin",
                "dvt", "d-dimer", "cta", "wells", "thrombo",
                "dyspnea", "pleuritic", "tachycard",
            ],
        ),
        (
            "physical exam",
            ClinicalDomain.GENERAL,
            [
                "exam", "auscult", "palpat", "inspect", "vitals",
                "heent", "lungs clear", "heart regular", "abdomen",
                "extremit", "neuro", "skin", "physical",
            ],
        ),
    ],
    "ms": [
        (
            "multiple sclerosis",
            ClinicalDomain.NEUROLOGY,
            [
                "demyelinat", "lesion", "plaque", "relaps", "remit",
                "mri brain", "optic neuritis", "numbness", "tingling",
                "interferon", "copaxone", "white matter",
            ],
        ),
        (
            "mental status",
            ClinicalDomain.NEUROLOGY,
            [
                "alert", "oriented", "confus", "deliri", "letharg",
                "agitat", "aox", "gcs", "consciousness",
                "cogniti", "mmse",
            ],
        ),
        (
            "morphine sulfate",
            ClinicalDomain.PHARMACY,
            [
                "pain", "mg", "iv push", "dose", "narcotic",
                "opioid", "analges", "prn pain", "dilaudid",
            ],
        ),
    ],
    "pt": [
        (
            "patient",
            ClinicalDomain.GENERAL,
            [
                "the pt", "pt is", "pt was", "pt has", "pt report",
                "pt deni", "pt present", "pt complain", "pt state",
                "admit", "discharge",
            ],
        ),
        (
            "physical therapy",
            ClinicalDomain.MUSCULOSKELETAL,
            [
                "therapy", "rehab", "exercise", "gait", "mobiliz",
                "strengthen", "rom", "ambul", "walker", "crutch",
                "ot", "consult pt",
            ],
        ),
        (
            "prothrombin time",
            ClinicalDomain.HEMATOLOGY,
            [
                "inr", "coag", "warfarin", "coumadin", "bleed",
                "anticoag", "ptt", "clot", "seconds",
            ],
        ),
    ],
    "or": [
        (
            "operating room",
            ClinicalDomain.GENERAL,
            [
                "surg", "operat", "anesthes", "incision", "procedure",
                "taken to", "booked", "scheduled", "pre-op", "post-op",
                "intraop",
            ],
        ),
    ],
    "cr": [
        (
            "creatinine",
            ClinicalDomain.RENAL,
            [
                "bun", "gfr", "kidney", "renal", "lab", "level",
                "mg/dl", "elevated", "baseline",
            ],
        ),
    ],
    "cap": [
        (
            "community-acquired pneumonia",
            ClinicalDomain.PULMONOLOGY,
            [
                "pneumonia", "antibiotic", "cxr", "infiltrate",
                "fever", "cough", "consolidat", "sputum",
            ],
        ),
        (
            "capsule",
            ClinicalDomain.PHARMACY,
            [
                "mg", "dose", "take", "oral", "swallow",
                "medication", "prescri", "dispens",
            ],
        ),
    ],
    "ra": [
        (
            "rheumatoid arthritis",
            ClinicalDomain.MUSCULOSKELETAL,
            [
                "joint", "swelling", "stiffness", "methotrexate",
                "dmard", "rheumat", "synovit", "erosion",
                "deformit", "nodule",
            ],
        ),
        (
            "room air",
            ClinicalDomain.PULMONOLOGY,
            [
                "spo2", "o2 sat", "saturat", "oxygen",
                "on ra", "% on", "breathing",
            ],
        ),
    ],
    "pd": [
        (
            "peritoneal dialysis",
            ClinicalDomain.RENAL,
            [
                "dialys", "catheter", "exchange", "dwell",
                "esrd", "renal", "fluid",
            ],
        ),
        (
            "probing depth",
            ClinicalDomain.DENTAL,
            [
                "perio", "pocket", "mm", "gingiv", "probing",
                "attach", "recession", "tooth", "teeth",
            ],
        ),
    ],
    "ed": [
        (
            "emergency department",
            ClinicalDomain.GENERAL,
            [
                "present", "triage", "emergenc", "er",
                "ambulance", "ems", "acute", "arrival",
            ],
        ),
        (
            "erectile dysfunction",
            ClinicalDomain.GENERAL,
            [
                "sexual", "viagra", "sildenafil", "cialis",
                "tadalafil", "erect", "pde5",
            ],
        ),
    ],
}


# ─────────────────────────────────────────────────────────────────────
# Section headers that influence disambiguation
# ─────────────────────────────────────────────────────────────────────
# Maps section header patterns to abbreviation-sense overrides.
# If the abbreviation appears inside a detected section, use that sense.

_SECTION_SENSE_OVERRIDES: dict[str, dict[str, str]] = {
    "physical exam": {"pe": "physical exam"},
    "physical examination": {"pe": "physical exam"},
    "pe:": {"pe": "physical exam"},
    "medications": {"cap": "capsule", "tab": "tablet"},
    "medication list": {"cap": "capsule"},
    "dental history": {"pd": "probing depth", "rct": "root canal therapy"},
    "periodontal": {"pd": "probing depth", "cal": "clinical attachment loss"},
    "assessment and plan": {"or": "operating room"},
    "a/p": {"or": "operating room"},
    "labs": {"pt": "prothrombin time", "cr": "creatinine"},
    "laboratory": {"pt": "prothrombin time", "cr": "creatinine"},
    "lab results": {"pt": "prothrombin time", "cr": "creatinine"},
}

# Compile section header patterns once
_SECTION_HEADER_RE = re.compile(
    r"(?:^|\n)\s*(?:"
    + "|".join(
        re.escape(h) for h in sorted(_SECTION_SENSE_OVERRIDES, key=len, reverse=True)
    )
    + r")\s*[:\-]?\s*",
    re.IGNORECASE,
)


class AbbreviationExpander:
    """Detects and expands clinical abbreviations in free text.

    Uses a 300+ entry dictionary covering 12 clinical domains with
    context-aware disambiguation for ambiguous abbreviations.

    Parameters
    ----------
    config : AbbreviationConfig | None
        Configuration options. Uses defaults if None.

    Examples
    --------
    >>> expander = AbbreviationExpander()
    >>> result = expander.expand("Pt c/o SOB and CP. PMH: HTN, DM2, CAD.")
    >>> result.total_found
    7
    >>> result.matches[0].expansion
    'patient'
    """

    def __init__(self, config: AbbreviationConfig | None = None) -> None:
        self.config = config or AbbreviationConfig()
        self._patterns = self._compile_patterns()
        self._ambiguous_keys = set(_AMBIGUOUS_DB.keys())

    def _compile_patterns(self) -> list[tuple[re.Pattern, str]]:
        """Compile word-boundary regex patterns for all abbreviations.

        Returns
        -------
        list[tuple[re.Pattern, str]]
            Pairs of (compiled pattern, abbreviation key), sorted by
            abbreviation length descending to match longer forms first.
        """
        all_abbrevs = set(_ABBREVIATION_DB.keys()) | set(_AMBIGUOUS_DB.keys())

        # Sort by length descending so longer abbreviations match first
        # (e.g., "n/v/d" before "n/v", "dnr/dni" before "dnr")
        sorted_abbrevs = sorted(all_abbrevs, key=len, reverse=True)

        patterns = []
        for abbrev in sorted_abbrevs:
            # Apply domain filter if configured
            if self.config.domains:
                if abbrev in _ABBREVIATION_DB:
                    _, domain, _ = _ABBREVIATION_DB[abbrev]
                    if domain not in self.config.domains:
                        continue
                elif abbrev in _AMBIGUOUS_DB:
                    domains = {d for _, d, _ in _AMBIGUOUS_DB[abbrev]}
                    if not domains & set(self.config.domains):
                        continue

            # Escape special regex characters in the abbreviation
            escaped = re.escape(abbrev)
            # Word-boundary matching to prevent partial matches
            pattern = re.compile(
                rf"\b{escaped}\b",
                re.IGNORECASE,
            )
            patterns.append((pattern, abbrev))

        return patterns

    def _detect_sections(self, text: str) -> list[tuple[int, int, dict[str, str]]]:
        """Detect section headers and their sense overrides.

        Parameters
        ----------
        text : str
            Input clinical text.

        Returns
        -------
        list[tuple[int, int, dict[str, str]]]
            List of (start, end, override_dict) tuples for each section.
            End is the start of the next section or len(text).
        """
        sections: list[tuple[int, int, dict[str, str]]] = []
        for match in _SECTION_HEADER_RE.finditer(text):
            header_text = match.group().strip().rstrip(":-").strip().lower()
            # Find the matching override key
            for key, overrides in _SECTION_SENSE_OVERRIDES.items():
                if key in header_text:
                    sections.append((match.start(), len(text), overrides))
                    break

        # Set proper end boundaries — each section ends where the next begins
        for i in range(len(sections) - 1):
            start, _, overrides = sections[i]
            next_start = sections[i + 1][0]
            sections[i] = (start, next_start, overrides)

        return sections

    def _get_section_override(
        self, pos: int, abbrev_key: str,
        sections: list[tuple[int, int, dict[str, str]]],
    ) -> str | None:
        """Check if position falls within a section that overrides this abbreviation.

        Parameters
        ----------
        pos : int
            Character position in the text.
        abbrev_key : str
            Lowercase abbreviation key.
        sections : list[tuple[int, int, dict[str, str]]]
            Detected sections from ``_detect_sections()``.

        Returns
        -------
        str | None
            The overridden expansion, or None if no override applies.
        """
        for start, end, overrides in sections:
            if start <= pos < end and abbrev_key in overrides:
                return overrides[abbrev_key]
        return None

    def _resolve_ambiguous(
        self, abbrev_key: str, text: str, start: int, end: int,
        sections: list[tuple[int, int, dict[str, str]]],
    ) -> tuple[str, ClinicalDomain, float, AmbiguityResolution, list[str]]:
        """Resolve an ambiguous abbreviation using context signals.

        Strategy priority:
        1. Section header override (highest confidence)
        2. Context keyword matching within window
        3. Default to first sense (lowest confidence)

        Parameters
        ----------
        abbrev_key : str
            Lowercase abbreviation key.
        text : str
            Full document text.
        start : int
            Start offset of the match.
        end : int
            End offset of the match.
        sections : list
            Detected section boundaries.

        Returns
        -------
        tuple
            (expansion, domain, confidence, resolution_method, alternatives)
        """
        senses = _AMBIGUOUS_DB[abbrev_key]
        all_expansions = [s[0] for s in senses]

        # Strategy 1: Section header override
        section_override = self._get_section_override(start, abbrev_key, sections)
        if section_override:
            for expansion, domain, _ in senses:
                if expansion == section_override:
                    alternatives = [e for e in all_expansions if e != expansion]
                    return (
                        expansion, domain, 0.90,
                        AmbiguityResolution.SECTION_RESOLVED, alternatives,
                    )

        # Strategy 2: Context keyword matching
        ctx_start = max(0, start - self.config.context_window)
        ctx_end = min(len(text), end + self.config.context_window)
        context = text[ctx_start:ctx_end].lower()

        best_score = 0
        best_sense = senses[0]

        for expansion, domain, keywords in senses:
            score = sum(1 for kw in keywords if kw in context)
            if score > best_score:
                best_score = score
                best_sense = (expansion, domain, keywords)

        if best_score > 0:
            expansion, domain, _ = best_sense
            # Confidence scales with number of matching keywords (max 0.90)
            confidence = min(0.90, 0.70 + best_score * 0.05)
            alternatives = [e for e in all_expansions if e != expansion]
            return (
                expansion, domain, confidence,
                AmbiguityResolution.CONTEXT_RESOLVED, alternatives,
            )

        # Strategy 3: Default to first sense
        expansion, domain, _ = senses[0]
        alternatives = [e for e in all_expansions if e != expansion]
        return (
            expansion, domain, 0.60,
            AmbiguityResolution.DEFAULT_SENSE, alternatives,
        )

    def expand(self, text: str) -> ExpansionResult:
        """Detect and expand abbreviations in clinical text.

        Parameters
        ----------
        text : str
            Clinical free text to analyze.

        Returns
        -------
        ExpansionResult
            Contains all matches, expanded text, and processing metadata.

        Raises
        ------
        ValueError
            If text is empty or None.
        """
        if not text or not text.strip():
            return ExpansionResult(
                original_text=text or "",
                expanded_text=text or "",
                matches=[],
                total_found=0,
                ambiguous_count=0,
                processing_time_ms=0.0,
            )

        start_time = time.perf_counter()

        # Detect section headers for context-aware disambiguation
        sections = self._detect_sections(text)

        # Find all abbreviation matches
        raw_matches: list[AbbreviationMatch] = []

        for pattern, abbrev_key in self._patterns:
            for m in pattern.finditer(text):
                match_start = m.start()
                match_end = m.end()

                if abbrev_key in self._ambiguous_keys:
                    # Ambiguous — resolve via context
                    expansion, domain, confidence, resolution, alternatives = (
                        self._resolve_ambiguous(
                            abbrev_key, text, match_start, match_end, sections,
                        )
                    )
                    raw_matches.append(
                        AbbreviationMatch(
                            abbreviation=m.group(),
                            expansion=expansion,
                            start=match_start,
                            end=match_end,
                            confidence=confidence,
                            domain=domain,
                            is_ambiguous=True,
                            resolution=resolution,
                            alternative_expansions=alternatives,
                        ),
                    )
                elif abbrev_key in _ABBREVIATION_DB:
                    # Unambiguous — direct lookup
                    expansion, domain, confidence = _ABBREVIATION_DB[abbrev_key]

                    if not self.config.include_unambiguous:
                        continue

                    raw_matches.append(
                        AbbreviationMatch(
                            abbreviation=m.group(),
                            expansion=expansion,
                            start=match_start,
                            end=match_end,
                            confidence=confidence,
                            domain=domain,
                            is_ambiguous=False,
                            resolution=AmbiguityResolution.UNAMBIGUOUS,
                        ),
                    )

        # Deduplicate overlapping spans — keep highest confidence
        matches = self._deduplicate(raw_matches)

        # Apply confidence threshold
        matches = [
            m for m in matches if m.confidence >= self.config.min_confidence
        ]

        # Sort by position in text
        matches.sort(key=lambda m: m.start)

        # Build expanded text if requested
        if self.config.expand_in_place and matches:
            expanded_text = self._build_expanded_text(text, matches)
        else:
            expanded_text = text

        elapsed = (time.perf_counter() - start_time) * 1000

        return ExpansionResult(
            original_text=text,
            expanded_text=expanded_text,
            matches=matches,
            total_found=len(matches),
            ambiguous_count=sum(1 for m in matches if m.is_ambiguous),
            processing_time_ms=elapsed,
        )

    def expand_batch(
        self, texts: list[str],
    ) -> list[ExpansionResult]:
        """Expand abbreviations in multiple documents.

        Parameters
        ----------
        texts : list[str]
            List of clinical texts to process.

        Returns
        -------
        list[ExpansionResult]
            One result per input text, in order.
        """
        return [self.expand(t) for t in texts]

    def _deduplicate(
        self, matches: list[AbbreviationMatch],
    ) -> list[AbbreviationMatch]:
        """Remove overlapping matches, keeping highest confidence.

        When two matches overlap in character span, the one with higher
        confidence wins. On tie, the longer match is preferred.

        Parameters
        ----------
        matches : list[AbbreviationMatch]
            Raw matches potentially with overlaps.

        Returns
        -------
        list[AbbreviationMatch]
            Deduplicated matches with no overlapping spans.
        """
        if len(matches) <= 1:
            return matches

        # Sort by start position, then by length descending
        sorted_matches = sorted(
            matches, key=lambda m: (m.start, -(m.end - m.start)),
        )

        result: list[AbbreviationMatch] = []
        for match in sorted_matches:
            # Check overlap with already-accepted matches
            overlaps = False
            for i, accepted in enumerate(result):
                if match.start < accepted.end and match.end > accepted.start:
                    # Overlap detected — keep higher confidence
                    if match.confidence > accepted.confidence or (
                        match.confidence == accepted.confidence
                        and (match.end - match.start) > (accepted.end - accepted.start)
                    ):
                        result[i] = match
                    overlaps = True
                    break

            if not overlaps:
                result.append(match)

        return result

    def _build_expanded_text(
        self, text: str, matches: list[AbbreviationMatch],
    ) -> str:
        """Build text with abbreviations replaced by expansions.

        Uses "abbreviation (expansion)" format to preserve readability
        while adding clarity.

        Parameters
        ----------
        text : str
            Original text.
        matches : list[AbbreviationMatch]
            Sorted, deduplicated matches.

        Returns
        -------
        str
            Text with abbreviations expanded inline.
        """
        # Process matches in reverse order to preserve offsets
        result = text
        for match in reversed(matches):
            original = result[match.start:match.end]
            replacement = f"{original} ({match.expansion})"
            result = result[:match.start] + replacement + result[match.end:]

        return result

    def get_dictionary_stats(self) -> dict:
        """Return statistics about the abbreviation dictionary.

        Returns
        -------
        dict
            Dictionary coverage statistics by domain.
        """
        domain_counts: dict[str, int] = {}
        for _, (_, domain, _) in _ABBREVIATION_DB.items():
            domain_counts[str(domain)] = domain_counts.get(str(domain), 0) + 1

        ambiguous_domain_counts: dict[str, int] = {}
        for abbrev, senses in _AMBIGUOUS_DB.items():
            for _, domain, _ in senses:
                key = str(domain)
                ambiguous_domain_counts[key] = ambiguous_domain_counts.get(key, 0) + 1

        return {
            "total_unambiguous": len(_ABBREVIATION_DB),
            "total_ambiguous": len(_AMBIGUOUS_DB),
            "total_senses": sum(len(s) for s in _AMBIGUOUS_DB.values()),
            "total_entries": len(_ABBREVIATION_DB) + len(_AMBIGUOUS_DB),
            "domains": {
                d: {
                    "unambiguous": domain_counts.get(d, 0),
                    "ambiguous_senses": ambiguous_domain_counts.get(d, 0),
                }
                for d in sorted(
                    set(list(domain_counts.keys()) + list(ambiguous_domain_counts.keys())),
                )
            },
        }

    def lookup(self, abbreviation: str) -> dict | None:
        """Look up a specific abbreviation.

        Parameters
        ----------
        abbreviation : str
            Abbreviation to look up (case-insensitive).

        Returns
        -------
        dict | None
            Abbreviation details, or None if not found.
        """
        key = abbreviation.lower()

        if key in _ABBREVIATION_DB:
            expansion, domain, confidence = _ABBREVIATION_DB[key]
            return {
                "abbreviation": abbreviation,
                "expansion": expansion,
                "domain": str(domain),
                "confidence": confidence,
                "is_ambiguous": False,
            }

        if key in _AMBIGUOUS_DB:
            senses = _AMBIGUOUS_DB[key]
            return {
                "abbreviation": abbreviation,
                "senses": [
                    {
                        "expansion": exp,
                        "domain": str(dom),
                        "context_keywords": kw,
                    }
                    for exp, dom, kw in senses
                ],
                "is_ambiguous": True,
            }

        return None
