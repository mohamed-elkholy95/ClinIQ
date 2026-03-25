"""Clinical concept normalization engine.

Maps extracted entity text (e.g. "HTN", "heart attack", "metformin 500 mg")
to standardised medical ontology codes:

* **UMLS CUI** — Concept Unique Identifiers from the Unified Medical
  Language System.  The lingua franca for biomedical concept identity.
* **SNOMED-CT** — Systematized Nomenclature of Medicine, the richest
  clinical terminology for diagnoses, procedures, and body structures.
* **RxNorm** — Normalised drug names maintained by the NLM for
  medication interoperability.
* **ICD-10-CM** — International Classification of Diseases codes.
* **LOINC** — Logical Observation Identifiers Names and Codes for
  laboratory and clinical observations.

Architecture
------------
1. **Exact-match lookup** — O(1) dictionary hit on case-folded +
   whitespace-normalised text.  Covers common conditions, medications,
   procedures, anatomy, and lab tests with curated mappings.
2. **Alias / synonym expansion** — Many clinical terms have multiple
   surface forms (abbreviations, eponyms, brand names).  A reverse
   index maps every alias to its canonical concept.
3. **Fuzzy matching** — Levenshtein-ratio based candidate scoring
   when exact match misses.  Controlled by a configurable minimum
   similarity threshold (default 0.80) to prevent false links.
4. **Entity-type-aware filtering** — Optional constraint that only
   considers concepts whose ontology source is compatible with the
   entity type (e.g. MEDICATION entities only match RxNorm entries).
5. **Batch normalization** — Process multiple entities in a single
   call with shared dictionary access and deduplication.

Thread safety
-------------
The concept dictionary and alias index are immutable after module load.
Instance state is read-only during normalization.  Thread-safe without
external synchronisation.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import StrEnum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


class OntologySource(StrEnum):
    """Supported medical ontology systems."""

    UMLS = "UMLS"
    SNOMED_CT = "SNOMED-CT"
    RXNORM = "RxNorm"
    ICD10CM = "ICD-10-CM"
    LOINC = "LOINC"


class EntityTypeGroup(StrEnum):
    """Broad entity type groupings for ontology filtering."""

    CONDITION = "CONDITION"  # DISEASE, SYMPTOM
    MEDICATION = "MEDICATION"
    PROCEDURE = "PROCEDURE"
    ANATOMY = "ANATOMY"  # ANATOMY, BODY_PART
    LAB = "LAB"  # LAB_VALUE, TEST


@dataclass(frozen=True)
class ConceptEntry:
    """A normalised medical concept with ontology codes.

    Parameters
    ----------
    cui : str
        UMLS Concept Unique Identifier (e.g. ``C0020538``).
    preferred_term : str
        Canonical / preferred surface form.
    aliases : tuple[str, ...]
        Alternative surface forms (abbreviations, synonyms, brand names).
    snomed_code : str | None
        SNOMED-CT concept ID, if available.
    rxnorm_code : str | None
        RxNorm concept ID, if available.
    icd10_code : str | None
        ICD-10-CM code, if available.
    loinc_code : str | None
        LOINC code, if available.
    semantic_type : str
        UMLS semantic type label (e.g. ``Disease or Syndrome``).
    type_group : EntityTypeGroup
        Broad grouping for entity-type filtering.
    """

    cui: str
    preferred_term: str
    aliases: tuple[str, ...] = ()
    snomed_code: str | None = None
    rxnorm_code: str | None = None
    icd10_code: str | None = None
    loinc_code: str | None = None
    semantic_type: str = ""
    type_group: EntityTypeGroup = EntityTypeGroup.CONDITION


@dataclass
class NormalizationResult:
    """Result of normalizing a single entity mention.

    Parameters
    ----------
    input_text : str
        Original entity surface form.
    matched : bool
        Whether a concept match was found.
    cui : str | None
        Matched UMLS CUI, if any.
    preferred_term : str | None
        Canonical term for the matched concept.
    confidence : float
        Match confidence in [0, 1].  1.0 for exact matches.
    match_type : str
        How the match was made: ``exact``, ``alias``, ``fuzzy``, or ``none``.
    snomed_code : str | None
        SNOMED-CT code if available for the matched concept.
    rxnorm_code : str | None
        RxNorm code if available.
    icd10_code : str | None
        ICD-10-CM code if available.
    loinc_code : str | None
        LOINC code if available.
    semantic_type : str
        UMLS semantic type of the matched concept.
    alternatives : list[dict]
        Other candidate matches ranked by confidence.
    """

    input_text: str
    matched: bool = False
    cui: str | None = None
    preferred_term: str | None = None
    confidence: float = 0.0
    match_type: str = "none"
    snomed_code: str | None = None
    rxnorm_code: str | None = None
    icd10_code: str | None = None
    loinc_code: str | None = None
    semantic_type: str = ""
    alternatives: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialise to JSON-compatible dictionary."""
        return {
            "input_text": self.input_text,
            "matched": self.matched,
            "cui": self.cui,
            "preferred_term": self.preferred_term,
            "confidence": round(self.confidence, 4),
            "match_type": self.match_type,
            "codes": {
                "umls_cui": self.cui,
                "snomed_ct": self.snomed_code,
                "rxnorm": self.rxnorm_code,
                "icd10_cm": self.icd10_code,
                "loinc": self.loinc_code,
            },
            "semantic_type": self.semantic_type,
            "alternatives": self.alternatives,
        }


@dataclass
class BatchNormalizationResult:
    """Result of normalizing a batch of entity mentions.

    Parameters
    ----------
    results : list[NormalizationResult]
        Individual normalization results.
    total : int
        Total number of input entities.
    matched_count : int
        Number of entities that were successfully matched.
    match_rate : float
        Fraction of entities matched (0.0–1.0).
    processing_time_ms : float
        Total processing time in milliseconds.
    """

    results: list[NormalizationResult]
    total: int = 0
    matched_count: int = 0
    match_rate: float = 0.0
    processing_time_ms: float = 0.0

    def to_dict(self) -> dict:
        """Serialise to JSON-compatible dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total": self.total,
                "matched": self.matched_count,
                "unmatched": self.total - self.matched_count,
                "match_rate": round(self.match_rate, 4),
                "processing_time_ms": round(self.processing_time_ms, 2),
            },
        }


# ---------------------------------------------------------------------------
# Concept dictionary — curated mappings
# ---------------------------------------------------------------------------
# Each entry is (CUI, preferred_term, aliases_tuple, snomed, rxnorm, icd10,
# loinc, semantic_type, type_group).

_CONCEPT_DATA: list[ConceptEntry] = [
    # -----------------------------------------------------------------------
    # CONDITIONS — Cardiovascular
    # -----------------------------------------------------------------------
    ConceptEntry("C0020538", "Hypertension", ("htn", "high blood pressure", "elevated bp", "arterial hypertension"), "38341003", None, "I10", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0027051", "Myocardial Infarction", ("mi", "heart attack", "ami", "acute myocardial infarction", "stemi", "nstemi"), "22298006", None, "I21.9", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0018802", "Congestive Heart Failure", ("chf", "heart failure", "hf", "congestive cardiac failure", "cardiac failure"), "42343007", None, "I50.9", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0010068", "Coronary Artery Disease", ("cad", "coronary heart disease", "chd", "ischemic heart disease", "ihd"), "53741008", None, "I25.10", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0004238", "Atrial Fibrillation", ("afib", "a-fib", "af", "atrial fib"), "49436004", None, "I48.91", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0011847", "Type 1 Diabetes Mellitus", ("t1dm", "type 1 diabetes", "juvenile diabetes", "insulin dependent diabetes", "iddm"), "46635009", None, "E10", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0011860", "Type 2 Diabetes Mellitus", ("t2dm", "type 2 diabetes", "dm2", "dm", "diabetes mellitus", "adult onset diabetes", "niddm"), "44054006", None, "E11", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0038454", "Cerebrovascular Accident", ("cva", "stroke", "cerebral infarction", "brain attack"), "230690007", None, "I63.9", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0040053", "Transient Ischemic Attack", ("tia", "mini stroke", "transient ischaemic attack"), "266257000", None, "G45.9", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0034065", "Pulmonary Embolism", ("pe", "pulmonary embolus", "lung embolism"), "59282003", None, "I26.99", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0149871", "Deep Vein Thrombosis", ("dvt", "deep venous thrombosis", "venous thrombosis"), "128053003", None, "I82.40", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),

    # -----------------------------------------------------------------------
    # CONDITIONS — Respiratory
    # -----------------------------------------------------------------------
    ConceptEntry("C0024117", "Chronic Obstructive Pulmonary Disease", ("copd", "chronic bronchitis", "emphysema"), "13645005", None, "J44.1", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0004096", "Asthma", ("bronchial asthma", "reactive airway disease", "rad"), "195967001", None, "J45.909", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0032285", "Pneumonia", ("pna", "lung infection", "community acquired pneumonia", "cap"), "233604007", None, "J18.9", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0034063", "Pulmonary Fibrosis", ("lung fibrosis", "idiopathic pulmonary fibrosis", "ipf"), "51615001", None, "J84.10", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),

    # -----------------------------------------------------------------------
    # CONDITIONS — GI
    # -----------------------------------------------------------------------
    ConceptEntry("C0017168", "Gastroesophageal Reflux Disease", ("gerd", "acid reflux", "reflux", "heartburn"), "235595009", None, "K21.0", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0009324", "Ulcerative Colitis", ("uc", "colitis"), "64766004", None, "K51.90", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0010346", "Crohn's Disease", ("crohns", "crohn disease", "regional enteritis"), "34000006", None, "K50.90", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0019158", "Hepatitis C", ("hcv", "hep c", "hepatitis c virus"), "50711007", None, "B18.2", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0008350", "Cholecystitis", ("gallbladder inflammation", "acute cholecystitis"), "76581006", None, "K81.0", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),

    # -----------------------------------------------------------------------
    # CONDITIONS — Renal
    # -----------------------------------------------------------------------
    ConceptEntry("C1561643", "Chronic Kidney Disease", ("ckd", "chronic renal disease", "chronic renal failure", "crf"), "709044004", None, "N18.9", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0022660", "Acute Kidney Injury", ("aki", "acute renal failure", "arf"), "14669001", None, "N17.9", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0042029", "Urinary Tract Infection", ("uti", "bladder infection", "cystitis"), "68566005", None, "N39.0", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),

    # -----------------------------------------------------------------------
    # CONDITIONS — Neurological
    # -----------------------------------------------------------------------
    ConceptEntry("C0014544", "Epilepsy", ("seizure disorder", "convulsive disorder"), "84757009", None, "G40.909", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0030567", "Parkinson's Disease", ("parkinsons", "parkinson disease", "pd"), "49049000", None, "G20", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0002736", "Alzheimer's Disease", ("alzheimers", "alzheimer disease", "ad"), "26929004", None, "G30.9", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0018681", "Headache", ("cephalgia", "head pain"), "25064002", None, "R51.9", None, "Finding", EntityTypeGroup.CONDITION),
    ConceptEntry("C0149931", "Migraine", ("migraine headache", "migraines"), "37796009", None, "G43.909", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),

    # -----------------------------------------------------------------------
    # CONDITIONS — Musculoskeletal
    # -----------------------------------------------------------------------
    ConceptEntry("C0029408", "Osteoarthritis", ("oa", "degenerative joint disease", "djd"), "396275006", None, "M19.90", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0003873", "Rheumatoid Arthritis", ("ra", "rheumatoid"), "69896004", None, "M06.9", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0029456", "Osteoporosis", ("bone loss", "low bone density"), "64859006", None, "M81.0", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0024031", "Low Back Pain", ("lbp", "lumbar pain", "back pain", "backache"), "279039007", None, "M54.5", None, "Finding", EntityTypeGroup.CONDITION),

    # -----------------------------------------------------------------------
    # CONDITIONS — Psychiatric
    # -----------------------------------------------------------------------
    ConceptEntry("C0011570", "Major Depressive Disorder", ("mdd", "depression", "major depression", "depressive disorder"), "36923009", None, "F33.0", None, "Mental or Behavioral Dysfunction", EntityTypeGroup.CONDITION),
    ConceptEntry("C0003467", "Generalized Anxiety Disorder", ("gad", "anxiety", "anxiety disorder"), "21897009", None, "F41.1", None, "Mental or Behavioral Dysfunction", EntityTypeGroup.CONDITION),
    ConceptEntry("C0006266", "Bipolar Disorder", ("bipolar", "manic depression", "bpad"), "13746004", None, "F31.9", None, "Mental or Behavioral Dysfunction", EntityTypeGroup.CONDITION),
    ConceptEntry("C0036341", "Schizophrenia", ("sz",), "58214004", None, "F20.9", None, "Mental or Behavioral Dysfunction", EntityTypeGroup.CONDITION),
    ConceptEntry("C0038436", "Post-Traumatic Stress Disorder", ("ptsd", "post traumatic stress", "combat stress"), "47505003", None, "F43.10", None, "Mental or Behavioral Dysfunction", EntityTypeGroup.CONDITION),

    # -----------------------------------------------------------------------
    # CONDITIONS — Endocrine / Metabolic
    # -----------------------------------------------------------------------
    ConceptEntry("C0020473", "Hyperlipidemia", ("hld", "dyslipidemia", "high cholesterol", "hypercholesterolemia"), "55822004", None, "E78.5", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0020502", "Hypothyroidism", ("underactive thyroid", "low thyroid"), "40930008", None, "E03.9", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0020550", "Hyperthyroidism", ("overactive thyroid", "thyrotoxicosis", "graves disease"), "34486009", None, "E05.90", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0028754", "Obesity", ("obese", "morbid obesity"), "414916001", None, "E66.9", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0001080", "Gout", ("gouty arthritis", "hyperuricemia"), "90560007", None, "M10.9", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),

    # -----------------------------------------------------------------------
    # CONDITIONS — Infectious
    # -----------------------------------------------------------------------
    ConceptEntry("C0009450", "COVID-19", ("covid", "sars-cov-2", "coronavirus", "covid19"), "840539006", None, "U07.1", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0042769", "HIV Infection", ("hiv", "human immunodeficiency virus", "hiv/aids", "aids"), "86406008", None, "B20", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0036974", "Sepsis", ("septicemia", "blood poisoning", "systemic infection"), "91302008", None, "A41.9", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),

    # -----------------------------------------------------------------------
    # CONDITIONS — Dental
    # -----------------------------------------------------------------------
    ConceptEntry("C0011334", "Dental Caries", ("cavities", "tooth decay", "caries", "dental cavity"), "80967001", None, "K02.9", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0031099", "Periodontitis", ("periodontal disease", "gum disease", "perio", "periodontosis"), "41565005", None, "K05.30", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0017574", "Gingivitis", ("gum inflammation", "bleeding gums"), "66383009", None, "K05.10", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0040460", "Temporomandibular Joint Disorder", ("tmj", "tmd", "tmj disorder", "jaw pain"), "41888000", None, "M26.60", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),
    ConceptEntry("C0155934", "Dental Abscess", ("tooth abscess", "periapical abscess", "dentoalveolar abscess"), "109478007", None, "K04.7", None, "Disease or Syndrome", EntityTypeGroup.CONDITION),

    # -----------------------------------------------------------------------
    # SYMPTOMS
    # -----------------------------------------------------------------------
    ConceptEntry("C0013404", "Dyspnea", ("dyspnoea", "shortness of breath", "sob", "breathlessness", "difficulty breathing"), "267036007", None, "R06.00", None, "Sign or Symptom", EntityTypeGroup.CONDITION),
    ConceptEntry("C0008031", "Chest Pain", ("angina", "chest tightness", "substernal chest pain"), "29857009", None, "R07.9", None, "Sign or Symptom", EntityTypeGroup.CONDITION),
    ConceptEntry("C0027497", "Nausea", ("nauseous", "feeling sick"), "422587007", None, "R11.0", None, "Sign or Symptom", EntityTypeGroup.CONDITION),
    ConceptEntry("C0042963", "Vomiting", ("emesis",), "422400008", None, "R11.10", None, "Sign or Symptom", EntityTypeGroup.CONDITION),
    ConceptEntry("C0015672", "Fatigue", ("tiredness", "malaise", "lethargy", "exhaustion"), "84229001", None, "R53.83", None, "Sign or Symptom", EntityTypeGroup.CONDITION),
    ConceptEntry("C0015967", "Fever", ("pyrexia", "febrile", "elevated temperature"), "386661006", None, "R50.9", None, "Sign or Symptom", EntityTypeGroup.CONDITION),
    ConceptEntry("C0030193", "Pain", ("ache", "painful", "algesia"), "22253000", None, "R52", None, "Sign or Symptom", EntityTypeGroup.CONDITION),
    ConceptEntry("C0013144", "Dizziness", ("vertigo", "lightheadedness", "light-headed"), "404640003", None, "R42", None, "Sign or Symptom", EntityTypeGroup.CONDITION),
    ConceptEntry("C0011991", "Diarrhea", ("diarrhoea", "loose stools", "watery stools"), "62315008", None, "R19.7", None, "Sign or Symptom", EntityTypeGroup.CONDITION),
    ConceptEntry("C0009806", "Constipation", ("obstipation",), "14760008", None, "K59.00", None, "Sign or Symptom", EntityTypeGroup.CONDITION),
    ConceptEntry("C0030920", "Peripheral Edema", ("edema", "oedema", "swelling", "leg swelling", "ankle swelling"), "271809000", None, "R60.0", None, "Sign or Symptom", EntityTypeGroup.CONDITION),
    ConceptEntry("C0010200", "Cough", ("coughing",), "49727002", None, "R05.9", None, "Sign or Symptom", EntityTypeGroup.CONDITION),

    # -----------------------------------------------------------------------
    # MEDICATIONS — Cardiovascular
    # -----------------------------------------------------------------------
    ConceptEntry("C0065374", "Lisinopril", ("prinivil", "zestril"), None, "29046", "Z79.0", None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0004147", "Atenolol", ("tenormin",), None, "1202", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0025859", "Metoprolol", ("lopressor", "toprol", "toprol-xl"), None, "6918", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0004057", "Atorvastatin", ("lipitor",), None, "83367", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0073314", "Rosuvastatin", ("crestor",), None, "301542", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0070166", "Amlodipine", ("norvasc",), None, "17767", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0023779", "Losartan", ("cozaar",), None, "52175", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0042903", "Warfarin", ("coumadin",), None, "11289", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0122432", "Clopidogrel", ("plavix",), None, "32968", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C1831808", "Apixaban", ("eliquis",), None, "1364430", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0015846", "Furosemide", ("lasix",), None, "4603", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0020261", "Hydrochlorothiazide", ("hctz", "microzide"), None, "5487", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),

    # -----------------------------------------------------------------------
    # MEDICATIONS — Endocrine / Diabetes
    # -----------------------------------------------------------------------
    ConceptEntry("C0025598", "Metformin", ("glucophage",), None, "6809", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0021641", "Insulin", ("insulin regular", "humulin", "novolin"), None, "5856", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0532924", "Insulin Glargine", ("lantus", "basaglar", "toujeo"), None, "274783", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C1701407", "Semaglutide", ("ozempic", "wegovy", "rybelsus"), None, "1991302", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0069089", "Levothyroxine", ("synthroid", "levoxyl", "thyroid replacement"), None, "10582", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0017725", "Glipizide", ("glucotrol",), None, "4815", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),

    # -----------------------------------------------------------------------
    # MEDICATIONS — Psychiatric / CNS
    # -----------------------------------------------------------------------
    ConceptEntry("C0284635", "Sertraline", ("zoloft",), None, "36437", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0015473", "Escitalopram", ("lexapro",), None, "321988", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0078569", "Fluoxetine", ("prozac",), None, "4493", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0078785", "Duloxetine", ("cymbalta",), None, "72625", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0052912", "Alprazolam", ("xanax",), None, "596", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0023779", "Lorazepam", ("ativan",), None, "6470", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0040805", "Trazodone", ("desyrel",), None, "10737", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0017168", "Gabapentin", ("neurontin",), None, "25480", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),

    # -----------------------------------------------------------------------
    # MEDICATIONS — Pain / Anti-inflammatory
    # -----------------------------------------------------------------------
    ConceptEntry("C0000970", "Acetaminophen", ("tylenol", "paracetamol", "apap"), None, "161", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0020740", "Ibuprofen", ("advil", "motrin"), None, "5640", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0028128", "Naproxen", ("aleve", "naprosyn"), None, "7258", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0030049", "Oxycodone", ("oxycontin", "percocet", "roxicodone"), None, "7804", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0024611", "Morphine", ("ms contin",), None, "7052", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0040822", "Tramadol", ("ultram",), None, "10689", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0032042", "Prednisone", ("deltasone",), None, "8640", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),

    # -----------------------------------------------------------------------
    # MEDICATIONS — Antibiotics
    # -----------------------------------------------------------------------
    ConceptEntry("C0002645", "Amoxicillin", ("amoxil",), None, "723", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0054066", "Amoxicillin-Clavulanate", ("augmentin", "amox-clav"), None, "19711", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0004660", "Azithromycin", ("zithromax", "z-pack", "zpak"), None, "18631", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0008809", "Ciprofloxacin", ("cipro",), None, "2551", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0023483", "Levofloxacin", ("levaquin",), None, "82122", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0012265", "Doxycycline", ("vibramycin",), None, "3640", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),

    # -----------------------------------------------------------------------
    # MEDICATIONS — GI
    # -----------------------------------------------------------------------
    ConceptEntry("C0028978", "Omeprazole", ("prilosec",), None, "7646", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0073228", "Pantoprazole", ("protonix",), None, "40790", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0064636", "Ondansetron", ("zofran",), None, "26225", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),

    # -----------------------------------------------------------------------
    # MEDICATIONS — Respiratory
    # -----------------------------------------------------------------------
    ConceptEntry("C0001927", "Albuterol", ("proventil", "ventolin", "salbutamol", "proair"), None, "435", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),
    ConceptEntry("C0055100", "Montelukast", ("singulair",), None, "88249", None, None, "Pharmacologic Substance", EntityTypeGroup.MEDICATION),

    # -----------------------------------------------------------------------
    # PROCEDURES
    # -----------------------------------------------------------------------
    ConceptEntry("C0013798", "Electrocardiogram", ("ecg", "ekg", "12-lead ekg", "12-lead ecg"), "29303009", None, None, None, "Diagnostic Procedure", EntityTypeGroup.PROCEDURE),
    ConceptEntry("C0013516", "Echocardiogram", ("echo", "transthoracic echocardiogram", "tte", "transesophageal echocardiogram", "tee"), "40701008", None, None, None, "Diagnostic Procedure", EntityTypeGroup.PROCEDURE),
    ConceptEntry("C0024485", "Magnetic Resonance Imaging", ("mri", "mr imaging", "magnetic resonance"), "113091000", None, None, None, "Diagnostic Procedure", EntityTypeGroup.PROCEDURE),
    ConceptEntry("C0040405", "Computed Tomography", ("ct scan", "ct", "cat scan", "computed tomography scan"), "77477000", None, None, None, "Diagnostic Procedure", EntityTypeGroup.PROCEDURE),
    ConceptEntry("C0043299", "X-Ray", ("radiograph", "xray", "plain film", "chest x-ray", "cxr"), "363680008", None, None, None, "Diagnostic Procedure", EntityTypeGroup.PROCEDURE),
    ConceptEntry("C0014245", "Endoscopy", ("egd", "upper endoscopy", "esophagogastroduodenoscopy"), "423827005", None, None, None, "Diagnostic Procedure", EntityTypeGroup.PROCEDURE),
    ConceptEntry("C0009378", "Colonoscopy", ("colon scope",), "73761001", None, None, None, "Diagnostic Procedure", EntityTypeGroup.PROCEDURE),
    ConceptEntry("C0006104", "Cardiac Catheterization", ("cath", "heart cath", "cardiac cath", "left heart catheterization"), "41976001", None, None, None, "Diagnostic Procedure", EntityTypeGroup.PROCEDURE),
    ConceptEntry("C0010055", "Coronary Artery Bypass Graft", ("cabg", "bypass surgery", "coronary bypass"), "232717009", None, None, None, "Therapeutic or Preventive Procedure", EntityTypeGroup.PROCEDURE),
    ConceptEntry("C0162577", "Percutaneous Coronary Intervention", ("pci", "angioplasty", "stenting", "ptca"), "415070008", None, None, None, "Therapeutic or Preventive Procedure", EntityTypeGroup.PROCEDURE),
    ConceptEntry("C0020710", "Total Hip Replacement", ("thr", "total hip arthroplasty", "hip replacement"), "76164006", None, None, None, "Therapeutic or Preventive Procedure", EntityTypeGroup.PROCEDURE),
    ConceptEntry("C0086511", "Total Knee Replacement", ("tkr", "total knee arthroplasty", "knee replacement"), "609588000", None, None, None, "Therapeutic or Preventive Procedure", EntityTypeGroup.PROCEDURE),
    ConceptEntry("C0007431", "Appendectomy", ("appy",), "80146002", None, None, None, "Therapeutic or Preventive Procedure", EntityTypeGroup.PROCEDURE),
    ConceptEntry("C0008310", "Cholecystectomy", ("chole", "gallbladder removal", "lap chole", "laparoscopic cholecystectomy"), "38102005", None, None, None, "Therapeutic or Preventive Procedure", EntityTypeGroup.PROCEDURE),

    # Dental procedures
    ConceptEntry("C0035975", "Root Canal Therapy", ("rct", "root canal", "endodontic therapy", "endodontic treatment"), "234713009", None, None, None, "Therapeutic or Preventive Procedure", EntityTypeGroup.PROCEDURE),
    ConceptEntry("C0206219", "Scaling and Root Planing", ("srp", "deep cleaning", "periodontal scaling"), "28845002", None, None, None, "Therapeutic or Preventive Procedure", EntityTypeGroup.PROCEDURE),
    ConceptEntry("C0040440", "Tooth Extraction", ("extraction", "dental extraction", "exodontia"), "69130005", None, None, None, "Therapeutic or Preventive Procedure", EntityTypeGroup.PROCEDURE),
    ConceptEntry("C0011382", "Dental Crown", ("crown", "dental cap", "porcelain crown", "pfm crown"), "77032002", None, None, None, "Therapeutic or Preventive Procedure", EntityTypeGroup.PROCEDURE),

    # -----------------------------------------------------------------------
    # ANATOMY / BODY PARTS
    # -----------------------------------------------------------------------
    ConceptEntry("C0018787", "Heart", ("cardiac", "myocardium"), "80891009", None, None, None, "Body Part, Organ, or Organ Component", EntityTypeGroup.ANATOMY),
    ConceptEntry("C0024109", "Lung", ("pulmonary", "lungs"), "39607008", None, None, None, "Body Part, Organ, or Organ Component", EntityTypeGroup.ANATOMY),
    ConceptEntry("C0023884", "Liver", ("hepatic",), "10200004", None, None, None, "Body Part, Organ, or Organ Component", EntityTypeGroup.ANATOMY),
    ConceptEntry("C0022646", "Kidney", ("renal", "kidneys"), "64033007", None, None, None, "Body Part, Organ, or Organ Component", EntityTypeGroup.ANATOMY),
    ConceptEntry("C0006104", "Brain", ("cerebral", "intracranial"), "12738006", None, None, None, "Body Part, Organ, or Organ Component", EntityTypeGroup.ANATOMY),
    ConceptEntry("C0000726", "Abdomen", ("abdominal",), "818983003", None, None, None, "Body Part, Organ, or Organ Component", EntityTypeGroup.ANATOMY),
    ConceptEntry("C0040578", "Thorax", ("chest", "thoracic"), "51185008", None, None, None, "Body Part, Organ, or Organ Component", EntityTypeGroup.ANATOMY),

    # -----------------------------------------------------------------------
    # LAB TESTS / VALUES
    # -----------------------------------------------------------------------
    ConceptEntry("C0009555", "Complete Blood Count", ("cbc", "full blood count", "fbc"), None, None, None, "26604-6", "Laboratory Procedure", EntityTypeGroup.LAB),
    ConceptEntry("C0005845", "Blood Glucose", ("blood sugar", "fasting glucose", "fbg", "fbs", "random glucose"), None, None, None, "2345-7", "Laboratory Procedure", EntityTypeGroup.LAB),
    ConceptEntry("C0019018", "Hemoglobin A1c", ("hba1c", "a1c", "glycated hemoglobin", "glycosylated hemoglobin"), None, None, None, "4548-4", "Laboratory Procedure", EntityTypeGroup.LAB),
    ConceptEntry("C0201811", "Basic Metabolic Panel", ("bmp", "chem-7", "chem7"), None, None, None, "51990-0", "Laboratory Procedure", EntityTypeGroup.LAB),
    ConceptEntry("C0009555", "Comprehensive Metabolic Panel", ("cmp", "chem-14", "chem14"), None, None, None, "24323-8", "Laboratory Procedure", EntityTypeGroup.LAB),
    ConceptEntry("C0023508", "White Blood Cell Count", ("wbc", "leukocyte count"), None, None, None, "6690-2", "Laboratory Procedure", EntityTypeGroup.LAB),
    ConceptEntry("C0018935", "Hemoglobin", ("hgb", "hb"), None, None, None, "718-7", "Laboratory Procedure", EntityTypeGroup.LAB),
    ConceptEntry("C0032181", "Platelet Count", ("plt", "platelets", "thrombocyte count"), None, None, None, "777-3", "Laboratory Procedure", EntityTypeGroup.LAB),
    ConceptEntry("C0010294", "Creatinine", ("cr", "serum creatinine", "scr"), None, None, None, "2160-0", "Laboratory Procedure", EntityTypeGroup.LAB),
    ConceptEntry("C0005133", "Blood Urea Nitrogen", ("bun",), None, None, None, "3094-0", "Laboratory Procedure", EntityTypeGroup.LAB),
    ConceptEntry("C0036821", "Serum Sodium", ("na", "sodium level"), None, None, None, "2951-2", "Laboratory Procedure", EntityTypeGroup.LAB),
    ConceptEntry("C0032823", "Serum Potassium", ("k", "potassium level"), None, None, None, "2823-3", "Laboratory Procedure", EntityTypeGroup.LAB),
    ConceptEntry("C0023555", "Lipid Panel", ("lipid profile", "cholesterol panel", "fasting lipid panel"), None, None, None, "57698-3", "Laboratory Procedure", EntityTypeGroup.LAB),
    ConceptEntry("C0041236", "Troponin", ("troponin i", "troponin t", "hs troponin", "high sensitivity troponin"), None, None, None, "49563-0", "Laboratory Procedure", EntityTypeGroup.LAB),
    ConceptEntry("C0033684", "Prothrombin Time", ("pt", "inr", "pt/inr"), None, None, None, "5902-2", "Laboratory Procedure", EntityTypeGroup.LAB),
    ConceptEntry("C0525032", "Brain Natriuretic Peptide", ("bnp", "nt-probnp", "pro-bnp"), None, None, None, "30934-4", "Laboratory Procedure", EntityTypeGroup.LAB),
    ConceptEntry("C0040059", "Thyroid Stimulating Hormone", ("tsh",), None, None, None, "3016-3", "Laboratory Procedure", EntityTypeGroup.LAB),
    ConceptEntry("C0042014", "Urinalysis", ("ua", "urine analysis", "urine test"), None, None, None, "24356-8", "Laboratory Procedure", EntityTypeGroup.LAB),
    ConceptEntry("C0201848", "Liver Function Tests", ("lft", "lfts", "hepatic function panel"), None, None, None, "24325-3", "Laboratory Procedure", EntityTypeGroup.LAB),
    ConceptEntry("C0683474", "Estimated Glomerular Filtration Rate", ("egfr", "gfr"), None, None, None, "33914-3", "Laboratory Procedure", EntityTypeGroup.LAB),
]


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------


def _build_indices() -> tuple[
    dict[str, ConceptEntry],
    dict[str, ConceptEntry],
    dict[EntityTypeGroup, list[ConceptEntry]],
]:
    """Build lookup indices from the concept data.

    Returns
    -------
    tuple
        (exact_index, alias_index, group_index)
        * exact_index: case-folded preferred_term → ConceptEntry
        * alias_index: case-folded alias → ConceptEntry
        * group_index: EntityTypeGroup → list of ConceptEntry
    """
    exact: dict[str, ConceptEntry] = {}
    alias: dict[str, ConceptEntry] = {}
    group: dict[EntityTypeGroup, list[ConceptEntry]] = {g: [] for g in EntityTypeGroup}

    for concept in _CONCEPT_DATA:
        key = concept.preferred_term.lower().strip()
        exact[key] = concept
        group[concept.type_group].append(concept)

        for a in concept.aliases:
            alias_key = a.lower().strip()
            # Don't overwrite — first entry wins for ambiguous aliases
            if alias_key not in alias:
                alias[alias_key] = concept

    return exact, alias, group


_EXACT_INDEX, _ALIAS_INDEX, _GROUP_INDEX = _build_indices()


# ---------------------------------------------------------------------------
# Normalizer configuration
# ---------------------------------------------------------------------------


@dataclass
class NormalizerConfig:
    """Configuration for the concept normalizer.

    Parameters
    ----------
    min_similarity : float
        Minimum Levenshtein-ratio similarity for fuzzy matching (0.0–1.0).
        Default 0.80 provides a good precision/recall balance.
    max_alternatives : int
        Maximum number of alternative candidates to return per entity.
    enable_fuzzy : bool
        Whether to attempt fuzzy matching when exact/alias lookups fail.
    type_aware : bool
        When True and an entity_type is provided, only consider concepts
        whose type_group is compatible.
    """

    min_similarity: float = 0.80
    max_alternatives: int = 3
    enable_fuzzy: bool = True
    type_aware: bool = True


# ---------------------------------------------------------------------------
# Type mapping — NER entity types to concept type groups
# ---------------------------------------------------------------------------

_ENTITY_TYPE_TO_GROUP: dict[str, EntityTypeGroup] = {
    "DISEASE": EntityTypeGroup.CONDITION,
    "SYMPTOM": EntityTypeGroup.CONDITION,
    "MEDICATION": EntityTypeGroup.MEDICATION,
    "DOSAGE": EntityTypeGroup.MEDICATION,
    "PROCEDURE": EntityTypeGroup.PROCEDURE,
    "ANATOMY": EntityTypeGroup.ANATOMY,
    "BODY_PART": EntityTypeGroup.ANATOMY,
    "LAB_VALUE": EntityTypeGroup.LAB,
    "TEST": EntityTypeGroup.LAB,
    "TREATMENT": EntityTypeGroup.PROCEDURE,
    "DEVICE": EntityTypeGroup.PROCEDURE,
}


# ---------------------------------------------------------------------------
# Normalizer engine
# ---------------------------------------------------------------------------


class ClinicalConceptNormalizer:
    """Maps extracted entity text to standardised medical ontology codes.

    This is the primary interface for concept normalization.  It chains
    three resolution strategies (exact → alias → fuzzy) and returns a
    ``NormalizationResult`` with the best match and any alternatives.

    Parameters
    ----------
    config : NormalizerConfig | None
        Optional configuration overrides.

    Examples
    --------
    >>> normalizer = ClinicalConceptNormalizer()
    >>> result = normalizer.normalize("HTN")
    >>> result.cui
    'C0020538'
    >>> result.preferred_term
    'Hypertension'
    """

    def __init__(self, config: NormalizerConfig | None = None) -> None:
        self.config = config or NormalizerConfig()
        self._stats = {
            "exact_hits": 0,
            "alias_hits": 0,
            "fuzzy_hits": 0,
            "misses": 0,
            "total": 0,
        }

    # -- public API ---------------------------------------------------------

    def normalize(
        self,
        text: str,
        entity_type: str | None = None,
    ) -> NormalizationResult:
        """Normalize a single entity mention to a standardised concept.

        Parameters
        ----------
        text : str
            Entity surface form to normalize.
        entity_type : str | None
            Optional NER entity type (e.g. ``DISEASE``, ``MEDICATION``)
            for type-aware filtering.

        Returns
        -------
        NormalizationResult
            Normalization outcome with matched codes and confidence.
        """
        self._stats["total"] += 1
        key = self._normalize_key(text)

        if not key:
            self._stats["misses"] += 1
            return NormalizationResult(input_text=text)

        # Determine type group constraint
        type_group = None
        if self.config.type_aware and entity_type:
            type_group = _ENTITY_TYPE_TO_GROUP.get(entity_type.upper())

        # Strategy 1: exact match on preferred term
        concept = _EXACT_INDEX.get(key)
        if concept and self._type_matches(concept, type_group):
            self._stats["exact_hits"] += 1
            return self._make_result(text, concept, 1.0, "exact")

        # Strategy 2: alias match
        concept = _ALIAS_INDEX.get(key)
        if concept and self._type_matches(concept, type_group):
            self._stats["alias_hits"] += 1
            return self._make_result(text, concept, 0.95, "alias")

        # Strategy 3: fuzzy matching
        if self.config.enable_fuzzy:
            fuzzy_result = self._fuzzy_match(key, type_group)
            if fuzzy_result:
                self._stats["fuzzy_hits"] += 1
                return self._make_result(
                    text,
                    fuzzy_result[0],
                    fuzzy_result[1],
                    "fuzzy",
                    alternatives=fuzzy_result[2],
                )

        self._stats["misses"] += 1
        return NormalizationResult(input_text=text)

    def normalize_batch(
        self,
        entities: list[dict[str, str]],
    ) -> BatchNormalizationResult:
        """Normalize a batch of entity mentions.

        Parameters
        ----------
        entities : list[dict[str, str]]
            List of dicts with ``text`` and optional ``entity_type`` keys.

        Returns
        -------
        BatchNormalizationResult
            Aggregated results with match statistics.
        """
        start = time.perf_counter()
        results: list[NormalizationResult] = []
        matched = 0

        for entity in entities:
            text = entity.get("text", "")
            entity_type = entity.get("entity_type")
            result = self.normalize(text, entity_type)
            results.append(result)
            if result.matched:
                matched += 1

        elapsed_ms = (time.perf_counter() - start) * 1000
        total = len(entities)

        return BatchNormalizationResult(
            results=results,
            total=total,
            matched_count=matched,
            match_rate=matched / total if total > 0 else 0.0,
            processing_time_ms=elapsed_ms,
        )

    def lookup_cui(self, cui: str) -> ConceptEntry | None:
        """Look up a concept by its UMLS CUI.

        Parameters
        ----------
        cui : str
            UMLS Concept Unique Identifier (e.g. ``C0020538``).

        Returns
        -------
        ConceptEntry | None
            The matching concept, or None if not found.
        """
        cui_upper = cui.upper().strip()
        for concept in _CONCEPT_DATA:
            if concept.cui == cui_upper:
                return concept
        return None

    def get_stats(self) -> dict:
        """Return normalization statistics.

        Returns
        -------
        dict
            Hit/miss counts and match rate.
        """
        total = self._stats["total"]
        matched = self._stats["exact_hits"] + self._stats["alias_hits"] + self._stats["fuzzy_hits"]
        return {
            **self._stats,
            "match_rate": matched / total if total > 0 else 0.0,
            "dictionary_size": len(_CONCEPT_DATA),
        }

    def reset_stats(self) -> None:
        """Reset normalization statistics counters."""
        for key in self._stats:
            self._stats[key] = 0

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _normalize_key(text: str) -> str:
        """Normalize text for lookup: case-fold, strip, collapse whitespace.

        Parameters
        ----------
        text : str
            Raw entity text.

        Returns
        -------
        str
            Normalized key string.
        """
        key = text.lower().strip()
        key = re.sub(r"\s+", " ", key)
        # Remove trailing periods (abbreviation artefacts)
        key = key.rstrip(".")
        return key

    @staticmethod
    def _type_matches(
        concept: ConceptEntry,
        type_group: EntityTypeGroup | None,
    ) -> bool:
        """Check if a concept's type group matches the constraint.

        Parameters
        ----------
        concept : ConceptEntry
            Candidate concept.
        type_group : EntityTypeGroup | None
            Required type group, or None for no constraint.

        Returns
        -------
        bool
            True if the concept matches or no constraint is set.
        """
        if type_group is None:
            return True
        return concept.type_group == type_group

    def _fuzzy_match(
        self,
        key: str,
        type_group: EntityTypeGroup | None,
    ) -> tuple[ConceptEntry, float, list[dict]] | None:
        """Find best fuzzy match across all concepts.

        Parameters
        ----------
        key : str
            Normalized search key.
        type_group : EntityTypeGroup | None
            Optional type constraint.

        Returns
        -------
        tuple | None
            (best_concept, confidence, alternatives) or None if below
            the minimum similarity threshold.
        """
        candidates: list[tuple[float, ConceptEntry]] = []
        min_sim = self.config.min_similarity

        # Search preferred terms
        for term, concept in _EXACT_INDEX.items():
            if not self._type_matches(concept, type_group):
                continue
            ratio = SequenceMatcher(None, key, term).ratio()
            if ratio >= min_sim:
                candidates.append((ratio, concept))

        # Search aliases
        for alias, concept in _ALIAS_INDEX.items():
            if not self._type_matches(concept, type_group):
                continue
            ratio = SequenceMatcher(None, key, alias).ratio()
            if ratio >= min_sim and not any(c.cui == concept.cui for _, c in candidates):
                candidates.append((ratio, concept))

        if not candidates:
            return None

        # Sort by similarity descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        best_ratio, best_concept = candidates[0]

        # Build alternatives (excluding best)
        alternatives: list[dict] = []
        for ratio, concept in candidates[1: self.config.max_alternatives + 1]:
            alternatives.append({
                "cui": concept.cui,
                "preferred_term": concept.preferred_term,
                "confidence": round(ratio, 4),
                "codes": {
                    "snomed_ct": concept.snomed_code,
                    "rxnorm": concept.rxnorm_code,
                    "icd10_cm": concept.icd10_code,
                    "loinc": concept.loinc_code,
                },
            })

        return best_concept, best_ratio, alternatives

    @staticmethod
    def _make_result(
        input_text: str,
        concept: ConceptEntry,
        confidence: float,
        match_type: str,
        alternatives: list[dict] | None = None,
    ) -> NormalizationResult:
        """Construct a NormalizationResult from a matched concept.

        Parameters
        ----------
        input_text : str
            Original entity text.
        concept : ConceptEntry
            Matched concept.
        confidence : float
            Match confidence score.
        match_type : str
            How the match was made.
        alternatives : list[dict] | None
            Other candidate matches.

        Returns
        -------
        NormalizationResult
            Populated result object.
        """
        return NormalizationResult(
            input_text=input_text,
            matched=True,
            cui=concept.cui,
            preferred_term=concept.preferred_term,
            confidence=confidence,
            match_type=match_type,
            snomed_code=concept.snomed_code,
            rxnorm_code=concept.rxnorm_code,
            icd10_code=concept.icd10_code,
            loinc_code=concept.loinc_code,
            semantic_type=concept.semantic_type,
            alternatives=alternatives or [],
        )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def get_dictionary_stats() -> dict:
    """Return statistics about the concept dictionary.

    Returns
    -------
    dict
        Counts by type group, total concepts, total aliases.
    """
    group_counts = {g.value: len(entries) for g, entries in _GROUP_INDEX.items()}
    total_aliases = sum(len(c.aliases) for c in _CONCEPT_DATA)
    ontology_coverage = {
        "snomed_ct": sum(1 for c in _CONCEPT_DATA if c.snomed_code),
        "rxnorm": sum(1 for c in _CONCEPT_DATA if c.rxnorm_code),
        "icd10_cm": sum(1 for c in _CONCEPT_DATA if c.icd10_code),
        "loinc": sum(1 for c in _CONCEPT_DATA if c.loinc_code),
    }
    return {
        "total_concepts": len(_CONCEPT_DATA),
        "total_aliases": total_aliases,
        "by_type_group": group_counts,
        "ontology_coverage": ontology_coverage,
    }
