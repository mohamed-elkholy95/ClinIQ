"""Medical query expansion for clinical document search.

Enhances recall by expanding user queries with medical synonyms,
abbreviations, and related terms before they reach the search engine.
This addresses a core challenge in clinical NLP: clinicians use
inconsistent terminology — "HTN" vs "hypertension", "MI" vs "myocardial
infarction", "DM" vs "diabetes mellitus" — and a pure lexical search
would miss relevant documents that use the alternate form.

Architecture
------------
* **Abbreviation map** — Bidirectional lookup between common medical
  abbreviations and their full forms.  Sourced from standard medical
  abbreviation lists (JCAHO, ISMP) augmented with dental terminology.
* **Synonym groups** — Clusters of interchangeable terms.  When any
  member appears in a query, all other members are appended as
  expansion terms with a configurable weight discount.
* **Spelling normalisation** — British/American spelling variants
  (anaemia/anemia, oedema/edema) are treated as synonyms.
* **Controlled expansion** — A maximum expansion factor caps the
  number of added terms to prevent query drift, where too many
  expansions dilute the original intent.

Thread safety
-------------
The synonym and abbreviation dictionaries are immutable after module
load and can be read concurrently without synchronisation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Medical abbreviation map (bidirectional)
# ---------------------------------------------------------------------------

# fmt: off
_ABBREVIATION_TO_FULL: dict[str, str] = {
    # Cardiovascular
    "htn": "hypertension",
    "mi": "myocardial infarction",
    "chf": "congestive heart failure",
    "afib": "atrial fibrillation",
    "cad": "coronary artery disease",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "bp": "blood pressure",
    "hr": "heart rate",
    "ef": "ejection fraction",
    "lvef": "left ventricular ejection fraction",
    "cabg": "coronary artery bypass graft",
    "pci": "percutaneous coronary intervention",
    "aaa": "abdominal aortic aneurysm",
    "avr": "aortic valve replacement",

    # Endocrine / Metabolic
    "dm": "diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
    "t1dm": "type 1 diabetes mellitus",
    "hba1c": "glycated hemoglobin",
    "a1c": "glycated hemoglobin",
    "tsh": "thyroid stimulating hormone",
    "dka": "diabetic ketoacidosis",
    "bmi": "body mass index",

    # Respiratory
    "copd": "chronic obstructive pulmonary disease",
    "sob": "shortness of breath",
    "ards": "acute respiratory distress syndrome",
    "osa": "obstructive sleep apnea",
    "pft": "pulmonary function test",
    "o2": "oxygen",
    "spo2": "oxygen saturation",

    # Gastrointestinal
    "gerd": "gastroesophageal reflux disease",
    "gi": "gastrointestinal",
    "gib": "gastrointestinal bleeding",
    "egd": "esophagogastroduodenoscopy",
    "ercp": "endoscopic retrograde cholangiopancreatography",
    "lfts": "liver function tests",
    "alt": "alanine aminotransferase",
    "ast": "aspartate aminotransferase",

    # Renal
    "ckd": "chronic kidney disease",
    "esrd": "end stage renal disease",
    "aki": "acute kidney injury",
    "bun": "blood urea nitrogen",
    "gfr": "glomerular filtration rate",
    "egfr": "estimated glomerular filtration rate",
    "uti": "urinary tract infection",

    # Neurological
    "cva": "cerebrovascular accident",
    "tia": "transient ischemic attack",
    "ms": "multiple sclerosis",
    "sz": "seizure",
    "loc": "loss of consciousness",
    "aox3": "alert and oriented times three",

    # Infectious
    "uri": "upper respiratory infection",
    "rti": "respiratory tract infection",
    "mrsa": "methicillin resistant staphylococcus aureus",
    "vre": "vancomycin resistant enterococcus",
    "hiv": "human immunodeficiency virus",
    "tb": "tuberculosis",

    # Musculoskeletal
    "oa": "osteoarthritis",
    "ra": "rheumatoid arthritis",
    "rom": "range of motion",
    "fx": "fracture",
    "orif": "open reduction internal fixation",
    "tkr": "total knee replacement",
    "thr": "total hip replacement",

    # Dental / Oral
    "perio": "periodontal",
    "cdt": "current dental terminology",
    "bwx": "bitewing xray",
    "pano": "panoramic radiograph",
    "srp": "scaling and root planing",
    "rct": "root canal treatment",
    "rpd": "removable partial denture",
    "fpd": "fixed partial denture",
    "tmj": "temporomandibular joint",
    "tmd": "temporomandibular disorder",
    "cal": "clinical attachment loss",
    "bop": "bleeding on probing",

    # Labs / Diagnostics
    "cbc": "complete blood count",
    "bmp": "basic metabolic panel",
    "cmp": "comprehensive metabolic panel",
    "wbc": "white blood cell",
    "rbc": "red blood cell",
    "hgb": "hemoglobin",
    "hct": "hematocrit",
    "plt": "platelet",
    "inr": "international normalized ratio",
    "pt": "prothrombin time",
    "ptt": "partial thromboplastin time",
    "bnp": "brain natriuretic peptide",
    "crp": "c reactive protein",
    "esr": "erythrocyte sedimentation rate",
    "ct": "computed tomography",
    "mri": "magnetic resonance imaging",
    "us": "ultrasound",
    "ekg": "electrocardiogram",
    "ecg": "electrocardiogram",
    "echo": "echocardiogram",

    # Medications / Orders
    "abx": "antibiotics",
    "nsaid": "nonsteroidal anti inflammatory drug",
    "ppi": "proton pump inhibitor",
    "ace": "angiotensin converting enzyme",
    "arb": "angiotensin receptor blocker",
    "ssri": "selective serotonin reuptake inhibitor",
    "prn": "as needed",
    "bid": "twice daily",
    "tid": "three times daily",
    "qid": "four times daily",
    "qd": "daily",

    # General clinical
    "hx": "history",
    "dx": "diagnosis",
    "rx": "prescription",
    "tx": "treatment",
    "sx": "symptoms",
    "pmh": "past medical history",
    "fh": "family history",
    "sh": "social history",
    "ros": "review of systems",
    "a&p": "assessment and plan",
    "dc": "discharge",
    "f/u": "follow up",
    "nkda": "no known drug allergies",
    "nka": "no known allergies",
    "aox": "alert and oriented",
    "wn": "well nourished",
    "wd": "well developed",
    "nad": "no acute distress",
    "nsr": "normal sinus rhythm",
    "rrr": "regular rate and rhythm",
    "ctab": "clear to auscultation bilaterally",
    "ntnd": "nontender nondistended",
}
# fmt: on

# Build reverse map (full form → abbreviation)
_FULL_TO_ABBREVIATION: dict[str, str] = {}
for _abbr, _full in _ABBREVIATION_TO_FULL.items():
    _FULL_TO_ABBREVIATION[_full.lower()] = _abbr

# ---------------------------------------------------------------------------
# Synonym groups — sets of interchangeable clinical terms
# ---------------------------------------------------------------------------

_SYNONYM_GROUPS: list[set[str]] = [
    # Cardiovascular
    {"hypertension", "high blood pressure", "elevated blood pressure", "htn"},
    {"myocardial infarction", "heart attack", "mi", "stemi", "nstemi"},
    {"stroke", "cerebrovascular accident", "cva"},
    {"atrial fibrillation", "afib", "a-fib", "af"},

    # Endocrine
    {"diabetes mellitus", "diabetes", "dm", "diabetic"},
    {"type 2 diabetes", "t2dm", "dm2", "type ii diabetes", "niddm"},
    {"type 1 diabetes", "t1dm", "dm1", "type i diabetes", "iddm"},
    {"hypothyroidism", "underactive thyroid", "low thyroid"},
    {"hyperthyroidism", "overactive thyroid", "thyrotoxicosis"},

    # Respiratory
    {"shortness of breath", "dyspnea", "dyspnoea", "sob", "breathlessness"},
    {"chronic obstructive pulmonary disease", "copd", "emphysema"},
    {"pneumonia", "pna", "lung infection"},

    # Pain / Symptoms
    {"pain", "ache", "discomfort", "tenderness", "soreness"},
    {"headache", "cephalalgia", "ha"},
    {"nausea", "queasy", "nauseated"},
    {"fever", "pyrexia", "febrile", "elevated temperature"},
    {"edema", "oedema", "swelling", "fluid retention"},
    {"fatigue", "tiredness", "malaise", "lethargy", "asthenia"},

    # GI
    {"gastroesophageal reflux", "gerd", "acid reflux", "heartburn"},
    {"constipation", "obstipation", "decreased bowel movements"},
    {"diarrhea", "diarrhoea", "loose stools"},

    # Renal
    {"chronic kidney disease", "ckd", "chronic renal failure", "crf"},
    {"acute kidney injury", "aki", "acute renal failure", "arf"},

    # Spelling variants (British / American)
    {"anemia", "anaemia"},
    {"hemorrhage", "haemorrhage"},
    {"leukemia", "leukaemia"},
    {"tumor", "tumour"},
    {"color", "colour"},
    {"fiber", "fibre"},
    {"hemoglobin", "haemoglobin"},
    {"pediatric", "paediatric"},
    {"orthopedic", "orthopaedic"},
    {"gynecology", "gynaecology"},
    {"esophagus", "oesophagus"},
    {"estrogen", "oestrogen"},

    # Dental
    {"periodontal disease", "gum disease", "periodontitis", "perio"},
    {"dental caries", "tooth decay", "cavity", "cavities"},
    {"extraction", "exodontia", "tooth removal"},
    {"root canal", "endodontic treatment", "rct"},
    {"scaling and root planing", "srp", "deep cleaning"},
    {"gingivitis", "gum inflammation"},
    {"bruxism", "teeth grinding", "tooth grinding"},
    {"malocclusion", "bad bite", "misaligned teeth"},
]

# Build fast lookup: term → set of synonyms (excluding self)
_SYNONYM_LOOKUP: dict[str, set[str]] = {}
for _group in _SYNONYM_GROUPS:
    for _term in _group:
        key = _term.lower()
        if key not in _SYNONYM_LOOKUP:
            _SYNONYM_LOOKUP[key] = set()
        _SYNONYM_LOOKUP[key].update(t.lower() for t in _group if t.lower() != key)


# ---------------------------------------------------------------------------
# Tokeniser (shared with hybrid.py)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:/[a-z0-9]+)?")


def _tokenize(text: str) -> list[str]:
    """Tokenise text to lowercase alphanumeric tokens.

    Parameters
    ----------
    text:
        Input string.

    Returns
    -------
    list[str]
        Lowercase tokens.
    """
    return _TOKEN_RE.findall(text.lower())


# ---------------------------------------------------------------------------
# Expanded query result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExpandedQuery:
    """Result of query expansion.

    Attributes
    ----------
    original:
        The original query string.
    expanded_terms:
        Additional terms added via synonym/abbreviation expansion.
    expansion_sources:
        Mapping of each expanded term to the reason it was added
        (e.g. ``"synonym of hypertension"`` or ``"abbreviation: htn"``).
    expanded_query:
        Combined query string with original + expanded terms.
    expansion_count:
        Number of terms added.
    """

    original: str
    expanded_terms: list[str] = field(default_factory=list)
    expansion_sources: dict[str, str] = field(default_factory=dict)
    expanded_query: str = ""
    expansion_count: int = 0


# ---------------------------------------------------------------------------
# Query expander
# ---------------------------------------------------------------------------


class MedicalQueryExpander:
    """Expands clinical search queries with medical synonyms and abbreviations.

    Parameters
    ----------
    max_expansions:
        Maximum number of expansion terms to add per query.
        Prevents query drift when a term has many synonyms.
    include_abbreviations:
        Whether to expand abbreviations to full forms (and vice versa).
    include_synonyms:
        Whether to expand with synonym groups.
    synonym_weight:
        Not used in the text expansion itself but surfaced in the
        ``ExpandedQuery`` for downstream weighting by the search engine.

    Examples
    --------
    >>> expander = MedicalQueryExpander(max_expansions=5)
    >>> result = expander.expand("patient with htn and dm")
    >>> "hypertension" in result.expanded_terms
    True
    >>> "diabetes mellitus" in result.expanded_terms
    True
    """

    def __init__(
        self,
        max_expansions: int = 8,
        include_abbreviations: bool = True,
        include_synonyms: bool = True,
        synonym_weight: float = 0.7,
    ) -> None:
        self.max_expansions = max_expansions
        self.include_abbreviations = include_abbreviations
        self.include_synonyms = include_synonyms
        self.synonym_weight = synonym_weight

    def expand(self, query: str) -> ExpandedQuery:
        """Expand a search query with medical synonyms and abbreviations.

        Parameters
        ----------
        query:
            Natural-language search query.

        Returns
        -------
        ExpandedQuery
            The original query plus any expansion terms and their sources.
        """
        if not query or not query.strip():
            return ExpandedQuery(original=query, expanded_query=query)

        tokens = _tokenize(query)
        original_lower = query.lower()

        expanded_terms: list[str] = []
        sources: dict[str, str] = {}

        # Check individual tokens and multi-word phrases
        checked_phrases: set[str] = set()

        # 1. Single-token abbreviation expansion
        if self.include_abbreviations:
            for token in tokens:
                if token in checked_phrases:
                    continue
                checked_phrases.add(token)

                # Abbreviation → full form
                full = _ABBREVIATION_TO_FULL.get(token)
                if full and full not in original_lower:
                    expanded_terms.append(full)
                    sources[full] = f"abbreviation expansion: {token}"

                # Full form → abbreviation (for multi-word lookups below)

        # 2. Multi-word phrase expansion (check bigrams and trigrams)
        if self.include_abbreviations or self.include_synonyms:
            for n in (2, 3, 4):
                for i in range(len(tokens) - n + 1):
                    phrase = " ".join(tokens[i : i + n])
                    if phrase in checked_phrases:
                        continue
                    checked_phrases.add(phrase)

                    # Check if this phrase has an abbreviation
                    if self.include_abbreviations:
                        abbr = _FULL_TO_ABBREVIATION.get(phrase)
                        if abbr and abbr not in original_lower:
                            expanded_terms.append(abbr)
                            sources[abbr] = f"abbreviation: {phrase}"

                    # Check synonym groups for the phrase
                    if self.include_synonyms:
                        syns = _SYNONYM_LOOKUP.get(phrase, set())
                        for syn in syns:
                            if syn not in original_lower and syn not in sources:
                                expanded_terms.append(syn)
                                sources[syn] = f"synonym of: {phrase}"

        # 3. Single-token synonym expansion
        if self.include_synonyms:
            for token in tokens:
                syns = _SYNONYM_LOOKUP.get(token, set())
                for syn in syns:
                    if syn not in original_lower and syn not in sources:
                        expanded_terms.append(syn)
                        sources[syn] = f"synonym of: {token}"

        # 4. Full-form → abbreviation for single tokens
        if self.include_abbreviations:
            for token in tokens:
                abbr = _FULL_TO_ABBREVIATION.get(token)
                if abbr and abbr not in original_lower and abbr not in sources:
                    expanded_terms.append(abbr)
                    sources[abbr] = f"abbreviation for: {token}"

        # Deduplicate while preserving order, cap at max_expansions
        seen: set[str] = set()
        unique_expansions: list[str] = []
        for term in expanded_terms:
            if term not in seen and len(unique_expansions) < self.max_expansions:
                seen.add(term)
                unique_expansions.append(term)

        # Build expanded query string
        expanded_query = query
        if unique_expansions:
            expanded_query = query + " " + " ".join(unique_expansions)

        logger.debug(
            "Query expansion: '%s' → +%d terms %s",
            query,
            len(unique_expansions),
            unique_expansions,
        )

        return ExpandedQuery(
            original=query,
            expanded_terms=unique_expansions,
            expansion_sources={k: v for k, v in sources.items() if k in seen},
            expanded_query=expanded_query,
            expansion_count=len(unique_expansions),
        )

    def get_abbreviation(self, term: str) -> str | None:
        """Look up the abbreviation for a full medical term.

        Parameters
        ----------
        term:
            Full medical term (case-insensitive).

        Returns
        -------
        str or None
            The abbreviation if found.
        """
        return _FULL_TO_ABBREVIATION.get(term.lower())

    def get_full_form(self, abbreviation: str) -> str | None:
        """Look up the full form for a medical abbreviation.

        Parameters
        ----------
        abbreviation:
            Medical abbreviation (case-insensitive).

        Returns
        -------
        str or None
            The full form if found.
        """
        return _ABBREVIATION_TO_FULL.get(abbreviation.lower())

    def get_synonyms(self, term: str) -> set[str]:
        """Get all synonyms for a medical term.

        Parameters
        ----------
        term:
            Medical term (case-insensitive).

        Returns
        -------
        set[str]
            Set of synonym strings (may be empty).
        """
        return _SYNONYM_LOOKUP.get(term.lower(), set())
