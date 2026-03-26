"""Text preprocessing utilities for clinical text.

Clinical text is fundamentally different from general-domain text:

* **Non-standard abbreviations** — "pt" (patient), "htn" (hypertension),
  "bid" (twice daily) are everywhere and rarely appear in dictionaries.
* **Section-structured layout** — Notes follow templates (H&P, SOAP,
  discharge summary) with recognisable headers.
* **Mixed formatting** — OCR artifacts, copy-paste from EHRs, bullet
  lists, ALL-CAPS headers, and Unicode variants from different systems.
* **Meaningful whitespace** — Line breaks often separate list items
  (e.g., medication lists), so aggressive whitespace normalisation
  can destroy structure.

This module handles the messy reality of real-world clinical documents
while preserving the structural cues that downstream NLP modules rely
on (section boundaries, sentence boundaries, list formatting).

Design decisions
----------------
* **Preserve structure** — We collapse runs of 3+ newlines to 2, but
  never strip all newlines.  Section detection depends on line breaks.
* **Abbreviation expansion is opt-in** — Many downstream modules
  (allergy extractor, medication extractor) have their own abbreviation
  handling tuned to their domain.  Global expansion can introduce
  ambiguity (e.g., "PE" = pulmonary embolism or physical exam).
* **Sentence segmentation protects decimals and titles** — Clinical
  text is full of "Dr. Smith" and "HbA1c 7.2" which naive splitters
  break on the period.
"""

import hashlib
import re
import string
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class TextSection:
    """Represents a detected section in clinical text."""

    name: str
    content: str
    start_char: int
    end_char: int
    confidence: float = 1.0


@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing."""

    normalize_whitespace: bool = True
    remove_extra_newlines: bool = True
    handle_abbreviations: bool = True
    detect_sections: bool = True
    preserve_case: bool = False
    min_section_length: int = 10
    max_document_length: int = 100000


# Common clinical section headers
CLINICAL_SECTIONS = {
    "chief_complaint": [
        "chief complaint",
        "cc",
        "reason for visit",
        "presenting complaint",
        "chief complaint / hpi",
    ],
    "hpi": [
        "history of present illness",
        "hpi",
        "history of present",
        "present illness",
        "history",
    ],
    "pmh": [
        "past medical history",
        "pmh",
        "medical history",
        "past history",
    ],
    "psh": [
        "past surgical history",
        "psh",
        "surgical history",
    ],
    "fh": [
        "family history",
        "fh",
        "fam hx",
    ],
    "sh": [
        "social history",
        "sh",
        "soc hx",
    ],
    "ros": [
        "review of systems",
        "ros",
        "review systems",
    ],
    "pe": [
        "physical exam",
        "pe",
        "physical examination",
        "exam",
    ],
    "assessment": [
        "assessment",
        "assessment and plan",
        "a/p",
        "impression",
        "clinical impression",
    ],
    "plan": [
        "plan",
        "treatment plan",
        "recommendations",
        "follow-up",
        "follow up",
    ],
    "labs": [
        "laboratory",
        "labs",
        "laboratory data",
        "lab results",
    ],
    "imaging": [
        "imaging",
        "radiology",
        "xr",
        "ct",
        "mri",
        "ultrasound",
    ],
    "medications": [
        "medications",
        "meds",
        "current medications",
        "home medications",
    ],
    "allergies": [
        "allergies",
        "allergy",
        "nkda",
        "no known drug allergies",
    ],
}

# Common medical abbreviations
MEDICAL_ABBREVIATIONS = {
    "pt": "patient",
    "pt.": "patient",
    "y/o": "year old",
    "yo": "year old",
    "m": "male",
    "f": "female",
    "h/o": "history of",
    "c/o": "complains of",
    "sob": "shortness of breath",
    "cp": "chest pain",
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "dm2": "diabetes mellitus type 2",
    "dm1": "diabetes mellitus type 1",
    "cad": "coronary artery disease",
    "chf": "congestive heart failure",
    "afib": "atrial fibrillation",
    "mi": "myocardial infarction",
    "cva": "cerebrovascular accident",
    "tia": "transient ischemic attack",
    "copd": "chronic obstructive pulmonary disease",
    "ckd": "chronic kidney disease",
    "esrd": "end stage renal disease",
    "djd": "degenerative joint disease",
    "oa": "osteoarthritis",
    "ra": "rheumatoid arthritis",
    "dvt": "deep vein thrombosis",
    "pe": "pulmonary embolism",
    "uti": "urinary tract infection",
    "n/v": "nausea and vomiting",
    "abd": "abdomen",
    "lmp": "last menstrual period",
    "fh": "family history",
    "sh": "social history",
    "pmh": "past medical history",
    "psh": "past surgical history",
    "ros": "review of systems",
    "wnl": "within normal limits",
    "nml": "normal",
    "abnl": "abnormal",
    "dx": "diagnosis",
    "tx": "treatment",
    "sx": "symptoms",
    "hx": "history",
    "rx": "prescription",
    "prn": "as needed",
    "bid": "twice daily",
    "tid": "three times daily",
    "qid": "four times daily",
    "qd": "once daily",
    "qhs": "at bedtime",
    "po": "by mouth",
    "iv": "intravenous",
    "im": "intramuscular",
    "sc": "subcutaneous",
    "stat": "immediately",
    "nkda": "no known drug allergies",
    "nka": "no known allergies",
}


class ClinicalTextPreprocessor:
    """Preprocessor for clinical text documents."""

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()
        self._section_patterns = self._build_section_patterns()

    def _build_section_patterns(self) -> dict[str, re.Pattern]:
        """Build regex patterns for section detection."""
        patterns = {}
        for section_type, headers in CLINICAL_SECTIONS.items():
            # Match headers at start of line, with optional colon
            header_pattern = "|".join(re.escape(h) for h in headers)
            pattern = rf"^\s*({header_pattern})\s*:?\s*$"
            patterns[section_type] = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        return patterns

    def preprocess(self, text: str) -> str:
        """Apply preprocessing pipeline to text."""
        if len(text) > self.config.max_document_length:
            text = text[: self.config.max_document_length]

        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)

        if self.config.remove_extra_newlines:
            text = self._remove_extra_newlines(text)

        return text.strip()

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        # Replace various unicode whitespace with standard space
        text = re.sub(r"[\u00a0\u2000-\u200b\u202f\u205f\u3000]", " ", text)
        # Normalize tabs to spaces
        text = re.sub(r"\t+", " ", text)
        # Collapse multiple spaces
        text = re.sub(r" +", " ", text)
        return text

    def _remove_extra_newlines(self, text: str) -> str:
        """Remove excessive newlines while preserving structure."""
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Collapse more than 2 consecutive newlines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text

    def expand_abbreviations(self, text: str) -> str:
        """Expand common medical abbreviations."""
        result = text
        for abbr, expansion in MEDICAL_ABBREVIATIONS.items():
            # Use word boundaries to avoid partial matches
            pattern = rf"\b{re.escape(abbr)}\b"
            result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)
        return result

    def detect_sections(self, text: str) -> list[TextSection]:
        """Detect clinical sections in the text."""
        if not self.config.detect_sections:
            return []

        sections: list[TextSection] = []
        lines = text.split("\n")

        # Find all section headers
        section_starts: list[tuple[int, str, int]] = []

        for line_idx, line in enumerate(lines):
            for section_type, pattern in self._section_patterns.items():
                if pattern.match(line):
                    char_pos = sum(len(l) + 1 for l in lines[:line_idx])
                    section_starts.append((char_pos, section_type, line_idx))
                    break

        # Create sections with content
        for i, (start_pos, section_type, _start_line) in enumerate(section_starts):
            # End position is start of next section or end of text
            end_pos = section_starts[i + 1][0] if i + 1 < len(section_starts) else len(text)

            content = text[start_pos:end_pos].strip()

            if len(content) >= self.config.min_section_length:
                sections.append(
                    TextSection(
                        name=section_type,
                        content=content,
                        start_char=start_pos,
                        end_char=end_pos,
                    )
                )

        return sections

    def segment_sentences(self, text: str) -> list[str]:
        """Segment clinical text into sentences."""
        # Clinical text has unusual sentence boundaries
        # Handle abbreviations with periods, decimal numbers, etc.

        # First, protect common patterns that shouldn't be split
        protected = text
        protected = re.sub(r"(\d+)\.(\d+)", r"\1__DECIMAL__\2", protected)  # Decimal numbers
        protected = re.sub(
            r"\b([A-Z])\.", r"\1__ABBR__", protected
        )  # Single letter abbreviations
        protected = re.sub(r"\b(dr|mr|mrs|ms|vs)\.", r"\1__ABBR__", protected, flags=re.IGNORECASE)

        # Split on sentence-ending punctuation followed by space and capital
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)

        # Restore protected patterns
        result = []
        for sentence in sentences:
            sentence = sentence.replace("__DECIMAL__", ".")
            sentence = sentence.replace("__ABBR__", ".")
            result.append(sentence.strip())

        return [s for s in result if s]

    def clean_text(self, text: str) -> str:
        """Clean text by removing artifacts and normalizing."""
        # Remove form feed characters
        text = text.replace("\f", " ")
        # Remove control characters except newlines and tabs
        text = "".join(char for char in text if char in string.printable or char in "\n\t")
        # Normalize unicode dashes
        text = re.sub(r"[\u2013\u2014]", "-", text)
        # Normalize quotes
        text = re.sub(r"[\u2018\u2019]", "'", text)
        text = re.sub(r"[\u201c\u201d]", '"', text)
        return text

    def extract_hash(self, text: str) -> str:
        """Generate SHA256 hash of text for deduplication."""
        normalized = self.preprocess(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get_text_stats(self, text: str) -> dict:
        """Get basic statistics about the text."""
        preprocessed = self.preprocess(text)
        words = preprocessed.split()
        sentences = self.segment_sentences(preprocessed)
        sections = self.detect_sections(preprocessed)

        return {
            "char_count": len(preprocessed),
            "word_count": len(words),
            "sentence_count": len(sentences),
            "section_count": len(sections),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0,
            "sections_found": [s.name for s in sections],
        }


def preprocess_clinical_text(
    text: str,
    config: PreprocessingConfig | None = None,
) -> str:
    """Convenience function to preprocess clinical text.

    Parameters
    ----------
    text:
        Raw clinical text to preprocess.  Empty or whitespace-only
        strings are returned as-is (empty string).
    config:
        Optional preprocessing configuration.

    Returns
    -------
    str
        Cleaned, normalised clinical text.

    Raises
    ------
    TypeError
        If *text* is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    if not text or not text.strip():
        return ""
    preprocessor = ClinicalTextPreprocessor(config)
    return preprocessor.preprocess(text)
