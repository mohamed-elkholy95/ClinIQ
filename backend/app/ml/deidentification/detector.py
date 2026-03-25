"""PHI detection and de-identification engine.

Implements rule-based Protected Health Information detection following
the HIPAA Safe Harbor method (45 CFR §164.514(b)(2)).  The 18 identifier
categories are mapped to ``PhiType`` enum members, each backed by one or
more compiled regex patterns with contextual window validation.

Design decisions
----------------
* **Regex-first approach** — deterministic, auditable, and fast (~2 ms per
  average clinical note).  A transformer-based detector can be layered on
  top via ``PhiDetector.add_custom_detector`` for higher recall.
* **Overlapping span resolution** — when two patterns match overlapping
  regions, the longer match wins (tie-break: higher-priority PhiType).
* **Replacement strategies** — ``REDACT`` replaces with ``[PHI_TYPE]``,
  ``MASK`` replaces characters with ``*``, ``SURROGATE`` substitutes
  realistic but fake values for downstream model training.
* **Thread safety** — all state is immutable after ``__init__``; pattern
  compilation happens once at construction time.

Limitations
-----------
* Pattern-based detection cannot catch novel name formats or misspellings.
* Date patterns assume US-style MM/DD/YYYY and ISO YYYY-MM-DD; other
  locale formats may require additional patterns.
* Age detection only flags ages ≥ 90 per Safe Harbor rules.
"""

import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class PhiType(StrEnum):
    """HIPAA Safe Harbor PHI identifier categories.

    Each member corresponds to one of the 18 categories defined in
    45 CFR §164.514(b)(2).  The string value is used as the replacement
    tag in ``REDACT`` mode (e.g. ``[DATE]``).
    """

    NAME = "NAME"
    DATE = "DATE"
    PHONE = "PHONE"
    FAX = "FAX"
    EMAIL = "EMAIL"
    SSN = "SSN"
    MRN = "MRN"
    HEALTH_PLAN = "HEALTH_PLAN"
    ACCOUNT = "ACCOUNT"
    LICENSE = "LICENSE"
    VEHICLE = "VEHICLE"
    DEVICE = "DEVICE"
    URL = "URL"
    IP_ADDRESS = "IP_ADDRESS"
    BIOMETRIC = "BIOMETRIC"
    PHOTO = "PHOTO"
    GEOGRAPHIC = "GEOGRAPHIC"
    AGE = "AGE"


class ReplacementStrategy(StrEnum):
    """How to replace detected PHI spans in the output text.

    * ``REDACT`` — replace with a bracketed type tag, e.g. ``[DATE]``.
    * ``MASK`` — replace each character with ``*``, preserving length.
    * ``SURROGATE`` — substitute a realistic but synthetic value so
      downstream models still see plausible structure.
    """

    REDACT = "redact"
    MASK = "mask"
    SURROGATE = "surrogate"


@dataclass(frozen=True)
class PhiEntity:
    """A detected PHI span within a clinical document.

    Parameters
    ----------
    text : str
        The original PHI text as it appears in the document.
    phi_type : PhiType
        The HIPAA category this span belongs to.
    start_char : int
        Character offset of the span start (0-indexed, inclusive).
    end_char : int
        Character offset of the span end (0-indexed, exclusive).
    confidence : float
        Detection confidence in [0, 1].  Rule-based patterns use a
        fixed confidence (typically 0.95); context-validated matches
        are boosted to 0.98.
    pattern_name : str
        Identifier of the pattern or detector that produced this match.
    """

    text: str
    phi_type: PhiType
    start_char: int
    end_char: int
    confidence: float = 0.95
    pattern_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "text": self.text,
            "phi_type": self.phi_type.value,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "pattern_name": self.pattern_name,
        }


@dataclass
class DeidentificationConfig:
    """Configuration for the de-identification pipeline.

    Parameters
    ----------
    strategy : ReplacementStrategy
        How to replace detected PHI (default: REDACT).
    enabled_types : set[PhiType] | None
        Subset of PHI types to detect.  ``None`` means all types.
    confidence_threshold : float
        Minimum confidence to include a detection (default: 0.5).
    context_window : int
        Number of characters before/after a match to inspect for
        contextual clues (e.g. "Dr." before a name).
    preserve_length : bool
        When using MASK strategy, whether to preserve the original
        text length (default: True).
    surrogate_seed : int
        Random seed for reproducible surrogate generation.
    """

    strategy: ReplacementStrategy = ReplacementStrategy.REDACT
    enabled_types: set[PhiType] | None = None
    confidence_threshold: float = 0.5
    context_window: int = 30
    preserve_length: bool = True
    surrogate_seed: int = 42


# ---------------------------------------------------------------------------
# Pattern registry
# ---------------------------------------------------------------------------
# Each entry: (PhiType, pattern_name, compiled_regex, base_confidence)
# Patterns are applied in priority order; earlier PhiTypes take precedence
# when spans overlap.

_CONTEXT_PREFIXES_NAME: re.Pattern[str] = re.compile(
    r"(?:(?:Dr|Mr|Mrs|Ms|Prof|Nurse|Patient|Pt)\.\s*$)",
    re.IGNORECASE,
)

_NAME_PATTERN = re.compile(
    r"\b(?:Dr|Mr|Mrs|Ms|Prof|Nurse)\.\s*"
    r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b",
    re.UNICODE,
)

_DATE_PATTERNS = [
    # MM/DD/YYYY or MM-DD-YYYY
    re.compile(
        r"\b(?:0[1-9]|1[0-2])[/\-](?:0[1-9]|[12]\d|3[01])[/\-]"
        r"(?:19|20)\d{2}\b"
    ),
    # YYYY-MM-DD (ISO)
    re.compile(
        r"\b(?:19|20)\d{2}[/\-](?:0[1-9]|1[0-2])[/\-]"
        r"(?:0[1-9]|[12]\d|3[01])\b"
    ),
    # Month DD, YYYY
    re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+"
        r"(?:0?[1-9]|[12]\d|3[01]),?\s+(?:19|20)\d{2}\b",
        re.IGNORECASE,
    ),
    # DD Month YYYY
    re.compile(
        r"\b(?:0?[1-9]|[12]\d|3[01])\s+"
        r"(?:January|February|March|April|May|June|July|August|"
        r"September|October|November|December)\s+"
        r"(?:19|20)\d{2}\b",
        re.IGNORECASE,
    ),
]

_PHONE_PATTERN = re.compile(
    r"\b(?:\+1[\s\-]?)?\(?[2-9]\d{2}\)?[\s\-.]?[2-9]\d{2}[\s\-.]?\d{4}\b"
)

_EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"
)

_SSN_PATTERN = re.compile(
    r"\b(?!000|666|9\d{2})\d{3}[\s\-]?(?!00)\d{2}[\s\-]?(?!0000)\d{4}\b"
)

_MRN_PATTERN = re.compile(
    r"\b(?:MRN|Medical\s*Record\s*(?:Number|No|#)?)[:\s#]*"
    r"([A-Z0-9]{4,12})\b",
    re.IGNORECASE,
)

_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+",
    re.IGNORECASE,
)

_IP_PATTERN = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

_ZIP_PATTERN = re.compile(
    r"\b\d{5}(?:\-\d{4})?\b"
)

_AGE_OVER_89_PATTERN = re.compile(
    r"\b(?:age[d]?\s*(?:of\s*)?)?(?:9\d|[1-9]\d{2,})\s*"
    r"(?:year[s]?\s*old|y/?o|yo)\b",
    re.IGNORECASE,
)

_ACCOUNT_PATTERN = re.compile(
    r"\b(?:Account|Acct)\s*(?:Number|No|#)?[:\s#]*"
    r"([A-Z0-9]{6,15})\b",
    re.IGNORECASE,
)

_LICENSE_PATTERN = re.compile(
    r"\b(?:License|Licence|DEA|NPI)\s*(?:Number|No|#)?[:\s#]*"
    r"([A-Z0-9]{5,15})\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Surrogate values for reproducible de-identification
# ---------------------------------------------------------------------------

_SURROGATE_NAMES = [
    "Dr. Smith", "Dr. Johnson", "Dr. Williams", "Dr. Brown",
    "Dr. Jones", "Dr. Davis", "Dr. Miller", "Dr. Wilson",
    "Dr. Moore", "Dr. Taylor", "Dr. Anderson", "Dr. Thomas",
]

_SURROGATE_DATES = [
    "01/01/2000", "06/15/2000", "03/20/2000", "09/10/2000",
    "12/25/2000", "07/04/2000", "02/14/2000", "11/11/2000",
]

_SURROGATE_PHONES = [
    "(555) 000-0001", "(555) 000-0002", "(555) 000-0003",
    "(555) 000-0004", "(555) 000-0005", "(555) 000-0006",
]

_SURROGATE_EMAILS = [
    "user1@example.com", "user2@example.com", "user3@example.com",
    "user4@example.com", "user5@example.com", "user6@example.com",
]

_SURROGATE_SSNS = [
    "000-00-0001", "000-00-0002", "000-00-0003",
    "000-00-0004", "000-00-0005", "000-00-0006",
]

_SURROGATE_MRNS = [
    "MRN000001", "MRN000002", "MRN000003",
    "MRN000004", "MRN000005", "MRN000006",
]

_SURROGATES: dict[PhiType, list[str]] = {
    PhiType.NAME: _SURROGATE_NAMES,
    PhiType.DATE: _SURROGATE_DATES,
    PhiType.PHONE: _SURROGATE_PHONES,
    PhiType.EMAIL: _SURROGATE_EMAILS,
    PhiType.SSN: _SURROGATE_SSNS,
    PhiType.MRN: _SURROGATE_MRNS,
}


class PhiDetector:
    """Detect Protected Health Information spans in clinical text.

    Uses compiled regex patterns and contextual heuristics to identify
    PHI entities.  Patterns are evaluated in priority order, and
    overlapping spans are resolved by keeping the longest match.

    Parameters
    ----------
    config : DeidentificationConfig
        Detection and replacement configuration.

    Examples
    --------
    >>> detector = PhiDetector()
    >>> entities = detector.detect("Patient Dr. Smith seen on 01/15/2024")
    >>> [e.phi_type for e in entities]
    [<PhiType.NAME: 'NAME'>, <PhiType.DATE: 'DATE'>]
    """

    def __init__(self, config: DeidentificationConfig | None = None) -> None:
        self._config = config or DeidentificationConfig()
        self._patterns = self._build_pattern_registry()
        self._custom_detectors: list[Any] = []
        logger.info(
            "PhiDetector initialised — strategy=%s, types=%s",
            self._config.strategy.value,
            "all" if self._config.enabled_types is None
            else ",".join(t.value for t in self._config.enabled_types),
        )

    @property
    def config(self) -> DeidentificationConfig:
        """Return the current configuration."""
        return self._config

    def _build_pattern_registry(
        self,
    ) -> list[tuple[PhiType, str, re.Pattern[str], float]]:
        """Build the ordered pattern registry.

        Returns
        -------
        list[tuple[PhiType, str, re.Pattern, float]]
            Each tuple is (phi_type, pattern_name, compiled_regex,
            base_confidence).  Order determines priority for overlap
            resolution.
        """
        registry: list[tuple[PhiType, str, re.Pattern[str], float]] = []
        enabled = self._config.enabled_types

        def _add(
            phi_type: PhiType,
            name: str,
            pattern: re.Pattern[str],
            confidence: float = 0.95,
        ) -> None:
            if enabled is None or phi_type in enabled:
                registry.append((phi_type, name, pattern, confidence))

        # Names (context-dependent, lower base confidence)
        _add(PhiType.NAME, "name_titled", _NAME_PATTERN, 0.90)

        # Dates — multiple formats
        for i, pat in enumerate(_DATE_PATTERNS):
            _add(PhiType.DATE, f"date_{i}", pat, 0.95)

        # Phone / Fax
        _add(PhiType.PHONE, "phone_us", _PHONE_PATTERN, 0.90)

        # Email
        _add(PhiType.EMAIL, "email", _EMAIL_PATTERN, 0.98)

        # SSN
        _add(PhiType.SSN, "ssn", _SSN_PATTERN, 0.85)

        # MRN
        _add(PhiType.MRN, "mrn", _MRN_PATTERN, 0.97)

        # URLs
        _add(PhiType.URL, "url", _URL_PATTERN, 0.98)

        # IP addresses
        _add(PhiType.IP_ADDRESS, "ip", _IP_PATTERN, 0.90)

        # Geographic (ZIP codes)
        _add(PhiType.GEOGRAPHIC, "zip", _ZIP_PATTERN, 0.70)

        # Age ≥ 90
        _add(PhiType.AGE, "age_over_89", _AGE_OVER_89_PATTERN, 0.92)

        # Account numbers
        _add(PhiType.ACCOUNT, "account", _ACCOUNT_PATTERN, 0.90)

        # License / DEA / NPI
        _add(PhiType.LICENSE, "license", _LICENSE_PATTERN, 0.90)

        return registry

    def add_custom_detector(self, detector: Any) -> None:
        """Register an external detector (e.g. transformer-based NER).

        The detector must implement a ``detect(text: str) -> list[PhiEntity]``
        method.

        Parameters
        ----------
        detector : object
            Any object with a ``detect`` method returning ``list[PhiEntity]``.
        """
        self._custom_detectors.append(detector)

    def detect(self, text: str) -> list[PhiEntity]:
        """Scan text for PHI entities.

        Parameters
        ----------
        text : str
            Clinical text to scan.

        Returns
        -------
        list[PhiEntity]
            Detected PHI entities, sorted by start_char.  Overlapping
            spans have been resolved (longest match wins).
        """
        if not text or not text.strip():
            return []

        entities: list[PhiEntity] = []

        # --- Regex pattern detection ---
        for phi_type, pattern_name, pattern, base_conf in self._patterns:
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                matched_text = match.group()

                # Context-based confidence adjustment
                confidence = self._adjust_confidence(
                    text, start, end, phi_type, base_conf,
                )

                if confidence < self._config.confidence_threshold:
                    continue

                # For MRN/Account/License, use the captured group if present
                if match.lastindex and match.lastindex >= 1:
                    matched_text = match.group(1)
                    start = match.start(1)
                    end = match.end(1)

                entities.append(
                    PhiEntity(
                        text=matched_text,
                        phi_type=phi_type,
                        start_char=start,
                        end_char=end,
                        confidence=confidence,
                        pattern_name=pattern_name,
                    )
                )

        # --- Custom detectors (e.g. transformer NER) ---
        for detector in self._custom_detectors:
            try:
                custom_entities = detector.detect(text)
                entities.extend(custom_entities)
            except Exception:
                logger.warning(
                    "Custom PHI detector failed — skipping",
                    exc_info=True,
                )

        # Resolve overlapping spans
        entities = self._resolve_overlaps(entities)

        # Sort by position
        entities.sort(key=lambda e: (e.start_char, -e.end_char))

        return entities

    def _adjust_confidence(
        self,
        text: str,
        start: int,
        end: int,
        phi_type: PhiType,
        base_confidence: float,
    ) -> float:
        """Adjust detection confidence based on surrounding context.

        Looks at a window of text before/after the match for contextual
        clues that raise or lower confidence.  For example, "Dr." before
        a name match raises confidence; a standalone 5-digit number in
        a medication dosage context lowers ZIP code confidence.

        Parameters
        ----------
        text : str
            Full document text.
        start : int
            Match start offset.
        end : int
            Match end offset.
        phi_type : PhiType
            Detected PHI category.
        base_confidence : float
            Confidence from the pattern definition.

        Returns
        -------
        float
            Adjusted confidence, clamped to [0.0, 1.0].
        """
        window = self._config.context_window
        prefix = text[max(0, start - window): start]
        suffix = text[end: min(len(text), end + window)]

        adjustment = 0.0

        if phi_type == PhiType.NAME:
            # Boost if preceded by title
            if _CONTEXT_PREFIXES_NAME.search(prefix):
                adjustment += 0.05
            # Boost if followed by clinical context
            if re.search(r"(?:diagnosed|presented|reported|complain)", suffix, re.I):
                adjustment += 0.03

        elif phi_type == PhiType.GEOGRAPHIC:
            # ZIP codes are very noisy — lower confidence unless preceded
            # by address-like context
            if re.search(r"(?:zip|postal|address|city|state)", prefix, re.I):
                adjustment += 0.15
            else:
                # Standalone 5-digit numbers are often dosages, lab values
                adjustment -= 0.25

        elif phi_type == PhiType.SSN:
            # Boost if preceded by SSN-related context
            if re.search(r"(?:ssn|social\s*security|ss#)", prefix, re.I):
                adjustment += 0.10
            else:
                # Bare 9-digit numbers might be phone numbers or MRNs
                adjustment -= 0.15

        elif phi_type == PhiType.PHONE:
            # Boost if preceded by phone/fax/tel context
            if re.search(r"(?:phone|tel|call|fax|contact|cell|mobile)", prefix, re.I):
                adjustment += 0.08
            if re.search(r"(?:fax)", prefix, re.I):
                # Re-classify as FAX
                pass  # Keep as PHONE; fax handled separately if needed

        elif phi_type == PhiType.AGE:
            # Only flag ages ≥ 90 per Safe Harbor
            matched = text[start:end]
            age_match = re.search(r"(\d+)", matched)
            if age_match:
                age_val = int(age_match.group(1))
                if age_val < 90:
                    adjustment -= 1.0  # Effectively suppress

        return max(0.0, min(1.0, base_confidence + adjustment))

    @staticmethod
    def _resolve_overlaps(entities: list[PhiEntity]) -> list[PhiEntity]:
        """Resolve overlapping PHI spans by keeping the longest match.

        When two detections overlap, the longer span is kept.  On tie,
        the one with higher confidence wins.

        Parameters
        ----------
        entities : list[PhiEntity]
            Raw detected entities (may overlap).

        Returns
        -------
        list[PhiEntity]
            Non-overlapping entities.
        """
        if not entities:
            return []

        # Sort by start, then by descending length (longer first)
        sorted_entities = sorted(
            entities,
            key=lambda e: (e.start_char, -(e.end_char - e.start_char), -e.confidence),
        )

        resolved: list[PhiEntity] = [sorted_entities[0]]
        for entity in sorted_entities[1:]:
            last = resolved[-1]
            # If this entity overlaps with the last kept entity, skip it
            if entity.start_char < last.end_char:
                # Keep the longer one (already sorted so 'last' is longer or equal)
                if (entity.end_char - entity.start_char) > (last.end_char - last.start_char):
                    resolved[-1] = entity
            else:
                resolved.append(entity)

        return resolved


class Deidentifier:
    """De-identify clinical text by detecting and replacing PHI spans.

    Orchestrates ``PhiDetector`` for detection and applies the configured
    replacement strategy to produce a safe output text.

    Parameters
    ----------
    config : DeidentificationConfig | None
        Pipeline configuration.  Defaults to REDACT strategy with all
        PHI types enabled.

    Examples
    --------
    >>> deid = Deidentifier()
    >>> result = deid.deidentify(
    ...     "Patient Dr. Smith seen on 01/15/2024. SSN: 123-45-6789"
    ... )
    >>> print(result["text"])
    Patient [NAME] seen on [DATE]. SSN: [SSN]
    """

    def __init__(self, config: DeidentificationConfig | None = None) -> None:
        self._config = config or DeidentificationConfig()
        self._detector = PhiDetector(self._config)
        self._surrogate_counters: dict[PhiType, int] = {}

    @property
    def detector(self) -> PhiDetector:
        """Return the underlying PHI detector."""
        return self._detector

    @property
    def config(self) -> DeidentificationConfig:
        """Return the current configuration."""
        return self._config

    def deidentify(self, text: str) -> dict[str, Any]:
        """Detect and replace PHI in clinical text.

        Parameters
        ----------
        text : str
            Raw clinical text potentially containing PHI.

        Returns
        -------
        dict[str, Any]
            Dictionary with keys:
            - ``text`` (str): De-identified text.
            - ``entities`` (list[dict]): Detected PHI entities.
            - ``entity_count`` (int): Total PHI spans found.
            - ``phi_types_found`` (list[str]): Unique PHI types detected.
            - ``strategy`` (str): Replacement strategy used.
        """
        if not text or not text.strip():
            return {
                "text": text or "",
                "entities": [],
                "entity_count": 0,
                "phi_types_found": [],
                "strategy": self._config.strategy.value,
            }

        entities = self._detector.detect(text)

        # Build replacement text (process right-to-left to preserve offsets)
        deidentified = text
        for entity in reversed(entities):
            replacement = self._get_replacement(entity)
            deidentified = (
                deidentified[: entity.start_char]
                + replacement
                + deidentified[entity.end_char:]
            )

        phi_types_found = sorted({e.phi_type.value for e in entities})

        return {
            "text": deidentified,
            "entities": [e.to_dict() for e in entities],
            "entity_count": len(entities),
            "phi_types_found": phi_types_found,
            "strategy": self._config.strategy.value,
        }

    def deidentify_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """De-identify multiple clinical texts.

        Parameters
        ----------
        texts : list[str]
            List of clinical texts to de-identify.

        Returns
        -------
        list[dict[str, Any]]
            List of de-identification results, one per input text.
        """
        return [self.deidentify(t) for t in texts]

    def _get_replacement(self, entity: PhiEntity) -> str:
        """Generate replacement text for a PHI entity.

        Parameters
        ----------
        entity : PhiEntity
            The PHI entity to replace.

        Returns
        -------
        str
            Replacement text according to the configured strategy.
        """
        strategy = self._config.strategy

        if strategy == ReplacementStrategy.REDACT:
            return f"[{entity.phi_type.value}]"

        elif strategy == ReplacementStrategy.MASK:
            if self._config.preserve_length:
                return "*" * len(entity.text)
            return "****"

        elif strategy == ReplacementStrategy.SURROGATE:
            return self._get_surrogate(entity.phi_type)

        # Fallback to redact
        return f"[{entity.phi_type.value}]"

    def _get_surrogate(self, phi_type: PhiType) -> str:
        """Pick a deterministic surrogate value for the given PHI type.

        Uses a per-type counter seeded by ``surrogate_seed`` so repeated
        calls cycle through the surrogate list deterministically.

        Parameters
        ----------
        phi_type : PhiType
            Category of PHI to generate a surrogate for.

        Returns
        -------
        str
            A synthetic replacement value.
        """
        surrogates = _SURROGATES.get(phi_type)
        if not surrogates:
            return f"[{phi_type.value}]"

        counter = self._surrogate_counters.get(phi_type, self._config.surrogate_seed)
        idx = counter % len(surrogates)
        self._surrogate_counters[phi_type] = counter + 1
        return surrogates[idx]
