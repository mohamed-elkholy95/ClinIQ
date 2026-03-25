"""Clinical vital signs extraction from unstructured clinical text.

This module extracts structured vital sign measurements from free-text
clinical notes, including:

- **Blood Pressure** (systolic/diastolic, MAP calculation)
- **Heart Rate / Pulse** (beats per minute)
- **Temperature** (Fahrenheit/Celsius with auto-conversion)
- **Respiratory Rate** (breaths per minute)
- **Oxygen Saturation** (SpO2 percentage)
- **Weight** (kg/lbs with conversion)
- **Height** (cm/inches/feet-inches with conversion)
- **BMI** (extracted or calculated from weight+height)
- **Pain Scale** (0–10 numeric rating)

Architecture
------------
``ClinicalVitalSignsExtractor`` uses compiled regex patterns with:

1. **Multi-format matching** — handles abbreviations (BP, HR, T, RR, SpO2),
   full names ("blood pressure", "heart rate"), and common clinical shorthand
   ("afebrile", "tachycardic").
2. **Unit normalisation** — converts all values to standard units (mmHg, bpm,
   °F, breaths/min, %, kg, cm, kg/m²).
3. **Range validation** — physiologically impossible values are rejected
   (e.g., HR > 350, temp > 115°F).
4. **Clinical interpretation** — each reading is classified as normal,
   low, high, or critical based on standard adult reference ranges.
5. **Section awareness** — boosts confidence when found inside "Vital Signs"
   or "Physical Exam" sections.
6. **Trend detection** — identifies comparative language ("improved",
   "worsening", "stable") adjacent to vital signs.

References
----------
- AHA Blood Pressure Categories (2017 Guidelines)
- WHO Reference Ranges for Adult Vital Signs
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class VitalSignType(StrEnum):
    """Recognised vital sign categories."""

    BLOOD_PRESSURE = "blood_pressure"
    HEART_RATE = "heart_rate"
    TEMPERATURE = "temperature"
    RESPIRATORY_RATE = "respiratory_rate"
    OXYGEN_SATURATION = "oxygen_saturation"
    WEIGHT = "weight"
    HEIGHT = "height"
    BMI = "bmi"
    PAIN_SCALE = "pain_scale"


class ClinicalInterpretation(StrEnum):
    """Clinical interpretation of a vital sign value."""

    NORMAL = "normal"
    LOW = "low"
    HIGH = "high"
    CRITICAL_LOW = "critical_low"
    CRITICAL_HIGH = "critical_high"


class VitalTrend(StrEnum):
    """Directional trend relative to previous reading."""

    IMPROVING = "improving"
    WORSENING = "worsening"
    STABLE = "stable"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class VitalSignReading:
    """A single extracted vital sign measurement.

    Parameters
    ----------
    vital_type : VitalSignType
        Category of vital sign.
    value : float
        Primary numeric value in standard units.
    unit : str
        Standard unit string (e.g., "mmHg", "bpm", "°F").
    raw_text : str
        Original text span that was matched.
    start : int
        Character offset start in source text.
    end : int
        Character offset end in source text.
    confidence : float
        Extraction confidence [0.0, 1.0].
    interpretation : ClinicalInterpretation
        Clinical classification of the value.
    secondary_value : float | None
        For blood pressure: diastolic value.
    trend : VitalTrend
        Detected trend modifier if any.
    metadata : dict[str, Any]
        Additional context (MAP, original unit, conversion info).
    """

    vital_type: VitalSignType
    value: float
    unit: str
    raw_text: str
    start: int
    end: int
    confidence: float
    interpretation: ClinicalInterpretation
    secondary_value: float | None = None
    trend: VitalTrend = VitalTrend.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        d = asdict(self)
        d["vital_type"] = self.vital_type.value
        d["interpretation"] = self.interpretation.value
        d["trend"] = self.trend.value
        return d


@dataclass
class VitalSignsResult:
    """Container for all vital signs extracted from a document.

    Parameters
    ----------
    readings : list[VitalSignReading]
        All extracted readings, sorted by document position.
    text_hash : str
        SHA-256 of the input text for deduplication.
    extraction_time_ms : float
        Wall-clock extraction time in milliseconds.
    summary : dict[str, Any]
        Aggregate stats: counts per type, any critical findings.
    """

    readings: list[VitalSignReading] = field(default_factory=list)
    text_hash: str = ""
    extraction_time_ms: float = 0.0
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to plain dictionary."""
        return {
            "readings": [r.to_dict() for r in self.readings],
            "text_hash": self.text_hash,
            "extraction_time_ms": round(self.extraction_time_ms, 2),
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Reference ranges (adult)
# ---------------------------------------------------------------------------

# Format: (critical_low, low, high, critical_high)
# Values at or below critical_low / at or above critical_high → critical
# Values between critical_low–low or high–critical_high → low/high

_ADULT_RANGES: dict[VitalSignType, tuple[float, float, float, float]] = {
    # Systolic BP
    VitalSignType.BLOOD_PRESSURE: (70.0, 90.0, 140.0, 180.0),
    # Diastolic reference (stored in metadata for interpretation)
    VitalSignType.HEART_RATE: (30.0, 60.0, 100.0, 150.0),
    # Temperature in °F
    VitalSignType.TEMPERATURE: (93.0, 97.0, 99.5, 104.0),
    VitalSignType.RESPIRATORY_RATE: (6.0, 12.0, 20.0, 30.0),
    VitalSignType.OXYGEN_SATURATION: (85.0, 95.0, 100.0, 101.0),
    # BMI
    VitalSignType.BMI: (10.0, 18.5, 25.0, 40.0),
    # Pain scale
    VitalSignType.PAIN_SCALE: (-1.0, 0.0, 3.0, 7.0),
}

# Diastolic ranges
_DIASTOLIC_RANGES = (40.0, 60.0, 90.0, 120.0)


def _interpret(vital_type: VitalSignType, value: float) -> ClinicalInterpretation:
    """Classify a vital sign value against adult reference ranges.

    Parameters
    ----------
    vital_type : VitalSignType
        The type of vital sign.
    value : float
        The numeric value in standard units.

    Returns
    -------
    ClinicalInterpretation
        One of NORMAL, LOW, HIGH, CRITICAL_LOW, CRITICAL_HIGH.
    """
    ranges = _ADULT_RANGES.get(vital_type)
    if ranges is None:
        return ClinicalInterpretation.NORMAL

    crit_low, low, high, crit_high = ranges
    if value <= crit_low:
        return ClinicalInterpretation.CRITICAL_LOW
    if value < low:
        return ClinicalInterpretation.LOW
    if value >= crit_high:
        return ClinicalInterpretation.CRITICAL_HIGH
    if value > high:
        return ClinicalInterpretation.HIGH
    return ClinicalInterpretation.NORMAL


def _interpret_diastolic(value: float) -> ClinicalInterpretation:
    """Classify diastolic blood pressure."""
    crit_low, low, high, crit_high = _DIASTOLIC_RANGES
    if value <= crit_low:
        return ClinicalInterpretation.CRITICAL_LOW
    if value < low:
        return ClinicalInterpretation.LOW
    if value >= crit_high:
        return ClinicalInterpretation.CRITICAL_HIGH
    if value > high:
        return ClinicalInterpretation.HIGH
    return ClinicalInterpretation.NORMAL


# ---------------------------------------------------------------------------
# Physiological validation ranges (reject values outside these)
# ---------------------------------------------------------------------------

_VALID_RANGES: dict[VitalSignType, tuple[float, float]] = {
    VitalSignType.BLOOD_PRESSURE: (20.0, 350.0),  # systolic
    VitalSignType.HEART_RATE: (10.0, 350.0),
    VitalSignType.TEMPERATURE: (80.0, 115.0),  # °F
    VitalSignType.RESPIRATORY_RATE: (2.0, 80.0),
    VitalSignType.OXYGEN_SATURATION: (30.0, 100.0),
    VitalSignType.WEIGHT: (0.5, 700.0),  # kg
    VitalSignType.HEIGHT: (20.0, 280.0),  # cm
    VitalSignType.BMI: (5.0, 100.0),
    VitalSignType.PAIN_SCALE: (0.0, 10.0),
}


def _is_valid(vital_type: VitalSignType, value: float) -> bool:
    """Check if a value is physiologically plausible.

    Parameters
    ----------
    vital_type : VitalSignType
        The vital sign type.
    value : float
        The extracted value.

    Returns
    -------
    bool
        True if the value falls within plausible physiological bounds.
    """
    bounds = _VALID_RANGES.get(vital_type)
    if bounds is None:
        return True
    return bounds[0] <= value <= bounds[1]


# ---------------------------------------------------------------------------
# Trend detection patterns
# ---------------------------------------------------------------------------

_TREND_PATTERNS: list[tuple[re.Pattern[str], VitalTrend]] = [
    (re.compile(r"\b(?:improv(?:ed|ing)|better|decreas(?:ed|ing)|normaliz(?:ed|ing)|resolv(?:ed|ing))\b", re.IGNORECASE), VitalTrend.IMPROVING),
    (re.compile(r"\b(?:worsen(?:ed|ing)|deteriorat(?:ed|ing)|increas(?:ed|ing)|elevat(?:ed|ing)|rising|rose)\b", re.IGNORECASE), VitalTrend.WORSENING),
    (re.compile(r"\b(?:stable|unchanged|steady|maintained|consistent)\b", re.IGNORECASE), VitalTrend.STABLE),
]


def _detect_trend(text: str, start: int, end: int) -> VitalTrend:
    """Look for trend language within 60 characters of the vital sign.

    Parameters
    ----------
    text : str
        Full document text.
    start : int
        Vital sign span start.
    end : int
        Vital sign span end.

    Returns
    -------
    VitalTrend
        Detected trend or UNKNOWN.
    """
    window_start = max(0, start - 60)
    window_end = min(len(text), end + 60)
    window = text[window_start:window_end]
    for pattern, trend in _TREND_PATTERNS:
        if pattern.search(window):
            return trend
    return VitalTrend.UNKNOWN


# ---------------------------------------------------------------------------
# Section awareness
# ---------------------------------------------------------------------------

_VITALS_SECTION_HEADERS = re.compile(
    r"(?:^|\n)\s*(?:vital\s*signs?|vitals|v/?s|physical\s*exam(?:ination)?|"
    r"objective|review\s*of\s*systems|assessment\s*(?:and|&)\s*plan)\s*[:|\n]",
    re.IGNORECASE,
)

_SECTION_TERMINATOR = re.compile(
    r"(?:^|\n)\s*(?:[A-Z][A-Z ]{3,}|[A-Z][a-z]+(?: [A-Z][a-z]+)+)\s*:",
)

_SECTION_CONFIDENCE_BOOST = 0.05


def _in_vitals_section(text: str, position: int) -> bool:
    """Determine if *position* falls inside a Vital Signs section.

    Parameters
    ----------
    text : str
        Full document text.
    position : int
        Character offset to test.

    Returns
    -------
    bool
        True if inside a vitals-related section.
    """
    # Find the last vitals header before position
    last_header_end = -1
    for m in _VITALS_SECTION_HEADERS.finditer(text):
        if m.end() <= position:
            last_header_end = m.end()

    if last_header_end < 0:
        return False

    # Check if a new section header appears between the vitals header and position
    for m in _SECTION_TERMINATOR.finditer(text, last_header_end):
        if m.start() < position:
            # A new section started before our position — we left vitals
            return False
        break  # next section is after our position
    return True


# ---------------------------------------------------------------------------
# Compiled extraction patterns
# ---------------------------------------------------------------------------


def _bp_patterns() -> list[tuple[re.Pattern[str], str]]:
    """Blood pressure patterns returning (systolic, diastolic) groups.

    Returns
    -------
    list[tuple[re.Pattern[str], str]]
        (pattern, label) pairs.
    """
    return [
        # BP 120/80, BP: 120/80 mmHg, Blood Pressure 120/80
        (re.compile(
            r"(?:(?:blood\s*pressure|b\.?p\.?)\s*[:=]?\s*)"
            r"(\d{2,3})\s*/\s*(\d{2,3})"
            r"(?:\s*(?:mmHg|mm\s*Hg))?",
            re.IGNORECASE,
        ), "labeled"),
        # Standalone systolic/diastolic: 120/80 mmHg (must have unit)
        (re.compile(
            r"(?<!\d[./])"
            r"(\d{2,3})\s*/\s*(\d{2,3})"
            r"\s*(?:mmHg|mm\s*Hg)",
            re.IGNORECASE,
        ), "unit_required"),
    ]


def _hr_patterns() -> list[re.Pattern[str]]:
    """Heart rate / pulse patterns.

    Returns
    -------
    list[re.Pattern[str]]
        Compiled patterns with value in group(1).
    """
    return [
        # HR 80, HR: 80 bpm, Heart Rate: 80, Pulse: 80
        re.compile(
            r"(?:(?:heart\s*rate|h\.?r\.?|pulse)\s*[:=]?\s*)"
            r"(\d{2,3})"
            r"(?:\s*(?:bpm|beats?\s*/?\s*min(?:ute)?|/min))?",
            re.IGNORECASE,
        ),
        # Standalone with unit: 80 bpm
        re.compile(
            r"(?<![/\d])"
            r"(\d{2,3})\s*(?:bpm|beats\s*/?\s*min(?:ute)?)",
            re.IGNORECASE,
        ),
    ]


def _temp_patterns() -> list[tuple[re.Pattern[str], str]]:
    """Temperature patterns.

    Returns
    -------
    list[tuple[re.Pattern[str], str]]
        (pattern, unit_hint) — unit_hint is "F", "C", or "auto".
    """
    return [
        # Temp 98.6 F, T: 98.6°F, Temperature: 37.0 C
        (re.compile(
            r"(?:(?:temp(?:erature)?|t(?:emp)?)\s*[:=]?\s*)"
            r"(\d{2,3}(?:\.\d{1,2})?)\s*°?\s*([FfCc])\b",
            re.IGNORECASE,
        ), "explicit"),
        # Labeled without unit — assume °F if > 50 else °C
        (re.compile(
            r"(?:(?:temp(?:erature)?|t(?:emp)?)\s*[:=]\s*)"
            r"(\d{2,3}(?:\.\d{1,2})?)"
            r"(?!\s*°?\s*[FfCc])",
            re.IGNORECASE,
        ), "auto"),
    ]


def _rr_patterns() -> list[re.Pattern[str]]:
    """Respiratory rate patterns.

    Returns
    -------
    list[re.Pattern[str]]
        Compiled patterns.
    """
    return [
        # RR 18, Resp Rate: 18, Respiratory Rate: 18
        re.compile(
            r"(?:(?:resp(?:iratory)?\s*rate|r\.?r\.?)\s*[:=]?\s*)"
            r"(\d{1,2})"
            r"(?:\s*(?:breaths?\s*/?\s*min(?:ute)?|/min|bpm))?",
            re.IGNORECASE,
        ),
    ]


def _spo2_patterns() -> list[re.Pattern[str]]:
    """Oxygen saturation patterns.

    Returns
    -------
    list[re.Pattern[str]]
        Compiled patterns.
    """
    return [
        # SpO2 98%, O2 Sat 98%, SaO2: 98%
        re.compile(
            r"(?:(?:sp\s*o\s*2|o2\s*sat(?:uration)?|sa\s*o\s*2|oxygen\s*sat(?:uration)?|sat)\s*[:=]?\s*)"
            r"(\d{2,3})\s*%?",
            re.IGNORECASE,
        ),
        # Standalone: 98% on RA / 98% on 2L NC
        re.compile(
            r"(\d{2,3})\s*%\s*(?:on\s+(?:RA|room\s*air|\d+\s*L))",
            re.IGNORECASE,
        ),
    ]


def _weight_patterns() -> list[tuple[re.Pattern[str], str]]:
    """Weight patterns.

    Returns
    -------
    list[tuple[re.Pattern[str], str]]
        (pattern, unit_hint) pairs.
    """
    return [
        # Weight: 70 kg, Wt: 154 lbs
        (re.compile(
            r"(?:(?:weight|wt|body\s*weight)\s*[:=]?\s*)"
            r"(\d{1,3}(?:\.\d{1,2})?)\s*(kg|kgs|lbs?|pounds?|kilograms?)",
            re.IGNORECASE,
        ), "explicit"),
        # Standalone with unit: 70 kg
        (re.compile(
            r"(?<![/\d])"
            r"(\d{1,3}(?:\.\d{1,2})?)\s*(kg|kgs)\b",
            re.IGNORECASE,
        ), "kg"),
    ]


def _height_patterns() -> list[tuple[re.Pattern[str], str]]:
    """Height patterns.

    Returns
    -------
    list[tuple[re.Pattern[str], str]]
        (pattern, format_hint) pairs.
    """
    return [
        # Height: 5'10", Ht: 5 ft 10 in
        (re.compile(
            r"(?:(?:height|ht|stature)\s*[:=]?\s*)"
            r"(\d{1})\s*['\u2032ft]\s*(\d{1,2})\s*(?:[\"″\u2033]|in(?:ches?)?)?",
            re.IGNORECASE,
        ), "feet_inches"),
        # Height: 170 cm
        (re.compile(
            r"(?:(?:height|ht|stature)\s*[:=]?\s*)"
            r"(\d{2,3}(?:\.\d{1,2})?)\s*(cm|centimeters?|in(?:ches?)?|inches)",
            re.IGNORECASE,
        ), "metric_or_inches"),
    ]


def _bmi_patterns() -> list[re.Pattern[str]]:
    """BMI patterns.

    Returns
    -------
    list[re.Pattern[str]]
        Compiled patterns.
    """
    return [
        # BMI 24.5, BMI: 24.5 kg/m2
        re.compile(
            r"(?:(?:bmi|body\s*mass\s*index)\s*[:=]?\s*)"
            r"(\d{1,2}(?:\.\d{1,2})?)"
            r"(?:\s*(?:kg\s*/?\s*m\s*[²2]?))?",
            re.IGNORECASE,
        ),
    ]


def _pain_patterns() -> list[re.Pattern[str]]:
    """Pain scale patterns.

    Returns
    -------
    list[re.Pattern[str]]
        Compiled patterns.
    """
    return [
        # Pain: 5/10, Pain Scale: 5/10, Pain score 7
        re.compile(
            r"(?:(?:pain\s*(?:score|scale|level|rating)?)\s*[:=]?\s*)"
            r"(\d{1,2})\s*(?:/\s*10)?",
            re.IGNORECASE,
        ),
    ]


# ---------------------------------------------------------------------------
# Qualitative vital sign patterns
# ---------------------------------------------------------------------------

_QUALITATIVE_VITALS: list[tuple[re.Pattern[str], VitalSignType, float, ClinicalInterpretation, str]] = [
    # Afebrile → normal temperature
    (re.compile(r"\b(?:afebrile)\b", re.IGNORECASE),
     VitalSignType.TEMPERATURE, 98.6, ClinicalInterpretation.NORMAL, "afebrile (implied 98.6°F)"),
    # Febrile → elevated temperature
    (re.compile(r"\b(?:febrile|fever(?:ish)?)\b", re.IGNORECASE),
     VitalSignType.TEMPERATURE, 101.0, ClinicalInterpretation.HIGH, "febrile (implied ≥101°F)"),
    # Tachycardic → elevated HR
    (re.compile(r"\b(?:tachycardi[ac])\b", re.IGNORECASE),
     VitalSignType.HEART_RATE, 110.0, ClinicalInterpretation.HIGH, "tachycardic (implied >100 bpm)"),
    # Bradycardic → low HR
    (re.compile(r"\b(?:bradycardi[ac])\b", re.IGNORECASE),
     VitalSignType.HEART_RATE, 50.0, ClinicalInterpretation.LOW, "bradycardic (implied <60 bpm)"),
    # Tachypneic → elevated RR
    (re.compile(r"\b(?:tachypne[ia]c?)\b", re.IGNORECASE),
     VitalSignType.RESPIRATORY_RATE, 24.0, ClinicalInterpretation.HIGH, "tachypneic (implied >20/min)"),
    # Hypotensive
    (re.compile(r"\b(?:hypotensive?)\b", re.IGNORECASE),
     VitalSignType.BLOOD_PRESSURE, 85.0, ClinicalInterpretation.LOW, "hypotensive (implied SBP <90)"),
    # Hypertensive
    (re.compile(r"\b(?:hypertensive?)\b", re.IGNORECASE),
     VitalSignType.BLOOD_PRESSURE, 160.0, ClinicalInterpretation.HIGH, "hypertensive (implied SBP >140)"),
    # Hypoxic
    (re.compile(r"\b(?:hypoxic|hypoxemi[ac])\b", re.IGNORECASE),
     VitalSignType.OXYGEN_SATURATION, 88.0, ClinicalInterpretation.LOW, "hypoxic (implied SpO2 <90%)"),
]


# ---------------------------------------------------------------------------
# Unit conversions
# ---------------------------------------------------------------------------


def _celsius_to_fahrenheit(c: float) -> float:
    """Convert Celsius to Fahrenheit.

    Parameters
    ----------
    c : float
        Temperature in degrees Celsius.

    Returns
    -------
    float
        Temperature in degrees Fahrenheit.
    """
    return round(c * 9.0 / 5.0 + 32.0, 1)


def _lbs_to_kg(lbs: float) -> float:
    """Convert pounds to kilograms.

    Parameters
    ----------
    lbs : float
        Weight in pounds.

    Returns
    -------
    float
        Weight in kilograms.
    """
    return round(lbs / 2.2046, 1)


def _inches_to_cm(inches: float) -> float:
    """Convert inches to centimetres.

    Parameters
    ----------
    inches : float
        Length in inches.

    Returns
    -------
    float
        Length in centimetres.
    """
    return round(inches * 2.54, 1)


def _feet_inches_to_cm(feet: int, inches: int) -> float:
    """Convert feet + inches to centimetres.

    Parameters
    ----------
    feet : int
        Number of feet.
    inches : int
        Remaining inches.

    Returns
    -------
    float
        Height in centimetres.
    """
    total_inches = feet * 12 + inches
    return _inches_to_cm(total_inches)


def _calculate_bmi(weight_kg: float, height_cm: float) -> float | None:
    """Calculate BMI from weight (kg) and height (cm).

    Parameters
    ----------
    weight_kg : float
        Weight in kilograms.
    height_cm : float
        Height in centimetres.

    Returns
    -------
    float | None
        BMI value, or None if height is zero.
    """
    if height_cm <= 0:
        return None
    height_m = height_cm / 100.0
    return round(weight_kg / (height_m * height_m), 1)


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------


class ClinicalVitalSignsExtractor:
    """Extract structured vital signs from clinical free text.

    The extractor is stateless and thread-safe.  All patterns are compiled
    at class-instantiation time and reused across calls.

    Parameters
    ----------
    min_confidence : float
        Minimum confidence threshold for returned readings (default 0.50).

    Examples
    --------
    >>> ext = ClinicalVitalSignsExtractor()
    >>> result = ext.extract("BP 120/80 mmHg, HR 72 bpm, T 98.6 F")
    >>> len(result.readings)
    3
    """

    def __init__(self, min_confidence: float = 0.50) -> None:
        self.min_confidence = min_confidence
        # Pre-compile all patterns
        self._bp_patterns = _bp_patterns()
        self._hr_patterns = _hr_patterns()
        self._temp_patterns = _temp_patterns()
        self._rr_patterns = _rr_patterns()
        self._spo2_patterns = _spo2_patterns()
        self._weight_patterns = _weight_patterns()
        self._height_patterns = _height_patterns()
        self._bmi_patterns = _bmi_patterns()
        self._pain_patterns = _pain_patterns()

    # -------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------

    def extract(self, text: str) -> VitalSignsResult:
        """Extract all vital signs from clinical text.

        Parameters
        ----------
        text : str
            Clinical note text.

        Returns
        -------
        VitalSignsResult
            All extracted readings with metadata.
        """
        t0 = time.perf_counter()

        if not text or not text.strip():
            return VitalSignsResult(
                text_hash=hashlib.sha256(b"").hexdigest(),
                extraction_time_ms=0.0,
                summary={"total": 0, "by_type": {}, "critical_findings": []},
            )

        text_hash = hashlib.sha256(text.encode()).hexdigest()
        readings: list[VitalSignReading] = []

        # Extract each vital type
        readings.extend(self._extract_blood_pressure(text))
        readings.extend(self._extract_heart_rate(text))
        readings.extend(self._extract_temperature(text))
        readings.extend(self._extract_respiratory_rate(text))
        readings.extend(self._extract_spo2(text))
        readings.extend(self._extract_weight(text))
        readings.extend(self._extract_height(text))
        readings.extend(self._extract_bmi(text))
        readings.extend(self._extract_pain(text))
        readings.extend(self._extract_qualitative(text))

        # Deduplicate overlapping spans
        readings = self._deduplicate(readings)

        # Apply section-aware confidence boosting
        for r in readings:
            if _in_vitals_section(text, r.start):
                r.confidence = min(1.0, r.confidence + _SECTION_CONFIDENCE_BOOST)

        # Detect trends
        for r in readings:
            if r.trend == VitalTrend.UNKNOWN:
                r.trend = _detect_trend(text, r.start, r.end)

        # Filter by min confidence
        readings = [r for r in readings if r.confidence >= self.min_confidence]

        # Sort by position
        readings.sort(key=lambda r: r.start)

        # Try to calculate BMI if we have weight and height but no BMI
        bmi_types = {r.vital_type for r in readings}
        if VitalSignType.BMI not in bmi_types:
            weight_r = next((r for r in readings if r.vital_type == VitalSignType.WEIGHT), None)
            height_r = next((r for r in readings if r.vital_type == VitalSignType.HEIGHT), None)
            if weight_r and height_r:
                bmi_val = _calculate_bmi(weight_r.value, height_r.value)
                if bmi_val and _is_valid(VitalSignType.BMI, bmi_val):
                    readings.append(VitalSignReading(
                        vital_type=VitalSignType.BMI,
                        value=bmi_val,
                        unit="kg/m²",
                        raw_text=f"(calculated: {bmi_val})",
                        start=weight_r.start,
                        end=height_r.end,
                        confidence=min(weight_r.confidence, height_r.confidence) * 0.9,
                        interpretation=_interpret(VitalSignType.BMI, bmi_val),
                        metadata={"calculated": True, "weight_kg": weight_r.value, "height_cm": height_r.value},
                    ))

        elapsed = (time.perf_counter() - t0) * 1000.0

        # Build summary
        summary = self._build_summary(readings)

        return VitalSignsResult(
            readings=readings,
            text_hash=text_hash,
            extraction_time_ms=elapsed,
            summary=summary,
        )

    def extract_batch(
        self, texts: list[str]
    ) -> list[VitalSignsResult]:
        """Extract vital signs from multiple documents.

        Parameters
        ----------
        texts : list[str]
            List of clinical note texts.

        Returns
        -------
        list[VitalSignsResult]
            One result per input text.
        """
        return [self.extract(t) for t in texts]

    # -------------------------------------------------------------------
    # Per-type extractors
    # -------------------------------------------------------------------

    def _extract_blood_pressure(self, text: str) -> list[VitalSignReading]:
        """Extract blood pressure readings.

        Parameters
        ----------
        text : str
            Clinical text.

        Returns
        -------
        list[VitalSignReading]
            Extracted BP readings with systolic as primary value.
        """
        results: list[VitalSignReading] = []
        for pattern, label in self._bp_patterns:
            for m in pattern.finditer(text):
                systolic = float(m.group(1))
                diastolic = float(m.group(2))

                if not _is_valid(VitalSignType.BLOOD_PRESSURE, systolic):
                    continue
                if diastolic < 10 or diastolic > 200:
                    continue
                if systolic <= diastolic:
                    continue

                # MAP = (SBP + 2*DBP) / 3
                mean_arterial = round((systolic + 2 * diastolic) / 3.0, 1)

                confidence = 0.90 if label == "labeled" else 0.80
                sys_interp = _interpret(VitalSignType.BLOOD_PRESSURE, systolic)
                dia_interp = _interpret_diastolic(diastolic)
                # Use the worse interpretation
                interp = max(sys_interp, dia_interp, key=lambda i: [
                    ClinicalInterpretation.NORMAL,
                    ClinicalInterpretation.LOW,
                    ClinicalInterpretation.HIGH,
                    ClinicalInterpretation.CRITICAL_LOW,
                    ClinicalInterpretation.CRITICAL_HIGH,
                ].index(i))

                results.append(VitalSignReading(
                    vital_type=VitalSignType.BLOOD_PRESSURE,
                    value=systolic,
                    unit="mmHg",
                    raw_text=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    confidence=confidence,
                    interpretation=interp,
                    secondary_value=diastolic,
                    metadata={
                        "systolic": systolic,
                        "diastolic": diastolic,
                        "mean_arterial_pressure": mean_arterial,
                        "systolic_interpretation": sys_interp.value,
                        "diastolic_interpretation": dia_interp.value,
                    },
                ))
        return results

    def _extract_heart_rate(self, text: str) -> list[VitalSignReading]:
        """Extract heart rate readings.

        Parameters
        ----------
        text : str
            Clinical text.

        Returns
        -------
        list[VitalSignReading]
            Extracted HR readings.
        """
        results: list[VitalSignReading] = []
        for pattern in self._hr_patterns:
            for m in pattern.finditer(text):
                value = float(m.group(1))
                if not _is_valid(VitalSignType.HEART_RATE, value):
                    continue
                results.append(VitalSignReading(
                    vital_type=VitalSignType.HEART_RATE,
                    value=value,
                    unit="bpm",
                    raw_text=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    confidence=0.88,
                    interpretation=_interpret(VitalSignType.HEART_RATE, value),
                ))
        return results

    def _extract_temperature(self, text: str) -> list[VitalSignReading]:
        """Extract temperature readings with unit conversion.

        Parameters
        ----------
        text : str
            Clinical text.

        Returns
        -------
        list[VitalSignReading]
            Extracted temperature readings normalised to °F.
        """
        results: list[VitalSignReading] = []
        for pattern, unit_hint in self._temp_patterns:
            for m in pattern.finditer(text):
                raw_value = float(m.group(1))
                metadata: dict[str, Any] = {}

                if unit_hint == "explicit":
                    unit_char = m.group(2).upper()
                    if unit_char == "C":
                        value_f = _celsius_to_fahrenheit(raw_value)
                        metadata["original_value"] = raw_value
                        metadata["original_unit"] = "°C"
                        metadata["converted"] = True
                    else:
                        value_f = raw_value
                else:
                    # Auto-detect: if > 50, assume °F; else °C
                    if raw_value <= 50:
                        value_f = _celsius_to_fahrenheit(raw_value)
                        metadata["original_value"] = raw_value
                        metadata["original_unit"] = "°C"
                        metadata["converted"] = True
                    else:
                        value_f = raw_value

                if not _is_valid(VitalSignType.TEMPERATURE, value_f):
                    continue

                results.append(VitalSignReading(
                    vital_type=VitalSignType.TEMPERATURE,
                    value=value_f,
                    unit="°F",
                    raw_text=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    confidence=0.88 if unit_hint == "explicit" else 0.78,
                    interpretation=_interpret(VitalSignType.TEMPERATURE, value_f),
                    metadata=metadata,
                ))
        return results

    def _extract_respiratory_rate(self, text: str) -> list[VitalSignReading]:
        """Extract respiratory rate readings.

        Parameters
        ----------
        text : str
            Clinical text.

        Returns
        -------
        list[VitalSignReading]
            Extracted RR readings.
        """
        results: list[VitalSignReading] = []
        for pattern in self._rr_patterns:
            for m in pattern.finditer(text):
                value = float(m.group(1))
                if not _is_valid(VitalSignType.RESPIRATORY_RATE, value):
                    continue
                results.append(VitalSignReading(
                    vital_type=VitalSignType.RESPIRATORY_RATE,
                    value=value,
                    unit="breaths/min",
                    raw_text=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    confidence=0.88,
                    interpretation=_interpret(VitalSignType.RESPIRATORY_RATE, value),
                ))
        return results

    def _extract_spo2(self, text: str) -> list[VitalSignReading]:
        """Extract oxygen saturation readings.

        Parameters
        ----------
        text : str
            Clinical text.

        Returns
        -------
        list[VitalSignReading]
            Extracted SpO2 readings.
        """
        results: list[VitalSignReading] = []
        for pattern in self._spo2_patterns:
            for m in pattern.finditer(text):
                value = float(m.group(1))
                if not _is_valid(VitalSignType.OXYGEN_SATURATION, value):
                    continue
                results.append(VitalSignReading(
                    vital_type=VitalSignType.OXYGEN_SATURATION,
                    value=value,
                    unit="%",
                    raw_text=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    confidence=0.88,
                    interpretation=_interpret(VitalSignType.OXYGEN_SATURATION, value),
                ))
        return results

    def _extract_weight(self, text: str) -> list[VitalSignReading]:
        """Extract weight readings with unit conversion to kg.

        Parameters
        ----------
        text : str
            Clinical text.

        Returns
        -------
        list[VitalSignReading]
            Extracted weight readings normalised to kg.
        """
        results: list[VitalSignReading] = []
        for pattern, unit_hint in self._weight_patterns:
            for m in pattern.finditer(text):
                raw_value = float(m.group(1))
                metadata: dict[str, Any] = {}

                if unit_hint == "kg":
                    value_kg = raw_value
                else:
                    unit_str = m.group(2).lower()
                    if unit_str in ("lb", "lbs", "pound", "pounds"):
                        value_kg = _lbs_to_kg(raw_value)
                        metadata["original_value"] = raw_value
                        metadata["original_unit"] = "lbs"
                        metadata["converted"] = True
                    else:
                        value_kg = raw_value

                if not _is_valid(VitalSignType.WEIGHT, value_kg):
                    continue

                results.append(VitalSignReading(
                    vital_type=VitalSignType.WEIGHT,
                    value=value_kg,
                    unit="kg",
                    raw_text=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    confidence=0.85,
                    interpretation=ClinicalInterpretation.NORMAL,
                    metadata=metadata,
                ))
        return results

    def _extract_height(self, text: str) -> list[VitalSignReading]:
        """Extract height readings with unit conversion to cm.

        Parameters
        ----------
        text : str
            Clinical text.

        Returns
        -------
        list[VitalSignReading]
            Extracted height readings normalised to cm.
        """
        results: list[VitalSignReading] = []
        for pattern, fmt in self._height_patterns:
            for m in pattern.finditer(text):
                metadata: dict[str, Any] = {}

                if fmt == "feet_inches":
                    feet = int(m.group(1))
                    inches = int(m.group(2))
                    value_cm = _feet_inches_to_cm(feet, inches)
                    metadata["original_feet"] = feet
                    metadata["original_inches"] = inches
                    metadata["converted"] = True
                else:
                    raw_value = float(m.group(1))
                    unit_str = m.group(2).lower()
                    if unit_str.startswith("in"):
                        value_cm = _inches_to_cm(raw_value)
                        metadata["original_value"] = raw_value
                        metadata["original_unit"] = "inches"
                        metadata["converted"] = True
                    else:
                        value_cm = raw_value

                if not _is_valid(VitalSignType.HEIGHT, value_cm):
                    continue

                results.append(VitalSignReading(
                    vital_type=VitalSignType.HEIGHT,
                    value=value_cm,
                    unit="cm",
                    raw_text=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    confidence=0.85,
                    interpretation=ClinicalInterpretation.NORMAL,
                    metadata=metadata,
                ))
        return results

    def _extract_bmi(self, text: str) -> list[VitalSignReading]:
        """Extract BMI readings.

        Parameters
        ----------
        text : str
            Clinical text.

        Returns
        -------
        list[VitalSignReading]
            Extracted BMI readings.
        """
        results: list[VitalSignReading] = []
        for pattern in self._bmi_patterns:
            for m in pattern.finditer(text):
                value = float(m.group(1))
                if not _is_valid(VitalSignType.BMI, value):
                    continue
                results.append(VitalSignReading(
                    vital_type=VitalSignType.BMI,
                    value=value,
                    unit="kg/m²",
                    raw_text=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    confidence=0.88,
                    interpretation=_interpret(VitalSignType.BMI, value),
                ))
        return results

    def _extract_pain(self, text: str) -> list[VitalSignReading]:
        """Extract pain scale readings.

        Parameters
        ----------
        text : str
            Clinical text.

        Returns
        -------
        list[VitalSignReading]
            Extracted pain scale readings.
        """
        results: list[VitalSignReading] = []
        for pattern in self._pain_patterns:
            for m in pattern.finditer(text):
                value = float(m.group(1))
                if not _is_valid(VitalSignType.PAIN_SCALE, value):
                    continue
                results.append(VitalSignReading(
                    vital_type=VitalSignType.PAIN_SCALE,
                    value=value,
                    unit="/10",
                    raw_text=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    confidence=0.85,
                    interpretation=_interpret(VitalSignType.PAIN_SCALE, value),
                ))
        return results

    def _extract_qualitative(self, text: str) -> list[VitalSignReading]:
        """Extract qualitative vital sign descriptors.

        Matches terms like "afebrile", "tachycardic", "hypotensive" and
        converts them to approximate numeric values with lower confidence.

        Parameters
        ----------
        text : str
            Clinical text.

        Returns
        -------
        list[VitalSignReading]
            Extracted qualitative readings.
        """
        results: list[VitalSignReading] = []
        for pattern, vtype, implied_value, interp, description in _QUALITATIVE_VITALS:
            for m in pattern.finditer(text):
                results.append(VitalSignReading(
                    vital_type=vtype,
                    value=implied_value,
                    unit={
                        VitalSignType.TEMPERATURE: "°F",
                        VitalSignType.HEART_RATE: "bpm",
                        VitalSignType.RESPIRATORY_RATE: "breaths/min",
                        VitalSignType.BLOOD_PRESSURE: "mmHg",
                        VitalSignType.OXYGEN_SATURATION: "%",
                    }.get(vtype, ""),
                    raw_text=m.group(0),
                    start=m.start(),
                    end=m.end(),
                    confidence=0.70,
                    interpretation=interp,
                    metadata={"qualitative": True, "description": description},
                ))
        return results

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _deduplicate(readings: list[VitalSignReading]) -> list[VitalSignReading]:
        """Remove overlapping spans, keeping the higher-confidence reading.

        Parameters
        ----------
        readings : list[VitalSignReading]
            Potentially overlapping readings.

        Returns
        -------
        list[VitalSignReading]
            Deduplicated readings.
        """
        if len(readings) <= 1:
            return readings

        # Sort by confidence descending
        readings.sort(key=lambda r: -r.confidence)
        kept: list[VitalSignReading] = []
        for r in readings:
            overlap = False
            for k in kept:
                if r.start < k.end and r.end > k.start and r.vital_type == k.vital_type:
                    overlap = True
                    break
            if not overlap:
                kept.append(r)
        return kept

    @staticmethod
    def _build_summary(readings: list[VitalSignReading]) -> dict[str, Any]:
        """Build aggregate summary from readings.

        Parameters
        ----------
        readings : list[VitalSignReading]
            All extracted readings.

        Returns
        -------
        dict[str, Any]
            Summary with total count, per-type counts, and critical findings.
        """
        by_type: dict[str, int] = {}
        critical: list[dict[str, Any]] = []

        for r in readings:
            by_type[r.vital_type.value] = by_type.get(r.vital_type.value, 0) + 1
            if r.interpretation in (
                ClinicalInterpretation.CRITICAL_LOW,
                ClinicalInterpretation.CRITICAL_HIGH,
            ):
                critical.append({
                    "vital_type": r.vital_type.value,
                    "value": r.value,
                    "unit": r.unit,
                    "interpretation": r.interpretation.value,
                })

        return {
            "total": len(readings),
            "by_type": by_type,
            "critical_findings": critical,
        }
