"""Clinical temporal information extraction.

Parses dates, durations, frequencies, relative time references, and temporal
relations from clinical notes.  Produces normalised :class:`TemporalExpression`
objects that can be anchored to a reference date for patient timeline
construction.

Design decisions
----------------
* **Pattern-first** — Clinical temporal expressions follow highly regular
  patterns ("on 03/15/2024", "for 6 weeks", "q8h").  Regex-based extraction
  is fast, deterministic, and requires no external dependencies.
* **Reference-date anchoring** — Relative expressions ("3 days ago",
  "last week") are resolved against a configurable reference date, defaulting
  to today.
* **Temporal relation modelling** — Beyond extracting *when*, the module
  identifies temporal ordering between events (BEFORE, AFTER, OVERLAP,
  CONTAINS, SIMULTANEOUS) following a simplified TimeML taxonomy.
* **Clinical frequency normalisation** — Medical shorthand (BID, TID, q6h,
  PRN) is mapped to structured :class:`Frequency` objects with daily
  occurrence counts for downstream dosing calculations.

Architecture
-----------
::

    Clinical text ─► DateExtractor ──────────────┐
                 ├► DurationExtractor ────────────┤
                 ├► FrequencyExtractor ───────────┼─► TemporalExtractionResult
                 ├► RelativeTimeExtractor ────────┤
                 └► TemporalRelationResolver ─────┘
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums & constants
# ---------------------------------------------------------------------------

class TemporalType(StrEnum):
    """Category of temporal expression."""

    DATE = "date"                # Specific date: "March 15, 2024"
    DATETIME = "datetime"        # Date + time: "03/15/2024 at 14:30"
    DURATION = "duration"        # Time span: "for 6 weeks"
    FREQUENCY = "frequency"      # Recurrence: "twice daily", "q8h"
    RELATIVE = "relative"        # Relative reference: "3 days ago"
    AGE = "age"                  # Patient age: "72-year-old"
    PERIOD = "period"            # Named period: "postoperative day 3"


class TemporalRelation(StrEnum):
    """Temporal ordering between two events.

    Based on a simplified Allen's interval algebra, matching the
    relations most commonly needed in clinical timeline construction.
    """

    BEFORE = "before"            # Event A ends before B starts
    AFTER = "after"              # Event A starts after B ends
    OVERLAP = "overlap"          # Events share some time period
    CONTAINS = "contains"        # Event A fully contains B
    SIMULTANEOUS = "simultaneous"  # Events occur at the same time
    BEGINS = "begins"            # Event A starts when B starts
    ENDS = "ends"                # Event A ends when B ends


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TemporalExpression:
    """A normalised temporal expression extracted from text.

    Parameters
    ----------
    text:
        The raw text span matched in the source document.
    temporal_type:
        Category of the expression.
    start_char:
        Character offset of the expression start.
    end_char:
        Character offset of the expression end.
    confidence:
        Extraction confidence in [0, 1].
    normalised_value:
        ISO-8601 normalised string (date, datetime, duration).
    resolved_date:
        Absolute date after resolving relative references.
    duration_days:
        For durations, the length in days.
    metadata:
        Arbitrary additional fields.
    """

    text: str
    temporal_type: TemporalType
    start_char: int
    end_char: int
    confidence: float
    normalised_value: str | None = None
    resolved_date: date | None = None
    duration_days: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "text": self.text,
            "temporal_type": self.temporal_type.value,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": round(self.confidence, 4),
            "normalised_value": self.normalised_value,
            "resolved_date": self.resolved_date.isoformat() if self.resolved_date else None,
            "duration_days": self.duration_days,
            "metadata": self.metadata,
        }


@dataclass
class Frequency:
    """Normalised medication/treatment frequency.

    Parameters
    ----------
    text:
        Original text (e.g. "BID", "every 8 hours").
    times_per_day:
        Number of occurrences per 24 hours.
    interval_hours:
        Hours between doses.
    as_needed:
        Whether the frequency is PRN (as needed).
    """

    text: str
    times_per_day: float
    interval_hours: float
    as_needed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "text": self.text,
            "times_per_day": self.times_per_day,
            "interval_hours": round(self.interval_hours, 2),
            "as_needed": self.as_needed,
        }


@dataclass
class TemporalLink:
    """A temporal ordering relation between two text spans.

    Parameters
    ----------
    source_span:
        The first event or time expression text.
    target_span:
        The second event or time expression text.
    relation:
        The temporal ordering.
    confidence:
        Confidence in [0, 1].
    evidence:
        The connecting text that signals the relation.
    """

    source_span: str
    target_span: str
    relation: TemporalRelation
    confidence: float
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "source": self.source_span,
            "target": self.target_span,
            "relation": self.relation.value,
            "confidence": round(self.confidence, 4),
            "evidence": self.evidence,
        }


@dataclass
class TemporalExtractionResult:
    """Container for all temporal information extracted from a document.

    Parameters
    ----------
    expressions:
        All temporal expressions found.
    frequencies:
        Normalised medication/treatment frequencies.
    temporal_links:
        Ordering relations between events.
    reference_date:
        The anchor date used for resolving relative expressions.
    processing_time_ms:
        Wall-clock extraction time.
    """

    expressions: list[TemporalExpression]
    frequencies: list[Frequency]
    temporal_links: list[TemporalLink]
    reference_date: date
    processing_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-safe dictionary."""
        return {
            "expressions": [e.to_dict() for e in self.expressions],
            "frequencies": [f.to_dict() for f in self.frequencies],
            "temporal_links": [t.to_dict() for t in self.temporal_links],
            "reference_date": self.reference_date.isoformat(),
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


# ---------------------------------------------------------------------------
# Clinical frequency abbreviation map
# ---------------------------------------------------------------------------

# Maps medical abbreviations to (times_per_day, interval_hours, as_needed).
# Sources: ISMP abbreviation list, Joint Commission "Do Not Use" list.
FREQUENCY_MAP: dict[str, tuple[float, float, bool]] = {
    # Latin-derived abbreviations
    "qd": (1.0, 24.0, False),
    "od": (1.0, 24.0, False),
    "daily": (1.0, 24.0, False),
    "once daily": (1.0, 24.0, False),
    "bid": (2.0, 12.0, False),
    "twice daily": (2.0, 12.0, False),
    "tid": (3.0, 8.0, False),
    "three times daily": (3.0, 8.0, False),
    "qid": (4.0, 6.0, False),
    "four times daily": (4.0, 6.0, False),
    "qhs": (1.0, 24.0, False),
    "at bedtime": (1.0, 24.0, False),
    "hs": (1.0, 24.0, False),
    "qam": (1.0, 24.0, False),
    "every morning": (1.0, 24.0, False),
    "qpm": (1.0, 24.0, False),
    "every evening": (1.0, 24.0, False),
    "qod": (0.5, 48.0, False),
    "every other day": (0.5, 48.0, False),
    "qwk": (1 / 7, 168.0, False),
    "weekly": (1 / 7, 168.0, False),
    "once weekly": (1 / 7, 168.0, False),
    "biweekly": (1 / 14, 336.0, False),
    "monthly": (1 / 30, 720.0, False),
    "once monthly": (1 / 30, 720.0, False),

    # Interval-based
    "q2h": (12.0, 2.0, False),
    "q3h": (8.0, 3.0, False),
    "q4h": (6.0, 4.0, False),
    "q6h": (4.0, 6.0, False),
    "q8h": (3.0, 8.0, False),
    "q12h": (2.0, 12.0, False),
    "q24h": (1.0, 24.0, False),
    "q48h": (0.5, 48.0, False),
    "q72h": (1 / 3, 72.0, False),

    # PRN (as-needed) combinations
    "prn": (0.0, 0.0, True),
    "as needed": (0.0, 0.0, True),
    "q4h prn": (6.0, 4.0, True),
    "q6h prn": (4.0, 6.0, True),
    "q8h prn": (3.0, 8.0, True),

    # Meal-related
    "ac": (3.0, 8.0, False),  # before meals
    "pc": (3.0, 8.0, False),  # after meals
    "with meals": (3.0, 8.0, False),

    # Stat / one-time
    "stat": (1.0, 0.0, False),
    "once": (1.0, 0.0, False),
    "x1": (1.0, 0.0, False),
}


# ---------------------------------------------------------------------------
# Month name map for date parsing
# ---------------------------------------------------------------------------

MONTH_NAMES: dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


# ---------------------------------------------------------------------------
# Duration unit map (unit → days)
# ---------------------------------------------------------------------------

DURATION_UNITS: dict[str, float] = {
    "minute": 1 / 1440,
    "minutes": 1 / 1440,
    "min": 1 / 1440,
    "mins": 1 / 1440,
    "hour": 1 / 24,
    "hours": 1 / 24,
    "hr": 1 / 24,
    "hrs": 1 / 24,
    "day": 1.0,
    "days": 1.0,
    "week": 7.0,
    "weeks": 7.0,
    "wk": 7.0,
    "wks": 7.0,
    "month": 30.0,
    "months": 30.0,
    "mo": 30.0,
    "mos": 30.0,
    "year": 365.0,
    "years": 365.0,
    "yr": 365.0,
    "yrs": 365.0,
}


# ---------------------------------------------------------------------------
# Temporal relation signal patterns
# ---------------------------------------------------------------------------

_BEFORE_SIGNALS = [
    r"\b(?:before|prior to|preceding|pre-|earlier than|leading up to)\b",
]

_AFTER_SIGNALS = [
    r"\b(?:after|following|subsequent to|post-|later than|since)\b",
]

_SIMULTANEOUS_SIGNALS = [
    r"\b(?:during|while|at the (?:same )?time|concurrent|simultaneously|upon)\b",
]

_OVERLAP_SIGNALS = [
    r"\b(?:throughout|over the course of|spanning)\b",
]


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class ClinicalTemporalExtractor:
    """Extract and normalise temporal information from clinical text.

    Combines date, duration, frequency, relative-time, and temporal-relation
    extraction into a single pass.  All sub-extractors use compiled regexes
    and require no external dependencies.

    Parameters
    ----------
    reference_date:
        Anchor date for resolving relative expressions.  Defaults to today.
    """

    # ---- Date patterns ----
    # MM/DD/YYYY or MM-DD-YYYY
    _RE_DATE_MDY = re.compile(
        r"\b(0?[1-9]|1[0-2])[/\-](0?[1-9]|[12]\d|3[01])[/\-]((?:19|20)\d{2})\b"
    )
    # YYYY-MM-DD (ISO)
    _RE_DATE_ISO = re.compile(
        r"\b((?:19|20)\d{2})-(0?[1-9]|1[0-2])-(0?[1-9]|[12]\d|3[01])\b"
    )
    # Month DD, YYYY  /  DD Month YYYY
    _RE_DATE_WRITTEN = re.compile(
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?)"
        r"\s+(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?\s+"
        r"((?:19|20)\d{2})\b",
        re.IGNORECASE,
    )
    _RE_DATE_DMY = re.compile(
        r"\b(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+"
        r"((?:January|February|March|April|May|June|July|August|September|October|November|December|"
        r"Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?)"
        r",?\s+((?:19|20)\d{2})\b",
        re.IGNORECASE,
    )

    # ---- Duration patterns ----
    _RE_DURATION = re.compile(
        r"\b(?:for|over|lasting|duration of|x)\s*"
        r"(\d+(?:\.\d+)?)\s*"
        r"(minutes?|mins?|hours?|hrs?|days?|weeks?|wks?|months?|mos?|years?|yrs?)\b",
        re.IGNORECASE,
    )
    _RE_DURATION_RANGE = re.compile(
        r"\b(\d+)\s*(?:to|-)\s*(\d+)\s*"
        r"(minutes?|mins?|hours?|hrs?|days?|weeks?|wks?|months?|mos?|years?|yrs?)\b",
        re.IGNORECASE,
    )

    # ---- Relative time patterns ----
    _RE_RELATIVE_AGO = re.compile(
        r"\b(\d+)\s*(minutes?|mins?|hours?|hrs?|days?|weeks?|wks?|months?|mos?|years?|yrs?)\s+ago\b",
        re.IGNORECASE,
    )
    _RE_RELATIVE_LAST = re.compile(
        r"\blast\s+(week|month|year|monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
        re.IGNORECASE,
    )
    _RE_RELATIVE_NAMED = re.compile(
        r"\b(today|yesterday|tomorrow|tonight|this morning|this afternoon|this evening)\b",
        re.IGNORECASE,
    )

    # ---- Frequency patterns ----
    _RE_FREQ_Q = re.compile(
        r"\bq(\d+)h\s*(?:prn)?\b",
        re.IGNORECASE,
    )
    _RE_FREQ_ABBREV = re.compile(
        r"\b(QD|OD|BID|TID|QID|QHS|QAM|QPM|QOD|QWK|PRN|STAT|AC|PC)\b",
        re.IGNORECASE,
    )
    _RE_FREQ_WRITTEN = re.compile(
        r"\b(once|twice|three times|four times)\s+(daily|weekly|monthly)\b",
        re.IGNORECASE,
    )
    _RE_FREQ_EVERY = re.compile(
        r"\bevery\s+(\d+)\s*(hours?|hrs?|days?|weeks?|months?)\b",
        re.IGNORECASE,
    )

    # ---- Age pattern ----
    _RE_AGE = re.compile(
        r"\b(\d{1,3})\s*[-–]?\s*year\s*[-–]?\s*old\b",
        re.IGNORECASE,
    )

    # ---- Postoperative day ----
    _RE_POD = re.compile(
        r"\b(?:post[- ]?(?:operative|op)|POD)\s*(?:day\s*)?#?\s*(\d+)\b",
        re.IGNORECASE,
    )

    def __init__(self, reference_date: date | None = None) -> None:
        self.reference_date = reference_date or date.today()

    def extract(self, text: str) -> TemporalExtractionResult:
        """Extract all temporal information from *text*.

        Parameters
        ----------
        text:
            Clinical note text.

        Returns
        -------
        TemporalExtractionResult
            All expressions, frequencies, and temporal links found.
        """
        start = time.perf_counter()

        expressions: list[TemporalExpression] = []
        frequencies: list[Frequency] = []

        # 1) Dates
        expressions.extend(self._extract_dates(text))

        # 2) Durations
        expressions.extend(self._extract_durations(text))

        # 3) Relative time references
        expressions.extend(self._extract_relative(text))

        # 4) Ages
        expressions.extend(self._extract_ages(text))

        # 5) Postoperative days
        expressions.extend(self._extract_pod(text))

        # 6) Frequencies
        frequencies.extend(self._extract_frequencies(text))

        # 7) Temporal relations between events
        temporal_links = self._extract_temporal_links(text, expressions)

        # De-duplicate overlapping expressions (keep higher confidence)
        expressions = self._deduplicate(expressions)

        elapsed = (time.perf_counter() - start) * 1000

        return TemporalExtractionResult(
            expressions=expressions,
            frequencies=frequencies,
            temporal_links=temporal_links,
            reference_date=self.reference_date,
            processing_time_ms=elapsed,
        )

    # ------------------------------------------------------------------
    # Date extraction
    # ------------------------------------------------------------------

    def _extract_dates(self, text: str) -> list[TemporalExpression]:
        """Extract explicit date mentions."""
        results: list[TemporalExpression] = []

        # MM/DD/YYYY
        for m in self._RE_DATE_MDY.finditer(text):
            try:
                month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
                d = date(year, month, day)
                results.append(TemporalExpression(
                    text=m.group(0),
                    temporal_type=TemporalType.DATE,
                    start_char=m.start(),
                    end_char=m.end(),
                    confidence=0.95,
                    normalised_value=d.isoformat(),
                    resolved_date=d,
                ))
            except ValueError:
                pass

        # YYYY-MM-DD
        for m in self._RE_DATE_ISO.finditer(text):
            try:
                year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
                d = date(year, month, day)
                results.append(TemporalExpression(
                    text=m.group(0),
                    temporal_type=TemporalType.DATE,
                    start_char=m.start(),
                    end_char=m.end(),
                    confidence=0.95,
                    normalised_value=d.isoformat(),
                    resolved_date=d,
                ))
            except ValueError:
                pass

        # Month DD, YYYY
        for m in self._RE_DATE_WRITTEN.finditer(text):
            try:
                month_name = m.group(1).rstrip(".").lower()
                month = MONTH_NAMES.get(month_name)
                if month is None:
                    continue
                day = int(m.group(2))
                year = int(m.group(3))
                d = date(year, month, day)
                results.append(TemporalExpression(
                    text=m.group(0),
                    temporal_type=TemporalType.DATE,
                    start_char=m.start(),
                    end_char=m.end(),
                    confidence=0.90,
                    normalised_value=d.isoformat(),
                    resolved_date=d,
                ))
            except ValueError:
                pass

        # DD Month YYYY
        for m in self._RE_DATE_DMY.finditer(text):
            try:
                day = int(m.group(1))
                month_name = m.group(2).rstrip(".").lower()
                month = MONTH_NAMES.get(month_name)
                if month is None:
                    continue
                year = int(m.group(3))
                d = date(year, month, day)
                results.append(TemporalExpression(
                    text=m.group(0),
                    temporal_type=TemporalType.DATE,
                    start_char=m.start(),
                    end_char=m.end(),
                    confidence=0.90,
                    normalised_value=d.isoformat(),
                    resolved_date=d,
                ))
            except ValueError:
                pass

        return results

    # ------------------------------------------------------------------
    # Duration extraction
    # ------------------------------------------------------------------

    def _extract_durations(self, text: str) -> list[TemporalExpression]:
        """Extract duration expressions (e.g. 'for 6 weeks')."""
        results: list[TemporalExpression] = []

        for m in self._RE_DURATION.finditer(text):
            amount = float(m.group(1))
            unit = m.group(2).lower().rstrip("s")
            # Normalise unit aliases
            if unit in ("min",):
                unit = "minute"
            elif unit in ("hr",):
                unit = "hour"
            elif unit in ("wk",):
                unit = "week"
            elif unit in ("mo",):
                unit = "month"
            elif unit in ("yr",):
                unit = "year"

            days = amount * DURATION_UNITS.get(unit, DURATION_UNITS.get(unit + "s", 1.0))

            results.append(TemporalExpression(
                text=m.group(0),
                temporal_type=TemporalType.DURATION,
                start_char=m.start(),
                end_char=m.end(),
                confidence=0.85,
                normalised_value=f"P{amount}{unit[0].upper()}",
                duration_days=days,
            ))

        # Range durations: "3 to 5 days"
        for m in self._RE_DURATION_RANGE.finditer(text):
            lo = float(m.group(1))
            hi = float(m.group(2))
            unit = m.group(3).lower().rstrip("s")
            if unit in ("min",):
                unit = "minute"
            elif unit in ("hr",):
                unit = "hour"
            elif unit in ("wk",):
                unit = "week"
            elif unit in ("mo",):
                unit = "month"
            elif unit in ("yr",):
                unit = "year"

            avg = (lo + hi) / 2
            days = avg * DURATION_UNITS.get(unit, DURATION_UNITS.get(unit + "s", 1.0))

            results.append(TemporalExpression(
                text=m.group(0),
                temporal_type=TemporalType.DURATION,
                start_char=m.start(),
                end_char=m.end(),
                confidence=0.80,
                normalised_value=f"P{lo}-{hi}{unit[0].upper()}",
                duration_days=days,
                metadata={"range_low": lo, "range_high": hi},
            ))

        return results

    # ------------------------------------------------------------------
    # Relative time extraction
    # ------------------------------------------------------------------

    def _extract_relative(self, text: str) -> list[TemporalExpression]:
        """Extract relative time references ('3 days ago', 'yesterday')."""
        results: list[TemporalExpression] = []

        # "N units ago"
        for m in self._RE_RELATIVE_AGO.finditer(text):
            amount = int(m.group(1))
            unit = m.group(2).lower().rstrip("s")
            if unit in ("min",):
                unit = "minute"
            elif unit in ("hr",):
                unit = "hour"
            elif unit in ("wk",):
                unit = "week"
            elif unit in ("mo",):
                unit = "month"
            elif unit in ("yr",):
                unit = "year"

            days_back = amount * DURATION_UNITS.get(unit, DURATION_UNITS.get(unit + "s", 1.0))
            resolved = self.reference_date - timedelta(days=days_back)

            results.append(TemporalExpression(
                text=m.group(0),
                temporal_type=TemporalType.RELATIVE,
                start_char=m.start(),
                end_char=m.end(),
                confidence=0.80,
                normalised_value=f"-P{amount}{unit[0].upper()}",
                resolved_date=resolved,
                metadata={"direction": "past", "amount": amount, "unit": unit},
            ))

        # Named relative: today, yesterday, tomorrow
        named_offsets = {
            "today": 0,
            "tonight": 0,
            "this morning": 0,
            "this afternoon": 0,
            "this evening": 0,
            "yesterday": -1,
            "tomorrow": 1,
        }
        for m in self._RE_RELATIVE_NAMED.finditer(text):
            name = m.group(1).lower()
            offset = named_offsets.get(name, 0)
            resolved = self.reference_date + timedelta(days=offset)

            results.append(TemporalExpression(
                text=m.group(0),
                temporal_type=TemporalType.RELATIVE,
                start_char=m.start(),
                end_char=m.end(),
                confidence=0.90,
                normalised_value=resolved.isoformat(),
                resolved_date=resolved,
                metadata={"named_reference": name},
            ))

        # "last week/month/year"
        last_offsets = {
            "week": 7,
            "month": 30,
            "year": 365,
        }
        for m in self._RE_RELATIVE_LAST.finditer(text):
            unit = m.group(1).lower()
            days_back = last_offsets.get(unit, 7)
            resolved = self.reference_date - timedelta(days=days_back)

            results.append(TemporalExpression(
                text=m.group(0),
                temporal_type=TemporalType.RELATIVE,
                start_char=m.start(),
                end_char=m.end(),
                confidence=0.75,
                normalised_value=f"-P{days_back}D",
                resolved_date=resolved,
                metadata={"direction": "past", "reference": unit},
            ))

        return results

    # ------------------------------------------------------------------
    # Age extraction
    # ------------------------------------------------------------------

    def _extract_ages(self, text: str) -> list[TemporalExpression]:
        """Extract patient age references."""
        results: list[TemporalExpression] = []

        for m in self._RE_AGE.finditer(text):
            age = int(m.group(1))
            if age > 130:  # Sanity check
                continue
            results.append(TemporalExpression(
                text=m.group(0),
                temporal_type=TemporalType.AGE,
                start_char=m.start(),
                end_char=m.end(),
                confidence=0.90,
                normalised_value=f"P{age}Y",
                duration_days=age * 365.25,
                metadata={"age_years": age},
            ))

        return results

    # ------------------------------------------------------------------
    # Postoperative day extraction
    # ------------------------------------------------------------------

    def _extract_pod(self, text: str) -> list[TemporalExpression]:
        """Extract postoperative day references (POD #3, post-op day 5)."""
        results: list[TemporalExpression] = []

        for m in self._RE_POD.finditer(text):
            pod_num = int(m.group(1))
            results.append(TemporalExpression(
                text=m.group(0),
                temporal_type=TemporalType.PERIOD,
                start_char=m.start(),
                end_char=m.end(),
                confidence=0.90,
                normalised_value=f"POD{pod_num}",
                duration_days=float(pod_num),
                metadata={"postoperative_day": pod_num},
            ))

        return results

    # ------------------------------------------------------------------
    # Frequency extraction
    # ------------------------------------------------------------------

    def _extract_frequencies(self, text: str) -> list[Frequency]:
        """Extract and normalise medication/treatment frequencies."""
        results: list[Frequency] = []

        # q<N>h patterns
        for m in self._RE_FREQ_Q.finditer(text):
            hours = int(m.group(1))
            full_match = m.group(0).lower()
            is_prn = "prn" in full_match
            times = 24.0 / hours if hours > 0 else 0
            results.append(Frequency(
                text=m.group(0),
                times_per_day=times,
                interval_hours=float(hours),
                as_needed=is_prn,
            ))

        # Standard abbreviations (BID, TID, etc.)
        for m in self._RE_FREQ_ABBREV.finditer(text):
            abbrev = m.group(1).lower()
            mapping = FREQUENCY_MAP.get(abbrev)
            if mapping:
                times_per_day, interval_hours, as_needed = mapping
                results.append(Frequency(
                    text=m.group(0),
                    times_per_day=times_per_day,
                    interval_hours=interval_hours,
                    as_needed=as_needed,
                ))

        # Written frequencies ("twice daily")
        written_map = {
            ("once", "daily"): (1.0, 24.0),
            ("twice", "daily"): (2.0, 12.0),
            ("three times", "daily"): (3.0, 8.0),
            ("four times", "daily"): (4.0, 6.0),
            ("once", "weekly"): (1 / 7, 168.0),
            ("twice", "weekly"): (2 / 7, 84.0),
            ("once", "monthly"): (1 / 30, 720.0),
        }
        for m in self._RE_FREQ_WRITTEN.finditer(text):
            count = m.group(1).lower()
            period = m.group(2).lower()
            mapping = written_map.get((count, period))
            if mapping:
                results.append(Frequency(
                    text=m.group(0),
                    times_per_day=mapping[0],
                    interval_hours=mapping[1],
                ))

        # "every N hours/days"
        for m in self._RE_FREQ_EVERY.finditer(text):
            amount = int(m.group(1))
            unit = m.group(2).lower().rstrip("s")
            if unit in ("hr",):
                unit = "hour"

            if unit == "hour":
                interval_h = float(amount)
                times = 24.0 / amount if amount > 0 else 0
            elif unit == "day":
                interval_h = float(amount * 24)
                times = 1.0 / amount if amount > 0 else 0
            elif unit == "week":
                interval_h = float(amount * 168)
                times = 1.0 / (amount * 7) if amount > 0 else 0
            elif unit == "month":
                interval_h = float(amount * 720)
                times = 1.0 / (amount * 30) if amount > 0 else 0
            else:
                continue

            results.append(Frequency(
                text=m.group(0),
                times_per_day=times,
                interval_hours=interval_h,
            ))

        return results

    # ------------------------------------------------------------------
    # Temporal relation extraction
    # ------------------------------------------------------------------

    def _extract_temporal_links(
        self,
        text: str,
        expressions: list[TemporalExpression],
    ) -> list[TemporalLink]:
        """Identify temporal ordering relations between events in the text.

        Scans for temporal signal words (before, after, during, etc.) and
        connects them to nearby temporal expressions or event mentions.
        """
        links: list[TemporalLink] = []

        # Define signal → relation mapping
        signal_map: list[tuple[str, TemporalRelation]] = []
        for pat in _BEFORE_SIGNALS:
            signal_map.append((pat, TemporalRelation.BEFORE))
        for pat in _AFTER_SIGNALS:
            signal_map.append((pat, TemporalRelation.AFTER))
        for pat in _SIMULTANEOUS_SIGNALS:
            signal_map.append((pat, TemporalRelation.SIMULTANEOUS))
        for pat in _OVERLAP_SIGNALS:
            signal_map.append((pat, TemporalRelation.OVERLAP))

        # Split text into sentences for local context
        sentences = re.split(r'[.!?]+', text)
        offset = 0

        for sentence in sentences:
            sentence_stripped = sentence.strip()
            if not sentence_stripped:
                offset += len(sentence) + 1
                continue

            for pattern_str, relation in signal_map:
                for m in re.finditer(pattern_str, sentence_stripped, re.IGNORECASE):
                    # Find the clause before and after the signal word
                    before_text = sentence_stripped[:m.start()].strip()
                    after_text = sentence_stripped[m.end():].strip()

                    if before_text and after_text:
                        # Truncate to reasonable lengths
                        source = before_text[-80:].strip()
                        target = after_text[:80].strip()

                        links.append(TemporalLink(
                            source_span=source,
                            target_span=target,
                            relation=relation,
                            confidence=0.70,
                            evidence=m.group(0),
                        ))

            offset += len(sentence) + 1

        return links

    # ------------------------------------------------------------------
    # De-duplication
    # ------------------------------------------------------------------

    def _deduplicate(
        self,
        expressions: list[TemporalExpression],
    ) -> list[TemporalExpression]:
        """Remove overlapping expressions, keeping higher-confidence ones.

        When two expressions overlap in character span, the one with higher
        confidence wins.  Ties are broken by longer span (more specific).
        """
        if not expressions:
            return []

        # Sort by start position, then by confidence (descending)
        sorted_exprs = sorted(
            expressions,
            key=lambda e: (e.start_char, -e.confidence, -(e.end_char - e.start_char)),
        )

        result: list[TemporalExpression] = [sorted_exprs[0]]

        for expr in sorted_exprs[1:]:
            last = result[-1]
            # Check overlap
            if expr.start_char < last.end_char:
                # Overlapping — keep the one with higher confidence
                if expr.confidence > last.confidence:
                    result[-1] = expr
                # Otherwise keep the existing one
            else:
                result.append(expr)

        return result
