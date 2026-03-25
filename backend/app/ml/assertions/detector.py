"""Clinical assertion detection using ConText/NegEx-inspired algorithms.

This module classifies the assertion status of clinical entities extracted
from medical text, determining whether conditions, symptoms, medications,
and other medical concepts are:

- **present**: Affirmed in the clinical context
- **absent**: Negated (e.g., "no fever", "denies chest pain")
- **possible**: Uncertain (e.g., "possible pneumonia", "rule out DVT")
- **conditional**: Dependent on a condition (e.g., "if symptoms worsen")
- **hypothetical**: Future/planned (e.g., "will start metformin")
- **family**: Associated with family member (e.g., "family history of diabetes")

The algorithm uses a trigger-based approach inspired by the ConText algorithm
(Harkema et al., 2009) with configurable scope windows and sentence-boundary
termination.

Architecture:
    AssertionDetector (ABC)
    ├── RuleBasedAssertionDetector — Compiled regex triggers with configurable scope
    └── ConTextAssertionDetector — Extended ConText with forward/backward triggers

Design decisions:
    - Sentence segmentation uses regex splitting on sentence-ending punctuation
      plus common clinical abbreviations, rather than requiring an NLP library.
    - Trigger scope is bounded by both character distance and sentence boundaries,
      whichever is more restrictive.
    - Triggers have priority levels so that more specific patterns (e.g.,
      "family history of") take precedence over general ones (e.g., "history").
    - Thread-safe: all state is instance-level and patterns are compiled once at
      load time.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AssertionStatus(str, Enum):
    """Assertion status for a clinical entity.

    Values follow the HL7/i2b2 assertion taxonomy:
    - present: condition is affirmed
    - absent: condition is negated
    - possible: condition is uncertain/suspected
    - conditional: condition depends on a future event
    - hypothetical: planned/future condition or treatment
    - family: condition attributed to family member, not patient
    """

    PRESENT = "present"
    ABSENT = "absent"
    POSSIBLE = "possible"
    CONDITIONAL = "conditional"
    HYPOTHETICAL = "hypothetical"
    FAMILY = "family"


class TriggerType(str, Enum):
    """Direction and type of an assertion trigger.

    - pre: trigger occurs BEFORE the entity (e.g., "no [entity]")
    - post: trigger occurs AFTER the entity (e.g., "[entity] was ruled out")
    - pseudo: looks like a trigger but isn't (e.g., "no increase" in "no increase in pain"
              should not negate pain — this prevents false positives)
    - terminator: stops the scope of a preceding trigger (e.g., "but" between trigger and entity)
    """

    PRE = "pre"
    POST = "post"
    PSEUDO = "pseudo"
    TERMINATOR = "terminator"


@dataclass
class Trigger:
    """A single assertion trigger pattern.

    Parameters
    ----------
    pattern : str
        Regex pattern string for matching (case-insensitive).
    trigger_type : TriggerType
        Whether this is a pre-entity, post-entity, pseudo, or terminator trigger.
    assertion : AssertionStatus
        The assertion status this trigger implies (ignored for terminators/pseudo).
    priority : int
        Higher priority triggers take precedence (0 = lowest).
    max_scope : int
        Maximum character distance between trigger and entity.
    """

    pattern: str
    trigger_type: TriggerType
    assertion: AssertionStatus
    priority: int = 0
    max_scope: int = 50


@dataclass
class AssertionResult:
    """Result of assertion detection for a single entity.

    Parameters
    ----------
    status : AssertionStatus
        The determined assertion status.
    confidence : float
        Confidence in the assertion (0.0–1.0).
    trigger_text : str | None
        The text that triggered the assertion, if any.
    trigger_type : TriggerType | None
        The type of trigger that fired.
    entity_text : str
        The entity text being classified.
    entity_start : int
        Character offset of entity start.
    entity_end : int
        Character offset of entity end.
    sentence : str
        The sentence containing the entity.
    metadata : dict[str, Any]
        Additional metadata (e.g., matched pattern, priority).
    """

    status: AssertionStatus
    confidence: float
    trigger_text: str | None
    trigger_type: TriggerType | None
    entity_text: str
    entity_start: int
    entity_end: int
    sentence: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation.

        Returns
        -------
        dict[str, Any]
            Dictionary with all assertion result fields.
        """
        return {
            "status": self.status.value,
            "confidence": self.confidence,
            "trigger_text": self.trigger_text,
            "trigger_type": self.trigger_type.value if self.trigger_type else None,
            "entity_text": self.entity_text,
            "entity_start": self.entity_start,
            "entity_end": self.entity_end,
            "sentence": self.sentence,
            "metadata": self.metadata,
        }


class AssertionDetector(ABC):
    """Abstract base class for clinical assertion detection.

    Subclasses implement the ``detect`` method to classify entities
    as present, absent, possible, conditional, hypothetical, or family.
    """

    @abstractmethod
    def detect(
        self,
        text: str,
        entity_start: int,
        entity_end: int,
    ) -> AssertionResult:
        """Detect assertion status for an entity in clinical text.

        Parameters
        ----------
        text : str
            The full clinical text.
        entity_start : int
            Start character offset of the entity.
        entity_end : int
            End character offset of the entity.

        Returns
        -------
        AssertionResult
            The assertion classification result.
        """
        ...

    def detect_batch(
        self,
        text: str,
        entities: list[tuple[int, int]],
    ) -> list[AssertionResult]:
        """Detect assertions for multiple entities in the same text.

        Parameters
        ----------
        text : str
            The full clinical text.
        entities : list[tuple[int, int]]
            List of (start, end) character offsets for entities.

        Returns
        -------
        list[AssertionResult]
            Assertion results for each entity.
        """
        return [self.detect(text, start, end) for start, end in entities]


# -- Sentence segmentation utilities ---


# Sentence-ending patterns for clinical text
_SENTENCE_SPLIT_PATTERN = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])"  # Standard sentence boundary
    r"|(?<=\n)\s*(?=\S)"  # Newline-separated lines
    r"|(?<=[:;])\s*\n"  # Colon/semicolon followed by newline
)


def _segment_sentences(text: str) -> list[tuple[int, int]]:
    """Segment text into sentence spans.

    Returns a list of (start, end) character offsets for each sentence.
    Handles clinical text conventions like section headers and list items.

    Parameters
    ----------
    text : str
        Clinical text to segment.

    Returns
    -------
    list[tuple[int, int]]
        Sentence boundary offsets.
    """
    if not text:
        return []

    sentences = []
    last_end = 0

    for match in _SENTENCE_SPLIT_PATTERN.finditer(text):
        start = match.start()
        if start > last_end:
            sentences.append((last_end, start))
        last_end = match.end()

    # Final sentence
    if last_end < len(text):
        sentences.append((last_end, len(text)))

    return sentences


def _find_entity_sentence(
    text: str,
    entity_start: int,
    entity_end: int,
    sentences: list[tuple[int, int]],
) -> tuple[int, int]:
    """Find the sentence containing the entity.

    Parameters
    ----------
    text : str
        Full text.
    entity_start : int
        Entity start offset.
    entity_end : int
        Entity end offset.
    sentences : list[tuple[int, int]]
        Pre-computed sentence boundaries.

    Returns
    -------
    tuple[int, int]
        (sentence_start, sentence_end) offsets.
    """
    for sent_start, sent_end in sentences:
        if sent_start <= entity_start and entity_end <= sent_end:
            return sent_start, sent_end

    # Fallback: use a window around the entity
    window = 200
    return max(0, entity_start - window), min(len(text), entity_end + window)


# -- Trigger libraries ---


def _build_negation_triggers() -> list[Trigger]:
    """Build the negation trigger library.

    Returns
    -------
    list[Trigger]
        Negation trigger patterns.
    """
    return [
        # Pre-entity negation triggers (appear before the entity)
        Trigger(r"\bno\s+(?:evidence\s+of\s+)?", TriggerType.PRE, AssertionStatus.ABSENT, priority=3, max_scope=40),
        Trigger(r"\bnot\s+(?:any\s+)?", TriggerType.PRE, AssertionStatus.ABSENT, priority=3, max_scope=40),
        Trigger(r"\bdenies?\s+(?:any\s+)?", TriggerType.PRE, AssertionStatus.ABSENT, priority=4, max_scope=60),
        Trigger(r"\bdenying\s+", TriggerType.PRE, AssertionStatus.ABSENT, priority=4, max_scope=60),
        Trigger(r"\bwithout\s+(?:any\s+)?", TriggerType.PRE, AssertionStatus.ABSENT, priority=3, max_scope=50),
        Trigger(r"\bnegative\s+for\s+", TriggerType.PRE, AssertionStatus.ABSENT, priority=5, max_scope=50),
        Trigger(r"\babsence\s+of\s+", TriggerType.PRE, AssertionStatus.ABSENT, priority=5, max_scope=40),
        Trigger(r"\bno\s+signs?\s+of\s+", TriggerType.PRE, AssertionStatus.ABSENT, priority=5, max_scope=50),
        Trigger(r"\bno\s+complaints?\s+of\s+", TriggerType.PRE, AssertionStatus.ABSENT, priority=5, max_scope=50),
        Trigger(r"\bfree\s+of\s+", TriggerType.PRE, AssertionStatus.ABSENT, priority=4, max_scope=40),
        Trigger(r"\bresolved\s+", TriggerType.PRE, AssertionStatus.ABSENT, priority=3, max_scope=30),
        Trigger(r"\bno\s+further\s+", TriggerType.PRE, AssertionStatus.ABSENT, priority=4, max_scope=50),
        Trigger(r"\bno\s+(?:new|acute|significant)\s+", TriggerType.PRE, AssertionStatus.ABSENT, priority=4, max_scope=50),
        Trigger(r"\bno\s+longer\s+(?:has|have|experiencing)\s+", TriggerType.PRE, AssertionStatus.ABSENT, priority=5, max_scope=60),
        Trigger(r"\bfailed\s+to\s+reveal\s+", TriggerType.PRE, AssertionStatus.ABSENT, priority=5, max_scope=50),
        Trigger(r"\bnot\s+demonstrate\s+", TriggerType.PRE, AssertionStatus.ABSENT, priority=4, max_scope=50),
        Trigger(r"\bno\s+radiographic\s+evidence\s+of\s+", TriggerType.PRE, AssertionStatus.ABSENT, priority=6, max_scope=60),
        Trigger(r"\bruled\s+out\s+", TriggerType.PRE, AssertionStatus.ABSENT, priority=4, max_scope=40),
        Trigger(r"\bwas\s+not\s+", TriggerType.PRE, AssertionStatus.ABSENT, priority=3, max_scope=40),
        Trigger(r"\bare\s+not\s+", TriggerType.PRE, AssertionStatus.ABSENT, priority=3, max_scope=40),
        Trigger(r"\bdoes\s+not\s+(?:have|show|demonstrate|exhibit|indicate)\s+", TriggerType.PRE, AssertionStatus.ABSENT, priority=5, max_scope=60),
        Trigger(r"\bno\s+history\s+of\s+", TriggerType.PRE, AssertionStatus.ABSENT, priority=5, max_scope=50),
        Trigger(r"\bnever\s+(?:had|has|have)\s+", TriggerType.PRE, AssertionStatus.ABSENT, priority=5, max_scope=50),
        Trigger(r"\bunremarkable\s+(?:for\s+)?", TriggerType.PRE, AssertionStatus.ABSENT, priority=4, max_scope=40),
        # Post-entity negation triggers (appear after the entity)
        Trigger(r"\s+(?:was|were|is|are)\s+(?:not\s+)?(?:absent|negative|ruled\s+out)", TriggerType.POST, AssertionStatus.ABSENT, priority=4, max_scope=50),
        Trigger(r"\s+(?:has|have)\s+been\s+ruled\s+out", TriggerType.POST, AssertionStatus.ABSENT, priority=5, max_scope=50),
        Trigger(r"\s+(?:has|have)\s+resolved", TriggerType.POST, AssertionStatus.ABSENT, priority=4, max_scope=40),
        Trigger(r"\s+(?:was|were)\s+not\s+(?:seen|found|detected|identified|observed)", TriggerType.POST, AssertionStatus.ABSENT, priority=5, max_scope=60),
        Trigger(r"\s+(?:is|are)\s+unlikely", TriggerType.POST, AssertionStatus.ABSENT, priority=3, max_scope=40),
    ]


def _build_uncertainty_triggers() -> list[Trigger]:
    """Build the uncertainty trigger library.

    Returns
    -------
    list[Trigger]
        Uncertainty trigger patterns.
    """
    return [
        # Pre-entity uncertainty triggers
        Trigger(r"\bpossible\s+", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=3, max_scope=40),
        Trigger(r"\bprobable\s+", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=3, max_scope=40),
        Trigger(r"\blikely\s+", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=2, max_scope=40),
        Trigger(r"\bsuspected\s+", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=4, max_scope=40),
        Trigger(r"\bsuspicious\s+(?:for\s+)?", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=4, max_scope=50),
        Trigger(r"\bmay\s+(?:have\s+|be\s+)?", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=2, max_scope=40),
        Trigger(r"\bmight\s+(?:have\s+|be\s+)?", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=2, max_scope=40),
        Trigger(r"\bconcern\s+(?:for|about)\s+", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=4, max_scope=50),
        Trigger(r"\brule\s+out\s+", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=5, max_scope=40),
        Trigger(r"\br/o\s+", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=5, max_scope=30),
        Trigger(r"\bquestionable\s+", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=3, max_scope=40),
        Trigger(r"\buncertain\s+(?:for\s+|about\s+)?", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=4, max_scope=50),
        Trigger(r"\bequivocal\s+(?:for\s+)?", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=4, max_scope=40),
        Trigger(r"\bindeterminate\s+", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=3, max_scope=40),
        Trigger(r"\bcannot\s+(?:be\s+)?(?:excluded?|ruled\s+out)\s*", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=5, max_scope=60),
        Trigger(r"\bdifferential\s+(?:diagnosis\s+)?(?:includes?\s+)?", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=4, max_scope=60),
        Trigger(r"\b(?:would\s+)?suggest(?:s|ive)?\s+(?:of\s+)?", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=2, max_scope=50),
        Trigger(r"\bconsistent\s+with\s+", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=2, max_scope=40),
        Trigger(r"\bclinically\s+consistent\s+with\s+", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=3, max_scope=50),
        Trigger(r"\bappears?\s+(?:to\s+be\s+)?", TriggerType.PRE, AssertionStatus.POSSIBLE, priority=2, max_scope=40),
        # Post-entity uncertainty triggers
        Trigger(r"\s+(?:is|are)\s+(?:suspected|questionable|equivocal)", TriggerType.POST, AssertionStatus.POSSIBLE, priority=3, max_scope=50),
        Trigger(r"\s+cannot\s+be\s+(?:excluded|ruled\s+out)", TriggerType.POST, AssertionStatus.POSSIBLE, priority=5, max_scope=50),
        Trigger(r"\s*\?\s*$", TriggerType.POST, AssertionStatus.POSSIBLE, priority=2, max_scope=10),
    ]


def _build_family_triggers() -> list[Trigger]:
    """Build the family history trigger library.

    Returns
    -------
    list[Trigger]
        Family history trigger patterns.
    """
    return [
        Trigger(r"\bfamily\s+history\s+(?:of\s+)?(?:significant\s+for\s+)?", TriggerType.PRE, AssertionStatus.FAMILY, priority=6, max_scope=60),
        Trigger(r"\bfamilial\s+", TriggerType.PRE, AssertionStatus.FAMILY, priority=5, max_scope=40),
        Trigger(r"\b(?:mother|father|sister|brother|parent|sibling|grandmother|grandfather|aunt|uncle|cousin)\s+(?:has|had|with|diagnosed\s+with)\s+", TriggerType.PRE, AssertionStatus.FAMILY, priority=6, max_scope=80),
        Trigger(r"\b(?:maternal|paternal)\s+(?:history\s+of\s+|grandmother|grandfather|aunt|uncle)\s*", TriggerType.PRE, AssertionStatus.FAMILY, priority=6, max_scope=80),
        Trigger(r"\bFH:?\s+", TriggerType.PRE, AssertionStatus.FAMILY, priority=5, max_scope=40),
        Trigger(r"\bfamily\s+hx\s+(?:of\s+)?", TriggerType.PRE, AssertionStatus.FAMILY, priority=6, max_scope=50),
        Trigger(r"\binherited\s+", TriggerType.PRE, AssertionStatus.FAMILY, priority=4, max_scope=40),
        Trigger(r"\bhereditary\s+", TriggerType.PRE, AssertionStatus.FAMILY, priority=5, max_scope=40),
        Trigger(r"\bruns?\s+in\s+(?:the\s+)?family\b", TriggerType.POST, AssertionStatus.FAMILY, priority=5, max_scope=40),
    ]


def _build_hypothetical_triggers() -> list[Trigger]:
    """Build the hypothetical/future trigger library.

    Returns
    -------
    list[Trigger]
        Hypothetical trigger patterns.
    """
    return [
        Trigger(r"\bwill\s+(?:start|begin|initiate|need|require|undergo)\s+", TriggerType.PRE, AssertionStatus.HYPOTHETICAL, priority=4, max_scope=50),
        Trigger(r"\bplan\s+(?:to|for)\s+", TriggerType.PRE, AssertionStatus.HYPOTHETICAL, priority=4, max_scope=40),
        Trigger(r"\bplanned\s+", TriggerType.PRE, AssertionStatus.HYPOTHETICAL, priority=4, max_scope=30),
        Trigger(r"\bscheduled\s+for\s+", TriggerType.PRE, AssertionStatus.HYPOTHETICAL, priority=4, max_scope=40),
        Trigger(r"\bto\s+be\s+(?:started|initiated|given)\s+", TriggerType.PRE, AssertionStatus.HYPOTHETICAL, priority=4, max_scope=50),
        Trigger(r"\bshould\s+(?:be\s+)?(?:started|given|considered)\s*", TriggerType.PRE, AssertionStatus.HYPOTHETICAL, priority=3, max_scope=50),
        Trigger(r"\bconsider\s+(?:starting\s+)?", TriggerType.PRE, AssertionStatus.HYPOTHETICAL, priority=3, max_scope=40),
        Trigger(r"\bif\s+(?:the\s+)?(?:patient\s+)?(?:develops?|experiences?|has|shows?)\s+", TriggerType.PRE, AssertionStatus.HYPOTHETICAL, priority=4, max_scope=70),
        Trigger(r"\bwould\s+recommend\s+", TriggerType.PRE, AssertionStatus.HYPOTHETICAL, priority=3, max_scope=40),
        Trigger(r"\bpending\s+", TriggerType.PRE, AssertionStatus.HYPOTHETICAL, priority=3, max_scope=30),
        Trigger(r"\bawaiting\s+", TriggerType.PRE, AssertionStatus.HYPOTHETICAL, priority=3, max_scope=30),
        Trigger(r"\bfollow\s*-?\s*up\s+(?:with\s+)?", TriggerType.PRE, AssertionStatus.HYPOTHETICAL, priority=3, max_scope=40),
    ]


def _build_conditional_triggers() -> list[Trigger]:
    """Build the conditional trigger library.

    Returns
    -------
    list[Trigger]
        Conditional trigger patterns.
    """
    return [
        Trigger(r"\bif\s+(?:symptoms?\s+)?(?:worsens?|persists?|recurs?|returns?|continues?)\s*,?\s*", TriggerType.PRE, AssertionStatus.CONDITIONAL, priority=5, max_scope=70),
        Trigger(r"\bshould\s+(?:the\s+)?(?:symptoms?\s+)?(?:worsen|persist|recur|return|continue)\s*,?\s*", TriggerType.PRE, AssertionStatus.CONDITIONAL, priority=5, max_scope=80),
        Trigger(r"\bin\s+(?:the\s+)?(?:event|case)\s+(?:of|that)\s+", TriggerType.PRE, AssertionStatus.CONDITIONAL, priority=4, max_scope=50),
        Trigger(r"\bunless\s+", TriggerType.PRE, AssertionStatus.CONDITIONAL, priority=3, max_scope=40),
        Trigger(r"\bprovided\s+(?:that\s+)?", TriggerType.PRE, AssertionStatus.CONDITIONAL, priority=3, max_scope=40),
        Trigger(r"\bas\s+needed\s+for\s+", TriggerType.PRE, AssertionStatus.CONDITIONAL, priority=3, max_scope=40),
    ]


def _build_pseudo_triggers() -> list[Trigger]:
    """Build pseudo-trigger patterns that prevent false-positive negations.

    These patterns look like negation triggers but should NOT trigger assertion
    changes. For example, "no increase in pain" should not negate "pain" — the
    patient still has pain.

    Returns
    -------
    list[Trigger]
        Pseudo-trigger patterns.
    """
    return [
        Trigger(r"\bno\s+(?:increase|decrease|change|improvement|worsening)\s+(?:in|of)\s+", TriggerType.PSEUDO, AssertionStatus.PRESENT, priority=10, max_scope=60),
        Trigger(r"\bnot\s+(?:cause|causing|caused)\s+", TriggerType.PSEUDO, AssertionStatus.PRESENT, priority=10, max_scope=50),
        Trigger(r"\bnot\s+(?:only|just|merely)\s+", TriggerType.PSEUDO, AssertionStatus.PRESENT, priority=10, max_scope=40),
        Trigger(r"\bno\s+(?:reason\s+to\s+)?(?:doubt|question)\s+", TriggerType.PSEUDO, AssertionStatus.PRESENT, priority=10, max_scope=50),
        Trigger(r"\bnot\s+necessarily\s+", TriggerType.PSEUDO, AssertionStatus.PRESENT, priority=10, max_scope=40),
        Trigger(r"\bgram\s+negative\b", TriggerType.PSEUDO, AssertionStatus.PRESENT, priority=10, max_scope=30),
        Trigger(r"\bnot\s+(?:certain|sure)\s+(?:if|whether)\s+", TriggerType.PSEUDO, AssertionStatus.PRESENT, priority=10, max_scope=50),
    ]


def _build_terminator_triggers() -> list[Trigger]:
    """Build scope terminator triggers.

    Terminators stop the scope of a preceding pre-entity trigger. For example,
    in "no fever, but patient reports chest pain", the terminator "but" prevents
    "no" from negating "chest pain".

    Returns
    -------
    list[Trigger]
        Terminator trigger patterns.
    """
    return [
        Trigger(r"\bbut\b", TriggerType.TERMINATOR, AssertionStatus.PRESENT, priority=5, max_scope=0),
        Trigger(r"\bhowever\b", TriggerType.TERMINATOR, AssertionStatus.PRESENT, priority=5, max_scope=0),
        Trigger(r"\byet\b", TriggerType.TERMINATOR, AssertionStatus.PRESENT, priority=4, max_scope=0),
        Trigger(r"\bthough\b", TriggerType.TERMINATOR, AssertionStatus.PRESENT, priority=4, max_scope=0),
        Trigger(r"\balthough\b", TriggerType.TERMINATOR, AssertionStatus.PRESENT, priority=4, max_scope=0),
        Trigger(r"\baside\s+from\b", TriggerType.TERMINATOR, AssertionStatus.PRESENT, priority=5, max_scope=0),
        Trigger(r"\bexcept\s+(?:for\s+)?", TriggerType.TERMINATOR, AssertionStatus.PRESENT, priority=5, max_scope=0),
        Trigger(r"\bother\s+than\b", TriggerType.TERMINATOR, AssertionStatus.PRESENT, priority=5, max_scope=0),
        Trigger(r"\bwhich\b", TriggerType.TERMINATOR, AssertionStatus.PRESENT, priority=3, max_scope=0),
        Trigger(r"\bthat\s+(?:is|was|are|were)\b", TriggerType.TERMINATOR, AssertionStatus.PRESENT, priority=3, max_scope=0),
        Trigger(r"[;:]", TriggerType.TERMINATOR, AssertionStatus.PRESENT, priority=6, max_scope=0),
    ]


class RuleBasedAssertionDetector(AssertionDetector):
    """Rule-based assertion detector using compiled regex triggers.

    Uses a library of clinical language triggers to determine assertion
    status. Supports pre-entity and post-entity triggers, pseudo-triggers
    (to prevent false positives), and scope terminators.

    Parameters
    ----------
    custom_triggers : list[Trigger] | None
        Additional custom triggers to include.
    default_status : AssertionStatus
        Default status when no trigger matches (default: PRESENT).
    default_confidence : float
        Confidence for default (no-trigger) assertions.
    trigger_confidence : float
        Base confidence when a trigger matches.

    Examples
    --------
    >>> detector = RuleBasedAssertionDetector()
    >>> result = detector.detect("Patient denies chest pain.", 22, 32)
    >>> result.status
    <AssertionStatus.ABSENT: 'absent'>
    >>> result.trigger_text
    'denies '
    """

    def __init__(
        self,
        custom_triggers: list[Trigger] | None = None,
        default_status: AssertionStatus = AssertionStatus.PRESENT,
        default_confidence: float = 0.80,
        trigger_confidence: float = 0.90,
    ):
        self.default_status = default_status
        self.default_confidence = default_confidence
        self.trigger_confidence = trigger_confidence

        # Build trigger library
        self._triggers: list[Trigger] = []
        self._triggers.extend(_build_negation_triggers())
        self._triggers.extend(_build_uncertainty_triggers())
        self._triggers.extend(_build_family_triggers())
        self._triggers.extend(_build_hypothetical_triggers())
        self._triggers.extend(_build_conditional_triggers())
        self._triggers.extend(_build_pseudo_triggers())
        self._triggers.extend(_build_terminator_triggers())

        if custom_triggers:
            self._triggers.extend(custom_triggers)

        # Compile all patterns
        self._compiled: list[tuple[Trigger, re.Pattern]] = [
            (trigger, re.compile(trigger.pattern, re.IGNORECASE))
            for trigger in self._triggers
        ]

        # Separate by type for efficient lookup
        self._pre_triggers = [
            (t, p) for t, p in self._compiled if t.trigger_type == TriggerType.PRE
        ]
        self._post_triggers = [
            (t, p) for t, p in self._compiled if t.trigger_type == TriggerType.POST
        ]
        self._pseudo_triggers = [
            (t, p) for t, p in self._compiled if t.trigger_type == TriggerType.PSEUDO
        ]
        self._terminators = [
            (t, p) for t, p in self._compiled if t.trigger_type == TriggerType.TERMINATOR
        ]

        logger.info(
            "Initialized RuleBasedAssertionDetector with %d triggers "
            "(%d pre, %d post, %d pseudo, %d terminators)",
            len(self._triggers),
            len(self._pre_triggers),
            len(self._post_triggers),
            len(self._pseudo_triggers),
            len(self._terminators),
        )

    @property
    def trigger_count(self) -> int:
        """Total number of configured triggers.

        Returns
        -------
        int
            Trigger count.
        """
        return len(self._triggers)

    def detect(
        self,
        text: str,
        entity_start: int,
        entity_end: int,
    ) -> AssertionResult:
        """Detect assertion status for an entity span.

        Algorithm:
        1. Find the sentence containing the entity.
        2. Extract pre-entity and post-entity context within the sentence.
        3. Check for pseudo-triggers first (these block negation).
        4. Check pre-entity context for triggers, respecting scope terminators.
        5. Check post-entity context for triggers.
        6. Return highest-priority match, or default (PRESENT).

        Parameters
        ----------
        text : str
            Full clinical text.
        entity_start : int
            Entity start character offset.
        entity_end : int
            Entity end character offset.

        Returns
        -------
        AssertionResult
            Assertion classification.
        """
        entity_text = text[entity_start:entity_end]
        sentences = _segment_sentences(text)
        sent_start, sent_end = _find_entity_sentence(
            text, entity_start, entity_end, sentences
        )
        sentence = text[sent_start:sent_end].strip()

        # Extract context windows
        pre_context = text[sent_start:entity_start]
        post_context = text[entity_end:sent_end]

        # Step 1: Check for pseudo-triggers (these prevent false positives)
        for trigger, pattern in self._pseudo_triggers:
            match = pattern.search(pre_context)
            if match:
                distance = len(pre_context) - match.end()
                if distance <= trigger.max_scope:
                    logger.debug(
                        "Pseudo-trigger '%s' matched for entity '%s' — "
                        "blocking assertion change",
                        match.group(),
                        entity_text,
                    )
                    return AssertionResult(
                        status=AssertionStatus.PRESENT,
                        confidence=self.trigger_confidence,
                        trigger_text=match.group(),
                        trigger_type=TriggerType.PSEUDO,
                        entity_text=entity_text,
                        entity_start=entity_start,
                        entity_end=entity_end,
                        sentence=sentence,
                        metadata={"pseudo_trigger": True},
                    )

        # Step 2: Check pre-entity triggers
        best_pre = self._find_best_pre_trigger(pre_context, entity_text)

        # Step 3: Check post-entity triggers
        best_post = self._find_best_post_trigger(post_context, entity_text)

        # Step 4: Choose highest priority match
        best_match = None
        if best_pre and best_post:
            # Prefer higher priority; on tie, prefer pre-entity
            if best_pre[0].priority >= best_post[0].priority:
                best_match = best_pre
            else:
                best_match = best_post
        elif best_pre:
            best_match = best_pre
        elif best_post:
            best_match = best_post

        if best_match:
            trigger, match = best_match
            # Calculate confidence based on priority and distance
            confidence = self._calculate_confidence(trigger, match, pre_context, post_context)
            return AssertionResult(
                status=trigger.assertion,
                confidence=confidence,
                trigger_text=match.group().strip(),
                trigger_type=trigger.trigger_type,
                entity_text=entity_text,
                entity_start=entity_start,
                entity_end=entity_end,
                sentence=sentence,
                metadata={
                    "pattern": trigger.pattern,
                    "priority": trigger.priority,
                    "max_scope": trigger.max_scope,
                },
            )

        # No trigger matched — entity is present
        return AssertionResult(
            status=self.default_status,
            confidence=self.default_confidence,
            trigger_text=None,
            trigger_type=None,
            entity_text=entity_text,
            entity_start=entity_start,
            entity_end=entity_end,
            sentence=sentence,
        )

    def _find_best_pre_trigger(
        self,
        pre_context: str,
        entity_text: str,
    ) -> tuple[Trigger, re.Match] | None:
        """Find the best-matching pre-entity trigger.

        Checks for scope terminators between trigger and entity.

        Parameters
        ----------
        pre_context : str
            Text before the entity (within same sentence).
        entity_text : str
            The entity text (for logging).

        Returns
        -------
        tuple[Trigger, re.Match] | None
            Best (trigger, match) pair, or None.
        """
        candidates: list[tuple[Trigger, re.Match, int]] = []

        for trigger, pattern in self._pre_triggers:
            # Search for rightmost match (closest to entity)
            match = None
            for m in pattern.finditer(pre_context):
                match = m  # Keep updating to get rightmost

            if match is None:
                continue

            # Check distance (from end of trigger match to entity)
            distance = len(pre_context) - match.end()
            if distance > trigger.max_scope:
                continue

            # Check for terminators between trigger and entity
            between = pre_context[match.end():]
            terminated = False
            for term_trigger, term_pattern in self._terminators:
                if term_pattern.search(between):
                    terminated = True
                    break

            if terminated:
                logger.debug(
                    "Trigger '%s' terminated before entity '%s'",
                    match.group(),
                    entity_text,
                )
                continue

            candidates.append((trigger, match, distance))

        if not candidates:
            return None

        # Sort by priority (descending), then by distance (ascending = closer)
        candidates.sort(key=lambda x: (-x[0].priority, x[2]))
        best = candidates[0]
        return best[0], best[1]

    def _find_best_post_trigger(
        self,
        post_context: str,
        entity_text: str,
    ) -> tuple[Trigger, re.Match] | None:
        """Find the best-matching post-entity trigger.

        Parameters
        ----------
        post_context : str
            Text after the entity (within same sentence).
        entity_text : str
            The entity text (for logging).

        Returns
        -------
        tuple[Trigger, re.Match] | None
            Best (trigger, match) pair, or None.
        """
        candidates: list[tuple[Trigger, re.Match, int]] = []

        for trigger, pattern in self._post_triggers:
            match = pattern.search(post_context)
            if match is None:
                continue

            distance = match.start()
            if distance > trigger.max_scope:
                continue

            candidates.append((trigger, match, distance))

        if not candidates:
            return None

        candidates.sort(key=lambda x: (-x[0].priority, x[2]))
        best = candidates[0]
        return best[0], best[1]

    def _calculate_confidence(
        self,
        trigger: Trigger,
        match: re.Match,
        pre_context: str,
        post_context: str,
    ) -> float:
        """Calculate assertion confidence based on trigger properties.

        Factors:
        - Base confidence (self.trigger_confidence)
        - Priority bonus (+0.02 per priority level)
        - Distance penalty (-0.01 per 10 chars of distance)

        Parameters
        ----------
        trigger : Trigger
            The matched trigger.
        match : re.Match
            The regex match object.
        pre_context : str
            Pre-entity context.
        post_context : str
            Post-entity context.

        Returns
        -------
        float
            Confidence value clamped to [0.5, 1.0].
        """
        confidence = self.trigger_confidence

        # Priority bonus
        confidence += trigger.priority * 0.02

        # Distance penalty
        if trigger.trigger_type == TriggerType.PRE:
            distance = len(pre_context) - match.end()
        else:
            distance = match.start()

        confidence -= (distance / 10) * 0.01

        return max(0.50, min(1.0, confidence))


class ConTextAssertionDetector(RuleBasedAssertionDetector):
    """Extended ConText assertion detector with section-aware scope.

    Extends RuleBasedAssertionDetector with:
    - Section header detection (e.g., "Family History:" section implies
      family assertion for all entities within that section)
    - Configurable scope windows per assertion category
    - Aggregate statistics for batch processing

    Parameters
    ----------
    section_headers : dict[AssertionStatus, list[str]] | None
        Map of assertion statuses to section header patterns.
    custom_triggers : list[Trigger] | None
        Additional custom triggers.

    Examples
    --------
    >>> detector = ConTextAssertionDetector()
    >>> result = detector.detect(
    ...     "Family History:\\nMother had breast cancer. Father had diabetes.",
    ...     42, 55
    ... )
    >>> result.status
    <AssertionStatus.FAMILY: 'family'>
    """

    # Default section header patterns
    DEFAULT_SECTION_HEADERS: dict[AssertionStatus, list[str]] = {
        AssertionStatus.FAMILY: [
            r"(?:^|\n)\s*(?:family\s+history|FH|family\s+hx)\s*[:\-]",
        ],
        AssertionStatus.ABSENT: [
            r"(?:^|\n)\s*(?:pertinent\s+negatives?|negatives?)\s*[:\-]",
        ],
        AssertionStatus.HYPOTHETICAL: [
            r"(?:^|\n)\s*(?:plan|assessment\s+(?:and|&)\s+plan|A/?P|recommendations?|discharge\s+plan)\s*[:\-]",
        ],
    }

    def __init__(
        self,
        section_headers: dict[AssertionStatus, list[str]] | None = None,
        custom_triggers: list[Trigger] | None = None,
        **kwargs: Any,
    ):
        super().__init__(custom_triggers=custom_triggers, **kwargs)

        self._section_headers = section_headers or self.DEFAULT_SECTION_HEADERS
        self._compiled_sections: dict[AssertionStatus, list[re.Pattern]] = {}

        for status, patterns in self._section_headers.items():
            self._compiled_sections[status] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        self._stats: dict[str, int] = {
            "total_detections": 0,
            "present": 0,
            "absent": 0,
            "possible": 0,
            "conditional": 0,
            "hypothetical": 0,
            "family": 0,
            "section_overrides": 0,
        }

    @property
    def stats(self) -> dict[str, int]:
        """Detection statistics.

        Returns
        -------
        dict[str, int]
            Counts of each assertion type detected.
        """
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset detection statistics to zero."""
        for key in self._stats:
            self._stats[key] = 0

    def detect(
        self,
        text: str,
        entity_start: int,
        entity_end: int,
    ) -> AssertionResult:
        """Detect assertion with section-aware scope.

        First checks if the entity falls within a section whose header
        implies a specific assertion (e.g., "Family History:" section).
        If so, applies the section-level assertion. Otherwise, delegates
        to the parent rule-based detection.

        Parameters
        ----------
        text : str
            Full clinical text.
        entity_start : int
            Entity start offset.
        entity_end : int
            Entity end offset.

        Returns
        -------
        AssertionResult
            Assertion result with section awareness.
        """
        # Check section headers first
        section_assertion = self._check_section_context(text, entity_start)

        # Get rule-based result
        result = super().detect(text, entity_start, entity_end)

        # Section header overrides default (PRESENT) but not explicit triggers
        if section_assertion and result.status == AssertionStatus.PRESENT:
            result = AssertionResult(
                status=section_assertion,
                confidence=0.85,
                trigger_text=None,
                trigger_type=None,
                entity_text=result.entity_text,
                entity_start=entity_start,
                entity_end=entity_end,
                sentence=result.sentence,
                metadata={"section_override": True, "section_assertion": section_assertion.value},
            )
            self._stats["section_overrides"] += 1

        # Update stats
        self._stats["total_detections"] += 1
        self._stats[result.status.value] += 1

        return result

    def _check_section_context(
        self,
        text: str,
        entity_start: int,
    ) -> AssertionStatus | None:
        """Check if entity falls within a section with an implied assertion.

        Searches backwards from the entity for section headers. The nearest
        section header determines the section context.

        Parameters
        ----------
        text : str
            Full clinical text.
        entity_start : int
            Entity start offset.

        Returns
        -------
        AssertionStatus | None
            Section-implied assertion status, or None.
        """
        # Look at text before entity for section headers
        pre_text = text[:entity_start]

        best_match: tuple[int, AssertionStatus] | None = None

        for status, patterns in self._compiled_sections.items():
            for pattern in patterns:
                for match in pattern.finditer(pre_text):
                    pos = match.end()
                    if best_match is None or pos > best_match[0]:
                        best_match = (pos, status)

        if best_match is None:
            return None

        # Check that no other section header appears between the match and entity
        # (i.e., we're still in that section)
        between = text[best_match[0]:entity_start]

        # Generic section header pattern (any "Word:" at start of line)
        generic_header = re.compile(r"(?:^|\n)\s*[A-Z][A-Za-z\s]+:\s*(?:\n|$)")
        if generic_header.search(between):
            return None  # A new section started, so section context doesn't apply

        return best_match[1]
