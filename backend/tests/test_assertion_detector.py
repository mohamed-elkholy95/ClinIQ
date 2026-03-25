"""Tests for the clinical assertion detection module.

Tests cover:
- AssertionStatus enum completeness and string serialization
- AssertionResult dataclass serialization
- Trigger dataclass construction
- Sentence segmentation utility
- RuleBasedAssertionDetector:
  - Negation detection (pre-entity and post-entity triggers)
  - Uncertainty/possibility detection
  - Family history attribution
  - Hypothetical/future assertions
  - Conditional assertions
  - Pseudo-trigger blocking (prevents false negation)
  - Scope terminator handling (but/however/except)
  - Distance-based scope limiting
  - Confidence calculation
  - Batch detection
  - Default (present) classification
  - Custom triggers
- ConTextAssertionDetector:
  - Section-header-based assertion override
  - Statistics tracking
  - Stats reset
  - Section boundary detection (new section cancels previous)
  - No section override when explicit trigger found
"""

import pytest

from app.ml.assertions.detector import (
    AssertionResult,
    AssertionStatus,
    ConTextAssertionDetector,
    RuleBasedAssertionDetector,
    Trigger,
    TriggerType,
    _find_entity_sentence,
    _segment_sentences,
)

# ---------------------------------------------------------------------------
# Enum and dataclass tests
# ---------------------------------------------------------------------------


class TestAssertionStatus:
    """Tests for the AssertionStatus enum."""

    def test_all_statuses_present(self) -> None:
        """Verify all expected statuses exist."""
        expected = {"present", "absent", "possible", "conditional", "hypothetical", "family"}
        actual = {s.value for s in AssertionStatus}
        assert actual == expected

    def test_string_enum_values(self) -> None:
        """Status values serialize as strings."""
        assert AssertionStatus.PRESENT == "present"
        assert AssertionStatus.ABSENT == "absent"
        assert AssertionStatus.POSSIBLE == "possible"

    def test_enum_count(self) -> None:
        """Exactly 6 assertion statuses."""
        assert len(AssertionStatus) == 6


class TestTriggerType:
    """Tests for the TriggerType enum."""

    def test_all_types_present(self) -> None:
        """Verify all trigger types."""
        expected = {"pre", "post", "pseudo", "terminator"}
        actual = {t.value for t in TriggerType}
        assert actual == expected


class TestTrigger:
    """Tests for the Trigger dataclass."""

    def test_default_values(self) -> None:
        """Trigger has sensible defaults."""
        t = Trigger(
            pattern=r"\bno\s+",
            trigger_type=TriggerType.PRE,
            assertion=AssertionStatus.ABSENT,
        )
        assert t.priority == 0
        assert t.max_scope == 50

    def test_custom_values(self) -> None:
        """Trigger respects custom values."""
        t = Trigger(
            pattern=r"\bdenies\s+",
            trigger_type=TriggerType.PRE,
            assertion=AssertionStatus.ABSENT,
            priority=5,
            max_scope=80,
        )
        assert t.priority == 5
        assert t.max_scope == 80


class TestAssertionResult:
    """Tests for the AssertionResult dataclass."""

    def test_to_dict(self) -> None:
        """Result serializes to dictionary."""
        result = AssertionResult(
            status=AssertionStatus.ABSENT,
            confidence=0.95,
            trigger_text="denies",
            trigger_type=TriggerType.PRE,
            entity_text="chest pain",
            entity_start=20,
            entity_end=30,
            sentence="Patient denies chest pain.",
        )
        d = result.to_dict()
        assert d["status"] == "absent"
        assert d["confidence"] == 0.95
        assert d["trigger_text"] == "denies"
        assert d["trigger_type"] == "pre"
        assert d["entity_text"] == "chest pain"
        assert d["entity_start"] == 20
        assert d["entity_end"] == 30

    def test_to_dict_no_trigger(self) -> None:
        """Result with no trigger serializes correctly."""
        result = AssertionResult(
            status=AssertionStatus.PRESENT,
            confidence=0.80,
            trigger_text=None,
            trigger_type=None,
            entity_text="fever",
            entity_start=10,
            entity_end=15,
            sentence="Patient has fever.",
        )
        d = result.to_dict()
        assert d["trigger_text"] is None
        assert d["trigger_type"] is None

    def test_metadata_default(self) -> None:
        """Metadata defaults to empty dict."""
        result = AssertionResult(
            status=AssertionStatus.PRESENT,
            confidence=0.80,
            trigger_text=None,
            trigger_type=None,
            entity_text="fever",
            entity_start=0,
            entity_end=5,
            sentence="fever",
        )
        assert result.metadata == {}


# ---------------------------------------------------------------------------
# Sentence segmentation tests
# ---------------------------------------------------------------------------


class TestSentenceSegmentation:
    """Tests for sentence segmentation utility."""

    def test_empty_text(self) -> None:
        """Empty text returns no sentences."""
        assert _segment_sentences("") == []

    def test_single_sentence(self) -> None:
        """Single sentence returns one span."""
        sentences = _segment_sentences("Patient has fever.")
        assert len(sentences) == 1
        assert sentences[0] == (0, 18)

    def test_two_sentences(self) -> None:
        """Two sentences are split correctly."""
        text = "Patient has fever. Temperature is 101F."
        sentences = _segment_sentences(text)
        assert len(sentences) == 2

    def test_newline_separated(self) -> None:
        """Newline-separated lines are treated as sentences."""
        text = "No fever\nNo cough\nNo headache"
        sentences = _segment_sentences(text)
        assert len(sentences) >= 2

    def test_find_entity_sentence(self) -> None:
        """Entity is located in the correct sentence."""
        text = "Patient has fever. No chest pain. Vitals stable."
        sentences = _segment_sentences(text)
        # "chest pain" starts at ~22
        start = text.index("chest pain")
        end = start + len("chest pain")
        sent_start, sent_end = _find_entity_sentence(text, start, end, sentences)
        sentence = text[sent_start:sent_end].strip()
        assert "chest pain" in sentence

    def test_find_entity_fallback(self) -> None:
        """Fallback when entity doesn't fall in any sentence span."""
        # With no proper sentences, fallback window is used
        sent_start, sent_end = _find_entity_sentence("abc", 0, 3, [])
        assert sent_start == 0
        assert sent_end == 3


# ---------------------------------------------------------------------------
# RuleBasedAssertionDetector tests
# ---------------------------------------------------------------------------


class TestRuleBasedNegation:
    """Tests for negation detection."""

    @pytest.fixture()
    def detector(self) -> RuleBasedAssertionDetector:
        return RuleBasedAssertionDetector()

    def test_no_fever(self, detector: RuleBasedAssertionDetector) -> None:
        """'No fever' should be negated."""
        text = "Patient has no fever."
        start = text.index("fever")
        result = detector.detect(text, start, start + 5)
        assert result.status == AssertionStatus.ABSENT

    def test_denies_chest_pain(self, detector: RuleBasedAssertionDetector) -> None:
        """'Denies chest pain' should be negated."""
        text = "Patient denies chest pain and shortness of breath."
        start = text.index("chest pain")
        result = detector.detect(text, start, start + len("chest pain"))
        assert result.status == AssertionStatus.ABSENT

    def test_without_symptoms(self, detector: RuleBasedAssertionDetector) -> None:
        """'Without headache' should be negated."""
        text = "Patient presents without headache or dizziness."
        start = text.index("headache")
        result = detector.detect(text, start, start + len("headache"))
        assert result.status == AssertionStatus.ABSENT

    def test_negative_for(self, detector: RuleBasedAssertionDetector) -> None:
        """'Negative for' should negate."""
        text = "Blood cultures negative for bacteremia."
        start = text.index("bacteremia")
        result = detector.detect(text, start, start + len("bacteremia"))
        assert result.status == AssertionStatus.ABSENT

    def test_absence_of(self, detector: RuleBasedAssertionDetector) -> None:
        """'Absence of' should negate."""
        text = "Absence of lymphadenopathy."
        start = text.index("lymphadenopathy")
        result = detector.detect(text, start, start + len("lymphadenopathy"))
        assert result.status == AssertionStatus.ABSENT

    def test_no_signs_of(self, detector: RuleBasedAssertionDetector) -> None:
        """'No signs of' should negate."""
        text = "No signs of infection."
        start = text.index("infection")
        result = detector.detect(text, start, start + len("infection"))
        assert result.status == AssertionStatus.ABSENT

    def test_no_evidence_of(self, detector: RuleBasedAssertionDetector) -> None:
        """'No evidence of' should negate."""
        text = "No evidence of malignancy."
        start = text.index("malignancy")
        result = detector.detect(text, start, start + len("malignancy"))
        assert result.status == AssertionStatus.ABSENT

    def test_free_of(self, detector: RuleBasedAssertionDetector) -> None:
        """'Free of' should negate."""
        text = "Lungs are free of infiltrate."
        start = text.index("infiltrate")
        result = detector.detect(text, start, start + len("infiltrate"))
        assert result.status == AssertionStatus.ABSENT

    def test_post_negation_ruled_out(self, detector: RuleBasedAssertionDetector) -> None:
        """Post-entity 'ruled out' should negate."""
        text = "Pulmonary embolism has been ruled out."
        start = text.index("Pulmonary embolism")
        result = detector.detect(text, start, start + len("Pulmonary embolism"))
        assert result.status == AssertionStatus.ABSENT

    def test_post_negation_resolved(self, detector: RuleBasedAssertionDetector) -> None:
        """Post-entity 'has resolved' should negate."""
        text = "Pneumonia has resolved completely."
        start = text.index("Pneumonia")
        result = detector.detect(text, start, start + len("Pneumonia"))
        assert result.status == AssertionStatus.ABSENT

    def test_not_demonstrate(self, detector: RuleBasedAssertionDetector) -> None:
        """'Not demonstrate' should negate."""
        text = "CT scan did not demonstrate pulmonary embolism."
        start = text.index("pulmonary embolism")
        result = detector.detect(text, start, start + len("pulmonary embolism"))
        assert result.status == AssertionStatus.ABSENT

    def test_no_history_of(self, detector: RuleBasedAssertionDetector) -> None:
        """'No history of' should negate."""
        text = "No history of hypertension."
        start = text.index("hypertension")
        result = detector.detect(text, start, start + len("hypertension"))
        assert result.status == AssertionStatus.ABSENT

    def test_unremarkable(self, detector: RuleBasedAssertionDetector) -> None:
        """'Unremarkable for' should negate."""
        text = "Physical exam unremarkable for edema."
        start = text.index("edema")
        result = detector.detect(text, start, start + len("edema"))
        assert result.status == AssertionStatus.ABSENT


class TestRuleBasedUncertainty:
    """Tests for uncertainty/possibility detection."""

    @pytest.fixture()
    def detector(self) -> RuleBasedAssertionDetector:
        return RuleBasedAssertionDetector()

    def test_possible_pneumonia(self, detector: RuleBasedAssertionDetector) -> None:
        """'Possible pneumonia' should be uncertain."""
        text = "Chest X-ray shows possible pneumonia."
        start = text.index("pneumonia")
        result = detector.detect(text, start, start + len("pneumonia"))
        assert result.status == AssertionStatus.POSSIBLE

    def test_rule_out(self, detector: RuleBasedAssertionDetector) -> None:
        """'Rule out DVT' should be uncertain."""
        text = "Rule out DVT in the left lower extremity."
        start = text.index("DVT")
        result = detector.detect(text, start, start + 3)
        assert result.status == AssertionStatus.POSSIBLE

    def test_r_slash_o(self, detector: RuleBasedAssertionDetector) -> None:
        """'r/o PE' should be uncertain."""
        text = "Presenting with dyspnea, r/o PE."
        start = text.index("PE")
        result = detector.detect(text, start, start + 2)
        assert result.status == AssertionStatus.POSSIBLE

    def test_suspected(self, detector: RuleBasedAssertionDetector) -> None:
        """'Suspected appendicitis' should be uncertain."""
        text = "Suspected appendicitis, CT ordered."
        start = text.index("appendicitis")
        result = detector.detect(text, start, start + len("appendicitis"))
        assert result.status == AssertionStatus.POSSIBLE

    def test_concern_for(self, detector: RuleBasedAssertionDetector) -> None:
        """'Concern for sepsis' should be uncertain."""
        text = "There is concern for sepsis."
        start = text.index("sepsis")
        result = detector.detect(text, start, start + len("sepsis"))
        assert result.status == AssertionStatus.POSSIBLE

    def test_may_have(self, detector: RuleBasedAssertionDetector) -> None:
        """'May have' should be uncertain."""
        text = "Patient may have early diabetes."
        start = text.index("early diabetes")
        result = detector.detect(text, start, start + len("early diabetes"))
        assert result.status == AssertionStatus.POSSIBLE

    def test_differential_includes(self, detector: RuleBasedAssertionDetector) -> None:
        """'Differential includes' should be uncertain."""
        text = "Differential diagnosis includes lymphoma."
        start = text.index("lymphoma")
        result = detector.detect(text, start, start + len("lymphoma"))
        assert result.status == AssertionStatus.POSSIBLE

    def test_cannot_be_excluded(self, detector: RuleBasedAssertionDetector) -> None:
        """'Cannot be excluded' should be uncertain."""
        text = "Fracture cannot be excluded on plain film."
        start = text.index("Fracture")
        result = detector.detect(text, start, start + len("Fracture"))
        assert result.status == AssertionStatus.POSSIBLE

    def test_question_mark(self, detector: RuleBasedAssertionDetector) -> None:
        """Trailing question mark implies uncertainty."""
        text = "Pneumonia?"
        start = 0
        result = detector.detect(text, start, start + len("Pneumonia"))
        assert result.status == AssertionStatus.POSSIBLE

    def test_equivocal(self, detector: RuleBasedAssertionDetector) -> None:
        """'Equivocal for' should be uncertain."""
        text = "Results equivocal for malignancy."
        start = text.index("malignancy")
        result = detector.detect(text, start, start + len("malignancy"))
        assert result.status == AssertionStatus.POSSIBLE


class TestRuleBasedFamily:
    """Tests for family history attribution."""

    @pytest.fixture()
    def detector(self) -> RuleBasedAssertionDetector:
        return RuleBasedAssertionDetector()

    def test_family_history_of(self, detector: RuleBasedAssertionDetector) -> None:
        """'Family history of diabetes' should be family."""
        text = "Family history of diabetes mellitus."
        start = text.index("diabetes mellitus")
        result = detector.detect(text, start, start + len("diabetes mellitus"))
        assert result.status == AssertionStatus.FAMILY

    def test_mother_had(self, detector: RuleBasedAssertionDetector) -> None:
        """'Mother had breast cancer' should be family."""
        text = "Mother had breast cancer at age 55."
        start = text.index("breast cancer")
        result = detector.detect(text, start, start + len("breast cancer"))
        assert result.status == AssertionStatus.FAMILY

    def test_father_diagnosed_with(self, detector: RuleBasedAssertionDetector) -> None:
        """'Father diagnosed with' should be family."""
        text = "Father diagnosed with coronary artery disease."
        start = text.index("coronary artery disease")
        result = detector.detect(text, start, start + len("coronary artery disease"))
        assert result.status == AssertionStatus.FAMILY

    def test_fh_abbreviation(self, detector: RuleBasedAssertionDetector) -> None:
        """'FH: diabetes' should be family."""
        text = "FH: diabetes, hypertension."
        start = text.index("diabetes")
        result = detector.detect(text, start, start + len("diabetes"))
        assert result.status == AssertionStatus.FAMILY

    def test_familial(self, detector: RuleBasedAssertionDetector) -> None:
        """'Familial hypercholesterolemia' should be family."""
        text = "Familial hypercholesterolemia suspected."
        start = text.index("hypercholesterolemia")
        result = detector.detect(text, start, start + len("hypercholesterolemia"))
        assert result.status == AssertionStatus.FAMILY

    def test_runs_in_family_post(self, detector: RuleBasedAssertionDetector) -> None:
        """'Runs in the family' (post-trigger) should be family."""
        text = "Heart disease runs in the family."
        start = text.index("Heart disease")
        result = detector.detect(text, start, start + len("Heart disease"))
        assert result.status == AssertionStatus.FAMILY

    def test_hereditary(self, detector: RuleBasedAssertionDetector) -> None:
        """'Hereditary' trigger should detect family assertion."""
        text = "Hereditary hemochromatosis."
        start = text.index("hemochromatosis")
        result = detector.detect(text, start, start + len("hemochromatosis"))
        assert result.status == AssertionStatus.FAMILY


class TestRuleBasedHypothetical:
    """Tests for hypothetical/future assertion detection."""

    @pytest.fixture()
    def detector(self) -> RuleBasedAssertionDetector:
        return RuleBasedAssertionDetector()

    def test_will_start(self, detector: RuleBasedAssertionDetector) -> None:
        """'Will start metformin' should be hypothetical."""
        text = "Will start metformin 500mg daily."
        start = text.index("metformin")
        result = detector.detect(text, start, start + len("metformin"))
        assert result.status == AssertionStatus.HYPOTHETICAL

    def test_plan_for(self, detector: RuleBasedAssertionDetector) -> None:
        """'Plan for surgery' should be hypothetical."""
        text = "Plan for knee replacement surgery."
        start = text.index("knee replacement surgery")
        result = detector.detect(text, start, start + len("knee replacement surgery"))
        assert result.status == AssertionStatus.HYPOTHETICAL

    def test_scheduled_for(self, detector: RuleBasedAssertionDetector) -> None:
        """'Scheduled for' should be hypothetical."""
        text = "Scheduled for colonoscopy next week."
        start = text.index("colonoscopy")
        result = detector.detect(text, start, start + len("colonoscopy"))
        assert result.status == AssertionStatus.HYPOTHETICAL

    def test_consider_starting(self, detector: RuleBasedAssertionDetector) -> None:
        """'Consider starting' should be hypothetical."""
        text = "Consider starting statin therapy."
        start = text.index("statin therapy")
        result = detector.detect(text, start, start + len("statin therapy"))
        assert result.status == AssertionStatus.HYPOTHETICAL

    def test_pending(self, detector: RuleBasedAssertionDetector) -> None:
        """'Pending biopsy' should be hypothetical."""
        text = "Pending biopsy results from pathology."
        start = text.index("biopsy")
        result = detector.detect(text, start, start + len("biopsy"))
        assert result.status == AssertionStatus.HYPOTHETICAL


class TestRuleBasedConditional:
    """Tests for conditional assertion detection."""

    @pytest.fixture()
    def detector(self) -> RuleBasedAssertionDetector:
        return RuleBasedAssertionDetector()

    def test_if_symptoms_worsen(self, detector: RuleBasedAssertionDetector) -> None:
        """'If symptoms worsen' should be conditional."""
        text = "If symptoms worsen, start antibiotics."
        start = text.index("antibiotics")
        result = detector.detect(text, start, start + len("antibiotics"))
        assert result.status == AssertionStatus.CONDITIONAL

    def test_unless(self, detector: RuleBasedAssertionDetector) -> None:
        """'Unless' trigger should detect conditional."""
        text = "Continue current regimen unless hypertension develops."
        start = text.index("hypertension")
        result = detector.detect(text, start, start + len("hypertension"))
        assert result.status == AssertionStatus.CONDITIONAL

    def test_in_event_of(self, detector: RuleBasedAssertionDetector) -> None:
        """'In the event of' should be conditional."""
        text = "In the event of anaphylaxis, administer epinephrine."
        start = text.index("anaphylaxis")
        result = detector.detect(text, start, start + len("anaphylaxis"))
        assert result.status == AssertionStatus.CONDITIONAL


class TestRuleBasedPseudoTriggers:
    """Tests for pseudo-trigger handling (preventing false positives)."""

    @pytest.fixture()
    def detector(self) -> RuleBasedAssertionDetector:
        return RuleBasedAssertionDetector()

    def test_no_increase_in_pain(self, detector: RuleBasedAssertionDetector) -> None:
        """'No increase in pain' should NOT negate pain."""
        text = "There is no increase in pain since last visit."
        start = text.index("pain")
        result = detector.detect(text, start, start + len("pain"))
        assert result.status == AssertionStatus.PRESENT
        assert result.metadata.get("pseudo_trigger") is True

    def test_no_change_in_symptoms(self, detector: RuleBasedAssertionDetector) -> None:
        """'No change in symptoms' should NOT negate symptoms."""
        text = "No change in symptoms this week."
        start = text.index("symptoms")
        result = detector.detect(text, start, start + len("symptoms"))
        assert result.status == AssertionStatus.PRESENT

    def test_not_causing(self, detector: RuleBasedAssertionDetector) -> None:
        """'Not causing' should not negate the subject condition."""
        text = "Medication is not causing headache."
        start = text.index("headache")
        result = detector.detect(text, start, start + len("headache"))
        assert result.status == AssertionStatus.PRESENT

    def test_gram_negative(self, detector: RuleBasedAssertionDetector) -> None:
        """'Gram negative' should not trigger negation."""
        text = "Gram negative rods seen on culture."
        start = text.index("rods")
        result = detector.detect(text, start, start + len("rods"))
        assert result.status == AssertionStatus.PRESENT


class TestRuleBasedTerminators:
    """Tests for scope terminator handling."""

    @pytest.fixture()
    def detector(self) -> RuleBasedAssertionDetector:
        return RuleBasedAssertionDetector()

    def test_but_terminates_negation(self, detector: RuleBasedAssertionDetector) -> None:
        """'No fever, but has chest pain' — 'but' blocks negation of chest pain."""
        text = "No fever, but has chest pain."
        start = text.index("chest pain")
        result = detector.detect(text, start, start + len("chest pain"))
        assert result.status == AssertionStatus.PRESENT

    def test_however_terminates(self, detector: RuleBasedAssertionDetector) -> None:
        """'However' should terminate negation scope."""
        text = "Denies nausea, however reports abdominal pain."
        start = text.index("abdominal pain")
        result = detector.detect(text, start, start + len("abdominal pain"))
        assert result.status == AssertionStatus.PRESENT

    def test_except_terminates(self, detector: RuleBasedAssertionDetector) -> None:
        """'Except for' should terminate negation scope."""
        text = "No complaints except for mild headache."
        start = text.index("mild headache")
        result = detector.detect(text, start, start + len("mild headache"))
        assert result.status == AssertionStatus.PRESENT

    def test_aside_from_terminates(self, detector: RuleBasedAssertionDetector) -> None:
        """'Aside from' should terminate negation scope."""
        text = "No issues, aside from persistent cough."
        start = text.index("persistent cough")
        result = detector.detect(text, start, start + len("persistent cough"))
        assert result.status == AssertionStatus.PRESENT


class TestRuleBasedScopeAndConfidence:
    """Tests for scope limiting and confidence scoring."""

    @pytest.fixture()
    def detector(self) -> RuleBasedAssertionDetector:
        return RuleBasedAssertionDetector()

    def test_trigger_out_of_scope(self, detector: RuleBasedAssertionDetector) -> None:
        """Trigger too far from entity should not apply."""
        # "no" is >60 chars from "pneumonia" — beyond most trigger scopes
        text = "No symptoms were reported by the patient during the extended comprehensive visit today. Pneumonia."
        start = text.index("Pneumonia")
        result = detector.detect(text, start, start + len("Pneumonia"))
        assert result.status == AssertionStatus.PRESENT

    def test_confidence_is_reasonable(self, detector: RuleBasedAssertionDetector) -> None:
        """Confidence should be between 0.5 and 1.0."""
        text = "Patient denies fever."
        start = text.index("fever")
        result = detector.detect(text, start, start + 5)
        assert 0.5 <= result.confidence <= 1.0

    def test_present_default_confidence(self, detector: RuleBasedAssertionDetector) -> None:
        """Default (present) assertions use default confidence."""
        text = "Patient has fever."
        start = text.index("fever")
        result = detector.detect(text, start, start + 5)
        assert result.confidence == detector.default_confidence

    def test_trigger_count(self, detector: RuleBasedAssertionDetector) -> None:
        """Detector should have a substantial trigger library."""
        assert detector.trigger_count > 60


class TestRuleBasedDefaultAndCustom:
    """Tests for default classification and custom triggers."""

    def test_present_when_no_trigger(self) -> None:
        """Entity with no trigger match should be PRESENT."""
        detector = RuleBasedAssertionDetector()
        text = "Patient presents with severe headache."
        start = text.index("severe headache")
        result = detector.detect(text, start, start + len("severe headache"))
        assert result.status == AssertionStatus.PRESENT
        assert result.trigger_text is None
        assert result.trigger_type is None

    def test_custom_trigger(self) -> None:
        """Custom triggers should be recognized."""
        custom = [
            Trigger(
                pattern=r"\bcategory\s+Z\s+",
                trigger_type=TriggerType.PRE,
                assertion=AssertionStatus.ABSENT,
                priority=10,
                max_scope=30,
            )
        ]
        detector = RuleBasedAssertionDetector(custom_triggers=custom)
        text = "category Z diabetes."
        start = text.index("diabetes")
        result = detector.detect(text, start, start + len("diabetes"))
        assert result.status == AssertionStatus.ABSENT

    def test_custom_default_status(self) -> None:
        """Custom default status is used when no trigger matches."""
        detector = RuleBasedAssertionDetector(
            default_status=AssertionStatus.POSSIBLE
        )
        text = "Patient has a rash."
        start = text.index("rash")
        result = detector.detect(text, start, start + 4)
        assert result.status == AssertionStatus.POSSIBLE


class TestRuleBasedBatch:
    """Tests for batch assertion detection."""

    @pytest.fixture()
    def detector(self) -> RuleBasedAssertionDetector:
        return RuleBasedAssertionDetector()

    def test_batch_detection(self, detector: RuleBasedAssertionDetector) -> None:
        """Batch detection processes multiple entities."""
        text = "No fever. Possible pneumonia. Patient has headache."
        entities = [
            (text.index("fever"), text.index("fever") + 5),
            (text.index("pneumonia"), text.index("pneumonia") + 9),
            (text.index("headache"), text.index("headache") + 8),
        ]
        results = detector.detect_batch(text, entities)
        assert len(results) == 3
        assert results[0].status == AssertionStatus.ABSENT
        assert results[1].status == AssertionStatus.POSSIBLE
        assert results[2].status == AssertionStatus.PRESENT

    def test_batch_empty(self, detector: RuleBasedAssertionDetector) -> None:
        """Empty entity list returns empty results."""
        results = detector.detect_batch("Some text.", [])
        assert results == []


class TestRuleBasedRealisticNotes:
    """Tests with realistic clinical note snippets."""

    @pytest.fixture()
    def detector(self) -> RuleBasedAssertionDetector:
        return RuleBasedAssertionDetector()

    def test_review_of_systems_negatives(self, detector: RuleBasedAssertionDetector) -> None:
        """ROS negatives should be properly detected."""
        text = "ROS: Patient denies any chest pain, shortness of breath, or palpitations."
        start = text.index("chest pain")
        result = detector.detect(text, start, start + len("chest pain"))
        assert result.status == AssertionStatus.ABSENT

    def test_mixed_positive_negative(self, detector: RuleBasedAssertionDetector) -> None:
        """Mixed positive/negative note parsed correctly."""
        text = "Patient reports fatigue but denies fever or chills."
        # "fatigue" is affirmed
        start = text.index("fatigue")
        result = detector.detect(text, start, start + len("fatigue"))
        assert result.status == AssertionStatus.PRESENT

    def test_assessment_uncertainty(self, detector: RuleBasedAssertionDetector) -> None:
        """Assessment section often has uncertain findings."""
        text = "Assessment: Likely community-acquired pneumonia."
        start = text.index("community-acquired pneumonia")
        result = detector.detect(text, start, start + len("community-acquired pneumonia"))
        assert result.status == AssertionStatus.POSSIBLE


# ---------------------------------------------------------------------------
# ConTextAssertionDetector tests
# ---------------------------------------------------------------------------


class TestConTextDetector:
    """Tests for the ConText assertion detector with section awareness."""

    @pytest.fixture()
    def detector(self) -> ConTextAssertionDetector:
        return ConTextAssertionDetector()

    def test_family_history_section(self, detector: ConTextAssertionDetector) -> None:
        """Entities in 'Family History:' section should be family."""
        text = "Chief Complaint: Chest pain.\nFamily History:\nDiabetes mellitus type 2."
        start = text.index("Diabetes mellitus")
        result = detector.detect(text, start, start + len("Diabetes mellitus"))
        assert result.status == AssertionStatus.FAMILY
        assert result.metadata.get("section_override") is True

    def test_fh_section_header(self, detector: ConTextAssertionDetector) -> None:
        """'FH:' section header should also trigger family."""
        text = "FH:\nBreast cancer, colon cancer."
        start = text.index("Breast cancer")
        result = detector.detect(text, start, start + len("Breast cancer"))
        assert result.status == AssertionStatus.FAMILY

    def test_pertinent_negatives_section(self, detector: ConTextAssertionDetector) -> None:
        """Entities in 'Pertinent Negatives:' section should be absent."""
        text = "Pertinent Negatives:\nPulmonary embolism."
        start = text.index("Pulmonary embolism")
        result = detector.detect(text, start, start + len("Pulmonary embolism"))
        assert result.status == AssertionStatus.ABSENT

    def test_section_does_not_override_explicit_trigger(self, detector: ConTextAssertionDetector) -> None:
        """Explicit trigger should take precedence over section context.

        If an entity is in a family history section but has an explicit
        negation trigger, the explicit trigger wins.
        """
        text = "Family History:\nNo diabetes."
        start = text.index("diabetes")
        result = detector.detect(text, start, start + len("diabetes"))
        # "no" negation trigger should override family section
        assert result.status == AssertionStatus.ABSENT

    def test_new_section_cancels_previous(self, detector: ConTextAssertionDetector) -> None:
        """A new section header should cancel the previous section context."""
        text = "Family History:\nDiabetes.\nPhysical Exam:\nHeart murmur."
        start = text.index("Heart murmur")
        result = detector.detect(text, start, start + len("Heart murmur"))
        # "Physical Exam:" is a new section, cancelling family history context
        assert result.status == AssertionStatus.PRESENT

    def test_stats_tracking(self, detector: ConTextAssertionDetector) -> None:
        """Detection stats should be updated."""
        text = "No fever. Possible pneumonia. Headache present."
        detector.detect(text, text.index("fever"), text.index("fever") + 5)
        detector.detect(text, text.index("pneumonia"), text.index("pneumonia") + 9)
        detector.detect(text, text.index("Headache"), text.index("Headache") + 8)

        stats = detector.stats
        assert stats["total_detections"] == 3
        assert stats["absent"] >= 1
        assert stats["possible"] >= 1

    def test_stats_reset(self, detector: ConTextAssertionDetector) -> None:
        """Stats should reset to zero."""
        text = "No fever."
        detector.detect(text, text.index("fever"), text.index("fever") + 5)
        assert detector.stats["total_detections"] > 0

        detector.reset_stats()
        assert detector.stats["total_detections"] == 0

    def test_inherits_negation(self, detector: ConTextAssertionDetector) -> None:
        """ConText detector inherits rule-based negation."""
        text = "Patient denies chest pain."
        start = text.index("chest pain")
        result = detector.detect(text, start, start + len("chest pain"))
        assert result.status == AssertionStatus.ABSENT

    def test_plan_section_hypothetical(self, detector: ConTextAssertionDetector) -> None:
        """Entities in 'Plan:' section should be hypothetical."""
        text = "Assessment: COPD exacerbation.\nPlan:\nAlbuterol nebulizer."
        start = text.index("Albuterol nebulizer")
        result = detector.detect(text, start, start + len("Albuterol nebulizer"))
        assert result.status == AssertionStatus.HYPOTHETICAL
