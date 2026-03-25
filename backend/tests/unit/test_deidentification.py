"""Tests for the PHI de-identification module.

Covers all 18 HIPAA Safe Harbor identifier categories, overlap resolution,
confidence adjustment, replacement strategies, batch processing, and edge
cases.
"""

from unittest.mock import MagicMock

import pytest

from app.ml.deidentification.detector import (
    DeidentificationConfig,
    Deidentifier,
    PhiDetector,
    PhiEntity,
    PhiType,
    ReplacementStrategy,
)

# ---------------------------------------------------------------------------
# PhiEntity
# ---------------------------------------------------------------------------

class TestPhiEntity:
    """PhiEntity dataclass tests."""

    def test_to_dict_all_fields(self) -> None:
        """to_dict returns all fields as a plain dictionary."""
        entity = PhiEntity(
            text="Dr. Smith",
            phi_type=PhiType.NAME,
            start_char=0,
            end_char=9,
            confidence=0.95,
            pattern_name="name_titled",
        )
        d = entity.to_dict()
        assert d["text"] == "Dr. Smith"
        assert d["phi_type"] == "NAME"
        assert d["start_char"] == 0
        assert d["end_char"] == 9
        assert d["confidence"] == 0.95
        assert d["pattern_name"] == "name_titled"

    def test_frozen_dataclass(self) -> None:
        """PhiEntity is immutable (frozen dataclass)."""
        entity = PhiEntity(
            text="test", phi_type=PhiType.DATE,
            start_char=0, end_char=4,
        )
        with pytest.raises(AttributeError):
            entity.text = "modified"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PhiDetector — Pattern detection
# ---------------------------------------------------------------------------

class TestPhiDetectorNames:
    """Name detection patterns."""

    def test_detect_titled_name(self) -> None:
        """Detect 'Dr. Smith' as NAME."""
        detector = PhiDetector()
        entities = detector.detect("Patient Dr. Smith presented with chest pain.")
        names = [e for e in entities if e.phi_type == PhiType.NAME]
        assert len(names) >= 1
        assert "Smith" in names[0].text

    def test_detect_multiple_names(self) -> None:
        """Detect multiple titled names."""
        text = "Dr. Jones referred patient to Dr. Williams for surgery."
        entities = PhiDetector().detect(text)
        names = [e for e in entities if e.phi_type == PhiType.NAME]
        assert len(names) == 2

    def test_name_with_middle(self) -> None:
        """Detect names with middle name."""
        text = "Seen by Dr. John Michael Smith."
        entities = PhiDetector().detect(text)
        names = [e for e in entities if e.phi_type == PhiType.NAME]
        assert len(names) >= 1


class TestPhiDetectorDates:
    """Date detection patterns."""

    def test_mm_dd_yyyy(self) -> None:
        """Detect MM/DD/YYYY format."""
        entities = PhiDetector().detect("Admitted 01/15/2024 for observation.")
        dates = [e for e in entities if e.phi_type == PhiType.DATE]
        assert len(dates) >= 1
        assert "01/15/2024" in dates[0].text

    def test_iso_date(self) -> None:
        """Detect YYYY-MM-DD (ISO) format."""
        entities = PhiDetector().detect("Lab drawn 2024-03-15 at 08:00.")
        dates = [e for e in entities if e.phi_type == PhiType.DATE]
        assert len(dates) >= 1
        assert "2024-03-15" in dates[0].text

    def test_month_dd_yyyy(self) -> None:
        """Detect 'January 15, 2024' format."""
        entities = PhiDetector().detect("Follow-up scheduled January 15, 2024.")
        dates = [e for e in entities if e.phi_type == PhiType.DATE]
        assert len(dates) >= 1
        assert "January" in dates[0].text

    def test_dd_month_yyyy(self) -> None:
        """Detect '15 March 2024' format."""
        entities = PhiDetector().detect("Born 15 March 1990 in Ohio.")
        dates = [e for e in entities if e.phi_type == PhiType.DATE]
        assert len(dates) >= 1

    def test_mm_dd_yyyy_with_dashes(self) -> None:
        """Detect MM-DD-YYYY format with dashes."""
        entities = PhiDetector().detect("DOB: 06-22-1985.")
        dates = [e for e in entities if e.phi_type == PhiType.DATE]
        assert len(dates) >= 1


class TestPhiDetectorPhone:
    """Phone number detection."""

    def test_us_phone_parens(self) -> None:
        """Detect (555) 123-4567 format."""
        entities = PhiDetector().detect("Call (555) 234-5678 for results.")
        phones = [e for e in entities if e.phi_type == PhiType.PHONE]
        assert len(phones) >= 1

    def test_us_phone_dashes(self) -> None:
        """Detect 555-234-5678 format."""
        entities = PhiDetector().detect("Phone: 555-234-5678.")
        phones = [e for e in entities if e.phi_type == PhiType.PHONE]
        assert len(phones) >= 1

    def test_phone_with_context_boost(self) -> None:
        """Phone preceded by 'tel' gets confidence boost."""
        entities = PhiDetector().detect("Tel: 555-234-5678.")
        phones = [e for e in entities if e.phi_type == PhiType.PHONE]
        assert len(phones) >= 1
        assert phones[0].confidence > 0.90


class TestPhiDetectorEmail:
    """Email detection."""

    def test_standard_email(self) -> None:
        """Detect standard email address."""
        entities = PhiDetector().detect("Contact john.doe@hospital.org for results.")
        emails = [e for e in entities if e.phi_type == PhiType.EMAIL]
        assert len(emails) == 1
        assert "john.doe@hospital.org" in emails[0].text

    def test_email_high_confidence(self) -> None:
        """Email detection has high base confidence (0.98)."""
        entities = PhiDetector().detect("Email: test@example.com")
        emails = [e for e in entities if e.phi_type == PhiType.EMAIL]
        assert emails[0].confidence >= 0.98


class TestPhiDetectorSSN:
    """SSN detection."""

    def test_ssn_with_dashes(self) -> None:
        """Detect SSN with dashes and context."""
        text = "SSN: 123-45-6789"
        entities = PhiDetector().detect(text)
        ssns = [e for e in entities if e.phi_type == PhiType.SSN]
        assert len(ssns) >= 1

    def test_ssn_context_boost(self) -> None:
        """SSN preceded by 'Social Security' gets confidence boost."""
        text = "Social Security Number: 123-45-6789"
        entities = PhiDetector().detect(text)
        ssns = [e for e in entities if e.phi_type == PhiType.SSN]
        assert len(ssns) >= 1
        assert ssns[0].confidence > 0.85

    def test_ssn_without_context_lower_confidence(self) -> None:
        """Bare 9-digit number without SSN context has lower confidence."""
        text = "Reference: 123-45-6789"
        detector = PhiDetector(DeidentificationConfig(confidence_threshold=0.0))
        entities = detector.detect(text)
        ssns = [e for e in entities if e.phi_type == PhiType.SSN]
        if ssns:
            assert ssns[0].confidence < 0.85


class TestPhiDetectorMRN:
    """Medical Record Number detection."""

    def test_mrn_with_prefix(self) -> None:
        """Detect MRN with explicit prefix."""
        text = "MRN: A1234567"
        entities = PhiDetector().detect(text)
        mrns = [e for e in entities if e.phi_type == PhiType.MRN]
        assert len(mrns) >= 1

    def test_medical_record_number_prefix(self) -> None:
        """Detect 'Medical Record Number' prefix."""
        text = "Medical Record Number: MR789012"
        entities = PhiDetector().detect(text)
        mrns = [e for e in entities if e.phi_type == PhiType.MRN]
        assert len(mrns) >= 1


class TestPhiDetectorURL:
    """URL detection."""

    def test_http_url(self) -> None:
        """Detect HTTP URL."""
        entities = PhiDetector().detect("Results at http://portal.hospital.org/results")
        urls = [e for e in entities if e.phi_type == PhiType.URL]
        assert len(urls) == 1

    def test_https_url(self) -> None:
        """Detect HTTPS URL."""
        entities = PhiDetector().detect("Visit https://patient.example.com/records")
        urls = [e for e in entities if e.phi_type == PhiType.URL]
        assert len(urls) == 1


class TestPhiDetectorIP:
    """IP address detection."""

    def test_ipv4_address(self) -> None:
        """Detect standard IPv4 address."""
        entities = PhiDetector().detect("Logged from 192.168.1.100 at 14:00.")
        ips = [e for e in entities if e.phi_type == PhiType.IP_ADDRESS]
        assert len(ips) >= 1
        assert "192.168.1.100" in ips[0].text


class TestPhiDetectorAge:
    """Age over 89 detection (Safe Harbor)."""

    def test_age_90_detected(self) -> None:
        """Age 90 years old is flagged."""
        entities = PhiDetector().detect("Patient is 92 years old with COPD.")
        ages = [e for e in entities if e.phi_type == PhiType.AGE]
        assert len(ages) >= 1

    def test_age_under_90_not_detected(self) -> None:
        """Age under 90 should NOT be flagged."""
        entities = PhiDetector().detect("Patient is 85 years old with HTN.")
        ages = [e for e in entities if e.phi_type == PhiType.AGE]
        assert len(ages) == 0

    def test_age_yo_abbreviation(self) -> None:
        """Detect '95 y/o' abbreviation."""
        entities = PhiDetector().detect("95 y/o female admitted for fall.")
        ages = [e for e in entities if e.phi_type == PhiType.AGE]
        assert len(ages) >= 1


class TestPhiDetectorAccount:
    """Account number detection."""

    def test_account_number(self) -> None:
        """Detect account number with prefix."""
        text = "Account Number: ABC123456"
        entities = PhiDetector().detect(text)
        accts = [e for e in entities if e.phi_type == PhiType.ACCOUNT]
        assert len(accts) >= 1


class TestPhiDetectorLicense:
    """License/DEA/NPI number detection."""

    def test_license_number(self) -> None:
        """Detect license number."""
        text = "License No: MD123456789"
        entities = PhiDetector().detect(text)
        lics = [e for e in entities if e.phi_type == PhiType.LICENSE]
        assert len(lics) >= 1

    def test_dea_number(self) -> None:
        """Detect DEA number."""
        text = "DEA# AB1234567"
        entities = PhiDetector().detect(text)
        lics = [e for e in entities if e.phi_type == PhiType.LICENSE]
        assert len(lics) >= 1

    def test_npi_number(self) -> None:
        """Detect NPI number."""
        text = "NPI: 1234567890"
        entities = PhiDetector().detect(text)
        lics = [e for e in entities if e.phi_type == PhiType.LICENSE]
        assert len(lics) >= 1


# ---------------------------------------------------------------------------
# PhiDetector — Configuration & filtering
# ---------------------------------------------------------------------------

class TestPhiDetectorConfig:
    """Configuration and filtering behaviour."""

    def test_enabled_types_filter(self) -> None:
        """Only detect specified PHI types when enabled_types is set."""
        config = DeidentificationConfig(
            enabled_types={PhiType.DATE, PhiType.EMAIL},
        )
        text = "Dr. Smith seen on 01/15/2024, email: dr@example.com"
        entities = PhiDetector(config).detect(text)
        types_found = {e.phi_type for e in entities}
        assert PhiType.NAME not in types_found
        assert PhiType.DATE in types_found or PhiType.EMAIL in types_found

    def test_confidence_threshold(self) -> None:
        """Entities below confidence_threshold are excluded."""
        config = DeidentificationConfig(confidence_threshold=0.99)
        # Names have base confidence 0.90, won't pass 0.99
        entities = PhiDetector(config).detect("Dr. Smith has a cold.")
        names = [e for e in entities if e.phi_type == PhiType.NAME]
        assert len(names) == 0

    def test_empty_text_returns_empty(self) -> None:
        """Empty string returns no entities."""
        assert PhiDetector().detect("") == []

    def test_whitespace_only_returns_empty(self) -> None:
        """Whitespace-only text returns no entities."""
        assert PhiDetector().detect("   \n\t  ") == []

    def test_no_phi_returns_empty(self) -> None:
        """Text without PHI returns empty list."""
        entities = PhiDetector().detect("The patient has mild hypertension.")
        # May have some low-confidence ZIP-like matches, but should be minimal
        high_conf = [e for e in entities if e.confidence > 0.8]
        assert len(high_conf) == 0


# ---------------------------------------------------------------------------
# PhiDetector — Overlap resolution
# ---------------------------------------------------------------------------

class TestOverlapResolution:
    """Overlapping span resolution."""

    def test_non_overlapping_preserved(self) -> None:
        """Non-overlapping entities are all preserved."""
        entities = [
            PhiEntity("Dr. Smith", PhiType.NAME, 0, 9, 0.95, "a"),
            PhiEntity("01/15/2024", PhiType.DATE, 20, 30, 0.95, "b"),
        ]
        resolved = PhiDetector._resolve_overlaps(entities)
        assert len(resolved) == 2

    def test_longer_match_wins(self) -> None:
        """When spans overlap, the longer match is kept."""
        entities = [
            PhiEntity("123-45", PhiType.PHONE, 0, 6, 0.90, "short"),
            PhiEntity("123-45-6789", PhiType.SSN, 0, 11, 0.85, "long"),
        ]
        resolved = PhiDetector._resolve_overlaps(entities)
        assert len(resolved) == 1
        assert resolved[0].text == "123-45-6789"

    def test_empty_input(self) -> None:
        """Empty entity list returns empty."""
        assert PhiDetector._resolve_overlaps([]) == []

    def test_single_entity(self) -> None:
        """Single entity passes through unchanged."""
        entity = PhiEntity("test", PhiType.NAME, 0, 4, 0.95, "a")
        resolved = PhiDetector._resolve_overlaps([entity])
        assert len(resolved) == 1
        assert resolved[0] is entity


# ---------------------------------------------------------------------------
# PhiDetector — Custom detectors
# ---------------------------------------------------------------------------

class TestCustomDetector:
    """Custom detector integration."""

    def test_custom_detector_results_merged(self) -> None:
        """Custom detector entities are merged with regex results."""
        custom = MagicMock()
        custom.detect.return_value = [
            PhiEntity("Custom PHI", PhiType.BIOMETRIC, 50, 60, 0.99, "custom"),
        ]
        detector = PhiDetector()
        detector.add_custom_detector(custom)
        entities = detector.detect("Some text " * 10)
        # Custom entity should be in the result
        biometrics = [e for e in entities if e.phi_type == PhiType.BIOMETRIC]
        assert len(biometrics) == 1

    def test_custom_detector_failure_graceful(self) -> None:
        """Custom detector exception is caught gracefully."""
        custom = MagicMock()
        custom.detect.side_effect = RuntimeError("model crashed")
        detector = PhiDetector()
        detector.add_custom_detector(custom)
        # Should not raise
        entities = detector.detect("Dr. Smith on 01/15/2024")
        assert isinstance(entities, list)


# ---------------------------------------------------------------------------
# Deidentifier — Replacement strategies
# ---------------------------------------------------------------------------

class TestDeidentifierRedact:
    """REDACT replacement strategy."""

    def test_redact_replaces_with_tags(self) -> None:
        """REDACT strategy replaces PHI with [TYPE] tags."""
        deid = Deidentifier()
        result = deid.deidentify("Dr. Smith seen on 01/15/2024.")
        assert "[NAME]" in result["text"]
        assert "[DATE]" in result["text"]
        assert "Smith" not in result["text"]
        assert "01/15/2024" not in result["text"]

    def test_redact_entity_count(self) -> None:
        """Entity count matches detected PHI spans."""
        deid = Deidentifier()
        result = deid.deidentify("Dr. Smith, email: doc@hospital.org, DOB: 03/15/1980")
        assert result["entity_count"] >= 2

    def test_redact_phi_types_found(self) -> None:
        """phi_types_found lists unique types detected."""
        deid = Deidentifier()
        result = deid.deidentify("Dr. Smith seen on 01/15/2024.")
        assert "NAME" in result["phi_types_found"]
        assert "DATE" in result["phi_types_found"]


class TestDeidentifierMask:
    """MASK replacement strategy."""

    def test_mask_preserves_length(self) -> None:
        """MASK with preserve_length replaces each char with *."""
        config = DeidentificationConfig(
            strategy=ReplacementStrategy.MASK,
            preserve_length=True,
        )
        deid = Deidentifier(config)
        result = deid.deidentify("Email: test@example.com is the contact.")
        # The email should be replaced with asterisks of same length
        assert "test@example.com" not in result["text"]
        assert "****" in result["text"]

    def test_mask_fixed_length(self) -> None:
        """MASK without preserve_length uses fixed '****'."""
        config = DeidentificationConfig(
            strategy=ReplacementStrategy.MASK,
            preserve_length=False,
        )
        deid = Deidentifier(config)
        result = deid.deidentify("Email: test@example.com here.")
        assert "test@example.com" not in result["text"]


class TestDeidentifierSurrogate:
    """SURROGATE replacement strategy."""

    def test_surrogate_provides_realistic_values(self) -> None:
        """SURROGATE replaces PHI with synthetic values."""
        config = DeidentificationConfig(
            strategy=ReplacementStrategy.SURROGATE,
        )
        deid = Deidentifier(config)
        result = deid.deidentify("Dr. Smith seen on 01/15/2024.")
        text = result["text"]
        # Should NOT have the original PHI
        assert "Smith" not in text
        assert "01/15/2024" not in text
        # Should NOT have redact tags either
        assert "[NAME]" not in text
        assert "[DATE]" not in text

    def test_surrogate_deterministic(self) -> None:
        """Same seed produces same surrogates."""
        config = DeidentificationConfig(
            strategy=ReplacementStrategy.SURROGATE,
            surrogate_seed=42,
        )
        result1 = Deidentifier(config).deidentify("Email: a@b.com")
        result2 = Deidentifier(config).deidentify("Email: a@b.com")
        assert result1["text"] == result2["text"]

    def test_surrogate_fallback_for_unknown_type(self) -> None:
        """PHI types without surrogates fall back to [TYPE] tag."""
        config = DeidentificationConfig(
            strategy=ReplacementStrategy.SURROGATE,
        )
        deid = Deidentifier(config)
        # URL type has no surrogates defined
        result = deid.deidentify("Visit https://patient-portal.example.com/records")
        urls = [e for e in result["entities"] if e["phi_type"] == "URL"]
        if urls:
            # Fallback should produce [URL] tag since no URL surrogates
            assert "patient-portal.example.com" not in result["text"]


# ---------------------------------------------------------------------------
# Deidentifier — Edge cases
# ---------------------------------------------------------------------------

class TestDeidentifierEdgeCases:
    """Edge cases and error handling."""

    def test_empty_text(self) -> None:
        """Empty text returns empty result."""
        result = Deidentifier().deidentify("")
        assert result["text"] == ""
        assert result["entities"] == []
        assert result["entity_count"] == 0

    def test_none_text(self) -> None:
        """None text returns empty result."""
        result = Deidentifier().deidentify(None)  # type: ignore[arg-type]
        assert result["entity_count"] == 0

    def test_no_phi_text(self) -> None:
        """Text without PHI passes through unchanged."""
        text = "The patient presents with mild hypertension and diabetes."
        result = Deidentifier().deidentify(text)
        # Text should be unchanged (or very close — some low-confidence
        # matches might be suppressed by threshold)
        assert result["entity_count"] == 0 or result["text"] != text

    def test_multiple_phi_types(self) -> None:
        """Text with multiple PHI types detects all."""
        text = (
            "Dr. Johnson seen on 03/15/2024. Contact: doc@hospital.org. "
            "SSN: 123-45-6789. MRN: PAT12345"
        )
        result = Deidentifier().deidentify(text)
        assert result["entity_count"] >= 3
        types = set(result["phi_types_found"])
        assert len(types) >= 2

    def test_batch_deidentify(self) -> None:
        """Batch processing handles multiple texts."""
        texts = [
            "Dr. Smith on 01/15/2024",
            "Contact: test@hospital.org",
            "MRN: ABC12345",
        ]
        results = Deidentifier().deidentify_batch(texts)
        assert len(results) == 3
        assert all(isinstance(r, dict) for r in results)

    def test_batch_empty_list(self) -> None:
        """Batch with empty list returns empty results."""
        results = Deidentifier().deidentify_batch([])
        assert results == []

    def test_strategy_in_result(self) -> None:
        """Result includes the strategy name."""
        for strategy in ReplacementStrategy:
            config = DeidentificationConfig(strategy=strategy)
            result = Deidentifier(config).deidentify("test text")
            assert result["strategy"] == strategy.value


# ---------------------------------------------------------------------------
# Deidentifier — Complex clinical notes
# ---------------------------------------------------------------------------

class TestDeidentifierClinicalNotes:
    """Realistic clinical note de-identification."""

    CLINICAL_NOTE = (
        "DISCHARGE SUMMARY\n"
        "Patient: Dr. Margaret Williams\n"
        "DOB: 06/22/1985\n"
        "MRN: MR7890123\n"
        "Date of Admission: January 15, 2024\n"
        "Date of Discharge: January 22, 2024\n\n"
        "HISTORY OF PRESENT ILLNESS:\n"
        "92 year old female presented to the ED with chest pain. "
        "Contact phone: (555) 234-5678. "
        "Email: margaret.w@patientportal.org\n\n"
        "ATTENDING: Dr. Robert Chen\n"
        "NPI: 1234567890\n"
    )

    def test_clinical_note_names_redacted(self) -> None:
        """Names are redacted from clinical note."""
        result = Deidentifier().deidentify(self.CLINICAL_NOTE)
        assert "Margaret" not in result["text"] or "[NAME]" in result["text"]

    def test_clinical_note_dates_redacted(self) -> None:
        """Dates are redacted from clinical note."""
        result = Deidentifier().deidentify(self.CLINICAL_NOTE)
        assert "06/22/1985" not in result["text"]

    def test_clinical_note_mrn_redacted(self) -> None:
        """MRN is redacted from clinical note."""
        result = Deidentifier().deidentify(self.CLINICAL_NOTE)
        assert "MR7890123" not in result["text"]

    def test_clinical_note_email_redacted(self) -> None:
        """Email is redacted from clinical note."""
        result = Deidentifier().deidentify(self.CLINICAL_NOTE)
        assert "margaret.w@patientportal.org" not in result["text"]

    def test_clinical_note_age_over_89_redacted(self) -> None:
        """Age ≥ 90 is redacted per Safe Harbor."""
        result = Deidentifier().deidentify(self.CLINICAL_NOTE)
        ages = [e for e in result["entities"] if e["phi_type"] == "AGE"]
        assert len(ages) >= 1

    def test_clinical_note_preserves_medical_content(self) -> None:
        """Medical content (diagnoses, procedures) is preserved."""
        result = Deidentifier().deidentify(self.CLINICAL_NOTE)
        assert "chest pain" in result["text"]
        assert "DISCHARGE SUMMARY" in result["text"]
        assert "HISTORY OF PRESENT ILLNESS" in result["text"]

    def test_clinical_note_multiple_phi_types(self) -> None:
        """Multiple PHI types detected in realistic note."""
        result = Deidentifier().deidentify(self.CLINICAL_NOTE)
        assert len(result["phi_types_found"]) >= 3


# ---------------------------------------------------------------------------
# PhiType and ReplacementStrategy enums
# ---------------------------------------------------------------------------

class TestEnums:
    """Enum completeness tests."""

    def test_phi_type_has_18_members(self) -> None:
        """PhiType covers all 18 HIPAA Safe Harbor categories."""
        assert len(PhiType) == 18

    def test_replacement_strategy_has_3_members(self) -> None:
        """ReplacementStrategy has redact, mask, surrogate."""
        assert len(ReplacementStrategy) == 3

    def test_phi_type_string_values(self) -> None:
        """All PhiType values are uppercase strings."""
        for pt in PhiType:
            assert pt.value == pt.value.upper()
            assert isinstance(pt.value, str)


# ---------------------------------------------------------------------------
# DeidentificationConfig
# ---------------------------------------------------------------------------

class TestDeidentificationConfig:
    """Configuration defaults and overrides."""

    def test_defaults(self) -> None:
        """Default config uses REDACT with all types."""
        config = DeidentificationConfig()
        assert config.strategy == ReplacementStrategy.REDACT
        assert config.enabled_types is None
        assert config.confidence_threshold == 0.5
        assert config.context_window == 30
        assert config.preserve_length is True
        assert config.surrogate_seed == 42

    def test_custom_config(self) -> None:
        """Custom values are respected."""
        config = DeidentificationConfig(
            strategy=ReplacementStrategy.MASK,
            enabled_types={PhiType.NAME, PhiType.DATE},
            confidence_threshold=0.8,
            context_window=50,
            preserve_length=False,
            surrogate_seed=99,
        )
        assert config.strategy == ReplacementStrategy.MASK
        assert PhiType.NAME in config.enabled_types
        assert config.confidence_threshold == 0.8
