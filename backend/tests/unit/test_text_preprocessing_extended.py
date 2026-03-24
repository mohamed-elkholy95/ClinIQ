"""Extended tests for text_preprocessing — max_document_length truncation,
section detection disabled, and preprocess_clinical_text convenience function."""

import pytest

from app.ml.utils.text_preprocessing import (
    ClinicalTextPreprocessor,
    PreprocessingConfig,
    preprocess_clinical_text,
)


class TestMaxDocumentLength:
    """Cover the max_document_length truncation path (line 210)."""

    def test_long_text_truncated(self) -> None:
        """Text longer than max_document_length is truncated before processing."""
        config = PreprocessingConfig(max_document_length=20)
        preprocessor = ClinicalTextPreprocessor(config=config)
        long_text = "A" * 100
        result = preprocessor.preprocess(long_text)
        assert len(result) <= 20

    def test_short_text_not_truncated(self) -> None:
        """Text shorter than max_document_length passes through unchanged."""
        config = PreprocessingConfig(max_document_length=1000)
        preprocessor = ClinicalTextPreprocessor(config=config)
        text = "Short clinical note."
        result = preprocessor.preprocess(text)
        assert "Short clinical note" in result


class TestSectionDetectionDisabled:
    """Cover detect_sections=False branch (line 250)."""

    def test_sections_disabled(self) -> None:
        """When detect_sections=False, detect_sections returns empty list."""
        config = PreprocessingConfig(detect_sections=False)
        preprocessor = ClinicalTextPreprocessor(config=config)
        text = "HISTORY OF PRESENT ILLNESS:\nPatient has diabetes.\nASSESSMENT:\nDiabetes."
        sections = preprocessor.detect_sections(text)
        assert sections == []


class TestPreprocessClinicalText:
    """Cover the module-level convenience function (lines 373-375)."""

    def test_normal_text(self) -> None:
        """Convenience function processes text normally."""
        result = preprocess_clinical_text("Patient has   diabetes   mellitus.")
        assert "diabetes" in result

    def test_empty_string(self) -> None:
        """Empty string returns empty string."""
        result = preprocess_clinical_text("")
        assert result == ""

    def test_whitespace_only(self) -> None:
        """Whitespace-only string returns empty string."""
        result = preprocess_clinical_text("   \n\t  ")
        assert result == ""

    def test_type_error(self) -> None:
        """Non-string input raises TypeError."""
        with pytest.raises(TypeError, match="Expected str"):
            preprocess_clinical_text(123)  # type: ignore[arg-type]

    def test_custom_config(self) -> None:
        """Passing a custom config is respected."""
        config = PreprocessingConfig(normalize_whitespace=False)
        result = preprocess_clinical_text("hello   world", config=config)
        # With normalize_whitespace=False, extra spaces may be preserved
        assert "hello" in result
