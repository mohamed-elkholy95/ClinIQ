"""Tests for clinical text input validation."""


from app.ml.utils.validation import validate_clinical_text


class TestValidateClinicalText:
    """Verify the validate_clinical_text function covers key edge cases."""

    def test_valid_clinical_note(self) -> None:
        """A typical clinical note should pass validation."""
        text = (
            "Patient is a 55-year-old male presenting with chest pain. "
            "History of hypertension and diabetes mellitus type 2."
        )
        result = validate_clinical_text(text)
        assert result.is_valid is True
        assert result.errors == []

    def test_empty_string_rejected(self) -> None:
        result = validate_clinical_text("")
        assert result.is_valid is False
        assert any("empty" in e.lower() for e in result.errors)

    def test_whitespace_only_rejected(self) -> None:
        result = validate_clinical_text("   \n\t  ")
        assert result.is_valid is False

    def test_too_short_rejected(self) -> None:
        result = validate_clinical_text("short")
        assert result.is_valid is False
        assert any("too short" in e.lower() for e in result.errors)

    def test_null_bytes_rejected(self) -> None:
        result = validate_clinical_text("Patient has \x00 diabetes")
        assert result.is_valid is False
        assert any("null" in e.lower() for e in result.errors)

    def test_over_max_length_warns(self) -> None:
        text = "Patient has diabetes. " * 10_000  # ~220k chars
        result = validate_clinical_text(text, max_length=100_000)
        assert result.is_valid is True  # still valid, just warns
        assert len(result.warnings) > 0

    def test_non_string_input_rejected(self) -> None:
        result = validate_clinical_text(12345)  # type: ignore[arg-type]
        assert result.is_valid is False
        assert any("str" in e for e in result.errors)

    def test_high_noise_ratio_warns(self) -> None:
        # Mostly non-alphanumeric characters
        text = "###$$$%%%^^^&&&***!!!" * 10
        result = validate_clinical_text(text, min_length=5)
        assert any("noise" in w.lower() for w in result.warnings)

    def test_few_words_warns(self) -> None:
        result = validate_clinical_text("Diabetes mellitus.", min_length=5)
        assert result.is_valid is True
        assert any("few words" in w.lower() for w in result.warnings)

    def test_oversized_bytes_rejected(self) -> None:
        # Create text that exceeds byte limit
        text = "A" * (11 * 1024 * 1024)  # 11 MB
        result = validate_clinical_text(text, max_raw_bytes=10 * 1024 * 1024)
        assert result.is_valid is False
        assert any("byte size" in e.lower() for e in result.errors)
