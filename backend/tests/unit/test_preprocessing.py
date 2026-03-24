"""Unit tests for text preprocessing."""

import pytest

from app.ml.utils.text_preprocessing import (
    ClinicalTextPreprocessor,
    PreprocessingConfig,
    TextSection,
    preprocess_clinical_text,
)


class TestClinicalTextPreprocessor:
    """Tests for ClinicalTextPreprocessor."""

    @pytest.fixture
    def preprocessor(self) -> ClinicalTextPreprocessor:
        """Create preprocessor instance."""
        return ClinicalTextPreprocessor()

    @pytest.fixture
    def sample_note(self) -> str:
        """Sample clinical note."""
        return """
        CHIEF COMPLAINT: Chest pain

        HISTORY OF PRESENT ILLNESS:
        Patient is a 45-year-old male presenting with substernal chest pain
        that started 2 hours ago. Pain is described as pressure-like, 7/10 severity.
        No radiation to arm or jaw. No associated shortness of breath or diaphoresis.

        PAST MEDICAL HISTORY:
        Hypertension
        Hyperlipidemia

        MEDICATIONS:
        Aspirin 81mg daily
        Lisinopril 10mg daily

        ASSESSMENT AND PLAN:
        Chest pain - rule out acute coronary syndrome
        ECG ordered
        Troponins ordered
        """

    def test_preprocess_normalizes_whitespace(self, preprocessor: ClinicalTextPreprocessor):
        """Test whitespace normalization."""
        text = "Hello   world\t\ttest\n\n\n\nmultiple"
        result = preprocessor.preprocess(text)

        assert "   " not in result
        assert "\t\t" not in result
        assert "\n\n\n\n" not in result

    def test_preprocess_preserves_structure(self, preprocessor: ClinicalTextPreprocessor):
        """Test that preprocessing preserves document structure."""
        text = "Line 1\n\nLine 2\n\nLine 3"
        result = preprocessor.preprocess(text)

        assert "Line 1" in result
        assert "Line 2" in result
        assert "Line 3" in result

    def test_detect_sections(self, preprocessor: ClinicalTextPreprocessor, sample_note: str):
        """Test section detection."""
        sections = preprocessor.detect_sections(sample_note)

        assert len(sections) > 0

        section_names = {s.name for s in sections}
        assert "chief_complaint" in section_names or "hpi" in section_names

    def test_segment_sentences(self, preprocessor: ClinicalTextPreprocessor):
        """Test sentence segmentation."""
        text = "This is sentence 1. This is sentence 2. This is sentence 3."
        sentences = preprocessor.segment_sentences(text)

        assert len(sentences) == 3
        assert sentences[0].startswith("This is sentence 1")

    def test_segment_sentences_preserves_decimals(self, preprocessor: ClinicalTextPreprocessor):
        """Test that decimal numbers are not split."""
        text = "The value is 3.14 and the dose is 100.5 mg."
        sentences = preprocessor.segment_sentences(text)

        # Should be single sentence
        assert len(sentences) == 1
        assert "3.14" in sentences[0]
        assert "100.5" in sentences[0]

    def test_expand_abbreviations(self, preprocessor: ClinicalTextPreprocessor):
        """Test medical abbreviation expansion."""
        text = "Pt is a 55 y/o male with h/o DM2 and HTN."
        expanded = preprocessor.expand_abbreviations(text)

        assert "patient" in expanded.lower()
        assert "year old" in expanded.lower()
        assert "history of" in expanded.lower()
        assert "diabetes mellitus" in expanded.lower()
        assert "hypertension" in expanded.lower()

    def test_extract_hash(self, preprocessor: ClinicalTextPreprocessor):
        """Test text hash extraction."""
        text1 = "Hello world"
        text2 = "Hello world"
        text3 = "Different text"

        hash1 = preprocessor.extract_hash(text1)
        hash2 = preprocessor.extract_hash(text2)
        hash3 = preprocessor.extract_hash(text3)

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64  # SHA256 hex length

    def test_get_text_stats(self, preprocessor: ClinicalTextPreprocessor, sample_note: str):
        """Test text statistics extraction."""
        stats = preprocessor.get_text_stats(sample_note)

        assert "char_count" in stats
        assert "word_count" in stats
        assert "sentence_count" in stats
        assert "section_count" in stats
        assert stats["word_count"] > 0
        assert stats["sentence_count"] > 0

    def test_clean_text(self, preprocessor: ClinicalTextPreprocessor):
        """Test text cleaning."""
        text = "Hello\u2014world\u2013test\u201cquoted\u201d"
        cleaned = preprocessor.clean_text(text)

        assert "\u2014" not in cleaned  # Em dash
        assert "\u2013" not in cleaned  # En dash
        assert "\u201c" not in cleaned  # Left quote
        assert "\u201d" not in cleaned  # Right quote


class TestPreprocessingConfig:
    """Tests for PreprocessingConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PreprocessingConfig()

        assert config.normalize_whitespace is True
        assert config.remove_extra_newlines is True
        assert config.detect_sections is True
        assert config.max_document_length == 100000

    def test_custom_config(self):
        """Test custom configuration."""
        config = PreprocessingConfig(
            normalize_whitespace=False,
            max_document_length=50000,
        )

        assert config.normalize_whitespace is False
        assert config.max_document_length == 50000


class TestTextSection:
    """Tests for TextSection dataclass."""

    def test_text_section_creation(self):
        """Test TextSection creation."""
        section = TextSection(
            name="hpi",
            content="Patient has diabetes",
            start_char=0,
            end_char=20,
            confidence=0.9,
        )

        assert section.name == "hpi"
        assert section.content == "Patient has diabetes"
        assert section.confidence == 0.9


def test_preprocess_clinical_text_convenience():
    """Test the convenience function."""
    text = "  Test   text  \n\n\n"
    result = preprocess_clinical_text(text)

    assert "  " not in result
    assert result.startswith("Test")
