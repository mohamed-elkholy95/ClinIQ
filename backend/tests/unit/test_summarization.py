"""Unit tests for the clinical text summarization module."""

import pytest

from app.ml.summarization.model import ExtractiveSummarizer, SummarizationResult

SAMPLE_CLINICAL_NOTE = """
CHIEF COMPLAINT: Follow-up for type 2 diabetes mellitus.

HISTORY OF PRESENT ILLNESS:
Patient is a 55-year-old male with a history of type 2 diabetes mellitus,
hypertension, and hyperlipidemia presenting for routine follow-up.
He reports good compliance with medications including metformin 1000mg twice
daily and lisinopril 10mg daily. Blood glucose levels have been well controlled.
No symptoms of hypoglycemia. Patient denies chest pain, shortness of breath,
or peripheral edema.

PAST MEDICAL HISTORY:
- Type 2 diabetes mellitus diagnosed 8 years ago
- Hypertension controlled with lisinopril
- Hyperlipidemia on atorvastatin
- Obesity BMI 32

MEDICATIONS:
- Metformin 1000mg PO BID
- Lisinopril 10mg PO daily
- Atorvastatin 20mg PO daily
- Aspirin 81mg PO daily

ALLERGIES: NKDA

ASSESSMENT AND PLAN:
1. Type 2 DM - HbA1c 6.8%, well controlled. Continue current regimen.
2. Hypertension - BP 128/78 today, well controlled. Continue lisinopril.
3. Hyperlipidemia - LDL 89, at goal. Continue atorvastatin.
4. Obesity - Counselled on diet and regular exercise.

Follow up in 3 months or sooner if concerns arise.
"""


class TestSummarizationResult:
    """Tests for the SummarizationResult dataclass."""

    def test_creation(self):
        """Test creating a SummarizationResult."""
        result = SummarizationResult(
            summary="Patient has diabetes managed with metformin.",
            key_findings=["HbA1c 6.8%, well controlled."],
            detail_level="standard",
            processing_time_ms=50.0,
            model_name="extractive-textrank",
            model_version="1.0.0",
        )

        assert result.summary == "Patient has diabetes managed with metformin."
        assert result.detail_level == "standard"
        assert result.model_name == "extractive-textrank"

    def test_optional_defaults(self):
        """Test default values for optional fields."""
        result = SummarizationResult(
            summary="Summary text.",
            key_findings=[],
            detail_level="brief",
            processing_time_ms=10.0,
            model_name="test",
            model_version="1.0",
        )

        assert result.sentence_count_original == 0
        assert result.sentence_count_summary == 0
        assert result.metadata == {}

    def test_to_dict_keys(self):
        """Test that to_dict() includes all expected keys."""
        result = SummarizationResult(
            summary="A clinical summary.",
            key_findings=["Finding 1", "Finding 2"],
            detail_level="detailed",
            processing_time_ms=75.0,
            model_name="extractive",
            model_version="1.0.0",
            sentence_count_original=20,
            sentence_count_summary=5,
        )

        d = result.to_dict()

        expected_keys = {
            "summary",
            "key_findings",
            "detail_level",
            "processing_time_ms",
            "model_name",
            "model_version",
            "sentence_count_original",
            "sentence_count_summary",
            "metadata",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self):
        """Test that to_dict() values are correctly serialised."""
        result = SummarizationResult(
            summary="Test summary.",
            key_findings=["Key finding."],
            detail_level="brief",
            processing_time_ms=20.0,
            model_name="test-model",
            model_version="2.0.0",
            sentence_count_original=10,
            sentence_count_summary=2,
            metadata={"chunk_count": 1},
        )

        d = result.to_dict()

        assert d["summary"] == "Test summary."
        assert d["key_findings"] == ["Key finding."]
        assert d["detail_level"] == "brief"
        assert d["processing_time_ms"] == 20.0
        assert d["sentence_count_original"] == 10
        assert d["sentence_count_summary"] == 2
        assert d["metadata"] == {"chunk_count": 1}

    def test_detail_level_values(self):
        """Test that all valid detail_level values are accepted."""
        for level in ("brief", "standard", "detailed"):
            result = SummarizationResult(
                summary="Test.",
                key_findings=[],
                detail_level=level,
                processing_time_ms=0.0,
                model_name="test",
                model_version="1.0",
            )
            assert result.detail_level == level


class TestExtractiveSummarizer:
    """Tests for ExtractiveSummarizer."""

    @pytest.fixture
    def summarizer(self) -> ExtractiveSummarizer:
        """Create a loaded ExtractiveSummarizer instance."""
        s = ExtractiveSummarizer()
        s.load()
        return s

    def test_load_sets_is_loaded(self):
        """Test that load() sets is_loaded to True."""
        s = ExtractiveSummarizer()
        assert s.is_loaded is False
        s.load()
        assert s.is_loaded is True

    def test_ensure_loaded_triggers_load(self):
        """Test that ensure_loaded() calls load when not loaded."""
        s = ExtractiveSummarizer()
        s.ensure_loaded()
        assert s.is_loaded is True

    def test_summarize_returns_summarization_result(
        self, summarizer: ExtractiveSummarizer
    ):
        """Test that summarize() returns a SummarizationResult."""
        result = summarizer.summarize(SAMPLE_CLINICAL_NOTE)
        assert isinstance(result, SummarizationResult)

    def test_summarize_returns_non_empty_summary(
        self, summarizer: ExtractiveSummarizer
    ):
        """Test that the summary text is non-empty for a valid input."""
        result = summarizer.summarize(SAMPLE_CLINICAL_NOTE)
        assert len(result.summary.strip()) > 0

    def test_summarize_default_detail_level_is_standard(
        self, summarizer: ExtractiveSummarizer
    ):
        """Test that the default detail_level is 'standard'."""
        result = summarizer.summarize(SAMPLE_CLINICAL_NOTE)
        assert result.detail_level == "standard"

    @pytest.mark.parametrize("detail_level", ["brief", "standard", "detailed"])
    def test_summarize_all_detail_levels(
        self, summarizer: ExtractiveSummarizer, detail_level: str
    ):
        """Test that all detail levels produce valid results."""
        result = summarizer.summarize(SAMPLE_CLINICAL_NOTE, detail_level=detail_level)

        assert isinstance(result, SummarizationResult)
        assert result.detail_level == detail_level
        assert len(result.summary.strip()) > 0

    def test_brief_shorter_than_detailed(self, summarizer: ExtractiveSummarizer):
        """Test that 'brief' summary is no longer than 'detailed' summary."""
        brief = summarizer.summarize(SAMPLE_CLINICAL_NOTE, detail_level="brief")
        detailed = summarizer.summarize(SAMPLE_CLINICAL_NOTE, detail_level="detailed")

        brief_count = brief.sentence_count_summary
        detailed_count = detailed.sentence_count_summary

        assert brief_count <= detailed_count, (
            f"Brief ({brief_count} sentences) should be <= detailed ({detailed_count} sentences)"
        )

    def test_sentence_counts_set(self, summarizer: ExtractiveSummarizer):
        """Test that sentence_count_original and sentence_count_summary are set."""
        result = summarizer.summarize(SAMPLE_CLINICAL_NOTE)
        assert result.sentence_count_original > 0
        assert result.sentence_count_summary > 0

    def test_summary_shorter_than_original(self, summarizer: ExtractiveSummarizer):
        """Test that the summary is shorter than the original text."""
        result = summarizer.summarize(SAMPLE_CLINICAL_NOTE, detail_level="brief")
        assert len(result.summary) < len(SAMPLE_CLINICAL_NOTE)

    def test_key_findings_is_list(self, summarizer: ExtractiveSummarizer):
        """Test that key_findings is a list."""
        result = summarizer.summarize(SAMPLE_CLINICAL_NOTE)
        assert isinstance(result.key_findings, list)

    def test_key_findings_are_strings(self, summarizer: ExtractiveSummarizer):
        """Test that key_findings entries are strings."""
        result = summarizer.summarize(SAMPLE_CLINICAL_NOTE)
        for finding in result.key_findings:
            assert isinstance(finding, str)

    def test_processing_time_positive(self, summarizer: ExtractiveSummarizer):
        """Test that processing_time_ms is a positive number."""
        result = summarizer.summarize(SAMPLE_CLINICAL_NOTE)
        assert result.processing_time_ms > 0.0

    def test_model_name_in_result(self, summarizer: ExtractiveSummarizer):
        """Test that model_name is included in the result."""
        result = summarizer.summarize(SAMPLE_CLINICAL_NOTE)
        assert result.model_name == "extractive-textrank"

    def test_model_version_in_result(self, summarizer: ExtractiveSummarizer):
        """Test that model_version is included in the result."""
        result = summarizer.summarize(SAMPLE_CLINICAL_NOTE)
        assert isinstance(result.model_version, str)
        assert len(result.model_version) > 0

    def test_single_sentence_text(self, summarizer: ExtractiveSummarizer):
        """Test behaviour with a very short single-sentence text."""
        text = "Patient presents for routine follow-up appointment today."
        result = summarizer.summarize(text)

        assert isinstance(result, SummarizationResult)
        assert len(result.summary) > 0

    def test_empty_text_returns_result(self, summarizer: ExtractiveSummarizer):
        """Test that empty text does not raise an exception."""
        result = summarizer.summarize("")
        assert isinstance(result, SummarizationResult)

    def test_custom_model_name(self):
        """Test that a custom model_name is preserved in results."""
        s = ExtractiveSummarizer(model_name="custom-summarizer", version="2.0.0")
        s.load()
        result = s.summarize(SAMPLE_CLINICAL_NOTE)
        assert result.model_name == "custom-summarizer"

    def test_summarize_preserves_clinical_terms(
        self, summarizer: ExtractiveSummarizer
    ):
        """Test that key clinical terms appear in the summary of a clinical note."""
        result = summarizer.summarize(SAMPLE_CLINICAL_NOTE, detail_level="detailed")

        # With detailed level, at least some clinical terms should survive
        clinical_terms = ["diabetes", "metformin", "hypertension", "lisinopril"]
        summary_lower = result.summary.lower()
        found = [term for term in clinical_terms if term in summary_lower]

        assert len(found) > 0, (
            f"None of {clinical_terms} found in summary: {result.summary!r}"
        )

    def test_sentence_count_summary_lte_original(
        self, summarizer: ExtractiveSummarizer
    ):
        """Test that summary sentence count never exceeds original sentence count."""
        result = summarizer.summarize(SAMPLE_CLINICAL_NOTE)
        assert result.sentence_count_summary <= result.sentence_count_original

    def test_default_damping_factor(self):
        """Test that the default PageRank damping factor is 0.85."""
        s = ExtractiveSummarizer()
        assert s.damping == 0.85

    def test_to_dict_is_serialisable(self, summarizer: ExtractiveSummarizer):
        """Test that to_dict() produces a JSON-serialisable structure."""
        import json

        result = summarizer.summarize(SAMPLE_CLINICAL_NOTE)
        d = result.to_dict()

        # Should not raise
        serialised = json.dumps(d)
        assert len(serialised) > 0
