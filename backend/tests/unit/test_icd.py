"""Unit tests for the ICD-10 model module."""

import pytest

from app.ml.icd.model import (
    ICD10_CHAPTERS,
    ICDCodePrediction,
    ICDPredictionResult,
    get_chapter_for_code,
)


class TestICDCodePrediction:
    """Tests for the ICDCodePrediction dataclass."""

    def test_creation_required_fields(self):
        """Test creating ICDCodePrediction with required fields."""
        pred = ICDCodePrediction(
            code="E11.9",
            description="Type 2 diabetes mellitus without complications",
            confidence=0.85,
        )

        assert pred.code == "E11.9"
        assert pred.description == "Type 2 diabetes mellitus without complications"
        assert pred.confidence == 0.85

    def test_optional_field_defaults(self):
        """Test that optional fields default to None."""
        pred = ICDCodePrediction(
            code="I10",
            description="Essential (primary) hypertension",
            confidence=0.90,
        )

        assert pred.chapter is None
        assert pred.category is None
        assert pred.contributing_text is None

    def test_to_dict_returns_all_keys(self):
        """Test that to_dict() includes all expected keys."""
        pred = ICDCodePrediction(
            code="J18.9",
            description="Pneumonia, unspecified organism",
            confidence=0.75,
            chapter="Diseases of the respiratory system",
        )

        result = pred.to_dict()

        expected_keys = {
            "code",
            "description",
            "confidence",
            "chapter",
            "category",
            "contributing_text",
        }
        assert set(result.keys()) == expected_keys

    def test_to_dict_values_correct(self):
        """Test that to_dict() contains correct values."""
        pred = ICDCodePrediction(
            code="E11.9",
            description="Type 2 diabetes mellitus without complications",
            confidence=0.85,
            chapter="Endocrine, nutritional and metabolic diseases",
            category="E11",
            contributing_text=["diabetes", "metformin"],
        )

        result = pred.to_dict()

        assert result["code"] == "E11.9"
        assert result["description"] == "Type 2 diabetes mellitus without complications"
        assert result["confidence"] == 0.85
        assert result["chapter"] == "Endocrine, nutritional and metabolic diseases"
        assert result["category"] == "E11"
        assert result["contributing_text"] == ["diabetes", "metformin"]

    def test_confidence_can_be_zero(self):
        """Test that confidence of 0.0 is valid."""
        pred = ICDCodePrediction(code="Z00.00", description="Encounter for exam", confidence=0.0)
        assert pred.confidence == 0.0

    def test_confidence_can_be_one(self):
        """Test that confidence of 1.0 is valid."""
        pred = ICDCodePrediction(code="I10", description="Hypertension", confidence=1.0)
        assert pred.confidence == 1.0


class TestICDPredictionResult:
    """Tests for the ICDPredictionResult dataclass."""

    @pytest.fixture
    def sample_predictions(self) -> list[ICDCodePrediction]:
        """Provide a list of sample ICD predictions."""
        return [
            ICDCodePrediction(code="E11.9", description="Type 2 DM", confidence=0.85),
            ICDCodePrediction(code="I10", description="Hypertension", confidence=0.75),
            ICDCodePrediction(code="E78.5", description="Hyperlipidemia", confidence=0.65),
            ICDCodePrediction(code="Z68.35", description="BMI 35.0-35.9", confidence=0.55),
            ICDCodePrediction(code="J45.20", description="Mild intermittent asthma", confidence=0.40),
        ]

    @pytest.fixture
    def prediction_result(
        self, sample_predictions: list[ICDCodePrediction]
    ) -> ICDPredictionResult:
        """Provide a complete ICDPredictionResult fixture."""
        return ICDPredictionResult(
            predictions=sample_predictions,
            processing_time_ms=42.5,
            model_name="test-icd-model",
            model_version="1.0.0",
        )

    def test_creation(self, prediction_result: ICDPredictionResult):
        """Test that ICDPredictionResult is created correctly."""
        assert prediction_result.model_name == "test-icd-model"
        assert prediction_result.model_version == "1.0.0"
        assert prediction_result.processing_time_ms == 42.5
        assert len(prediction_result.predictions) == 5

    def test_optional_document_summary_defaults_to_none(self):
        """Test that document_summary defaults to None."""
        result = ICDPredictionResult(
            predictions=[],
            processing_time_ms=10.0,
            model_name="test",
            model_version="1.0",
        )
        assert result.document_summary is None

    def test_top_k_returns_correct_count(
        self, prediction_result: ICDPredictionResult
    ):
        """Test that top_k() returns exactly k predictions."""
        top3 = prediction_result.top_k(k=3)
        assert len(top3) == 3

    def test_top_k_sorted_by_confidence_descending(
        self, prediction_result: ICDPredictionResult
    ):
        """Test that top_k() results are sorted highest confidence first."""
        top3 = prediction_result.top_k(k=3)

        for i in range(len(top3) - 1):
            assert top3[i].confidence >= top3[i + 1].confidence

    def test_top_k_highest_confidence_selected(
        self, prediction_result: ICDPredictionResult
    ):
        """Test that top_k() returns the highest-confidence predictions."""
        top1 = prediction_result.top_k(k=1)
        assert len(top1) == 1
        assert top1[0].code == "E11.9"
        assert top1[0].confidence == 0.85

    def test_top_k_exceeds_available(
        self, prediction_result: ICDPredictionResult
    ):
        """Test that top_k() with k > len(predictions) returns all predictions."""
        top100 = prediction_result.top_k(k=100)
        assert len(top100) == 5

    def test_top_k_zero(self, prediction_result: ICDPredictionResult):
        """Test that top_k(0) returns empty list."""
        result = prediction_result.top_k(k=0)
        assert result == []

    def test_to_dict_structure(self, prediction_result: ICDPredictionResult):
        """Test the structure of to_dict() output."""
        result = prediction_result.to_dict()

        expected_keys = {
            "predictions",
            "processing_time_ms",
            "model_name",
            "model_version",
            "document_summary",
        }
        assert set(result.keys()) == expected_keys

    def test_to_dict_predictions_are_dicts(
        self, prediction_result: ICDPredictionResult
    ):
        """Test that predictions in to_dict() output are serialised as dicts."""
        result = prediction_result.to_dict()

        for pred in result["predictions"]:
            assert isinstance(pred, dict)
            assert "code" in pred
            assert "confidence" in pred

    def test_to_dict_processing_time(
        self, prediction_result: ICDPredictionResult
    ):
        """Test that processing_time_ms is preserved in to_dict()."""
        result = prediction_result.to_dict()
        assert result["processing_time_ms"] == 42.5

    def test_empty_predictions(self):
        """Test ICDPredictionResult with no predictions."""
        result = ICDPredictionResult(
            predictions=[],
            processing_time_ms=5.0,
            model_name="test",
            model_version="1.0",
        )

        assert result.top_k(k=5) == []
        assert result.to_dict()["predictions"] == []


class TestICD10Chapters:
    """Tests for the ICD10_CHAPTERS mapping."""

    def test_chapters_is_dict(self):
        """Test that ICD10_CHAPTERS is a dict."""
        assert isinstance(ICD10_CHAPTERS, dict)

    def test_chapters_not_empty(self):
        """Test that ICD10_CHAPTERS has entries."""
        assert len(ICD10_CHAPTERS) > 0

    def test_common_chapters_present(self):
        """Test that well-known ICD-10 chapter ranges are present."""
        expected_ranges = [
            "E00-E89",  # Endocrine
            "I00-I99",  # Circulatory
            "J00-J99",  # Respiratory
            "K00-K95",  # Digestive
            "C00-D49",  # Neoplasms
        ]
        for chapter_range in expected_ranges:
            assert chapter_range in ICD10_CHAPTERS, (
                f"Expected chapter range {chapter_range!r} not found in ICD10_CHAPTERS"
            )

    def test_chapter_values_are_strings(self):
        """Test that all chapter descriptions are non-empty strings."""
        for chapter_range, description in ICD10_CHAPTERS.items():
            assert isinstance(description, str), (
                f"Description for {chapter_range} is not a string"
            )
            assert len(description) > 0, (
                f"Description for {chapter_range} is empty"
            )

    def test_chapter_keys_format(self):
        """Test that chapter keys follow expected format (e.g. 'A00-B99')."""
        import re

        pattern = re.compile(r"^[A-Z]\d+-.+$")
        for key in ICD10_CHAPTERS:
            assert pattern.match(key), f"Chapter key {key!r} does not match expected format"


class TestGetChapterForCode:
    """Tests for the get_chapter_for_code() function."""

    @pytest.mark.parametrize(
        "code,expected_chapter_fragment",
        [
            ("E11.9", "Endocrine"),
            ("I10", "circulatory"),
            ("J18.9", "respiratory"),
            ("K21.0", "digestive"),
            ("C50.9", "Neoplasm"),
            ("A41.9", "infectious"),
            ("B34.9", "infectious"),
            ("F32.9", "Mental"),
            ("G35", "nervous"),
            ("M79.3", "musculoskeletal"),
            ("R06.0", "Symptoms"),
            ("Z00.00", "health status"),
        ],
    )
    def test_returns_correct_chapter(self, code: str, expected_chapter_fragment: str):
        """Parametrised test that each code maps to the correct chapter."""
        chapter = get_chapter_for_code(code)
        assert chapter is not None, f"Expected a chapter for code {code!r}, got None"
        assert expected_chapter_fragment.lower() in chapter.lower(), (
            f"Expected '{expected_chapter_fragment}' in chapter for code {code!r}, "
            f"got: {chapter!r}"
        )

    def test_returns_none_for_unknown_prefix(self):
        """Test that an unrecognised code prefix returns None."""
        # Use a prefix that isn't in the chapter ranges mapping
        result = get_chapter_for_code("9XX.0")
        assert result is None

    def test_returns_none_for_empty_string(self):
        """Test that an empty string returns None."""
        result = get_chapter_for_code("")
        assert result is None

    def test_case_insensitive_lookup(self):
        """Test that lowercase code is handled the same as uppercase."""
        upper = get_chapter_for_code("E11.9")
        lower = get_chapter_for_code("e11.9")
        assert upper == lower

    def test_returns_string(self):
        """Test that a valid code always returns a string."""
        result = get_chapter_for_code("I10")
        assert isinstance(result, str)

    def test_circulatory_chapter_codes(self):
        """Test multiple codes from the circulatory chapter."""
        codes = ["I10", "I21.3", "I50.9", "I48.0"]
        for code in codes:
            chapter = get_chapter_for_code(code)
            assert chapter is not None
            assert "circulatory" in chapter.lower(), (
                f"Expected circulatory chapter for {code!r}, got {chapter!r}"
            )

    def test_endocrine_chapter_codes(self):
        """Test multiple codes from the endocrine chapter."""
        codes = ["E11.9", "E10.9", "E78.5", "E03.9"]
        for code in codes:
            chapter = get_chapter_for_code(code)
            assert chapter is not None
            assert "endocrine" in chapter.lower(), (
                f"Expected endocrine chapter for {code!r}, got {chapter!r}"
            )
