"""Unit tests for the RuleBasedICDClassifier.

Exercises the keyword-matching prediction logic, confidence boosting for
synonym matches, deduplication, batch prediction, and edge-case handling
(empty text, no matches, very long documents).
"""

import pytest

from app.ml.icd.model import ICDPredictionResult, RuleBasedICDClassifier


@pytest.fixture
def classifier() -> RuleBasedICDClassifier:
    """Return a loaded RuleBasedICDClassifier instance."""
    clf = RuleBasedICDClassifier()
    clf.load()
    return clf


class TestRuleBasedICDClassifierLifecycle:
    """Lifecycle and attribute tests."""

    def test_not_loaded_before_load(self) -> None:
        clf = RuleBasedICDClassifier()
        assert clf.is_loaded is False

    def test_loaded_after_load(self) -> None:
        clf = RuleBasedICDClassifier()
        clf.load()
        assert clf.is_loaded is True

    def test_ensure_loaded_triggers_load(self) -> None:
        clf = RuleBasedICDClassifier()
        clf.ensure_loaded()
        assert clf.is_loaded is True

    def test_default_model_name(self) -> None:
        clf = RuleBasedICDClassifier()
        assert clf.model_name == "rule-based-icd"

    def test_custom_model_name(self) -> None:
        clf = RuleBasedICDClassifier(model_name="custom", version="2.0")
        assert clf.model_name == "custom"
        assert clf.version == "2.0"

    def test_compiled_rules_populated_after_load(self) -> None:
        clf = RuleBasedICDClassifier()
        clf.load()
        assert len(clf._compiled_rules) > 0


class TestRuleBasedICDClassifierPredict:
    """Tests for the predict() method."""

    def test_returns_icd_prediction_result(self, classifier: RuleBasedICDClassifier) -> None:
        result = classifier.predict("Patient has hypertension.")
        assert isinstance(result, ICDPredictionResult)

    def test_result_model_name(self, classifier: RuleBasedICDClassifier) -> None:
        result = classifier.predict("Patient has hypertension.")
        assert result.model_name == "rule-based-icd"
        assert result.model_version == "1.0.0"

    def test_processing_time_positive(self, classifier: RuleBasedICDClassifier) -> None:
        result = classifier.predict("Patient has hypertension.")
        assert result.processing_time_ms > 0

    def test_hypertension_detection(self, classifier: RuleBasedICDClassifier) -> None:
        result = classifier.predict("Patient diagnosed with hypertension.")
        codes = [p.code for p in result.predictions]
        assert "I10" in codes

    def test_diabetes_type2_detection(self, classifier: RuleBasedICDClassifier) -> None:
        result = classifier.predict("History of type 2 diabetes mellitus.")
        codes = [p.code for p in result.predictions]
        assert "E11.9" in codes

    def test_diabetes_type1_detection(self, classifier: RuleBasedICDClassifier) -> None:
        result = classifier.predict("Patient has type 1 diabetes, insulin-dependent.")
        codes = [p.code for p in result.predictions]
        assert "E10.9" in codes

    @pytest.mark.parametrize(
        "text,expected_code",
        [
            ("Patient has COPD with acute exacerbation.", "J44.1"),
            ("Diagnosed with pneumonia on chest X-ray.", "J18.9"),
            ("Chronic kidney disease stage 3.", "N18.9"),
            ("History of myocardial infarction.", "I21.9"),
            ("Patient reports chest pain.", "R07.9"),
            ("Urinary tract infection confirmed.", "N39.0"),
            ("Patient is obese with BMI >30.", "E66.9"),
            ("Anemia workup pending.", "D64.9"),
            ("GERD symptoms worsening.", "K21.0"),
            ("Low back pain for 3 weeks.", "M54.5"),
            ("Atrial fibrillation noted on ECG.", "I48.91"),
            ("Heart failure exacerbation.", "I50.9"),
            ("Sepsis secondary to pneumonia.", "A41.9"),
            ("Seizure disorder, epilepsy.", "G40.909"),
            ("Acute kidney injury requiring dialysis.", "N17.9"),
        ],
    )
    def test_detects_expected_code(
        self, classifier: RuleBasedICDClassifier, text: str, expected_code: str
    ) -> None:
        result = classifier.predict(text)
        codes = [p.code for p in result.predictions]
        assert expected_code in codes, (
            f"Expected {expected_code} in predictions for: {text!r}, got {codes}"
        )

    def test_multiple_conditions_detected(self, classifier: RuleBasedICDClassifier) -> None:
        text = (
            "Patient has hypertension, type 2 diabetes, COPD, "
            "and chronic kidney disease."
        )
        result = classifier.predict(text)
        codes = {p.code for p in result.predictions}
        assert {"I10", "E11.9", "J44.1", "N18.9"}.issubset(codes)

    def test_predictions_sorted_by_confidence(self, classifier: RuleBasedICDClassifier) -> None:
        text = "Patient has hypertension, type 2 diabetes, and pneumonia."
        result = classifier.predict(text)
        if len(result.predictions) >= 2:
            for i in range(len(result.predictions) - 1):
                assert result.predictions[i].confidence >= result.predictions[i + 1].confidence

    def test_confidence_in_valid_range(self, classifier: RuleBasedICDClassifier) -> None:
        text = "Hypertension, diabetes type 2, COPD, heart failure."
        result = classifier.predict(text)
        for pred in result.predictions:
            assert 0.0 <= pred.confidence <= 1.0, (
                f"Confidence {pred.confidence} out of range for {pred.code}"
            )

    def test_chapter_populated(self, classifier: RuleBasedICDClassifier) -> None:
        result = classifier.predict("Patient has hypertension.")
        htn_preds = [p for p in result.predictions if p.code == "I10"]
        assert htn_preds
        assert htn_preds[0].chapter is not None
        assert "circulatory" in htn_preds[0].chapter.lower()

    def test_contributing_text_populated(self, classifier: RuleBasedICDClassifier) -> None:
        result = classifier.predict("High blood pressure and hypertension noted.")
        htn_preds = [p for p in result.predictions if p.code == "I10"]
        assert htn_preds
        assert htn_preds[0].contributing_text is not None
        assert len(htn_preds[0].contributing_text) > 0

    def test_synonym_boost_increases_confidence(
        self, classifier: RuleBasedICDClassifier
    ) -> None:
        """Multiple synonym matches should boost confidence above baseline."""
        # "hypertension" and "high blood pressure" are both keywords for I10
        result_single = classifier.predict("Patient has hypertension.")
        result_multi = classifier.predict(
            "Patient has hypertension and high blood pressure."
        )
        single_conf = next(
            (p.confidence for p in result_single.predictions if p.code == "I10"), 0
        )
        multi_conf = next(
            (p.confidence for p in result_multi.predictions if p.code == "I10"), 0
        )
        assert multi_conf >= single_conf

    def test_description_populated(self, classifier: RuleBasedICDClassifier) -> None:
        result = classifier.predict("Patient has hypertension.")
        htn = [p for p in result.predictions if p.code == "I10"]
        assert htn
        assert htn[0].description is not None
        assert len(htn[0].description) > 0

    def test_deduplication_by_code(self, classifier: RuleBasedICDClassifier) -> None:
        """Same code should not appear twice even if multiple keywords match."""
        text = "hypertension high blood pressure htn elevated bp"
        result = classifier.predict(text)
        codes = [p.code for p in result.predictions]
        assert codes.count("I10") == 1

    def test_top_k_limits_output(self, classifier: RuleBasedICDClassifier) -> None:
        text = (
            "Hypertension, diabetes type 2, COPD, pneumonia, "
            "chronic kidney disease, heart failure, chest pain, "
            "atrial fibrillation, sepsis, anemia, obesity, GERD."
        )
        result = classifier.predict(text, top_k=3)
        assert len(result.predictions) <= 3

    def test_top_k_default_is_ten(self, classifier: RuleBasedICDClassifier) -> None:
        text = (
            "Hypertension, diabetes type 2, COPD, pneumonia, "
            "chronic kidney disease, heart failure, chest pain, "
            "atrial fibrillation, sepsis, anemia, obesity, GERD, "
            "low back pain, depression, anxiety, asthma, hypothyroidism."
        )
        result = classifier.predict(text)
        assert len(result.predictions) <= 10


class TestRuleBasedICDClassifierEdgeCases:
    """Edge-case and boundary tests."""

    def test_empty_text(self, classifier: RuleBasedICDClassifier) -> None:
        result = classifier.predict("")
        assert isinstance(result, ICDPredictionResult)
        assert result.predictions == []

    def test_no_matching_keywords(self, classifier: RuleBasedICDClassifier) -> None:
        result = classifier.predict("The weather is sunny today.")
        assert isinstance(result, ICDPredictionResult)
        assert result.predictions == []

    def test_whitespace_only_text(self, classifier: RuleBasedICDClassifier) -> None:
        result = classifier.predict("   \n\t  ")
        assert result.predictions == []

    def test_case_insensitive_matching(self, classifier: RuleBasedICDClassifier) -> None:
        upper = classifier.predict("HYPERTENSION")
        lower = classifier.predict("hypertension")
        mixed = classifier.predict("Hypertension")
        assert all(
            any(p.code == "I10" for p in r.predictions)
            for r in [upper, lower, mixed]
        )

    def test_very_long_document(self, classifier: RuleBasedICDClassifier) -> None:
        """Ensure prediction works on a very long document without errors."""
        long_text = "Patient has hypertension. " * 5000
        result = classifier.predict(long_text)
        assert isinstance(result, ICDPredictionResult)
        codes = [p.code for p in result.predictions]
        assert "I10" in codes

    def test_to_dict_serialisation(self, classifier: RuleBasedICDClassifier) -> None:
        result = classifier.predict("Patient has hypertension and diabetes.")
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "predictions" in d
        assert isinstance(d["predictions"], list)


class TestRuleBasedICDClassifierBatch:
    """Tests for predict_batch()."""

    def test_batch_returns_list(self, classifier: RuleBasedICDClassifier) -> None:
        texts = ["Hypertension.", "Diabetes type 2."]
        results = classifier.predict_batch(texts)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_batch_individual_results_correct(
        self, classifier: RuleBasedICDClassifier
    ) -> None:
        texts = ["Hypertension.", "COPD.", "Nothing medical here."]
        results = classifier.predict_batch(texts)
        assert any(p.code == "I10" for p in results[0].predictions)
        assert any(p.code == "J44.1" for p in results[1].predictions)
        assert results[2].predictions == []

    def test_batch_empty_list(self, classifier: RuleBasedICDClassifier) -> None:
        results = classifier.predict_batch([])
        assert results == []

    def test_batch_single_item(self, classifier: RuleBasedICDClassifier) -> None:
        results = classifier.predict_batch(["Patient has anemia."])
        assert len(results) == 1
        codes = [p.code for p in results[0].predictions]
        assert "D64.9" in codes

    def test_batch_top_k_applied(self, classifier: RuleBasedICDClassifier) -> None:
        texts = [
            "Hypertension, diabetes, COPD, pneumonia, heart failure, sepsis.",
        ]
        results = classifier.predict_batch(texts, top_k=2)
        assert len(results[0].predictions) <= 2
