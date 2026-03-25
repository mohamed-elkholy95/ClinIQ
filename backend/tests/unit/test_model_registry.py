"""Unit tests for the singleton model registry.

Validates lazy loading, caching, thread safety, and reset behaviour
of the central model registry used by API route handlers.
"""

import threading

import pytest

from app.services.model_registry import (
    get_icd_model,
    get_ner_model,
    get_risk_scorer,
    get_summarizer,
    reset_all,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the registry before and after every test."""
    reset_all()
    yield
    reset_all()


# ------------------------------------------------------------------
# Lazy loading
# ------------------------------------------------------------------

class TestLazyLoading:
    """Models should only be loaded on first access."""

    def test_ner_model_returns_loaded_instance(self):
        model = get_ner_model()
        assert model is not None
        assert model.is_loaded

    def test_icd_model_returns_loaded_instance(self):
        model = get_icd_model()
        assert model is not None
        assert model.is_loaded

    def test_summarizer_returns_loaded_instance(self):
        model = get_summarizer()
        assert model is not None
        assert model.is_loaded

    def test_risk_scorer_returns_loaded_instance(self):
        model = get_risk_scorer()
        assert model is not None
        assert model.is_loaded


# ------------------------------------------------------------------
# Caching (singleton)
# ------------------------------------------------------------------

class TestCaching:
    """Repeated calls must return the same instance."""

    def test_ner_model_cached(self):
        m1 = get_ner_model()
        m2 = get_ner_model()
        assert m1 is m2

    def test_icd_model_cached(self):
        m1 = get_icd_model()
        m2 = get_icd_model()
        assert m1 is m2

    def test_summarizer_cached(self):
        m1 = get_summarizer()
        m2 = get_summarizer()
        assert m1 is m2

    def test_risk_scorer_cached(self):
        m1 = get_risk_scorer()
        m2 = get_risk_scorer()
        assert m1 is m2


# ------------------------------------------------------------------
# Reset
# ------------------------------------------------------------------

class TestReset:
    """reset_all() must discard all cached models."""

    def test_reset_clears_ner(self):
        m1 = get_ner_model()
        reset_all()
        m2 = get_ner_model()
        assert m1 is not m2

    def test_reset_clears_all_models(self):
        ner = get_ner_model()
        icd = get_icd_model()
        summ = get_summarizer()
        risk = get_risk_scorer()
        reset_all()
        assert get_ner_model() is not ner
        assert get_icd_model() is not icd
        assert get_summarizer() is not summ
        assert get_risk_scorer() is not risk


# ------------------------------------------------------------------
# Thread safety
# ------------------------------------------------------------------

class TestThreadSafety:
    """Concurrent access must not create duplicate model instances."""

    def test_concurrent_ner_access_returns_same_instance(self):
        results: list = [None] * 10
        barrier = threading.Barrier(10)

        def _fetch(idx: int) -> None:
            barrier.wait()
            results[idx] = get_ner_model()

        threads = [threading.Thread(target=_fetch, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same instance
        assert all(r is results[0] for r in results)


# ------------------------------------------------------------------
# Functional smoke tests
# ------------------------------------------------------------------

class TestFunctionalSmoke:
    """Quick checks that loaded models can actually run inference."""

    def test_ner_model_extracts_medication(self):
        model = get_ner_model()
        entities = model.extract_entities("Patient takes aspirin 81mg daily.")
        med_texts = [e.text.lower() for e in entities]
        assert any("aspirin" in t for t in med_texts)

    def test_icd_model_returns_predictions(self):
        model = get_icd_model()
        result = model.predict("Patient diagnosed with type 2 diabetes mellitus.")
        assert result is not None
        assert hasattr(result, "predictions")

    def test_summarizer_returns_text(self):
        model = get_summarizer()
        text = (
            "Patient presents with chest pain radiating to the left arm. "
            "ECG shows ST elevation. Troponin levels elevated. "
            "Started on heparin drip and aspirin. Cardiology consulted."
        )
        result = model.summarize(text)
        assert result is not None

    def test_risk_scorer_returns_assessment(self):
        model = get_risk_scorer()
        result = model.assess_risk(
            "Patient has diabetes, hypertension, and chronic kidney disease."
        )
        assert result is not None
