"""Tests for the evaluation REST API endpoints.

Covers all 7 endpoints: classification, agreement, NER, ROUGE,
ICD, AUPRC, and metrics catalogue.
"""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client():
    """Async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# =========================================================================
# POST /evaluate/classification
# =========================================================================


class TestClassificationEval:
    """Tests for POST /api/v1/evaluate/classification."""

    @pytest.mark.anyio
    async def test_basic_mcc(self, client: AsyncClient) -> None:
        """Compute MCC for a simple case."""
        resp = await client.post(
            "/api/v1/evaluate/classification",
            json={"y_true": [1, 0, 1, 0], "y_pred": [1, 0, 1, 0]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["mcc"] == pytest.approx(1.0)
        assert data["tp"] == 2
        assert data["tn"] == 2

    @pytest.mark.anyio
    async def test_with_calibration(self, client: AsyncClient) -> None:
        """Include y_prob to get calibration metrics."""
        resp = await client.post(
            "/api/v1/evaluate/classification",
            json={
                "y_true": [1, 0, 1, 0],
                "y_pred": [1, 0, 1, 0],
                "y_prob": [0.9, 0.1, 0.8, 0.2],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["calibration"] is not None
        assert "expected_calibration_error" in data["calibration"]
        assert "brier_score" in data["calibration"]

    @pytest.mark.anyio
    async def test_mismatched_lengths(self, client: AsyncClient) -> None:
        """Mismatched y_true/y_pred should return 400."""
        resp = await client.post(
            "/api/v1/evaluate/classification",
            json={"y_true": [1, 0], "y_pred": [1]},
        )
        assert resp.status_code == 400

    @pytest.mark.anyio
    async def test_no_calibration_without_prob(self, client: AsyncClient) -> None:
        """Without y_prob, calibration should be null."""
        resp = await client.post(
            "/api/v1/evaluate/classification",
            json={"y_true": [1, 0], "y_pred": [1, 0]},
        )
        data = resp.json()
        assert data["calibration"] is None

    @pytest.mark.anyio
    async def test_processing_time(self, client: AsyncClient) -> None:
        """Response should include processing_time_ms."""
        resp = await client.post(
            "/api/v1/evaluate/classification",
            json={"y_true": [1, 0], "y_pred": [1, 0]},
        )
        data = resp.json()
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] >= 0


# =========================================================================
# POST /evaluate/agreement
# =========================================================================


class TestAgreementEval:
    """Tests for POST /api/v1/evaluate/agreement."""

    @pytest.mark.anyio
    async def test_perfect_kappa(self, client: AsyncClient) -> None:
        """Identical raters should yield kappa = 1.0."""
        resp = await client.post(
            "/api/v1/evaluate/agreement",
            json={"rater_a": ["A", "B", "A"], "rater_b": ["A", "B", "A"]},
        )
        assert resp.status_code == 200
        assert resp.json()["kappa"] == pytest.approx(1.0)

    @pytest.mark.anyio
    async def test_mismatched_lengths(self, client: AsyncClient) -> None:
        """Mismatched rater lengths should return 400."""
        resp = await client.post(
            "/api/v1/evaluate/agreement",
            json={"rater_a": ["A"], "rater_b": ["A", "B"]},
        )
        assert resp.status_code == 400

    @pytest.mark.anyio
    async def test_response_fields(self, client: AsyncClient) -> None:
        """Response should have all kappa fields."""
        resp = await client.post(
            "/api/v1/evaluate/agreement",
            json={"rater_a": ["A", "B"], "rater_b": ["A", "A"]},
        )
        data = resp.json()
        assert "kappa" in data
        assert "observed_agreement" in data
        assert "expected_agreement" in data
        assert "n_items" in data


# =========================================================================
# POST /evaluate/ner
# =========================================================================


class TestNEREval:
    """Tests for POST /api/v1/evaluate/ner."""

    @pytest.mark.anyio
    async def test_exact_match(self, client: AsyncClient) -> None:
        """Exact entity matches should yield F1 = 1.0."""
        entities = [
            {"entity_type": "DISEASE", "start": 0, "end": 10},
            {"entity_type": "DRUG", "start": 20, "end": 30},
        ]
        resp = await client.post(
            "/api/v1/evaluate/ner",
            json={"gold_entities": entities, "pred_entities": entities},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["exact_f1"] == pytest.approx(1.0)
        assert data["n_exact_matches"] == 2

    @pytest.mark.anyio
    async def test_partial_match(self, client: AsyncClient) -> None:
        """Overlapping spans should get partial credit."""
        resp = await client.post(
            "/api/v1/evaluate/ner",
            json={
                "gold_entities": [{"entity_type": "DISEASE", "start": 0, "end": 20}],
                "pred_entities": [{"entity_type": "DISEASE", "start": 5, "end": 25}],
            },
        )
        data = resp.json()
        assert data["partial_f1"] > 0
        assert data["n_partial_matches"] >= 1

    @pytest.mark.anyio
    async def test_no_entities(self, client: AsyncClient) -> None:
        """Both empty → perfect scores."""
        resp = await client.post(
            "/api/v1/evaluate/ner",
            json={"gold_entities": [], "pred_entities": []},
        )
        data = resp.json()
        assert data["exact_f1"] == 1.0

    @pytest.mark.anyio
    async def test_overlap_threshold(self, client: AsyncClient) -> None:
        """Custom threshold should filter low-overlap matches."""
        resp = await client.post(
            "/api/v1/evaluate/ner",
            json={
                "gold_entities": [{"entity_type": "DRUG", "start": 0, "end": 100}],
                "pred_entities": [{"entity_type": "DRUG", "start": 95, "end": 105}],
                "overlap_threshold": 0.5,
            },
        )
        data = resp.json()
        assert data["n_partial_matches"] == 0


# =========================================================================
# POST /evaluate/rouge
# =========================================================================


class TestROUGEEval:
    """Tests for POST /api/v1/evaluate/rouge."""

    @pytest.mark.anyio
    async def test_identical_texts(self, client: AsyncClient) -> None:
        """Identical texts should yield F1 = 1.0."""
        text = "Patient has diabetes and hypertension"
        resp = await client.post(
            "/api/v1/evaluate/rouge",
            json={"reference": text, "hypothesis": text},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["rouge1"]["f1"] == pytest.approx(1.0)

    @pytest.mark.anyio
    async def test_length_ratio(self, client: AsyncClient) -> None:
        """Length ratio should be computed correctly."""
        resp = await client.post(
            "/api/v1/evaluate/rouge",
            json={
                "reference": "a b c d e",
                "hypothesis": "a b",
            },
        )
        data = resp.json()
        assert data["reference_length"] == 5
        assert data["hypothesis_length"] == 2
        assert data["length_ratio"] == pytest.approx(0.4)

    @pytest.mark.anyio
    async def test_full_prf(self, client: AsyncClient) -> None:
        """Each ROUGE variant should have precision, recall, f1."""
        resp = await client.post(
            "/api/v1/evaluate/rouge",
            json={"reference": "a b c", "hypothesis": "a b d"},
        )
        data = resp.json()
        for key in ["rouge1", "rouge2", "rougeL"]:
            assert "precision" in data[key]
            assert "recall" in data[key]
            assert "f1" in data[key]


# =========================================================================
# POST /evaluate/icd
# =========================================================================


class TestICDEval:
    """Tests for POST /api/v1/evaluate/icd."""

    @pytest.mark.anyio
    async def test_perfect(self, client: AsyncClient) -> None:
        """Exact code matches."""
        codes = ["E11.65", "I10"]
        resp = await client.post(
            "/api/v1/evaluate/icd",
            json={"gold_codes": codes, "pred_codes": codes},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["full_code_accuracy"] == 1.0

    @pytest.mark.anyio
    async def test_hierarchical(self, client: AsyncClient) -> None:
        """Block match without full match."""
        resp = await client.post(
            "/api/v1/evaluate/icd",
            json={"gold_codes": ["E11.65"], "pred_codes": ["E11.9"]},
        )
        data = resp.json()
        assert data["full_code_accuracy"] == 0.0
        assert data["block_accuracy"] == 1.0
        assert data["chapter_accuracy"] == 1.0

    @pytest.mark.anyio
    async def test_mismatched_lengths(self, client: AsyncClient) -> None:
        """Mismatched code lists should return 400."""
        resp = await client.post(
            "/api/v1/evaluate/icd",
            json={"gold_codes": ["E11"], "pred_codes": ["E11", "I10"]},
        )
        assert resp.status_code == 400


# =========================================================================
# POST /evaluate/auprc
# =========================================================================


class TestAUPRCEval:
    """Tests for POST /api/v1/evaluate/auprc."""

    @pytest.mark.anyio
    async def test_perfect(self, client: AsyncClient) -> None:
        """Perfect ranking should yield AUPRC = 1.0."""
        resp = await client.post(
            "/api/v1/evaluate/auprc",
            json={
                "y_true": [1, 1, 0, 0],
                "y_scores": [0.9, 0.8, 0.2, 0.1],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["auprc"] == pytest.approx(1.0)

    @pytest.mark.anyio
    async def test_custom_label(self, client: AsyncClient) -> None:
        """Custom label should propagate."""
        resp = await client.post(
            "/api/v1/evaluate/auprc",
            json={
                "y_true": [1, 0],
                "y_scores": [0.9, 0.1],
                "label": "pneumonia",
            },
        )
        data = resp.json()
        assert data["label"] == "pneumonia"

    @pytest.mark.anyio
    async def test_mismatched_lengths(self, client: AsyncClient) -> None:
        """Mismatched lengths should return 400."""
        resp = await client.post(
            "/api/v1/evaluate/auprc",
            json={"y_true": [1], "y_scores": [0.9, 0.1]},
        )
        assert resp.status_code == 400


# =========================================================================
# GET /evaluate/metrics
# =========================================================================


class TestMetricsCatalogue:
    """Tests for GET /api/v1/evaluate/metrics."""

    @pytest.mark.anyio
    async def test_catalogue(self, client: AsyncClient) -> None:
        """Should return the full catalogue of available metrics."""
        resp = await client.get("/api/v1/evaluate/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "metrics" in data
        names = {m["name"] for m in data["metrics"]}
        assert "classification" in names
        assert "agreement" in names
        assert "ner" in names
        assert "rouge" in names
        assert "icd" in names
        assert "auprc" in names

    @pytest.mark.anyio
    async def test_catalogue_structure(self, client: AsyncClient) -> None:
        """Each metric entry should have name, endpoint, description, use_case."""
        resp = await client.get("/api/v1/evaluate/metrics")
        for m in resp.json()["metrics"]:
            assert "name" in m
            assert "endpoint" in m
            assert "description" in m
            assert "use_case" in m
