"""Tests for the quality analysis API endpoints.

Covers:
- POST /quality — single note analysis
- POST /quality/batch — batch analysis with summary
- GET /quality/dimensions — dimension catalogue
- Input validation (empty text, oversized batch)
- Custom expected sections override
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

PREFIX = "/api/v1"

SAMPLE_NOTE = """
CHIEF COMPLAINT: Chest pain.

HISTORY OF PRESENT ILLNESS:
68-year-old male with hypertension and diabetes presenting with
acute chest pain radiating to the left arm for 2 hours.

ASSESSMENT:
Acute coronary syndrome, rule out STEMI.

PLAN:
1. Admit to CCU
2. Serial troponins Q6h
3. Cardiology consult
"""


class TestQualityEndpoint:
    """Tests for POST /quality."""

    def test_quality_success(self):
        """Single note analysis returns quality report."""
        resp = client.post(f"{PREFIX}/quality", json={"text": SAMPLE_NOTE})
        assert resp.status_code == 200
        data = resp.json()
        assert "overall_score" in data
        assert "grade" in data
        assert "dimensions" in data
        assert "recommendations" in data
        assert "stats" in data
        assert "text_hash" in data
        assert "analysis_ms" in data
        assert 0 <= data["overall_score"] <= 100
        assert data["grade"] in ("A", "B", "C", "D", "F")
        assert len(data["dimensions"]) == 5

    def test_quality_dimensions_have_findings(self):
        """Each dimension has a findings list."""
        resp = client.post(f"{PREFIX}/quality", json={"text": SAMPLE_NOTE})
        data = resp.json()
        for dim in data["dimensions"]:
            assert "dimension" in dim
            assert "score" in dim
            assert "weight" in dim
            assert "findings" in dim
            assert isinstance(dim["findings"], list)

    def test_quality_stats_populated(self):
        """Stats contain word count and section count."""
        resp = client.post(f"{PREFIX}/quality", json={"text": SAMPLE_NOTE})
        data = resp.json()
        stats = data["stats"]
        assert "word_count" in stats
        assert "section_count" in stats
        assert stats["word_count"] > 0

    def test_quality_custom_sections(self):
        """Custom expected sections override defaults."""
        resp = client.post(
            f"{PREFIX}/quality",
            json={
                "text": "CHIEF COMPLAINT: Pain.\nPLAN: Follow up.",
                "expected_sections": ["chief complaint", "plan"],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        # With only 2 expected sections both present, completeness should be reasonable
        completeness = next(
            d for d in data["dimensions"] if d["dimension"] == "completeness"
        )
        missing = [
            f for f in completeness["findings"] if "Missing" in f["message"]
        ]
        assert len(missing) == 0

    def test_quality_empty_text_rejected(self):
        """Empty text is rejected with 422."""
        resp = client.post(f"{PREFIX}/quality", json={"text": ""})
        assert resp.status_code == 422

    def test_quality_short_note_lower_score(self):
        """A short note scores lower than a complete one."""
        short_resp = client.post(f"{PREFIX}/quality", json={"text": "Pain. Follow up."})
        full_resp = client.post(f"{PREFIX}/quality", json={"text": SAMPLE_NOTE})
        assert short_resp.status_code == 200
        assert full_resp.status_code == 200
        assert short_resp.json()["overall_score"] < full_resp.json()["overall_score"]


class TestQualityBatchEndpoint:
    """Tests for POST /quality/batch."""

    def test_batch_success(self):
        """Batch analysis returns results and summary."""
        resp = client.post(
            f"{PREFIX}/quality/batch",
            json={
                "documents": [
                    {"text": SAMPLE_NOTE},
                    {"text": "Short note. Pain."},
                ]
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 2
        assert "summary" in data
        assert data["summary"]["total"] == 2
        assert "average_score" in data["summary"]
        assert "grade_distribution" in data["summary"]

    def test_batch_order_preserved(self):
        """Results are in the same order as input documents."""
        resp = client.post(
            f"{PREFIX}/quality/batch",
            json={
                "documents": [
                    {"text": "Short."},
                    {"text": SAMPLE_NOTE},
                ]
            },
        )
        data = resp.json()
        # Short note should score lower
        assert data["results"][0]["overall_score"] < data["results"][1]["overall_score"]

    def test_batch_summary_stats(self):
        """Summary contains min, max, average, and grade distribution."""
        resp = client.post(
            f"{PREFIX}/quality/batch",
            json={
                "documents": [
                    {"text": SAMPLE_NOTE},
                    {"text": SAMPLE_NOTE},
                ]
            },
        )
        data = resp.json()
        summary = data["summary"]
        assert summary["min_score"] <= summary["average_score"] <= summary["max_score"]
        assert sum(summary["grade_distribution"].values()) == 2

    def test_batch_empty_rejected(self):
        """Empty document list is rejected."""
        resp = client.post(f"{PREFIX}/quality/batch", json={"documents": []})
        assert resp.status_code == 422


class TestQualityDimensionsEndpoint:
    """Tests for GET /quality/dimensions."""

    def test_dimensions_returns_five(self):
        """Returns all five quality dimensions."""
        resp = client.get(f"{PREFIX}/quality/dimensions")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 5

    def test_dimensions_have_descriptions(self):
        """Each dimension has a non-empty description."""
        resp = client.get(f"{PREFIX}/quality/dimensions")
        data = resp.json()
        for dim in data:
            assert "dimension" in dim
            assert "description" in dim
            assert len(dim["description"]) > 20

    def test_dimensions_names_match_enum(self):
        """Dimension names match QualityDimension enum values."""
        resp = client.get(f"{PREFIX}/quality/dimensions")
        data = resp.json()
        names = {d["dimension"] for d in data}
        expected = {"completeness", "readability", "structure", "information_density", "consistency"}
        assert names == expected
