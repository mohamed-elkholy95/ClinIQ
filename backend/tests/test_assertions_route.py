"""Tests for the assertion detection API endpoints.

Tests cover:
- POST /assertions — single entity assertion detection
- POST /assertions/batch — batch entity assertion detection
- GET /assertions/statuses — list all assertion status types
- GET /assertions/stats — detection statistics
- Validation errors (out-of-bounds offsets, invalid spans)
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

API_PREFIX = "/api/v1"


@pytest.fixture(autouse=True)
def _setup_client():
    global client
    client = TestClient(app)
    yield


class TestSingleAssertionEndpoint:
    """Tests for POST /assertions."""

    def test_negated_entity(self) -> None:
        """Negated entity returns absent status."""
        response = client.post(
            f"{API_PREFIX}/assertions",
            json={
                "text": "Patient denies chest pain.",
                "entity_start": 15,
                "entity_end": 25,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "absent"
        assert data["entity_text"] == "chest pain"
        assert data["confidence"] > 0.5

    def test_present_entity(self) -> None:
        """Affirmed entity returns present status."""
        response = client.post(
            f"{API_PREFIX}/assertions",
            json={
                "text": "Patient presents with fever.",
                "entity_start": 22,
                "entity_end": 27,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "present"
        assert data["entity_text"] == "fever"

    def test_possible_entity(self) -> None:
        """Uncertain entity returns possible status."""
        response = client.post(
            f"{API_PREFIX}/assertions",
            json={
                "text": "Possible pneumonia on chest X-ray.",
                "entity_start": 9,
                "entity_end": 18,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "possible"
        assert data["entity_text"] == "pneumonia"

    def test_family_entity(self) -> None:
        """Family history entity returns family status."""
        response = client.post(
            f"{API_PREFIX}/assertions",
            json={
                "text": "Family history of diabetes.",
                "entity_start": 18,
                "entity_end": 26,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "family"

    def test_hypothetical_entity(self) -> None:
        """Future/planned entity returns hypothetical status."""
        response = client.post(
            f"{API_PREFIX}/assertions",
            json={
                "text": "Will start metformin 500mg daily.",
                "entity_start": 11,
                "entity_end": 20,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "hypothetical"

    def test_response_includes_sentence(self) -> None:
        """Response includes the sentence context."""
        response = client.post(
            f"{API_PREFIX}/assertions",
            json={
                "text": "No fever noted.",
                "entity_start": 3,
                "entity_end": 8,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "sentence" in data
        assert len(data["sentence"]) > 0

    def test_entity_end_exceeds_text_length(self) -> None:
        """Out-of-bounds entity_end returns 422."""
        response = client.post(
            f"{API_PREFIX}/assertions",
            json={
                "text": "Short text.",
                "entity_start": 0,
                "entity_end": 999,
            },
        )
        assert response.status_code == 422

    def test_entity_end_before_start(self) -> None:
        """entity_end <= entity_start returns validation error."""
        response = client.post(
            f"{API_PREFIX}/assertions",
            json={
                "text": "Some text here.",
                "entity_start": 10,
                "entity_end": 5,
            },
        )
        assert response.status_code == 422

    def test_empty_text_rejected(self) -> None:
        """Empty text returns validation error."""
        response = client.post(
            f"{API_PREFIX}/assertions",
            json={
                "text": "",
                "entity_start": 0,
                "entity_end": 1,
            },
        )
        assert response.status_code == 422


class TestBatchAssertionEndpoint:
    """Tests for POST /assertions/batch."""

    def test_batch_success(self) -> None:
        """Batch endpoint processes multiple entities."""
        text = "No fever. Possible pneumonia. Patient has headache."
        response = client.post(
            f"{API_PREFIX}/assertions/batch",
            json={
                "text": text,
                "entities": [
                    {"start": text.index("fever"), "end": text.index("fever") + 5},
                    {"start": text.index("pneumonia"), "end": text.index("pneumonia") + 9},
                    {"start": text.index("headache"), "end": text.index("headache") + 8},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["results"]) == 3
        assert data["processing_time_ms"] >= 0
        assert isinstance(data["summary"], dict)

    def test_batch_preserves_order(self) -> None:
        """Results are in same order as input entities."""
        text = "No fever. Patient has headache."
        response = client.post(
            f"{API_PREFIX}/assertions/batch",
            json={
                "text": text,
                "entities": [
                    {"start": text.index("fever"), "end": text.index("fever") + 5},
                    {"start": text.index("headache"), "end": text.index("headache") + 8},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["results"][0]["entity_text"] == "fever"
        assert data["results"][1]["entity_text"] == "headache"

    def test_batch_summary_counts(self) -> None:
        """Summary provides correct status counts."""
        text = "No fever. No cough. Patient has headache."
        response = client.post(
            f"{API_PREFIX}/assertions/batch",
            json={
                "text": text,
                "entities": [
                    {"start": text.index("fever"), "end": text.index("fever") + 5},
                    {"start": text.index("cough"), "end": text.index("cough") + 5},
                    {"start": text.index("headache"), "end": text.index("headache") + 8},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["summary"].get("absent", 0) >= 2
        assert data["summary"].get("present", 0) >= 1

    def test_batch_out_of_bounds_entity(self) -> None:
        """Batch with out-of-bounds entity returns 422."""
        response = client.post(
            f"{API_PREFIX}/assertions/batch",
            json={
                "text": "Short.",
                "entities": [{"start": 0, "end": 999}],
            },
        )
        assert response.status_code == 422

    def test_batch_empty_entities_rejected(self) -> None:
        """Empty entities list returns validation error."""
        response = client.post(
            f"{API_PREFIX}/assertions/batch",
            json={
                "text": "Some text.",
                "entities": [],
            },
        )
        assert response.status_code == 422


class TestAssertionStatusesEndpoint:
    """Tests for GET /assertions/statuses."""

    def test_list_statuses(self) -> None:
        """Returns all 6 assertion statuses."""
        response = client.get(f"{API_PREFIX}/assertions/statuses")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 6
        statuses = {item["status"] for item in data}
        assert "present" in statuses
        assert "absent" in statuses
        assert "possible" in statuses
        assert "family" in statuses

    def test_statuses_have_descriptions(self) -> None:
        """Each status has a non-empty description."""
        response = client.get(f"{API_PREFIX}/assertions/statuses")
        data = response.json()
        for item in data:
            assert len(item["description"]) > 10


class TestAssertionStatsEndpoint:
    """Tests for GET /assertions/stats."""

    def test_stats_endpoint(self) -> None:
        """Stats endpoint returns trigger count and version."""
        response = client.get(f"{API_PREFIX}/assertions/stats")
        assert response.status_code == 200
        data = response.json()
        assert "detection_stats" in data
        assert "trigger_count" in data
        assert data["trigger_count"] > 60
        assert data["version"] == "1.0.0"
