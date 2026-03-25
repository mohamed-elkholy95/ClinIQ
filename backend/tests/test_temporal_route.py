"""Tests for the temporal API endpoint.

Covers:
- POST /temporal — happy path, reference date, validation
- GET /temporal/frequency-map — catalogue response
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

API_PREFIX = "/api/v1"


@pytest.fixture(autouse=True)
def _setup_client():
    global client
    client = TestClient(app)
    yield


class TestTemporalEndpoint:
    """Tests for POST /temporal."""

    def test_happy_path(self):
        body = {
            "text": "Patient admitted on 03/15/2024 with chest pain starting 3 days ago. Take aspirin BID.",
            "reference_date": "2024-03-18",
        }
        resp = client.post(f"{API_PREFIX}/temporal", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["expression_count"] >= 1
        assert data["frequency_count"] >= 1
        assert data["reference_date"] == "2024-03-18"
        assert data["processing_time_ms"] >= 0

    def test_default_reference_date(self):
        body = {"text": "Symptoms began today"}
        resp = client.post(f"{API_PREFIX}/temporal", json=body)
        assert resp.status_code == 200
        data = resp.json()
        # Should use today's date
        assert "reference_date" in data

    def test_date_extraction(self):
        body = {
            "text": "Surgery scheduled for March 15, 2024",
            "reference_date": "2024-03-10",
        }
        resp = client.post(f"{API_PREFIX}/temporal", json=body)
        assert resp.status_code == 200
        data = resp.json()
        dates = [e for e in data["expressions"] if e["temporal_type"] == "date"]
        assert len(dates) >= 1
        assert dates[0]["resolved_date"] == "2024-03-15"

    def test_frequency_extraction(self):
        body = {"text": "Metformin 500mg PO BID with meals. Morphine 2mg q4h PRN."}
        resp = client.post(f"{API_PREFIX}/temporal", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert data["frequency_count"] >= 2

    def test_duration_extraction(self):
        body = {"text": "Treatment for 6 weeks, then reassess"}
        resp = client.post(f"{API_PREFIX}/temporal", json=body)
        assert resp.status_code == 200
        data = resp.json()
        assert "expressions" in data
        durations = [e for e in data["expressions"] if e["temporal_type"] == "duration"]
        assert len(durations) >= 1

    def test_empty_text_rejected(self):
        body = {"text": ""}
        resp = client.post(f"{API_PREFIX}/temporal", json=body)
        assert resp.status_code == 422

    def test_no_temporals_returns_empty(self):
        body = {"text": "Patient denies chest pain and shortness of breath"}
        resp = client.post(f"{API_PREFIX}/temporal", json=body)
        assert resp.status_code == 200
        data = resp.json()
        # No dates should be found
        dates = [e for e in data["expressions"] if e["temporal_type"] == "date"]
        assert len(dates) == 0


class TestFrequencyMapEndpoint:
    """Tests for GET /temporal/frequency-map."""

    def test_list_frequencies(self):
        resp = client.get(f"{API_PREFIX}/temporal/frequency-map")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 30
        assert "bid" in data["frequencies"]
        assert "prn" in data["frequencies"]

    def test_frequency_entry_structure(self):
        resp = client.get(f"{API_PREFIX}/temporal/frequency-map")
        data = resp.json()
        bid = data["frequencies"]["bid"]
        assert bid["times_per_day"] == 2.0
        assert bid["interval_hours"] == 12.0
        assert bid["as_needed"] is False
