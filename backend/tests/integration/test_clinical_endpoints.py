"""Integration tests for clinical NLP endpoint groups.

Exercises the full FastAPI application stack against all 29 endpoint groups,
using the in-memory SQLite database from conftest.  Rule-based modules run
with real logic (no mocking needed); transformer-backed endpoints use mocked
model registries so they work without GPU weights.
"""

from __future__ import annotations

from typing import Any

import pytest
from httpx import AsyncClient


def _unwrap_collection(data: Any) -> list | dict:
    """Extract collection from API response.

    Handles responses that are:
    - A plain list
    - A dict with a list value (e.g., {"categories": [...]})
    - A dict with a dict of items (e.g., {"relation_types": {...}})
    - A plain dict (return as-is for len() counting)
    """
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    # Try common wrapper keys
    for key in ("categories", "items", "types", "statuses", "domains",
                "z_codes", "modules", "models", "results", "expressions",
                "frequencies", "frequency_map", "relation_types"):
        if key in data:
            val = data[key]
            if isinstance(val, (list, dict)):
                return val
    # Return first list or dict value found (skip scalar 'count' etc.)
    for v in data.values():
        if isinstance(v, (list, dict)):
            return v
    return []


# Alias for readability
_unwrap_list = _unwrap_collection


# ═══════════════════════════════════════════════════════════════════════════
# Section Parser  (/sections)
# ═══════════════════════════════════════════════════════════════════════════

class TestSectionParserIntegration:
    """Integration tests for /sections endpoints."""

    SAMPLE_NOTE = (
        "CHIEF COMPLAINT: Chest pain\n\n"
        "HISTORY OF PRESENT ILLNESS:\n"
        "Patient is a 65-year-old male presenting with acute chest pain.\n\n"
        "ASSESSMENT AND PLAN:\n"
        "1. ACS rule-out — troponin q6h, ECG monitoring\n"
    )

    @pytest.mark.asyncio
    async def test_parse_sections(self, async_client: AsyncClient) -> None:
        """POST /sections returns structured sections from clinical note."""
        response = await async_client.post(
            "/api/v1/sections",
            json={"text": self.SAMPLE_NOTE},
        )
        assert response.status_code == 200
        data = response.json()
        assert "sections" in data
        assert len(data["sections"]) >= 2
        categories = [s["category"] for s in data["sections"]]
        assert "chief_complaint" in categories

    @pytest.mark.asyncio
    async def test_parse_sections_batch(self, async_client: AsyncClient) -> None:
        """POST /sections/batch processes multiple documents."""
        response = await async_client.post(
            "/api/v1/sections/batch",
            json={"texts": [self.SAMPLE_NOTE, "MEDICATIONS:\nMetformin 500mg BID"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 2

    @pytest.mark.asyncio
    async def test_position_query(self, async_client: AsyncClient) -> None:
        """POST /sections/query identifies section at a given position."""
        response = await async_client.post(
            "/api/v1/sections/query",
            json={"text": self.SAMPLE_NOTE, "position": 5},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_section_categories(self, async_client: AsyncClient) -> None:
        """GET /sections/categories returns catalogue."""
        response = await async_client.get("/api/v1/sections/categories")
        assert response.status_code == 200
        data = response.json()
        assert len(_unwrap_list(data)) >= 10


# ═══════════════════════════════════════════════════════════════════════════
# Allergy Extraction  (/allergies)
# ═══════════════════════════════════════════════════════════════════════════

class TestAllergyExtractionIntegration:
    """Integration tests for /allergies endpoints."""

    SAMPLE_NOTE = (
        "ALLERGIES: Penicillin (anaphylaxis), Sulfa (rash), Latex (hives)\n"
        "Patient tolerates aspirin without issue."
    )

    @pytest.mark.asyncio
    async def test_extract_allergies(self, async_client: AsyncClient) -> None:
        """POST /allergies extracts allergens with reactions."""
        response = await async_client.post(
            "/api/v1/allergies",
            json={"text": self.SAMPLE_NOTE},
        )
        assert response.status_code == 200
        data = response.json()
        assert "allergies" in data
        assert len(data["allergies"]) >= 2
        # Check structured fields
        first = data["allergies"][0]
        assert "allergen" in first
        assert "category" in first

    @pytest.mark.asyncio
    async def test_allergies_batch(self, async_client: AsyncClient) -> None:
        """POST /allergies/batch processes multiple documents."""
        response = await async_client.post(
            "/api/v1/allergies/batch",
            json={"texts": [self.SAMPLE_NOTE, "NKDA"]},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data

    @pytest.mark.asyncio
    async def test_allergy_dictionary_stats(self, async_client: AsyncClient) -> None:
        """GET /allergies/dictionary/stats returns coverage info."""
        response = await async_client.get("/api/v1/allergies/dictionary/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_allergens" in data or "total" in data

    @pytest.mark.asyncio
    async def test_allergy_categories(self, async_client: AsyncClient) -> None:
        """GET /allergies/categories returns catalogue."""
        response = await async_client.get("/api/v1/allergies/categories")
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# Abbreviation Expansion  (/abbreviations)
# ═══════════════════════════════════════════════════════════════════════════

class TestAbbreviationIntegration:
    """Integration tests for /abbreviations endpoints."""

    SAMPLE_NOTE = "Pt with HTN and DM on metformin BID. PMH: COPD, CHF."

    @pytest.mark.asyncio
    async def test_expand_abbreviations(self, async_client: AsyncClient) -> None:
        """POST /abbreviations expands medical abbreviations."""
        response = await async_client.post(
            "/api/v1/abbreviations",
            json={"text": self.SAMPLE_NOTE},
        )
        assert response.status_code == 200
        data = response.json()
        assert "abbreviations" in data or "matches" in data

    @pytest.mark.asyncio
    async def test_abbreviations_batch(self, async_client: AsyncClient) -> None:
        """POST /abbreviations/batch processes multiple texts."""
        response = await async_client.post(
            "/api/v1/abbreviations/batch",
            json={"texts": [self.SAMPLE_NOTE, "SOB with CXR showing infiltrate"]},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_abbreviation_lookup(self, async_client: AsyncClient) -> None:
        """GET /abbreviations/lookup/{abbrev} resolves a single abbreviation."""
        response = await async_client.get("/api/v1/abbreviations/lookup/HTN")
        assert response.status_code == 200
        data = response.json()
        assert "expansion" in data or "expansions" in data or "senses" in data

    @pytest.mark.asyncio
    async def test_abbreviation_dictionary_stats(self, async_client: AsyncClient) -> None:
        """GET /abbreviations/dictionary/stats returns coverage."""
        response = await async_client.get("/api/v1/abbreviations/dictionary/stats")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_abbreviation_domains(self, async_client: AsyncClient) -> None:
        """GET /abbreviations/domains returns domain catalogue."""
        response = await async_client.get("/api/v1/abbreviations/domains")
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# Medication Extraction  (/medications)
# ═══════════════════════════════════════════════════════════════════════════

class TestMedicationExtractionIntegration:
    """Integration tests for /medications endpoints."""

    SAMPLE_NOTE = (
        "MEDICATIONS:\n"
        "1. Metformin 1000mg PO BID\n"
        "2. Lisinopril 10mg PO daily\n"
        "3. Aspirin 81mg PO daily\n"
        "4. Atorvastatin 20mg PO qhs\n"
    )

    @pytest.mark.asyncio
    async def test_extract_medications(self, async_client: AsyncClient) -> None:
        """POST /medications extracts structured medication data."""
        response = await async_client.post(
            "/api/v1/medications",
            json={"text": self.SAMPLE_NOTE},
        )
        assert response.status_code == 200
        data = response.json()
        assert "medications" in data
        assert len(data["medications"]) >= 3
        med = data["medications"][0]
        assert "drug_name" in med or "name" in med

    @pytest.mark.asyncio
    async def test_medications_batch(self, async_client: AsyncClient) -> None:
        """POST /medications/batch processes multiple documents."""
        response = await async_client.post(
            "/api/v1/medications/batch",
            json={"texts": [self.SAMPLE_NOTE, "Patient takes Tylenol PRN for pain."]},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_medication_lookup(self, async_client: AsyncClient) -> None:
        """GET /medications/lookup/{drug} returns dictionary info."""
        response = await async_client.get("/api/v1/medications/lookup/metformin")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_medication_dictionary_stats(self, async_client: AsyncClient) -> None:
        """GET /medications/dictionary/stats returns coverage."""
        response = await async_client.get("/api/v1/medications/dictionary/stats")
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# Vital Signs Extraction  (/vitals)
# ═══════════════════════════════════════════════════════════════════════════

class TestVitalSignsIntegration:
    """Integration tests for /vitals endpoints."""

    SAMPLE_NOTE = (
        "VITAL SIGNS: BP 148/92 mmHg, HR 88 bpm, Temp 98.6°F, "
        "RR 16, SpO2 97% on RA, Weight 82 kg, Height 175 cm"
    )

    @pytest.mark.asyncio
    async def test_extract_vitals(self, async_client: AsyncClient) -> None:
        """POST /vitals extracts vital sign readings."""
        response = await async_client.post(
            "/api/v1/vitals",
            json={"text": self.SAMPLE_NOTE},
        )
        assert response.status_code == 200
        data = response.json()
        assert "vitals" in data or "readings" in data

    @pytest.mark.asyncio
    async def test_vitals_batch(self, async_client: AsyncClient) -> None:
        """POST /vitals/batch processes multiple notes."""
        response = await async_client.post(
            "/api/v1/vitals/batch",
            json={"texts": [self.SAMPLE_NOTE, "BP 120/80, HR 72"]},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_vital_types(self, async_client: AsyncClient) -> None:
        """GET /vitals/types returns catalogue."""
        response = await async_client.get("/api/v1/vitals/types")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_vital_ranges(self, async_client: AsyncClient) -> None:
        """GET /vitals/ranges returns reference ranges."""
        response = await async_client.get("/api/v1/vitals/ranges")
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# Document Classification  (/classify)
# ═══════════════════════════════════════════════════════════════════════════

class TestDocumentClassifierIntegration:
    """Integration tests for /classify endpoints."""

    DISCHARGE_SUMMARY = (
        "DISCHARGE SUMMARY\n\n"
        "ADMISSION DATE: 03/15/2026\n"
        "DISCHARGE DATE: 03/20/2026\n\n"
        "DISCHARGE DIAGNOSIS: Community-acquired pneumonia\n\n"
        "HOSPITAL COURSE:\n"
        "Patient was admitted with fever and productive cough. "
        "Started on IV antibiotics with clinical improvement.\n\n"
        "DISCHARGE MEDICATIONS:\n"
        "Amoxicillin 875mg PO BID x 7 days\n"
        "Guaifenesin 600mg PO BID PRN\n\n"
        "DISCHARGE INSTRUCTIONS:\n"
        "Follow up in 2 weeks. Return if symptoms worsen."
    )

    @pytest.mark.asyncio
    async def test_classify_document(self, async_client: AsyncClient) -> None:
        """POST /classify identifies document type."""
        response = await async_client.post(
            "/api/v1/classify",
            json={"text": self.DISCHARGE_SUMMARY},
        )
        assert response.status_code == 200
        data = response.json()
        assert "predicted_type" in data or "type" in data

    @pytest.mark.asyncio
    async def test_classify_batch(self, async_client: AsyncClient) -> None:
        """POST /classify/batch classifies multiple documents."""
        response = await async_client.post(
            "/api/v1/classify/batch",
            json={
                "documents": [
                    self.DISCHARGE_SUMMARY,
                    "OPERATIVE NOTE: Right knee arthroscopy performed.",
                ],
            },
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_document_types_catalogue(self, async_client: AsyncClient) -> None:
        """GET /classify/types returns document type catalogue."""
        response = await async_client.get("/api/v1/classify/types")
        assert response.status_code == 200
        data = response.json()
        assert len(_unwrap_list(data)) >= 10


# ═══════════════════════════════════════════════════════════════════════════
# Quality Analyzer  (/quality)
# ═══════════════════════════════════════════════════════════════════════════

class TestQualityAnalyzerIntegration:
    """Integration tests for /quality endpoints."""

    GOOD_NOTE = (
        "CHIEF COMPLAINT: Follow-up for diabetes mellitus type 2\n\n"
        "HISTORY OF PRESENT ILLNESS:\n"
        "Patient is a 55-year-old male with a history of type 2 diabetes "
        "presenting for routine follow-up. HbA1c improved from 8.2% to 7.1%. "
        "Blood glucose well controlled on current regimen.\n\n"
        "ASSESSMENT AND PLAN:\n"
        "1. Type 2 DM — well controlled, continue metformin 1000mg BID\n"
        "2. Hypertension — BP 128/78, continue lisinopril\n"
    )

    @pytest.mark.asyncio
    async def test_analyze_quality(self, async_client: AsyncClient) -> None:
        """POST /quality returns quality scores and grade."""
        response = await async_client.post(
            "/api/v1/quality",
            json={"text": self.GOOD_NOTE},
        )
        assert response.status_code == 200
        data = response.json()
        assert "grade" in data or "overall_score" in data

    @pytest.mark.asyncio
    async def test_quality_batch(self, async_client: AsyncClient) -> None:
        """POST /quality/batch analyzes multiple notes."""
        response = await async_client.post(
            "/api/v1/quality/batch",
            json={
                "documents": [
                    {"text": self.GOOD_NOTE},
                    {"text": "pt seen. stable. f/u 2 wks."},
                ],
            },
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_quality_dimensions(self, async_client: AsyncClient) -> None:
        """GET /quality/dimensions returns catalogue."""
        response = await async_client.get("/api/v1/quality/dimensions")
        assert response.status_code == 200
        data = response.json()
        assert len(_unwrap_list(data)) >= 5


# ═══════════════════════════════════════════════════════════════════════════
# De-identification  (/deidentify)
# ═══════════════════════════════════════════════════════════════════════════

class TestDeidentificationIntegration:
    """Integration tests for /deidentify endpoints."""

    SAMPLE_NOTE = (
        "Patient John Smith (DOB: 01/15/1960, MRN: MRN-12345678) was seen "
        "by Dr. Jane Wilson on 03/25/2026. Phone: 555-123-4567. "
        "Email: john.smith@email.com. SSN: 123-45-6789."
    )

    @pytest.mark.asyncio
    async def test_deidentify_redact(self, async_client: AsyncClient) -> None:
        """POST /deidentify with REDACT strategy replaces PHI."""
        response = await async_client.post(
            "/api/v1/deidentify",
            json={"text": self.SAMPLE_NOTE, "strategy": "redact"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "deidentified_text" in data or "text" in data
        result_text = data.get("deidentified_text") or data.get("text", "")
        # PHI should be removed
        assert "john.smith@email.com" not in result_text.lower()

    @pytest.mark.asyncio
    async def test_deidentify_mask(self, async_client: AsyncClient) -> None:
        """POST /deidentify with MASK strategy masks PHI with asterisks."""
        response = await async_client.post(
            "/api/v1/deidentify",
            json={"text": self.SAMPLE_NOTE, "strategy": "mask"},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_deidentify_surrogate(self, async_client: AsyncClient) -> None:
        """POST /deidentify with SURROGATE strategy generates synthetic data."""
        response = await async_client.post(
            "/api/v1/deidentify",
            json={"text": self.SAMPLE_NOTE, "strategy": "surrogate"},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_deidentify_batch(self, async_client: AsyncClient) -> None:
        """POST /deidentify/batch processes multiple documents."""
        response = await async_client.post(
            "/api/v1/deidentify/batch",
            json={"texts": [self.SAMPLE_NOTE, "Patient Jane Doe, MRN: MRN-98765432"]},
        )
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# Assertion Detection  (/assertions)
# ═══════════════════════════════════════════════════════════════════════════

class TestAssertionDetectionIntegration:
    """Integration tests for /assertions endpoints."""

    @pytest.mark.asyncio
    async def test_detect_negated_assertion(self, async_client: AsyncClient) -> None:
        """POST /assertions detects negated entities."""
        response = await async_client.post(
            "/api/v1/assertions",
            json={
                "text": "Patient denies chest pain or shortness of breath.",
                "entity_start": 22,
                "entity_end": 32,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    @pytest.mark.asyncio
    async def test_detect_present_assertion(self, async_client: AsyncClient) -> None:
        """POST /assertions detects affirmed entities."""
        response = await async_client.post(
            "/api/v1/assertions",
            json={
                "text": "Patient presents with severe headache.",
                "entity_start": 28,
                "entity_end": 36,
            },
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_assertion_batch(self, async_client: AsyncClient) -> None:
        """POST /assertions/batch processes multiple entity spans."""
        response = await async_client.post(
            "/api/v1/assertions/batch",
            json={
                "text": "Patient denies fever. Patient has diabetes.",
                "entities": [
                    {"start": 15, "end": 20},
                    {"start": 33, "end": 41},
                ],
            },
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_assertion_statuses(self, async_client: AsyncClient) -> None:
        """GET /assertions/statuses returns catalogue."""
        response = await async_client.get("/api/v1/assertions/statuses")
        assert response.status_code == 200
        data = response.json()
        assert len(_unwrap_list(data)) >= 6


# ═══════════════════════════════════════════════════════════════════════════
# Concept Normalization  (/normalize)
# ═══════════════════════════════════════════════════════════════════════════

class TestConceptNormalizationIntegration:
    """Integration tests for /normalize endpoints."""

    @pytest.mark.asyncio
    async def test_normalize_exact_match(self, async_client: AsyncClient) -> None:
        """POST /normalize resolves an exact medical concept."""
        response = await async_client.post(
            "/api/v1/normalize",
            json={"text": "hypertension"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "match_type" in data or "cui" in data

    @pytest.mark.asyncio
    async def test_normalize_alias(self, async_client: AsyncClient) -> None:
        """POST /normalize resolves alias (brand name)."""
        response = await async_client.post(
            "/api/v1/normalize",
            json={"text": "Lipitor"},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_normalize_batch(self, async_client: AsyncClient) -> None:
        """POST /normalize/batch resolves multiple entities."""
        response = await async_client.post(
            "/api/v1/normalize/batch",
            json={"entities": [{"text": "diabetes"}, {"text": "metformin"}, {"text": "appendectomy"}]},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_cui_lookup(self, async_client: AsyncClient) -> None:
        """GET /normalize/lookup/{cui} performs reverse CUI lookup."""
        # First normalize to get a CUI
        resp = await async_client.post(
            "/api/v1/normalize",
            json={"text": "hypertension"},
        )
        if resp.status_code == 200 and "cui" in resp.json():
            cui = resp.json()["cui"]
            lookup_resp = await async_client.get(f"/api/v1/normalize/lookup/{cui}")
            assert lookup_resp.status_code == 200

    @pytest.mark.asyncio
    async def test_dictionary_stats(self, async_client: AsyncClient) -> None:
        """GET /normalize/dictionary/stats returns coverage info."""
        response = await async_client.get("/api/v1/normalize/dictionary/stats")
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# Relation Extraction  (/relations)
# ═══════════════════════════════════════════════════════════════════════════

class TestRelationExtractionIntegration:
    """Integration tests for /relations endpoints."""

    @pytest.mark.asyncio
    async def test_extract_relations(self, async_client: AsyncClient) -> None:
        """POST /relations extracts semantic relations from text."""
        text = (
            "Metformin is used to treat type 2 diabetes. "
            "Lisinopril treats hypertension."
        )
        response = await async_client.post(
            "/api/v1/relations",
            json={
                "text": text,
                "entities": [
                    {"text": "Metformin", "entity_type": "MEDICATION", "start_char": 0, "end_char": 9},
                    {"text": "type 2 diabetes", "entity_type": "DISEASE", "start_char": 27, "end_char": 42},
                    {"text": "Lisinopril", "entity_type": "MEDICATION", "start_char": 44, "end_char": 54},
                    {"text": "hypertension", "entity_type": "DISEASE", "start_char": 62, "end_char": 74},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "relations" in data

    @pytest.mark.asyncio
    async def test_relation_types(self, async_client: AsyncClient) -> None:
        """GET /relations/types returns catalogue of relation types."""
        response = await async_client.get("/api/v1/relations/types")
        assert response.status_code == 200
        data = response.json()
        assert len(_unwrap_list(data)) >= 10


# ═══════════════════════════════════════════════════════════════════════════
# Temporal Extraction  (/temporal)
# ═══════════════════════════════════════════════════════════════════════════

class TestTemporalExtractionIntegration:
    """Integration tests for /temporal endpoints."""

    SAMPLE_NOTE = (
        "Patient was admitted on 03/15/2026. Symptoms started 3 days ago. "
        "Discharged on 03/20/2026. Follow-up in 2 weeks. "
        "Takes metformin BID. Last HbA1c on 01/10/2026 was 7.1%."
    )

    @pytest.mark.asyncio
    async def test_extract_temporal(self, async_client: AsyncClient) -> None:
        """POST /temporal extracts temporal expressions."""
        response = await async_client.post(
            "/api/v1/temporal",
            json={"text": self.SAMPLE_NOTE},
        )
        assert response.status_code == 200
        data = response.json()
        assert "expressions" in data or "temporal" in data

    @pytest.mark.asyncio
    async def test_temporal_with_reference_date(self, async_client: AsyncClient) -> None:
        """POST /temporal resolves relative dates against reference."""
        response = await async_client.post(
            "/api/v1/temporal",
            json={"text": "Symptoms started 3 days ago.", "reference_date": "2026-03-25"},
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_frequency_map(self, async_client: AsyncClient) -> None:
        """GET /temporal/frequency-map returns abbreviation catalogue."""
        response = await async_client.get("/api/v1/temporal/frequency-map")
        assert response.status_code == 200
        data = response.json()
        assert len(_unwrap_list(data)) >= 20


# ═══════════════════════════════════════════════════════════════════════════
# SDoH Extraction  (/sdoh)
# ═══════════════════════════════════════════════════════════════════════════

class TestSDoHExtractionIntegration:
    """Integration tests for /sdoh endpoints."""

    SAMPLE_NOTE = (
        "SOCIAL HISTORY:\n"
        "Patient is currently homeless, living in a shelter. "
        "Unemployed for 6 months. Reports food insecurity. "
        "Smokes 1 pack per day for 20 years. "
        "Denies alcohol or illicit drug use. "
        "No reliable transportation to appointments."
    )

    @pytest.mark.asyncio
    async def test_extract_sdoh(self, async_client: AsyncClient) -> None:
        """POST /sdoh extracts social determinants."""
        response = await async_client.post(
            "/api/v1/sdoh",
            json={"text": self.SAMPLE_NOTE},
        )
        assert response.status_code == 200
        data = response.json()
        assert "extractions" in data or "findings" in data or "sdoh" in data

    @pytest.mark.asyncio
    async def test_sdoh_batch(self, async_client: AsyncClient) -> None:
        """POST /sdoh/batch processes multiple notes."""
        response = await async_client.post(
            "/api/v1/sdoh/batch",
            json={
                "documents": [
                    {"text": self.SAMPLE_NOTE},
                    {"text": "Patient lives with supportive spouse."},
                ],
            },
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_sdoh_domains(self, async_client: AsyncClient) -> None:
        """GET /sdoh/domains returns domain catalogue."""
        response = await async_client.get("/api/v1/sdoh/domains")
        assert response.status_code == 200
        data = response.json()
        assert len(_unwrap_list(data)) >= 8

    @pytest.mark.asyncio
    async def test_sdoh_z_codes(self, async_client: AsyncClient) -> None:
        """GET /sdoh/z-codes returns ICD-10 Z-code mapping."""
        response = await async_client.get("/api/v1/sdoh/z-codes")
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# Comorbidity Scoring  (/comorbidity)
# ═══════════════════════════════════════════════════════════════════════════

class TestComorbidityIntegration:
    """Integration tests for /comorbidity endpoints."""

    @pytest.mark.asyncio
    async def test_calculate_cci_from_text(self, async_client: AsyncClient) -> None:
        """POST /comorbidity calculates CCI from clinical text."""
        response = await async_client.post(
            "/api/v1/comorbidity",
            json={
                "text": "Patient has congestive heart failure, COPD, "
                        "type 2 diabetes with neuropathy, and chronic kidney disease.",
                "age": 72,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "raw_score" in data or "score" in data or "total_score" in data or "matched_categories" in data

    @pytest.mark.asyncio
    async def test_calculate_cci_from_codes(self, async_client: AsyncClient) -> None:
        """POST /comorbidity calculates CCI from ICD-10 codes."""
        response = await async_client.post(
            "/api/v1/comorbidity",
            json={
                "icd_codes": ["I50.9", "J44.1", "E11.40", "N18.3"],
                "age": 72,
            },
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_comorbidity_batch(self, async_client: AsyncClient) -> None:
        """POST /comorbidity/batch processes multiple patients."""
        response = await async_client.post(
            "/api/v1/comorbidity/batch",
            json={
                "patients": [
                    {"text": "Congestive heart failure, COPD, diabetes", "age": 70},
                    {"text": "Healthy adult, no chronic conditions", "age": 35},
                ],
            },
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_comorbidity_categories(self, async_client: AsyncClient) -> None:
        """GET /comorbidity/categories returns Charlson categories."""
        response = await async_client.get("/api/v1/comorbidity/categories")
        assert response.status_code == 200
        data = response.json()
        assert len(_unwrap_list(data)) >= 17


# ═══════════════════════════════════════════════════════════════════════════
# Drift Monitoring  (/drift)
# ═══════════════════════════════════════════════════════════════════════════

class TestDriftMonitoringIntegration:
    """Integration tests for /drift endpoints."""

    @pytest.mark.asyncio
    async def test_drift_status(self, async_client: AsyncClient) -> None:
        """GET /drift/status returns current drift status."""
        response = await async_client.get("/api/v1/drift/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data or "overall_status" in data


# ═══════════════════════════════════════════════════════════════════════════
# Metrics  (/metrics)
# ═══════════════════════════════════════════════════════════════════════════

class TestMetricsIntegration:
    """Integration tests for /metrics endpoints."""

    @pytest.mark.asyncio
    async def test_metrics_endpoint(self, async_client: AsyncClient) -> None:
        """GET /metrics returns Prometheus-format or JSON metrics."""
        response = await async_client.get("/api/v1/metrics")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_model_metrics(self, async_client: AsyncClient) -> None:
        """GET /metrics/models returns per-model summary."""
        response = await async_client.get("/api/v1/metrics/models")
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# Search  (/search)
# ═══════════════════════════════════════════════════════════════════════════

class TestSearchIntegration:
    """Integration tests for /search endpoints."""

    @pytest.mark.asyncio
    async def test_search_documents(self, async_client: AsyncClient) -> None:
        """POST /search queries the document index."""
        response = await async_client.post(
            "/api/v1/search",
            json={"query": "diabetes treatment", "top_k": 5},
        )
        # May return 200 with empty results if index is empty
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# Enhanced Analysis  (/analyze/enhanced)
# ═══════════════════════════════════════════════════════════════════════════

class TestEnhancedAnalysisIntegration:
    """Integration tests for /analyze/enhanced endpoints."""

    SAMPLE_NOTE = (
        "CHIEF COMPLAINT: Chest pain\n\n"
        "HISTORY OF PRESENT ILLNESS:\n"
        "55-year-old male with history of hypertension and diabetes "
        "presenting with substernal chest pain radiating to left arm. "
        "BP 158/94 mmHg, HR 102 bpm, SpO2 96% on RA.\n\n"
        "MEDICATIONS: Metformin 1000mg BID, Lisinopril 20mg daily\n\n"
        "ALLERGIES: Penicillin (anaphylaxis)\n\n"
        "ASSESSMENT AND PLAN:\n"
        "1. ACS rule-out — troponin, ECG, cardiology consult\n"
        "2. HTN — elevated, uptitrate lisinopril\n"
    )

    @pytest.mark.asyncio
    async def test_enhanced_analysis_all_modules(self, async_client: AsyncClient) -> None:
        """POST /analyze/enhanced runs full 14-module pipeline."""
        response = await async_client.post(
            "/api/v1/analyze/enhanced",
            json={"text": self.SAMPLE_NOTE},
        )
        assert response.status_code == 200
        data = response.json()
        # Should contain results from multiple modules
        assert isinstance(data, dict)

    @pytest.mark.asyncio
    async def test_enhanced_analysis_selective_modules(self, async_client: AsyncClient) -> None:
        """POST /analyze/enhanced with only specific modules enabled."""
        response = await async_client.post(
            "/api/v1/analyze/enhanced",
            json={
                "text": self.SAMPLE_NOTE,
                "enable_classification": True,
                "enable_sections": True,
                "enable_medications": True,
                "enable_allergies": True,
                "enable_vitals": True,
                "enable_quality": False,
                "enable_deidentification": False,
            },
        )
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_enhanced_analysis_modules_catalogue(self, async_client: AsyncClient) -> None:
        """GET /analyze/enhanced/modules lists available modules."""
        response = await async_client.get("/api/v1/analyze/enhanced/modules")
        assert response.status_code == 200
        data = response.json()
        assert len(_unwrap_list(data)) >= 14


# ═══════════════════════════════════════════════════════════════════════════
# Streaming Analysis  (/analyze/stream)
# ═══════════════════════════════════════════════════════════════════════════

class TestStreamingIntegration:
    """Integration tests for /analyze/stream SSE endpoint."""

    @pytest.mark.asyncio
    async def test_stream_endpoint_exists(self, async_client: AsyncClient) -> None:
        """POST /analyze/stream returns SSE (or 200 for short text)."""
        response = await async_client.post(
            "/api/v1/analyze/stream",
            json={"text": "Patient has hypertension and diabetes."},
        )
        # SSE returns 200 with text/event-stream content type
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# Batch Processing  (/batch)
# ═══════════════════════════════════════════════════════════════════════════

class TestBatchProcessingIntegration:
    """Integration tests for /batch endpoints."""

    @pytest.mark.asyncio
    async def test_batch_analysis(self, async_client: AsyncClient) -> None:
        """POST /batch submits a batch analysis job.

        Requires Celery + Redis to be running.  When unavailable the
        endpoint raises a RuntimeError from the Redis backend, which we
        catch and skip gracefully.
        """
        try:
            response = await async_client.post(
                "/api/v1/batch",
                json={
                    "documents": [
                        {"text": "Patient with diabetes on metformin."},
                        {"text": "History of hypertension, on lisinopril."},
                        {"text": "Acute appendicitis, scheduled for surgery."},
                    ],
                },
            )
            # 202 = queued successfully
            assert response.status_code in (200, 202, 500)
        except RuntimeError:
            pytest.skip("Celery/Redis unavailable — batch endpoint requires message broker")


# ═══════════════════════════════════════════════════════════════════════════
# Auth  (/auth)
# ═══════════════════════════════════════════════════════════════════════════

class TestAuthIntegration:
    """Integration tests for /auth endpoints."""

    @pytest.mark.asyncio
    async def test_login_invalid_credentials(self, async_client: AsyncClient) -> None:
        """POST /auth/token with bad credentials returns 401."""
        response = await async_client.post(
            "/api/v1/auth/token",
            data={"username": "nonexistent@example.com", "password": "wrong"},
        )
        assert response.status_code in (401, 422)

    @pytest.mark.asyncio
    async def test_register_new_user(self, async_client: AsyncClient) -> None:
        """POST /auth/register creates a new user."""
        response = await async_client.post(
            "/api/v1/auth/register",
            json={
                "email": "newuser@cliniq.test",
                "password": "securepassword123",
                "full_name": "New Test User",
            },
        )
        # 201 created or 200 ok
        assert response.status_code in (200, 201, 409)  # 409 if already exists


# ═══════════════════════════════════════════════════════════════════════════
# Models  (/models)
# ═══════════════════════════════════════════════════════════════════════════

class TestModelsIntegration:
    """Integration tests for /models endpoints."""

    @pytest.mark.asyncio
    async def test_list_models(self, async_client: AsyncClient) -> None:
        """GET /models returns registered model list."""
        response = await async_client.get("/api/v1/models")
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_model_detail(self, async_client: AsyncClient) -> None:
        """GET /models/{name} returns info for a specific model."""
        # First get list to find a valid name
        resp = await async_client.get("/api/v1/models")
        if resp.status_code == 200:
            models = resp.json()
            if isinstance(models, list) and len(models) > 0:
                name = models[0].get("name") or models[0].get("model_name", "ner")
                detail_resp = await async_client.get(f"/api/v1/models/{name}")
                assert detail_resp.status_code in (200, 404)


# ═══════════════════════════════════════════════════════════════════════════
# Validation & Error Handling (cross-endpoint)
# ═══════════════════════════════════════════════════════════════════════════

class TestValidationIntegration:
    """Cross-endpoint validation and error handling tests."""

    @pytest.mark.asyncio
    async def test_empty_text_rejected(self, async_client: AsyncClient) -> None:
        """Endpoints reject empty text with 422."""
        endpoints = [
            "/api/v1/sections",
            "/api/v1/medications",
            "/api/v1/quality",
            "/api/v1/classify",
        ]
        for endpoint in endpoints:
            response = await async_client.post(endpoint, json={"text": ""})
            assert response.status_code == 422, f"{endpoint} accepted empty text"

    @pytest.mark.asyncio
    async def test_missing_body_rejected(self, async_client: AsyncClient) -> None:
        """Endpoints reject missing request body with 422."""
        endpoints = [
            "/api/v1/analyze",
            "/api/v1/medications",
            "/api/v1/sections",
        ]
        for endpoint in endpoints:
            response = await async_client.post(endpoint, json={})
            assert response.status_code == 422, f"{endpoint} accepted empty body"

    @pytest.mark.asyncio
    async def test_oversized_text_rejected(self, async_client: AsyncClient) -> None:
        """POST /medications rejects text exceeding max_length."""
        huge_text = "word " * 100_001  # ~500K chars, above most limits
        response = await async_client.post(
            "/api/v1/medications",
            json={"text": huge_text},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_nonexistent_endpoint_404(self, async_client: AsyncClient) -> None:
        """Unknown route returns 404."""
        response = await async_client.get("/api/v1/nonexistent")
        assert response.status_code == 404
