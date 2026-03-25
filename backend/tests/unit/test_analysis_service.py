"""Unit tests for the AnalysisService database persistence layer.

Tests document storage, entity extraction persistence, prediction storage,
audit logging, and document processing state transitions.  Uses mock
AsyncSession to avoid database dependencies.
"""

from __future__ import annotations

import hashlib
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from app.services.analysis import AnalysisService

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db() -> AsyncMock:
    """Mock AsyncSession with flush and commit support."""
    db = AsyncMock()
    db.add = MagicMock()
    db.flush = AsyncMock()
    db.commit = AsyncMock()
    db.execute = AsyncMock()
    return db


@pytest.fixture
def service(mock_db: AsyncMock) -> AnalysisService:
    """AnalysisService wired to a mock database session."""
    return AnalysisService(db=mock_db)


@pytest.fixture
def sample_entities() -> list[dict]:
    """Entity dicts matching the expected store_entities schema."""
    return [
        {
            "entity_type": "MEDICATION",
            "text": "warfarin",
            "normalized_text": "warfarin sodium",
            "umls_cui": "C0043031",
            "start_char": 10,
            "end_char": 18,
            "confidence": 0.95,
            "is_negated": False,
            "is_uncertain": False,
        },
        {
            "entity_type": "DISEASE",
            "text": "heart failure",
            "normalized_text": None,
            "umls_cui": None,
            "start_char": 30,
            "end_char": 43,
            "confidence": 0.88,
            "is_negated": False,
            "is_uncertain": True,
        },
    ]


# ---------------------------------------------------------------------------
# store_document
# ---------------------------------------------------------------------------


class TestStoreDocument:
    """Tests for AnalysisService.store_document."""

    @pytest.mark.asyncio
    async def test_creates_document_with_content_hash(
        self, service: AnalysisService, mock_db: AsyncMock
    ):
        content = "Patient presents with chest pain."
        await service.store_document(content, title="Note #1")

        # Verify db.add was called
        mock_db.add.assert_called_once()
        mock_db.flush.assert_awaited_once()

        # The returned object should have the content hash
        added_obj = mock_db.add.call_args[0][0]
        expected_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        assert added_obj.content_hash == expected_hash
        assert added_obj.content == content
        assert added_obj.title == "Note #1"
        assert added_obj.source == "api"

    @pytest.mark.asyncio
    async def test_accepts_optional_parameters(
        self, service: AnalysisService, mock_db: AsyncMock
    ):
        user_id = uuid4()
        await service.store_document(
            "text",
            user_id=user_id,
            title="Custom",
            source="upload",
            specialty="cardiology",
        )
        added_obj = mock_db.add.call_args[0][0]
        assert added_obj.user_id == user_id
        assert added_obj.source == "upload"
        assert added_obj.specialty == "cardiology"

    @pytest.mark.asyncio
    async def test_document_gets_uuid(
        self, service: AnalysisService, mock_db: AsyncMock
    ):
        await service.store_document("text")
        added_obj = mock_db.add.call_args[0][0]
        assert isinstance(added_obj.id, UUID)


# ---------------------------------------------------------------------------
# store_entities
# ---------------------------------------------------------------------------


class TestStoreEntities:
    """Tests for AnalysisService.store_entities."""

    @pytest.mark.asyncio
    async def test_stores_all_entities(
        self, service: AnalysisService, mock_db: AsyncMock, sample_entities: list[dict]
    ):
        doc_id = uuid4()
        await service.store_entities(doc_id, sample_entities, model_version="1.0")

        # db.add should be called once per entity
        assert mock_db.add.call_count == len(sample_entities)
        mock_db.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_entity_fields_mapped_correctly(
        self, service: AnalysisService, mock_db: AsyncMock, sample_entities: list[dict]
    ):
        doc_id = uuid4()
        await service.store_entities(doc_id, sample_entities, model_version="2.0")

        # Check first entity
        first_call = mock_db.add.call_args_list[0]
        entity_obj = first_call[0][0]
        assert entity_obj.entity_type == "MEDICATION"
        assert entity_obj.text == "warfarin"
        assert entity_obj.confidence == 0.95
        assert entity_obj.model_version == "2.0"
        assert entity_obj.document_id == doc_id

    @pytest.mark.asyncio
    async def test_empty_entity_list(
        self, service: AnalysisService, mock_db: AsyncMock
    ):
        result = await service.store_entities(uuid4(), [])
        mock_db.add.assert_not_called()
        mock_db.flush.assert_awaited_once()
        assert result == []

    @pytest.mark.asyncio
    async def test_optional_fields_default_correctly(
        self, service: AnalysisService, mock_db: AsyncMock
    ):
        """Entities without optional fields should use defaults."""
        entities = [
            {
                "entity_type": "SYMPTOM",
                "text": "fever",
                "start_char": 0,
                "end_char": 5,
                "confidence": 0.7,
            }
        ]
        await service.store_entities(uuid4(), entities)
        entity_obj = mock_db.add.call_args[0][0]
        assert entity_obj.is_negated is False
        assert entity_obj.is_uncertain is False
        assert entity_obj.normalized_text is None


# ---------------------------------------------------------------------------
# store_prediction
# ---------------------------------------------------------------------------


class TestStorePrediction:
    """Tests for AnalysisService.store_prediction."""

    @pytest.mark.asyncio
    async def test_stores_prediction(
        self, service: AnalysisService, mock_db: AsyncMock
    ):
        doc_id = uuid4()
        result_data = {"codes": ["I21.0"], "top_score": 0.9}

        await service.store_prediction(
            document_id=doc_id,
            prediction_type="icd_10",
            model_name="rule-based-icd",
            model_version="1.0.0",
            result=result_data,
            confidence=0.9,
            processing_time_ms=42,
        )

        mock_db.add.assert_called_once()
        mock_db.flush.assert_awaited_once()

        pred_obj = mock_db.add.call_args[0][0]
        assert pred_obj.document_id == doc_id
        assert pred_obj.prediction_type == "icd_10"
        assert pred_obj.model_name == "rule-based-icd"
        assert pred_obj.result == result_data
        assert pred_obj.confidence == 0.9
        assert isinstance(pred_obj.id, UUID)

    @pytest.mark.asyncio
    async def test_optional_fields(
        self, service: AnalysisService, mock_db: AsyncMock
    ):
        await service.store_prediction(
            document_id=uuid4(),
            prediction_type="ner",
            model_name="model",
            model_version="1.0",
            result={},
        )
        pred_obj = mock_db.add.call_args[0][0]
        assert pred_obj.confidence is None
        assert pred_obj.processing_time_ms is None


# ---------------------------------------------------------------------------
# log_audit
# ---------------------------------------------------------------------------


class TestLogAudit:
    """Tests for AnalysisService.log_audit."""

    @pytest.mark.asyncio
    async def test_creates_audit_entry(
        self, service: AnalysisService, mock_db: AsyncMock
    ):
        await service.log_audit(
            action="document.analyze",
            user_id=uuid4(),
            resource_type="document",
            ip_address="192.168.1.1",
            status_code=200,
            response_time_ms=150,
        )
        mock_db.add.assert_called_once()
        mock_db.flush.assert_awaited_once()

        audit_obj = mock_db.add.call_args[0][0]
        assert audit_obj.action == "document.analyze"
        assert audit_obj.ip_address == "192.168.1.1"
        assert audit_obj.status_code == 200

    @pytest.mark.asyncio
    async def test_minimal_audit_entry(
        self, service: AnalysisService, mock_db: AsyncMock
    ):
        """Only action is required; everything else can be None."""
        await service.log_audit(action="health.check")
        audit_obj = mock_db.add.call_args[0][0]
        assert audit_obj.action == "health.check"
        assert audit_obj.user_id is None
        assert audit_obj.ip_address is None

    @pytest.mark.asyncio
    async def test_audit_gets_uuid(
        self, service: AnalysisService, mock_db: AsyncMock
    ):
        await service.log_audit(action="test")
        audit_obj = mock_db.add.call_args[0][0]
        assert isinstance(audit_obj.id, UUID)


# ---------------------------------------------------------------------------
# mark_document_processed
# ---------------------------------------------------------------------------


class TestMarkDocumentProcessed:
    """Tests for AnalysisService.mark_document_processed."""

    @pytest.mark.asyncio
    async def test_executes_update(
        self, service: AnalysisService, mock_db: AsyncMock
    ):
        doc_id = uuid4()
        await service.mark_document_processed(doc_id)

        # Should call execute (with an UPDATE statement) and flush
        mock_db.execute.assert_awaited_once()
        mock_db.flush.assert_awaited_once()
