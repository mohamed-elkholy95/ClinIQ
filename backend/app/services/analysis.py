"""Analysis service orchestrating ML pipeline and data persistence."""

import hashlib
import logging
import time
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AuditLog, Document, Entity, Prediction

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service for orchestrating clinical text analysis."""

    def __init__(self, db: AsyncSession):
        self.db = db

    async def store_document(
        self,
        content: str,
        user_id: UUID | None = None,
        title: str | None = None,
        source: str = "api",
        specialty: str | None = None,
    ) -> Document:
        """Store a clinical document in the database."""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        document = Document(
            id=uuid4(),
            user_id=user_id,
            content=content,
            content_hash=content_hash,
            title=title,
            source=source,
            specialty=specialty,
        )
        self.db.add(document)
        await self.db.flush()
        return document

    async def store_entities(
        self,
        document_id: UUID,
        entities: list[dict],
        model_version: str | None = None,
    ) -> list[Entity]:
        """Store extracted entities in the database."""
        db_entities = []
        for entity_data in entities:
            entity = Entity(
                id=uuid4(),
                document_id=document_id,
                entity_type=entity_data["entity_type"],
                text=entity_data["text"],
                normalized_text=entity_data.get("normalized_text"),
                umls_cui=entity_data.get("umls_cui"),
                start_char=entity_data["start_char"],
                end_char=entity_data["end_char"],
                confidence=entity_data["confidence"],
                is_negated=entity_data.get("is_negated", False),
                is_uncertain=entity_data.get("is_uncertain", False),
                model_version=model_version,
            )
            db_entities.append(entity)
            self.db.add(entity)

        await self.db.flush()
        return db_entities

    async def store_prediction(
        self,
        document_id: UUID,
        prediction_type: str,
        model_name: str,
        model_version: str,
        result: dict,
        confidence: float | None = None,
        processing_time_ms: int | None = None,
    ) -> Prediction:
        """Store a model prediction in the database."""
        prediction = Prediction(
            id=uuid4(),
            document_id=document_id,
            prediction_type=prediction_type,
            model_name=model_name,
            model_version=model_version,
            result=result,
            confidence=confidence,
            processing_time_ms=processing_time_ms,
        )
        self.db.add(prediction)
        await self.db.flush()
        return prediction

    async def log_audit(
        self,
        action: str,
        user_id: UUID | None = None,
        api_key_id: UUID | None = None,
        resource_type: str | None = None,
        resource_id: UUID | None = None,
        document_hash: str | None = None,
        model_name: str | None = None,
        model_version: str | None = None,
        ip_address: str | None = None,
        user_agent: str | None = None,
        status_code: int | None = None,
        response_time_ms: int | None = None,
        metadata: dict | None = None,
    ) -> AuditLog:
        """Write an audit log entry."""
        audit = AuditLog(
            id=uuid4(),
            user_id=user_id,
            api_key_id=api_key_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            document_hash=document_hash,
            model_name=model_name,
            model_version=model_version,
            ip_address=ip_address,
            user_agent=user_agent,
            status_code=status_code,
            response_time_ms=response_time_ms,
            metadata_=metadata,
        )
        self.db.add(audit)
        await self.db.flush()
        return audit

    async def mark_document_processed(self, document_id: UUID) -> None:
        """Mark a document as processed."""
        from datetime import datetime, timezone

        from sqlalchemy import update

        await self.db.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(is_processed=True, processed_at=datetime.now(timezone.utc))
        )
        await self.db.flush()
