"""Initial database schema

Revision ID: 001
Revises: None
Create Date: 2026-03-24
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False, index=True),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255)),
        sa.Column("is_active", sa.Boolean, default=True, nullable=False),
        sa.Column("is_superuser", sa.Boolean, default=False, nullable=False),
        sa.Column("role", sa.String(50), default="user", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "api_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("hashed_key", sa.String(255), nullable=False),
        sa.Column("key_prefix", sa.String(10), nullable=False),
        sa.Column("is_active", sa.Boolean, default=True, nullable=False),
        sa.Column("rate_limit", sa.Integer, default=100),
        sa.Column("expires_at", sa.DateTime(timezone=True)),
        sa.Column("last_used_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
            index=True,
        ),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False, index=True),
        sa.Column("title", sa.String(500)),
        sa.Column("source", sa.String(100)),
        sa.Column("specialty", sa.String(100)),
        sa.Column("metadata", postgresql.JSONB, default=dict),
        sa.Column("is_processed", sa.Boolean, default=False, nullable=False),
        sa.Column("processed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_documents_user_created", "documents", ["user_id", "created_at"])

    op.create_table(
        "entities",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("entity_type", sa.String(50), nullable=False, index=True),
        sa.Column("text", sa.String(500), nullable=False),
        sa.Column("normalized_text", sa.String(500)),
        sa.Column("umls_cui", sa.String(20)),
        sa.Column("start_char", sa.Integer, nullable=False),
        sa.Column("end_char", sa.Integer, nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("is_negated", sa.Boolean, default=False, nullable=False),
        sa.Column("is_uncertain", sa.Boolean, default=False, nullable=False),
        sa.Column("metadata", postgresql.JSONB, default=dict),
        sa.Column("model_version", sa.String(50)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_entities_document_type", "entities", ["document_id", "entity_type"])

    op.create_table(
        "predictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("prediction_type", sa.String(50), nullable=False, index=True),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("model_version", sa.String(50), nullable=False),
        sa.Column("result", postgresql.JSONB, nullable=False),
        sa.Column("confidence", sa.Float),
        sa.Column("processing_time_ms", sa.Integer),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "ix_predictions_document_type", "predictions", ["document_id", "prediction_type"]
    )
    op.create_index(
        "ix_predictions_model_version", "predictions", ["model_name", "model_version"]
    )

    op.create_table(
        "icd_codes",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("code", sa.String(10), unique=True, nullable=False, index=True),
        sa.Column("description", sa.String(500), nullable=False),
        sa.Column("chapter", sa.String(100)),
        sa.Column("category", sa.String(100)),
        sa.Column("is_active", sa.Boolean, default=True, nullable=False),
    )

    op.create_table(
        "audit_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "timestamp",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
            index=True,
        ),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), index=True),
        sa.Column("api_key_id", postgresql.UUID(as_uuid=True), index=True),
        sa.Column("action", sa.String(100), nullable=False, index=True),
        sa.Column("resource_type", sa.String(50)),
        sa.Column("resource_id", postgresql.UUID(as_uuid=True)),
        sa.Column("document_hash", sa.String(64)),
        sa.Column("model_name", sa.String(100)),
        sa.Column("model_version", sa.String(50)),
        sa.Column("ip_address", sa.String(45)),
        sa.Column("user_agent", sa.String(500)),
        sa.Column("status_code", sa.Integer),
        sa.Column("response_time_ms", sa.Integer),
        sa.Column("metadata", postgresql.JSONB, default=dict),
    )
    op.create_index("ix_audit_log_user_action", "audit_log", ["user_id", "action"])
    op.create_index("ix_audit_log_action_timestamp", "audit_log", ["action", "timestamp"])

    op.create_table(
        "batch_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("status", sa.String(20), nullable=False, default="pending", index=True),
        sa.Column("total_documents", sa.Integer, nullable=False),
        sa.Column("processed_documents", sa.Integer, default=0, nullable=False),
        sa.Column("failed_documents", sa.Integer, default=0, nullable=False),
        sa.Column("pipeline_config", postgresql.JSONB, nullable=False),
        sa.Column("result_file", sa.String(500)),
        sa.Column("error_message", sa.Text),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    op.create_table(
        "model_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("model_name", sa.String(100), nullable=False, index=True),
        sa.Column("version", sa.String(50), nullable=False),
        sa.Column("stage", sa.String(20), nullable=False, default="staging"),
        sa.Column("mlflow_run_id", sa.String(100)),
        sa.Column("metrics", postgresql.JSONB),
        sa.Column("config", postgresql.JSONB),
        sa.Column("deployed_at", sa.DateTime(timezone=True)),
        sa.Column("deployed_by", postgresql.UUID(as_uuid=True)),
        sa.Column("is_active", sa.Boolean, default=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index(
        "ix_model_versions_name_version",
        "model_versions",
        ["model_name", "version"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_table("model_versions")
    op.drop_table("batch_jobs")
    op.drop_table("audit_log")
    op.drop_table("icd_codes")
    op.drop_table("predictions")
    op.drop_table("entities")
    op.drop_table("documents")
    op.drop_table("api_keys")
    op.drop_table("users")
