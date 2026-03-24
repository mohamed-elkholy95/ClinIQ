"""NER request and response schemas."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

MAX_TEXT_LENGTH = 100_000

_ENTITY_TYPES = Literal[
    "DISEASE",
    "SYMPTOM",
    "MEDICATION",
    "DOSAGE",
    "PROCEDURE",
    "ANATOMY",
    "LAB_VALUE",
    "TEST",
    "TREATMENT",
    "DEVICE",
    "BODY_PART",
    "DURATION",
    "FREQUENCY",
    "TEMPORAL",
]

# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class NERRequest(BaseModel):
    """Request body for named entity recognition."""

    text: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_TEXT_LENGTH,
            description="Clinical text to analyse (plain text, up to 100 000 characters)",
        ),
    ]
    model: str = Field(
        default="rule-based",
        description=(
            "NER backend to use. "
            "One of 'rule-based', 'spacy' (en_ner_bc5cdr_md), or 'transformer' (BioBERT). "
            "Defaults to the fast rule-based model."
        ),
    )
    entity_types: list[_ENTITY_TYPES] | None = Field(
        default=None,
        description=(
            "Restrict output to these entity types. "
            "Omit or pass null to return all supported types."
        ),
    )
    include_negated: bool = Field(
        default=True,
        description="Include entities that appear in a negated context (is_negated=True).",
    )
    include_uncertain: bool = Field(
        default=True,
        description="Include entities marked as uncertain (is_uncertain=True).",
    )
    min_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold; entities below this score are dropped.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": (
                    "Patient is a 62-year-old male with a history of hypertension and "
                    "type 2 diabetes mellitus. Currently on metformin 500 mg BID and "
                    "lisinopril 10 mg daily. Denies chest pain or dyspnoea."
                ),
                "model": "rule-based",
                "entity_types": ["DISEASE", "MEDICATION", "DOSAGE"],
                "include_negated": False,
                "include_uncertain": True,
                "min_confidence": 0.5,
            }
        }
    }


# ---------------------------------------------------------------------------
# Response atoms
# ---------------------------------------------------------------------------


class EntityResponse(BaseModel):
    """A single extracted medical entity."""

    text: str = Field(description="Exact surface form of the entity as it appears in the source text")
    entity_type: str = Field(
        description=(
            "Semantic type label, e.g. DISEASE, MEDICATION, LAB_VALUE. "
            "See /api/v1/ner/entity-types for the full vocabulary."
        )
    )
    start_char: int = Field(ge=0, description="Character offset of the first character of the entity (inclusive)")
    end_char: int = Field(ge=0, description="Character offset of the character after the last character (exclusive)")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence score in [0, 1]")
    normalized_text: str | None = Field(
        default=None,
        description="Canonicalised / preferred form of the entity text, if available",
    )
    umls_cui: str | None = Field(
        default=None,
        description="UMLS Concept Unique Identifier (e.g. 'C0011849'), when available",
    )
    is_negated: bool = Field(
        default=False,
        description="True when the entity appears within a negation scope (e.g. 'denies chest pain')",
    )
    is_uncertain: bool = Field(
        default=False,
        description="True when the entity is qualified by uncertainty language (e.g. 'possible PE')",
    )
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional model-specific properties",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "metformin",
                "entity_type": "MEDICATION",
                "start_char": 92,
                "end_char": 101,
                "confidence": 0.94,
                "normalized_text": "Metformin",
                "umls_cui": "C0025598",
                "is_negated": False,
                "is_uncertain": False,
                "metadata": None,
            }
        }
    }


# ---------------------------------------------------------------------------
# Response envelope
# ---------------------------------------------------------------------------


class NERResponse(BaseModel):
    """Top-level NER response."""

    text_length: int = Field(ge=0, description="Number of characters in the input text")
    entity_count: int = Field(ge=0, description="Total number of entities returned after filtering")
    entities: list[EntityResponse] = Field(description="Extracted entities, ordered by start_char")
    model_name: str = Field(description="Name of the model that produced these predictions")
    model_version: str = Field(description="Version string of the deployed model")
    processing_time_ms: float = Field(ge=0, description="End-to-end inference latency in milliseconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text_length": 210,
                "entity_count": 4,
                "entities": [
                    {
                        "text": "hypertension",
                        "entity_type": "DISEASE",
                        "start_char": 56,
                        "end_char": 68,
                        "confidence": 0.9,
                        "normalized_text": "Hypertension",
                        "umls_cui": "C0020538",
                        "is_negated": False,
                        "is_uncertain": False,
                        "metadata": None,
                    }
                ],
                "model_name": "rule-based",
                "model_version": "1.0.0",
                "processing_time_ms": 12.3,
            }
        }
    }
