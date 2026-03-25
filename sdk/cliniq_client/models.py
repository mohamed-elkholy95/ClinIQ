"""ClinIQ SDK data models.

Typed dataclasses representing API response objects for all ClinIQ
endpoints.  Each model provides a ``from_dict`` class method for safe
deserialization from raw JSON dicts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Core analysis models
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """Extracted medical entity."""

    text: str
    entity_type: str
    start_char: int
    end_char: int
    confidence: float
    normalized_text: str | None = None
    umls_cui: str | None = None
    is_negated: bool = False
    is_uncertain: bool = False


@dataclass
class ICDPrediction:
    """ICD-10 code prediction."""

    code: str
    description: str | None
    confidence: float
    chapter: str | None = None
    contributing_text: list[str] | None = None


@dataclass
class Summary:
    """Clinical text summary."""

    summary: str
    key_findings: list[str] = field(default_factory=list)
    detail_level: str = "standard"
    word_count: int = 0


@dataclass
class RiskFactor:
    """Risk assessment factor."""

    name: str
    score: float
    weight: float
    category: str
    description: str = ""


@dataclass
class RiskAssessment:
    """Patient risk assessment."""

    overall_score: float
    risk_level: str
    factors: list[RiskFactor] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Full pipeline analysis result."""

    entities: list[Entity] = field(default_factory=list)
    icd_predictions: list[ICDPrediction] = field(default_factory=list)
    summary: Summary | None = None
    risk_assessment: RiskAssessment | None = None
    processing_time_ms: float = 0.0
    model_versions: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisResult:
        """Create from API response dict."""
        entities = [Entity(**e) for e in (data.get("entities") or [])]
        icd_preds = [ICDPrediction(**p) for p in (data.get("icd_predictions") or [])]
        summary = None
        if data.get("summary"):
            summary = Summary(**data["summary"])
        risk = None
        if data.get("risk_assessment"):
            ra = data["risk_assessment"]
            factors = [RiskFactor(**f) for f in (ra.get("factors") or [])]
            risk = RiskAssessment(
                overall_score=ra["overall_score"],
                risk_level=ra["risk_level"],
                factors=factors,
                recommendations=ra.get("recommendations", []),
            )
        return cls(
            entities=entities,
            icd_predictions=icd_preds,
            summary=summary,
            risk_assessment=risk,
            processing_time_ms=data.get("processing_time_ms", 0),
            model_versions=data.get("model_versions", {}),
        )


@dataclass
class BatchJob:
    """Batch processing job status."""

    job_id: str
    status: str
    total_documents: int
    processed_documents: int = 0
    failed_documents: int = 0
    progress: float = 0.0
    result_file: str | None = None


# ---------------------------------------------------------------------------
# Document classification
# ---------------------------------------------------------------------------


@dataclass
class ClassificationScore:
    """Single document type score."""

    document_type: str
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class ClassificationResult:
    """Document classification result."""

    predicted_type: str
    scores: list[ClassificationScore] = field(default_factory=list)
    processing_time_ms: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassificationResult:
        """Create from API response dict."""
        scores = [
            ClassificationScore(**s) for s in (data.get("scores") or [])
        ]
        return cls(
            predicted_type=data.get("predicted_type", "unknown"),
            scores=scores,
            processing_time_ms=data.get("processing_time_ms", 0),
        )


# ---------------------------------------------------------------------------
# Medication extraction
# ---------------------------------------------------------------------------


@dataclass
class Medication:
    """Extracted medication with components."""

    drug_name: str
    generic_name: str | None = None
    dosage: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: str | None = None
    indication: str | None = None
    prn: bool = False
    status: str = "active"
    confidence: float = 0.0


@dataclass
class MedicationResult:
    """Medication extraction result."""

    medication_count: int
    medications: list[Medication] = field(default_factory=list)
    processing_time_ms: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MedicationResult:
        """Create from API response dict."""
        meds = [Medication(**m) for m in (data.get("medications") or [])]
        return cls(
            medication_count=data.get("medication_count", len(meds)),
            medications=meds,
            processing_time_ms=data.get("processing_time_ms", 0),
        )


# ---------------------------------------------------------------------------
# Allergy extraction
# ---------------------------------------------------------------------------


@dataclass
class Allergy:
    """Extracted allergen with reactions."""

    allergen: str
    category: str = "unknown"
    reactions: list[dict[str, Any]] = field(default_factory=list)
    severity: str = "unknown"
    status: str = "active"
    confidence: float = 0.0


@dataclass
class AllergyResult:
    """Allergy extraction result."""

    allergy_count: int
    no_known_allergies: bool = False
    allergies: list[Allergy] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AllergyResult:
        """Create from API response dict."""
        allergies = [Allergy(**a) for a in (data.get("allergies") or [])]
        return cls(
            allergy_count=data.get("allergy_count", len(allergies)),
            no_known_allergies=data.get("no_known_allergies", False),
            allergies=allergies,
        )


# ---------------------------------------------------------------------------
# Vital signs extraction
# ---------------------------------------------------------------------------


@dataclass
class VitalSign:
    """Extracted vital sign measurement."""

    vital_type: str
    value: float
    unit: str
    interpretation: str = "normal"
    confidence: float = 0.0


@dataclass
class VitalSignResult:
    """Vital signs extraction result."""

    vital_count: int
    vitals: list[VitalSign] = field(default_factory=list)
    processing_time_ms: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VitalSignResult:
        """Create from API response dict."""
        vitals = [VitalSign(**v) for v in (data.get("vitals") or [])]
        return cls(
            vital_count=data.get("vital_count", len(vitals)),
            vitals=vitals,
            processing_time_ms=data.get("processing_time_ms", 0),
        )


# ---------------------------------------------------------------------------
# Section parsing
# ---------------------------------------------------------------------------


@dataclass
class Section:
    """Parsed document section."""

    category: str
    header: str
    header_normalised: str = ""
    header_start: int = 0
    header_end: int = 0
    body_end: int = 0
    confidence: float = 0.0


@dataclass
class SectionResult:
    """Section parsing result."""

    section_count: int
    sections: list[Section] = field(default_factory=list)
    categories_found: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SectionResult:
        """Create from API response dict."""
        sections = [Section(**s) for s in (data.get("sections") or [])]
        return cls(
            section_count=data.get("section_count", len(sections)),
            sections=sections,
            categories_found=data.get("categories_found", []),
        )


# ---------------------------------------------------------------------------
# Abbreviation expansion
# ---------------------------------------------------------------------------


@dataclass
class AbbreviationMatch:
    """Detected abbreviation with expansion."""

    abbreviation: str
    expansion: str
    start: int = 0
    end: int = 0
    confidence: float = 0.0
    domain: str = "general"
    is_ambiguous: bool = False


@dataclass
class AbbreviationResult:
    """Abbreviation expansion result."""

    total_found: int
    expanded_text: str = ""
    matches: list[AbbreviationMatch] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AbbreviationResult:
        """Create from API response dict."""
        matches = [AbbreviationMatch(**m) for m in (data.get("matches") or [])]
        return cls(
            total_found=data.get("total_found", len(matches)),
            expanded_text=data.get("expanded_text", ""),
            matches=matches,
        )


# ---------------------------------------------------------------------------
# Quality analysis
# ---------------------------------------------------------------------------


@dataclass
class QualityDimension:
    """Quality score for a single dimension."""

    dimension: str
    score: float
    weight: float
    finding_count: int = 0


@dataclass
class QualityReport:
    """Clinical note quality report."""

    overall_score: float
    grade: str
    dimensions: list[QualityDimension] = field(default_factory=list)
    recommendation_count: int = 0
    top_recommendations: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityReport:
        """Create from API response dict."""
        dims = [QualityDimension(**d) for d in (data.get("dimensions") or [])]
        return cls(
            overall_score=data.get("overall_score", 0),
            grade=data.get("grade", "F"),
            dimensions=dims,
            recommendation_count=data.get("recommendation_count", 0),
            top_recommendations=data.get("top_recommendations", []),
        )


# ---------------------------------------------------------------------------
# Assertion detection
# ---------------------------------------------------------------------------


@dataclass
class AssertionResult:
    """Assertion status for an entity."""

    entity_text: str
    entity_type: str = ""
    status: str = "present"
    confidence: float = 0.0
    trigger_text: str | None = None


# ---------------------------------------------------------------------------
# Concept normalization
# ---------------------------------------------------------------------------


@dataclass
class NormalizationResult:
    """Normalized entity with ontology codes."""

    entity_text: str
    entity_type: str = ""
    cui: str | None = None
    preferred_term: str | None = None
    match_type: str = "exact"
    confidence: float = 0.0
    snomed_code: str | None = None
    rxnorm_code: str | None = None
    icd10_code: str | None = None
    loinc_code: str | None = None


# ---------------------------------------------------------------------------
# SDoH extraction
# ---------------------------------------------------------------------------


@dataclass
class SDoHExtraction:
    """Social Determinant of Health extraction."""

    domain: str
    text: str
    sentiment: str = "adverse"
    confidence: float = 0.0
    z_codes: list[str] = field(default_factory=list)


@dataclass
class SDoHResult:
    """SDoH extraction result."""

    extraction_count: int
    adverse_count: int = 0
    protective_count: int = 0
    domain_summary: dict[str, int] = field(default_factory=dict)
    extractions: list[SDoHExtraction] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SDoHResult:
        """Create from API response dict."""
        extractions = [SDoHExtraction(**e) for e in (data.get("extractions") or [])]
        return cls(
            extraction_count=data.get("extraction_count", len(extractions)),
            adverse_count=data.get("adverse_count", 0),
            protective_count=data.get("protective_count", 0),
            domain_summary=data.get("domain_summary", {}),
            extractions=extractions,
        )


# ---------------------------------------------------------------------------
# Comorbidity scoring
# ---------------------------------------------------------------------------


@dataclass
class MatchedCategory:
    """Matched comorbidity category."""

    category: str
    weight: int
    source: str = ""
    evidence: str = ""
    confidence: float = 0.0


@dataclass
class ComorbidityResult:
    """Charlson Comorbidity Index result."""

    raw_score: int
    age_adjusted_score: int | None = None
    risk_group: str = "low"
    ten_year_mortality: float = 0.0
    category_count: int = 0
    matched_categories: list[MatchedCategory] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ComorbidityResult:
        """Create from API response dict."""
        cats = [MatchedCategory(**c) for c in (data.get("matched_categories") or [])]
        return cls(
            raw_score=data.get("raw_score", 0),
            age_adjusted_score=data.get("age_adjusted_score"),
            risk_group=data.get("risk_group", "low"),
            ten_year_mortality=data.get("ten_year_mortality", 0),
            category_count=data.get("category_count", len(cats)),
            matched_categories=cats,
        )


# ---------------------------------------------------------------------------
# Relation extraction
# ---------------------------------------------------------------------------


@dataclass
class Relation:
    """Extracted clinical relation."""

    subject: str
    subject_type: str
    object: str
    object_type: str
    relation_type: str
    confidence: float = 0.0
    evidence: str = ""


@dataclass
class RelationResult:
    """Relation extraction result."""

    relation_count: int
    pair_count: int = 0
    relations: list[Relation] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationResult:
        """Create from API response dict."""
        rels = [Relation(**r) for r in (data.get("relations") or [])]
        return cls(
            relation_count=data.get("relation_count", len(rels)),
            pair_count=data.get("pair_count", 0),
            relations=rels,
        )


# ---------------------------------------------------------------------------
# Enhanced pipeline result
# ---------------------------------------------------------------------------


@dataclass
class EnhancedAnalysisResult:
    """Full enhanced pipeline analysis result."""

    base_result: AnalysisResult | None = None
    classification: dict[str, Any] | None = None
    sections: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None
    deidentification: dict[str, Any] | None = None
    abbreviations: dict[str, Any] | None = None
    medications: dict[str, Any] | None = None
    allergies: dict[str, Any] | None = None
    vitals: dict[str, Any] | None = None
    temporal: dict[str, Any] | None = None
    assertions: list[dict[str, Any]] | None = None
    normalization: list[dict[str, Any]] | None = None
    sdoh: dict[str, Any] | None = None
    relations: dict[str, Any] | None = None
    comorbidity: dict[str, Any] | None = None
    processing_time_ms: float = 0.0
    component_errors: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnhancedAnalysisResult:
        """Create from API response dict."""
        base = None
        if data.get("base_result"):
            base = AnalysisResult.from_dict(data["base_result"])
        return cls(
            base_result=base,
            classification=data.get("classification"),
            sections=data.get("sections"),
            quality=data.get("quality"),
            deidentification=data.get("deidentification"),
            abbreviations=data.get("abbreviations"),
            medications=data.get("medications"),
            allergies=data.get("allergies"),
            vitals=data.get("vitals"),
            temporal=data.get("temporal"),
            assertions=data.get("assertions"),
            normalization=data.get("normalization"),
            sdoh=data.get("sdoh"),
            relations=data.get("relations"),
            comorbidity=data.get("comorbidity"),
            processing_time_ms=data.get("processing_time_ms", 0),
            component_errors=data.get("component_errors", {}),
        )


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


@dataclass
class SearchHit:
    """Single search result."""

    document_id: str
    score: float
    snippet: str = ""
    title: str = ""


@dataclass
class SearchResult:
    """Document search result."""

    hits: list[SearchHit] = field(default_factory=list)
    total: int = 0
    query_expansion: dict[str, Any] | None = None
    reranked: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchResult:
        """Create from API response dict."""
        hits = [SearchHit(**h) for h in (data.get("hits") or [])]
        return cls(
            hits=hits,
            total=data.get("total", len(hits)),
            query_expansion=data.get("query_expansion"),
            reranked=data.get("reranked", False),
        )
